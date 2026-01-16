#!/usr/bin/env python3
"""
02_annotate_corpus.py - Annotate poems with LLM using Gemini API.

This script processes raw poems from data/raw/poems.jsonl and produces
annotated poems in data/annotated/poems.parquet.

Key features:
- Fugashi tokenization (source of truth for tokens)
- LLM annotation via Gemini API for readings, grammar points, vocabulary
- Two-level grammar IDs (canonical_id + sense_id)
- Deterministic difficulty scoring from factors
- Batch processing with checkpointing (resume on failure)
- Schema validation with retry on malformed responses

Usage:
  python scripts/02_annotate_corpus.py --input data/raw/poems.jsonl --output data/annotated/poems.parquet
  python scripts/02_annotate_corpus.py --batch-size 5 --resume
  python scripts/02_annotate_corpus.py --max-poems 50  # For testing
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

# Import Fugashi for tokenization
try:
    import fugashi
except ImportError:
    print("ERROR: fugashi not installed. Run: pip install fugashi unidic-lite")
    sys.exit(1)

# Import Google GenAI (new SDK)
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

from wakadecoder.schemas.annotation import (
    PoemAnnotation,
    FugashiToken,
    TokenReading,
    GrammarPoint,
    VocabularyAnnotation,
    DifficultyFactor,
    compute_difficulty_score,
)
from wakadecoder.utils.prompt_loader import load_prompt, format_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-2.0-flash"  # Faster than preview; use --model gemini-3-flash-preview for higher quality
DEFAULT_BATCH_SIZE = 10
DEFAULT_API_SLEEP = 1.0
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "annotated" / ".checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "annotated" / ".cache"


# -----------------------------------------------------------------------------
# Fugashi Tokenization
# -----------------------------------------------------------------------------

def tokenize_poem(text: str, tagger: fugashi.Tagger) -> list[FugashiToken]:
    """
    Tokenize a poem using Fugashi and compute character spans.

    This is the SOURCE OF TRUTH for tokenization - LLM does not invent tokens.

    Args:
        text: Poem text
        tagger: Fugashi tagger instance

    Returns:
        List of FugashiToken with spans
    """
    tokens = []
    current_pos = 0

    for word in tagger(text):
        surface = word.surface

        # Find the surface in text starting from current position
        start = text.find(surface, current_pos)
        if start == -1:
            # Fallback: use current position
            start = current_pos

        end = start + len(surface)

        # Extract POS information
        pos = word.pos if hasattr(word, 'pos') else str(word.feature)
        pos_detail = str(word.feature) if hasattr(word, 'feature') else pos

        # Get lemma (base form)
        lemma = surface
        if hasattr(word, 'feature'):
            features = str(word.feature).split(',')
            # UniDic format: POS,POS_detail,inflection_type,inflection_form,lemma,...
            if len(features) >= 7 and features[6]:
                lemma = features[6]

        tokens.append(FugashiToken(
            surface=surface,
            pos=pos.split(',')[0] if ',' in pos else pos,
            pos_detail=pos_detail,
            lemma=lemma,
            span=[start, end]
        ))

        current_pos = end

    return tokens


def tokens_to_json(tokens: list[FugashiToken]) -> str:
    """Convert tokens to JSON for LLM prompt."""
    token_dicts = []
    for i, t in enumerate(tokens):
        token_dicts.append({
            "index": i,
            "surface": t.surface,
            "pos": t.pos,
            "lemma": t.lemma,
            "span": t.span
        })
    return json.dumps(token_dicts, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Gemini API Client
# -----------------------------------------------------------------------------

class GeminiClient:
    """Wrapper for Gemini API with rate limiting and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        sleep_seconds: float = DEFAULT_API_SLEEP
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")

        self.client = genai.Client(api_key=self.api_key)
        self.temperature = temperature
        self.sleep_seconds = sleep_seconds
        self.model_name = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> str:
        """
        Generate text using Gemini API.

        Args:
            system_prompt: System context
            user_prompt: User message
            max_retries: Number of retries on failure

        Returns:
            Generated text response
        """
        # Combine system and user prompts (Gemini uses single prompt)
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=self.temperature,
                    )
                )

                # Rate limiting
                time.sleep(self.sleep_seconds)

                # Handle potential None response
                if response.text is None:
                    # Try to extract text from candidates
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            return candidate.content.parts[0].text
                    raise ValueError("Empty response from API")

                return response.text

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.sleep_seconds * (attempt + 1))  # Exponential backoff
                else:
                    raise


# -----------------------------------------------------------------------------
# LLM Response Parsing and Validation
# -----------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response

    Returns:
        Parsed JSON dict
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text)

    if matches:
        # Try each code block
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON (starts with { and ends with })
    text = text.strip()
    if text.startswith('{'):
        # Find the matching closing brace
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos > 0:
            try:
                return json.loads(text[:end_pos])
            except json.JSONDecodeError:
                pass

    # Last resort: try to parse the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not extract JSON from response: {e}\n\nResponse:\n{text[:500]}...")


def validate_and_fix_annotation(
    llm_response: dict,
    poem: dict,
    tokens: list[FugashiToken]
) -> dict:
    """
    Validate and fix LLM annotation response.

    Args:
        llm_response: Parsed JSON from LLM
        poem: Original poem dict
        tokens: Fugashi tokens

    Returns:
        Fixed annotation dict ready for PoemAnnotation
    """
    text = poem["text"]
    text_len = len(text)

    # Ensure required fields exist
    if "reading_hiragana" not in llm_response:
        llm_response["reading_hiragana"] = ""
    if "reading_romaji" not in llm_response:
        llm_response["reading_romaji"] = ""
    if "token_readings" not in llm_response:
        llm_response["token_readings"] = []
    if "grammar_points" not in llm_response:
        llm_response["grammar_points"] = []
    if "vocabulary" not in llm_response:
        llm_response["vocabulary"] = []
    if "difficulty_factors" not in llm_response:
        llm_response["difficulty_factors"] = []

    # Fix token_readings to match token count
    token_readings = llm_response.get("token_readings", [])
    if len(token_readings) < len(tokens):
        # Pad with surface forms
        for i in range(len(token_readings), len(tokens)):
            token_readings.append({
                "token_index": i,
                "reading_kana": tokens[i].surface
            })
    elif len(token_readings) > len(tokens):
        # Truncate
        token_readings = token_readings[:len(tokens)]

    # Ensure token indices are valid
    for i, tr in enumerate(token_readings):
        tr["token_index"] = i
        if "reading_kana" not in tr or not tr["reading_kana"]:
            tr["reading_kana"] = tokens[i].surface

    llm_response["token_readings"] = token_readings

    # Validate and fix grammar point spans
    valid_grammar_points = []
    for gp in llm_response.get("grammar_points", []):
        # Ensure required fields
        if not gp.get("canonical_id"):
            continue
        if not gp.get("surface"):
            continue

        # Fix canonical_id format (must match pattern ^[a-z]+_[a-z0-9_]+$)
        canonical_id = gp["canonical_id"]
        canonical_id = canonical_id.lower().replace(" ", "_").replace("-", "_")
        if not re.match(r'^[a-z]+_[a-z0-9_]+$', canonical_id):
            # Try to fix by adding category prefix
            category = gp.get("category", "other")
            canonical_id = f"{category}_{canonical_id}".replace("__", "_")
            canonical_id = re.sub(r'[^a-z0-9_]', '', canonical_id)
        gp["canonical_id"] = canonical_id

        # Fix sense_id format
        if gp.get("sense_id"):
            sense_id = gp["sense_id"].lower().replace(" ", "_").replace("-", "_")
            sense_id = re.sub(r'[^a-z0-9_]', '', sense_id)
            gp["sense_id"] = sense_id if sense_id else None

        # Fix category
        valid_categories = {"particle", "auxiliary", "conjugation", "kireji", "syntax", "other"}
        if gp.get("category") not in valid_categories:
            gp["category"] = "other"

        # Ensure description exists
        if not gp.get("description"):
            gp["description"] = f"{gp['surface']} - {gp.get('category', 'grammar point')}"

        # Validate and fix span
        span = gp.get("span", [0, 1])
        if isinstance(span, list) and len(span) == 2:
            start, end = span
            # Clamp to valid range
            start = max(0, min(start, text_len - 1))
            end = max(start + 1, min(end, text_len))

            # Try to find actual surface position
            surface = gp["surface"]
            actual_pos = text.find(surface)
            if actual_pos >= 0:
                start = actual_pos
                end = actual_pos + len(surface)

            gp["span"] = [start, end]
        else:
            # Create span from surface
            surface = gp["surface"]
            pos = text.find(surface)
            if pos >= 0:
                gp["span"] = [pos, pos + len(surface)]
            else:
                gp["span"] = [0, 1]

        valid_grammar_points.append(gp)

    llm_response["grammar_points"] = valid_grammar_points

    # Validate and fix vocabulary spans
    valid_vocabulary = []
    for v in llm_response.get("vocabulary", []):
        if not v.get("word") or not v.get("reading") or not v.get("meaning"):
            continue

        # Fix span
        span = v.get("span", [0, 1])
        if isinstance(span, list) and len(span) == 2:
            start, end = span
            start = max(0, min(start, text_len - 1))
            end = max(start + 1, min(end, text_len))

            word = v["word"]
            actual_pos = text.find(word)
            if actual_pos >= 0:
                start = actual_pos
                end = actual_pos + len(word)

            v["span"] = [start, end]
        else:
            word = v["word"]
            pos = text.find(word)
            if pos >= 0:
                v["span"] = [pos, pos + len(word)]
            else:
                v["span"] = [0, 1]

        valid_vocabulary.append(v)

    llm_response["vocabulary"] = valid_vocabulary

    # Validate difficulty factors
    valid_factors = []
    for f in llm_response.get("difficulty_factors", []):
        if not f.get("factor"):
            continue

        weight = f.get("weight", 0.1)
        if not isinstance(weight, (int, float)):
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                weight = 0.1

        weight = max(0.0, min(1.0, weight))
        f["weight"] = weight

        valid_factors.append(f)

    llm_response["difficulty_factors"] = valid_factors

    return llm_response


def build_poem_annotation(
    poem: dict,
    tokens: list[FugashiToken],
    llm_response: dict
) -> PoemAnnotation:
    """
    Build a PoemAnnotation from poem data, tokens, and LLM response.

    Args:
        poem: Raw poem dict from ingest
        tokens: Fugashi tokens
        llm_response: Validated LLM response

    Returns:
        PoemAnnotation instance
    """
    # Build token readings
    token_readings = [
        TokenReading(
            token_index=tr["token_index"],
            reading_kana=tr["reading_kana"]
        )
        for tr in llm_response.get("token_readings", [])
    ]

    # Build grammar points
    grammar_points = []
    for gp in llm_response.get("grammar_points", []):
        try:
            grammar_points.append(GrammarPoint(
                canonical_id=gp["canonical_id"],
                sense_id=gp.get("sense_id"),
                surface=gp["surface"],
                category=gp["category"],
                description=gp["description"],
                span=gp["span"]
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid grammar point: {gp} - {e}")

    # Build vocabulary
    vocabulary = []
    for v in llm_response.get("vocabulary", []):
        try:
            vocabulary.append(VocabularyAnnotation(
                word=v["word"],
                reading=v["reading"],
                span=v["span"],
                meaning=v["meaning"],
                chinese_cognate_note=v.get("chinese_cognate_note")
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid vocabulary: {v} - {e}")

    # Build difficulty factors
    difficulty_factors = []
    for f in llm_response.get("difficulty_factors", []):
        try:
            difficulty_factors.append(DifficultyFactor(
                factor=f["factor"],
                weight=f["weight"],
                note=f.get("note")
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid difficulty factor: {f} - {e}")

    # Build annotation
    return PoemAnnotation(
        poem_id=poem["poem_id"],
        text=poem["text"],
        text_hash=poem.get("text_hash", PoemAnnotation.compute_text_hash(poem["text"])),
        source=poem["source"],
        author=poem.get("author"),
        collection=poem.get("collection"),
        fugashi_tokens=tokens,
        token_readings=token_readings,
        reading_hiragana=llm_response.get("reading_hiragana", ""),
        reading_romaji=llm_response.get("reading_romaji", ""),
        grammar_points=grammar_points,
        vocabulary=vocabulary,
        difficulty_factors=difficulty_factors,
        semantic_notes=llm_response.get("semantic_notes")
    )


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------

def get_cache_key(poem_id: str, model: str, prompt_version: str) -> str:
    """Generate cache key for LLM response."""
    key = f"{poem_id}_{model}_{prompt_version}"
    return hashlib.md5(key.encode()).hexdigest()


def load_cached_response(cache_key: str) -> dict | None:
    """Load cached LLM response if available."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cached_response(cache_key: str, response: dict):
    """Save LLM response to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(response, ensure_ascii=False, fp=f, indent=2)


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def load_checkpoint(checkpoint_file: Path) -> set[str]:
    """Load set of already processed poem IDs."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_checkpoint(checkpoint_file: Path, poem_id: str):
    """Append poem ID to checkpoint file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(f"{poem_id}\n")


# -----------------------------------------------------------------------------
# Main Annotation Pipeline
# -----------------------------------------------------------------------------

def annotate_poem(
    poem: dict,
    tagger: fugashi.Tagger,
    client: GeminiClient,
    prompt_config: dict,
    use_cache: bool = True
) -> PoemAnnotation | None:
    """
    Annotate a single poem.

    Args:
        poem: Raw poem dict
        tagger: Fugashi tagger
        client: Gemini client
        prompt_config: Loaded prompt configuration
        use_cache: Whether to use response caching

    Returns:
        PoemAnnotation or None on failure
    """
    poem_id = poem["poem_id"]
    text = poem["text"]

    # Tokenize
    tokens = tokenize_poem(text, tagger)

    if not tokens:
        logger.warning(f"No tokens for poem {poem_id}, skipping")
        return None

    # Check cache
    cache_key = get_cache_key(poem_id, client.model_name, prompt_config["meta"]["version"])

    if use_cache:
        cached = load_cached_response(cache_key)
        if cached:
            logger.debug(f"Using cached response for {poem_id}")
            try:
                fixed = validate_and_fix_annotation(cached, poem, tokens)
                return build_poem_annotation(poem, tokens, fixed)
            except Exception as e:
                logger.warning(f"Cached response invalid for {poem_id}: {e}")

    # Build prompt
    tokens_json = tokens_to_json(tokens)
    user_prompt = format_prompt(
        prompt_config["user_template"],
        poem_text=text,
        text_hash=poem.get("text_hash", PoemAnnotation.compute_text_hash(text)),
        fugashi_tokens_json=tokens_json,
        source=poem["source"],
        author=poem.get("author") or "unknown",
        collection=poem.get("collection") or "unknown"
    )

    # Call LLM
    try:
        response_text = client.generate(
            prompt_config["system"],
            user_prompt
        )

        # Parse response
        llm_response = extract_json_from_response(response_text)

        # Cache response
        if use_cache:
            save_cached_response(cache_key, llm_response)

        # Validate and build annotation
        fixed = validate_and_fix_annotation(llm_response, poem, tokens)
        return build_poem_annotation(poem, tokens, fixed)

    except Exception as e:
        logger.error(f"Failed to annotate poem {poem_id}: {e}")
        return None


def annotate_corpus(
    input_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_poems: int | None = None,
    resume: bool = False,
    use_cache: bool = True,
    api_sleep: float = DEFAULT_API_SLEEP
) -> int:
    """
    Annotate all poems in input file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output parquet file
        model: Gemini model name
        batch_size: Number of poems per checkpoint save
        max_poems: Maximum poems to process
        resume: Whether to resume from checkpoint
        use_cache: Whether to use LLM response caching
        api_sleep: Seconds to sleep between API calls

    Returns:
        Number of poems successfully annotated
    """
    # Initialize
    logger.info(f"Loading prompt configuration...")
    prompt_config = load_prompt("annotate")

    logger.info(f"Initializing Fugashi tagger...")
    tagger = fugashi.Tagger()

    logger.info(f"Initializing Gemini client (model: {model})...")
    client = GeminiClient(model=model, sleep_seconds=api_sleep)

    # Checkpointing setup
    checkpoint_file = CHECKPOINT_DIR / f"annotate_{output_path.stem}.txt"
    processed_ids = load_checkpoint(checkpoint_file) if resume else set()

    if resume and processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} poems already processed")

    # Load input poems
    logger.info(f"Loading poems from {input_path}...")
    poems = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            poems.append(json.loads(line))

    logger.info(f"Loaded {len(poems)} poems")

    if max_poems:
        poems = poems[:max_poems]
        logger.info(f"Limited to {len(poems)} poems (--max-poems)")

    # Filter already processed
    if resume:
        poems = [p for p in poems if p["poem_id"] not in processed_ids]
        logger.info(f"{len(poems)} poems remaining after checkpoint filter")

    if not poems:
        logger.info("No poems to process")
        return 0

    # Process poems
    annotations = []
    success_count = 0
    error_count = 0

    # Load existing annotations if resuming
    if resume and output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            annotations = existing_df.to_dict(orient="records")
            logger.info(f"Loaded {len(annotations)} existing annotations")
        except Exception as e:
            logger.warning(f"Could not load existing annotations: {e}")

    for i, poem in enumerate(poems):
        poem_id = poem["poem_id"]

        try:
            logger.info(f"[{i+1}/{len(poems)}] Annotating {poem_id}...")

            annotation = annotate_poem(poem, tagger, client, prompt_config, use_cache)

            if annotation:
                # Convert to dict for DataFrame
                ann_dict = annotation.model_dump()

                # Convert nested Pydantic models to dicts
                ann_dict["fugashi_tokens"] = [t.model_dump() for t in annotation.fugashi_tokens]
                ann_dict["token_readings"] = [t.model_dump() for t in annotation.token_readings]
                ann_dict["grammar_points"] = [g.model_dump() for g in annotation.grammar_points]
                ann_dict["vocabulary"] = [v.model_dump() for v in annotation.vocabulary]
                ann_dict["difficulty_factors"] = [d.model_dump() for d in annotation.difficulty_factors]

                annotations.append(ann_dict)
                success_count += 1

                # Checkpoint
                save_checkpoint(checkpoint_file, poem_id)

                # Save intermediate results
                if (success_count % batch_size) == 0:
                    logger.info(f"Saving checkpoint ({success_count} poems)...")
                    df = pd.DataFrame(annotations)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(output_path, index=False)
            else:
                error_count += 1
                logger.warning(f"Failed to annotate {poem_id}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {poem_id}: {e}")

    # Final save
    if annotations:
        logger.info(f"Saving final results ({len(annotations)} annotations)...")
        df = pd.DataFrame(annotations)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    logger.info(f"Annotation complete: {success_count} success, {error_count} errors")
    return success_count


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate poems with LLM using Gemini API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/02_annotate_corpus.py --input data/raw/poems.jsonl --output data/annotated/poems.parquet
  python scripts/02_annotate_corpus.py --batch-size 5 --resume
  python scripts/02_annotate_corpus.py --max-poems 10  # Quick test
  python scripts/02_annotate_corpus.py --model gemini-2.0-flash --no-cache
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "poems.jsonl",
        help="Input JSONL file with raw poems (default: data/raw/poems.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Output parquet file (default: data/annotated/poems.parquet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Poems per checkpoint save (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-poems",
        type=int,
        default=None,
        help="Maximum poems to annotate (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching",
    )
    parser.add_argument(
        "--api-sleep",
        type=float,
        default=DEFAULT_API_SLEEP,
        help=f"Seconds between API calls (default: {DEFAULT_API_SLEEP})",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    # Run annotation
    count = annotate_corpus(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        batch_size=args.batch_size,
        max_poems=args.max_poems,
        resume=args.resume,
        use_cache=not args.no_cache,
        api_sleep=args.api_sleep
    )

    print(f"\nAnnotated {count} poems to {args.output}")

    # Verification hint
    print(f"\nVerify with:")
    print(f'  python -c "import pandas as pd; df = pd.read_parquet(\'{args.output}\'); print(f\'Poems: {{len(df)}}\'); print(df.columns.tolist())"')


if __name__ == "__main__":
    main()
