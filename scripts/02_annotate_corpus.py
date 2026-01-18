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
- Parallel processing with configurable workers (5-10x speedup)
- Batch checkpointing (resume on failure)
- Schema validation with retry on malformed responses

Usage:
  python scripts/02_annotate_corpus.py --workers 5 --resume  # Fast parallel
  python scripts/02_annotate_corpus.py --max-poems 10        # Quick test
  python scripts/02_annotate_corpus.py --batch-size 5 --resume
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
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

from wakawaka.schemas.annotation import (
    PoemAnnotation,
    FugashiToken,
    TokenReading,
    GrammarPoint,
    VocabularyAnnotation,
    DifficultyFactor,
    compute_difficulty_score,
)
from wakawaka.utils.prompt_loader import load_prompt, format_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview" # high quality is required
DEFAULT_BATCH_SIZE = 10
DEFAULT_API_SLEEP = 1.0
DEFAULT_WORKERS = 100  # Parallel workers for annotation
WORKER_TIMEOUT = 120  # Timeout per poem in seconds (2 minutes)
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "annotated" / ".checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "annotated" / ".cache"

# Thread-local storage for Fugashi tagger (not thread-safe)
_thread_local = threading.local()


def get_thread_tagger() -> "fugashi.Tagger":
    """Get thread-local Fugashi tagger instance with UniDic-Waka for classical Japanese."""
    if not hasattr(_thread_local, 'tagger'):
        dict_path = PROJECT_ROOT / "data" / "dict"
        if (dict_path / "dicrc").exists():
            from fugashi import GenericTagger
            _thread_local.tagger = GenericTagger(f'-d {dict_path} -r {dict_path}/dicrc')
            logger.info("Using UniDic-Waka dictionary for classical Japanese")
        else:
            _thread_local.tagger = fugashi.Tagger()
            logger.warning("UniDic-Waka not found at data/dict/, using default unidic-lite")
    return _thread_local.tagger


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
        sleep_seconds: float = DEFAULT_API_SLEEP,
        request_timeout: float = 60.0  # HTTP request timeout in seconds
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")

        # Create client with HTTP timeout via http_options (timeout in milliseconds)
        http_options = genai_types.HttpOptions(timeout=int(request_timeout * 1000))
        self.client = genai.Client(api_key=self.api_key, http_options=http_options)
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
    if "kanji_transcription" not in llm_response:
        # Default to original text if no transcription provided
        llm_response["kanji_transcription"] = text
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

    # Compute difficulty score from factors
    difficulty_score = compute_difficulty_score(difficulty_factors) if difficulty_factors else 0.0

    # Build annotation
    return PoemAnnotation(
        poem_id=poem["poem_id"],
        text=poem["text"],
        text_hash=poem.get("text_hash", PoemAnnotation.compute_text_hash(poem["text"])),
        source=poem["source"],
        author=poem.get("author"),
        collection=poem.get("collection"),
        kanji_transcription=llm_response.get("kanji_transcription", poem["text"]),
        fugashi_tokens=tokens,
        token_readings=token_readings,
        reading_hiragana=llm_response.get("reading_hiragana", ""),
        reading_romaji=llm_response.get("reading_romaji", ""),
        grammar_points=grammar_points,
        vocabulary=vocabulary,
        difficulty_factors=difficulty_factors,
        difficulty_score=difficulty_score,
        semantic_notes=llm_response.get("semantic_notes")
    )


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------

def get_cache_key(poem_id: str, model: str, prompt_version: str, shot: int = 1) -> str:
    """Generate cache key for LLM response with shot number for 2-shot approach."""
    key = f"{poem_id}_{model}_{prompt_version}_shot{shot}"
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


def save_checkpoint(checkpoint_file: Path, poem_id: str, all_ids: set[str] | None = None):
    """Save poem ID to checkpoint file (deduplicated).

    If all_ids is provided, writes the full set. Otherwise appends and dedupes.
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    if all_ids is not None:
        # Write full set (used for bulk updates)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for pid in sorted(all_ids):
                f.write(f"{pid}\n")
    else:
        # Load existing, add new, dedupe, write back
        existing = load_checkpoint(checkpoint_file)
        existing.add(poem_id)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for pid in sorted(existing):
                f.write(f"{pid}\n")


# -----------------------------------------------------------------------------
# Two-Shot Merge Logic
# -----------------------------------------------------------------------------

def merge_annotations(shot1: dict, shot2: dict) -> dict:
    """
    Merge two annotation shots, preferring Shot 2 for corrected values.

    Strategy:
    - Readings: prefer Shot 2 (refined/corrected)
    - Grammar points: merge by canonical_id, Shot 2 overwrites duplicates
    - Vocabulary: merge by word, Shot 2 overwrites duplicates
    - Difficulty factors: average weights, prefer Shot 2 notes
    """
    result = shot1.copy()

    # Prefer Shot 2 for kanji transcription (more refined)
    if shot2.get("kanji_transcription"):
        result["kanji_transcription"] = shot2["kanji_transcription"]

    # Prefer Shot 2 for readings (more refined)
    if shot2.get("reading_hiragana"):
        result["reading_hiragana"] = shot2["reading_hiragana"]
    if shot2.get("reading_romaji"):
        result["reading_romaji"] = shot2["reading_romaji"]
    if shot2.get("token_readings"):
        result["token_readings"] = shot2["token_readings"]

    # Merge grammar points (Shot 2 overwrites duplicates by canonical_id)
    shot1_gps = {gp["canonical_id"]: gp for gp in shot1.get("grammar_points", [])}
    shot2_gps = {gp["canonical_id"]: gp for gp in shot2.get("grammar_points", [])}
    shot1_gps.update(shot2_gps)  # Shot 2 overwrites
    result["grammar_points"] = list(shot1_gps.values())

    # Merge vocabulary (Shot 2 overwrites duplicates by word)
    shot1_vocab = {v["word"]: v for v in shot1.get("vocabulary", [])}
    shot2_vocab = {v["word"]: v for v in shot2.get("vocabulary", [])}
    shot1_vocab.update(shot2_vocab)
    result["vocabulary"] = list(shot1_vocab.values())

    # Average difficulty factors
    factors1 = {f["factor"]: f for f in shot1.get("difficulty_factors", [])}
    factors2 = {f["factor"]: f for f in shot2.get("difficulty_factors", [])}

    merged_factors = {}
    all_factor_names = set(factors1.keys()) | set(factors2.keys())
    for fname in all_factor_names:
        f1 = factors1.get(fname, {"weight": 0.0})
        f2 = factors2.get(fname, {"weight": 0.0})

        # If factor exists in both, average; otherwise take the one that exists
        if fname in factors1 and fname in factors2:
            avg_weight = (f1.get("weight", 0.0) + f2.get("weight", 0.0)) / 2.0
        else:
            avg_weight = f1.get("weight", 0.0) or f2.get("weight", 0.0)

        merged_factors[fname] = {
            "factor": fname,
            "weight": avg_weight,
            "note": f2.get("note") or f1.get("note")  # Prefer Shot 2 note
        }

    result["difficulty_factors"] = list(merged_factors.values())

    # Prefer Shot 2 semantic notes if available
    if shot2.get("semantic_notes"):
        result["semantic_notes"] = shot2["semantic_notes"]

    return result


# -----------------------------------------------------------------------------
# Main Annotation Pipeline
# -----------------------------------------------------------------------------

def annotate_poem(
    poem: dict,
    tagger: fugashi.Tagger,
    client: GeminiClient,
    prompt_config: dict,
    use_cache: bool = True,
    prompt_config_review: dict | None = None,
    use_two_shot: bool = True
) -> PoemAnnotation | None:
    """
    Annotate a single poem with optional 2-shot approach.

    Args:
        poem: Raw poem dict
        tagger: Fugashi tagger
        client: Gemini client
        prompt_config: Loaded prompt configuration (Shot 1)
        use_cache: Whether to use response caching
        prompt_config_review: Review prompt configuration (Shot 2), None to disable
        use_two_shot: Whether to use 2-shot annotation (requires prompt_config_review)

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

    tokens_json = tokens_to_json(tokens)
    text_hash = poem.get("text_hash", PoemAnnotation.compute_text_hash(text))

    # Determine if we can do 2-shot
    can_two_shot = use_two_shot and prompt_config_review is not None

    # --- SHOT 1: Initial Annotation ---
    cache_key_shot1 = get_cache_key(poem_id, client.model_name, prompt_config["meta"]["version"], shot=1)

    shot1_response = None
    if use_cache:
        cached = load_cached_response(cache_key_shot1)
        if cached:
            logger.debug(f"Using cached Shot 1 response for {poem_id}")
            shot1_response = cached

    if shot1_response is None:
        # Build Shot 1 prompt
        user_prompt = format_prompt(
            prompt_config["user_template"],
            poem_text=text,
            text_hash=text_hash,
            fugashi_tokens_json=tokens_json,
            source=poem["source"],
            author=poem.get("author") or "unknown",
            collection=poem.get("collection") or "unknown"
        )

        try:
            response_text = client.generate(
                prompt_config["system"],
                user_prompt
            )
            shot1_response = extract_json_from_response(response_text)

            if use_cache:
                save_cached_response(cache_key_shot1, shot1_response)

        except Exception as e:
            logger.error(f"Shot 1 failed for {poem_id}: {e}")
            return None

    # Validate Shot 1
    try:
        shot1_fixed = validate_and_fix_annotation(shot1_response, poem, tokens)
    except Exception as e:
        logger.error(f"Shot 1 validation failed for {poem_id}: {e}")
        return None

    # --- SHOT 2: Review & Correction (if enabled) ---
    if can_two_shot:
        cache_key_shot2 = get_cache_key(poem_id, client.model_name, prompt_config_review["meta"]["version"], shot=2)

        shot2_response = None
        if use_cache:
            cached = load_cached_response(cache_key_shot2)
            if cached:
                logger.debug(f"Using cached Shot 2 response for {poem_id}")
                shot2_response = cached

        if shot2_response is None:
            # Build Shot 2 prompt with Shot 1 output
            shot1_json = json.dumps(shot1_fixed, ensure_ascii=False, indent=2)

            user_prompt_review = format_prompt(
                prompt_config_review["user_template"],
                poem_text=text,
                text_hash=text_hash,
                kanji_transcription=shot1_fixed.get("kanji_transcription", text),
                fugashi_tokens_json=tokens_json,
                shot1_json=shot1_json,
                source=poem["source"],
                author=poem.get("author") or "unknown",
                collection=poem.get("collection") or "unknown"
            )

            try:
                response_text = client.generate(
                    prompt_config_review["system"],
                    user_prompt_review
                )
                shot2_response = extract_json_from_response(response_text)

                if use_cache:
                    save_cached_response(cache_key_shot2, shot2_response)

            except Exception as e:
                logger.warning(f"Shot 2 failed for {poem_id}: {e}, using Shot 1 only")
                # Fall back to Shot 1 only
                return build_poem_annotation(poem, tokens, shot1_fixed)

        # Validate Shot 2
        try:
            shot2_fixed = validate_and_fix_annotation(shot2_response, poem, tokens)
        except Exception as e:
            logger.warning(f"Shot 2 validation failed for {poem_id}: {e}, using Shot 1 only")
            return build_poem_annotation(poem, tokens, shot1_fixed)

        # Merge both shots
        merged = merge_annotations(shot1_fixed, shot2_fixed)
        return build_poem_annotation(poem, tokens, merged)

    else:
        # Single-shot mode
        return build_poem_annotation(poem, tokens, shot1_fixed)


def _annotation_to_dict(annotation: PoemAnnotation) -> dict:
    """Convert PoemAnnotation to dict for DataFrame storage."""
    ann_dict = annotation.model_dump()
    ann_dict["fugashi_tokens"] = [t.model_dump() for t in annotation.fugashi_tokens]
    ann_dict["token_readings"] = [t.model_dump() for t in annotation.token_readings]
    ann_dict["grammar_points"] = [g.model_dump() for g in annotation.grammar_points]
    ann_dict["vocabulary"] = [v.model_dump() for v in annotation.vocabulary]
    ann_dict["difficulty_factors"] = [d.model_dump() for d in annotation.difficulty_factors]
    return ann_dict


def _annotate_worker(
    poem: dict,
    client: GeminiClient,
    prompt_config: dict,
    use_cache: bool,
    prompt_config_review: dict | None = None,
    use_two_shot: bool = True
) -> tuple[str, PoemAnnotation | None, str | None]:
    """
    Worker function for parallel annotation with 2-shot support.

    Returns:
        Tuple of (poem_id, annotation_or_none, error_message_or_none)
    """
    poem_id = poem["poem_id"]
    try:
        tagger = get_thread_tagger()
        annotation = annotate_poem(
            poem, tagger, client, prompt_config, use_cache,
            prompt_config_review=prompt_config_review,
            use_two_shot=use_two_shot
        )
        return (poem_id, annotation, None)
    except Exception as e:
        return (poem_id, None, str(e))


def annotate_corpus(
    input_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_poems: int | None = None,
    resume: bool = False,
    use_cache: bool = True,
    api_sleep: float = DEFAULT_API_SLEEP,
    workers: int = DEFAULT_WORKERS,
    use_two_shot: bool = True
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
        workers: Number of parallel workers (default: 1)
        use_two_shot: Whether to use 2-shot annotation (default: True)

    Returns:
        Number of poems successfully annotated
    """
    # Initialize
    logger.info(f"Loading prompt configuration...")
    prompt_config = load_prompt("annotate")

    # Load review prompt for 2-shot (if enabled)
    prompt_config_review = None
    if use_two_shot:
        try:
            prompt_config_review = load_prompt("annotate_review")
            logger.info("Loaded review prompt for 2-shot annotation")
        except Exception as e:
            logger.warning(f"Could not load review prompt: {e}, falling back to single-shot")
            use_two_shot = False

    if use_two_shot:
        logger.info("Using 2-shot annotation (initial + review)")
    else:
        logger.info("Using single-shot annotation")

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

    # Thread-safe lock for shared state
    lock = threading.Lock()

    def process_result(poem_id: str, annotation: PoemAnnotation | None, error: str | None):
        """Process a completed annotation result (thread-safe)."""
        nonlocal success_count, error_count

        with lock:
            if annotation:
                ann_dict = _annotation_to_dict(annotation)
                annotations.append(ann_dict)
                success_count += 1
                save_checkpoint(checkpoint_file, poem_id)

                # Save intermediate results
                if (success_count % batch_size) == 0:
                    logger.info(f"Saving checkpoint ({success_count} poems)...")
                    df = pd.DataFrame(annotations)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(output_path, index=False)
            else:
                error_count += 1
                if error:
                    logger.error(f"Failed to annotate {poem_id}: {error}")
                else:
                    logger.warning(f"Failed to annotate {poem_id}")

    if workers > 1:
        # Parallel processing
        logger.info(f"Starting parallel annotation with {workers} workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _annotate_worker, poem, client, prompt_config, use_cache,
                    prompt_config_review, use_two_shot
                ): poem
                for poem in poems
            }

            # Process completed tasks with timeout handling
            # Overall timeout scales with number of poems: (poems/workers) * timeout_per_poem * 2 (for 2-shot)
            overall_timeout = max(300, (len(poems) / workers) * WORKER_TIMEOUT * 2 + 60)
            logger.info(f"Overall timeout: {overall_timeout:.0f}s for {len(poems)} poems with {workers} workers")

            completed = 0
            timed_out = 0
            try:
                for future in as_completed(futures, timeout=overall_timeout):
                    poem = futures[future]
                    completed += 1

                    try:
                        # Add timeout to result retrieval
                        poem_id, annotation, error = future.result(timeout=WORKER_TIMEOUT)
                        logger.info(f"[{completed}/{len(poems)}] Completed {poem_id}")
                        process_result(poem_id, annotation, error)
                    except FuturesTimeoutError:
                        logger.warning(f"[{completed}/{len(poems)}] Timeout for {poem['poem_id']}, skipping")
                        timed_out += 1
                        with lock:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"[{completed}/{len(poems)}] Worker exception for {poem['poem_id']}: {e}")
                        with lock:
                            error_count += 1

            except FuturesTimeoutError:
                # Overall timeout for as_completed - some futures may still be pending
                pending = [f for f in futures if not f.done()]
                logger.warning(f"Overall timeout reached. {len(pending)} tasks still pending, cancelling...")
                for future in pending:
                    future.cancel()
                timed_out += len(pending)

            except KeyboardInterrupt:
                logger.info("Interrupted by user, cancelling remaining tasks...")
                for future in futures:
                    future.cancel()

            if timed_out > 0:
                logger.warning(f"Total timed out poems: {timed_out}")
    else:
        # Sequential processing (original behavior)
        logger.info("Starting sequential annotation...")
        tagger = fugashi.Tagger()

        for i, poem in enumerate(poems):
            poem_id = poem["poem_id"]

            try:
                logger.info(f"[{i+1}/{len(poems)}] Annotating {poem_id}...")
                annotation = annotate_poem(
                    poem, tagger, client, prompt_config, use_cache,
                    prompt_config_review=prompt_config_review,
                    use_two_shot=use_two_shot
                )
                process_result(poem_id, annotation, None)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing {poem_id}: {e}")
                error_count += 1

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
  python scripts/02_annotate_corpus.py --single-shot  # Faster, single-shot annotation
  python scripts/02_annotate_corpus.py --max-poems 3 --no-cache  # Test 2-shot with 3 poems
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
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers for annotation (default: {DEFAULT_WORKERS}). Use 5-10 for faster processing.",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Disable 2-shot annotation (use only initial pass, faster but less accurate)",
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
        api_sleep=args.api_sleep,
        workers=args.workers,
        use_two_shot=not args.single_shot
    )

    print(f"\nAnnotated {count} poems to {args.output}")

    # Verification hint
    print(f"\nVerify with:")
    print(f'  python -c "import pandas as pd; df = pd.read_parquet(\'{args.output}\'); print(f\'Poems: {{len(df)}}\'); print(df.columns.tolist())"')


if __name__ == "__main__":
    main()
