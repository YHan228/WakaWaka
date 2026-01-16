#!/usr/bin/env python3
"""
06_annotate_literary.py - Literary analysis annotation for poems.

This script processes annotated poems from data/annotated/poems.parquet and
produces literary analysis in data/literary/poems_literary.parquet.

This is SEPARATE from grammar annotations - it focuses on:
- Poem interpretation and meaning
- Literary techniques and how they function
- Poetic devices (kakekotoba, makurakotoba, etc.)
- Cultural and historical context
- Comparisons to Chinese poetry

Key features:
- Parallel processing with configurable workers
- Batch checkpointing (resume on failure)
- LLM response caching
- Schema validation

Usage:
  python scripts/06_annotate_literary.py --workers 5 --resume
  python scripts/06_annotate_literary.py --max-poems 10  # Quick test
  python scripts/06_annotate_literary.py --model gemini-2.5-flash-preview-05-20
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

# Import Google GenAI
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

from wakawaka.schemas.literary import (
    PoemLiteraryAnalysis,
    PoeticDevice,
    Allusion,
    ImageryNote,
)
from wakawaka.utils.prompt_loader import load_prompt, format_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_BATCH_SIZE = 10
DEFAULT_API_SLEEP = 0.5
DEFAULT_WORKERS = 1
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "literary" / ".checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "literary" / ".cache"


# -----------------------------------------------------------------------------
# Gemini API Client
# -----------------------------------------------------------------------------

class GeminiClient:
    """Wrapper for Gemini API with rate limiting and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
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
        """Generate text using Gemini API."""
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

                time.sleep(self.sleep_seconds)

                if response.text is None:
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            return candidate.content.parts[0].text
                    raise ValueError("Empty response from API")

                return response.text

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.sleep_seconds * (attempt + 1) * 2)
                else:
                    raise


# -----------------------------------------------------------------------------
# JSON Extraction and Validation
# -----------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try raw JSON
    text = text.strip()
    if text.startswith('{'):
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

    # Last resort
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not extract JSON: {e}\n\nResponse:\n{text[:500]}...")


def validate_and_fix_analysis(llm_response: dict, poem_id: str, text: str) -> dict:
    """Validate and fix LLM literary analysis response."""

    # Ensure poem_id and text
    llm_response["poem_id"] = poem_id
    llm_response["text"] = text

    # Ensure required fields
    if "interpretation" not in llm_response or not llm_response["interpretation"]:
        llm_response["interpretation"] = "Interpretation not provided."

    if "emotional_tone" not in llm_response or not llm_response["emotional_tone"]:
        llm_response["emotional_tone"] = "Tone not analyzed."

    if "literary_techniques" not in llm_response or not llm_response["literary_techniques"]:
        llm_response["literary_techniques"] = "Techniques not analyzed."

    # Ensure list fields are lists
    for field in ["poetic_devices", "allusions"]:
        if field not in llm_response:
            llm_response[field] = []
        elif not isinstance(llm_response[field], list):
            llm_response[field] = []

    # Handle imagery_analysis
    if "imagery_analysis" in llm_response:
        if llm_response["imagery_analysis"] is None:
            del llm_response["imagery_analysis"]
        elif not isinstance(llm_response["imagery_analysis"], list):
            del llm_response["imagery_analysis"]

    # Validate poetic_devices structure
    valid_devices = []
    for device in llm_response.get("poetic_devices", []):
        if isinstance(device, dict) and "name" in device and "explanation" in device:
            valid_devices.append({
                "name": device.get("name", ""),
                "location": device.get("location"),
                "explanation": device.get("explanation", ""),
                "effect": device.get("effect")
            })
    llm_response["poetic_devices"] = valid_devices

    # Validate allusions structure
    valid_allusions = []
    for allusion in llm_response.get("allusions", []):
        if isinstance(allusion, dict) and "reference" in allusion and "connection" in allusion:
            valid_allusions.append({
                "reference": allusion.get("reference", ""),
                "connection": allusion.get("connection", "")
            })
    llm_response["allusions"] = valid_allusions

    # Validate imagery_analysis structure
    if "imagery_analysis" in llm_response:
        valid_imagery = []
        for img in llm_response.get("imagery_analysis", []):
            if isinstance(img, dict) and "image" in img and "significance" in img:
                valid_imagery.append({
                    "image": img.get("image", ""),
                    "significance": img.get("significance", "")
                })
        if valid_imagery:
            llm_response["imagery_analysis"] = valid_imagery
        else:
            del llm_response["imagery_analysis"]

    # Remove null optional fields
    optional_fields = [
        "structure_and_flow", "sound_and_rhythm", "seasonal_context",
        "historical_background", "cultural_notes", "chinese_poetry_parallel",
        "critical_notes"
    ]
    for field in optional_fields:
        if field in llm_response and llm_response[field] is None:
            del llm_response[field]

    return llm_response


def build_literary_analysis(llm_response: dict) -> PoemLiteraryAnalysis:
    """Build a PoemLiteraryAnalysis from validated LLM response."""

    # Build poetic devices
    poetic_devices = [
        PoeticDevice(**d) for d in llm_response.get("poetic_devices", [])
    ]

    # Build allusions
    allusions = [
        Allusion(**a) for a in llm_response.get("allusions", [])
    ]

    # Build imagery analysis
    imagery_analysis = None
    if "imagery_analysis" in llm_response:
        imagery_analysis = [
            ImageryNote(**i) for i in llm_response["imagery_analysis"]
        ]

    return PoemLiteraryAnalysis(
        poem_id=llm_response["poem_id"],
        text=llm_response["text"],
        interpretation=llm_response["interpretation"],
        emotional_tone=llm_response["emotional_tone"],
        literary_techniques=llm_response["literary_techniques"],
        imagery_analysis=imagery_analysis,
        structure_and_flow=llm_response.get("structure_and_flow"),
        sound_and_rhythm=llm_response.get("sound_and_rhythm"),
        poetic_devices=poetic_devices,
        seasonal_context=llm_response.get("seasonal_context"),
        historical_background=llm_response.get("historical_background"),
        allusions=allusions,
        cultural_notes=llm_response.get("cultural_notes"),
        chinese_poetry_parallel=llm_response.get("chinese_poetry_parallel"),
        critical_notes=llm_response.get("critical_notes")
    )


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------

def get_cache_key(poem_id: str, model: str, prompt_version: str) -> str:
    """Generate cache key for LLM response."""
    key = f"literary_{poem_id}_{model}_{prompt_version}"
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
# Helper Functions
# -----------------------------------------------------------------------------

def summarize_vocabulary(vocabulary) -> str:
    """Summarize vocabulary list for prompt."""
    # Handle numpy arrays and None
    if vocabulary is None:
        return "None extracted"
    try:
        vocab_list = list(vocabulary) if hasattr(vocabulary, '__iter__') and not isinstance(vocabulary, (str, dict)) else []
    except (TypeError, ValueError):
        return "None extracted"

    if not vocab_list:
        return "None extracted"

    items = []
    for v in vocab_list[:10]:
        if isinstance(v, dict):
            word = v.get("word", "")
            meaning = v.get("meaning", "")
            items.append(f"{word}: {meaning}")

    result = "; ".join(items)
    if len(vocab_list) > 10:
        result += f" ... and {len(vocab_list) - 10} more"
    return result if items else "None extracted"


def summarize_grammar(grammar_points) -> str:
    """Summarize grammar points for prompt."""
    # Handle numpy arrays and None
    if grammar_points is None:
        return "None extracted"
    try:
        gp_list = list(grammar_points) if hasattr(grammar_points, '__iter__') and not isinstance(grammar_points, (str, dict)) else []
    except (TypeError, ValueError):
        return "None extracted"

    if not gp_list:
        return "None extracted"

    items = []
    for gp in gp_list[:8]:
        if isinstance(gp, dict):
            canonical_id = gp.get("canonical_id", "")
            surface = gp.get("surface", "")
            items.append(f"{surface} ({canonical_id})")

    result = "; ".join(items)
    if len(gp_list) > 8:
        result += f" ... and {len(gp_list) - 8} more"
    return result if items else "None extracted"


# -----------------------------------------------------------------------------
# Main Annotation Pipeline
# -----------------------------------------------------------------------------

def analyze_poem(
    poem_row: dict,
    client: GeminiClient,
    prompt_config: dict,
    use_cache: bool = True
) -> PoemLiteraryAnalysis | None:
    """
    Generate literary analysis for a single poem.

    Args:
        poem_row: Row from annotated poems parquet
        client: Gemini client
        prompt_config: Loaded prompt configuration
        use_cache: Whether to use response caching

    Returns:
        PoemLiteraryAnalysis or None on failure
    """
    poem_id = poem_row["poem_id"]
    text = poem_row["text"]

    # Check cache
    cache_key = get_cache_key(poem_id, client.model_name, prompt_config["meta"]["version"])

    if use_cache:
        cached = load_cached_response(cache_key)
        if cached:
            logger.debug(f"Using cached response for {poem_id}")
            try:
                fixed = validate_and_fix_analysis(cached, poem_id, text)
                return build_literary_analysis(fixed)
            except Exception as e:
                logger.warning(f"Cached response invalid for {poem_id}: {e}")

    # Build prompt with context
    vocabulary = poem_row.get("vocabulary", [])
    grammar_points = poem_row.get("grammar_points", [])

    user_prompt = format_prompt(
        prompt_config["user_template"],
        poem_id=poem_id,
        text=text,
        reading_hiragana=poem_row.get("reading_hiragana", ""),
        reading_romaji=poem_row.get("reading_romaji", ""),
        author=poem_row.get("author") or "Unknown",
        collection=poem_row.get("collection") or "Unknown",
        vocabulary_summary=summarize_vocabulary(vocabulary),
        grammar_summary=summarize_grammar(grammar_points),
        semantic_notes=poem_row.get("semantic_notes") or "None"
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

        # Validate and build
        fixed = validate_and_fix_analysis(llm_response, poem_id, text)
        return build_literary_analysis(fixed)

    except Exception as e:
        logger.error(f"Failed to analyze poem {poem_id}: {e}")
        return None


def _analysis_to_dict(analysis: PoemLiteraryAnalysis) -> dict:
    """Convert PoemLiteraryAnalysis to dict for DataFrame storage."""
    data = analysis.model_dump()

    # Nested objects are already dicts from model_dump
    return data


def _analyze_worker(
    poem_row: dict,
    client: GeminiClient,
    prompt_config: dict,
    use_cache: bool
) -> tuple[str, PoemLiteraryAnalysis | None, str | None]:
    """Worker function for parallel analysis."""
    poem_id = poem_row["poem_id"]
    try:
        analysis = analyze_poem(poem_row, client, prompt_config, use_cache)
        return (poem_id, analysis, None)
    except Exception as e:
        return (poem_id, None, str(e))


def analyze_corpus(
    input_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_poems: int | None = None,
    resume: bool = False,
    use_cache: bool = True,
    api_sleep: float = DEFAULT_API_SLEEP,
    workers: int = DEFAULT_WORKERS
) -> int:
    """
    Generate literary analysis for all poems.

    Args:
        input_path: Path to annotated poems parquet
        output_path: Path to output parquet file
        model: Gemini model name
        batch_size: Poems per checkpoint save
        max_poems: Maximum poems to process
        resume: Whether to resume from checkpoint
        use_cache: Whether to use LLM response caching
        api_sleep: Seconds between API calls
        workers: Number of parallel workers

    Returns:
        Number of poems successfully analyzed
    """
    # Initialize
    logger.info("Loading prompt configuration...")
    prompt_config = load_prompt("literary_analysis")

    logger.info(f"Initializing Gemini client (model: {model})...")
    client = GeminiClient(model=model, sleep_seconds=api_sleep)

    # Checkpointing
    checkpoint_file = CHECKPOINT_DIR / f"literary_{output_path.stem}.txt"
    processed_ids = load_checkpoint(checkpoint_file) if resume else set()

    if resume and processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} poems already processed")

    # Load input poems
    logger.info(f"Loading annotated poems from {input_path}...")
    df = pd.read_parquet(input_path)
    poems = df.to_dict(orient="records")
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
    analyses = []
    success_count = 0
    error_count = 0

    # Load existing analyses if resuming
    if resume and output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            analyses = existing_df.to_dict(orient="records")
            logger.info(f"Loaded {len(analyses)} existing analyses")
        except Exception as e:
            logger.warning(f"Could not load existing analyses: {e}")

    # Thread-safe lock
    lock = threading.Lock()

    def process_result(poem_id: str, analysis: PoemLiteraryAnalysis | None, error: str | None):
        """Process a completed analysis result (thread-safe)."""
        nonlocal success_count, error_count

        with lock:
            if analysis:
                ann_dict = _analysis_to_dict(analysis)
                analyses.append(ann_dict)
                success_count += 1
                save_checkpoint(checkpoint_file, poem_id)

                # Save intermediate results
                if (success_count % batch_size) == 0:
                    logger.info(f"Saving checkpoint ({success_count} poems)...")
                    save_df = pd.DataFrame(analyses)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    save_df.to_parquet(output_path, index=False)
            else:
                error_count += 1
                if error:
                    logger.error(f"Failed to analyze {poem_id}: {error}")

    if workers > 1:
        # Parallel processing
        logger.info(f"Starting parallel analysis with {workers} workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_analyze_worker, poem, client, prompt_config, use_cache): poem
                for poem in poems
            }

            completed = 0
            try:
                for future in as_completed(futures):
                    poem = futures[future]
                    completed += 1

                    try:
                        poem_id, analysis, error = future.result()
                        logger.info(f"[{completed}/{len(poems)}] Completed {poem_id}")
                        process_result(poem_id, analysis, error)
                    except Exception as e:
                        logger.error(f"[{completed}/{len(poems)}] Worker exception: {e}")
                        with lock:
                            error_count += 1

            except KeyboardInterrupt:
                logger.info("Interrupted by user, cancelling remaining tasks...")
                for future in futures:
                    future.cancel()
    else:
        # Sequential processing
        logger.info("Starting sequential analysis...")

        for i, poem in enumerate(poems):
            poem_id = poem["poem_id"]

            try:
                logger.info(f"[{i+1}/{len(poems)}] Analyzing {poem_id}...")
                analysis = analyze_poem(poem, client, prompt_config, use_cache)
                process_result(poem_id, analysis, None)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing {poem_id}: {e}")
                error_count += 1

    # Final save
    if analyses:
        logger.info(f"Saving final results ({len(analyses)} analyses)...")
        final_df = pd.DataFrame(analyses)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_path, index=False)

    logger.info(f"Analysis complete: {success_count} success, {error_count} errors")
    return success_count


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate literary analysis for poems using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/06_annotate_literary.py --workers 5 --resume
  python scripts/06_annotate_literary.py --max-poems 10  # Quick test
  python scripts/06_annotate_literary.py --model gemini-2.5-flash-preview-05-20
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Input parquet with annotated poems (default: data/annotated/poems.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "literary" / "poems_literary.parquet",
        help="Output parquet file (default: data/literary/poems_literary.parquet)",
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
        help="Maximum poems to analyze (default: all)",
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
        help=f"Parallel workers (default: {DEFAULT_WORKERS}). Use 5-10 for faster processing.",
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

    # Run analysis
    count = analyze_corpus(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        batch_size=args.batch_size,
        max_poems=args.max_poems,
        resume=args.resume,
        use_cache=not args.no_cache,
        api_sleep=args.api_sleep,
        workers=args.workers
    )

    print(f"\nAnalyzed {count} poems to {args.output}")

    # Verification hint
    print(f"\nVerify with:")
    print(f'  python -c "import pandas as pd; df = pd.read_parquet(\'{args.output}\'); print(f\'Poems: {{len(df)}}\'); print(df[[\'poem_id\', \'interpretation\']].head())"')


if __name__ == "__main__":
    main()
