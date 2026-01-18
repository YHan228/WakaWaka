#!/usr/bin/env python3
"""
backfill_kanji_transcription.py - Add kanji transcription to existing annotated poems.

This one-time script adds kanji_transcription field to poems that are mostly hiragana.
For poems that already have kanji or where transcription is uncertain, it keeps the original text.

Usage:
  python scripts/backfill_kanji_transcription.py --input data/annotated/poems.parquet
  python scripts/backfill_kanji_transcription.py --dry-run  # Preview without API calls
  python scripts/backfill_kanji_transcription.py --workers 10
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
import numpy as np

# Import Google GenAI
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "gemini-2.0-flash"
WORKER_TIMEOUT = 60  # seconds per poem
HIRAGANA_THRESHOLD = 0.80  # Only transcribe poems with >80% hiragana
CACHE_DIR = PROJECT_ROOT / "data" / "annotated" / ".cache_kanji"

# Transcription prompt
TRANSCRIPTION_SYSTEM = """你是古典日语专家。任务：将纯平假名和歌转写为汉字版本。

**规则:**
1. 将和语词汇转写为对应汉字（如 "やま" → "山"、"はな" → "花"）
2. 保持助词/助动词为假名（如 "に"、"を"、"けり"、"らむ"）
3. 若原文已含足够汉字，直接返回原文
4. 若无法确定正确汉字，直接返回原文，不要猜测
5. 保持诗歌的原有分词和换行

**示例:**
- "つひにゆくみちとはかねてききしかど" → "終に行く道とは兼ねて聞きしかど"
- "はるすぎてなつきにけらし" → "春過ぎて夏来にけらし"
- "あきのたのかりほのいほのとまをあらみ" → "秋の田のかりほの庵のとまを荒み"

仅返回转写结果，不要任何解释。"""

TRANSCRIPTION_USER = """将此和歌转写为汉字版本（若无法确定则返回原文）:

{text}"""


def hiragana_ratio(text: str) -> float:
    """Calculate the ratio of hiragana characters in text."""
    if not text:
        return 0.0
    hiragana_count = sum(1 for c in text if '\u3040' <= c <= '\u309f')
    return hiragana_count / len(text)


def get_cache_path(poem_id: str) -> Path:
    """Get cache file path for a poem."""
    return CACHE_DIR / f"{poem_id}_kanji.json"


def load_cached_transcription(poem_id: str) -> str | None:
    """Load cached transcription if available."""
    cache_path = get_cache_path(poem_id)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("kanji_transcription")
        except Exception:
            pass
    return None


def save_cached_transcription(poem_id: str, transcription: str):
    """Save transcription to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(poem_id)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({"kanji_transcription": transcription}, f, ensure_ascii=False)


class GeminiClient:
    """Simple Gemini client for transcription."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model_name = model
        http_options = genai_types.HttpOptions(timeout=WORKER_TIMEOUT * 1000)
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def transcribe(self, text: str) -> str:
        """Get kanji transcription for a poem."""
        user_prompt = TRANSCRIPTION_USER.format(text=text)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=f"{TRANSCRIPTION_SYSTEM}\n\n{user_prompt}")]
                )
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500,
            )
        )

        if response.text:
            return response.text.strip()
        return text  # Return original if no response


def transcribe_poem(
    poem_id: str,
    text: str,
    client: GeminiClient,
    use_cache: bool = True
) -> tuple[str, str, str | None]:
    """
    Transcribe a single poem.

    Returns:
        Tuple of (poem_id, transcription, error_message)
    """
    try:
        # Check cache first
        if use_cache:
            cached = load_cached_transcription(poem_id)
            if cached:
                return (poem_id, cached, None)

        # Call LLM
        transcription = client.transcribe(text)

        # Validate: transcription should not be empty or much longer than original
        if not transcription or len(transcription) > len(text) * 1.5:
            transcription = text

        # Cache result
        if use_cache:
            save_cached_transcription(poem_id, transcription)

        return (poem_id, transcription, None)

    except Exception as e:
        return (poem_id, text, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill kanji transcription for existing poems"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Input parquet file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output parquet file (default: overwrite input)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=20,
        help="Number of parallel workers (default: 20)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=HIRAGANA_THRESHOLD,
        help=f"Hiragana ratio threshold for transcription (default: {HIRAGANA_THRESHOLD})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without API calls"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading poems from {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} poems")

    # Calculate hiragana ratio for each poem
    df['_hiragana_ratio'] = df['text'].apply(hiragana_ratio)

    # Find poems needing transcription
    needs_transcription = df[df['_hiragana_ratio'] > args.threshold]
    already_has_kanji = df[df['_hiragana_ratio'] <= args.threshold]

    logger.info(f"Poems needing transcription (>{args.threshold:.0%} hiragana): {len(needs_transcription)}")
    logger.info(f"Poems already with kanji: {len(already_has_kanji)}")

    if args.dry_run:
        logger.info("\n=== DRY RUN - Preview ===")
        for _, row in needs_transcription.head(5).iterrows():
            print(f"\n{row['poem_id']} ({row['_hiragana_ratio']:.1%} hiragana):")
            print(f"  {row['text']}")
        logger.info(f"\n... and {len(needs_transcription) - 5} more poems")
        return

    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment")
        sys.exit(1)

    client = GeminiClient(api_key, args.model)
    logger.info(f"Using model: {args.model}")

    # Initialize kanji_transcription column
    if 'kanji_transcription' not in df.columns:
        df['kanji_transcription'] = df['text']  # Default to original text

    # For poems that already have kanji, set transcription = original
    df.loc[df['_hiragana_ratio'] <= args.threshold, 'kanji_transcription'] = df.loc[df['_hiragana_ratio'] <= args.threshold, 'text']

    # Process poems needing transcription
    poems_to_process = [
        (row['poem_id'], row['text'])
        for _, row in needs_transcription.iterrows()
    ]

    use_cache = not args.no_cache
    success_count = 0
    error_count = 0

    logger.info(f"Processing {len(poems_to_process)} poems with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(transcribe_poem, pid, text, client, use_cache): pid
            for pid, text in poems_to_process
        }

        try:
            for future in as_completed(futures, timeout=WORKER_TIMEOUT * len(poems_to_process) / args.workers + 60):
                poem_id = futures[future]
                try:
                    pid, transcription, error = future.result(timeout=WORKER_TIMEOUT)

                    if error:
                        logger.warning(f"Error for {pid}: {error}")
                        error_count += 1
                    else:
                        # Update dataframe
                        df.loc[df['poem_id'] == pid, 'kanji_transcription'] = transcription
                        success_count += 1

                        if success_count % 50 == 0:
                            logger.info(f"Progress: {success_count}/{len(poems_to_process)}")

                except FuturesTimeoutError:
                    logger.warning(f"Timeout for {poem_id}")
                    error_count += 1
                except Exception as e:
                    logger.warning(f"Error for {poem_id}: {e}")
                    error_count += 1

        except FuturesTimeoutError:
            logger.warning("Overall timeout reached")

    logger.info(f"Completed: {success_count} success, {error_count} errors")

    # Clean up temporary column
    df = df.drop(columns=['_hiragana_ratio'])

    # Save result
    output_path = args.output or args.input
    logger.info(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)

    # Show sample results
    logger.info("\n=== Sample Transcriptions ===")
    sample = df[df['text'] != df['kanji_transcription']].head(5)
    for _, row in sample.iterrows():
        print(f"\n{row['poem_id']}:")
        print(f"  Original: {row['text']}")
        print(f"  Kanji:    {row['kanji_transcription']}")

    logger.info(f"\nDone! {len(df[df['text'] != df['kanji_transcription']])} poems have kanji transcription different from original")


if __name__ == "__main__":
    main()
