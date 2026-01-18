#!/usr/bin/env python3
"""
07_standardize_season_theme.py - Standardize season/theme labels for literary annotations.

Adds standardized season and theme labels to literary annotations using LLM extraction.
Output is a simple JSON mapping: {poem_id: {season, themes}}.

Usage:
  python scripts/07_standardize_season_theme.py --workers 10
  python scripts/07_standardize_season_theme.py --resume
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

from google import genai
from google.genai import types as genai_types

from wakawaka.utils.prompt_loader import load_prompt, format_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_WORKERS = 50
WORKER_TIMEOUT = 30
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "literary" / ".checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "literary" / ".cache_season"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "literary" / "poems_literary.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "literary" / "standardized_season_theme.json"


class GeminiClient:
    """Minimal Gemini API wrapper."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        http_options = genai_types.HttpOptions(timeout=30000)
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model = model
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(temperature=self.temperature),
                )
                return response.text or ""
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


def extract_json(text: str) -> dict:
    """Extract JSON from response."""
    text = text.strip()
    # Try code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return json.loads(match.group(1).strip())
    # Try raw JSON
    if text.startswith('{'):
        brace_count = 0
        for i, c in enumerate(text):
            if c == '{': brace_count += 1
            elif c == '}': brace_count -= 1
            if brace_count == 0:
                return json.loads(text[:i+1])
    return json.loads(text)


def get_cache_key(poem_id: str, model: str, version: str) -> str:
    return hashlib.md5(f"{poem_id}_{model}_{version}".encode()).hexdigest()


def load_cache(cache_key: str) -> dict | None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.load(open(cache_file, encoding="utf-8"))
        except:
            pass
    return None


def save_cache(cache_key: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(CACHE_DIR / f"{cache_key}.json", "w", encoding="utf-8"), ensure_ascii=False)


def load_checkpoint(checkpoint_file: Path) -> set[str]:
    if checkpoint_file.exists():
        return {line.strip() for line in open(checkpoint_file, encoding="utf-8") if line.strip()}
    return set()


def save_checkpoint(checkpoint_file: Path, poem_id: str):
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(f"{poem_id}\n")


def process_poem(poem: dict, client: GeminiClient, prompt_config: dict, use_cache: bool) -> tuple[str, dict | None, str | None]:
    """Process single poem. Returns (poem_id, result_or_none, error_or_none)."""
    poem_id = poem["poem_id"]
    try:
        cache_key = get_cache_key(poem_id, client.model, prompt_config["meta"]["version"])
        if use_cache:
            cached = load_cache(cache_key)
            if cached:
                return (poem_id, cached, None)

        user_prompt = format_prompt(
            prompt_config["user_template"],
            poem_id=poem_id,
            text=poem.get("text", ""),
            seasonal_context=poem.get("seasonal_context", "") or "",
            interpretation=(poem.get("interpretation", "") or "")[:500],
        )

        response = client.generate(prompt_config["system"], user_prompt)
        result = extract_json(response)

        # Validate
        season = result.get("season", "無季")
        if season not in ["春", "夏", "秋", "冬", "無季"]:
            season = "無季"
        themes = result.get("themes", ["其他"])
        if not isinstance(themes, list):
            themes = [themes] if themes else ["其他"]

        data = {"season": season, "themes": themes}
        if use_cache:
            save_cache(cache_key, data)
        return (poem_id, data, None)

    except Exception as e:
        return (poem_id, None, str(e))


def main():
    parser = argparse.ArgumentParser(description="Standardize season/theme labels")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)

    logger.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    poems = df.to_dict(orient="records")
    logger.info(f"Loaded {len(poems)} poems")

    prompt_config = load_prompt("standardize_season_theme")
    client = GeminiClient(model=args.model, temperature=prompt_config["meta"].get("temperature", 0.1))

    checkpoint_file = CHECKPOINT_DIR / "season_theme.txt"
    processed_ids = load_checkpoint(checkpoint_file) if args.resume else set()

    # Load existing results if resuming
    results = {}
    if args.resume and args.output.exists():
        try:
            results = json.load(open(args.output, encoding="utf-8"))
            logger.info(f"Loaded {len(results)} existing results")
        except:
            pass

    if args.resume:
        poems = [p for p in poems if p["poem_id"] not in processed_ids]
        logger.info(f"{len(poems)} poems remaining")

    if not poems:
        logger.info("No poems to process")
        return

    lock = threading.Lock()
    success_count = 0
    error_count = 0

    logger.info(f"Processing with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_poem, p, client, prompt_config, not args.no_cache): p for p in poems}

        for future in as_completed(futures, timeout=max(300, len(poems) * 2)):
            try:
                poem_id, data, error = future.result(timeout=WORKER_TIMEOUT)
                with lock:
                    if data:
                        results[poem_id] = data
                        save_checkpoint(checkpoint_file, poem_id)
                        success_count += 1
                        if success_count % 50 == 0:
                            logger.info(f"Progress: {success_count}/{len(poems)}")
                            json.dump(results, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    else:
                        error_count += 1
                        logger.warning(f"Failed {poem_id}: {error}")
            except FuturesTimeoutError:
                error_count += 1

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Summary stats
    season_counts = {}
    theme_counts = {}
    for data in results.values():
        season_counts[data["season"]] = season_counts.get(data["season"], 0) + 1
        for t in data["themes"]:
            theme_counts[t] = theme_counts.get(t, 0) + 1

    logger.info(f"Done: {success_count} success, {error_count} errors")
    logger.info(f"Seasons: {season_counts}")
    logger.info(f"Themes: {dict(sorted(theme_counts.items(), key=lambda x: -x[1]))}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
