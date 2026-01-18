#!/usr/bin/env python3
"""
03c_select_poems.py - Select teaching poems for each lesson with frequency tracking.

This script separates poem selection from lesson generation to enable better
tracking of poem usage across the curriculum. It processes lessons in
topological order and tracks how many times each poem has been used.

Key features:
- Sequential processing with frequency tracking (no parallel mode)
- LLM selects 2-3 poems from 10-poem candidate pool
- Previously used poems are NOT forbidden, but less-used poems are preferred
- Outputs curriculum with poem_ids populated

Usage:
  python scripts/03c_select_poems.py --input data/curriculum/lesson_graph.json
  python scripts/03c_select_poems.py --resume  # Resume from checkpoint
  python scripts/03c_select_poems.py --dry-run  # Preview without API calls
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
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

from wakawaka.utils.prompt_loader import load_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_API_SLEEP = 0.5
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "curriculum" / ".checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "data" / "curriculum"


# -----------------------------------------------------------------------------
# Gemini API Client
# -----------------------------------------------------------------------------

class GeminiClient:
    """Wrapper for Gemini API with rate limiting and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        sleep_seconds: float = DEFAULT_API_SLEEP
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.model = model
        self.temperature = temperature
        self.sleep_seconds = sleep_seconds
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """Generate response with retry logic."""
        import time

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[user_prompt],
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        response_mime_type="application/json"
                    )
                )

                time.sleep(self.sleep_seconds)

                if response.text:
                    return response.text

                raise ValueError("Empty response from API")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_lesson_graph(input_path: Path) -> dict:
    """Load curriculum lesson graph."""
    with open(input_path) as f:
        return json.load(f)


def load_poems_df(poems_path: Path) -> pd.DataFrame:
    """Load annotated poems parquet."""
    return pd.read_parquet(poems_path)


def load_grammar_index(grammar_index_path: Path) -> dict:
    """Load grammar index for category/description lookup."""
    with open(grammar_index_path) as f:
        return json.load(f)


def load_literary_df(literary_path: Path | None) -> pd.DataFrame | None:
    """Load literary annotations if available."""
    if literary_path and literary_path.exists():
        return pd.read_parquet(literary_path)
    return None


# -----------------------------------------------------------------------------
# Poem Data Formatting
# -----------------------------------------------------------------------------

def convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


def get_poems_by_ids(poem_ids: list[str], poems_df: pd.DataFrame) -> list[dict]:
    """Get poem data for a list of poem IDs."""
    poems = []
    for poem_id in poem_ids:
        rows = poems_df[poems_df["poem_id"] == poem_id]
        if not rows.empty:
            poem = convert_numpy(rows.iloc[0].to_dict())
            poems.append(poem)
    return poems


def format_candidates_for_selection(
    candidates: list[dict],
    canonical_id: str,
    literary_df: pd.DataFrame | None = None
) -> str:
    """Format candidate poems for LLM selection prompt."""
    formatted = []

    for poem in candidates:
        poem_id = poem.get("poem_id", "unknown")

        # Get literary devices if available
        literary_devices = []
        if literary_df is not None:
            lit_rows = literary_df[literary_df["poem_id"] == poem_id]
            if not lit_rows.empty:
                devices = lit_rows.iloc[0].get("poetic_devices", [])
                devices = convert_numpy(devices)
                if isinstance(devices, list):
                    literary_devices = devices

        # Get grammar points and vocabulary
        grammar_points = convert_numpy(poem.get("grammar_points", []))
        vocabulary = convert_numpy(poem.get("vocabulary", []))

        entry = {
            "poem_id": poem_id,
            "text": poem.get("text", ""),
            "reading_romaji": poem.get("reading_romaji", ""),
            "difficulty_score": round(float(poem.get("difficulty_score_computed", 0.5)), 3),
            "grammar_points": grammar_points,
            "vocabulary": vocabulary[:5] if isinstance(vocabulary, list) else [],
        }

        if literary_devices:
            entry["literary_devices"] = literary_devices[:5]

        formatted.append(entry)

    # Ensure all numpy types are converted
    formatted = convert_numpy(formatted)
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def format_poem_frequency(
    candidate_ids: list[str],
    poem_usage: Counter
) -> str:
    """Format poem usage frequency for LLM prompt."""
    lines = []
    for pid in candidate_ids:
        count = poem_usage.get(pid, 0)
        if count == 0:
            lines.append(f"- {pid}: 未使用")
        else:
            lines.append(f"- {pid}: 已用于 {count} 课")

    if not lines:
        return "（首次选择，所有诗歌均未使用）"

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LLM Poem Selection
# -----------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find JSON object directly
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def select_poems_for_lesson(
    lesson: dict,
    candidates: list[dict],
    grammar_index: dict,
    poem_usage: Counter,
    client: GeminiClient,
    prompt_config: dict,
    literary_df: pd.DataFrame | None = None
) -> list[str]:
    """
    Use LLM to select the best poems from candidates.

    Args:
        lesson: Lesson dict from curriculum
        candidates: List of candidate poem dicts
        grammar_index: Grammar index data
        poem_usage: Counter of poem usage across lessons
        client: Gemini client
        prompt_config: Loaded select_poems.yaml config
        literary_df: Literary annotations (optional)

    Returns:
        List of selected poem IDs (2-3 poems)
    """
    canonical_id = lesson["canonical_grammar_point"]
    grammar_info = grammar_index.get("entries", {}).get(canonical_id, {})

    # Get category and description
    category = grammar_info.get("category", "unknown")
    senses = grammar_info.get("senses", [])
    description = senses[0].get("description", "") if senses else ""

    # Format candidate IDs for frequency lookup
    candidate_ids = [p.get("poem_id") for p in candidates]

    # Build prompt
    system_prompt = prompt_config.get("system", "")
    user_template = prompt_config.get("user_template", "")

    user_prompt = user_template.format(
        grammar_point_id=canonical_id,
        category=category,
        description=description,
        poem_frequency=format_poem_frequency(candidate_ids, poem_usage),
        candidates_json=format_candidates_for_selection(candidates, canonical_id, literary_df)
    )

    # Call LLM
    try:
        response_text = client.generate(system_prompt, user_prompt)
        llm_response = extract_json_from_response(response_text)

        # Extract selected poem IDs
        selected_poems = llm_response.get("selected_poems", [])

        # Validate and extract IDs
        candidate_id_set = set(candidate_ids)
        selected_ids = []

        for sp in selected_poems:
            pid = sp.get("poem_id")
            if pid and pid in candidate_id_set:
                selected_ids.append(pid)

        # Ensure we have at least 2 poems
        if len(selected_ids) < 2:
            logger.warning(f"LLM selected fewer than 2 valid poems for {lesson['id']}, using first 3 candidates")
            selected_ids = candidate_ids[:3]

        return selected_ids[:3]  # Max 3 poems

    except Exception as e:
        logger.error(f"Poem selection failed for {lesson['id']}: {e}, using first 3 candidates")
        return candidate_ids[:3]


# -----------------------------------------------------------------------------
# Processing Pipeline
# -----------------------------------------------------------------------------

def get_topological_order(lesson_graph: dict) -> list[str]:
    """Get lessons in topological order (prerequisites first)."""
    import networkx as nx

    G = nx.DiGraph()

    # Add all lessons as nodes
    all_lessons = []
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            all_lessons.append(lesson)
            G.add_node(lesson["id"])

    # Add prerequisite edges
    for lesson in all_lessons:
        for prereq in lesson.get("prerequisites", []):
            if prereq in G.nodes:
                G.add_edge(prereq, lesson["id"])

    # Return topological sort
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        logger.warning("Cycle detected in prerequisites, using unit order")
        return [l["id"] for l in all_lessons]


def get_lesson_by_id(lesson_graph: dict, lesson_id: str) -> dict | None:
    """Get a specific lesson by ID."""
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            if lesson["id"] == lesson_id:
                return lesson
    return None


def save_checkpoint(
    checkpoint_path: Path,
    poem_usage: Counter,
    processed_lessons: set,
    lesson_poems: dict
):
    """Save checkpoint for resume."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "poem_usage": dict(poem_usage),
        "processed_lessons": list(processed_lessons),
        "lesson_poems": lesson_poems,
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> tuple[Counter, set, dict] | None:
    """Load checkpoint if exists."""
    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path) as f:
        data = json.load(f)

    return (
        Counter(data.get("poem_usage", {})),
        set(data.get("processed_lessons", [])),
        data.get("lesson_poems", {})
    )


def process_lessons(
    lesson_graph: dict,
    poems_df: pd.DataFrame,
    grammar_index: dict,
    client: GeminiClient | None,
    prompt_config: dict,
    literary_df: pd.DataFrame | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    dry_run: bool = False
) -> dict:
    """
    Process all lessons and select poems.

    Args:
        lesson_graph: Curriculum lesson graph
        poems_df: Annotated poems DataFrame
        grammar_index: Grammar index dict
        client: Gemini client (None for dry run)
        prompt_config: Select poems prompt config
        literary_df: Literary annotations (optional)
        checkpoint_path: Path to save checkpoints
        resume: Whether to resume from checkpoint
        dry_run: If True, skip LLM calls and use first 3 candidates

    Returns:
        Updated lesson graph with poem_ids populated
    """
    # Initialize tracking
    poem_usage = Counter()
    processed_lessons = set()
    lesson_poems = {}  # lesson_id -> list of poem_ids

    # Try to resume from checkpoint
    if resume and checkpoint_path:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data:
            poem_usage, processed_lessons, lesson_poems = checkpoint_data
            logger.info(f"Resumed from checkpoint: {len(processed_lessons)} lessons already processed")

    # Get topological order
    topo_order = get_topological_order(lesson_graph)
    logger.info(f"Processing {len(topo_order)} lessons in topological order")

    # Process each lesson
    for i, lesson_id in enumerate(topo_order):
        if lesson_id in processed_lessons:
            logger.debug(f"Skipping {lesson_id} (already processed)")
            continue

        lesson = get_lesson_by_id(lesson_graph, lesson_id)
        if not lesson:
            logger.warning(f"Lesson {lesson_id} not found in graph")
            continue

        # Get candidate poems
        candidate_ids = lesson.get("candidate_poem_ids", [])
        if not candidate_ids:
            logger.warning(f"No candidates for {lesson_id}, skipping")
            continue

        candidates = get_poems_by_ids(candidate_ids, poems_df)
        if not candidates:
            logger.warning(f"No valid candidate poems for {lesson_id}")
            continue

        # Select poems
        if dry_run:
            selected_ids = candidate_ids[:3]
            logger.info(f"[{i+1}/{len(topo_order)}] {lesson_id}: (dry-run) using first 3 candidates")
        else:
            selected_ids = select_poems_for_lesson(
                lesson, candidates, grammar_index, poem_usage,
                client, prompt_config, literary_df
            )
            logger.info(f"[{i+1}/{len(topo_order)}] {lesson_id}: selected {selected_ids}")

        # Update tracking
        lesson_poems[lesson_id] = selected_ids
        for pid in selected_ids:
            poem_usage[pid] += 1
        processed_lessons.add(lesson_id)

        # Save checkpoint periodically
        if checkpoint_path and (i + 1) % 10 == 0:
            save_checkpoint(checkpoint_path, poem_usage, processed_lessons, lesson_poems)
            logger.debug(f"Checkpoint saved at {i+1} lessons")

    # Final checkpoint
    if checkpoint_path:
        save_checkpoint(checkpoint_path, poem_usage, processed_lessons, lesson_poems)

    # Update lesson graph with selected poems
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            if lesson["id"] in lesson_poems:
                lesson["poem_ids"] = lesson_poems[lesson["id"]]

    # Add metadata
    lesson_graph["poem_selection"] = {
        "generated_at": datetime.now().isoformat(),
        "total_poems_used": len(poem_usage),
        "unique_poems": len([p for p, c in poem_usage.items() if c > 0]),
        "max_reuse_count": max(poem_usage.values()) if poem_usage else 0,
        "reuse_distribution": dict(Counter(poem_usage.values()))
    }

    return lesson_graph


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select teaching poems for curriculum lessons with frequency tracking"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=PROJECT_ROOT / "data" / "curriculum" / "lesson_graph.json",
        help="Input lesson graph JSON"
    )
    parser.add_argument(
        "--poems", "-p",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Annotated poems parquet"
    )
    parser.add_argument(
        "--grammar-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "curriculum" / "grammar_index.json",
        help="Grammar index JSON"
    )
    parser.add_argument(
        "--literary",
        type=Path,
        default=PROJECT_ROOT / "data" / "literary" / "poems_literary.parquet",
        help="Literary annotations parquet (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path (default: input path with _with_poems suffix)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without API calls (uses first 3 candidates)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.poems.exists():
        logger.error(f"Poems file not found: {args.poems}")
        sys.exit(1)

    if not args.grammar_index.exists():
        logger.error(f"Grammar index not found: {args.grammar_index}")
        sys.exit(1)

    # Set output path
    output_path = args.output
    if output_path is None:
        output_path = args.input.parent / "lesson_graph_with_poems.json"

    # Load data
    logger.info("Loading data...")
    lesson_graph = load_lesson_graph(args.input)
    poems_df = load_poems_df(args.poems)
    grammar_index = load_grammar_index(args.grammar_index)

    literary_df = None
    if args.literary.exists():
        literary_df = load_literary_df(args.literary)
        logger.info(f"Loaded literary data: {len(literary_df)} poems")

    logger.info(f"Loaded lesson graph: {sum(len(u.get('lessons', [])) for u in lesson_graph.get('units', []))} lessons")
    logger.info(f"Loaded poems: {len(poems_df)} poems")

    # Load prompt config
    prompt_config = load_prompt("select_poems")

    # Initialize client (unless dry run)
    client = None
    if not args.dry_run:
        client = GeminiClient(model=args.model, temperature=0.2)
        logger.info(f"Using model: {args.model}")

    # Process lessons
    checkpoint_path = CHECKPOINT_DIR / "poem_selection_checkpoint.json"

    updated_graph = process_lessons(
        lesson_graph=lesson_graph,
        poems_df=poems_df,
        grammar_index=grammar_index,
        client=client,
        prompt_config=prompt_config,
        literary_df=literary_df,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        dry_run=args.dry_run
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(updated_graph, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved updated lesson graph to: {output_path}")

    # Print summary
    selection_meta = updated_graph.get("poem_selection", {})
    logger.info("=" * 50)
    logger.info("Poem Selection Summary:")
    logger.info(f"  Total poems used: {selection_meta.get('total_poems_used', 0)}")
    logger.info(f"  Unique poems: {selection_meta.get('unique_poems', 0)}")
    logger.info(f"  Max reuse count: {selection_meta.get('max_reuse_count', 0)}")
    logger.info(f"  Reuse distribution: {selection_meta.get('reuse_distribution', {})}")


if __name__ == "__main__":
    main()
