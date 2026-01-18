#!/usr/bin/env python3
"""
03b_refine_curriculum_llm.py - LLM-assisted curriculum refinement.

This script takes the algorithmically-generated curriculum and asks an LLM
to review and improve it pedagogically. The original curriculum is preserved;
refined output goes to a separate directory.

Key features:
- Preserves original algorithmic curriculum
- LLM reviews: unit themes, lesson ordering, prerequisites, groupings
- Outputs refined curriculum to data/curriculum_refined/
- Uses high-quality model (gemini-3-pro-preview) for pedagogical reasoning
- Ensemble mode: run multiple trials and synthesize best curriculum

Usage:
  python scripts/03b_refine_curriculum_llm.py
  python scripts/03b_refine_curriculum_llm.py --ensemble 5
  python scripts/03b_refine_curriculum_llm.py --model gemini-2.5-pro
  python scripts/03b_refine_curriculum_llm.py --output-dir data/curriculum_v2
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from wakawaka.utils.prompt_loader import load_prompt

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

# Constants
DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "curriculum"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "curriculum_refined"
DEFAULT_LITERARY_INDEX = PROJECT_ROOT / "data" / "literary" / "literary_index.json"
DEFAULT_SEASON_THEME = PROJECT_ROOT / "data" / "literary" / "standardized_season_theme.json"
DEFAULT_LITERARY_PARQUET = PROJECT_ROOT / "data" / "literary" / "poems_literary.parquet"

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Gemini API Client
# -----------------------------------------------------------------------------

class GeminiClient:
    """Wrapper for Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set.")

        self.client = genai.Client(api_key=self.api_key)
        self.temperature = temperature
        self.model_name = model

    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
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
                    time.sleep(2 ** attempt)
                else:
                    raise


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_curriculum(curriculum_dir: Path) -> tuple[dict, dict]:
    """Load curriculum data."""
    lesson_graph_path = curriculum_dir / "lesson_graph.json"
    grammar_index_path = curriculum_dir / "grammar_index.json"

    with open(lesson_graph_path, "r", encoding="utf-8") as f:
        lesson_graph = json.load(f)

    with open(grammar_index_path, "r", encoding="utf-8") as f:
        grammar_index = json.load(f)

    return lesson_graph, grammar_index


def load_literary_index(literary_index_path: Path | None = None) -> dict | None:
    """Load literary index data for curriculum awareness."""
    path = literary_index_path or DEFAULT_LITERARY_INDEX
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            literary_index = json.load(f)
        logger.info(f"Loaded literary index with {len(literary_index)} devices")
        return literary_index
    else:
        logger.warning(f"Literary index not found at {path}")
        return None


def format_literary_summary(literary_index: dict | None) -> str:
    """Format literary device statistics for LLM context."""
    if not literary_index:
        return "（无文学分析数据）"

    lines = ["主要诗歌技法统计:"]

    # Sort by frequency
    sorted_devices = sorted(
        literary_index.items(),
        key=lambda x: x[1].get("frequency", 0),
        reverse=True
    )

    for device, stats in sorted_devices[:8]:
        freq = stats.get("frequency", 0)
        lines.append(f"  - {device}: {freq}首诗")

    return "\n".join(lines)


def load_season_theme(season_theme_path: Path | None = None) -> dict | None:
    """Load standardized season/theme data for curriculum awareness."""
    path = season_theme_path or DEFAULT_SEASON_THEME
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            season_theme = json.load(f)
        logger.info(f"Loaded season/theme for {len(season_theme)} poems")
        return season_theme
    else:
        logger.warning(f"Season/theme data not found at {path}")
        return None


def format_season_theme_summary(season_theme: dict | None) -> str:
    """Format season/theme distribution for LLM context."""
    if not season_theme:
        return "（无季节/主题数据）"

    from collections import Counter

    season_counter = Counter()
    theme_counter = Counter()

    for poem_id, data in season_theme.items():
        season = data.get("season", "")
        if season and season != "無季":
            season_counter[season] += 1
        for theme in data.get("themes", []):
            if theme and theme != "其他":
                theme_counter[theme] += 1

    lines = ["季节分布:"]
    for season, count in season_counter.most_common(5):
        lines.append(f"  - {season}: {count}首诗")

    lines.append("\n主题分布:")
    for theme, count in theme_counter.most_common(8):
        lines.append(f"  - {theme}: {count}首诗")

    return "\n".join(lines)


def load_poem_devices(literary_parquet_path: Path | None = None) -> dict[str, list[str]]:
    """
    Load poetic devices for each poem from literary parquet.

    Returns:
        Dict mapping poem_id -> list of device names
    """
    path = literary_parquet_path or DEFAULT_LITERARY_PARQUET
    if not path.exists():
        logger.warning(f"Literary parquet not found at {path}")
        return {}

    try:
        df = pd.read_parquet(path)
        poem_devices = {}

        for _, row in df.iterrows():
            poem_id = row.get("poem_id", "")
            devices_raw = row.get("poetic_devices")

            if devices_raw is None or (isinstance(devices_raw, float) and np.isnan(devices_raw)):
                poem_devices[poem_id] = []
                continue

            # Handle numpy array
            if isinstance(devices_raw, np.ndarray):
                devices_raw = devices_raw.tolist()

            # Extract device names
            device_names = []
            if isinstance(devices_raw, list):
                for device in devices_raw:
                    if isinstance(device, dict) and "name" in device:
                        device_names.append(device["name"])

            poem_devices[poem_id] = device_names

        logger.info(f"Loaded devices for {len(poem_devices)} poems")
        return poem_devices

    except Exception as e:
        logger.warning(f"Failed to load poem devices: {e}")
        return {}


def get_lesson_literary_distribution(
    candidate_poem_ids: list[str],
    season_theme: dict | None,
    poem_devices: dict | None
) -> dict:
    """
    Compute season/theme/device distribution for a lesson's candidate poems.

    Returns:
        Dict with seasons, themes, devices distributions
    """
    from collections import Counter

    seasons = Counter()
    themes = Counter()
    devices = Counter()

    for poem_id in candidate_poem_ids:
        # Season/theme from standardized data
        if season_theme and poem_id in season_theme:
            st = season_theme[poem_id]
            season = st.get("season", "")
            if season and season != "無季":
                seasons[season] += 1
            for theme in st.get("themes", []):
                if theme and theme != "其他":
                    themes[theme] += 1

        # Devices from literary parquet
        if poem_devices and poem_id in poem_devices:
            for device in poem_devices[poem_id]:
                devices[device] += 1

    return {
        "seasons": dict(seasons.most_common(4)),
        "themes": dict(themes.most_common(4)),
        "devices": dict(devices.most_common(5))
    }


def get_all_lesson_ids(lesson_graph: dict) -> list[str]:
    """Extract all lesson IDs from lesson graph."""
    ids = []
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            ids.append(lesson["id"])
    return ids


def format_lesson_metadata(lesson_graph: dict, grammar_index: dict) -> str:
    """Format concise lesson metadata for LLM."""
    lines = []

    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            lesson_id = lesson.get("id", "")
            grammar_point = lesson.get("canonical_grammar_point", "")

            # Get grammar info
            gp_info = grammar_index.get("entries", {}).get(grammar_point, {})
            frequency = gp_info.get("frequency", 0)
            category = gp_info.get("category", "unknown")
            surfaces = gp_info.get("surfaces", [])[:3]

            # One line per lesson
            surfaces_str = "/".join(surfaces) if surfaces else ""
            lines.append(f"{lesson_id}|{category}|{surfaces_str}|freq={frequency}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LLM Curriculum Refinement
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a curriculum designer for classical Japanese.

STUDENT: Chinese speaker, knows hiragana, no Japanese grammar.

TASK: Reorganize lessons into pedagogically-ordered units.

STRICT RULES:
1. Use ONLY lesson IDs from the PROVIDED LIST below
2. Every lesson ID must appear in EXACTLY ONE unit
3. Prerequisites must be lesson IDs that appear EARLIER in the curriculum
4. Unit IDs must be: unit_01, unit_02, ... unit_NN
5. Output ONLY valid JSON, no explanatory text

OUTPUT SCHEMA (follow exactly):
{
  "units": [
    {
      "id": "unit_01",
      "title": "Short Title (max 5 words)",
      "lessons": ["lesson_id_1", "lesson_id_2"]
    }
  ],
  "prerequisites": {
    "lesson_id": ["prereq_lesson_id"]
  }
}

CONSTRAINTS:
- titles: max 40 characters, no quotes or special chars
- lessons array: 3-10 lessons per unit
- prerequisites: only reference lessons in earlier units
- no extra fields, no comments, no analysis text"""


USER_TEMPLATE = """VALID LESSON IDS (use exactly these, no others):
{lesson_ids_list}

LESSON METADATA:
{lesson_metadata}

Reorganize into 5-10 thematic units. Output JSON only."""


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response."""
    # Try code blocks
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

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not extract JSON: {e}\n\nResponse:\n{text[:1000]}...")


def validate_refinements(refinements: dict, valid_ids: set[str]) -> tuple[bool, list[str]]:
    """Validate LLM output against constraints."""
    errors = []

    units = refinements.get("units", [])
    if not units:
        errors.append("No units in output")
        return False, errors

    if len(units) < 3 or len(units) > 15:
        errors.append(f"Expected 3-15 units, got {len(units)}")

    used_ids = set()
    for i, unit in enumerate(units):
        unit_id = unit.get("id", "")
        if not unit_id.startswith("unit_"):
            errors.append(f"Invalid unit ID: {unit_id}")

        title = unit.get("title", "")
        if len(title) > 50:
            errors.append(f"Unit {unit_id} title too long: {len(title)} chars")

        lessons = unit.get("lessons", [])
        if not lessons:
            errors.append(f"Unit {unit_id} has no lessons")

        for lid in lessons:
            if lid not in valid_ids:
                errors.append(f"Invalid lesson ID: {lid}")
            if lid in used_ids:
                errors.append(f"Duplicate lesson ID: {lid}")
            used_ids.add(lid)

    # Check all lessons are assigned
    missing = valid_ids - used_ids
    if missing:
        errors.append(f"Missing lessons: {missing}")

    # Validate prerequisites
    prereqs = refinements.get("prerequisites", {})
    assigned_before = set()
    for unit in units:
        for lid in unit.get("lessons", []):
            if lid in prereqs:
                for prereq_id in prereqs[lid]:
                    if prereq_id not in valid_ids:
                        errors.append(f"Invalid prereq ID: {prereq_id}")
                    if prereq_id not in assigned_before:
                        errors.append(f"Prereq {prereq_id} for {lid} not in earlier unit")
            assigned_before.add(lid)

    return len(errors) == 0, errors


def refine_curriculum(
    lesson_graph: dict,
    grammar_index: dict,
    client: GeminiClient
) -> dict:
    """Ask LLM to refine the curriculum with validation."""

    valid_ids = set(get_all_lesson_ids(lesson_graph))
    lesson_metadata = format_lesson_metadata(lesson_graph, grammar_index)

    user_prompt = USER_TEMPLATE.format(
        lesson_ids_list="\n".join(sorted(valid_ids)),
        lesson_metadata=lesson_metadata
    )

    logger.info("Sending curriculum to LLM for review...")
    logger.info(f"  Model: {client.model_name}")
    logger.info(f"  Lessons: {len(valid_ids)}")
    logger.info(f"  Prompt size: ~{len(user_prompt)} chars")

    response_text = client.generate(SYSTEM_PROMPT, user_prompt)

    logger.info("Parsing LLM response...")
    refined = extract_json_from_response(response_text)

    # Validate
    is_valid, errors = validate_refinements(refined, valid_ids)
    if not is_valid:
        logger.warning("Validation errors in LLM output:")
        for err in errors[:10]:
            logger.warning(f"  - {err}")
        # Try to fix missing lessons
        if any("Missing lessons" in e for e in errors):
            refined = fix_missing_lessons(refined, valid_ids)

    return refined


def fix_missing_lessons(refinements: dict, valid_ids: set[str]) -> dict:
    """Add missing lessons to a misc unit."""
    used_ids = set()
    for unit in refinements.get("units", []):
        used_ids.update(unit.get("lessons", []))

    missing = valid_ids - used_ids
    if missing:
        logger.info(f"  Adding {len(missing)} missing lessons to misc unit")
        refinements["units"].append({
            "id": f"unit_{len(refinements['units']) + 1:02d}",
            "title": "Additional Grammar",
            "lessons": sorted(missing)
        })

    return refinements


# -----------------------------------------------------------------------------
# Ensemble Mode Functions
# -----------------------------------------------------------------------------

def format_lessons_for_trial(
    lesson_graph: dict,
    grammar_index: dict,
    season_theme: dict | None = None,
    poem_devices: dict | None = None
) -> str:
    """
    Format lessons as JSON for trial prompt.

    Now includes per-lesson literary distributions so LLM can design
    thematic units and assign literary_focus with full awareness.
    """
    lessons = []
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            lid = lesson.get("id", "")
            gp = lesson.get("canonical_grammar_point", "")
            gp_info = grammar_index.get("entries", {}).get(gp, {})

            lesson_data = {
                "id": lid,
                "grammar_point": gp,
                "category": gp_info.get("category", "unknown"),
                "frequency": gp_info.get("frequency", 0),
                "difficulty_tier": lesson.get("difficulty_tier", 3),
                "surfaces": gp_info.get("surfaces", [])[:3]
            }

            # Include algorithmic literary data
            if lesson.get("literary_ready"):
                lesson_data["literary_ready"] = True
                if lesson.get("literary_difficulty"):
                    lesson_data["literary_difficulty"] = lesson["literary_difficulty"]

            # Include per-lesson literary distributions from candidate poems
            # This gives LLM full picture to design thematic progression
            candidate_ids = lesson.get("candidate_poem_ids", [])
            if candidate_ids and (season_theme or poem_devices):
                distribution = get_lesson_literary_distribution(
                    candidate_ids, season_theme, poem_devices
                )
                # Only include non-empty distributions
                if distribution["seasons"]:
                    lesson_data["available_seasons"] = distribution["seasons"]
                if distribution["themes"]:
                    lesson_data["available_themes"] = distribution["themes"]
                if distribution["devices"]:
                    lesson_data["available_devices"] = distribution["devices"]

            lessons.append(lesson_data)

    return json.dumps(lessons, ensure_ascii=False, indent=2)


def format_grammar_summary(grammar_index: dict) -> str:
    """Format grammar relationships for LLM context."""
    lines = []
    entries = grammar_index.get("entries", {})

    # Group by category
    by_category = {}
    for gid, info in entries.items():
        cat = info.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f"{gid} (freq={info.get('frequency', 0)})")

    for cat, items in sorted(by_category.items()):
        lines.append(f"[{cat}]: {', '.join(items[:10])}")
        if len(items) > 10:
            lines.append(f"  ... and {len(items) - 10} more")

    return "\n".join(lines)


def run_ensemble_trial(
    trial_num: int,
    lesson_graph: dict,
    grammar_index: dict,
    client: GeminiClient,
    trial_prompt_config: dict,
    valid_ids: set[str],
    literary_index: dict | None = None,
    season_theme: dict | None = None,
    poem_devices: dict | None = None
) -> dict | None:
    """Run a single ensemble trial."""
    logger.info(f"  Running trial {trial_num}...")

    # Include per-lesson distributions for LLM awareness
    lessons_json = format_lessons_for_trial(
        lesson_graph, grammar_index,
        season_theme=season_theme,
        poem_devices=poem_devices
    )
    grammar_summary = format_grammar_summary(grammar_index)
    literary_summary = format_literary_summary(literary_index)
    season_theme_summary = format_season_theme_summary(season_theme)

    system_prompt = trial_prompt_config.get("system", "")
    user_template = trial_prompt_config.get("user_template", "")

    user_prompt = user_template.format(
        num_lessons=len(valid_ids),
        trial_number=trial_num,
        lessons_json=lessons_json,
        grammar_summary=grammar_summary,
        literary_summary=literary_summary,
        season_theme_summary=season_theme_summary
    )

    try:
        response_text = client.generate(system_prompt, user_prompt)
        trial_result = extract_json_from_response(response_text)

        # Validate
        is_valid, errors = validate_refinements(trial_result, valid_ids)
        if not is_valid:
            logger.warning(f"  Trial {trial_num} validation errors: {errors[:3]}")
            # Try to fix
            if any("Missing lessons" in e for e in errors):
                trial_result = fix_missing_lessons(trial_result, valid_ids)

        return trial_result

    except Exception as e:
        logger.error(f"  Trial {trial_num} failed: {e}")
        return None


def synthesize_trials(
    trials: list[dict],
    valid_ids: set[str],
    client: GeminiClient,
    synthesis_prompt_config: dict,
    literary_index: dict | None = None,
    season_theme: dict | None = None,
    lessons_with_distributions: str | None = None
) -> dict:
    """Synthesize final curriculum from multiple trials."""
    logger.info("Synthesizing final curriculum from trials...")

    # Format trials for synthesis prompt
    master_lessons = [{"id": lid} for lid in sorted(valid_ids)]
    literary_summary = format_literary_summary(literary_index)
    season_theme_summary = format_season_theme_summary(season_theme)

    # Use per-lesson distributions if provided, otherwise just IDs
    if lessons_with_distributions:
        lessons_json_for_synthesis = lessons_with_distributions
    else:
        lessons_json_for_synthesis = json.dumps(master_lessons, indent=2)

    # Clean up trials for JSON
    trials_data = []
    for i, trial in enumerate(trials):
        if trial:
            trials_data.append({
                "curriculum_id": trial.get("curriculum_id", f"trial_{i+1}"),
                "design_philosophy": trial.get("design_philosophy", "Not specified"),
                "units": trial.get("units", []),
                "prerequisites": trial.get("prerequisites", {})
            })

    system_prompt = synthesis_prompt_config.get("system", "")
    user_template = synthesis_prompt_config.get("user_template", "")

    user_prompt = user_template.format(
        num_trials=len(trials_data),
        lessons_with_distributions=lessons_json_for_synthesis,
        trials_json=json.dumps(trials_data, ensure_ascii=False, indent=2),
        literary_summary=literary_summary,
        season_theme_summary=season_theme_summary
    )

    response_text = client.generate(system_prompt, user_prompt)
    synthesis = extract_json_from_response(response_text)

    # Validate
    is_valid, errors = validate_refinements(synthesis, valid_ids)
    if not is_valid:
        logger.warning(f"Synthesis validation errors: {errors[:5]}")
        if any("Missing lessons" in e for e in errors):
            synthesis = fix_missing_lessons(synthesis, valid_ids)

    return synthesis


def generate_lesson_context(
    refined_graph: dict,
    synthesis: dict,
    client: GeminiClient
) -> str:
    """
    Generate lesson context YAML based on finalized curriculum.

    Args:
        refined_graph: The final refined lesson graph
        synthesis: The synthesis result (with notes if available)
        client: Gemini client

    Returns:
        YAML string with lesson generation context
    """
    logger.info("Generating lesson context for lesson generation...")

    context_prompt = load_prompt("lesson_context")

    # Format curriculum for prompt
    curriculum_summary = {
        "units": [
            {
                "id": unit.get("id"),
                "title": unit.get("title"),
                "lessons": [
                    {
                        "id": lesson.get("id"),
                        "grammar_point": lesson.get("canonical_grammar_point"),
                        "prerequisites": lesson.get("prerequisites", [])
                    }
                    for lesson in unit.get("lessons", [])
                ]
            }
            for unit in refined_graph.get("units", [])
        ]
    }

    # Get synthesis notes if available
    synthesis_notes = synthesis.get("synthesis_notes", {})
    synthesis_notes_str = json.dumps(synthesis_notes, indent=2) if synthesis_notes else "Not available"

    system_prompt = context_prompt.get("system", "")
    user_template = context_prompt.get("user_template", "")

    user_prompt = user_template.format(
        curriculum_json=json.dumps(curriculum_summary, indent=2, ensure_ascii=False),
        synthesis_notes=synthesis_notes_str
    )

    response_text = client.generate(system_prompt, user_prompt)

    # Clean up response (remove markdown code blocks if present)
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first and last lines (code block markers)
        lines = [l for l in lines if not l.startswith("```")]
        response_text = "\n".join(lines)

    return response_text


def run_ensemble(
    lesson_graph: dict,
    grammar_index: dict,
    client: GeminiClient,
    num_trials: int = 5,
    max_parallel: int = 3,
    literary_index: dict | None = None,
    season_theme: dict | None = None,
    poem_devices: dict | None = None
) -> tuple[dict, list[dict]]:
    """
    Run ensemble curriculum generation with parallel trials.

    Args:
        lesson_graph: Original curriculum
        grammar_index: Grammar index
        client: Gemini client
        num_trials: Number of independent trials
        max_parallel: Maximum parallel API calls
        literary_index: Literary device index for context (optional)
        season_theme: Season/theme data for context (optional)
        poem_devices: Per-poem device lists for context (optional)

    Returns:
        Tuple of (synthesized_curriculum, all_trials)
    """
    # Load prompts
    trial_prompt = load_prompt("curriculum_trial")
    synthesis_prompt = load_prompt("curriculum_synthesis")

    valid_ids = set(get_all_lesson_ids(lesson_graph))
    trial_temp = trial_prompt.get("meta", {}).get("temperature", 0.7)

    logger.info(f"Running ensemble with {num_trials} trials (max {max_parallel} parallel)...")

    # Run trials in parallel
    trials = [None] * num_trials  # Pre-allocate to maintain order

    def run_single_trial(trial_num: int) -> tuple[int, dict | None]:
        """Worker function for parallel execution."""
        # Create a new client for each thread to avoid race conditions
        thread_client = GeminiClient(
            model=client.model_name,
            temperature=trial_temp
        )
        result = run_ensemble_trial(
            trial_num=trial_num,
            lesson_graph=lesson_graph,
            grammar_index=grammar_index,
            client=thread_client,
            trial_prompt_config=trial_prompt,
            valid_ids=valid_ids,
            literary_index=literary_index,
            season_theme=season_theme,
            poem_devices=poem_devices
        )
        return trial_num, result

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_single_trial, i): i
            for i in range(1, num_trials + 1)
        }

        for future in as_completed(futures):
            trial_num, result = future.result()
            if result:
                trials[trial_num - 1] = result
                logger.info(f"  Trial {trial_num}: {len(result.get('units', []))} units")
            else:
                logger.warning(f"  Trial {trial_num}: FAILED")

    # Filter out None values (failed trials)
    valid_trials = [t for t in trials if t is not None]

    if len(valid_trials) < 2:
        raise ValueError(f"Only {len(valid_trials)} trials succeeded, need at least 2 for synthesis")

    # Prepare lessons with distributions for synthesis (same as trials see)
    lessons_with_distributions = format_lessons_for_trial(
        lesson_graph, grammar_index,
        season_theme=season_theme,
        poem_devices=poem_devices
    )

    # Synthesize with lower temperature
    client.temperature = synthesis_prompt.get("meta", {}).get("temperature", 0.3)
    synthesis = synthesize_trials(
        valid_trials, valid_ids, client, synthesis_prompt,
        literary_index=literary_index,
        season_theme=season_theme,
        lessons_with_distributions=lessons_with_distributions
    )

    return synthesis, valid_trials


def apply_refinements(
    original_graph: dict,
    grammar_index: dict,
    refinements: dict
) -> dict:
    """Apply LLM refinements to create new lesson graph."""

    # Start with copy of original
    new_graph = {
        "units": [],
        "prerequisite_graph": original_graph.get("prerequisite_graph", {}),
        "meta": original_graph.get("meta", {}).copy()
    }

    # Build lesson lookup from original
    lesson_lookup = {}
    for unit in original_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            lesson_lookup[lesson["id"]] = lesson.copy()

    # Apply units from LLM (new schema uses "units" not "refined_units")
    llm_units = refinements.get("units", [])
    if llm_units:
        for unit_def in llm_units:
            # Get literary_ready from LLM's unit definition
            unit_literary_ready = unit_def.get("literary_ready", False)

            unit = {
                "id": unit_def.get("id", f"unit_{len(new_graph['units']) + 1:02d}"),
                "title": unit_def.get("title"),
                "theme": unit_def.get("theme", ""),
                "literary_ready": unit_literary_ready,
                "lessons": []
            }

            for lesson_id in unit_def.get("lessons", []):
                if lesson_id in lesson_lookup:
                    lesson = lesson_lookup[lesson_id].copy()

                    # Apply unit's literary_ready status to lessons
                    # This ensures lessons in literary-ready units get marked appropriately
                    if unit_literary_ready and not lesson.get("literary_ready"):
                        lesson["literary_ready"] = True

                    unit["lessons"].append(lesson)

            if unit["lessons"]:
                new_graph["units"].append(unit)
    else:
        new_graph["units"] = original_graph.get("units", [])

    # Apply prerequisites (new schema uses "prerequisites" not "refined_prerequisites")
    prereqs = refinements.get("prerequisites", {})
    if prereqs:
        for unit in new_graph["units"]:
            for lesson in unit["lessons"]:
                lesson_id = lesson["id"]
                if lesson_id in prereqs:
                    lesson["prerequisites"] = prereqs[lesson_id]

    # Apply LLM's literary_focus (key new feature)
    llm_literary_focus = refinements.get("literary_focus", {})
    if llm_literary_focus:
        logger.info(f"Applying LLM literary_focus to {len(llm_literary_focus)} lessons")
        for unit in new_graph["units"]:
            for lesson in unit["lessons"]:
                lesson_id = lesson["id"]
                if lesson_id in llm_literary_focus:
                    # Override algorithmic literary_focus with LLM's design
                    lesson["literary_focus"] = llm_literary_focus[lesson_id]
                    # Ensure lesson is marked literary_ready if it has focus
                    if llm_literary_focus[lesson_id]:
                        lesson["literary_ready"] = True

    # Update meta with literary integration stats
    literary_ready_count = sum(
        1 for u in new_graph["units"] for l in u.get("lessons", [])
        if l.get("literary_ready", False)
    )
    total_lessons = sum(len(u["lessons"]) for u in new_graph["units"])

    new_graph["meta"]["refined_at"] = datetime.now().isoformat()
    new_graph["meta"]["refinement_model"] = "llm"
    new_graph["meta"]["total_lessons"] = total_lessons

    # Update literary integration meta
    if "literary_integration" not in new_graph["meta"]:
        new_graph["meta"]["literary_integration"] = {}
    new_graph["meta"]["literary_integration"]["literary_ready_lessons"] = literary_ready_count

    return new_graph


def save_refined_curriculum(
    refined_graph: dict,
    grammar_index: dict,
    output_dir: Path,
    raw_refinements: dict
):
    """Save refined curriculum to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save refined lesson graph
    with open(output_dir / "lesson_graph.json", "w", encoding="utf-8") as f:
        json.dump(refined_graph, f, ensure_ascii=False, indent=2)

    # Copy grammar index (unchanged)
    with open(output_dir / "grammar_index.json", "w", encoding="utf-8") as f:
        json.dump(grammar_index, f, ensure_ascii=False, indent=2)

    # Save raw LLM refinements for reference
    with open(output_dir / "llm_refinements.json", "w", encoding="utf-8") as f:
        json.dump(raw_refinements, f, ensure_ascii=False, indent=2)

    # Generate report
    generate_report(refined_graph, output_dir)

    logger.info(f"Saved refined curriculum to {output_dir}")


def generate_report(refined_graph: dict, output_dir: Path):
    """Generate markdown report for refined curriculum."""
    meta = refined_graph.get('meta', {})
    lit_meta = meta.get('literary_integration', {})

    lines = [
        "# Refined Curriculum Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Total units**: {len(refined_graph.get('units', []))}",
        f"- **Total lessons**: {meta.get('total_lessons', 0)}",
    ]

    # Add literary integration summary
    if lit_meta:
        lines.extend([
            "",
            "### Literary Integration",
            "",
            f"- **Literary-ready lessons**: {lit_meta.get('literary_ready_lessons', 0)}",
        ])

    lines.extend([
        "",
        "## Units and Lessons",
        "",
    ])

    for unit in refined_graph.get("units", []):
        unit_id = unit.get("id", "unknown")
        unit_title = unit.get("title", unit_id)

        lines.append(f"### {unit_id}: {unit_title}")
        lines.append("")

        for lesson in unit.get("lessons", []):
            lesson_id = lesson.get("id", "")
            grammar = lesson.get("canonical_grammar_point", "")
            prereqs = lesson.get("prerequisites", [])
            lit_ready = lesson.get("literary_ready", False)

            prereq_str = f" (prereqs: {', '.join(prereqs)})" if prereqs else ""
            lit_str = " 📖" if lit_ready else ""
            lines.append(f"- `{lesson_id}` — {grammar}{prereq_str}{lit_str}")

            # Show literary focus if present
            if lesson.get("literary_focus"):
                lines.append(f"  - Literary focus: {', '.join(lesson['literary_focus'])}")

        lines.append("")

    # Legend
    lines.extend([
        "---",
        "",
        "**Legend**: 📖 = Literary-ready lesson (includes literary appreciation content)",
        "",
    ])

    with open(output_dir / "curriculum_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted curriculum refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/03b_refine_curriculum_llm.py                    # Single-shot refinement
  python scripts/03b_refine_curriculum_llm.py --ensemble 5       # Ensemble with 5 trials
  python scripts/03b_refine_curriculum_llm.py --ensemble 10      # Ensemble with 10 trials
"""
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input curriculum directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for refined curriculum"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=0,
        metavar="N",
        help="Run N independent trials and synthesize (recommended: 5-10)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        metavar="P",
        help="Max parallel API calls for ensemble mode (default: 3)"
    )
    parser.add_argument(
        "--literary-index",
        type=Path,
        default=DEFAULT_LITERARY_INDEX,
        help=f"Path to literary index JSON (default: {DEFAULT_LITERARY_INDEX})"
    )

    args = parser.parse_args()

    # Load original curriculum
    logger.info(f"Loading curriculum from {args.input_dir}...")
    lesson_graph, grammar_index = load_curriculum(args.input_dir)

    # Load literary index
    literary_index = load_literary_index(args.literary_index)

    # Load season/theme data
    season_theme = load_season_theme()

    # Load per-poem devices for detailed distributions
    poem_devices = load_poem_devices()

    original_lessons = sum(len(u.get("lessons", [])) for u in lesson_graph.get("units", []))
    logger.info(f"  Original: {len(lesson_graph.get('units', []))} units, {original_lessons} lessons")

    # Initialize client
    client = GeminiClient(model=args.model)

    # Get LLM refinements (single-shot or ensemble)
    trials = None
    try:
        if args.ensemble > 0:
            # Ensemble mode
            logger.info(f"Using ENSEMBLE mode with {args.ensemble} trials")
            refinements, trials = run_ensemble(
                lesson_graph, grammar_index, client,
                num_trials=args.ensemble,
                max_parallel=args.parallel,
                literary_index=literary_index,
                season_theme=season_theme,
                poem_devices=poem_devices
            )
        else:
            # Single-shot mode
            refinements = refine_curriculum(lesson_graph, grammar_index, client)
    except Exception as e:
        logger.error(f"Failed to get LLM refinements: {e}")
        return 1

    logger.info("LLM refinement received:")
    logger.info(f"  Units: {len(refinements.get('units', []))}")
    logger.info(f"  Prerequisites defined: {len(refinements.get('prerequisites', {}))}")
    if refinements.get("synthesis_notes"):
        notes = refinements["synthesis_notes"]
        logger.info("  Synthesis notes:")
        for decision in notes.get("key_decisions", [])[:3]:
            logger.info(f"    - {decision}")

    # Apply refinements
    logger.info("Applying refinements...")
    refined_graph = apply_refinements(lesson_graph, grammar_index, refinements)

    refined_lessons = sum(len(u.get("lessons", [])) for u in refined_graph.get("units", []))
    logger.info(f"  Refined: {len(refined_graph.get('units', []))} units, {refined_lessons} lessons")

    # Save
    save_refined_curriculum(refined_graph, grammar_index, args.output_dir, refinements)

    # Save trials if ensemble mode
    if trials:
        trials_dir = args.output_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        for i, trial in enumerate(trials):
            if trial:
                trial_file = trials_dir / f"trial_{i+1}.json"
                with open(trial_file, "w", encoding="utf-8") as f:
                    json.dump(trial, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved {len(trials)} trial proposals to {trials_dir}")

    # Generate lesson context for lesson generation
    try:
        lesson_context_yaml = generate_lesson_context(refined_graph, refinements, client)
        context_file = args.output_dir / "lesson_context.yaml"
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(lesson_context_yaml)
        logger.info(f"  Generated lesson context: {context_file}")
    except Exception as e:
        logger.warning(f"Failed to generate lesson context: {e}")

    logger.info("")
    logger.info("=" * 50)
    logger.info("CURRICULUM REFINEMENT COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Original: {args.input_dir}")
    logger.info(f"Refined:  {args.output_dir}")
    if args.ensemble > 0:
        logger.info(f"Mode:     Ensemble ({args.ensemble} trials + synthesis)")
    logger.info("")
    logger.info("To use refined curriculum for lesson generation:")
    logger.info(f"  python scripts/04_generate_lessons.py --curriculum {args.output_dir} --all --select-poems")

    return 0


if __name__ == "__main__":
    sys.exit(main())
