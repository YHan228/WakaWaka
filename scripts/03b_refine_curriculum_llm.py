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

def format_lessons_for_trial(lesson_graph: dict, grammar_index: dict) -> str:
    """Format lessons as JSON for trial prompt."""
    lessons = []
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            lid = lesson.get("id", "")
            gp = lesson.get("canonical_grammar_point", "")
            gp_info = grammar_index.get("entries", {}).get(gp, {})

            lessons.append({
                "id": lid,
                "grammar_point": gp,
                "category": gp_info.get("category", "unknown"),
                "frequency": gp_info.get("frequency", 0),
                "difficulty_tier": lesson.get("difficulty_tier", 3),
                "surfaces": gp_info.get("surfaces", [])[:3]
            })

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
    valid_ids: set[str]
) -> dict | None:
    """Run a single ensemble trial."""
    logger.info(f"  Running trial {trial_num}...")

    lessons_json = format_lessons_for_trial(lesson_graph, grammar_index)
    grammar_summary = format_grammar_summary(grammar_index)

    system_prompt = trial_prompt_config.get("system", "")
    user_template = trial_prompt_config.get("user_template", "")

    user_prompt = user_template.format(
        num_lessons=len(valid_ids),
        trial_number=trial_num,
        lessons_json=lessons_json,
        grammar_summary=grammar_summary
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
    synthesis_prompt_config: dict
) -> dict:
    """Synthesize final curriculum from multiple trials."""
    logger.info("Synthesizing final curriculum from trials...")

    # Format trials for synthesis prompt
    master_lessons = [{"id": lid} for lid in sorted(valid_ids)]

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
        master_lessons_json=json.dumps(master_lessons, indent=2),
        trials_json=json.dumps(trials_data, ensure_ascii=False, indent=2)
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
    max_parallel: int = 3
) -> tuple[dict, list[dict]]:
    """
    Run ensemble curriculum generation with parallel trials.

    Args:
        lesson_graph: Original curriculum
        grammar_index: Grammar index
        client: Gemini client
        num_trials: Number of independent trials
        max_parallel: Maximum parallel API calls

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
            valid_ids=valid_ids
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

    # Synthesize with lower temperature
    client.temperature = synthesis_prompt.get("meta", {}).get("temperature", 0.3)
    synthesis = synthesize_trials(valid_trials, valid_ids, client, synthesis_prompt)

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
            unit = {
                "id": unit_def.get("id", f"unit_{len(new_graph['units']) + 1:02d}"),
                "title": unit_def.get("title"),
                "lessons": []
            }

            for lesson_id in unit_def.get("lessons", []):
                if lesson_id in lesson_lookup:
                    lesson = lesson_lookup[lesson_id].copy()
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

    # Update meta
    new_graph["meta"]["refined_at"] = datetime.now().isoformat()
    new_graph["meta"]["refinement_model"] = "llm"
    new_graph["meta"]["total_lessons"] = sum(len(u["lessons"]) for u in new_graph["units"])

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
    lines = [
        "# Refined Curriculum Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Total units**: {len(refined_graph.get('units', []))}",
        f"- **Total lessons**: {refined_graph.get('meta', {}).get('total_lessons', 0)}",
        "",
        "## Units and Lessons",
        "",
    ]

    for unit in refined_graph.get("units", []):
        unit_id = unit.get("id", "unknown")
        unit_title = unit.get("title", unit_id)

        lines.append(f"### {unit_id}: {unit_title}")
        lines.append("")

        for lesson in unit.get("lessons", []):
            lesson_id = lesson.get("id", "")
            grammar = lesson.get("canonical_grammar_point", "")
            prereqs = lesson.get("prerequisites", [])

            prereq_str = f" (prereqs: {', '.join(prereqs)})" if prereqs else ""
            lines.append(f"- `{lesson_id}` — {grammar}{prereq_str}")

        lines.append("")

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

    args = parser.parse_args()

    # Load original curriculum
    logger.info(f"Loading curriculum from {args.input_dir}...")
    lesson_graph, grammar_index = load_curriculum(args.input_dir)

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
                max_parallel=args.parallel
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
