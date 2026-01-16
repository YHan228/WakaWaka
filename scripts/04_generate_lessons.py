#!/usr/bin/env python3
"""
04_generate_lessons.py - Generate lesson content using LLM.

This script processes the curriculum graph and generates complete lesson
content for each lesson, including teaching sequences, grammar explanations,
and reference cards.

Key features:
- LLM-generated pedagogical content via Gemini API
- Context-aware: provides prerequisite summaries and poem annotations
- Batch processing with checkpointing (resume on failure)
- Schema validation with retry on malformed responses
- Generates lessons_manifest.json for tracking

Usage:
  python scripts/04_generate_lessons.py --all                    # Generate all lessons
  python scripts/04_generate_lessons.py --lesson-ids lesson_particle_wa,lesson_auxiliary_keri
  python scripts/04_generate_lessons.py --all --resume           # Resume interrupted generation
  python scripts/04_generate_lessons.py --max-lessons 5          # Quick test
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

# Import Google GenAI (new SDK)
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

from wakadecoder.schemas.lesson import (
    LessonContent,
    GrammarExplanation,
    PoemDisplay,
    IntroductionStep,
    PoemPresentationStep,
    GrammarSpotlightStep,
    ContrastExampleStep,
    ComprehensionCheckStep,
    SummaryStep,
    ReferenceCard,
    ForwardReference,
    VocabularyItem,
)
from wakadecoder.utils.prompt_loader import load_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_API_SLEEP = 1.0
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "lessons" / ".checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "lessons" / ".cache"
OUTPUT_DIR = PROJECT_ROOT / "data" / "lessons"


# -----------------------------------------------------------------------------
# Gemini API Client
# -----------------------------------------------------------------------------

class GeminiClient:
    """Wrapper for Gemini API with rate limiting and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
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
                    time.sleep(self.sleep_seconds * (attempt + 1))
                else:
                    raise


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_curriculum(curriculum_dir: Path) -> tuple[dict, dict]:
    """
    Load curriculum data.

    Returns:
        Tuple of (lesson_graph, grammar_index)
    """
    lesson_graph_path = curriculum_dir / "lesson_graph.json"
    grammar_index_path = curriculum_dir / "grammar_index.json"

    with open(lesson_graph_path, "r", encoding="utf-8") as f:
        lesson_graph = json.load(f)

    with open(grammar_index_path, "r", encoding="utf-8") as f:
        grammar_index = json.load(f)

    return lesson_graph, grammar_index


def load_poems(poems_path: Path) -> pd.DataFrame:
    """Load annotated poems."""
    return pd.read_parquet(poems_path)


def get_all_lessons(lesson_graph: dict) -> list[dict]:
    """Extract all lessons from lesson graph."""
    lessons = []
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            lesson["unit_id"] = unit["id"]
            lessons.append(lesson)
    return lessons


def get_lesson_by_id(lesson_graph: dict, lesson_id: str) -> dict | None:
    """Get a specific lesson by ID."""
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            if lesson["id"] == lesson_id:
                lesson["unit_id"] = unit["id"]
                return lesson
    return None


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to native Python types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def get_poems_for_lesson(lesson: dict, poems_df: pd.DataFrame) -> list[dict]:
    """Get poem data for a lesson's assigned poems."""
    poem_ids = lesson.get("poem_ids", [])
    poems = []

    for poem_id in poem_ids:
        poem_rows = poems_df[poems_df["poem_id"] == poem_id]
        if not poem_rows.empty:
            poem = poem_rows.iloc[0].to_dict()
            # Recursively convert numpy arrays to lists for JSON serialization
            poem = convert_numpy_to_python(poem)
            poems.append(poem)

    return poems


def get_prerequisite_summaries(
    lesson: dict,
    lesson_graph: dict,
    generated_lessons: dict
) -> str:
    """
    Build prerequisite summaries for context.

    Args:
        lesson: Current lesson being generated
        lesson_graph: Full lesson graph
        generated_lessons: Dict of already generated lesson content

    Returns:
        String summarizing prerequisites
    """
    prereq_ids = lesson.get("prerequisites", [])

    if not prereq_ids:
        return "None - this is an introductory lesson."

    summaries = []
    for prereq_id in prereq_ids:
        if prereq_id in generated_lessons:
            content = generated_lessons[prereq_id]
            summaries.append(f"- {content.get('lesson_title', prereq_id)}: {content.get('lesson_summary', 'N/A')}")
        else:
            # Try to get from graph
            prereq_lesson = get_lesson_by_id(lesson_graph, prereq_id)
            if prereq_lesson:
                gp = prereq_lesson.get("canonical_grammar_point", prereq_id)
                summaries.append(f"- {prereq_id}: Teaches {gp}")
            else:
                summaries.append(f"- {prereq_id}: (content not yet generated)")

    return "\n".join(summaries) if summaries else "None"


# -----------------------------------------------------------------------------
# LLM Response Parsing and Validation
# -----------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON
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
        raise ValueError(f"Could not extract JSON from response: {e}\n\nResponse:\n{text[:500]}...")


def validate_and_fix_lesson(llm_response: dict, lesson: dict) -> dict:
    """
    Validate and fix LLM lesson response.

    Args:
        llm_response: Parsed JSON from LLM
        lesson: Original lesson dict from curriculum

    Returns:
        Fixed lesson dict ready for LessonContent
    """
    # Ensure required fields
    if "lesson_title" not in llm_response or not llm_response["lesson_title"]:
        llm_response["lesson_title"] = f"Lesson: {lesson['canonical_grammar_point']}"

    if "lesson_summary" not in llm_response or not llm_response["lesson_summary"]:
        llm_response["lesson_summary"] = f"Learn about {lesson['canonical_grammar_point']}"

    # Ensure grammar_explanation exists
    if "grammar_explanation" not in llm_response:
        llm_response["grammar_explanation"] = {
            "concept": f"The {lesson['canonical_grammar_point']} grammar point",
            "formation": None,
            "variations": [],
            "common_confusions": [],
            "logic_analogy": None
        }
    else:
        ge = llm_response["grammar_explanation"]
        if "concept" not in ge:
            ge["concept"] = f"The {lesson['canonical_grammar_point']} grammar point"
        if "variations" not in ge:
            ge["variations"] = []
        if "common_confusions" not in ge:
            ge["common_confusions"] = []

    # Ensure teaching_sequence exists and has required types
    if "teaching_sequence" not in llm_response:
        llm_response["teaching_sequence"] = []

    teaching_sequence = llm_response["teaching_sequence"]

    # Check for required types
    types_present = {step.get("type") for step in teaching_sequence}

    # Add introduction if missing
    if "introduction" not in types_present:
        teaching_sequence.insert(0, {
            "type": "introduction",
            "content": llm_response.get("lesson_summary", "Welcome to this lesson.")
        })

    # Add comprehension_check if missing
    if "comprehension_check" not in types_present:
        teaching_sequence.append({
            "type": "comprehension_check",
            "question": f"What is the function of {lesson['canonical_grammar_point']}?",
            "answer": "Review the lesson content above.",
            "hint": "Think about the examples shown."
        })

    # Fix poem_presentation steps
    for step in teaching_sequence:
        if step.get("type") == "poem_presentation":
            if "display" not in step:
                step["display"] = {
                    "text_with_furigana": "(poem text)",
                    "romaji": "(romaji)",
                    "translation": "(translation)"
                }
            else:
                display = step["display"]
                if "text_with_furigana" not in display:
                    display["text_with_furigana"] = "(poem text)"
                if "romaji" not in display:
                    display["romaji"] = "(romaji)"
                if "translation" not in display:
                    display["translation"] = "(translation)"

            if "vocabulary" not in step:
                step["vocabulary"] = []

            if "poem_id" not in step:
                step["poem_id"] = "unknown"

        elif step.get("type") == "comprehension_check":
            if "question" not in step:
                step["question"] = "What did you learn?"
            if "answer" not in step:
                step["answer"] = "Review the lesson."

        elif step.get("type") == "grammar_spotlight":
            if "content" not in step:
                step["content"] = "Focus on the grammar point."
            if "evidence" not in step:
                step["evidence"] = "See the poem above."

    # Ensure reference_card exists
    if "reference_card" not in llm_response:
        llm_response["reference_card"] = {
            "point": lesson["canonical_grammar_point"],
            "one_liner": f"A grammar point in classical Japanese",
            "example": "See lesson content",
            "see_also": []
        }
    else:
        rc = llm_response["reference_card"]
        if "point" not in rc:
            rc["point"] = lesson["canonical_grammar_point"]
        if "one_liner" not in rc:
            rc["one_liner"] = "A grammar point"
        if "example" not in rc:
            rc["example"] = "See lesson"
        if "see_also" not in rc:
            rc["see_also"] = []

    # Ensure forward_references exists
    if "forward_references" not in llm_response:
        llm_response["forward_references"] = []

    return llm_response


def build_lesson_content(lesson: dict, llm_response: dict) -> LessonContent:
    """
    Build LessonContent from lesson data and LLM response.

    Args:
        lesson: Lesson dict from curriculum
        llm_response: Validated LLM response

    Returns:
        LessonContent instance
    """
    # Build grammar explanation
    ge_data = llm_response.get("grammar_explanation", {})
    grammar_explanation = GrammarExplanation(
        concept=ge_data.get("concept", ""),
        formation=ge_data.get("formation"),
        variations=ge_data.get("variations", []),
        common_confusions=ge_data.get("common_confusions", []),
        logic_analogy=ge_data.get("logic_analogy")
    )

    # Build teaching sequence
    teaching_sequence = []
    for step in llm_response.get("teaching_sequence", []):
        step_type = step.get("type")

        try:
            if step_type == "introduction":
                teaching_sequence.append(IntroductionStep(
                    content=step.get("content", "")
                ))
            elif step_type == "poem_presentation":
                display_data = step.get("display", {})
                display = PoemDisplay(
                    text_with_furigana=display_data.get("text_with_furigana", ""),
                    romaji=display_data.get("romaji", ""),
                    translation=display_data.get("translation", "")
                )

                vocabulary = []
                for v in step.get("vocabulary", []):
                    try:
                        vocabulary.append(VocabularyItem(
                            word=v.get("word", ""),
                            reading=v.get("reading", ""),
                            meaning=v.get("meaning", ""),
                            span=v.get("span"),
                            chinese_cognate_note=v.get("chinese_cognate_note")
                        ))
                    except Exception as e:
                        logger.warning(f"Skipping invalid vocabulary item: {v} - {e}")

                teaching_sequence.append(PoemPresentationStep(
                    poem_id=step.get("poem_id", ""),
                    display=display,
                    vocabulary=vocabulary,
                    focus_highlight=step.get("focus_highlight")
                ))
            elif step_type == "grammar_spotlight":
                teaching_sequence.append(GrammarSpotlightStep(
                    content=step.get("content", ""),
                    evidence=step.get("evidence", "")
                ))
            elif step_type == "contrast_example":
                teaching_sequence.append(ContrastExampleStep(
                    content=step.get("content", "")
                ))
            elif step_type == "comprehension_check":
                teaching_sequence.append(ComprehensionCheckStep(
                    question=step.get("question", ""),
                    answer=step.get("answer", ""),
                    hint=step.get("hint")
                ))
            elif step_type == "summary":
                teaching_sequence.append(SummaryStep(
                    content=step.get("content", "")
                ))
            else:
                logger.warning(f"Unknown step type: {step_type}")
        except Exception as e:
            logger.warning(f"Error building step {step_type}: {e}")

    # Ensure minimum 3 steps
    if len(teaching_sequence) < 3:
        if not any(isinstance(s, IntroductionStep) for s in teaching_sequence):
            teaching_sequence.insert(0, IntroductionStep(
                content=llm_response.get("lesson_summary", "Welcome to this lesson.")
            ))
        if not any(isinstance(s, ComprehensionCheckStep) for s in teaching_sequence):
            teaching_sequence.append(ComprehensionCheckStep(
                question="What did you learn in this lesson?",
                answer="Review the grammar point covered.",
                hint=None
            ))
        if len(teaching_sequence) < 3:
            teaching_sequence.append(SummaryStep(
                content="Review the concepts covered in this lesson."
            ))

    # Build reference card
    rc_data = llm_response.get("reference_card", {})
    reference_card = ReferenceCard(
        point=rc_data.get("point", lesson["canonical_grammar_point"]),
        one_liner=rc_data.get("one_liner", ""),
        example=rc_data.get("example", ""),
        see_also=rc_data.get("see_also", [])
    )

    # Build forward references
    forward_references = []
    for fr in llm_response.get("forward_references", []):
        try:
            forward_references.append(ForwardReference(
                point=fr.get("point", ""),
                note=fr.get("note", "")
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid forward reference: {fr} - {e}")

    return LessonContent(
        lesson_id=lesson["id"],
        lesson_title=llm_response.get("lesson_title", ""),
        lesson_summary=llm_response.get("lesson_summary", ""),
        grammar_point=lesson["canonical_grammar_point"],
        grammar_explanation=grammar_explanation,
        teaching_sequence=teaching_sequence,
        reference_card=reference_card,
        forward_references=forward_references
    )


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------

def get_cache_key(lesson_id: str, model: str, prompt_version: str) -> str:
    """Generate cache key for LLM response."""
    key = f"{lesson_id}_{model}_{prompt_version}"
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
    """Load set of already processed lesson IDs."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_checkpoint(checkpoint_file: Path, lesson_id: str):
    """Append lesson ID to checkpoint file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(f"{lesson_id}\n")


# -----------------------------------------------------------------------------
# Lesson Generation
# -----------------------------------------------------------------------------

def format_poems_for_prompt(poems: list[dict]) -> str:
    """Format poem data for LLM prompt."""
    formatted = []

    for i, poem in enumerate(poems, 1):
        poem_text = poem.get("text", "")
        poem_id = poem.get("poem_id", f"poem_{i}")
        reading_hiragana = poem.get("reading_hiragana", "")
        reading_romaji = poem.get("reading_romaji", "")
        difficulty = poem.get("difficulty_score", poem.get("difficulty_score_computed", 0))

        # Format grammar points
        grammar_points = poem.get("grammar_points", [])
        gp_list = []
        for gp in grammar_points:
            if isinstance(gp, dict):
                gp_str = f"  - {gp.get('canonical_id', 'unknown')}"
                if gp.get('sense_id'):
                    gp_str += f" ({gp['sense_id']})"
                gp_str += f": {gp.get('surface', '')}"
                if gp.get('span'):
                    gp_str += f" [span: {gp['span']}]"
                gp_list.append(gp_str)

        # Format vocabulary
        vocabulary = poem.get("vocabulary", [])
        vocab_list = []
        for v in vocabulary:
            if isinstance(v, dict):
                vocab_list.append(f"  - {v.get('word', '')}: {v.get('meaning', '')}")

        formatted.append(f"""
POEM {i} (ID: {poem_id})
Text: {poem_text}
Hiragana: {reading_hiragana}
Romaji: {reading_romaji}
Difficulty: {difficulty:.2f}
Grammar Points:
{chr(10).join(gp_list) if gp_list else '  (none)'}
Vocabulary:
{chr(10).join(vocab_list) if vocab_list else '  (none)'}
""")

    return "\n---\n".join(formatted)


def generate_lesson(
    lesson: dict,
    grammar_index: dict,
    lesson_graph: dict,
    poems_df: pd.DataFrame,
    client: GeminiClient,
    prompt_config: dict,
    generated_lessons: dict,
    use_cache: bool = True
) -> LessonContent | None:
    """
    Generate content for a single lesson.

    Args:
        lesson: Lesson dict from curriculum
        grammar_index: Grammar index data
        lesson_graph: Full lesson graph
        poems_df: Annotated poems dataframe
        client: Gemini client
        prompt_config: Loaded prompt configuration
        generated_lessons: Dict of already generated lessons
        use_cache: Whether to use response caching

    Returns:
        LessonContent or None on failure
    """
    lesson_id = lesson["id"]
    canonical_id = lesson["canonical_grammar_point"]

    # Check cache
    cache_key = get_cache_key(lesson_id, client.model_name, prompt_config["meta"]["version"])

    if use_cache:
        cached = load_cached_response(cache_key)
        if cached:
            logger.debug(f"Using cached response for {lesson_id}")
            try:
                fixed = validate_and_fix_lesson(cached, lesson)
                return build_lesson_content(lesson, fixed)
            except Exception as e:
                logger.warning(f"Cached response invalid for {lesson_id}: {e}")

    # Get grammar point info
    grammar_info = grammar_index.get("entries", {}).get(canonical_id, {})

    # Get poems for this lesson
    poems = get_poems_for_lesson(lesson, poems_df)
    if not poems:
        logger.warning(f"No poems found for lesson {lesson_id}")

    # Get prerequisite summaries
    prereq_summary = get_prerequisite_summaries(lesson, lesson_graph, generated_lessons)

    # Build prompt
    system_prompt = prompt_config.get("system", "")

    user_template = prompt_config.get("user_template", "")
    user_prompt = user_template.format(
        grammar_point_id=canonical_id,
        category=grammar_info.get("category", "unknown"),
        frequency=grammar_info.get("frequency", 0),
        avg_difficulty=f"{grammar_info.get('avg_difficulty', 0):.2f}",
        prerequisites_summary=prereq_summary,
        poems_json=format_poems_for_prompt(poems)
    )

    # Call LLM
    try:
        logger.info(f"Generating lesson {lesson_id}...")
        response_text = client.generate(system_prompt, user_prompt)

        # Parse response
        llm_response = extract_json_from_response(response_text)

        # Cache the response
        if use_cache:
            save_cached_response(cache_key, llm_response)

        # Validate and fix
        fixed = validate_and_fix_lesson(llm_response, lesson)

        # Build lesson content
        return build_lesson_content(lesson, fixed)

    except Exception as e:
        logger.error(f"Failed to generate lesson {lesson_id}: {e}")
        return None


def save_lesson(lesson_content: LessonContent, output_dir: Path):
    """Save lesson content to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{lesson_content.lesson_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(lesson_content.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved lesson: {output_file}")


def generate_manifest(output_dir: Path, lessons: list[LessonContent]):
    """Generate lessons_manifest.json."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_lessons": len(lessons),
        "lessons": [
            {
                "lesson_id": l.lesson_id,
                "lesson_title": l.lesson_title,
                "grammar_point": l.grammar_point
            }
            for l in lessons
        ]
    }

    manifest_file = output_dir / "lessons_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Generated manifest: {manifest_file}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate lesson content using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=PROJECT_ROOT / "data" / "curriculum",
        help="Path to curriculum directory"
    )
    parser.add_argument(
        "--poems",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Path to annotated poems parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for lesson JSON files"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all lessons"
    )
    parser.add_argument(
        "--lesson-ids",
        type=str,
        help="Comma-separated list of lesson IDs to generate"
    )
    parser.add_argument(
        "--max-lessons",
        type=int,
        help="Maximum number of lessons to generate (for testing)"
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
        help="Resume from checkpoint (skip already generated lessons)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching"
    )
    parser.add_argument(
        "--api-sleep",
        type=float,
        default=DEFAULT_API_SLEEP,
        help=f"Sleep between API calls (default: {DEFAULT_API_SLEEP}s)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.lesson_ids:
        parser.error("Must specify --all or --lesson-ids")

    # Load data
    logger.info("Loading curriculum...")
    lesson_graph, grammar_index = load_curriculum(args.curriculum)

    logger.info("Loading poems...")
    poems_df = load_poems(args.poems)
    logger.info(f"  Loaded {len(poems_df)} poems")

    # Load prompt
    logger.info("Loading prompt template...")
    prompt_config = load_prompt("generate_lesson")

    # Get lessons to generate
    all_lessons = get_all_lessons(lesson_graph)
    logger.info(f"  Total lessons in curriculum: {len(all_lessons)}")

    if args.lesson_ids:
        lesson_ids = [lid.strip() for lid in args.lesson_ids.split(",")]
        lessons_to_generate = [l for l in all_lessons if l["id"] in lesson_ids]
        if len(lessons_to_generate) != len(lesson_ids):
            found_ids = {l["id"] for l in lessons_to_generate}
            missing = set(lesson_ids) - found_ids
            logger.warning(f"Some lesson IDs not found: {missing}")
    else:
        lessons_to_generate = all_lessons

    if args.max_lessons:
        lessons_to_generate = lessons_to_generate[:args.max_lessons]

    logger.info(f"  Lessons to generate: {len(lessons_to_generate)}")

    # Handle checkpoint
    checkpoint_file = CHECKPOINT_DIR / "generated_lessons.txt"
    completed = set()

    if args.resume:
        completed = load_checkpoint(checkpoint_file)
        logger.info(f"  Resuming: {len(completed)} lessons already generated")
        lessons_to_generate = [l for l in lessons_to_generate if l["id"] not in completed]
        logger.info(f"  Remaining: {len(lessons_to_generate)} lessons")

    if not lessons_to_generate:
        logger.info("No lessons to generate. All done!")
        return

    # Initialize client
    client = GeminiClient(
        model=args.model,
        sleep_seconds=args.api_sleep
    )

    # Track generated lessons (for prerequisite context)
    generated_lessons = {}

    # Load any existing lesson content for prerequisite context
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for lesson_file in args.output_dir.glob("lesson_*.json"):
        try:
            with open(lesson_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                generated_lessons[content.get("lesson_id", "")] = content
        except Exception:
            pass

    # Generate lessons
    generated = []
    failed = []

    for i, lesson in enumerate(lessons_to_generate, 1):
        lesson_id = lesson["id"]
        logger.info(f"[{i}/{len(lessons_to_generate)}] Generating {lesson_id}...")

        try:
            lesson_content = generate_lesson(
                lesson=lesson,
                grammar_index=grammar_index,
                lesson_graph=lesson_graph,
                poems_df=poems_df,
                client=client,
                prompt_config=prompt_config,
                generated_lessons=generated_lessons,
                use_cache=not args.no_cache
            )

            if lesson_content:
                save_lesson(lesson_content, args.output_dir)
                save_checkpoint(checkpoint_file, lesson_id)
                generated.append(lesson_content)
                generated_lessons[lesson_id] = lesson_content.model_dump()
                logger.info(f"  ✓ Generated: {lesson_content.lesson_title}")
            else:
                failed.append(lesson_id)
                logger.error(f"  ✗ Failed: {lesson_id}")

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Progress saved.")
            break
        except Exception as e:
            logger.error(f"  ✗ Error generating {lesson_id}: {e}")
            failed.append(lesson_id)

    # Generate manifest
    if generated:
        # Load all generated lessons for manifest
        all_generated = []
        for lesson_file in args.output_dir.glob("lesson_*.json"):
            try:
                with open(lesson_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    all_generated.append(LessonContent(**content))
            except Exception as e:
                logger.warning(f"Could not load {lesson_file}: {e}")

        generate_manifest(args.output_dir, all_generated)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Generated: {len(generated)} lessons")
    logger.info(f"Failed: {len(failed)} lessons")
    if failed:
        logger.info(f"Failed IDs: {', '.join(failed)}")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
