#!/usr/bin/env python3
"""
Generate Introduction Lesson using Ensemble LLM Generation.

This script generates the introductory lesson for WakaDecoder using:
1. Multiple independent LLM trials (ensemble)
2. A judge LLM to synthesize the best elements

Usage:
    python scripts/06_generate_introduction.py [--trials N] [--parallel N]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Import Google GenAI (new SDK)
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakawaka.classroom import ClassroomLoader

load_dotenv()


def load_prompt(prompt_name: str) -> dict:
    """Load a prompt template from YAML."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.yaml"
    with open(prompt_path) as f:
        return yaml.safe_load(f)


def get_curriculum_context(loader: ClassroomLoader) -> dict:
    """Extract curriculum context for the prompt."""
    units = loader.get_units()
    total_lessons = loader.get_lesson_count()
    total_poems = loader.get_poem_count()

    unit_summary = []
    for unit in units:
        lessons = loader.get_lessons_for_unit(unit.id)
        lesson_titles = [f"  - {l.title}" for l in lessons[:3]]
        if len(lessons) > 3:
            lesson_titles.append(f"  - ... and {len(lessons) - 3} more")
        unit_summary.append(f"**{unit.title}** ({len(lessons)} lessons):\n" + "\n".join(lesson_titles))

    return {
        "total_units": len(units),
        "total_lessons": total_lessons,
        "total_poems": total_poems,
        "unit_summary": "\n\n".join(unit_summary),
    }


def extract_json(text: str, debug: bool = False) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    original_text = text

    if debug:
        print(f"  [DEBUG] Raw response length: {len(text)}")
        print(f"  [DEBUG] First 300 chars: {repr(text[:300])}")

    # If it already starts with {, it's likely clean JSON
    text = text.strip()
    if text.startswith('{'):
        # Fix common JSON issues - trailing commas
        text = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  # Fall through to other methods

    # Try to find JSON in code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        text = json_match.group(1).strip()
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)

    # Clean up and parse
    text = text.strip()

    # Fix common JSON issues
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Original text preview: {repr(original_text[:500])}")
        print(f"Processed text preview: {repr(text[:500])}")
        raise


def generate_trial(
    client: genai.Client,
    model_name: str,
    prompt: dict,
    context: dict,
    trial_id: int,
    temperature: float = 0.9,
) -> dict:
    """Generate a single trial of the introduction."""
    print(f"  Trial {trial_id}: Starting...")

    # Format the prompts with context
    system = prompt["system"].format(**context)
    user = prompt["user"].format(**context)

    full_prompt = f"{system}\n\n---\n\n{user}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=16000,
            ),
        )

        # Handle potential empty response
        response_text = None
        if response.text:
            response_text = response.text
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text

        if not response_text:
            raise ValueError("Empty response from model")

        result = extract_json(response_text, debug=(trial_id == 1))
        print(f"  Trial {trial_id}: Complete ({len(result.get('sections', []))} sections)")
        return {"trial_id": trial_id, "result": result, "success": True}

    except Exception as e:
        print(f"  Trial {trial_id}: Failed - {e}")
        import traceback
        traceback.print_exc()
        return {"trial_id": trial_id, "error": str(e), "success": False}


def synthesize_trials(
    client: genai.Client,
    model_name: str,
    prompt: dict,
    trials: list[dict],
) -> dict:
    """Synthesize multiple trials into one optimal introduction."""
    print("\n  Synthesizing trials...")

    # Format drafts for the synthesis prompt
    drafts_text = ""
    for i, trial in enumerate(trials, 1):
        drafts_text += f"\n\n=== DRAFT {i} ===\n{json.dumps(trial['result'], indent=2, ensure_ascii=False)}"

    system = prompt["synthesis_system"].format(num_trials=len(trials))
    user = prompt["synthesis_user"].format(num_trials=len(trials), drafts=drafts_text)

    full_prompt = f"{system}\n\n---\n\n{user}"

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.3,  # Lower temperature for synthesis
            max_output_tokens=16000,
        ),
    )

    response_text = response.text
    if not response_text and response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            response_text = candidate.content.parts[0].text

    return extract_json(response_text)


def main():
    parser = argparse.ArgumentParser(description="Generate introduction lesson")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials (default: 5)")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel workers (default: 3)")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model to use")
    parser.add_argument("--output", default="data/introduction.json", help="Output path")
    parser.add_argument("--trials-dir", default="data/introduction_trials", help="Directory for trial outputs")
    args = parser.parse_args()

    # Setup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set in environment")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load curriculum context
    print("Loading curriculum context...")
    db_path = Path("data/classroom.db")
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run the build pipeline first to create classroom.db")
        sys.exit(1)

    loader = ClassroomLoader(db_path)
    context = get_curriculum_context(loader)
    print(f"  Units: {context['total_units']}")
    print(f"  Lessons: {context['total_lessons']}")
    print(f"  Poems: {context['total_poems']}")

    # Load prompt
    prompt = load_prompt("generate_introduction")

    # Create trials directory
    trials_dir = Path(args.trials_dir)
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Run trials in parallel
    print(f"\nRunning {args.trials} trials with {args.parallel} workers...")
    successful_trials = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                generate_trial, client, args.model, prompt, context, i, 0.9
            ): i for i in range(1, args.trials + 1)
        }

        for future in as_completed(futures):
            trial_id = futures[future]
            try:
                result = future.result()
                if result["success"]:
                    successful_trials.append(result)
                    # Save individual trial
                    trial_path = trials_dir / f"trial_{trial_id}.json"
                    with open(trial_path, "w") as f:
                        json.dump(result["result"], f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Trial {trial_id}: Exception - {e}")

    print(f"\n  Successful trials: {len(successful_trials)}/{args.trials}")

    if len(successful_trials) < 2:
        print("Error: Need at least 2 successful trials for synthesis")
        if successful_trials:
            # Use the single trial as fallback
            final = successful_trials[0]["result"]
        else:
            sys.exit(1)
    else:
        # Synthesize trials
        print("\nSynthesizing best elements from all trials...")
        final = synthesize_trials(client, args.model, prompt, successful_trials)

    # Save final result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Introduction saved to: {output_path}")
    print(f"  Title: {final.get('title', 'N/A')}")
    print(f"  Sections: {len(final.get('sections', []))}")


if __name__ == "__main__":
    main()
