#!/usr/bin/env python3
"""
fix_vocabulary_spans.py - Post-process lessons to add vocabulary spans.

Computes [start, end) spans for each vocabulary item by matching
word positions in the original poem text.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

def compute_vocab_spans(poem_text: str, vocabulary: list[dict]) -> list[dict]:
    """
    Compute spans for vocabulary items by matching positions in poem text.

    Uses greedy forward matching - finds next occurrence after last match.
    Handles repeated words (particles like の, に, etc.)

    Args:
        poem_text: Original poem text
        vocabulary: List of vocab dicts with 'word' field

    Returns:
        Updated vocabulary list with 'span' fields populated
    """
    last_end = 0  # Track position to search from

    for vocab in vocabulary:
        word = vocab.get('word', '')
        if not word:
            continue

        # Find next occurrence after last_end
        pos = poem_text.find(word, last_end)

        if pos >= 0:
            vocab['span'] = [pos, pos + len(word)]
            # Don't advance last_end for single-char particles that might be part of compounds
            # Only advance if word length > 1 or we found it after current position
            if len(word) > 1:
                last_end = pos + len(word)
            else:
                # For single chars, only advance past if it's clearly separate
                last_end = pos + 1
        else:
            # Word not found after last_end, try from beginning
            pos = poem_text.find(word)
            if pos >= 0:
                vocab['span'] = [pos, pos + len(word)]
            else:
                # Still not found - might be a conjugation variant or reading
                vocab['span'] = None

    return vocabulary


def fix_lesson_spans(lesson_path: Path, poems_df: pd.DataFrame) -> tuple[int, int]:
    """
    Fix vocabulary spans in a single lesson file.

    Returns:
        Tuple of (total_vocab_items, items_with_spans)
    """
    with open(lesson_path, 'r', encoding='utf-8') as f:
        lesson = json.load(f)

    total_items = 0
    items_with_spans = 0
    modified = False

    for step in lesson.get('teaching_sequence', []):
        if step.get('type') != 'poem_presentation':
            continue

        poem_id = step.get('poem_id')
        vocabulary = step.get('vocabulary', [])

        if not poem_id or not vocabulary:
            continue

        # Get poem text
        poem_rows = poems_df[poems_df['poem_id'] == poem_id]
        if poem_rows.empty:
            print(f"  Warning: poem {poem_id} not found in parquet")
            continue

        poem_text = poem_rows.iloc[0]['text']

        # Compute spans
        updated_vocab = compute_vocab_spans(poem_text, vocabulary)
        step['vocabulary'] = updated_vocab
        modified = True

        # Count stats
        for v in updated_vocab:
            total_items += 1
            if v.get('span'):
                items_with_spans += 1

    # Save updated lesson
    if modified:
        with open(lesson_path, 'w', encoding='utf-8') as f:
            json.dump(lesson, f, ensure_ascii=False, indent=2)

    return total_items, items_with_spans


def main():
    lessons_dir = PROJECT_ROOT / 'data' / 'lessons'
    poems_path = PROJECT_ROOT / 'data' / 'annotated' / 'poems.parquet'

    print("Loading poems...")
    poems_df = pd.read_parquet(poems_path)
    print(f"  Loaded {len(poems_df)} poems")

    lesson_files = sorted(lessons_dir.glob('lesson_*.json'))
    print(f"\nProcessing {len(lesson_files)} lessons...")

    total_all = 0
    spans_all = 0

    for lesson_path in lesson_files:
        total, with_spans = fix_lesson_spans(lesson_path, poems_df)
        total_all += total
        spans_all += with_spans

        pct = (with_spans / total * 100) if total > 0 else 0
        print(f"  {lesson_path.name}: {with_spans}/{total} vocab items with spans ({pct:.0f}%)")

    print(f"\n{'='*50}")
    print(f"TOTAL: {spans_all}/{total_all} vocab items with spans ({spans_all/total_all*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
