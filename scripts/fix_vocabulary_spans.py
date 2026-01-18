#!/usr/bin/env python3
"""
fix_vocabulary_spans.py - Post-process lessons to add vocabulary spans.

Computes [start, end) spans for each vocabulary item by matching
word positions in the poem text extracted from the lesson's furigana display.

Updated to handle new waka tokenization and LLM-generated vocabulary.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_plain_text(furigana_html: str) -> str:
    """Extract plain text from furigana HTML."""
    # Remove ruby annotations: <ruby>漢字<rt>かんじ</rt></ruby> -> 漢字
    text = re.sub(r'<ruby>([^<]+)<rt>[^<]+</rt></ruby>', r'\1', furigana_html)
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text


def compute_vocab_spans(poem_text: str, vocabulary: list[dict]) -> list[dict]:
    """
    Compute spans for vocabulary items by matching positions in poem text.

    Strategy:
    1. First pass: find all possible positions for each word
    2. Assign spans greedily in poem order (left to right)
    3. Track used character positions to avoid overlaps

    Args:
        poem_text: Plain poem text (no HTML)
        vocabulary: List of vocab dicts with 'word' field

    Returns:
        Updated vocabulary list with 'span' fields populated
    """
    # Track which positions are already assigned
    used_positions = set()

    # Find all possible matches for each vocab word
    vocab_matches = []
    for idx, vocab in enumerate(vocabulary):
        word = vocab.get('word', '')
        if not word:
            vocab_matches.append((idx, word, []))
            continue

        # Find all occurrences of this word in the text
        positions = []
        start = 0
        while True:
            pos = poem_text.find(word, start)
            if pos < 0:
                break
            positions.append(pos)
            start = pos + 1

        vocab_matches.append((idx, word, positions))

    # Sort vocabulary by their first appearance in the poem
    # This helps assign spans in reading order
    def first_pos(item):
        idx, word, positions = item
        return positions[0] if positions else float('inf')

    vocab_matches_sorted = sorted(vocab_matches, key=first_pos)

    # Assign spans greedily
    results = [None] * len(vocabulary)

    for idx, word, positions in vocab_matches_sorted:
        if not positions:
            results[idx] = None
            continue

        # Find first position that doesn't overlap with used positions
        assigned = False
        for pos in positions:
            span_positions = set(range(pos, pos + len(word)))
            if not span_positions & used_positions:
                # This position is available
                results[idx] = [pos, pos + len(word)]
                used_positions.update(span_positions)
                assigned = True
                break

        if not assigned:
            # All positions overlap - try to find any match
            # (might happen with repeated particles)
            results[idx] = None

    # Update vocabulary with spans
    for idx, vocab in enumerate(vocabulary):
        vocab['span'] = results[idx]

    return vocabulary


def fix_lesson_spans(lesson_path: Path) -> tuple[int, int]:
    """
    Fix vocabulary spans in a single lesson file.

    Uses the lesson's own display text (from furigana HTML) rather than
    external parquet file, ensuring consistency with LLM tokenization.

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

        vocabulary = step.get('vocabulary', [])
        display = step.get('display', {})
        furigana_html = display.get('text_with_furigana', '')

        if not vocabulary or not furigana_html:
            continue

        # Extract plain text from the lesson's own furigana display
        poem_text = extract_plain_text(furigana_html)

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

    lesson_files = sorted(lessons_dir.glob('lesson_*.json'))
    print(f"Processing {len(lesson_files)} lessons...")

    total_all = 0
    spans_all = 0

    for lesson_path in lesson_files:
        total, with_spans = fix_lesson_spans(lesson_path)
        total_all += total
        spans_all += with_spans

        pct = (with_spans / total * 100) if total > 0 else 0
        print(f"  {lesson_path.name}: {with_spans}/{total} vocab items with spans ({pct:.0f}%)")

    print(f"\n{'='*50}")
    print(f"TOTAL: {spans_all}/{total_all} vocab items with spans ({spans_all/total_all*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
