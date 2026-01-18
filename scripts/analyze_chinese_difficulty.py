#!/usr/bin/env python3
"""
analyze_chinese_difficulty.py - Chinese learner-adjusted difficulty analysis.

This script creates adjusted difficulty scores for Chinese speakers by:
1. Calculating a "kanji cognate bonus" - kanji helps Chinese learners
2. Adjusting the original difficulty score downward for kanji-rich poems
3. Computing marginal/incremental difficulty for curriculum sequencing

Output: data/analysis/chinese_difficulty.json, data/analysis/marginal_difficulty.json
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_utils import setup_plotting

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def count_kanji(text: str) -> int:
    """Count CJK unified ideographs (kanji) in text."""
    if not text:
        return 0
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')


def count_hiragana(text: str) -> int:
    """Count hiragana characters in text."""
    if not text:
        return 0
    return sum(1 for c in text if '\u3040' <= c <= '\u309f')


def count_katakana(text: str) -> int:
    """Count katakana characters in text."""
    if not text:
        return 0
    return sum(1 for c in text if '\u30a0' <= c <= '\u30ff')


def calculate_kanji_cognate_bonus(row: pd.Series) -> dict:
    """
    Calculate the kanji cognate bonus for a poem.

    Chinese learners benefit from kanji as they share meanings with Chinese characters.
    The bonus is based on:
    - Kanji count in the kanji_transcription (LLM-provided kanji version)
    - Kanji-to-total ratio (information density)

    Returns dict with bonus calculation details.
    """
    text = row['text']
    kanji_text = row.get('kanji_transcription', text) or text

    # Character counts from kanji transcription (more informative for Chinese learners)
    kanji_count = count_kanji(kanji_text)
    hiragana_count = count_hiragana(kanji_text)
    katakana_count = count_katakana(kanji_text)

    # Total meaningful characters (excluding punctuation)
    total_chars = kanji_count + hiragana_count + katakana_count
    if total_chars == 0:
        total_chars = len(kanji_text)

    # Kanji ratio: what fraction of the poem is kanji?
    kanji_ratio = kanji_count / total_chars if total_chars > 0 else 0

    # Cognate bonus formula:
    # - Base bonus scales with kanji ratio (more kanji = easier for Chinese learners)
    # - Maximum bonus is 0.25 (capped to not over-adjust)
    # - Minimum is 0 (no penalty for hiragana-heavy poems)
    raw_bonus = kanji_ratio * 0.35  # Scale factor
    cognate_bonus = min(0.25, raw_bonus)  # Cap at 0.25

    return {
        'kanji_count': kanji_count,
        'hiragana_count': hiragana_count,
        'total_chars': total_chars,
        'kanji_ratio': round(kanji_ratio, 4),
        'kanji_cognate_bonus': round(cognate_bonus, 4),
    }


def analyze_chinese_difficulty(df: pd.DataFrame) -> list[dict]:
    """
    Calculate Chinese-adjusted difficulty for all poems.

    Formula: adjusted_difficulty = original_difficulty - kanji_cognate_bonus
    """
    results = []

    for idx, row in df.iterrows():
        poem_id = row['poem_id']
        original_difficulty = row.get('difficulty_score_computed', 0.5) or 0.5

        # Calculate cognate bonus
        bonus_info = calculate_kanji_cognate_bonus(row)

        # Adjusted difficulty (clamped to [0, 1])
        adjusted_difficulty = max(0.0, original_difficulty - bonus_info['kanji_cognate_bonus'])
        adjusted_difficulty = round(adjusted_difficulty, 4)

        results.append({
            'poem_id': poem_id,
            'original_difficulty': round(original_difficulty, 4),
            'kanji_count': bonus_info['kanji_count'],
            'kanji_ratio': bonus_info['kanji_ratio'],
            'kanji_cognate_bonus': bonus_info['kanji_cognate_bonus'],
            'chinese_adjusted_difficulty': adjusted_difficulty,
        })

    return results


def analyze_marginal_difficulty(df: pd.DataFrame) -> dict:
    """
    Calculate marginal/incremental difficulty for curriculum sequencing.

    For each poem, compute:
    - What grammar points are NEW vs already seen in simpler poems
    - Marginal difficulty = new_grammar_weight / total_grammar

    This enables greedy curriculum construction: "If student knows X, what's easiest next?"
    """
    # First, sort poems by adjusted difficulty
    chinese_results = analyze_chinese_difficulty(df)
    difficulty_map = {r['poem_id']: r['chinese_adjusted_difficulty'] for r in chinese_results}

    # Get grammar points per poem
    poem_grammar = {}
    all_grammar_ids = set()

    for idx, row in df.iterrows():
        poem_id = row['poem_id']
        grammar_points = row.get('grammar_points', [])
        if hasattr(grammar_points, 'tolist'):
            grammar_points = grammar_points.tolist()

        # Extract canonical_ids
        gp_ids = set()
        for gp in grammar_points:
            if isinstance(gp, dict):
                canonical_id = gp.get('canonical_id', '')
                if canonical_id:
                    gp_ids.add(canonical_id)
                    all_grammar_ids.add(canonical_id)

        poem_grammar[poem_id] = gp_ids

    # Sort poems by difficulty
    sorted_poems = sorted(df['poem_id'].tolist(), key=lambda x: difficulty_map.get(x, 0.5))

    # Calculate marginal difficulty: simulate learning path
    known_grammar = set()
    marginal_results = []
    poem_sequence = []  # Optimal learning sequence

    for poem_id in sorted_poems:
        gp_ids = poem_grammar.get(poem_id, set())

        if not gp_ids:
            # No grammar points - easy poem
            marginal = 0.0
            new_count = 0
        else:
            new_points = gp_ids - known_grammar
            new_count = len(new_points)
            marginal = new_count / len(gp_ids)

        marginal_results.append({
            'poem_id': poem_id,
            'base_difficulty': difficulty_map.get(poem_id, 0.5),
            'total_grammar_points': len(gp_ids),
            'new_grammar_points': new_count,
            'marginal_difficulty': round(marginal, 4),
            'grammar_ids': list(gp_ids),
            'new_grammar_ids': list(gp_ids - known_grammar),
        })

        # Learn the grammar from this poem
        known_grammar.update(gp_ids)
        poem_sequence.append(poem_id)

    # Calculate "next easiest" recommendations for each poem
    # For each poem, find the easiest poem that introduces exactly 1 new grammar point
    next_recommendations = {}

    for i, current in enumerate(marginal_results):
        current_known = set()
        for j in range(i + 1):
            current_known.update(poem_grammar.get(marginal_results[j]['poem_id'], set()))

        # Find candidates with marginal difficulty
        candidates = []
        for j in range(i + 1, len(marginal_results)):
            candidate = marginal_results[j]
            gp_ids = poem_grammar.get(candidate['poem_id'], set())
            new_points = gp_ids - current_known

            if len(new_points) <= 2:  # Introduces at most 2 new grammar points
                candidates.append({
                    'poem_id': candidate['poem_id'],
                    'new_grammar_count': len(new_points),
                    'base_difficulty': candidate['base_difficulty'],
                })

        # Sort by new_grammar_count, then by difficulty
        candidates.sort(key=lambda x: (x['new_grammar_count'], x['base_difficulty']))
        next_recommendations[current['poem_id']] = candidates[:3]  # Top 3 recommendations

    return {
        'marginal_difficulty': marginal_results,
        'optimal_sequence': poem_sequence,
        'next_recommendations': next_recommendations,
        'total_grammar_points_discovered': len(all_grammar_ids),
    }


def plot_difficulty_comparison(chinese_results: list[dict], output_dir: Path):
    """Plot original vs adjusted difficulty distributions."""
    original = [r['original_difficulty'] for r in chinese_results]
    adjusted = [r['chinese_adjusted_difficulty'] for r in chinese_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram comparison
    bins = np.linspace(0, 1, 21)
    axes[0].hist(original, bins=bins, alpha=0.7, label='Original', color='steelblue')
    axes[0].hist(adjusted, bins=bins, alpha=0.7, label='Chinese-adjusted', color='coral')
    axes[0].set_xlabel('Difficulty Score')
    axes[0].set_ylabel('Number of Poems')
    axes[0].set_title('Difficulty Distribution Comparison')
    axes[0].legend()

    # Scatter plot: original vs adjusted
    axes[1].scatter(original, adjusted, alpha=0.5, s=20)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x (no change)')
    axes[1].set_xlabel('Original Difficulty')
    axes[1].set_ylabel('Chinese-adjusted Difficulty')
    axes[1].set_title('Difficulty Adjustment')
    axes[1].legend()

    # Kanji bonus distribution
    bonuses = [r['kanji_cognate_bonus'] for r in chinese_results]
    axes[2].hist(bonuses, bins=20, color='seagreen', alpha=0.7)
    axes[2].set_xlabel('Kanji Cognate Bonus')
    axes[2].set_ylabel('Number of Poems')
    axes[2].set_title('Kanji Cognate Bonus Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / "chinese_difficulty_comparison.png", dpi=150)
    plt.close()


def plot_marginal_difficulty(marginal_data: dict, output_dir: Path):
    """Plot marginal difficulty analysis."""
    results = marginal_data['marginal_difficulty']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Marginal difficulty over learning sequence
    marginal_scores = [r['marginal_difficulty'] for r in results]
    base_difficulties = [r['base_difficulty'] for r in results]

    # Rolling average for smoothing
    window = 20
    rolling_marginal = pd.Series(marginal_scores).rolling(window=window, min_periods=1).mean()

    axes[0].plot(rolling_marginal, color='coral', linewidth=2, label=f'Marginal (rolling avg {window})')
    axes[0].scatter(range(len(marginal_scores)), marginal_scores, alpha=0.2, s=10, color='coral')
    axes[0].set_xlabel('Poem Order (by difficulty)')
    axes[0].set_ylabel('Marginal Difficulty')
    axes[0].set_title('Marginal Difficulty Over Learning Sequence')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # New grammar points per poem
    new_counts = [r['new_grammar_points'] for r in results]
    cumulative_grammar = np.cumsum(new_counts)

    axes[1].bar(range(len(new_counts)), new_counts, alpha=0.5, color='steelblue', label='New grammar points')
    ax2 = axes[1].twinx()
    ax2.plot(cumulative_grammar, color='darkgreen', linewidth=2, label='Cumulative grammar')
    axes[1].set_xlabel('Poem Order (by difficulty)')
    axes[1].set_ylabel('New Grammar Points', color='steelblue')
    ax2.set_ylabel('Cumulative Grammar Learned', color='darkgreen')
    axes[1].set_title('Grammar Learning Progression')

    plt.tight_layout()
    plt.savefig(output_dir / "marginal_difficulty.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")

    # Chinese-adjusted difficulty
    print("\n" + "="*50)
    print("Calculating Chinese-adjusted difficulty...")
    chinese_results = analyze_chinese_difficulty(df)

    # Statistics
    original_diffs = [r['original_difficulty'] for r in chinese_results]
    adjusted_diffs = [r['chinese_adjusted_difficulty'] for r in chinese_results]

    print(f"\nOriginal difficulty:")
    print(f"  Mean: {np.mean(original_diffs):.3f}")
    print(f"  Std:  {np.std(original_diffs):.3f}")
    print(f"  Range: {min(original_diffs):.3f} - {max(original_diffs):.3f}")

    print(f"\nChinese-adjusted difficulty:")
    print(f"  Mean: {np.mean(adjusted_diffs):.3f}")
    print(f"  Std:  {np.std(adjusted_diffs):.3f}")
    print(f"  Range: {min(adjusted_diffs):.3f} - {max(adjusted_diffs):.3f}")

    # Check distribution spread
    orig_plateau = sum(1 for d in original_diffs if 0.4 <= d <= 0.6)
    adj_plateau = sum(1 for d in adjusted_diffs if 0.4 <= d <= 0.6)
    print(f"\nPoems in 0.4-0.6 range:")
    print(f"  Original: {orig_plateau} ({100*orig_plateau/len(original_diffs):.1f}%)")
    print(f"  Adjusted: {adj_plateau} ({100*adj_plateau/len(adjusted_diffs):.1f}%)")

    # Save Chinese difficulty results
    output_file = OUTPUT_DIR / "chinese_difficulty.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chinese_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_file}")

    # Marginal difficulty
    print("\n" + "="*50)
    print("Calculating marginal difficulty...")
    marginal_data = analyze_marginal_difficulty(df)

    print(f"\nMarginal difficulty stats:")
    marginal_scores = [r['marginal_difficulty'] for r in marginal_data['marginal_difficulty']]
    print(f"  Mean: {np.mean(marginal_scores):.3f}")
    print(f"  Total grammar points discovered: {marginal_data['total_grammar_points_discovered']}")

    # Save marginal difficulty results
    marginal_file = OUTPUT_DIR / "marginal_difficulty.json"
    with open(marginal_file, 'w', encoding='utf-8') as f:
        json.dump(marginal_data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {marginal_file}")

    # Generate plots
    print("\n" + "="*50)
    print("Generating plots...")
    plot_difficulty_comparison(chinese_results, OUTPUT_DIR)
    plot_marginal_difficulty(marginal_data, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR}")

    print("\n" + "="*50)
    print("Done!")


if __name__ == "__main__":
    main()
