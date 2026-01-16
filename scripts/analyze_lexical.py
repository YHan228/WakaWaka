#!/usr/bin/env python3
"""
analyze_lexical.py - Lexical diversity and distribution analysis.

Analyses:
- Zipf's law fit (log-log rank-frequency plot)
- Type-Token Ratio (TTR)
- Yule's K (vocabulary richness)
- MTLD (Measure of Textual Lexical Diversity)

Output: data/analysis/lexical/
"""

import sys
from pathlib import Path
from collections import Counter
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_utils import setup_plotting

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "lexical"


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def extract_all_tokens(df: pd.DataFrame) -> list[str]:
    """Extract all token surfaces from corpus."""
    tokens = []
    for token_list in df['fugashi_tokens']:
        if hasattr(token_list, 'tolist'):
            token_list = token_list.tolist()
        for t in token_list:
            if isinstance(t, dict):
                tokens.append(t['surface'])
    return tokens


def analyze_zipf(tokens: list[str], output_dir: Path):
    """Analyze Zipf's law fit."""
    freq = Counter(tokens)

    # Sort by frequency (descending)
    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1)
    frequencies = np.array(sorted_freq)

    # Log-log regression
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)

    # Filter out zeros for regression
    mask = frequencies > 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks[mask], log_freqs[mask]
    )

    # Zipf's law predicts slope ≈ -1
    print(f"Zipf's Law Analysis:")
    print(f"  Slope: {slope:.3f} (ideal: -1.0)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  Unique tokens: {len(freq)}")
    print(f"  Total tokens: {len(tokens)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ranks, frequencies, 'b.', alpha=0.5, markersize=3, label='Observed')

    # Fitted line
    fitted = 10**(intercept + slope * log_ranks)
    ax.loglog(ranks, fitted, 'r-', linewidth=2,
              label=f'Zipf fit (slope={slope:.2f}, R²={r_value**2:.3f})')

    # Ideal Zipf (slope = -1)
    ideal = frequencies[0] / ranks
    ax.loglog(ranks, ideal, 'g--', linewidth=1, alpha=0.7, label='Ideal Zipf (slope=-1)')

    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title("Zipf's Law Analysis - Waka Corpus")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "zipf_law.png", dpi=150)
    plt.close()

    # Save top tokens
    top_tokens = freq.most_common(100)
    with open(output_dir / "zipf_top_tokens.csv", "w", encoding="utf-8") as f:
        f.write("rank,token,frequency\n")
        for i, (token, count) in enumerate(top_tokens, 1):
            f.write(f"{i},{token},{count}\n")

    return {"slope": slope, "r_squared": r_value**2, "unique": len(freq), "total": len(tokens)}


def compute_ttr(tokens: list[str]) -> float:
    """Compute Type-Token Ratio."""
    types = len(set(tokens))
    return types / len(tokens) if tokens else 0


def compute_yules_k(tokens: list[str]) -> float:
    """
    Compute Yule's K (characteristic constant).

    K = 10^4 * (Σ(f_i² * V_i) - N) / N²
    where V_i is count of words appearing exactly i times, N is total tokens.

    Lower K = more diverse vocabulary.
    """
    freq = Counter(tokens)
    N = len(tokens)

    if N == 0:
        return 0

    # Count of counts
    freq_of_freq = Counter(freq.values())

    # Sum of f² * V_f
    sum_term = sum(f * f * v for f, v in freq_of_freq.items())

    K = 10000 * (sum_term - N) / (N * N)
    return K


def compute_mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).

    Counts how many times TTR drops below threshold when processing sequentially.
    """
    if len(tokens) < 10:
        return 0

    def mtld_forward(tokens, threshold):
        factors = 0
        types = set()
        start = 0

        for i, token in enumerate(tokens):
            types.add(token)
            ttr = len(types) / (i - start + 1)

            if ttr < threshold:
                factors += 1
                types = set()
                start = i + 1

        # Partial factor for remaining tokens
        if start < len(tokens):
            remaining_ttr = len(types) / (len(tokens) - start)
            factors += (1 - remaining_ttr) / (1 - threshold)

        return len(tokens) / factors if factors > 0 else len(tokens)

    # Average forward and backward
    forward = mtld_forward(tokens, threshold)
    backward = mtld_forward(tokens[::-1], threshold)

    return (forward + backward) / 2


def analyze_diversity(tokens: list[str], df: pd.DataFrame, output_dir: Path):
    """Compute lexical diversity metrics."""

    # Overall metrics
    ttr = compute_ttr(tokens)
    yules_k = compute_yules_k(tokens)
    mtld = compute_mtld(tokens)

    print(f"\nLexical Diversity Metrics:")
    print(f"  Type-Token Ratio (TTR): {ttr:.4f}")
    print(f"  Yule's K: {yules_k:.2f} (lower = more diverse)")
    print(f"  MTLD: {mtld:.2f} (higher = more diverse)")

    # Per-poem metrics
    poem_metrics = []
    for idx, row in df.iterrows():
        tokens_list = row['fugashi_tokens']
        if hasattr(tokens_list, 'tolist'):
            tokens_list = tokens_list.tolist()
        poem_tokens = [t['surface'] for t in tokens_list if isinstance(t, dict)]

        poem_metrics.append({
            'poem_id': row['poem_id'],
            'collection': row.get('collection', 'unknown'),
            'token_count': len(poem_tokens),
            'type_count': len(set(poem_tokens)),
            'ttr': compute_ttr(poem_tokens),
        })

    metrics_df = pd.DataFrame(poem_metrics)

    # By collection
    print(f"\nBy Collection:")
    for collection in metrics_df['collection'].unique():
        coll_df = metrics_df[metrics_df['collection'] == collection]
        print(f"  {collection}: avg TTR = {coll_df['ttr'].mean():.4f}, "
              f"avg tokens = {coll_df['token_count'].mean():.1f}")

    # Save metrics
    metrics_df.to_csv(output_dir / "poem_diversity_metrics.csv", index=False)

    # Plot TTR distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(metrics_df['ttr'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(ttr, color='red', linestyle='--', label=f'Corpus TTR: {ttr:.3f}')
    axes[0].set_xlabel('Type-Token Ratio')
    axes[0].set_ylabel('Poem Count')
    axes[0].set_title('TTR Distribution per Poem')
    axes[0].legend()

    # By collection boxplot
    collections = metrics_df['collection'].unique()
    data = [metrics_df[metrics_df['collection'] == c]['ttr'].values for c in collections]
    axes[1].boxplot(data, labels=[c[:10] for c in collections])
    axes[1].set_ylabel('Type-Token Ratio')
    axes[1].set_title('TTR by Collection')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "lexical_diversity.png", dpi=150)
    plt.close()

    return {
        "corpus_ttr": ttr,
        "yules_k": yules_k,
        "mtld": mtld
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")

    print("\nExtracting tokens...")
    tokens = extract_all_tokens(df)
    print(f"  Extracted {len(tokens)} tokens")

    print("\n" + "="*50)
    zipf_results = analyze_zipf(tokens, OUTPUT_DIR)

    print("\n" + "="*50)
    diversity_results = analyze_diversity(tokens, df, OUTPUT_DIR)

    # Save summary
    summary = {**zipf_results, **diversity_results}
    with open(OUTPUT_DIR / "lexical_summary.md", "w", encoding="utf-8") as f:
        f.write("# Lexical Analysis Summary\n\n")
        f.write("## Zipf's Law\n")
        f.write(f"- Slope: {summary['slope']:.3f} (ideal: -1.0)\n")
        f.write(f"- R²: {summary['r_squared']:.4f}\n")
        f.write(f"- Unique tokens: {summary['unique']:,}\n")
        f.write(f"- Total tokens: {summary['total']:,}\n\n")
        f.write("## Lexical Diversity\n")
        f.write(f"- Type-Token Ratio: {summary['corpus_ttr']:.4f}\n")
        f.write(f"- Yule's K: {summary['yules_k']:.2f}\n")
        f.write(f"- MTLD: {summary['mtld']:.2f}\n")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
