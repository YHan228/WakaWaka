#!/usr/bin/env python3
"""
analyze_pos.py - Part-of-Speech analysis.

Analyses:
- POS tag distribution
- POS bigram transitions (what follows what?)
- POS patterns in poems

Output: data/analysis/pos/
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_utils import setup_plotting

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "pos"


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def extract_pos_sequences(df: pd.DataFrame) -> list[list[str]]:
    """Extract POS sequences from all poems."""
    sequences = []

    for tokens in df['fugashi_tokens']:
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()

        pos_seq = []
        for t in tokens:
            if isinstance(t, dict):
                # Get main POS (first part before comma)
                pos = t.get('pos', 'UNK')
                if ',' in pos:
                    pos = pos.split(',')[0]
                pos_seq.append(pos)

        if pos_seq:
            sequences.append(pos_seq)

    return sequences


def analyze_pos_distribution(sequences: list[list[str]], output_dir: Path):
    """Analyze POS tag distribution."""

    print("Analyzing POS distribution...")

    pos_counts = Counter()
    for seq in sequences:
        pos_counts.update(seq)

    total = sum(pos_counts.values())
    print(f"\nPOS Distribution (top 15):")
    for pos, count in pos_counts.most_common(15):
        print(f"  {pos}: {count:,} ({100*count/total:.1f}%)")

    # Plot
    top_pos = pos_counts.most_common(20)
    labels = [p[0] for p in top_pos]
    counts = [p[1] for p in top_pos]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, counts, color='steelblue')
    ax.set_xlabel('Frequency')
    ax.set_title('POS Tag Distribution')
    ax.invert_yaxis()

    # Add percentage labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                f'{100*count/total:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "pos_distribution.png", dpi=150)
    plt.close()

    # Save data
    with open(output_dir / "pos_frequency.csv", "w", encoding="utf-8") as f:
        f.write("pos,count,percentage\n")
        for pos, count in pos_counts.most_common():
            f.write(f"{pos},{count},{100*count/total:.2f}\n")

    return pos_counts


def analyze_pos_bigrams(sequences: list[list[str]], output_dir: Path):
    """Analyze POS bigram transitions."""

    print("Analyzing POS bigrams...")

    bigram_counts = Counter()
    pos_set = set()

    for seq in sequences:
        pos_set.update(seq)
        for i in range(len(seq) - 1):
            bigram_counts[(seq[i], seq[i+1])] += 1

    # Get top POS tags for matrix
    pos_freq = Counter()
    for seq in sequences:
        pos_freq.update(seq)

    top_pos = [p for p, _ in pos_freq.most_common(15)]
    n = len(top_pos)
    pos_idx = {p: i for i, p in enumerate(top_pos)}

    # Build transition matrix
    trans_matrix = np.zeros((n, n))
    for (p1, p2), count in bigram_counts.items():
        if p1 in pos_idx and p2 in pos_idx:
            trans_matrix[pos_idx[p1], pos_idx[p2]] = count

    # Normalize by row (transition probabilities)
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_matrix / row_sums

    print(f"\nTop POS Bigrams:")
    for (p1, p2), count in bigram_counts.most_common(15):
        print(f"  {p1} → {p2}: {count:,}")

    # Plot transition matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw counts
    im1 = axes[0].imshow(trans_matrix, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(top_pos, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(top_pos, fontsize=8)
    axes[0].set_title('POS Bigram Counts')
    axes[0].set_xlabel('Next POS')
    axes[0].set_ylabel('Current POS')
    plt.colorbar(im1, ax=axes[0], label='Count')

    # Probabilities
    im2 = axes[1].imshow(trans_prob, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(top_pos, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(top_pos, fontsize=8)
    axes[1].set_title('POS Transition Probabilities')
    axes[1].set_xlabel('Next POS')
    axes[1].set_ylabel('Current POS')
    plt.colorbar(im2, ax=axes[1], label='Probability')

    plt.tight_layout()
    plt.savefig(output_dir / "pos_bigram_matrix.png", dpi=150)
    plt.close()

    # Save data
    with open(output_dir / "pos_bigrams.csv", "w", encoding="utf-8") as f:
        f.write("pos1,pos2,count\n")
        for (p1, p2), count in bigram_counts.most_common():
            f.write(f"{p1},{p2},{count}\n")

    # Save transition probabilities
    trans_df = pd.DataFrame(trans_prob, index=top_pos, columns=top_pos)
    trans_df.to_csv(output_dir / "pos_transition_prob.csv")

    return bigram_counts, trans_prob


def analyze_pos_trigrams(sequences: list[list[str]], output_dir: Path):
    """Analyze POS trigram patterns."""

    print("Analyzing POS trigrams...")

    trigram_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 2):
            trigram_counts[(seq[i], seq[i+1], seq[i+2])] += 1

    print(f"\nTop POS Trigrams:")
    for (p1, p2, p3), count in trigram_counts.most_common(15):
        print(f"  {p1} → {p2} → {p3}: {count:,}")

    # Save data
    with open(output_dir / "pos_trigrams.csv", "w", encoding="utf-8") as f:
        f.write("pos1,pos2,pos3,count\n")
        for (p1, p2, p3), count in trigram_counts.most_common(100):
            f.write(f"{p1},{p2},{p3},{count}\n")

    return trigram_counts


def analyze_pos_patterns_by_collection(df: pd.DataFrame, sequences: list[list[str]], output_dir: Path):
    """Compare POS patterns across collections."""

    print("Analyzing POS patterns by collection...")

    collection_pos = defaultdict(Counter)

    for (_, row), seq in zip(df.iterrows(), sequences):
        collection = row.get('collection', 'unknown')
        if pd.isna(collection):
            collection = 'unknown'
        collection_pos[collection].update(seq)

    # Get top POS for comparison
    all_pos = Counter()
    for pos_counts in collection_pos.values():
        all_pos.update(pos_counts)
    top_pos = [p for p, _ in all_pos.most_common(12)]

    # Build comparison matrix
    collections = list(collection_pos.keys())
    comparison = np.zeros((len(collections), len(top_pos)))

    for i, coll in enumerate(collections):
        total = sum(collection_pos[coll].values())
        for j, pos in enumerate(top_pos):
            comparison[i, j] = collection_pos[coll].get(pos, 0) / total if total > 0 else 0

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_pos))
    width = 0.8 / len(collections)

    colors = plt.cm.Set2(np.linspace(0, 1, len(collections)))
    for i, (coll, color) in enumerate(zip(collections, colors)):
        offset = (i - len(collections)/2 + 0.5) * width
        bars = ax.bar(x + offset, comparison[i], width, label=coll[:20], color=color)

    ax.set_xlabel('POS Tag')
    ax.set_ylabel('Proportion')
    ax.set_title('POS Distribution by Collection')
    ax.set_xticks(x)
    ax.set_xticklabels(top_pos, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / "pos_by_collection.png", dpi=150)
    plt.close()

    return collection_pos


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")

    print("\nExtracting POS sequences...")
    sequences = extract_pos_sequences(df)
    print(f"  Extracted {len(sequences)} sequences")
    print(f"  Total tokens: {sum(len(s) for s in sequences):,}")

    print("\n" + "="*50)
    pos_counts = analyze_pos_distribution(sequences, OUTPUT_DIR)

    print("\n" + "="*50)
    bigrams, trans_prob = analyze_pos_bigrams(sequences, OUTPUT_DIR)

    print("\n" + "="*50)
    trigrams = analyze_pos_trigrams(sequences, OUTPUT_DIR)

    print("\n" + "="*50)
    collection_pos = analyze_pos_patterns_by_collection(df, sequences, OUTPUT_DIR)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
