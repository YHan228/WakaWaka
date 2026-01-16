#!/usr/bin/env python3
"""
analyze_collections.py - Collection comparison analysis.

Analyses:
- Vocabulary overlap (Jaccard similarity matrix)
- Distinctive words per collection (TF-IDF)
- Statistical style fingerprints

Output: data/analysis/collections/
"""

import sys
from pathlib import Path
from collections import Counter
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_utils import setup_plotting

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "collections"


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def get_collection_vocabularies(df: pd.DataFrame) -> dict[str, set[str]]:
    """Extract vocabulary set per collection."""
    collections = {}

    for collection in df['collection'].unique():
        if pd.isna(collection):
            continue
        coll_df = df[df['collection'] == collection]

        vocab = set()
        for tokens in coll_df['fugashi_tokens']:
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            for t in tokens:
                if isinstance(t, dict):
                    vocab.add(t['surface'])

        collections[collection] = vocab

    return collections


def analyze_jaccard(vocabularies: dict[str, set[str]], output_dir: Path):
    """Compute Jaccard similarity matrix between collections."""
    collections = list(vocabularies.keys())
    n = len(collections)

    # Compute Jaccard matrix
    jaccard_matrix = np.zeros((n, n))

    for i, c1 in enumerate(collections):
        for j, c2 in enumerate(collections):
            v1, v2 = vocabularies[c1], vocabularies[c2]
            intersection = len(v1 & v2)
            union = len(v1 | v2)
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0

    print("Jaccard Similarity Matrix:")
    print(f"  Collections: {collections}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(jaccard_matrix, cmap='YlOrRd', aspect='auto')

    # Labels
    short_labels = [c[:15] + '...' if len(c) > 15 else c for c in collections]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)

    # Add values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{jaccard_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8)

    ax.set_title('Vocabulary Overlap (Jaccard Similarity)')
    plt.colorbar(im, label='Jaccard Index')
    plt.tight_layout()
    plt.savefig(output_dir / "jaccard_heatmap.png", dpi=150)
    plt.close()

    # Save matrix as CSV
    jaccard_df = pd.DataFrame(jaccard_matrix, index=collections, columns=collections)
    jaccard_df.to_csv(output_dir / "jaccard_matrix.csv")

    return jaccard_df


def analyze_tfidf(df: pd.DataFrame, output_dir: Path):
    """Find distinctive words per collection using TF-IDF."""

    # Build document per collection (all tokens concatenated)
    collection_docs = {}
    for collection in df['collection'].unique():
        if pd.isna(collection):
            continue
        coll_df = df[df['collection'] == collection]

        tokens = []
        for token_list in coll_df['fugashi_tokens']:
            if hasattr(token_list, 'tolist'):
                token_list = token_list.tolist()
            for t in token_list:
                if isinstance(t, dict):
                    tokens.append(t['surface'])

        collection_docs[collection] = ' '.join(tokens)

    collections = list(collection_docs.keys())
    documents = [collection_docs[c] for c in collections]

    # TF-IDF
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'[^\s]+')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Get top distinctive words per collection
    print("\nTop Distinctive Words per Collection (TF-IDF):")

    results = {}
    for i, collection in enumerate(collections):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-20:][::-1]

        top_words = [(feature_names[j], scores[j]) for j in top_indices if scores[j] > 0]
        results[collection] = top_words

        print(f"\n  {collection}:")
        for word, score in top_words[:10]:
            print(f"    {word}: {score:.4f}")

    # Save to CSV
    with open(output_dir / "distinctive_words_tfidf.csv", "w", encoding="utf-8") as f:
        f.write("collection,rank,word,tfidf_score\n")
        for collection, words in results.items():
            for rank, (word, score) in enumerate(words, 1):
                f.write(f"{collection},{rank},{word},{score:.6f}\n")

    return results


def analyze_style_fingerprints(df: pd.DataFrame, output_dir: Path):
    """Compute statistical style fingerprints per collection."""

    fingerprints = {}

    for collection in df['collection'].unique():
        if pd.isna(collection):
            continue
        coll_df = df[df['collection'] == collection]

        # Metrics
        token_counts = []
        char_counts = []
        vocab_counts = []
        grammar_counts = []
        hiragana_ratios = []
        difficulty_scores = []

        for _, row in coll_df.iterrows():
            text = row['text']
            tokens = row['fugashi_tokens']
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()

            token_counts.append(len(tokens))
            char_counts.append(len(text))
            vocab_counts.append(len(row.get('vocabulary', [])))
            grammar_counts.append(len(row.get('grammar_points', [])))
            difficulty_scores.append(row.get('difficulty_score', 0) or 0)

            # Hiragana ratio
            hiragana = sum(1 for c in text if '\u3040' <= c <= '\u309f')
            hiragana_ratios.append(hiragana / len(text) if text else 0)

        fingerprints[collection] = {
            'poem_count': len(coll_df),
            'avg_tokens': np.mean(token_counts),
            'std_tokens': np.std(token_counts),
            'avg_chars': np.mean(char_counts),
            'avg_vocab': np.mean(vocab_counts),
            'avg_grammar': np.mean(grammar_counts),
            'avg_difficulty': np.mean(difficulty_scores),
            'avg_hiragana_ratio': np.mean(hiragana_ratios),
        }

    print("\nStyle Fingerprints:")
    for collection, fp in fingerprints.items():
        print(f"\n  {collection} ({fp['poem_count']} poems):")
        print(f"    Avg tokens: {fp['avg_tokens']:.1f} ± {fp['std_tokens']:.1f}")
        print(f"    Avg chars: {fp['avg_chars']:.1f}")
        print(f"    Hiragana ratio: {fp['avg_hiragana_ratio']:.2%}")
        print(f"    Avg difficulty: {fp['avg_difficulty']:.3f}")

    # Save fingerprints
    fp_df = pd.DataFrame(fingerprints).T
    fp_df.to_csv(output_dir / "style_fingerprints.csv")

    # Radar chart for comparison
    collections = list(fingerprints.keys())[:6]  # Limit for readability
    if len(collections) >= 2:
        metrics = ['avg_tokens', 'avg_vocab', 'avg_grammar', 'avg_hiragana_ratio', 'avg_difficulty']
        metric_labels = ['Tokens', 'Vocabulary', 'Grammar', 'Hiragana%', 'Difficulty']

        # Normalize each metric to 0-1
        normalized = {}
        for m in metrics:
            values = [fingerprints[c][m] for c in collections]
            min_v, max_v = min(values), max(values)
            range_v = max_v - min_v if max_v > min_v else 1
            normalized[m] = [(fingerprints[c][m] - min_v) / range_v for c in collections]

        # Plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = plt.cm.Set2(np.linspace(0, 1, len(collections)))
        for i, collection in enumerate(collections):
            values = [normalized[m][i] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=collection[:20], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_title('Collection Style Fingerprints (Normalized)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(output_dir / "style_radar.png", dpi=150)
        plt.close()

    return fingerprints


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")
    print(f"  Collections: {df['collection'].value_counts().to_dict()}")

    print("\n" + "="*50)
    print("Analyzing vocabulary overlap...")
    vocabularies = get_collection_vocabularies(df)
    jaccard_df = analyze_jaccard(vocabularies, OUTPUT_DIR)

    print("\n" + "="*50)
    print("Analyzing distinctive words...")
    tfidf_results = analyze_tfidf(df, OUTPUT_DIR)

    print("\n" + "="*50)
    print("Computing style fingerprints...")
    fingerprints = analyze_style_fingerprints(df, OUTPUT_DIR)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
