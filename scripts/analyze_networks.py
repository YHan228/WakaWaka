#!/usr/bin/env python3
"""
analyze_networks.py - Network and clustering analysis.

Analyses:
- Word co-occurrence graph (top N nodes)
- Grammar point co-occurrence network
- Grammar point clustering dendrogram

Output: data/analysis/networks/
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_utils import setup_plotting, get_font

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "networks"

# Try to import networkx for graph visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Some visualizations will be skipped.")


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def build_word_cooccurrence(df: pd.DataFrame, min_freq: int = 10, top_n: int = 100):
    """Build word co-occurrence matrix from poems."""

    # Count word frequencies
    word_freq = Counter()
    for tokens in df['fugashi_tokens']:
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        for t in tokens:
            if isinstance(t, dict):
                word_freq[t['surface']] += 1

    # Filter to top N words
    top_words = [w for w, c in word_freq.most_common(top_n) if c >= min_freq]
    word_set = set(top_words)

    # Count co-occurrences (within same poem)
    cooccur = defaultdict(int)
    for tokens in df['fugashi_tokens']:
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()

        poem_words = set()
        for t in tokens:
            if isinstance(t, dict) and t['surface'] in word_set:
                poem_words.add(t['surface'])

        for w1, w2 in combinations(sorted(poem_words), 2):
            cooccur[(w1, w2)] += 1

    return top_words, word_freq, cooccur


def analyze_word_cooccurrence(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize word co-occurrence."""

    print("Building word co-occurrence matrix...")
    top_words, word_freq, cooccur = build_word_cooccurrence(df, min_freq=5, top_n=80)

    print(f"  Top words: {len(top_words)}")
    print(f"  Co-occurrence pairs: {len(cooccur)}")

    # Save co-occurrence data
    with open(output_dir / "word_cooccurrence.csv", "w", encoding="utf-8") as f:
        f.write("word1,word2,count\n")
        for (w1, w2), count in sorted(cooccur.items(), key=lambda x: -x[1])[:500]:
            f.write(f"{w1},{w2},{count}\n")

    if not HAS_NETWORKX:
        print("  Skipping graph visualization (networkx not installed)")
        return

    # Build graph
    G = nx.Graph()

    # Add nodes with frequency as attribute
    for word in top_words[:50]:  # Limit for visualization
        G.add_node(word, freq=word_freq[word])

    # Add edges (filter to stronger connections)
    min_cooccur = 3
    for (w1, w2), count in cooccur.items():
        if w1 in G.nodes and w2 in G.nodes and count >= min_cooccur:
            G.add_edge(w1, w2, weight=count)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Visualize
    plt.figure(figsize=(16, 16))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node sizes based on frequency
    node_sizes = [G.nodes[n]['freq'] * 3 for n in G.nodes]

    # Edge widths based on co-occurrence
    edge_widths = [G[u][v]['weight'] * 0.3 for u, v in G.edges]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                          alpha=0.7, edgecolors='navy')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=9, font_family=get_font())

    plt.title(f'Word Co-occurrence Network (Top {G.number_of_nodes()} words)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "word_cooccurrence_graph.png", dpi=150)
    plt.close()

    return G


def build_grammar_cooccurrence(df: pd.DataFrame):
    """Build grammar point co-occurrence matrix."""

    # Count grammar frequencies
    grammar_freq = Counter()
    for gps in df['grammar_points']:
        if hasattr(gps, 'tolist'):
            gps = gps.tolist()
        for gp in gps:
            if isinstance(gp, dict):
                grammar_freq[gp['canonical_id']] += 1

    # Filter to common grammar points
    min_freq = 5
    common_grammar = {g for g, c in grammar_freq.items() if c >= min_freq}

    # Count co-occurrences
    cooccur = defaultdict(int)
    for gps in df['grammar_points']:
        if hasattr(gps, 'tolist'):
            gps = gps.tolist()

        poem_grammar = set()
        for gp in gps:
            if isinstance(gp, dict) and gp['canonical_id'] in common_grammar:
                poem_grammar.add(gp['canonical_id'])

        for g1, g2 in combinations(sorted(poem_grammar), 2):
            cooccur[(g1, g2)] += 1

    return list(common_grammar), grammar_freq, cooccur


def analyze_grammar_cooccurrence(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize grammar point co-occurrence."""

    print("Building grammar co-occurrence matrix...")
    grammar_list, grammar_freq, cooccur = build_grammar_cooccurrence(df)

    print(f"  Grammar points: {len(grammar_list)}")
    print(f"  Co-occurrence pairs: {len(cooccur)}")

    # Save co-occurrence data
    with open(output_dir / "grammar_cooccurrence.csv", "w", encoding="utf-8") as f:
        f.write("grammar1,grammar2,count\n")
        for (g1, g2), count in sorted(cooccur.items(), key=lambda x: -x[1])[:500]:
            f.write(f"{g1},{g2},{count}\n")

    if not HAS_NETWORKX:
        print("  Skipping graph visualization (networkx not installed)")
        return

    # Build graph
    G = nx.Graph()

    # Top grammar points for visualization
    top_grammar = [g for g, _ in grammar_freq.most_common(60)]

    for grammar in top_grammar:
        G.add_node(grammar, freq=grammar_freq[grammar])

    min_cooccur = 5
    for (g1, g2), count in cooccur.items():
        if g1 in G.nodes and g2 in G.nodes and count >= min_cooccur:
            G.add_edge(g1, g2, weight=count)

    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Visualize
    plt.figure(figsize=(16, 16))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_sizes = [G.nodes[n]['freq'] * 2 for n in G.nodes]
    edge_widths = [G[u][v]['weight'] * 0.2 for u, v in G.edges]

    # Color by category
    colors = []
    for n in G.nodes:
        if n.startswith('particle_'):
            colors.append('lightblue')
        elif n.startswith('auxiliary_'):
            colors.append('lightgreen')
        elif n.startswith('conjugation_'):
            colors.append('lightyellow')
        else:
            colors.append('lightpink')

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors,
                          alpha=0.7, edgecolors='navy')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=7, font_family=get_font())

    plt.title(f'Grammar Point Co-occurrence Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "grammar_cooccurrence_graph.png", dpi=150)
    plt.close()

    return G


def analyze_grammar_clustering(df: pd.DataFrame, output_dir: Path):
    """Cluster grammar points based on co-occurrence patterns."""

    print("Building grammar clustering dendrogram...")

    # Build co-occurrence matrix
    grammar_list, grammar_freq, cooccur = build_grammar_cooccurrence(df)

    # Filter to top grammar points
    top_grammar = [g for g, _ in grammar_freq.most_common(50)]
    n = len(top_grammar)
    grammar_idx = {g: i for i, g in enumerate(top_grammar)}

    # Build co-occurrence matrix
    matrix = np.zeros((n, n))
    for (g1, g2), count in cooccur.items():
        if g1 in grammar_idx and g2 in grammar_idx:
            i, j = grammar_idx[g1], grammar_idx[g2]
            matrix[i, j] = count
            matrix[j, i] = count

    # Normalize by frequency (Jaccard-like)
    for i in range(n):
        for j in range(n):
            if i != j:
                freq_sum = grammar_freq[top_grammar[i]] + grammar_freq[top_grammar[j]]
                if freq_sum > 0:
                    matrix[i, j] = matrix[i, j] / freq_sum

    # Fill diagonal
    np.fill_diagonal(matrix, 1.0)

    # Convert similarity to distance
    distance_matrix = 1 - matrix
    np.fill_diagonal(distance_matrix, 0)

    # Ensure valid distance matrix
    distance_matrix = np.clip(distance_matrix, 0, 1)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Hierarchical clustering
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(14, 10))
    dendrogram(
        linkage_matrix,
        labels=top_grammar,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    plt.title('Grammar Point Clustering (by Co-occurrence)')
    plt.xlabel('Grammar Point')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_dir / "grammar_dendrogram.png", dpi=150)
    plt.close()

    print(f"  Clustered {n} grammar points")

    return linkage_matrix


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")

    print("\n" + "="*50)
    word_graph = analyze_word_cooccurrence(df, OUTPUT_DIR)

    print("\n" + "="*50)
    grammar_graph = analyze_grammar_cooccurrence(df, OUTPUT_DIR)

    print("\n" + "="*50)
    clustering = analyze_grammar_clustering(df, OUTPUT_DIR)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
