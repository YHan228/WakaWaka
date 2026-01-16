#!/usr/bin/env python3
"""
analyze_phonetics.py - Phonetic and sound pattern analysis.

Analyses:
- Vowel harmony patterns
- Sound/mora frequency distribution
- Hiragana character heatmap
- Alliteration and repetition patterns

Output: data/analysis/phonetics/
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_utils import setup_plotting

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "phonetics"

# Hiragana to vowel mapping
HIRAGANA_TO_VOWEL = {
    'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
    'か': 'a', 'き': 'i', 'く': 'u', 'け': 'e', 'こ': 'o',
    'さ': 'a', 'し': 'i', 'す': 'u', 'せ': 'e', 'そ': 'o',
    'た': 'a', 'ち': 'i', 'つ': 'u', 'て': 'e', 'と': 'o',
    'な': 'a', 'に': 'i', 'ぬ': 'u', 'ね': 'e', 'の': 'o',
    'は': 'a', 'ひ': 'i', 'ふ': 'u', 'へ': 'e', 'ほ': 'o',
    'ま': 'a', 'み': 'i', 'む': 'u', 'め': 'e', 'も': 'o',
    'や': 'a', 'ゆ': 'u', 'よ': 'o',
    'ら': 'a', 'り': 'i', 'る': 'u', 'れ': 'e', 'ろ': 'o',
    'わ': 'a', 'ゐ': 'i', 'ゑ': 'e', 'を': 'o',
    'ん': 'n',
    'が': 'a', 'ぎ': 'i', 'ぐ': 'u', 'げ': 'e', 'ご': 'o',
    'ざ': 'a', 'じ': 'i', 'ず': 'u', 'ぜ': 'e', 'ぞ': 'o',
    'だ': 'a', 'ぢ': 'i', 'づ': 'u', 'で': 'e', 'ど': 'o',
    'ば': 'a', 'び': 'i', 'ぶ': 'u', 'べ': 'e', 'ぼ': 'o',
    'ぱ': 'a', 'ぴ': 'i', 'ぷ': 'u', 'ぺ': 'e', 'ぽ': 'o',
}

# Hiragana to consonant mapping
HIRAGANA_TO_CONSONANT = {
    'あ': '', 'い': '', 'う': '', 'え': '', 'お': '',
    'か': 'k', 'き': 'k', 'く': 'k', 'け': 'k', 'こ': 'k',
    'さ': 's', 'し': 's', 'す': 's', 'せ': 's', 'そ': 's',
    'た': 't', 'ち': 't', 'つ': 't', 'て': 't', 'と': 't',
    'な': 'n', 'に': 'n', 'ぬ': 'n', 'ね': 'n', 'の': 'n',
    'は': 'h', 'ひ': 'h', 'ふ': 'h', 'へ': 'h', 'ほ': 'h',
    'ま': 'm', 'み': 'm', 'む': 'm', 'め': 'm', 'も': 'm',
    'や': 'y', 'ゆ': 'y', 'よ': 'y',
    'ら': 'r', 'り': 'r', 'る': 'r', 'れ': 'r', 'ろ': 'r',
    'わ': 'w', 'ゐ': 'w', 'ゑ': 'w', 'を': 'w',
    'ん': 'N',
    'が': 'g', 'ぎ': 'g', 'ぐ': 'g', 'げ': 'g', 'ご': 'g',
    'ざ': 'z', 'じ': 'z', 'ず': 'z', 'ぜ': 'z', 'ぞ': 'z',
    'だ': 'd', 'ぢ': 'd', 'づ': 'd', 'で': 'd', 'ど': 'd',
    'ば': 'b', 'び': 'b', 'ぶ': 'b', 'べ': 'b', 'ぼ': 'b',
    'ぱ': 'p', 'ぴ': 'p', 'ぷ': 'p', 'ぺ': 'p', 'ぽ': 'p',
}


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def text_to_hiragana(text: str) -> str:
    """Extract only hiragana from text."""
    return ''.join(c for c in text if '\u3040' <= c <= '\u309f')


def get_vowel_sequence(hiragana: str) -> str:
    """Convert hiragana to vowel sequence."""
    return ''.join(HIRAGANA_TO_VOWEL.get(c, '') for c in hiragana)


def get_consonant_sequence(hiragana: str) -> str:
    """Convert hiragana to consonant sequence."""
    return ''.join(HIRAGANA_TO_CONSONANT.get(c, '') for c in hiragana)


def analyze_vowel_harmony(df: pd.DataFrame, output_dir: Path):
    """Analyze vowel patterns and harmony."""

    print("Analyzing vowel patterns...")

    vowel_counts = Counter()
    vowel_bigrams = Counter()
    vowel_trigrams = Counter()

    for text in df['reading_hiragana']:
        if not text:
            continue
        hiragana = text_to_hiragana(str(text))
        vowels = get_vowel_sequence(hiragana)

        for v in vowels:
            vowel_counts[v] += 1

        for i in range(len(vowels) - 1):
            vowel_bigrams[vowels[i:i+2]] += 1

        for i in range(len(vowels) - 2):
            vowel_trigrams[vowels[i:i+3]] += 1

    # Vowel frequency
    print("\nVowel Frequencies:")
    total = sum(vowel_counts.values())
    for v in 'aiueo':
        count = vowel_counts.get(v, 0)
        print(f"  {v}: {count:,} ({100*count/total:.1f}%)")

    # Plot vowel distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Unigrams
    vowels = list('aiueo')
    counts = [vowel_counts.get(v, 0) for v in vowels]
    axes[0].bar(vowels, counts, color='steelblue')
    axes[0].set_title('Vowel Frequency')
    axes[0].set_xlabel('Vowel')
    axes[0].set_ylabel('Count')

    # Bigram heatmap
    bigram_matrix = np.zeros((5, 5))
    for i, v1 in enumerate(vowels):
        for j, v2 in enumerate(vowels):
            bigram_matrix[i, j] = vowel_bigrams.get(v1+v2, 0)

    im = axes[1].imshow(bigram_matrix, cmap='YlOrRd')
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(vowels)
    axes[1].set_yticklabels(vowels)
    axes[1].set_title('Vowel Bigram Frequency')
    axes[1].set_xlabel('Second Vowel')
    axes[1].set_ylabel('First Vowel')
    plt.colorbar(im, ax=axes[1])

    # Top trigrams
    top_trigrams = vowel_trigrams.most_common(15)
    axes[2].barh([t[0] for t in top_trigrams], [t[1] for t in top_trigrams], color='coral')
    axes[2].set_title('Top Vowel Trigrams')
    axes[2].set_xlabel('Count')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "vowel_patterns.png", dpi=150)
    plt.close()

    # Save data
    with open(output_dir / "vowel_bigrams.csv", "w") as f:
        f.write("bigram,count\n")
        for bg, count in vowel_bigrams.most_common():
            f.write(f"{bg},{count}\n")

    return vowel_counts, vowel_bigrams


def analyze_hiragana_frequency(df: pd.DataFrame, output_dir: Path):
    """Analyze hiragana character frequency."""

    print("Analyzing hiragana frequency...")

    char_counts = Counter()
    for text in df['reading_hiragana']:
        if not text:
            continue
        hiragana = text_to_hiragana(str(text))
        char_counts.update(hiragana)

    # Create heatmap in gojuon order
    gojuon_rows = ['あ', 'か', 'さ', 'た', 'な', 'は', 'ま', 'や', 'ら', 'わ']
    gojuon_cols = ['あ段', 'い段', 'う段', 'え段', 'お段']

    gojuon_grid = [
        ['あ', 'い', 'う', 'え', 'お'],
        ['か', 'き', 'く', 'け', 'こ'],
        ['さ', 'し', 'す', 'せ', 'そ'],
        ['た', 'ち', 'つ', 'て', 'と'],
        ['な', 'に', 'ぬ', 'ね', 'の'],
        ['は', 'ひ', 'ふ', 'へ', 'ほ'],
        ['ま', 'み', 'む', 'め', 'も'],
        ['や', '', 'ゆ', '', 'よ'],
        ['ら', 'り', 'る', 'れ', 'ろ'],
        ['わ', 'ゐ', '', 'ゑ', 'を'],
    ]

    heatmap_data = np.zeros((10, 5))
    for i, row in enumerate(gojuon_grid):
        for j, char in enumerate(row):
            if char:
                heatmap_data[i, j] = char_counts.get(char, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # Add character labels
    for i, row in enumerate(gojuon_grid):
        for j, char in enumerate(row):
            if char:
                count = char_counts.get(char, 0)
                ax.text(j, i, f'{char}\n{count}', ha='center', va='center', fontsize=10)

    ax.set_xticks(range(5))
    ax.set_yticks(range(10))
    ax.set_xticklabels(['a', 'i', 'u', 'e', 'o'])
    ax.set_yticklabels(['∅', 'k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w'])
    ax.set_title('Hiragana Frequency Heatmap (Gojuon Order)')
    ax.set_xlabel('Vowel')
    ax.set_ylabel('Consonant')

    plt.colorbar(im, label='Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / "hiragana_heatmap.png", dpi=150)
    plt.close()

    # Save top characters
    with open(output_dir / "hiragana_frequency.csv", "w", encoding="utf-8") as f:
        f.write("character,count\n")
        for char, count in char_counts.most_common():
            f.write(f"{char},{count}\n")

    print(f"  Total characters: {sum(char_counts.values()):,}")
    print(f"  Unique characters: {len(char_counts)}")
    print(f"  Top 5: {char_counts.most_common(5)}")

    return char_counts


def analyze_sound_patterns(df: pd.DataFrame, output_dir: Path):
    """Analyze alliteration and sound repetition patterns."""

    print("Analyzing sound patterns...")

    # Analyze consonant patterns (alliteration)
    alliteration_counts = Counter()
    repetition_poems = []

    for idx, row in df.iterrows():
        text = row.get('reading_hiragana', '')
        if not text:
            continue

        hiragana = text_to_hiragana(str(text))
        consonants = get_consonant_sequence(hiragana)

        # Find repeated consonant sequences (alliteration)
        for length in [2, 3]:
            for i in range(len(consonants) - length + 1):
                seq = consonants[i:i+length]
                if seq and len(set(seq)) == 1 and seq[0]:  # All same consonant
                    alliteration_counts[seq[0] * length] += 1

        # Find repeated hiragana sequences
        for length in [2, 3]:
            for i in range(len(hiragana) - length * 2 + 1):
                seq = hiragana[i:i+length]
                if seq in hiragana[i+length:]:
                    repetition_poems.append({
                        'poem_id': row['poem_id'],
                        'pattern': seq,
                        'text': row['text'][:30]
                    })

    print(f"\nAlliteration patterns (repeated consonants):")
    for pattern, count in alliteration_counts.most_common(10):
        print(f"  {pattern}: {count}")

    # Consonant transition analysis
    consonant_transitions = Counter()
    for text in df['reading_hiragana']:
        if not text:
            continue
        hiragana = text_to_hiragana(str(text))
        consonants = get_consonant_sequence(hiragana)

        for i in range(len(consonants) - 1):
            if consonants[i] and consonants[i+1]:
                consonant_transitions[consonants[i] + consonants[i+1]] += 1

    # Plot consonant transitions
    consonants = ['k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w', 'g', 'z', 'd', 'b']
    trans_matrix = np.zeros((len(consonants), len(consonants)))
    for i, c1 in enumerate(consonants):
        for j, c2 in enumerate(consonants):
            trans_matrix[i, j] = consonant_transitions.get(c1+c2, 0)

    # Normalize by row
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_matrix_norm = trans_matrix / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(trans_matrix_norm, cmap='YlOrRd')
    ax.set_xticks(range(len(consonants)))
    ax.set_yticks(range(len(consonants)))
    ax.set_xticklabels(consonants)
    ax.set_yticklabels(consonants)
    ax.set_title('Consonant Transition Probabilities')
    ax.set_xlabel('Next Consonant')
    ax.set_ylabel('Current Consonant')
    plt.colorbar(im, label='Probability')
    plt.tight_layout()
    plt.savefig(output_dir / "consonant_transitions.png", dpi=150)
    plt.close()

    # Save data
    with open(output_dir / "sound_patterns.csv", "w", encoding="utf-8") as f:
        f.write("pattern,count,type\n")
        for pattern, count in alliteration_counts.most_common():
            f.write(f"{pattern},{count},alliteration\n")

    pd.DataFrame(repetition_poems[:100]).to_csv(
        output_dir / "repetition_examples.csv", index=False
    )

    return alliteration_counts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")

    print("\n" + "="*50)
    vowel_results = analyze_vowel_harmony(df, OUTPUT_DIR)

    print("\n" + "="*50)
    hiragana_freq = analyze_hiragana_frequency(df, OUTPUT_DIR)

    print("\n" + "="*50)
    sound_patterns = analyze_sound_patterns(df, OUTPUT_DIR)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
