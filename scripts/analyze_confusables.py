#!/usr/bin/env python3
"""
analyze_confusables.py - Analyze grammar point homographs (confusables).

Finds grammar points with the same surface form but different meanings/functions.
Useful for generating quiz distractors and identifying common learner pitfalls.

Example confusables:
- ぬ: negative auxiliary (ず未然形) vs perfective auxiliary (完了)
- に: location particle vs copula (なり連用形)
- な: negative imperative vs exclamatory particle

Output: data/analysis/confusables.json
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"


def load_corpus():
    """Load annotated poems."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "annotated" / "poems.parquet")
    return df


def analyze_confusables(df: pd.DataFrame) -> dict:
    """
    Find grammar points with same surface but different canonical_id or sense_id.

    Groups all grammar points by surface form, then identifies surfaces that have
    multiple distinct functions/meanings.
    """
    # Group grammar points by surface
    surface_groups = defaultdict(list)

    for idx, row in df.iterrows():
        poem_id = row['poem_id']
        grammar_points = row.get('grammar_points', [])
        if hasattr(grammar_points, 'tolist'):
            grammar_points = grammar_points.tolist()

        for gp in grammar_points:
            if not isinstance(gp, dict):
                continue

            surface = gp.get('surface', '')
            if not surface:
                continue

            canonical_id = gp.get('canonical_id', '')
            sense_id = gp.get('sense_id', '')
            category = gp.get('category', '')
            description = gp.get('description', '')

            surface_groups[surface].append({
                'canonical_id': canonical_id,
                'sense_id': sense_id,
                'category': category,
                'description': description,
                'poem_id': poem_id,
            })

    # Find surfaces with multiple distinct meanings
    confusables = {}
    multi_meaning_count = 0

    for surface, occurrences in surface_groups.items():
        # Get unique (canonical_id, sense_id) pairs
        unique_meanings = {}
        for occ in occurrences:
            key = (occ['canonical_id'], occ['sense_id'] or '')
            if key not in unique_meanings:
                unique_meanings[key] = {
                    'canonical_id': occ['canonical_id'],
                    'sense_id': occ['sense_id'],
                    'category': occ['category'],
                    'description': occ['description'],
                    'frequency': 0,
                    'example_poems': [],
                }
            unique_meanings[key]['frequency'] += 1
            if len(unique_meanings[key]['example_poems']) < 3:
                unique_meanings[key]['example_poems'].append(occ['poem_id'])

        # Only include if there are multiple distinct meanings
        if len(unique_meanings) > 1:
            multi_meaning_count += 1
            meanings = list(unique_meanings.values())
            meanings.sort(key=lambda x: x['frequency'], reverse=True)

            confusables[surface] = {
                'surface': surface,
                'meaning_count': len(meanings),
                'total_occurrences': len(occurrences),
                'meanings': meanings,
            }

    return confusables


def categorize_confusables(confusables: dict) -> dict:
    """
    Categorize confusables by type of confusion.

    Categories:
    - particle_vs_auxiliary: Different POS entirely
    - same_category_different_sense: Same category, different function
    - conjugation_form_ambiguity: Different conjugation readings
    """
    categories = {
        'particle_vs_auxiliary': [],
        'same_category_different_sense': [],
        'other': [],
    }

    for surface, data in confusables.items():
        meanings = data['meanings']
        unique_categories = set(m['category'] for m in meanings)

        if len(unique_categories) > 1:
            # Different POS categories
            if 'particle' in unique_categories and 'auxiliary' in unique_categories:
                categories['particle_vs_auxiliary'].append(surface)
            else:
                categories['other'].append(surface)
        else:
            # Same category, different senses
            categories['same_category_different_sense'].append(surface)

    return categories


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    df = load_corpus()
    print(f"  Loaded {len(df)} poems")

    print("\n" + "="*50)
    print("Analyzing confusables (grammar homographs)...")
    confusables = analyze_confusables(df)

    # Statistics
    print(f"\nTotal surfaces with multiple meanings: {len(confusables)}")

    # Top confusables by occurrence
    sorted_confusables = sorted(
        confusables.items(),
        key=lambda x: x[1]['total_occurrences'],
        reverse=True
    )

    print("\nTop 15 confusables by frequency:")
    for surface, data in sorted_confusables[:15]:
        meanings = data['meanings']
        print(f"\n  '{surface}' ({data['total_occurrences']} occurrences, {data['meaning_count']} meanings):")
        for m in meanings[:3]:  # Show top 3 meanings
            sense = f" [{m['sense_id']}]" if m['sense_id'] else ""
            print(f"    - {m['canonical_id']}{sense}: {m['frequency']} ({m['description'][:50]}...)")

    # Categorize
    print("\n" + "="*50)
    print("Categorizing confusables...")
    categories = categorize_confusables(confusables)

    print(f"\nParticle vs Auxiliary confusions: {len(categories['particle_vs_auxiliary'])}")
    for surface in categories['particle_vs_auxiliary'][:10]:
        print(f"  - {surface}")

    print(f"\nSame-category different-sense: {len(categories['same_category_different_sense'])}")
    for surface in categories['same_category_different_sense'][:10]:
        print(f"  - {surface}")

    # Save results
    output_data = {
        'summary': {
            'total_confusable_surfaces': len(confusables),
            'particle_vs_auxiliary_count': len(categories['particle_vs_auxiliary']),
            'same_category_count': len(categories['same_category_different_sense']),
            'other_count': len(categories['other']),
        },
        'categories': categories,
        'confusables': {k: v for k, v in sorted_confusables},  # Keep sorted order
    }

    output_file = OUTPUT_DIR / "confusables.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_file}")

    # Also save a simplified CSV for quick reference
    csv_file = OUTPUT_DIR / "confusables_summary.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("surface,meaning_count,total_occurrences,canonical_ids\n")
        for surface, data in sorted_confusables:
            cids = ';'.join(m['canonical_id'] for m in data['meanings'])
            f.write(f"{surface},{data['meaning_count']},{data['total_occurrences']},{cids}\n")

    print(f"Summary CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
