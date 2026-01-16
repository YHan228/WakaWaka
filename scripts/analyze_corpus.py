#!/usr/bin/env python3
"""
Corpus Analysis Script for WakaDecoder (Phase 5.5)

This script generates comprehensive analysis reports for the annotated corpus
and curriculum structure to review before proceeding to lesson generation.

Generated Reports:
- corpus_summary.md           - Overview stats
- grammar_point_frequency.csv - canonical_id, sense_id, count, examples
- difficulty_distribution.png - Histogram
- vocabulary_coverage.csv     - Words extracted, by frequency
- lesson_graph.png            - Visual DAG of prerequisites
- sample_annotations.md       - Random poems with full annotations
- quality_flags.md            - Potential issues detected
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import random

import pandas as pd

# Optional visualization dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakawaka.schemas.annotation import PoemAnnotation
from wakawaka.schemas.curriculum import LessonGraph, GrammarIndex


def load_poems(poems_path: Path) -> pd.DataFrame:
    """Load annotated poems from parquet."""
    df = pd.read_parquet(poems_path)
    return df


def load_curriculum(curriculum_dir: Path) -> tuple[LessonGraph, GrammarIndex]:
    """Load curriculum data."""
    with open(curriculum_dir / 'lesson_graph.json') as f:
        lesson_graph = LessonGraph(**json.load(f))

    with open(curriculum_dir / 'grammar_index.json') as f:
        grammar_index = GrammarIndex(**json.load(f))

    return lesson_graph, grammar_index


def to_list(val):
    """Convert numpy arrays or other iterables to Python lists."""
    if val is None:
        return []
    if hasattr(val, 'tolist'):  # numpy array
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return []


def generate_corpus_summary(df: pd.DataFrame, lesson_graph: LessonGraph,
                           grammar_index: GrammarIndex, output_dir: Path):
    """Generate corpus_summary.md with overview statistics."""

    # Basic stats
    total_poems = len(df)
    sources = df['source'].value_counts().to_dict()

    # Token stats
    avg_tokens = df['fugashi_tokens'].apply(lambda x: len(to_list(x))).mean()

    # Grammar point stats
    grammar_counts = []
    for gps in df['grammar_points']:
        grammar_counts.append(len(to_list(gps)))
    avg_grammar = sum(grammar_counts) / len(grammar_counts) if grammar_counts else 0

    # Vocabulary stats
    vocab_counts = []
    for vs in df['vocabulary']:
        vocab_counts.append(len(to_list(vs)))
    avg_vocab = sum(vocab_counts) / len(vocab_counts) if vocab_counts else 0

    # Difficulty stats
    difficulties = df['difficulty_score_computed'].tolist()
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0

    # Curriculum stats
    total_units = len(lesson_graph.units)
    total_lessons = sum(len(u.lessons) for u in lesson_graph.units)
    total_grammar_points = len(grammar_index.entries)

    content = f"""# WakaDecoder Corpus Analysis Report

Generated: {datetime.now().isoformat()}

## Corpus Summary

| Metric | Value |
|--------|-------|
| Total Poems | {total_poems} |
| Sources | {', '.join(f'{k}: {v}' for k, v in sources.items())} |
| Average Tokens per Poem | {avg_tokens:.1f} |
| Average Grammar Points per Poem | {avg_grammar:.1f} |
| Average Vocabulary Items per Poem | {avg_vocab:.1f} |
| Average Difficulty Score | {avg_difficulty:.3f} |

## Curriculum Summary

| Metric | Value |
|--------|-------|
| Total Units | {total_units} |
| Total Lessons | {total_lessons} |
| Total Grammar Points Discovered | {total_grammar_points} |
| Active Prerequisite Edges | {len(lesson_graph.prerequisite_graph.edges)} |
| Removed Edges (Cycle Breaking) | {len(lesson_graph.prerequisite_graph.removed_edges)} |
| Stoplist Applied | {', '.join(lesson_graph.prerequisite_graph.stoplist_applied) or 'None'} |

## Difficulty Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0.0 - 0.2 (Easy) | {sum(1 for d in difficulties if 0.0 <= d < 0.2)} | {sum(1 for d in difficulties if 0.0 <= d < 0.2) / total_poems * 100:.1f}% |
| 0.2 - 0.4 | {sum(1 for d in difficulties if 0.2 <= d < 0.4)} | {sum(1 for d in difficulties if 0.2 <= d < 0.4) / total_poems * 100:.1f}% |
| 0.4 - 0.6 | {sum(1 for d in difficulties if 0.4 <= d < 0.6)} | {sum(1 for d in difficulties if 0.4 <= d < 0.6) / total_poems * 100:.1f}% |
| 0.6 - 0.8 | {sum(1 for d in difficulties if 0.6 <= d < 0.8)} | {sum(1 for d in difficulties if 0.6 <= d < 0.8) / total_poems * 100:.1f}% |
| 0.8 - 1.0 (Hard) | {sum(1 for d in difficulties if 0.8 <= d <= 1.0)} | {sum(1 for d in difficulties if 0.8 <= d <= 1.0) / total_poems * 100:.1f}% |

## Source Breakdown

"""
    for source, count in sources.items():
        content += f"- **{source}**: {count} poems ({count/total_poems*100:.1f}%)\n"

    (output_dir / 'corpus_summary.md').write_text(content, encoding='utf-8')
    print(f"  -> corpus_summary.md")


def generate_grammar_frequency(df: pd.DataFrame, grammar_index: GrammarIndex,
                               output_dir: Path):
    """Generate grammar_point_frequency.csv."""

    rows = []
    for canonical_id, entry in grammar_index.entries.items():
        # Main entry
        rows.append({
            'canonical_id': canonical_id,
            'sense_id': '',
            'category': entry.category,
            'frequency': entry.frequency,
            'avg_difficulty': round(entry.avg_difficulty, 3),
            'surfaces': ', '.join(entry.surfaces[:5]),  # first 5
            'example_poem_ids': ', '.join(entry.senses[0].example_poem_ids[:3]) if entry.senses else ''
        })

        # Sense breakdown
        for sense in entry.senses:
            rows.append({
                'canonical_id': canonical_id,
                'sense_id': sense.sense_id,
                'category': entry.category,
                'frequency': sense.frequency,
                'avg_difficulty': round(sense.avg_difficulty, 3),
                'surfaces': ', '.join(sense.surfaces[:5]),
                'example_poem_ids': ', '.join(sense.example_poem_ids[:3])
            })

    freq_df = pd.DataFrame(rows)
    freq_df = freq_df.sort_values(['frequency', 'canonical_id'], ascending=[False, True])
    freq_df.to_csv(output_dir / 'grammar_point_frequency.csv', index=False)
    print(f"  -> grammar_point_frequency.csv ({len(rows)} entries)")


def generate_vocabulary_coverage(df: pd.DataFrame, output_dir: Path):
    """Generate vocabulary_coverage.csv."""

    vocab_counter = Counter()
    vocab_readings = {}
    vocab_meanings = {}

    for _, row in df.iterrows():
        vocab_list = to_list(row.get('vocabulary', []))
        for v in vocab_list:
            if isinstance(v, dict):
                word = v.get('word', '')
                if word:
                    vocab_counter[word] += 1
                    if word not in vocab_readings:
                        vocab_readings[word] = v.get('reading', '')
                        vocab_meanings[word] = v.get('meaning', '')

    rows = []
    for word, count in vocab_counter.most_common():
        rows.append({
            'word': word,
            'reading': vocab_readings.get(word, ''),
            'meaning': vocab_meanings.get(word, ''),
            'frequency': count
        })

    vocab_df = pd.DataFrame(rows)
    vocab_df.to_csv(output_dir / 'vocabulary_coverage.csv', index=False)
    print(f"  -> vocabulary_coverage.csv ({len(rows)} unique words)")


def generate_difficulty_distribution(df: pd.DataFrame, output_dir: Path):
    """Generate difficulty_distribution.png histogram."""

    if not HAS_MATPLOTLIB:
        print("  -> difficulty_distribution.png (SKIPPED: matplotlib not installed)")
        return

    difficulties = df['difficulty_score_computed'].tolist()

    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Difficulty Score')
    plt.ylabel('Number of Poems')
    plt.title('Difficulty Distribution of Annotated Poems')
    plt.axvline(x=sum(difficulties)/len(difficulties), color='red',
                linestyle='--', label=f'Mean: {sum(difficulties)/len(difficulties):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_distribution.png', dpi=150)
    plt.close()
    print(f"  -> difficulty_distribution.png")


def generate_lesson_graph_viz(lesson_graph: LessonGraph, output_dir: Path):
    """Generate lesson_graph.png visualization of prerequisite DAG."""

    if not HAS_NETWORKX or not HAS_MATPLOTLIB:
        print("  -> lesson_graph.png (SKIPPED: networkx/matplotlib not installed)")
        return

    G = nx.DiGraph()

    # Add all lessons as nodes
    for unit in lesson_graph.units:
        for lesson in unit.lessons:
            G.add_node(lesson.id,
                      tier=lesson.difficulty_tier,
                      unit=unit.id)

    # Add prerequisite edges
    for unit in lesson_graph.units:
        for lesson in unit.lessons:
            for prereq in lesson.prerequisites:
                if prereq in G.nodes:
                    G.add_edge(prereq, lesson.id)

    if len(G.nodes) == 0:
        print("  -> lesson_graph.png (SKIPPED: no lessons)")
        return

    plt.figure(figsize=(16, 12))

    # Position nodes by tier
    pos = {}
    tier_counts = defaultdict(int)
    for node in G.nodes:
        tier = G.nodes[node].get('tier', 1)
        tier_counts[tier] += 1

    tier_current = defaultdict(int)
    for node in G.nodes:
        tier = G.nodes[node].get('tier', 1)
        tier_current[tier] += 1
        x = tier
        y = tier_current[tier] - tier_counts[tier] / 2
        pos[node] = (x, y)

    # Color by unit
    units = list(set(G.nodes[n].get('unit', '') for n in G.nodes))
    colors = plt.cm.tab10(range(len(units)))
    color_map = {u: colors[i] for i, u in enumerate(units)}
    node_colors = [color_map.get(G.nodes[n].get('unit', ''), 'gray') for n in G.nodes]

    nx.draw(G, pos,
            node_color=node_colors,
            node_size=300,
            font_size=6,
            with_labels=True,
            arrows=True,
            edge_color='gray',
            alpha=0.8)

    plt.title('Lesson Prerequisite Graph (by Difficulty Tier)')
    plt.tight_layout()
    plt.savefig(output_dir / 'lesson_graph.png', dpi=150)
    plt.close()
    print(f"  -> lesson_graph.png")


def generate_sample_annotations(df: pd.DataFrame, output_dir: Path, num_samples: int = 10):
    """Generate sample_annotations.md with random poem annotations."""

    # Sample random poems
    sample_indices = random.sample(range(len(df)), min(num_samples, len(df)))

    content = f"""# Sample Annotations for Manual Review

Generated: {datetime.now().isoformat()}
Samples: {len(sample_indices)} randomly selected poems

---

"""

    for idx in sample_indices:
        row = df.iloc[idx]

        tokens = to_list(row.get('fugashi_tokens', []))
        grammar = to_list(row.get('grammar_points', []))
        vocab = to_list(row.get('vocabulary', []))
        factors = to_list(row.get('difficulty_factors', []))

        content += f"""## {row['poem_id']}

**Source**: {row.get('source', 'unknown')}
**Author**: {row.get('author', 'unknown')}
**Collection**: {row.get('collection', 'unknown')}

### Text
```
{row['text']}
```

### Readings
- **Hiragana**: {row.get('reading_hiragana', 'N/A')}
- **Romaji**: {row.get('reading_romaji', 'N/A')}

### Fugashi Tokens ({len(tokens)})
"""
        for i, t in enumerate(tokens):
            if isinstance(t, dict):
                span = to_list(t.get('span', []))
                content += f"- `{t.get('surface', '')}` ({t.get('pos', '')}) [{span}]\n"

        content += f"""
### Grammar Points ({len(grammar)})
"""
        for gp in grammar:
            if isinstance(gp, dict):
                sense = f" ({gp.get('sense_id', '')})" if gp.get('sense_id') else ""
                content += f"- **{gp.get('canonical_id', '')}{sense}**: `{gp.get('surface', '')}` - {gp.get('description', '')}\n"

        content += f"""
### Vocabulary ({len(vocab)})
"""
        for v in vocab:
            if isinstance(v, dict):
                cognate = f" (cognate: {v.get('chinese_cognate_note', '')})" if v.get('chinese_cognate_note') else ""
                content += f"- **{v.get('word', '')}** ({v.get('reading', '')}): {v.get('meaning', '')}{cognate}\n"

        content += f"""
### Difficulty
- **Score**: {row.get('difficulty_score_computed', 0):.3f}
- **Factors**:
"""
        for f in factors:
            if isinstance(f, dict):
                content += f"  - {f.get('factor', '')}: {f.get('weight', 0):.2f}"
                if f.get('note'):
                    content += f" - {f.get('note')}"
                content += "\n"

        if row.get('semantic_notes'):
            content += f"""
### Semantic Notes
{row.get('semantic_notes', '')}
"""

        content += "\n---\n\n"

    (output_dir / 'sample_annotations.md').write_text(content, encoding='utf-8')
    print(f"  -> sample_annotations.md ({len(sample_indices)} poems)")


def generate_quality_flags(df: pd.DataFrame, grammar_index: GrammarIndex,
                          output_dir: Path):
    """Generate quality_flags.md with auto-detected issues."""

    flags = []

    # Check poems with 0 grammar points
    zero_grammar = []
    for idx, row in df.iterrows():
        gps = to_list(row.get('grammar_points', []))
        if len(gps) == 0:
            zero_grammar.append(row['poem_id'])
    if zero_grammar:
        flags.append(f"### Poems with 0 Grammar Points ({len(zero_grammar)})\n" +
                    "\n".join(f"- {pid}" for pid in zero_grammar[:20]) +
                    (f"\n- ... and {len(zero_grammar) - 20} more" if len(zero_grammar) > 20 else ""))

    # Check poems with 0 vocabulary
    zero_vocab = []
    for idx, row in df.iterrows():
        vs = to_list(row.get('vocabulary', []))
        if len(vs) == 0:
            zero_vocab.append(row['poem_id'])
    if zero_vocab:
        flags.append(f"### Poems with 0 Vocabulary Items ({len(zero_vocab)})\n" +
                    "\n".join(f"- {pid}" for pid in zero_vocab[:20]) +
                    (f"\n- ... and {len(zero_vocab) - 20} more" if len(zero_vocab) > 20 else ""))

    # Check grammar IDs that appear only once (potential typos)
    singleton_grammar = []
    for cid, entry in grammar_index.entries.items():
        if entry.frequency == 1:
            singleton_grammar.append(f"{cid} (surface: {entry.surfaces[0] if entry.surfaces else 'N/A'})")
    if singleton_grammar:
        flags.append(f"### Grammar Points Appearing Only Once ({len(singleton_grammar)})\n" +
                    "These may be typos or very rare forms:\n" +
                    "\n".join(f"- {s}" for s in singleton_grammar[:30]) +
                    (f"\n- ... and {len(singleton_grammar) - 30} more" if len(singleton_grammar) > 30 else ""))

    # Check for span issues
    span_issues = []
    for idx, row in df.iterrows():
        text_len = len(row['text'])

        # Check token spans
        tokens = to_list(row.get('fugashi_tokens', []))
        for t in tokens:
            if isinstance(t, dict) and 'span' in t:
                span = to_list(t['span'])
                if len(span) >= 2 and span[1] > text_len:
                    span_issues.append(f"{row['poem_id']}: token span {span} exceeds text length {text_len}")

        # Check grammar point spans
        gps = to_list(row.get('grammar_points', []))
        for gp in gps:
            if isinstance(gp, dict) and 'span' in gp:
                span = to_list(gp['span'])
                if len(span) >= 2 and span[1] > text_len:
                    span_issues.append(f"{row['poem_id']}: grammar span {span} exceeds text length {text_len}")

    if span_issues:
        flags.append(f"### Span Issues ({len(span_issues)})\n" +
                    "\n".join(f"- {s}" for s in span_issues[:20]) +
                    (f"\n- ... and {len(span_issues) - 20} more" if len(span_issues) > 20 else ""))

    # Check token reading count mismatch
    reading_mismatches = []
    for idx, row in df.iterrows():
        tokens = to_list(row.get('fugashi_tokens', []))
        readings = to_list(row.get('token_readings', []))
        if len(tokens) != len(readings):
            reading_mismatches.append(f"{row['poem_id']}: {len(tokens)} tokens, {len(readings)} readings")

    if reading_mismatches:
        flags.append(f"### Token/Reading Count Mismatches ({len(reading_mismatches)})\n" +
                    "\n".join(f"- {s}" for s in reading_mismatches[:20]) +
                    (f"\n- ... and {len(reading_mismatches) - 20} more" if len(reading_mismatches) > 20 else ""))

    # Check suspicious difficulty scores
    extreme_difficulty = []
    for idx, row in df.iterrows():
        score = row.get('difficulty_score_computed', 0)
        if score == 0.0 or score == 1.0:
            extreme_difficulty.append(f"{row['poem_id']}: difficulty = {score}")

    if extreme_difficulty:
        flags.append(f"### Extreme Difficulty Scores ({len(extreme_difficulty)})\n" +
                    "Scores of exactly 0.0 or 1.0 may be suspicious:\n" +
                    "\n".join(f"- {s}" for s in extreme_difficulty[:20]) +
                    (f"\n- ... and {len(extreme_difficulty) - 20} more" if len(extreme_difficulty) > 20 else ""))

    # Generate report
    content = f"""# Quality Flags Report

Generated: {datetime.now().isoformat()}
Total Poems Analyzed: {len(df)}

---

"""

    if flags:
        content += "\n\n".join(flags)
    else:
        content += "**No quality issues detected!**"

    content += """

---

## Next Steps

Based on these flags:
1. **Minor issues** (singleton grammar, missing vocab): Can be fixed in post-processing or ignored for MVP
2. **Span issues**: Indicate annotation bugs - may need prompt adjustment
3. **Reading mismatches**: Indicate alignment issues - review annotation logic
4. **Extreme difficulties**: Review if scores make sense for those poems

If issues are systematic, consider adjusting the annotation prompt and re-running Phase 4.
"""

    (output_dir / 'quality_flags.md').write_text(content, encoding='utf-8')
    print(f"  -> quality_flags.md ({len(flags)} issue categories)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze annotated corpus and curriculum for quality review (Phase 5.5)'
    )
    parser.add_argument(
        '--poems', type=Path, default=Path('data/annotated/poems.parquet'),
        help='Path to annotated poems parquet file'
    )
    parser.add_argument(
        '--curriculum', type=Path, default=Path('data/curriculum'),
        help='Path to curriculum directory'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('data/analysis'),
        help='Output directory for reports'
    )
    parser.add_argument(
        '--samples', type=int, default=10,
        help='Number of sample poems to include in manual review'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.poems.exists():
        print(f"Error: Poems file not found: {args.poems}")
        sys.exit(1)
    if not args.curriculum.exists():
        print(f"Error: Curriculum directory not found: {args.curriculum}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_poems(args.poems)
    print(f"  Loaded {len(df)} poems")

    lesson_graph, grammar_index = load_curriculum(args.curriculum)
    print(f"  Loaded curriculum: {len(lesson_graph.units)} units, {len(grammar_index.entries)} grammar points")

    print("\nGenerating reports...")

    # Generate all reports
    generate_corpus_summary(df, lesson_graph, grammar_index, args.output)
    generate_grammar_frequency(df, grammar_index, args.output)
    generate_vocabulary_coverage(df, args.output)
    generate_difficulty_distribution(df, args.output)
    generate_lesson_graph_viz(lesson_graph, args.output)
    generate_sample_annotations(df, args.output, args.samples)
    generate_quality_flags(df, grammar_index, args.output)

    print(f"\nAnalysis complete! Reports saved to: {args.output}")
    print("\nReview these files before proceeding to Phase 6:")
    print("  1. corpus_summary.md - Check overall stats")
    print("  2. quality_flags.md - Address any issues")
    print("  3. sample_annotations.md - Manual quality check")
    print("  4. grammar_point_frequency.csv - Check grammar coverage")


if __name__ == '__main__':
    main()
