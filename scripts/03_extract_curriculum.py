#!/usr/bin/env python3
"""
03_extract_curriculum.py - Extract curriculum structure from annotated poems.

This script derives lesson structure from annotation statistics — no hardcoding.
It uses two-level grammar IDs (canonical_id + sense_id) and creates a DAG
of prerequisites using co-occurrence analysis.

Key features:
- Two-level grammar identity (lessons driven by canonical_id)
- Multi-criteria prerequisite inference (co_ratio, difficulty_gap, support)
- Stoplist for ultra-common particles
- Cycle detection and breaking using networkx
- Topological sort for lesson ordering

Usage:
  python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum
  python scripts/03_extract_curriculum.py --min-poems-per-lesson 2 --max-lessons 30
"""

import argparse
import json
import logging
import sys
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import networkx as nx

from wakadecoder.schemas.curriculum import (
    STOPLIST_CANONICAL_IDS,
    GrammarIndexEntry,
    SenseEntry,
    GrammarIndex,
    PrerequisiteEdge,
    PrerequisiteGraph,
    LessonNode,
    Unit,
    LessonGraph,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MIN_POEMS_PER_LESSON = 2
DEFAULT_MAX_LESSONS = 50
DEFAULT_DIFFICULTY_TIERS = 5
DEFAULT_POEMS_PER_LESSON = 3
DEFAULT_CANDIDATE_POOL_SIZE = 10  # Larger pool for LLM poem selection

# Prerequisite inference thresholds
CO_RATIO_THRESHOLD = 0.7      # 70% co-occurrence required
DIFFICULTY_GAP_THRESHOLD = 0.05  # Minimum difficulty difference
MIN_SUPPORT_COUNT = 2         # Minimum co-occurrence count (lowered for small corpus)


# -----------------------------------------------------------------------------
# Step 1: Build Grammar Index
# -----------------------------------------------------------------------------

def build_grammar_index(
    poems_df: pd.DataFrame,
    min_frequency: int = 1
) -> dict[str, GrammarIndexEntry]:
    """
    Build grammar index from annotated poems.
    Groups by canonical_id, aggregates senses within each.

    Args:
        poems_df: DataFrame with annotated poems
        min_frequency: Minimum poem frequency for inclusion

    Returns:
        Dict of canonical_id -> GrammarIndexEntry
    """
    logger.info("Building grammar index...")

    # Collect all grammar points
    all_points = []
    for _, row in poems_df.iterrows():
        poem_id = row['poem_id']
        difficulty = row.get('difficulty_score_computed', 0.0)

        for gp in row['grammar_points']:
            all_points.append({
                'canonical_id': gp['canonical_id'],
                'sense_id': gp.get('sense_id'),
                'category': gp['category'],
                'surface': gp['surface'],
                'poem_id': poem_id,
                'poem_difficulty': difficulty
            })

    if not all_points:
        logger.warning("No grammar points found in corpus")
        return {}

    points_df = pd.DataFrame(all_points)
    logger.info(f"Found {len(points_df)} grammar point occurrences")

    # Build index grouped by canonical_id
    grammar_index = {}

    for canonical_id, group in points_df.groupby('canonical_id'):
        # Get unique poems containing this point
        poem_ids = group['poem_id'].unique().tolist()
        frequency = len(poem_ids)

        if frequency < min_frequency:
            continue

        # Aggregate surfaces and category
        surfaces = list(group['surface'].unique())
        category = group['category'].iloc[0]
        avg_difficulty = group['poem_difficulty'].mean()

        # Build sense breakdown
        senses = []
        for sense_id, sense_group in group.groupby('sense_id', dropna=False):
            if pd.isna(sense_id) or sense_id is None:
                continue

            sense_poem_ids = sense_group['poem_id'].unique().tolist()
            senses.append(SenseEntry(
                sense_id=str(sense_id),
                surfaces=list(sense_group['surface'].unique()),
                frequency=len(sense_poem_ids),
                avg_difficulty=sense_group['poem_difficulty'].mean(),
                example_poem_ids=sense_poem_ids[:3]
            ))

        grammar_index[canonical_id] = GrammarIndexEntry(
            canonical_id=canonical_id,
            category=category,
            surfaces=surfaces,
            frequency=frequency,
            avg_difficulty=avg_difficulty,
            senses=senses,
            co_occurrences={}  # Filled in next step
        )

    logger.info(f"Built index with {len(grammar_index)} canonical grammar points")
    return grammar_index


# -----------------------------------------------------------------------------
# Step 2: Compute Co-occurrence Matrix
# -----------------------------------------------------------------------------

def compute_co_occurrences(
    poems_df: pd.DataFrame,
    grammar_index: dict[str, GrammarIndexEntry]
) -> dict[str, GrammarIndexEntry]:
    """
    Compute co-occurrence counts between canonical grammar points.

    Args:
        poems_df: DataFrame with annotated poems
        grammar_index: Grammar index to update

    Returns:
        Updated grammar index with co_occurrences filled
    """
    logger.info("Computing co-occurrence matrix...")

    # Build poem -> canonical_ids mapping
    poem_to_canonical = defaultdict(set)
    for _, row in poems_df.iterrows():
        poem_id = row['poem_id']
        for gp in row['grammar_points']:
            canonical_id = gp['canonical_id']
            if canonical_id in grammar_index:
                poem_to_canonical[poem_id].add(canonical_id)

    # Count co-occurrences
    co_occurrence = defaultdict(Counter)
    for poem_id, canonical_set in poem_to_canonical.items():
        for c1 in canonical_set:
            for c2 in canonical_set:
                if c1 != c2:
                    co_occurrence[c1][c2] += 1

    # Update grammar index
    for canonical_id in grammar_index:
        grammar_index[canonical_id].co_occurrences = dict(co_occurrence[canonical_id])

    total_pairs = sum(len(v) for v in co_occurrence.values())
    logger.info(f"Computed {total_pairs} co-occurrence pairs")

    return grammar_index


# -----------------------------------------------------------------------------
# Step 3: Infer Prerequisites
# -----------------------------------------------------------------------------

def infer_prerequisites(
    grammar_index: dict[str, GrammarIndexEntry],
    co_ratio_threshold: float = CO_RATIO_THRESHOLD,
    difficulty_gap_threshold: float = DIFFICULTY_GAP_THRESHOLD,
    min_support: int = MIN_SUPPORT_COUNT
) -> tuple[list[PrerequisiteEdge], list[str]]:
    """
    Infer prerequisite relationships between grammar points.

    Criteria for A -> B (A is prerequisite of B):
    - co_ratio: B appears with A in >= threshold of B's poems
    - difficulty_gap: B is harder than A by >= threshold
    - support: Co-occurrence count >= min_support

    Args:
        grammar_index: Grammar index with co-occurrences
        co_ratio_threshold: Minimum co-occurrence ratio
        difficulty_gap_threshold: Minimum difficulty difference
        min_support: Minimum co-occurrence count

    Returns:
        Tuple of (edges, stoplist_applied)
    """
    logger.info("Inferring prerequisites...")

    edges = []
    stoplist_applied = []

    for canonical_id, entry in grammar_index.items():
        for other_id, co_count in entry.co_occurrences.items():
            other_entry = grammar_index.get(other_id)
            if other_entry is None:
                continue

            # Skip stoplist items as prerequisites
            if other_id in STOPLIST_CANONICAL_IDS:
                if other_id not in stoplist_applied:
                    stoplist_applied.append(other_id)
                continue

            # Calculate metrics
            co_ratio = co_count / entry.frequency if entry.frequency > 0 else 0
            difficulty_gap = entry.avg_difficulty - other_entry.avg_difficulty

            # Check all criteria
            # other_id -> canonical_id means other_id is prerequisite
            # (other_id appears often with canonical_id, and canonical_id is harder)
            if (co_ratio >= co_ratio_threshold and
                difficulty_gap >= difficulty_gap_threshold and
                co_count >= min_support):

                edges.append(PrerequisiteEdge(
                    from_id=other_id,      # prerequisite
                    to_id=canonical_id,     # dependent
                    co_ratio=round(co_ratio, 3),
                    difficulty_gap=round(difficulty_gap, 3),
                    support_count=co_count
                ))

    logger.info(f"Found {len(edges)} prerequisite edges (stoplist: {len(stoplist_applied)})")
    return edges, stoplist_applied


# -----------------------------------------------------------------------------
# Step 4: Detect and Break Cycles
# -----------------------------------------------------------------------------

def break_cycles(
    edges: list[PrerequisiteEdge]
) -> tuple[list[PrerequisiteEdge], list[PrerequisiteEdge]]:
    """
    Detect and break cycles in prerequisite graph.
    Removes weakest edges (by difficulty_gap, then co_ratio).

    Args:
        edges: List of prerequisite edges

    Returns:
        Tuple of (remaining_edges, removed_edges)
    """
    logger.info("Detecting and breaking cycles...")

    if not edges:
        return [], []

    # Build directed graph
    G = nx.DiGraph()
    edge_map = {}  # (from_id, to_id) -> edge

    for edge in edges:
        G.add_edge(edge.from_id, edge.to_id)
        edge_map[(edge.from_id, edge.to_id)] = edge

    removed_edges = []
    cycles_broken = 0

    # Iteratively find and break cycles
    while True:
        try:
            cycle = nx.find_cycle(G)
            cycles_broken += 1

            # Find weakest edge in cycle
            cycle_edges = []
            for u, v in cycle:
                if (u, v) in edge_map:
                    cycle_edges.append(edge_map[(u, v)])

            if not cycle_edges:
                # Edge not in our map, remove arbitrary edge
                u, v = cycle[0]
                G.remove_edge(u, v)
                continue

            # Sort by weakness: lowest difficulty_gap, then lowest co_ratio
            weakest = min(cycle_edges, key=lambda e: (e.difficulty_gap, e.co_ratio))

            # Remove weakest edge
            G.remove_edge(weakest.from_id, weakest.to_id)
            removed_edges.append(weakest)
            del edge_map[(weakest.from_id, weakest.to_id)]

            logger.debug(f"Broke cycle by removing: {weakest.from_id} -> {weakest.to_id}")

        except nx.NetworkXNoCycle:
            break

    # Collect remaining edges
    remaining_edges = list(edge_map.values())

    logger.info(f"Broke {cycles_broken} cycles, removed {len(removed_edges)} edges")
    return remaining_edges, removed_edges


# -----------------------------------------------------------------------------
# Step 5: Topological Sort and Group into Units
# -----------------------------------------------------------------------------

def build_lesson_graph(
    grammar_index: dict[str, GrammarIndexEntry],
    prerequisite_edges: list[PrerequisiteEdge],
    poems_df: pd.DataFrame,
    min_poems_per_lesson: int = DEFAULT_MIN_POEMS_PER_LESSON,
    max_lessons: int = DEFAULT_MAX_LESSONS,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    difficulty_tiers: int = DEFAULT_DIFFICULTY_TIERS
) -> tuple[list[Unit], dict[str, list[str]]]:
    """
    Build lesson graph from grammar index and prerequisites.

    Args:
        grammar_index: Grammar index
        prerequisite_edges: Prerequisite edges (after cycle breaking)
        poems_df: Annotated poems DataFrame
        min_poems_per_lesson: Minimum poems required per lesson
        max_lessons: Maximum number of lessons
        candidate_pool_size: Number of candidate poems per lesson (for LLM selection)
        difficulty_tiers: Number of difficulty tiers (1-5)

    Returns:
        Tuple of (units, prerequisite_map)
    """
    logger.info("Building lesson graph...")

    # Build prerequisite lookup (canonical_id -> list of prerequisite canonical_ids)
    prereq_lookup = defaultdict(list)
    for edge in prerequisite_edges:
        prereq_lookup[edge.to_id].append(edge.from_id)

    # Build directed graph for topological sort
    G = nx.DiGraph()
    for canonical_id in grammar_index:
        G.add_node(canonical_id)
    for edge in prerequisite_edges:
        G.add_edge(edge.from_id, edge.to_id)

    # Topological sort
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        logger.warning("Graph has cycles (shouldn't happen after cycle breaking)")
        topo_order = list(grammar_index.keys())

    # Filter to entries with enough poems
    valid_entries = [
        grammar_index[cid] for cid in topo_order
        if cid in grammar_index and grammar_index[cid].frequency >= min_poems_per_lesson
    ]

    # Limit to max_lessons
    if len(valid_entries) > max_lessons:
        # Prioritize by frequency (more common = more important)
        valid_entries = sorted(valid_entries, key=lambda e: -e.frequency)[:max_lessons]
        # Re-sort by topological order
        valid_ids = {e.canonical_id for e in valid_entries}
        valid_entries = [grammar_index[cid] for cid in topo_order if cid in valid_ids]

    # Build poem -> canonical_ids mapping for poem selection
    poem_to_canonical = defaultdict(set)
    for _, row in poems_df.iterrows():
        for gp in row['grammar_points']:
            poem_to_canonical[row['poem_id']].add(gp['canonical_id'])

    # Group by category into units
    units_dict = defaultdict(list)

    for entry in valid_entries:
        canonical_id = entry.canonical_id

        # Calculate difficulty tier (1-5)
        difficulty_tier = min(
            difficulty_tiers,
            max(1, int(entry.avg_difficulty * difficulty_tiers) + 1)
        )

        # Select candidate poems for this lesson (larger pool for LLM selection)
        candidate_poem_ids = select_poems_for_lesson(
            canonical_id, poems_df, poem_to_canonical, candidate_pool_size
        )

        if not candidate_poem_ids:
            logger.debug(f"No poems for {canonical_id}, skipping")
            continue

        # Build prerequisite lesson IDs
        prereq_lesson_ids = [
            f"lesson_{pid}" for pid in prereq_lookup.get(canonical_id, [])
            if pid in grammar_index and grammar_index[pid].frequency >= min_poems_per_lesson
        ]

        # Create lesson node with candidate pool (poem_ids populated by LLM selection later)
        lesson = LessonNode(
            id=f"lesson_{canonical_id}",
            canonical_grammar_point=canonical_id,
            senses_covered=[s.sense_id for s in entry.senses[:2]],  # First 2 senses
            prerequisites=prereq_lesson_ids,
            difficulty_tier=difficulty_tier,
            candidate_poem_ids=candidate_poem_ids,
            poem_ids=[]  # Will be populated by LLM poem selection step
        )

        unit_id = f"unit_{entry.category}"
        units_dict[unit_id].append(lesson)

    # Convert to Unit objects
    units = [
        Unit(id=unit_id, lessons=lessons)
        for unit_id, lessons in sorted(units_dict.items())
    ]

    total_lessons = sum(len(u.lessons) for u in units)
    logger.info(f"Built {len(units)} units with {total_lessons} lessons")

    return units, dict(prereq_lookup)


def select_poems_for_lesson(
    canonical_id: str,
    poems_df: pd.DataFrame,
    poem_to_canonical: dict[str, set],
    max_poems: int
) -> list[str]:
    """
    Select poems that best teach a grammar point.

    Criteria:
    - Contains the target canonical point
    - Sorted by difficulty (easier first)

    Args:
        canonical_id: Target grammar point
        poems_df: Annotated poems
        poem_to_canonical: Mapping of poem_id -> canonical_ids
        max_poems: Maximum poems to select

    Returns:
        List of poem IDs
    """
    # Filter to poems containing this canonical point
    candidate_ids = [
        pid for pid, cids in poem_to_canonical.items()
        if canonical_id in cids
    ]

    if not candidate_ids:
        return []

    # Get difficulties for sorting
    candidates = poems_df[poems_df['poem_id'].isin(candidate_ids)].copy()
    candidates = candidates.sort_values('difficulty_score_computed')

    return candidates.head(max_poems)['poem_id'].tolist()


# -----------------------------------------------------------------------------
# Main Extraction Pipeline
# -----------------------------------------------------------------------------

def extract_curriculum(
    input_path: Path,
    output_dir: Path,
    min_poems_per_lesson: int = DEFAULT_MIN_POEMS_PER_LESSON,
    max_lessons: int = DEFAULT_MAX_LESSONS,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    difficulty_tiers: int = DEFAULT_DIFFICULTY_TIERS
) -> LessonGraph:
    """
    Extract curriculum from annotated poems.

    Args:
        input_path: Path to annotated poems parquet
        output_dir: Directory for output files
        min_poems_per_lesson: Minimum poems per lesson
        max_lessons: Maximum lessons to generate
        candidate_pool_size: Number of candidate poems per lesson (for LLM selection)
        difficulty_tiers: Number of difficulty tiers

    Returns:
        LessonGraph object
    """
    # Load annotated poems
    logger.info(f"Loading annotated poems from {input_path}...")
    poems_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(poems_df)} annotated poems")

    # Step 1: Build grammar index
    grammar_index = build_grammar_index(poems_df, min_frequency=1)

    if not grammar_index:
        raise ValueError("No grammar points found in corpus")

    # Step 2: Compute co-occurrences
    grammar_index = compute_co_occurrences(poems_df, grammar_index)

    # Step 3: Infer prerequisites
    edges, stoplist_applied = infer_prerequisites(grammar_index)

    # Step 4: Break cycles
    final_edges, removed_edges = break_cycles(edges)

    # Step 5: Build lesson graph
    units, prereq_map = build_lesson_graph(
        grammar_index,
        final_edges,
        poems_df,
        min_poems_per_lesson=min_poems_per_lesson,
        max_lessons=max_lessons,
        candidate_pool_size=candidate_pool_size,
        difficulty_tiers=difficulty_tiers
    )

    # Build prerequisite graph
    prerequisite_graph = PrerequisiteGraph(
        edges=final_edges,
        removed_edges=removed_edges,
        stoplist_applied=stoplist_applied
    )

    # Build metadata
    total_lessons = sum(len(u.lessons) for u in units)
    meta = {
        'generated_at': datetime.now().isoformat(),
        'corpus_size': len(poems_df),
        'total_lessons': total_lessons,
        'total_units': len(units),
        'total_canonical_points': len(grammar_index),
        'cycles_broken': len(removed_edges),
        'stoplist_size': len(stoplist_applied),
        'config': {
            'min_poems_per_lesson': min_poems_per_lesson,
            'max_lessons': max_lessons,
            'candidate_pool_size': candidate_pool_size,
            'difficulty_tiers': difficulty_tiers,
            'co_ratio_threshold': CO_RATIO_THRESHOLD,
            'difficulty_gap_threshold': DIFFICULTY_GAP_THRESHOLD,
            'min_support_count': MIN_SUPPORT_COUNT
        }
    }

    # Create lesson graph
    lesson_graph = LessonGraph(
        units=units,
        prerequisite_graph=prerequisite_graph,
        meta=meta
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save grammar index
    grammar_index_obj = GrammarIndex(
        entries=grammar_index,
        generated_at=meta['generated_at'],
        corpus_size=len(poems_df)
    )
    grammar_index_path = output_dir / "grammar_index.json"
    with open(grammar_index_path, 'w', encoding='utf-8') as f:
        json.dump(grammar_index_obj.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"Saved grammar index to {grammar_index_path}")

    # Save lesson graph
    lesson_graph_path = output_dir / "lesson_graph.json"
    with open(lesson_graph_path, 'w', encoding='utf-8') as f:
        json.dump(lesson_graph.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"Saved lesson graph to {lesson_graph_path}")

    # Save prerequisite graph (for analysis)
    prereq_graph_path = output_dir / "prerequisite_graph.json"
    with open(prereq_graph_path, 'w', encoding='utf-8') as f:
        json.dump(prerequisite_graph.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"Saved prerequisite graph to {prereq_graph_path}")

    # Generate curriculum report
    report_path = output_dir / "curriculum_report.md"
    generate_curriculum_report(lesson_graph, grammar_index, report_path)
    logger.info(f"Saved curriculum report to {report_path}")

    return lesson_graph


def generate_curriculum_report(
    lesson_graph: LessonGraph,
    grammar_index: dict[str, GrammarIndexEntry],
    output_path: Path
):
    """Generate human-readable curriculum report."""

    lines = [
        "# WakaDecoder Curriculum Report",
        "",
        f"Generated: {lesson_graph.meta['generated_at']}",
        "",
        "## Summary",
        "",
        f"- **Corpus size**: {lesson_graph.meta['corpus_size']} poems",
        f"- **Total units**: {lesson_graph.meta['total_units']}",
        f"- **Total lessons**: {lesson_graph.meta['total_lessons']}",
        f"- **Grammar points discovered**: {lesson_graph.meta['total_canonical_points']}",
        f"- **Cycles broken**: {lesson_graph.meta['cycles_broken']}",
        "",
        "## Units and Lessons",
        "",
    ]

    for unit in lesson_graph.units:
        lines.append(f"### {unit.id}")
        lines.append("")

        for lesson in unit.lessons:
            entry = grammar_index.get(lesson.canonical_grammar_point)
            freq = entry.frequency if entry else 0
            surfaces = ", ".join(entry.surfaces[:3]) if entry else ""

            prereqs = ", ".join(lesson.prerequisites) if lesson.prerequisites else "None"

            lines.append(f"- **{lesson.id}** (Tier {lesson.difficulty_tier})")
            lines.append(f"  - Grammar: `{lesson.canonical_grammar_point}` ({surfaces})")
            lines.append(f"  - Frequency: {freq} poems")
            lines.append(f"  - Prerequisites: {prereqs}")
            lines.append(f"  - Candidate poems: {len(lesson.candidate_poem_ids)}")
            lines.append("")

    lines.extend([
        "## Prerequisite Graph",
        "",
        f"- **Active edges**: {len(lesson_graph.prerequisite_graph.edges)}",
        f"- **Removed edges** (cycle breaking): {len(lesson_graph.prerequisite_graph.removed_edges)}",
        f"- **Stoplist applied**: {', '.join(lesson_graph.prerequisite_graph.stoplist_applied)}",
        "",
    ])

    if lesson_graph.prerequisite_graph.edges:
        lines.append("### Active Prerequisites")
        lines.append("")
        lines.append("| From | To | Co-ratio | Difficulty Gap | Support |")
        lines.append("|------|-----|----------|----------------|---------|")

        for edge in lesson_graph.prerequisite_graph.edges[:20]:  # Top 20
            lines.append(
                f"| {edge.from_id} | {edge.to_id} | "
                f"{edge.co_ratio:.2f} | {edge.difficulty_gap:.2f} | {edge.support_count} |"
            )

        if len(lesson_graph.prerequisite_graph.edges) > 20:
            lines.append(f"| ... | ... | ... | ... | ... |")
            lines.append(f"*(showing 20 of {len(lesson_graph.prerequisite_graph.edges)})*")

    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract curriculum structure from annotated poems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum
  python scripts/03_extract_curriculum.py --min-poems-per-lesson 2 --max-lessons 30
  python scripts/03_extract_curriculum.py --difficulty-tiers 3
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Input parquet file with annotated poems (default: data/annotated/poems.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "curriculum",
        help="Output directory for curriculum files (default: data/curriculum)",
    )
    parser.add_argument(
        "--min-poems-per-lesson",
        type=int,
        default=DEFAULT_MIN_POEMS_PER_LESSON,
        help=f"Minimum poems required per lesson (default: {DEFAULT_MIN_POEMS_PER_LESSON})",
    )
    parser.add_argument(
        "--max-lessons",
        type=int,
        default=DEFAULT_MAX_LESSONS,
        help=f"Maximum number of lessons (default: {DEFAULT_MAX_LESSONS})",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help=f"Number of candidate poems per lesson for LLM selection (default: {DEFAULT_CANDIDATE_POOL_SIZE})",
    )
    parser.add_argument(
        "--difficulty-tiers",
        type=int,
        default=DEFAULT_DIFFICULTY_TIERS,
        help=f"Number of difficulty tiers 1-N (default: {DEFAULT_DIFFICULTY_TIERS})",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Run extraction
    lesson_graph = extract_curriculum(
        input_path=args.input,
        output_dir=args.output_dir,
        min_poems_per_lesson=args.min_poems_per_lesson,
        max_lessons=args.max_lessons,
        candidate_pool_size=args.candidate_pool_size,
        difficulty_tiers=args.difficulty_tiers
    )

    print(f"\nCurriculum extracted to {args.output_dir}/")
    print(f"  - Units: {len(lesson_graph.units)}")
    print(f"  - Lessons: {lesson_graph.meta['total_lessons']}")
    print(f"  - Grammar points: {lesson_graph.meta['total_canonical_points']}")

    print(f"\nVerify with:")
    print(f'  cat {args.output_dir}/curriculum_report.md')


if __name__ == "__main__":
    main()
