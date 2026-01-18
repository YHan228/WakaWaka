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
import numpy as np
import networkx as nx

from wakawaka.schemas.curriculum import (
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

# Literary integration configuration
DEFAULT_LITERARY_INPUT = PROJECT_ROOT / "data" / "literary" / "poems_literary.parquet"
DEFAULT_LITERARY_DIFFICULTY = PROJECT_ROOT / "data" / "literary" / "literary_difficulty.json"
DEFAULT_LITERARY_INDEX = PROJECT_ROOT / "data" / "literary" / "literary_index.json"

# Chinese-adjusted difficulty (use instead of raw difficulty)
DEFAULT_CHINESE_DIFFICULTY = PROJECT_ROOT / "data" / "analysis" / "chinese_difficulty.json"
DEFAULT_COMBINED_DIFFICULTY = PROJECT_ROOT / "data" / "literary" / "combined_difficulty.json"
DEFAULT_CONFUSABLES = PROJECT_ROOT / "data" / "analysis" / "confusables.json"
GRAMMAR_WEIGHT = 0.7  # Weight for grammar difficulty in combined score
LITERARY_WEIGHT = 0.3  # Weight for literary difficulty in combined score
FOUNDATION_UNIT_COUNT = 3  # First N units are "foundation" (no literary focus)

# Standardized season/theme data
DEFAULT_SEASON_THEME = PROJECT_ROOT / "data" / "literary" / "standardized_season_theme.json"

# Major poetic devices to track for literary_focus
# Includes common variants (Japanese/Chinese names, with/without okurigana)
MAJOR_DEVICES = {
    # Core waka devices
    '掛詞', '挂词', 'かけことば',  # Pivot word
    '縁語', '缘语', 'えんご',      # Associated words
    '枕詞', '枕词', 'まくらことば',  # Pillow word
    '序詞', '序词', 'じょことば',   # Preface
    '体言止め', '体言止', 'たいげんどめ',  # Noun ending
    '見立て', '见立', 'みたて',     # Conceit/metaphor
    '本歌取り', '本歌取', 'ほんかどり',  # Allusive variation
    # Kakarimusubi variants
    '係結', '係り結び', '係結び', '係り結', 'かかりむすび',
    '係結 (Kakarimusubi)',
    # Other common devices
    '歌枕', 'うたまくら',           # Poetic place name
    '切れ字', '切字', 'きれじ',      # Cutting word
    '倒置', '倒置法', '倒装',        # Inversion
    '拟人', '擬人法', '擬人',        # Personification
    '対句', '对句',                 # Parallelism
    '反語', '反问', '反実仮想',      # Rhetorical question/counterfactual
    '呼告', '呼びかけ',             # Apostrophe
}


# -----------------------------------------------------------------------------
# Literary Data Loading
# -----------------------------------------------------------------------------

def load_literary_data(
    literary_input: Path | None = None,
    literary_difficulty_path: Path | None = None,
    literary_index_path: Path | None = None
) -> tuple[pd.DataFrame | None, dict | None, dict | None]:
    """
    Load literary analysis data for curriculum integration.

    Returns:
        Tuple of (literary_df, literary_difficulty_dict, literary_index_dict)
    """
    literary_df = None
    literary_difficulty = None
    literary_index = None

    # Load literary annotations parquet
    if literary_input and literary_input.exists():
        logger.info(f"Loading literary annotations from {literary_input}...")
        literary_df = pd.read_parquet(literary_input)
        logger.info(f"Loaded literary data for {len(literary_df)} poems")
    else:
        logger.warning("No literary annotations found, skipping literary integration")

    # Load pre-computed literary difficulty scores
    if literary_difficulty_path and literary_difficulty_path.exists():
        with open(literary_difficulty_path, 'r', encoding='utf-8') as f:
            literary_difficulty = json.load(f)
        logger.info(f"Loaded literary difficulty scores for {len(literary_difficulty)} poems")
    else:
        logger.warning("No literary difficulty scores found")

    # Load literary index (device frequency)
    if literary_index_path and literary_index_path.exists():
        with open(literary_index_path, 'r', encoding='utf-8') as f:
            literary_index = json.load(f)
        logger.info(f"Loaded literary index with {len(literary_index)} devices")
    else:
        logger.warning("No literary index found")

    return literary_df, literary_difficulty, literary_index


def parse_poetic_devices(devices_raw) -> list[str]:
    """Parse and extract device names from various formats."""
    if devices_raw is None or (isinstance(devices_raw, float) and np.isnan(devices_raw)):
        return []

    if isinstance(devices_raw, np.ndarray):
        devices_raw = devices_raw.tolist()

    if isinstance(devices_raw, str):
        try:
            import ast
            devices_raw = ast.literal_eval(devices_raw)
        except (ValueError, SyntaxError):
            try:
                devices_raw = json.loads(devices_raw)
            except json.JSONDecodeError:
                return []

    if isinstance(devices_raw, list):
        names = []
        for device in devices_raw:
            if isinstance(device, dict) and 'name' in device:
                names.append(device['name'])
            elif isinstance(device, str):
                try:
                    import ast
                    parsed = ast.literal_eval(device)
                    if isinstance(parsed, dict) and 'name' in parsed:
                        names.append(parsed['name'])
                except:
                    pass
        return names

    return []


def get_poem_devices(
    poem_id: str,
    literary_df: pd.DataFrame | None
) -> list[str]:
    """Get list of poetic devices for a poem."""
    if literary_df is None:
        return []

    row = literary_df[literary_df['poem_id'] == poem_id]
    if row.empty:
        return []

    devices_raw = row.iloc[0].get('poetic_devices')
    return parse_poetic_devices(devices_raw)


# -----------------------------------------------------------------------------
# Load Chinese-Adjusted Difficulty
# -----------------------------------------------------------------------------

def load_chinese_difficulty(path: Path = DEFAULT_CHINESE_DIFFICULTY) -> dict[str, float]:
    """
    Load Chinese-adjusted difficulty scores.

    Returns:
        Dict of poem_id -> chinese_adjusted_difficulty
    """
    if not path.exists():
        logger.warning(f"Chinese difficulty file not found: {path}")
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for entry in data:
        result[entry['poem_id']] = entry['chinese_adjusted_difficulty']

    logger.info(f"Loaded Chinese-adjusted difficulty for {len(result)} poems")
    return result


def load_combined_difficulty(path: Path = DEFAULT_COMBINED_DIFFICULTY) -> dict[str, float]:
    """
    Load combined difficulty scores (grammar + literary).

    Returns:
        Dict of poem_id -> combined_difficulty
    """
    if not path.exists():
        logger.warning(f"Combined difficulty file not found: {path}")
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {pid: entry['combined_difficulty'] for pid, entry in data.items()}
    logger.info(f"Loaded combined difficulty for {len(result)} poems")
    return result


def load_confusables(path: Path = DEFAULT_CONFUSABLES) -> dict[str, list]:
    """
    Load grammar confusables for curriculum awareness.

    Returns:
        Dict of surface -> list of meanings
    """
    if not path.exists():
        logger.warning(f"Confusables file not found: {path}")
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    confusables = data.get('confusables', {})
    logger.info(f"Loaded {len(confusables)} confusable surfaces")
    return confusables


def load_season_theme(path: Path = DEFAULT_SEASON_THEME) -> dict[str, dict]:
    """Load standardized season/theme labels for poems."""
    if not path.exists():
        logger.warning(f"Season/theme file not found: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded season/theme for {len(data)} poems")
    return data


# -----------------------------------------------------------------------------
# Step 1: Build Grammar Index
# -----------------------------------------------------------------------------

def build_grammar_index(
    poems_df: pd.DataFrame,
    min_frequency: int = 1,
    chinese_difficulty: dict[str, float] | None = None
) -> dict[str, GrammarIndexEntry]:
    """
    Build grammar index from annotated poems.
    Groups by canonical_id, aggregates senses within each.

    Args:
        poems_df: DataFrame with annotated poems
        min_frequency: Minimum poem frequency for inclusion
        chinese_difficulty: Optional Chinese-adjusted difficulty scores

    Returns:
        Dict of canonical_id -> GrammarIndexEntry
    """
    logger.info("Building grammar index...")

    if chinese_difficulty is None:
        chinese_difficulty = {}

    # Collect all grammar points
    all_points = []
    for _, row in poems_df.iterrows():
        poem_id = row['poem_id']
        # Use Chinese-adjusted difficulty if available, otherwise fall back to raw
        difficulty = chinese_difficulty.get(poem_id, row.get('difficulty_score_computed', 0.0))

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
    difficulty_tiers: int = DEFAULT_DIFFICULTY_TIERS,
    literary_df: pd.DataFrame | None = None,
    literary_difficulty: dict | None = None,
    foundation_unit_count: int = FOUNDATION_UNIT_COUNT,
    combined_difficulty: dict[str, float] | None = None,
    season_theme: dict[str, dict] | None = None
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
        literary_df: DataFrame with literary annotations (optional)
        literary_difficulty: Dict of poem_id -> literary difficulty info (optional)
        foundation_unit_count: First N units are "foundation" (no literary focus)
        combined_difficulty: Dict of poem_id -> combined (grammar+literary) difficulty

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
        grammar_points = row.get('grammar_points', [])
        if grammar_points is None:
            continue
        if isinstance(grammar_points, np.ndarray):
            grammar_points = grammar_points.tolist()
        for gp in grammar_points:
            if isinstance(gp, dict) and 'canonical_id' in gp:
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
        # Note: literary_ready will be determined after units are formed
        candidate_poem_ids, _, _ = select_poems_for_lesson(
            canonical_id, poems_df, poem_to_canonical, candidate_pool_size,
            literary_difficulty=literary_difficulty,
            literary_df=literary_df,
            prefer_literary_rich=False,  # Will update after unit assignment
            combined_difficulty=combined_difficulty,
            season_theme=season_theme
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
            poem_ids=[],  # Will be populated by LLM poem selection step
            literary_ready=False,  # Will be updated after units are formed
            literary_focus=[],
            literary_difficulty=0.0
        )

        unit_id = f"unit_{entry.category}"
        units_dict[unit_id].append(lesson)

    # Convert to Unit objects
    units = [
        Unit(id=unit_id, lessons=lessons)
        for unit_id, lessons in sorted(units_dict.items())
    ]

    # Determine which lessons are "literary_ready" based on unit position
    # First N units are foundation (no literary focus)
    # After that, lessons are literary_ready
    logger.info(f"Marking lessons as literary_ready (foundation units: {foundation_unit_count})")
    foundation_lesson_ids = set()

    for i, unit in enumerate(units):
        is_foundation_unit = i < foundation_unit_count

        for lesson in unit.lessons:
            if is_foundation_unit:
                foundation_lesson_ids.add(lesson.id)
                lesson.literary_ready = False
            else:
                lesson.literary_ready = True

                # Re-select poems preferring literary-rich ones
                if literary_df is not None or season_theme:
                    new_poems, lit_focus, lit_diff = select_poems_for_lesson(
                        lesson.canonical_grammar_point, poems_df, poem_to_canonical,
                        candidate_pool_size,
                        literary_difficulty=literary_difficulty,
                        literary_df=literary_df,
                        prefer_literary_rich=True,
                        combined_difficulty=combined_difficulty,
                        season_theme=season_theme
                    )
                    lesson.candidate_poem_ids = new_poems
                    lesson.literary_focus = lit_focus
                    lesson.literary_difficulty = round(lit_diff, 3)

    # Log literary integration stats
    literary_ready_count = sum(
        1 for u in units for l in u.lessons if l.literary_ready
    )
    total_lessons = sum(len(u.lessons) for u in units)
    logger.info(f"Built {len(units)} units with {total_lessons} lessons")
    logger.info(f"Literary-ready lessons: {literary_ready_count}/{total_lessons}")

    return units, dict(prereq_lookup)


def select_poems_for_lesson(
    canonical_id: str,
    poems_df: pd.DataFrame,
    poem_to_canonical: dict[str, set],
    max_poems: int,
    literary_difficulty: dict | None = None,
    literary_df: pd.DataFrame | None = None,
    prefer_literary_rich: bool = False,
    combined_difficulty: dict[str, float] | None = None,
    season_theme: dict[str, dict] | None = None
) -> tuple[list[str], list[str], float]:
    """
    Select poems that best teach a grammar point using stratified sampling.

    Uses stratified sampling across difficulty tiers to give the LLM variety
    to choose from, rather than just the easiest poems. This ensures the
    candidate pool represents the full difficulty distribution of poems
    containing this grammar point.

    Criteria:
    - Contains the target canonical point
    - Stratified across 3-5 difficulty tiers (proportional sampling)
    - Within each tier: prefer poems with notable poetic devices if literary-ready
    - Final output sorted by difficulty for consistent ordering

    Args:
        canonical_id: Target grammar point
        poems_df: Annotated poems
        poem_to_canonical: Mapping of poem_id -> canonical_ids
        max_poems: Maximum poems to select
        literary_difficulty: Dict of poem_id -> literary difficulty info
        literary_df: DataFrame with literary annotations
        prefer_literary_rich: If True, prefer poems with more devices within each tier
        combined_difficulty: Dict of poem_id -> combined (grammar+literary) difficulty

    Returns:
        Tuple of (poem_ids, literary_focus_devices, avg_literary_difficulty)
    """
    # Filter to poems containing this canonical point
    candidate_ids = [
        pid for pid, cids in poem_to_canonical.items()
        if canonical_id in cids
    ]

    if not candidate_ids:
        return [], [], 0.0

    # Get difficulties for sorting
    candidates = poems_df[poems_df['poem_id'].isin(candidate_ids)].copy()

    # Use combined difficulty if available (grammar + literary), else fall back to grammar only
    if combined_difficulty and prefer_literary_rich:
        candidates['sort_difficulty'] = candidates['poem_id'].apply(
            lambda pid: combined_difficulty.get(pid, candidates.loc[candidates['poem_id'] == pid, 'difficulty_score_computed'].iloc[0] if len(candidates[candidates['poem_id'] == pid]) > 0 else 0.5)
        )
    else:
        candidates['sort_difficulty'] = candidates['difficulty_score_computed']

    # Add literary difficulty if available
    if literary_difficulty:
        candidates['literary_diff'] = candidates['poem_id'].apply(
            lambda pid: literary_difficulty.get(pid, {}).get('literary_difficulty', 0.0)
        )
    else:
        candidates['literary_diff'] = 0.0

    # Add device count if literary data available
    if prefer_literary_rich and literary_df is not None:
        candidates['device_count'] = candidates['poem_id'].apply(
            lambda pid: literary_difficulty.get(pid, {}).get('device_count', 0)
            if literary_difficulty else 0
        )
    else:
        candidates['device_count'] = 0

    # Stratified sampling: pick poems across the difficulty distribution
    # This gives LLM variety to choose from, not just easiest poems
    n_candidates = len(candidates)

    if n_candidates <= max_poems:
        # Not enough candidates, take all
        selected = candidates.sort_values('sort_difficulty')
    else:
        # Stratified sampling across difficulty tiers
        # Use 3-5 tiers depending on pool size
        n_tiers = min(5, max(3, n_candidates // 10))
        candidates['difficulty_tier'] = pd.qcut(
            candidates['sort_difficulty'],
            q=n_tiers,
            labels=False,
            duplicates='drop'
        )

        # Calculate poems per tier (distribute evenly, with extras going to easier tiers)
        actual_tiers = candidates['difficulty_tier'].nunique()
        base_per_tier = max_poems // actual_tiers
        extras = max_poems % actual_tiers

        selected_rows = []
        for tier in sorted(candidates['difficulty_tier'].unique()):
            tier_poems = candidates[candidates['difficulty_tier'] == tier]

            # Easier tiers (lower numbers) get priority for extra slots
            n_from_tier = base_per_tier + (1 if tier < extras else 0)

            if prefer_literary_rich and literary_df is not None:
                # Within tier, prefer poems with more devices
                tier_poems = tier_poems.sort_values('device_count', ascending=False)
            else:
                # Within tier, sort by difficulty for consistent ordering
                tier_poems = tier_poems.sort_values('sort_difficulty')

            selected_rows.append(tier_poems.head(n_from_tier))

        selected = pd.concat(selected_rows, ignore_index=True)
        # Final sort by difficulty for output ordering
        selected = selected.sort_values('sort_difficulty')

    selected_ids = selected['poem_id'].tolist()

    # Compute literary focus (devices + season/themes) and average difficulty
    literary_focus = []
    if literary_df is not None or season_theme:
        device_counter = Counter()
        season_counter = Counter()
        theme_counter = Counter()

        for pid in selected_ids:
            # Count poetic devices from literary_df
            if literary_df is not None:
                devices = get_poem_devices(pid, literary_df)
                for device in devices:
                    if device in MAJOR_DEVICES:
                        device_counter[device] += 1

            # Use standardized season/theme data
            if season_theme and pid in season_theme:
                st = season_theme[pid]
                if st.get('season') and st['season'] != '無季':
                    season_counter[st['season']] += 1
                for theme in st.get('themes', []):
                    if theme != '其他':
                        theme_counter[theme] += 1

        # Top devices (up to 3) - relaxed from 2
        top_devices = [d for d, _ in device_counter.most_common(3)]

        # Dominant season (25%+ threshold) - relaxed from 40%
        top_season = None
        if season_counter:
            most_common_season, season_count = season_counter.most_common(1)[0]
            if season_count >= len(selected_ids) * 0.25:
                top_season = f"【{most_common_season}】"

        # Dominant themes (25%+ threshold, up to 2) - relaxed from 40%/max 1
        top_themes = []
        if theme_counter:
            for theme, theme_count in theme_counter.most_common(2):
                if theme_count >= len(selected_ids) * 0.25:
                    top_themes.append(theme)

        # Combine: season > themes > devices (up to 5 total) - increased from 3
        if top_season:
            literary_focus.append(top_season)
        literary_focus.extend(top_themes[:max(0, 2 - (1 if top_season else 0))])
        literary_focus.extend(top_devices[:max(0, 5 - len(literary_focus))])

    avg_literary_diff = selected['literary_diff'].mean() if len(selected) > 0 else 0.0

    return selected_ids, literary_focus, avg_literary_diff


# -----------------------------------------------------------------------------
# Main Extraction Pipeline
# -----------------------------------------------------------------------------

def extract_curriculum(
    input_path: Path,
    output_dir: Path,
    min_poems_per_lesson: int = DEFAULT_MIN_POEMS_PER_LESSON,
    max_lessons: int = DEFAULT_MAX_LESSONS,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    difficulty_tiers: int = DEFAULT_DIFFICULTY_TIERS,
    literary_input: Path | None = None,
    literary_difficulty_path: Path | None = None,
    foundation_unit_count: int = FOUNDATION_UNIT_COUNT
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
        literary_input: Path to literary annotations parquet (optional)
        literary_difficulty_path: Path to literary difficulty JSON (optional)
        foundation_unit_count: First N units are "foundation" (no literary focus)

    Returns:
        LessonGraph object
    """
    # Load annotated poems
    logger.info(f"Loading annotated poems from {input_path}...")
    poems_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(poems_df)} annotated poems")

    # Load literary data (optional)
    if literary_input is not None or literary_difficulty_path is not None:
        literary_df, literary_difficulty, literary_index = load_literary_data(
            literary_input=literary_input or DEFAULT_LITERARY_INPUT,
            literary_difficulty_path=literary_difficulty_path or DEFAULT_LITERARY_DIFFICULTY,
            literary_index_path=DEFAULT_LITERARY_INDEX
        )
    else:
        logger.info("Literary integration disabled")
        literary_df, literary_difficulty, literary_index = None, None, None

    # Load Chinese-adjusted difficulty, combined difficulty, confusables, and season/theme
    chinese_difficulty = load_chinese_difficulty()
    combined_difficulty = load_combined_difficulty()
    confusables = load_confusables()
    season_theme = load_season_theme()

    # Step 1: Build grammar index (using Chinese-adjusted difficulty)
    grammar_index = build_grammar_index(poems_df, min_frequency=1, chinese_difficulty=chinese_difficulty)

    if not grammar_index:
        raise ValueError("No grammar points found in corpus")

    # Step 2: Compute co-occurrences
    grammar_index = compute_co_occurrences(poems_df, grammar_index)

    # Step 3: Infer prerequisites
    edges, stoplist_applied = infer_prerequisites(grammar_index)

    # Step 4: Break cycles
    final_edges, removed_edges = break_cycles(edges)

    # Step 5: Build lesson graph (with literary integration)
    units, prereq_map = build_lesson_graph(
        grammar_index,
        final_edges,
        poems_df,
        min_poems_per_lesson=min_poems_per_lesson,
        max_lessons=max_lessons,
        candidate_pool_size=candidate_pool_size,
        difficulty_tiers=difficulty_tiers,
        literary_df=literary_df,
        literary_difficulty=literary_difficulty,
        foundation_unit_count=foundation_unit_count,
        combined_difficulty=combined_difficulty,
        season_theme=season_theme
    )

    # Build prerequisite graph
    prerequisite_graph = PrerequisiteGraph(
        edges=final_edges,
        removed_edges=removed_edges,
        stoplist_applied=stoplist_applied
    )

    # Calculate literary integration stats
    literary_ready_count = sum(1 for u in units for l in u.lessons if l.literary_ready)
    total_lessons = sum(len(u.lessons) for u in units)

    # Build metadata
    meta = {
        'generated_at': datetime.now().isoformat(),
        'corpus_size': len(poems_df),
        'total_lessons': total_lessons,
        'total_units': len(units),
        'total_canonical_points': len(grammar_index),
        'cycles_broken': len(removed_edges),
        'stoplist_size': len(stoplist_applied),
        'literary_integration': {
            'enabled': literary_df is not None,
            'literary_ready_lessons': literary_ready_count,
            'foundation_units': foundation_unit_count,
            'grammar_weight': GRAMMAR_WEIGHT,
            'literary_weight': LITERARY_WEIGHT
        },
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

            # Literary integration info
            lit_ready = "✓" if lesson.literary_ready else "✗"
            lines.append(f"  - Literary ready: {lit_ready}")
            if lesson.literary_focus:
                lines.append(f"  - Literary focus: {', '.join(lesson.literary_focus)}")
            if lesson.literary_difficulty > 0:
                lines.append(f"  - Literary difficulty: {lesson.literary_difficulty:.2f}")

            lines.append("")

    # Literary integration summary
    lit_meta = lesson_graph.meta.get('literary_integration', {})
    if lit_meta.get('enabled'):
        lines.extend([
            "## Literary Integration",
            "",
            f"- **Enabled**: Yes",
            f"- **Foundation units**: {lit_meta.get('foundation_units', 0)}",
            f"- **Literary-ready lessons**: {lit_meta.get('literary_ready_lessons', 0)}",
            f"- **Grammar weight**: {lit_meta.get('grammar_weight', 0.7)}",
            f"- **Literary weight**: {lit_meta.get('literary_weight', 0.3)}",
            "",
        ])

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
    parser.add_argument(
        "--literary-input",
        type=Path,
        default=DEFAULT_LITERARY_INPUT,
        help=f"Path to literary annotations parquet (default: {DEFAULT_LITERARY_INPUT})",
    )
    parser.add_argument(
        "--literary-difficulty",
        type=Path,
        default=DEFAULT_LITERARY_DIFFICULTY,
        help=f"Path to literary difficulty JSON (default: {DEFAULT_LITERARY_DIFFICULTY})",
    )
    parser.add_argument(
        "--foundation-units",
        type=int,
        default=FOUNDATION_UNIT_COUNT,
        help=f"Number of foundation units (no literary focus) (default: {FOUNDATION_UNIT_COUNT})",
    )
    parser.add_argument(
        "--no-literary",
        action="store_true",
        help="Disable literary integration entirely",
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
        difficulty_tiers=args.difficulty_tiers,
        literary_input=None if args.no_literary else args.literary_input,
        literary_difficulty_path=None if args.no_literary else args.literary_difficulty,
        foundation_unit_count=args.foundation_units
    )

    print(f"\nCurriculum extracted to {args.output_dir}/")
    print(f"  - Units: {len(lesson_graph.units)}")
    print(f"  - Lessons: {lesson_graph.meta['total_lessons']}")
    print(f"  - Grammar points: {lesson_graph.meta['total_canonical_points']}")

    # Literary integration stats
    lit_meta = lesson_graph.meta.get('literary_integration', {})
    if lit_meta.get('enabled'):
        print(f"  - Literary-ready lessons: {lit_meta.get('literary_ready_lessons', 0)}")

    print(f"\nVerify with:")
    print(f'  cat {args.output_dir}/curriculum_report.md')


if __name__ == "__main__":
    main()
