#!/usr/bin/env python3
"""
analyze_literary_corpus.py - Statistical analysis of literary annotations.

This script analyzes the literary annotations in poems_literary.parquet to produce:
- Literary index (device frequency, co-occurrence)
- Thematic clusters (poems grouped by season/theme)
- Grammar-literary correlation matrix
- Literary complexity scores per poem (with Chinese adjustment)
- Combined difficulty scores (grammar + literary)
- Device progression paths for curriculum
- Kakarimusubi-literary correlation
- Visualization charts
- Human-readable analysis report

Outputs:
- data/literary/literary_index.json - Device frequency and co-occurrence
- data/literary/thematic_clusters.json - Poems grouped by theme/season
- data/literary/grammar_literary_correlation.json - Grammar↔Literary mapping
- data/literary/literary_difficulty.json - Literary complexity scores
- data/literary/combined_difficulty.json - Grammar + literary combined
- data/literary/device_progression.json - Device learning paths
- data/literary/kakarimusubi_literary.json - Kakarimusubi-literary effects
- data/literary/analysis_report.md - Human-readable summary
- data/literary/*.png - Visualization charts

Usage:
  python scripts/analyze_literary_corpus.py
  python scripts/analyze_literary_corpus.py --grammar-input data/annotated/poems.parquet
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project imports
from scripts.analysis_utils import setup_plotting

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_LITERARY_INPUT = PROJECT_ROOT / "data" / "literary" / "poems_literary.parquet"
DEFAULT_GRAMMAR_INPUT = PROJECT_ROOT / "data" / "annotated" / "poems.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "literary"
DEFAULT_CHINESE_DIFFICULTY = PROJECT_ROOT / "data" / "analysis" / "chinese_difficulty.json"
DEFAULT_CONFUSABLES = PROJECT_ROOT / "data" / "analysis" / "confusables.json"
DEFAULT_KAKARIMUSUBI = PROJECT_ROOT / "data" / "analysis" / "pos" / "kakarimusubi_patterns.csv"

# Chinese learner familiar devices (reduce difficulty for these)
CHINESE_FAMILIAR_DEVICES = {
    '対句': -0.10,      # Parallelism common in Chinese poetry
    '擬人法': -0.05,    # Personification familiar
    '見立て': -0.08,    # Conceit/metaphor similar to 比喻
    '反復': -0.03,      # Repetition
    '倒置': -0.02,      # Inversion
}

# Known poetic device name normalization
# Maps various forms (simplified Chinese, Japanese, romaji) to standard form
DEVICE_NORMALIZATION = {
    # 掛詞 kakekotoba (pivot word)
    '掛詞': '掛詞',
    '挂词': '掛詞',
    'kakekotoba': '掛詞',
    # 縁語 engo (associated words)
    '縁語': '縁語',
    '缘语': '縁語',
    'engo': '縁語',
    # 枕詞 makurakotoba (pillow word)
    '枕詞': '枕詞',
    '枕词': '枕詞',
    'makurakotoba': '枕詞',
    # 体言止め taigendome (noun ending)
    '体言止め': '体言止め',
    '体言止': '体言止め',
    'taigendome': '体言止め',
    # 序詞 jokotoba (preface)
    '序詞': '序詞',
    '序词': '序詞',
    'jokotoba': '序詞',
    # 本歌取り honkadori (allusive variation)
    '本歌取り': '本歌取り',
    '本歌取': '本歌取り',
    'honkadori': '本歌取り',
    # 見立て mitate (conceit/metaphor)
    '見立て': '見立て',
    '见立': '見立て',
    '见立て': '見立て',
    'mitate': '見立て',
    # 擬人法 (personification)
    '擬人法': '擬人法',
    '拟人': '擬人法',
    '拟人法': '擬人法',
    # 係結 (kakarimusubi)
    '係結': '係結',
    '係り結び': '係結',
    # 対句 (parallelism)
    '対句': '対句',
    '对句': '対句',
    # 反復 (repetition)
    '反復': '反復',
    '反复': '反復',
    # 倒置 (inversion)
    '倒置': '倒置',
    # 反問 (rhetorical question)
    '反問': '反問',
    '反问': '反問',
}

# Season keywords for classification
SEASON_KEYWORDS = {
    '春': ['春', '桜', 'さくら', '梅', 'うめ', '霞', 'かすみ', '鶯', 'うぐいす', '花'],
    '夏': ['夏', 'なつ', '蛍', 'ほたる', '杜鹃', 'ホトトギス', '時鳥', '涼', '暑'],
    '秋': ['秋', 'あき', '紅葉', 'もみじ', '月', '鹿', '虫', '露', '萩', '菊'],
    '冬': ['冬', 'ふゆ', '雪', 'ゆき', '霜', '枯', '寒', '氷'],
}

# Theme keywords for classification
THEME_KEYWORDS = {
    '恋': ['恋', '愛', '思', '人', '袖', '涙', '枕', '夢', '逢', '待'],
    '自然': ['山', '川', '海', '野', '風', '雨', '雲', '霧'],
    '無常': ['露', '儚', '消', '夢', '過', '散'],
    '旅': ['旅', '行', '道', '宿', '帰'],
    '別離': ['別', '離', '去', '惜'],
}


# -----------------------------------------------------------------------------
# Step 1: Parse Literary Data
# -----------------------------------------------------------------------------

def parse_poetic_devices(devices_raw) -> list[dict]:
    """
    Parse poetic devices from various formats.

    The data may be stored as:
    - List of dicts
    - String representation of list
    - None/NaN
    """
    if devices_raw is None or (isinstance(devices_raw, float) and np.isnan(devices_raw)):
        return []

    if isinstance(devices_raw, np.ndarray):
        devices_raw = devices_raw.tolist()

    if isinstance(devices_raw, str):
        # Try to parse as Python literal
        try:
            import ast
            devices_raw = ast.literal_eval(devices_raw)
        except (ValueError, SyntaxError):
            # Try JSON
            try:
                devices_raw = json.loads(devices_raw)
            except json.JSONDecodeError:
                return []

    if isinstance(devices_raw, list):
        result = []
        for device in devices_raw:
            if isinstance(device, dict):
                result.append(device)
            elif isinstance(device, str):
                # Try to parse string as dict
                try:
                    import ast
                    parsed = ast.literal_eval(device)
                    if isinstance(parsed, dict):
                        result.append(parsed)
                except:
                    pass
        return result

    return []


def normalize_device_name(name: str) -> str:
    """Normalize device name to standard form."""
    # Try direct mapping
    if name in DEVICE_NORMALIZATION:
        return DEVICE_NORMALIZATION[name]

    # Try lowercase
    name_lower = name.lower()
    if name_lower in DEVICE_NORMALIZATION:
        return DEVICE_NORMALIZATION[name_lower]

    # Extract Japanese term if present (e.g., "掛詞 kakekotoba" -> "掛詞")
    for key in DEVICE_NORMALIZATION:
        if key in name:
            return DEVICE_NORMALIZATION[key]

    return name


def extract_device_names(devices: list[dict]) -> list[str]:
    """Extract normalized device names from device list."""
    names = []
    for device in devices:
        if isinstance(device, dict) and 'name' in device:
            name = normalize_device_name(device['name'])
            names.append(name)
    return names


# -----------------------------------------------------------------------------
# Step 2: Build Literary Index
# -----------------------------------------------------------------------------

def build_literary_index(literary_df: pd.DataFrame) -> dict:
    """
    Build index of poetic devices with frequency and co-occurrence.

    Returns:
        Dict with structure:
        {
            "device_name": {
                "frequency": int,
                "co_occurring_devices": {"other_device": count},
                "example_poem_ids": [list of up to 5 poem IDs]
            }
        }
    """
    logger.info("Building literary index...")

    device_stats = defaultdict(lambda: {
        'frequency': 0,
        'co_occurring_devices': Counter(),
        'example_poem_ids': []
    })

    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']
        devices = parse_poetic_devices(row.get('poetic_devices'))
        device_names = extract_device_names(devices)

        if not device_names:
            continue

        # Deduplicate for this poem
        unique_names = list(set(device_names))

        for name in unique_names:
            stats = device_stats[name]
            stats['frequency'] += 1

            if len(stats['example_poem_ids']) < 5:
                stats['example_poem_ids'].append(poem_id)

            # Co-occurrence with other devices in same poem
            for other_name in unique_names:
                if other_name != name:
                    stats['co_occurring_devices'][other_name] += 1

    # Convert to serializable format
    result = {}
    for name, stats in device_stats.items():
        result[name] = {
            'frequency': stats['frequency'],
            'co_occurring_devices': dict(stats['co_occurring_devices']),
            'example_poem_ids': stats['example_poem_ids']
        }

    logger.info(f"Built index with {len(result)} unique devices")
    return result


# -----------------------------------------------------------------------------
# Step 3: Build Thematic Clusters
# -----------------------------------------------------------------------------

def classify_season(row: pd.Series) -> str | None:
    """Classify poem by season based on seasonal_context and text."""
    seasonal_context = row.get('seasonal_context', '')
    text = row.get('text', '')

    if pd.isna(seasonal_context):
        seasonal_context = ''
    if pd.isna(text):
        text = ''

    combined = f"{seasonal_context} {text}"

    # Count keyword matches for each season
    scores = {}
    for season, keywords in SEASON_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[season] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


def classify_theme(row: pd.Series) -> list[str]:
    """Classify poem themes based on interpretation and text."""
    interpretation = row.get('interpretation', '')
    text = row.get('text', '')
    emotional_tone = row.get('emotional_tone', '')

    if pd.isna(interpretation):
        interpretation = ''
    if pd.isna(text):
        text = ''
    if pd.isna(emotional_tone):
        emotional_tone = ''

    combined = f"{interpretation} {text} {emotional_tone}"

    themes = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            themes.append(theme)

    return themes if themes else ['その他']


def build_thematic_clusters(literary_df: pd.DataFrame) -> dict:
    """
    Build thematic clusters grouping poems by season and theme.

    Returns:
        {
            "by_season": {
                "春": ["poem_id_1", ...],
                "夏": [...],
                ...
            },
            "by_theme": {
                "恋": ["poem_id_1", ...],
                ...
            },
            "season_stats": {"春": count, ...},
            "theme_stats": {"恋": count, ...}
        }
    """
    logger.info("Building thematic clusters...")

    by_season = defaultdict(list)
    by_theme = defaultdict(list)

    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']

        # Season classification
        season = classify_season(row)
        if season:
            by_season[season].append(poem_id)
        else:
            by_season['季節不明'].append(poem_id)

        # Theme classification
        themes = classify_theme(row)
        for theme in themes:
            by_theme[theme].append(poem_id)

    result = {
        'by_season': {k: v for k, v in sorted(by_season.items())},
        'by_theme': {k: v for k, v in sorted(by_theme.items())},
        'season_stats': {k: len(v) for k, v in sorted(by_season.items())},
        'theme_stats': {k: len(v) for k, v in sorted(by_theme.items())}
    }

    logger.info(f"Season distribution: {result['season_stats']}")
    logger.info(f"Theme distribution: {result['theme_stats']}")

    return result


# -----------------------------------------------------------------------------
# Step 4: Grammar-Literary Correlation
# -----------------------------------------------------------------------------

def build_grammar_literary_correlation(
    literary_df: pd.DataFrame,
    grammar_df: pd.DataFrame | None
) -> dict:
    """
    Build correlation matrix between grammar points and literary devices.

    Returns:
        {
            "device_to_grammar": {
                "掛詞": {"particle_no": 50, "particle_wa": 45, ...}
            },
            "grammar_to_device": {
                "particle_no": {"掛詞": 50, "縁語": 30, ...}
            }
        }
    """
    logger.info("Building grammar-literary correlation...")

    if grammar_df is None or len(grammar_df) == 0:
        logger.warning("No grammar data available, skipping correlation")
        return {
            'device_to_grammar': {},
            'grammar_to_device': {},
            'note': 'No grammar annotations available'
        }

    # Build poem -> canonical_ids mapping from grammar data
    poem_to_grammar = {}
    for _, row in grammar_df.iterrows():
        poem_id = row['poem_id']
        grammar_points = row.get('grammar_points', [])

        # Handle numpy arrays and None values
        if grammar_points is None:
            continue
        if isinstance(grammar_points, np.ndarray):
            grammar_points = grammar_points.tolist()
        if not isinstance(grammar_points, list) or len(grammar_points) == 0:
            continue

        canonical_ids = set()
        for gp in grammar_points:
            if isinstance(gp, dict) and 'canonical_id' in gp:
                canonical_ids.add(gp['canonical_id'])
        if canonical_ids:
            poem_to_grammar[poem_id] = canonical_ids

    logger.info(f"Found grammar data for {len(poem_to_grammar)} poems")

    # Build correlation counters
    device_to_grammar = defaultdict(Counter)
    grammar_to_device = defaultdict(Counter)

    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']

        # Get devices for this poem
        devices = parse_poetic_devices(row.get('poetic_devices'))
        device_names = set(extract_device_names(devices))

        # Get grammar points for this poem
        grammar_ids = poem_to_grammar.get(poem_id, set())

        if not device_names or not grammar_ids:
            continue

        # Count co-occurrences
        for device in device_names:
            for grammar_id in grammar_ids:
                device_to_grammar[device][grammar_id] += 1
                grammar_to_device[grammar_id][device] += 1

    result = {
        'device_to_grammar': {k: dict(v) for k, v in device_to_grammar.items()},
        'grammar_to_device': {k: dict(v) for k, v in grammar_to_device.items()}
    }

    logger.info(f"Built correlation for {len(device_to_grammar)} devices")
    return result


# -----------------------------------------------------------------------------
# Step 5: Literary Difficulty Scoring
# -----------------------------------------------------------------------------

def compute_literary_difficulty(literary_df: pd.DataFrame) -> dict:
    """
    Compute literary difficulty score for each poem.

    Factors:
    - Number of poetic devices (more = harder)
    - Complexity of devices (掛詞, 本歌取り harder than 体言止め)
    - Allusion density
    - Interpretation length (longer = more complex meaning)

    Returns:
        {
            "poem_id": {
                "literary_difficulty": float (0-1),
                "device_count": int,
                "has_complex_device": bool,
                "interpretation_complexity": float
            }
        }
    """
    logger.info("Computing literary difficulty scores...")

    # Device complexity weights (higher = more sophisticated)
    DEVICE_WEIGHTS = {
        '掛詞': 0.4,          # Pivot word - requires understanding double meanings
        '本歌取り': 0.5,      # Allusive variation - requires knowing source poem
        '序詞': 0.35,         # Preface - complex structural device
        '枕詞': 0.25,         # Pillow word - conventional but requires knowledge
        '縁語': 0.2,          # Associated words - moderate
        '見立て': 0.25,       # Conceit/metaphor
        '体言止め': 0.1,      # Noun ending - simpler
        '対句': 0.15,         # Parallelism
        '擬人法': 0.15,       # Personification
        '倒置': 0.1,          # Inversion
        '反復': 0.1,          # Repetition
    }
    DEFAULT_WEIGHT = 0.15

    COMPLEX_DEVICES = {'掛詞', '本歌取り', '序詞'}

    results = {}

    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']

        # Get devices
        devices = parse_poetic_devices(row.get('poetic_devices'))
        device_names = extract_device_names(devices)

        # Device count score (normalized, max ~10 devices)
        device_count = len(device_names)
        device_count_score = min(device_count / 5, 1.0)

        # Device complexity score
        device_weights = [DEVICE_WEIGHTS.get(d, DEFAULT_WEIGHT) for d in device_names]
        complexity_score = sum(device_weights) / max(len(device_weights), 1)

        # Has complex device
        has_complex = any(d in COMPLEX_DEVICES for d in device_names)

        # Interpretation complexity (by length, normalized)
        interpretation = row.get('interpretation', '')
        if pd.isna(interpretation):
            interpretation = ''
        interp_len = len(interpretation)
        interp_score = min(interp_len / 500, 1.0)  # Normalize to ~500 chars

        # Allusion density
        allusions = row.get('allusions', [])
        if isinstance(allusions, np.ndarray):
            allusions = allusions.tolist() if allusions is not None else []
        elif pd.isna(allusions) if isinstance(allusions, float) else False:
            allusions = []
        allusion_score = min(len(allusions) / 3, 1.0) if allusions else 0.0

        # Combined score (weighted average)
        literary_difficulty = (
            0.3 * device_count_score +
            0.35 * complexity_score +
            0.2 * interp_score +
            0.15 * allusion_score
        )

        results[poem_id] = {
            'literary_difficulty': round(literary_difficulty, 3),
            'device_count': device_count,
            'has_complex_device': has_complex,
            'interpretation_complexity': round(interp_score, 3)
        }

    # Log distribution
    difficulties = [v['literary_difficulty'] for v in results.values()]
    logger.info(f"Literary difficulty range: {min(difficulties):.3f} - {max(difficulties):.3f}")
    logger.info(f"Mean difficulty: {sum(difficulties)/len(difficulties):.3f}")

    return results


# -----------------------------------------------------------------------------
# Step 5b: Chinese-Adjusted Literary Difficulty
# -----------------------------------------------------------------------------

def compute_chinese_adjusted_literary_difficulty(
    literary_difficulty: dict,
    literary_df: pd.DataFrame
) -> dict:
    """
    Adjust literary difficulty for Chinese learners.

    Some devices (対句, 擬人法) are familiar from Chinese poetry,
    so they should contribute less to difficulty.
    """
    logger.info("Computing Chinese-adjusted literary difficulty...")

    results = {}

    for poem_id, stats in literary_difficulty.items():
        # Get devices for this poem
        row = literary_df[literary_df['poem_id'] == poem_id]
        if row.empty:
            results[poem_id] = stats.copy()
            results[poem_id]['chinese_adjustment'] = 0.0
            results[poem_id]['chinese_adjusted_literary'] = stats['literary_difficulty']
            continue

        devices = parse_poetic_devices(row.iloc[0].get('poetic_devices'))
        device_names = extract_device_names(devices)

        # Calculate Chinese familiarity bonus
        adjustment = sum(
            CHINESE_FAMILIAR_DEVICES.get(d, 0)
            for d in device_names
        )

        adjusted = max(0.0, stats['literary_difficulty'] + adjustment)

        results[poem_id] = stats.copy()
        results[poem_id]['chinese_adjustment'] = round(adjustment, 4)
        results[poem_id]['chinese_adjusted_literary'] = round(adjusted, 4)

    adjustments = [v['chinese_adjustment'] for v in results.values()]
    logger.info(f"Chinese adjustment range: {min(adjustments):.3f} to {max(adjustments):.3f}")

    return results


# -----------------------------------------------------------------------------
# Step 5c: Combined Difficulty Score (Grammar + Literary)
# -----------------------------------------------------------------------------

def compute_combined_difficulty(
    literary_difficulty: dict,
    chinese_difficulty_path: Path
) -> dict:
    """
    Combine grammar difficulty with literary difficulty for unified curriculum scoring.

    Formula: combined = 0.5 * chinese_adjusted_grammar + 0.5 * chinese_adjusted_literary
    """
    logger.info("Computing combined difficulty scores...")

    # Load Chinese-adjusted grammar difficulty
    grammar_difficulty = {}
    if chinese_difficulty_path.exists():
        with open(chinese_difficulty_path, 'r', encoding='utf-8') as f:
            grammar_data = json.load(f)
        for entry in grammar_data:
            grammar_difficulty[entry['poem_id']] = entry['chinese_adjusted_difficulty']
        logger.info(f"Loaded grammar difficulty for {len(grammar_difficulty)} poems")
    else:
        logger.warning("Chinese difficulty file not found, using original literary only")

    results = {}

    for poem_id, lit_stats in literary_difficulty.items():
        grammar_diff = grammar_difficulty.get(poem_id, 0.5)
        literary_diff = lit_stats.get('chinese_adjusted_literary', lit_stats['literary_difficulty'])

        # Weighted combination
        combined = 0.5 * grammar_diff + 0.5 * literary_diff

        results[poem_id] = {
            'poem_id': poem_id,
            'grammar_difficulty': round(grammar_diff, 4),
            'literary_difficulty': round(literary_diff, 4),
            'combined_difficulty': round(combined, 4),
        }

    combined_scores = [v['combined_difficulty'] for v in results.values()]
    logger.info(f"Combined difficulty range: {min(combined_scores):.3f} - {max(combined_scores):.3f}")

    return results


# -----------------------------------------------------------------------------
# Step 5d: Device Progression Path
# -----------------------------------------------------------------------------

def build_device_progression(
    literary_index: dict,
    literary_df: pd.DataFrame,
    literary_difficulty: dict
) -> dict:
    """
    Build device learning progression for curriculum.

    Identifies:
    - Gateway poems (only 1 device - good for teaching)
    - Device frequency ordering (common → rare)
    - Recommended teaching sequence
    """
    logger.info("Building device progression paths...")

    # Get poems per device with their difficulties
    device_poems = {}
    for device, stats in literary_index.items():
        poem_ids = stats.get('example_poem_ids', [])
        # Get more poems if needed
        device_poems[device] = {
            'frequency': stats['frequency'],
            'poems': poem_ids,
        }

    # Find gateway poems (poems with exactly 1 device)
    gateway_poems = {}
    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']
        devices = parse_poetic_devices(row.get('poetic_devices'))
        device_names = extract_device_names(devices)

        if len(device_names) == 1:
            device = device_names[0]
            if device not in gateway_poems:
                gateway_poems[device] = []

            diff = literary_difficulty.get(poem_id, {}).get('literary_difficulty', 0.5)
            gateway_poems[device].append({
                'poem_id': poem_id,
                'difficulty': diff
            })

    # Sort gateways by difficulty (easiest first)
    for device in gateway_poems:
        gateway_poems[device].sort(key=lambda x: x['difficulty'])

    # Build recommended teaching sequence
    # Sort by frequency (common first), then by complexity
    DEVICE_COMPLEXITY_ORDER = {
        '体言止め': 1,  # Simple
        '反復': 2,
        '対句': 2,
        '倒置': 2,
        '縁語': 3,
        '枕詞': 3,
        '擬人法': 3,
        '見立て': 4,
        '掛詞': 5,
        '序詞': 5,
        '本歌取り': 6,  # Complex
        '係結': 4,
    }

    devices_sorted = sorted(
        literary_index.keys(),
        key=lambda d: (
            DEVICE_COMPLEXITY_ORDER.get(d, 4),  # Complexity first
            -literary_index[d]['frequency']  # Then frequency (higher = earlier)
        )
    )

    progression = {
        'teaching_sequence': devices_sorted,
        'gateway_poems': gateway_poems,
        'device_by_frequency': sorted(
            [(d, literary_index[d]['frequency']) for d in literary_index],
            key=lambda x: -x[1]
        ),
        'complexity_tiers': {
            'beginner': [d for d in devices_sorted if DEVICE_COMPLEXITY_ORDER.get(d, 4) <= 2],
            'intermediate': [d for d in devices_sorted if 2 < DEVICE_COMPLEXITY_ORDER.get(d, 4) <= 4],
            'advanced': [d for d in devices_sorted if DEVICE_COMPLEXITY_ORDER.get(d, 4) > 4],
        }
    }

    logger.info(f"Built progression with {len(devices_sorted)} devices")
    logger.info(f"Gateway poems found for {len(gateway_poems)} devices")

    return progression


# -----------------------------------------------------------------------------
# Step 5e: Kakarimusubi-Literary Correlation
# -----------------------------------------------------------------------------

def analyze_kakarimusubi_literary(
    literary_df: pd.DataFrame,
    grammar_df: pd.DataFrame,
    kakarimusubi_path: Path
) -> dict:
    """
    Correlate kakarimusubi patterns with literary effects.

    Analyzes which kakari particles associate with which emotional tones
    and literary devices.
    """
    logger.info("Analyzing kakarimusubi-literary correlation...")

    if not kakarimusubi_path.exists():
        logger.warning("Kakarimusubi patterns file not found")
        return {'note': 'No kakarimusubi data available'}

    # Load kakarimusubi patterns
    kakari_df = pd.read_csv(kakarimusubi_path)

    # Get kakari particles per poem
    poem_kakari = {}
    for _, row in kakari_df.iterrows():
        poem_id = row['poem_id']
        particle = row['kakari_particle']
        if poem_id not in poem_kakari:
            poem_kakari[poem_id] = []
        poem_kakari[poem_id].append(particle)

    # Correlate with emotional tones
    particle_tones = defaultdict(Counter)
    particle_devices = defaultdict(Counter)

    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']
        if poem_id not in poem_kakari:
            continue

        particles = poem_kakari[poem_id]
        tone = row.get('emotional_tone', '')
        if pd.isna(tone):
            tone = ''

        devices = parse_poetic_devices(row.get('poetic_devices'))
        device_names = extract_device_names(devices)

        for particle in particles:
            if tone:
                # Extract key tone words (first 4-8 chars often contain the essence)
                tone_key = tone[:8] if len(tone) > 8 else tone
                particle_tones[particle][tone_key] += 1

            for device in device_names:
                particle_devices[particle][device] += 1

    results = {
        'particle_emotional_tones': {k: dict(v.most_common(10)) for k, v in particle_tones.items()},
        'particle_literary_devices': {k: dict(v) for k, v in particle_devices.items()},
        'poems_with_kakarimusubi': len(poem_kakari),
    }

    logger.info(f"Analyzed {len(poem_kakari)} poems with kakarimusubi")

    return results


# -----------------------------------------------------------------------------
# Step 5f: Confusables Integration
# -----------------------------------------------------------------------------

def analyze_confusables_impact(
    literary_df: pd.DataFrame,
    grammar_df: pd.DataFrame,
    confusables_path: Path
) -> dict:
    """
    Analyze how grammar confusables affect literary difficulty.

    Poems with many homograph grammar points are harder to parse,
    which affects literary comprehension.
    """
    logger.info("Analyzing confusables impact on literary difficulty...")

    if not confusables_path.exists():
        logger.warning("Confusables file not found")
        return {'note': 'No confusables data available'}

    with open(confusables_path, 'r', encoding='utf-8') as f:
        confusables_data = json.load(f)

    # Get high-frequency confusable surfaces
    confusable_surfaces = set(confusables_data.get('confusables', {}).keys())

    # Count confusables per poem
    poem_confusable_counts = {}

    for _, row in grammar_df.iterrows():
        poem_id = row['poem_id']
        grammar_points = row.get('grammar_points', [])

        if grammar_points is None:
            continue
        if isinstance(grammar_points, np.ndarray):
            grammar_points = grammar_points.tolist()

        confusable_count = 0
        for gp in grammar_points:
            if isinstance(gp, dict):
                surface = gp.get('surface', '')
                if surface in confusable_surfaces:
                    confusable_count += 1

        poem_confusable_counts[poem_id] = confusable_count

    # Correlate with literary difficulty
    literary_by_confusables = defaultdict(list)
    for _, row in literary_df.iterrows():
        poem_id = row['poem_id']
        confusable_count = poem_confusable_counts.get(poem_id, 0)

        # Get literary metrics
        devices = parse_poetic_devices(row.get('poetic_devices'))
        device_count = len(extract_device_names(devices))

        bucket = min(confusable_count, 5)  # 0, 1, 2, 3, 4, 5+
        literary_by_confusables[bucket].append(device_count)

    # Calculate averages
    confusable_device_correlation = {}
    for bucket, device_counts in literary_by_confusables.items():
        avg = sum(device_counts) / len(device_counts) if device_counts else 0
        confusable_device_correlation[f"{bucket}_confusables"] = {
            'avg_devices': round(avg, 2),
            'poem_count': len(device_counts)
        }

    results = {
        'confusable_device_correlation': confusable_device_correlation,
        'total_confusable_surfaces': len(confusable_surfaces),
        'poems_analyzed': len(poem_confusable_counts),
    }

    logger.info(f"Analyzed confusables impact on {len(poem_confusable_counts)} poems")

    return results


# -----------------------------------------------------------------------------
# Step 7: Generate Visualizations
# -----------------------------------------------------------------------------

def generate_visualizations(
    literary_index: dict,
    thematic_clusters: dict,
    literary_difficulty: dict,
    combined_difficulty: dict,
    output_dir: Path
):
    """Generate visualization charts for literary analysis."""
    logger.info("Generating visualizations...")

    setup_plotting()

    # 1. Device frequency bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_devices = sorted(literary_index.items(), key=lambda x: x[1]['frequency'], reverse=True)[:15]
    devices = [d[0] for d in sorted_devices]
    freqs = [d[1]['frequency'] for d in sorted_devices]

    ax.barh(devices, freqs, color='steelblue')
    ax.set_xlabel('Frequency')
    ax.set_title('Top 15 Poetic Devices by Frequency')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "device_frequency.png", dpi=150)
    plt.close()

    # 2. Season distribution pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    season_stats = thematic_clusters.get('season_stats', {})
    if season_stats:
        seasons = list(season_stats.keys())
        counts = list(season_stats.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(seasons)))
        ax.pie(counts, labels=seasons, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Seasonal Distribution of Poems')
    plt.tight_layout()
    plt.savefig(output_dir / "season_distribution.png", dpi=150)
    plt.close()

    # 3. Literary difficulty histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    difficulties = [v['literary_difficulty'] for v in literary_difficulty.values()]
    ax.hist(difficulties, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Literary Difficulty')
    ax.set_ylabel('Number of Poems')
    ax.set_title('Literary Difficulty Distribution')
    ax.axvline(np.mean(difficulties), color='red', linestyle='--', label=f'Mean: {np.mean(difficulties):.2f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "literary_difficulty_histogram.png", dpi=150)
    plt.close()

    # 4. Combined difficulty comparison
    if combined_difficulty:
        fig, ax = plt.subplots(figsize=(10, 6))
        grammar = [v['grammar_difficulty'] for v in combined_difficulty.values()]
        literary = [v['literary_difficulty'] for v in combined_difficulty.values()]
        combined = [v['combined_difficulty'] for v in combined_difficulty.values()]

        bins = np.linspace(0, 1, 21)
        ax.hist(grammar, bins=bins, alpha=0.5, label='Grammar', color='steelblue')
        ax.hist(literary, bins=bins, alpha=0.5, label='Literary', color='coral')
        ax.hist(combined, bins=bins, alpha=0.5, label='Combined', color='green')
        ax.set_xlabel('Difficulty Score')
        ax.set_ylabel('Number of Poems')
        ax.set_title('Grammar vs Literary vs Combined Difficulty')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "difficulty_comparison.png", dpi=150)
        plt.close()

    # 5. Device co-occurrence heatmap
    top_devices = [d[0] for d in sorted_devices[:10]]
    n = len(top_devices)
    if n >= 2:
        co_matrix = np.zeros((n, n))
        for i, d1 in enumerate(top_devices):
            for j, d2 in enumerate(top_devices):
                if i != j:
                    co_matrix[i, j] = literary_index[d1].get('co_occurring_devices', {}).get(d2, 0)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(co_matrix, cmap='YlOrRd')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(top_devices, rotation=45, ha='right')
        ax.set_yticklabels(top_devices)
        ax.set_title('Device Co-occurrence Matrix')
        plt.colorbar(im, label='Co-occurrence Count')
        plt.tight_layout()
        plt.savefig(output_dir / "device_cooccurrence_heatmap.png", dpi=150)
        plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


# -----------------------------------------------------------------------------
# Step 8: Generate Report
# -----------------------------------------------------------------------------

def generate_analysis_report(
    literary_index: dict,
    thematic_clusters: dict,
    grammar_correlation: dict,
    literary_difficulty: dict,
    output_path: Path
):
    """Generate human-readable analysis report."""
    logger.info("Generating analysis report...")

    lines = [
        "# Literary Corpus Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Total poems analyzed: {len(literary_difficulty)}",
        "",
        "## Poetic Device Distribution",
        "",
        "| Device | Frequency | Example Poems |",
        "|--------|-----------|---------------|",
    ]

    # Sort devices by frequency
    sorted_devices = sorted(
        literary_index.items(),
        key=lambda x: x[1]['frequency'],
        reverse=True
    )

    for device, stats in sorted_devices[:15]:
        freq = stats['frequency']
        examples = ", ".join(stats['example_poem_ids'][:3])
        lines.append(f"| {device} | {freq} | {examples} |")

    if len(sorted_devices) > 15:
        lines.append(f"| ... | ... | ... |")
        lines.append(f"*({len(sorted_devices)} total devices)*")

    lines.extend([
        "",
        "## Device Co-occurrence (Top Pairs)",
        "",
        "| Device 1 | Device 2 | Count |",
        "|----------|----------|-------|",
    ])

    # Find top co-occurring pairs
    co_pairs = []
    for device, stats in literary_index.items():
        for other, count in stats.get('co_occurring_devices', {}).items():
            if device < other:  # Avoid duplicates
                co_pairs.append((device, other, count))

    co_pairs.sort(key=lambda x: x[2], reverse=True)
    for d1, d2, count in co_pairs[:10]:
        lines.append(f"| {d1} | {d2} | {count} |")

    lines.extend([
        "",
        "## Seasonal Distribution",
        "",
    ])

    for season, count in sorted(thematic_clusters.get('season_stats', {}).items()):
        pct = count / len(literary_difficulty) * 100
        lines.append(f"- **{season}**: {count} poems ({pct:.1f}%)")

    lines.extend([
        "",
        "## Theme Distribution",
        "",
    ])

    for theme, count in sorted(
        thematic_clusters.get('theme_stats', {}).items(),
        key=lambda x: x[1],
        reverse=True
    ):
        pct = count / len(literary_difficulty) * 100
        lines.append(f"- **{theme}**: {count} poems ({pct:.1f}%)")

    lines.extend([
        "",
        "## Literary Difficulty Distribution",
        "",
    ])

    difficulties = [v['literary_difficulty'] for v in literary_difficulty.values()]
    lines.append(f"- **Range**: {min(difficulties):.3f} - {max(difficulties):.3f}")
    lines.append(f"- **Mean**: {sum(difficulties)/len(difficulties):.3f}")

    # Tier distribution
    tiers = {
        'Tier 1 (0.0-0.2)': sum(1 for d in difficulties if d < 0.2),
        'Tier 2 (0.2-0.4)': sum(1 for d in difficulties if 0.2 <= d < 0.4),
        'Tier 3 (0.4-0.6)': sum(1 for d in difficulties if 0.4 <= d < 0.6),
        'Tier 4 (0.6-0.8)': sum(1 for d in difficulties if 0.6 <= d < 0.8),
        'Tier 5 (0.8-1.0)': sum(1 for d in difficulties if d >= 0.8),
    }

    lines.append("")
    for tier, count in tiers.items():
        lines.append(f"- **{tier}**: {count} poems")

    # Grammar-Literary correlation highlights
    if grammar_correlation.get('device_to_grammar'):
        lines.extend([
            "",
            "## Grammar-Literary Correlations",
            "",
            "Top grammar points co-occurring with each major device:",
            "",
        ])

        for device in ['掛詞', '縁語', '枕詞', '体言止め']:
            if device in grammar_correlation['device_to_grammar']:
                grammar_counts = grammar_correlation['device_to_grammar'][device]
                top_grammar = sorted(grammar_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                grammar_list = ", ".join(f"{g}({c})" for g, c in top_grammar)
                lines.append(f"- **{device}**: {grammar_list}")

    lines.extend([
        "",
        "---",
        "",
        "## Data Files Generated",
        "",
        "- `literary_index.json` - Device frequency and co-occurrence",
        "- `thematic_clusters.json` - Poems grouped by season/theme",
        "- `grammar_literary_correlation.json` - Grammar↔Literary mapping",
        "- `literary_difficulty.json` - Literary complexity scores",
        "",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    logger.info(f"Saved report to {output_path}")


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def analyze_literary_corpus(
    literary_input: Path,
    grammar_input: Path | None,
    output_dir: Path,
    chinese_difficulty_path: Path = DEFAULT_CHINESE_DIFFICULTY,
    confusables_path: Path = DEFAULT_CONFUSABLES,
    kakarimusubi_path: Path = DEFAULT_KAKARIMUSUBI
):
    """Run the full literary analysis pipeline with all improvements."""

    # Load literary data
    logger.info(f"Loading literary data from {literary_input}...")
    literary_df = pd.read_parquet(literary_input)
    logger.info(f"Loaded {len(literary_df)} poems with literary annotations")

    # Load grammar data (optional)
    grammar_df = None
    if grammar_input and grammar_input.exists():
        logger.info(f"Loading grammar data from {grammar_input}...")
        grammar_df = pd.read_parquet(grammar_input)
        logger.info(f"Loaded {len(grammar_df)} poems with grammar annotations")
    else:
        logger.warning("No grammar data available, some analyses will be skipped")

    # =========================================================================
    # Core Analysis Steps
    # =========================================================================
    print("\n" + "="*60)
    print("Step 1: Building literary index...")
    literary_index = build_literary_index(literary_df)

    print("\n" + "="*60)
    print("Step 2: Building thematic clusters...")
    thematic_clusters = build_thematic_clusters(literary_df)

    print("\n" + "="*60)
    print("Step 3: Building grammar-literary correlation...")
    grammar_correlation = build_grammar_literary_correlation(literary_df, grammar_df)

    print("\n" + "="*60)
    print("Step 4: Computing literary difficulty...")
    literary_difficulty = compute_literary_difficulty(literary_df)

    # =========================================================================
    # New Analysis Steps (Improvements)
    # =========================================================================
    print("\n" + "="*60)
    print("Step 5a: Computing Chinese-adjusted literary difficulty...")
    chinese_literary = compute_chinese_adjusted_literary_difficulty(literary_difficulty, literary_df)

    print("\n" + "="*60)
    print("Step 5b: Computing combined difficulty...")
    combined_difficulty = compute_combined_difficulty(chinese_literary, chinese_difficulty_path)

    print("\n" + "="*60)
    print("Step 5c: Building device progression...")
    device_progression = build_device_progression(literary_index, literary_df, literary_difficulty)

    print("\n" + "="*60)
    print("Step 5d: Analyzing kakarimusubi-literary correlation...")
    kakarimusubi_literary = {}
    if grammar_df is not None:
        kakarimusubi_literary = analyze_kakarimusubi_literary(literary_df, grammar_df, kakarimusubi_path)

    print("\n" + "="*60)
    print("Step 5e: Analyzing confusables impact...")
    confusables_impact = {}
    if grammar_df is not None:
        confusables_impact = analyze_confusables_impact(literary_df, grammar_df, confusables_path)

    # =========================================================================
    # Save outputs
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "literary_index.json", 'w', encoding='utf-8') as f:
        json.dump(literary_index, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved literary_index.json")

    with open(output_dir / "thematic_clusters.json", 'w', encoding='utf-8') as f:
        json.dump(thematic_clusters, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved thematic_clusters.json")

    with open(output_dir / "grammar_literary_correlation.json", 'w', encoding='utf-8') as f:
        json.dump(grammar_correlation, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved grammar_literary_correlation.json")

    with open(output_dir / "literary_difficulty.json", 'w', encoding='utf-8') as f:
        json.dump(chinese_literary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved literary_difficulty.json (with Chinese adjustment)")

    with open(output_dir / "combined_difficulty.json", 'w', encoding='utf-8') as f:
        json.dump(combined_difficulty, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved combined_difficulty.json")

    with open(output_dir / "device_progression.json", 'w', encoding='utf-8') as f:
        json.dump(device_progression, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved device_progression.json")

    if kakarimusubi_literary:
        with open(output_dir / "kakarimusubi_literary.json", 'w', encoding='utf-8') as f:
            json.dump(kakarimusubi_literary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved kakarimusubi_literary.json")

    if confusables_impact:
        with open(output_dir / "confusables_impact.json", 'w', encoding='utf-8') as f:
            json.dump(confusables_impact, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved confusables_impact.json")

    # =========================================================================
    # Generate visualizations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 6: Generating visualizations...")
    generate_visualizations(
        literary_index,
        thematic_clusters,
        chinese_literary,
        combined_difficulty,
        output_dir
    )

    # =========================================================================
    # Generate report
    # =========================================================================
    print("\n" + "="*60)
    print("Step 7: Generating report...")
    generate_analysis_report(
        literary_index,
        thematic_clusters,
        grammar_correlation,
        chinese_literary,
        output_dir / "analysis_report.md"
    )

    return {
        'literary_index': literary_index,
        'thematic_clusters': thematic_clusters,
        'grammar_correlation': grammar_correlation,
        'literary_difficulty': chinese_literary,
        'combined_difficulty': combined_difficulty,
        'device_progression': device_progression,
        'kakarimusubi_literary': kakarimusubi_literary,
        'confusables_impact': confusables_impact
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze literary annotations in the poetry corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_literary_corpus.py
  python scripts/analyze_literary_corpus.py --grammar-input data/annotated/poems.parquet
  python scripts/analyze_literary_corpus.py --output-dir data/literary_v2
        """
    )
    parser.add_argument(
        "--literary-input",
        type=Path,
        default=DEFAULT_LITERARY_INPUT,
        help=f"Input parquet with literary annotations (default: {DEFAULT_LITERARY_INPUT})"
    )
    parser.add_argument(
        "--grammar-input",
        type=Path,
        default=DEFAULT_GRAMMAR_INPUT,
        help=f"Input parquet with grammar annotations for correlation (default: {DEFAULT_GRAMMAR_INPUT})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for analysis files (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()

    # Validate input
    if not args.literary_input.exists():
        print(f"ERROR: Literary input file not found: {args.literary_input}")
        sys.exit(1)

    # Run analysis
    results = analyze_literary_corpus(
        literary_input=args.literary_input,
        grammar_input=args.grammar_input,
        output_dir=args.output_dir
    )

    print(f"\nLiterary corpus analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"\nSummary:")
    print(f"  - Poetic devices indexed: {len(results['literary_index'])}")
    print(f"  - Poems with difficulty scores: {len(results['literary_difficulty'])}")
    print(f"  - Seasonal clusters: {len(results['thematic_clusters']['by_season'])}")
    print(f"  - Theme clusters: {len(results['thematic_clusters']['by_theme'])}")
    print(f"\nView report: cat {args.output_dir}/analysis_report.md")


if __name__ == "__main__":
    main()
