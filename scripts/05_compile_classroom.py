#!/usr/bin/env python3
"""
05_compile_classroom.py - Bundle all content into deployable classroom.db.

Compiles lessons, curriculum, and poems into a single SQLite database
for runtime serving.

Usage:
  python scripts/05_compile_classroom.py
  python scripts/05_compile_classroom.py --curriculum data/curriculum_refined --output data/classroom.db
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SQLite Schema
# -----------------------------------------------------------------------------

SCHEMA = """
-- Units table
CREATE TABLE IF NOT EXISTS units (
    id TEXT PRIMARY KEY,
    title TEXT,
    theme TEXT,
    position INTEGER NOT NULL
);

-- Lessons table
CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    unit_id TEXT NOT NULL REFERENCES units(id),
    title TEXT NOT NULL,
    summary TEXT,
    grammar_point TEXT NOT NULL,
    position INTEGER NOT NULL,
    difficulty_tier INTEGER,
    content JSON NOT NULL,
    prerequisites JSON NOT NULL DEFAULT '[]'
);

-- Poems table
CREATE TABLE IF NOT EXISTS poems (
    id TEXT PRIMARY KEY,
    source TEXT,
    text TEXT NOT NULL,
    reading_hiragana TEXT,
    reading_romaji TEXT,
    author TEXT,
    collection TEXT,
    difficulty_score REAL,
    annotations JSON
);

-- Grammar points table
CREATE TABLE IF NOT EXISTS grammar_points (
    id TEXT PRIMARY KEY,
    category TEXT,
    frequency INTEGER,
    avg_difficulty REAL,
    senses JSON,
    lesson_id TEXT REFERENCES lessons(id)
);

-- Lesson-poem mapping (which poems are used in which lessons)
CREATE TABLE IF NOT EXISTS lesson_poems (
    lesson_id TEXT NOT NULL REFERENCES lessons(id),
    poem_id TEXT NOT NULL REFERENCES poems(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (lesson_id, poem_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_lessons_unit ON lessons(unit_id);
CREATE INDEX IF NOT EXISTS idx_lessons_grammar ON lessons(grammar_point);
CREATE INDEX IF NOT EXISTS idx_grammar_lesson ON grammar_points(lesson_id);
CREATE INDEX IF NOT EXISTS idx_lesson_poems_lesson ON lesson_poems(lesson_id);
CREATE INDEX IF NOT EXISTS idx_lesson_poems_poem ON lesson_poems(poem_id);

-- Metadata table
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_curriculum(curriculum_dir: Path) -> tuple[dict, dict]:
    """Load curriculum data."""
    lesson_graph_path = curriculum_dir / "lesson_graph.json"
    grammar_index_path = curriculum_dir / "grammar_index.json"

    with open(lesson_graph_path, "r", encoding="utf-8") as f:
        lesson_graph = json.load(f)

    grammar_index = {}
    if grammar_index_path.exists():
        with open(grammar_index_path, "r", encoding="utf-8") as f:
            grammar_index = json.load(f)

    return lesson_graph, grammar_index


def load_lessons(lessons_dir: Path) -> dict[str, dict]:
    """Load all lesson JSON files."""
    lessons = {}
    for lesson_file in lessons_dir.glob("lesson_*.json"):
        with open(lesson_file, "r", encoding="utf-8") as f:
            lesson = json.load(f)
            lessons[lesson["lesson_id"]] = lesson
    return lessons


def load_poems(poems_path: Path) -> pd.DataFrame:
    """Load annotated poems."""
    return pd.read_parquet(poems_path)


# -----------------------------------------------------------------------------
# Database Population
# -----------------------------------------------------------------------------

def create_database(db_path: Path) -> sqlite3.Connection:
    """Create database and schema."""
    if db_path.exists():
        db_path.unlink()
        logger.info(f"Removed existing database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    logger.info(f"Created database schema: {db_path}")
    return conn


def populate_units(conn: sqlite3.Connection, lesson_graph: dict):
    """Populate units table."""
    units = lesson_graph.get("units", [])
    for position, unit in enumerate(units, 1):
        conn.execute(
            "INSERT INTO units (id, title, theme, position) VALUES (?, ?, ?, ?)",
            (unit["id"], unit.get("title"), unit.get("theme"), position)
        )
    conn.commit()
    logger.info(f"Inserted {len(units)} units")


def populate_lessons(
    conn: sqlite3.Connection,
    lesson_graph: dict,
    lessons_content: dict[str, dict]
):
    """Populate lessons table."""
    inserted = 0
    missing = []

    for unit in lesson_graph.get("units", []):
        unit_id = unit["id"]
        for position, lesson in enumerate(unit.get("lessons", []), 1):
            lesson_id = lesson["id"]
            content = lessons_content.get(lesson_id)

            if not content:
                missing.append(lesson_id)
                continue

            prerequisites = lesson.get("prerequisites", [])

            conn.execute(
                """INSERT INTO lessons
                   (id, unit_id, title, summary, grammar_point, position, difficulty_tier, content, prerequisites)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    lesson_id,
                    unit_id,
                    content.get("lesson_title", lesson_id),
                    content.get("lesson_summary"),
                    content.get("grammar_point", lesson.get("canonical_grammar_point")),
                    position,
                    lesson.get("difficulty_tier"),
                    json.dumps(content, ensure_ascii=False),
                    json.dumps(prerequisites)
                )
            )
            inserted += 1

    conn.commit()
    logger.info(f"Inserted {inserted} lessons")
    if missing:
        logger.warning(f"Missing lesson content for: {missing}")


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def populate_poems(conn: sqlite3.Connection, poems_df: pd.DataFrame):
    """Populate poems table."""
    inserted = 0

    for _, row in poems_df.iterrows():
        # Convert numpy types and prepare annotations
        annotations = {}
        for col in ["grammar_points", "vocabulary", "difficulty_factors", "token_readings"]:
            if col in row and row[col] is not None:
                val = row[col]
                val = convert_to_json_serializable(val)
                annotations[col] = val

        conn.execute(
            """INSERT INTO poems
               (id, source, text, reading_hiragana, reading_romaji, author, collection, difficulty_score, annotations)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.get("poem_id"),
                row.get("source"),
                row.get("text"),
                row.get("reading_hiragana"),
                row.get("reading_romaji"),
                row.get("author"),
                row.get("collection"),
                float(row.get("difficulty_score_computed", 0)) if row.get("difficulty_score_computed") else None,
                json.dumps(annotations, ensure_ascii=False) if annotations else None
            )
        )
        inserted += 1

    conn.commit()
    logger.info(f"Inserted {inserted} poems")


def populate_grammar_points(
    conn: sqlite3.Connection,
    grammar_index: dict,
    lesson_graph: dict
):
    """Populate grammar_points table."""
    # Build lesson lookup
    lesson_lookup = {}
    for unit in lesson_graph.get("units", []):
        for lesson in unit.get("lessons", []):
            gp = lesson.get("canonical_grammar_point")
            if gp:
                lesson_lookup[gp] = lesson["id"]

    entries = grammar_index.get("entries", {})
    inserted = 0

    for gp_id, data in entries.items():
        lesson_id = lesson_lookup.get(gp_id)

        conn.execute(
            """INSERT INTO grammar_points
               (id, category, frequency, avg_difficulty, senses, lesson_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                gp_id,
                data.get("category"),
                data.get("frequency"),
                data.get("avg_difficulty"),
                json.dumps(data.get("senses", []), ensure_ascii=False),
                lesson_id
            )
        )
        inserted += 1

    conn.commit()
    logger.info(f"Inserted {inserted} grammar points")


def populate_lesson_poems(
    conn: sqlite3.Connection,
    lessons_content: dict[str, dict]
):
    """Populate lesson_poems mapping table."""
    inserted = 0

    for lesson_id, content in lessons_content.items():
        poem_ids_seen = set()
        position = 0

        for step in content.get("teaching_sequence", []):
            if step.get("type") == "poem_presentation":
                poem_id = step.get("poem_id")
                if poem_id and poem_id not in poem_ids_seen:
                    poem_ids_seen.add(poem_id)
                    position += 1
                    conn.execute(
                        "INSERT OR IGNORE INTO lesson_poems (lesson_id, poem_id, position) VALUES (?, ?, ?)",
                        (lesson_id, poem_id, position)
                    )
                    inserted += 1

    conn.commit()
    logger.info(f"Inserted {inserted} lesson-poem mappings")


def populate_metadata(conn: sqlite3.Connection, stats: dict):
    """Populate metadata table."""
    for key, value in stats.items():
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if isinstance(value, (dict, list)) else str(value))
        )
    conn.commit()
    logger.info("Inserted metadata")


# -----------------------------------------------------------------------------
# Integrity Checks
# -----------------------------------------------------------------------------

def run_integrity_checks(conn: sqlite3.Connection) -> list[str]:
    """Run integrity checks on the compiled database."""
    issues = []

    # Check: All lesson prerequisites exist
    cursor = conn.execute("""
        SELECT l.id, l.prerequisites FROM lessons l
    """)
    for row in cursor:
        lesson_id, prereqs_json = row
        prereqs = json.loads(prereqs_json) if prereqs_json else []
        for prereq in prereqs:
            check = conn.execute("SELECT 1 FROM lessons WHERE id = ?", (prereq,)).fetchone()
            if not check:
                issues.append(f"Lesson {lesson_id} has missing prerequisite: {prereq}")

    # Check: All poems referenced in lessons exist
    cursor = conn.execute("""
        SELECT lp.lesson_id, lp.poem_id FROM lesson_poems lp
        LEFT JOIN poems p ON lp.poem_id = p.id
        WHERE p.id IS NULL
    """)
    for row in cursor:
        issues.append(f"Lesson {row[0]} references missing poem: {row[1]}")

    # Check: All units have at least one lesson
    cursor = conn.execute("""
        SELECT u.id FROM units u
        LEFT JOIN lessons l ON u.id = l.unit_id
        GROUP BY u.id
        HAVING COUNT(l.id) = 0
    """)
    for row in cursor:
        issues.append(f"Unit {row[0]} has no lessons")

    return issues


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def compute_stats(conn: sqlite3.Connection) -> dict:
    """Compute classroom statistics."""
    stats = {
        "compiled_at": datetime.now().isoformat(),
        "total_units": conn.execute("SELECT COUNT(*) FROM units").fetchone()[0],
        "total_lessons": conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
        "total_poems": conn.execute("SELECT COUNT(*) FROM poems").fetchone()[0],
        "total_grammar_points": conn.execute("SELECT COUNT(*) FROM grammar_points").fetchone()[0],
        "poems_in_lessons": conn.execute("SELECT COUNT(DISTINCT poem_id) FROM lesson_poems").fetchone()[0],
    }

    # Estimate study time (rough: 10 min per lesson)
    stats["estimated_study_hours"] = round(stats["total_lessons"] * 10 / 60, 1)

    return stats


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compile classroom database from lessons and curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lessons",
        type=Path,
        default=PROJECT_ROOT / "data" / "lessons",
        help="Path to lessons directory"
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=PROJECT_ROOT / "data" / "curriculum_refined",
        help="Path to curriculum directory"
    )
    parser.add_argument(
        "--poems",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotated" / "poems.parquet",
        help="Path to annotated poems parquet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "classroom.db",
        help="Output database path"
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Output path for stats JSON (default: alongside database)"
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading curriculum...")
    lesson_graph, grammar_index = load_curriculum(args.curriculum)

    logger.info("Loading lessons...")
    lessons_content = load_lessons(args.lessons)
    logger.info(f"  Loaded {len(lessons_content)} lesson files")

    logger.info("Loading poems...")
    poems_df = load_poems(args.poems)
    logger.info(f"  Loaded {len(poems_df)} poems")

    # Create and populate database
    logger.info("Creating database...")
    conn = create_database(args.output)

    logger.info("Populating tables...")
    populate_units(conn, lesson_graph)
    populate_lessons(conn, lesson_graph, lessons_content)
    populate_poems(conn, poems_df)
    populate_grammar_points(conn, grammar_index, lesson_graph)
    populate_lesson_poems(conn, lessons_content)

    # Run integrity checks
    logger.info("Running integrity checks...")
    issues = run_integrity_checks(conn)
    if issues:
        logger.warning(f"Found {len(issues)} integrity issues:")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... and {len(issues) - 10} more")
    else:
        logger.info("  All integrity checks passed!")

    # Compute and save stats
    logger.info("Computing statistics...")
    stats = compute_stats(conn)
    populate_metadata(conn, stats)

    # Save stats JSON
    stats_path = args.stats_output or args.output.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved stats to: {stats_path}")

    # Close connection
    conn.close()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("COMPILATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Database: {args.output}")
    logger.info(f"Units: {stats['total_units']}")
    logger.info(f"Lessons: {stats['total_lessons']}")
    logger.info(f"Poems: {stats['total_poems']} ({stats['poems_in_lessons']} used in lessons)")
    logger.info(f"Grammar points: {stats['total_grammar_points']}")
    logger.info(f"Estimated study time: {stats['estimated_study_hours']} hours")
    if issues:
        logger.warning(f"Integrity issues: {len(issues)}")


if __name__ == "__main__":
    main()
