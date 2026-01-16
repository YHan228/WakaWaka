"""
ClassroomLoader - Load data from classroom.db SQLite database.

Provides read-only access to:
- Units and lessons
- Poems with annotations
- Grammar points
- Lesson-poem relationships
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wakadecoder.schemas import LessonContent, PoemAnnotation


@dataclass
class UnitData:
    """Unit information from database."""
    id: str
    title: str
    theme: Optional[str]
    position: int


@dataclass
class LessonSummary:
    """Lightweight lesson info for navigation (without full content)."""
    id: str
    unit_id: str
    title: str
    summary: str
    grammar_point: str
    position: int
    difficulty_tier: int
    prerequisites: list[str]


@dataclass
class PoemData:
    """Poem information from database."""
    id: str
    source: str
    text: str
    reading_hiragana: str
    reading_romaji: str
    author: Optional[str]
    collection: Optional[str]
    difficulty_score: float
    annotations: Optional[dict]


@dataclass
class GrammarPointData:
    """Grammar point information from database."""
    id: str
    category: str
    frequency: int
    avg_difficulty: float
    senses: list[dict]
    lesson_id: Optional[str]


class ClassroomLoader:
    """
    Load classroom data from SQLite database.

    Thread-safe for read operations. Each method creates a new connection.
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize loader with path to classroom.db.

        Args:
            db_path: Path to classroom.db file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Classroom database not found: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else None
        finally:
            conn.close()

    def get_all_metadata(self) -> dict[str, str]:
        """Get all metadata as a dictionary."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT key, value FROM metadata")
            return {row["key"]: row["value"] for row in cursor.fetchall()}
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Units
    # -------------------------------------------------------------------------

    def get_units(self) -> list[UnitData]:
        """Get all units ordered by position."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT id, title, theme, position FROM units ORDER BY position"
            )
            return [
                UnitData(
                    id=row["id"],
                    title=row["title"],
                    theme=row["theme"],
                    position=row["position"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_unit(self, unit_id: str) -> Optional[UnitData]:
        """Get a single unit by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT id, title, theme, position FROM units WHERE id = ?",
                (unit_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return UnitData(
                id=row["id"],
                title=row["title"],
                theme=row["theme"],
                position=row["position"],
            )
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Lessons
    # -------------------------------------------------------------------------

    def get_lessons_for_unit(self, unit_id: str) -> list[LessonSummary]:
        """Get all lessons for a unit, ordered by position."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, unit_id, title, summary, grammar_point,
                          position, difficulty_tier, prerequisites
                   FROM lessons
                   WHERE unit_id = ?
                   ORDER BY position""",
                (unit_id,)
            )
            return [
                LessonSummary(
                    id=row["id"],
                    unit_id=row["unit_id"],
                    title=row["title"],
                    summary=row["summary"],
                    grammar_point=row["grammar_point"],
                    position=row["position"],
                    difficulty_tier=row["difficulty_tier"],
                    prerequisites=json.loads(row["prerequisites"] or "[]"),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_all_lessons(self) -> list[LessonSummary]:
        """Get all lessons ordered by unit position then lesson position."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT l.id, l.unit_id, l.title, l.summary, l.grammar_point,
                          l.position, l.difficulty_tier, l.prerequisites
                   FROM lessons l
                   JOIN units u ON l.unit_id = u.id
                   ORDER BY u.position, l.position"""
            )
            return [
                LessonSummary(
                    id=row["id"],
                    unit_id=row["unit_id"],
                    title=row["title"],
                    summary=row["summary"],
                    grammar_point=row["grammar_point"],
                    position=row["position"],
                    difficulty_tier=row["difficulty_tier"],
                    prerequisites=json.loads(row["prerequisites"] or "[]"),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_lesson_summary(self, lesson_id: str) -> Optional[LessonSummary]:
        """Get lesson summary (without full content) by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, unit_id, title, summary, grammar_point,
                          position, difficulty_tier, prerequisites
                   FROM lessons WHERE id = ?""",
                (lesson_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return LessonSummary(
                id=row["id"],
                unit_id=row["unit_id"],
                title=row["title"],
                summary=row["summary"],
                grammar_point=row["grammar_point"],
                position=row["position"],
                difficulty_tier=row["difficulty_tier"],
                prerequisites=json.loads(row["prerequisites"] or "[]"),
            )
        finally:
            conn.close()

    def get_lesson_content(self, lesson_id: str) -> Optional[LessonContent]:
        """Get full lesson content by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT content FROM lessons WHERE id = ?", (lesson_id,)
            )
            row = cursor.fetchone()
            if not row or not row["content"]:
                return None
            content_dict = json.loads(row["content"])
            return LessonContent(**content_dict)
        finally:
            conn.close()

    def get_lesson_count(self) -> int:
        """Get total number of lessons."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) as count FROM lessons")
            return cursor.fetchone()["count"]
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Poems
    # -------------------------------------------------------------------------

    def get_poem(self, poem_id: str) -> Optional[PoemData]:
        """Get a single poem by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, source, text, reading_hiragana, reading_romaji,
                          author, collection, difficulty_score, annotations
                   FROM poems WHERE id = ?""",
                (poem_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return PoemData(
                id=row["id"],
                source=row["source"],
                text=row["text"],
                reading_hiragana=row["reading_hiragana"],
                reading_romaji=row["reading_romaji"],
                author=row["author"],
                collection=row["collection"],
                difficulty_score=row["difficulty_score"],
                annotations=json.loads(row["annotations"]) if row["annotations"] else None,
            )
        finally:
            conn.close()

    def get_poems_for_lesson(self, lesson_id: str) -> list[PoemData]:
        """Get all poems associated with a lesson, ordered by position."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT p.id, p.source, p.text, p.reading_hiragana, p.reading_romaji,
                          p.author, p.collection, p.difficulty_score, p.annotations
                   FROM poems p
                   JOIN lesson_poems lp ON p.id = lp.poem_id
                   WHERE lp.lesson_id = ?
                   ORDER BY lp.position""",
                (lesson_id,)
            )
            return [
                PoemData(
                    id=row["id"],
                    source=row["source"],
                    text=row["text"],
                    reading_hiragana=row["reading_hiragana"],
                    reading_romaji=row["reading_romaji"],
                    author=row["author"],
                    collection=row["collection"],
                    difficulty_score=row["difficulty_score"],
                    annotations=json.loads(row["annotations"]) if row["annotations"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_all_poems(self, limit: Optional[int] = None) -> list[PoemData]:
        """Get all poems, optionally limited."""
        conn = self._get_connection()
        try:
            query = """SELECT id, source, text, reading_hiragana, reading_romaji,
                              author, collection, difficulty_score, annotations
                       FROM poems ORDER BY difficulty_score"""
            if limit:
                query += f" LIMIT {limit}"
            cursor = conn.execute(query)
            return [
                PoemData(
                    id=row["id"],
                    source=row["source"],
                    text=row["text"],
                    reading_hiragana=row["reading_hiragana"],
                    reading_romaji=row["reading_romaji"],
                    author=row["author"],
                    collection=row["collection"],
                    difficulty_score=row["difficulty_score"],
                    annotations=json.loads(row["annotations"]) if row["annotations"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_poem_count(self) -> int:
        """Get total number of poems."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) as count FROM poems")
            return cursor.fetchone()["count"]
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Grammar Points
    # -------------------------------------------------------------------------

    def get_grammar_point(self, grammar_id: str) -> Optional[GrammarPointData]:
        """Get a grammar point by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, category, frequency, avg_difficulty, senses, lesson_id
                   FROM grammar_points WHERE id = ?""",
                (grammar_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return GrammarPointData(
                id=row["id"],
                category=row["category"],
                frequency=row["frequency"],
                avg_difficulty=row["avg_difficulty"],
                senses=json.loads(row["senses"]) if row["senses"] else [],
                lesson_id=row["lesson_id"],
            )
        finally:
            conn.close()

    def get_all_grammar_points(self) -> list[GrammarPointData]:
        """Get all grammar points."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, category, frequency, avg_difficulty, senses, lesson_id
                   FROM grammar_points ORDER BY frequency DESC"""
            )
            return [
                GrammarPointData(
                    id=row["id"],
                    category=row["category"],
                    frequency=row["frequency"],
                    avg_difficulty=row["avg_difficulty"],
                    senses=json.loads(row["senses"]) if row["senses"] else [],
                    lesson_id=row["lesson_id"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_grammar_points_for_lesson(self, lesson_id: str) -> list[GrammarPointData]:
        """Get grammar points taught in a specific lesson."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT id, category, frequency, avg_difficulty, senses, lesson_id
                   FROM grammar_points WHERE lesson_id = ?""",
                (lesson_id,)
            )
            return [
                GrammarPointData(
                    id=row["id"],
                    category=row["category"],
                    frequency=row["frequency"],
                    avg_difficulty=row["avg_difficulty"],
                    senses=json.loads(row["senses"]) if row["senses"] else [],
                    lesson_id=row["lesson_id"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_grammar_point_count(self) -> int:
        """Get total number of grammar points."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) as count FROM grammar_points")
            return cursor.fetchone()["count"]
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Curriculum Tree
    # -------------------------------------------------------------------------

    def get_curriculum_tree(self) -> list[dict]:
        """
        Get the full curriculum tree structure for sidebar navigation.

        Returns a list of units, each containing their lessons.
        """
        units = self.get_units()
        tree = []
        for unit in units:
            lessons = self.get_lessons_for_unit(unit.id)
            tree.append({
                "unit": unit,
                "lessons": lessons,
            })
        return tree
