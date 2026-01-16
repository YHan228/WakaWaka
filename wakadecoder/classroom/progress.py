"""
ProgressTracker - Track student progress in ~/.wakadecoder/progress.db.

Stores user progress separately from content database:
- Lesson completion status
- Quiz scores
- Current position
- Total study time
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from wakadecoder.schemas import LessonStatus, LessonProgress, StudentProgress


DEFAULT_PROGRESS_DIR = Path.home() / ".wakadecoder"
DEFAULT_PROGRESS_DB = DEFAULT_PROGRESS_DIR / "progress.db"


class ProgressTracker:
    """
    Track student progress in SQLite database.

    Progress is stored separately from content (classroom.db) so that:
    - Content can be updated without losing progress
    - Progress is user-specific, content is shared
    """

    def __init__(self, db_path: Optional[Path] = None, student_id: str = "default"):
        """
        Initialize progress tracker.

        Args:
            db_path: Path to progress.db (default: ~/.wakadecoder/progress.db)
            student_id: Student identifier for multi-user support
        """
        self.db_path = db_path or DEFAULT_PROGRESS_DB
        self.student_id = student_id
        self._ensure_database()

    def _ensure_database(self):
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS lesson_progress (
                    student_id TEXT NOT NULL,
                    lesson_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'not_started',
                    started_at TEXT,
                    completed_at TEXT,
                    quiz_score REAL,
                    PRIMARY KEY (student_id, lesson_id)
                );

                CREATE TABLE IF NOT EXISTS student_state (
                    student_id TEXT PRIMARY KEY,
                    current_lesson_id TEXT,
                    total_time_minutes INTEGER DEFAULT 0,
                    last_activity_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_lesson_progress_student
                ON lesson_progress(student_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # Lesson Progress
    # -------------------------------------------------------------------------

    def get_lesson_progress(self, lesson_id: str) -> LessonProgress:
        """Get progress for a specific lesson."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT lesson_id, status, started_at, completed_at, quiz_score
                   FROM lesson_progress
                   WHERE student_id = ? AND lesson_id = ?""",
                (self.student_id, lesson_id)
            )
            row = cursor.fetchone()
            if not row:
                return LessonProgress(lesson_id=lesson_id)

            return LessonProgress(
                lesson_id=row["lesson_id"],
                status=LessonStatus(row["status"]),
                started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                quiz_score=row["quiz_score"],
            )
        finally:
            conn.close()

    def get_all_lesson_progress(self) -> dict[str, LessonProgress]:
        """Get progress for all lessons."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT lesson_id, status, started_at, completed_at, quiz_score
                   FROM lesson_progress
                   WHERE student_id = ?""",
                (self.student_id,)
            )
            result = {}
            for row in cursor.fetchall():
                result[row["lesson_id"]] = LessonProgress(
                    lesson_id=row["lesson_id"],
                    status=LessonStatus(row["status"]),
                    started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    quiz_score=row["quiz_score"],
                )
            return result
        finally:
            conn.close()

    def start_lesson(self, lesson_id: str):
        """Mark a lesson as started (in progress)."""
        conn = self._get_connection()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO lesson_progress (student_id, lesson_id, status, started_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(student_id, lesson_id) DO UPDATE SET
                     status = CASE
                       WHEN status = 'not_started' THEN 'in_progress'
                       ELSE status
                     END,
                     started_at = CASE
                       WHEN started_at IS NULL THEN ?
                       ELSE started_at
                     END""",
                (self.student_id, lesson_id, LessonStatus.IN_PROGRESS.value, now, now)
            )
            # Update current lesson
            conn.execute(
                """INSERT INTO student_state (student_id, current_lesson_id, last_activity_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(student_id) DO UPDATE SET
                     current_lesson_id = ?,
                     last_activity_at = ?""",
                (self.student_id, lesson_id, now, lesson_id, now)
            )
            conn.commit()
        finally:
            conn.close()

    def complete_lesson(self, lesson_id: str, quiz_score: Optional[float] = None):
        """Mark a lesson as completed."""
        conn = self._get_connection()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO lesson_progress (student_id, lesson_id, status, completed_at, quiz_score)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(student_id, lesson_id) DO UPDATE SET
                     status = 'completed',
                     completed_at = ?,
                     quiz_score = COALESCE(?, quiz_score)""",
                (self.student_id, lesson_id, LessonStatus.COMPLETED.value, now, quiz_score, now, quiz_score)
            )
            conn.execute(
                """UPDATE student_state SET last_activity_at = ? WHERE student_id = ?""",
                (now, self.student_id)
            )
            conn.commit()
        finally:
            conn.close()

    def reset_lesson(self, lesson_id: str):
        """Reset a lesson to not started."""
        conn = self._get_connection()
        try:
            conn.execute(
                """DELETE FROM lesson_progress
                   WHERE student_id = ? AND lesson_id = ?""",
                (self.student_id, lesson_id)
            )
            conn.commit()
        finally:
            conn.close()

    def is_lesson_completed(self, lesson_id: str) -> bool:
        """Check if a lesson is completed."""
        progress = self.get_lesson_progress(lesson_id)
        return progress.status == LessonStatus.COMPLETED

    def get_completed_lesson_ids(self) -> set[str]:
        """Get set of completed lesson IDs."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT lesson_id FROM lesson_progress
                   WHERE student_id = ? AND status = 'completed'""",
                (self.student_id,)
            )
            return {row["lesson_id"] for row in cursor.fetchall()}
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Student State
    # -------------------------------------------------------------------------

    def get_current_lesson_id(self) -> Optional[str]:
        """Get the ID of the current lesson."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT current_lesson_id FROM student_state WHERE student_id = ?""",
                (self.student_id,)
            )
            row = cursor.fetchone()
            return row["current_lesson_id"] if row else None
        finally:
            conn.close()

    def set_current_lesson_id(self, lesson_id: str):
        """Set the current lesson ID."""
        conn = self._get_connection()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO student_state (student_id, current_lesson_id, last_activity_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(student_id) DO UPDATE SET
                     current_lesson_id = ?,
                     last_activity_at = ?""",
                (self.student_id, lesson_id, now, lesson_id, now)
            )
            conn.commit()
        finally:
            conn.close()

    def add_study_time(self, minutes: int):
        """Add study time to total."""
        conn = self._get_connection()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO student_state (student_id, total_time_minutes, last_activity_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(student_id) DO UPDATE SET
                     total_time_minutes = total_time_minutes + ?,
                     last_activity_at = ?""",
                (self.student_id, minutes, now, minutes, now)
            )
            conn.commit()
        finally:
            conn.close()

    def get_total_study_time(self) -> int:
        """Get total study time in minutes."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT total_time_minutes FROM student_state WHERE student_id = ?""",
                (self.student_id,)
            )
            row = cursor.fetchone()
            return row["total_time_minutes"] if row else 0
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Full Progress Export
    # -------------------------------------------------------------------------

    def get_student_progress(self) -> StudentProgress:
        """Get full student progress object."""
        all_progress = self.get_all_lesson_progress()
        current_lesson_id = self.get_current_lesson_id()
        total_time = self.get_total_study_time()

        return StudentProgress(
            student_id=self.student_id,
            lessons=all_progress,
            current_lesson_id=current_lesson_id,
            total_time_minutes=total_time,
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_completion_stats(self, total_lessons: int) -> dict:
        """
        Get completion statistics.

        Args:
            total_lessons: Total number of lessons in curriculum

        Returns:
            Dictionary with completion stats
        """
        completed = self.get_completed_lesson_ids()
        all_progress = self.get_all_lesson_progress()

        in_progress = sum(
            1 for p in all_progress.values()
            if p.status == LessonStatus.IN_PROGRESS
        )

        return {
            "total_lessons": total_lessons,
            "completed": len(completed),
            "in_progress": in_progress,
            "not_started": total_lessons - len(completed) - in_progress,
            "completion_percent": round(len(completed) / total_lessons * 100, 1) if total_lessons > 0 else 0,
            "total_study_minutes": self.get_total_study_time(),
        }

    def reset_all_progress(self):
        """Reset all progress for the current student."""
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM lesson_progress WHERE student_id = ?",
                (self.student_id,)
            )
            conn.execute(
                "DELETE FROM student_state WHERE student_id = ?",
                (self.student_id,)
            )
            conn.commit()
        finally:
            conn.close()
