"""
Navigator - Lesson sequencing, prerequisite checking, and navigation.

Provides:
- Next/previous lesson navigation
- Prerequisite validation
- Lesson availability based on progress
- Curriculum tree with status indicators
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from wakadecoder.schemas import LessonStatus, LessonContent

from .loader import ClassroomLoader, LessonSummary, UnitData
from .progress import ProgressTracker


class LessonAvailability(str, Enum):
    """Lesson availability status for UI display."""
    LOCKED = "locked"           # Prerequisites not met
    AVAILABLE = "available"     # Can start
    IN_PROGRESS = "in_progress" # Started but not completed
    COMPLETED = "completed"     # Finished


@dataclass
class NavigationLesson:
    """Lesson with navigation metadata."""
    summary: LessonSummary
    availability: LessonAvailability
    is_current: bool
    missing_prerequisites: list[str]  # IDs of unmet prerequisites


@dataclass
class NavigationUnit:
    """Unit with lessons and navigation metadata."""
    unit: UnitData
    lessons: list[NavigationLesson]
    completed_count: int
    total_count: int


class Navigator:
    """
    Navigate through the curriculum with prerequisite checking.

    Combines ClassroomLoader (content) with ProgressTracker (user state)
    to provide navigation with availability checking.
    """

    def __init__(self, loader: ClassroomLoader, progress: ProgressTracker):
        """
        Initialize navigator.

        Args:
            loader: ClassroomLoader instance for content access
            progress: ProgressTracker instance for user progress
        """
        self.loader = loader
        self.progress = progress
        self._lesson_order: list[str] = []
        self._lesson_index: dict[str, int] = {}
        self._refresh_lesson_order()

    def _refresh_lesson_order(self):
        """Build ordered list of lesson IDs for navigation."""
        lessons = self.loader.get_all_lessons()
        self._lesson_order = [lesson.id for lesson in lessons]
        self._lesson_index = {lid: idx for idx, lid in enumerate(self._lesson_order)}

    @property
    def total_lessons(self) -> int:
        """Total number of lessons."""
        return len(self._lesson_order)

    # -------------------------------------------------------------------------
    # Availability Checking
    # -------------------------------------------------------------------------

    def get_lesson_availability(self, lesson_id: str) -> tuple[LessonAvailability, list[str]]:
        """
        Check lesson availability based on progress and prerequisites.

        Returns:
            Tuple of (availability status, list of missing prerequisite IDs)
        """
        # Get lesson info
        lesson = self.loader.get_lesson_summary(lesson_id)
        if not lesson:
            return LessonAvailability.LOCKED, []

        # Get progress
        lesson_progress = self.progress.get_lesson_progress(lesson_id)

        # If completed, it's completed
        if lesson_progress.status == LessonStatus.COMPLETED:
            return LessonAvailability.COMPLETED, []

        # If in progress, it's in progress
        if lesson_progress.status == LessonStatus.IN_PROGRESS:
            return LessonAvailability.IN_PROGRESS, []

        # Check prerequisites
        completed = self.progress.get_completed_lesson_ids()
        missing = [
            prereq for prereq in lesson.prerequisites
            if prereq not in completed
        ]

        if missing:
            return LessonAvailability.LOCKED, missing

        return LessonAvailability.AVAILABLE, []

    def is_lesson_available(self, lesson_id: str) -> bool:
        """Check if a lesson can be started."""
        availability, _ = self.get_lesson_availability(lesson_id)
        return availability in (
            LessonAvailability.AVAILABLE,
            LessonAvailability.IN_PROGRESS,
            LessonAvailability.COMPLETED,
        )

    def get_available_lessons(self) -> list[LessonSummary]:
        """Get all lessons that can be started or continued."""
        result = []
        for lesson_id in self._lesson_order:
            if self.is_lesson_available(lesson_id):
                lesson = self.loader.get_lesson_summary(lesson_id)
                if lesson:
                    result.append(lesson)
        return result

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    def get_first_lesson(self) -> Optional[LessonContent]:
        """Get the first lesson in the curriculum."""
        if not self._lesson_order:
            return None
        return self.loader.get_lesson_content(self._lesson_order[0])

    def get_first_lesson_id(self) -> Optional[str]:
        """Get the ID of the first lesson."""
        return self._lesson_order[0] if self._lesson_order else None

    def get_next_lesson_id(self, current_id: str) -> Optional[str]:
        """Get the ID of the next lesson in order."""
        if current_id not in self._lesson_index:
            return None
        current_idx = self._lesson_index[current_id]
        if current_idx + 1 >= len(self._lesson_order):
            return None
        return self._lesson_order[current_idx + 1]

    def get_previous_lesson_id(self, current_id: str) -> Optional[str]:
        """Get the ID of the previous lesson in order."""
        if current_id not in self._lesson_index:
            return None
        current_idx = self._lesson_index[current_id]
        if current_idx <= 0:
            return None
        return self._lesson_order[current_idx - 1]

    def get_next_available_lesson_id(self, current_id: str) -> Optional[str]:
        """Get the next lesson that is available to start."""
        next_id = self.get_next_lesson_id(current_id)
        while next_id:
            if self.is_lesson_available(next_id):
                return next_id
            next_id = self.get_next_lesson_id(next_id)
        return None

    def get_recommended_lesson_id(self) -> Optional[str]:
        """
        Get the recommended next lesson for the student.

        Priority:
        1. Current lesson if in progress
        2. First incomplete lesson with prerequisites met
        3. First lesson
        """
        # Check current lesson
        current_id = self.progress.get_current_lesson_id()
        if current_id:
            availability, _ = self.get_lesson_availability(current_id)
            if availability == LessonAvailability.IN_PROGRESS:
                return current_id

        # Find first available lesson that's not completed
        for lesson_id in self._lesson_order:
            availability, _ = self.get_lesson_availability(lesson_id)
            if availability == LessonAvailability.AVAILABLE:
                return lesson_id

        # All completed or none available - return first lesson
        return self.get_first_lesson_id()

    def get_lesson_position(self, lesson_id: str) -> tuple[int, int]:
        """
        Get lesson position as (current, total).

        Returns (0, total) if lesson not found.
        """
        if lesson_id not in self._lesson_index:
            return (0, len(self._lesson_order))
        return (self._lesson_index[lesson_id] + 1, len(self._lesson_order))

    # -------------------------------------------------------------------------
    # Curriculum Tree
    # -------------------------------------------------------------------------

    def get_navigation_tree(self) -> list[NavigationUnit]:
        """
        Get full curriculum tree with navigation metadata.

        Returns list of units with lessons, each annotated with:
        - Availability status
        - Whether it's the current lesson
        - Missing prerequisites
        """
        current_lesson_id = self.progress.get_current_lesson_id()
        completed = self.progress.get_completed_lesson_ids()

        tree = []
        for unit in self.loader.get_units():
            lessons = self.loader.get_lessons_for_unit(unit.id)
            nav_lessons = []
            completed_count = 0

            for lesson in lessons:
                availability, missing = self.get_lesson_availability(lesson.id)
                is_current = lesson.id == current_lesson_id

                if availability == LessonAvailability.COMPLETED:
                    completed_count += 1

                nav_lessons.append(NavigationLesson(
                    summary=lesson,
                    availability=availability,
                    is_current=is_current,
                    missing_prerequisites=missing,
                ))

            tree.append(NavigationUnit(
                unit=unit,
                lessons=nav_lessons,
                completed_count=completed_count,
                total_count=len(lessons),
            ))

        return tree

    def get_status_indicator(self, lesson_id: str) -> str:
        """
        Get status indicator for sidebar display.

        Returns:
            ✓ for completed
            → for current
            ○ for available
            ◌ for locked
        """
        current_id = self.progress.get_current_lesson_id()
        availability, _ = self.get_lesson_availability(lesson_id)

        if lesson_id == current_id and availability == LessonAvailability.IN_PROGRESS:
            return "→"
        elif availability == LessonAvailability.COMPLETED:
            return "✓"
        elif availability == LessonAvailability.AVAILABLE:
            return "○"
        else:
            return "◌"

    # -------------------------------------------------------------------------
    # Lesson Actions
    # -------------------------------------------------------------------------

    def start_lesson(self, lesson_id: str) -> bool:
        """
        Start a lesson if available.

        Returns True if lesson was started, False if unavailable.
        """
        if not self.is_lesson_available(lesson_id):
            return False

        self.progress.start_lesson(lesson_id)
        return True

    def complete_lesson(self, lesson_id: str, quiz_score: Optional[float] = None) -> Optional[str]:
        """
        Complete a lesson and return next available lesson ID.

        Args:
            lesson_id: ID of lesson to complete
            quiz_score: Optional quiz score (0.0-1.0)

        Returns:
            ID of next available lesson, or None if curriculum complete
        """
        self.progress.complete_lesson(lesson_id, quiz_score)
        return self.get_next_available_lesson_id(lesson_id)

    # -------------------------------------------------------------------------
    # Progress Summary
    # -------------------------------------------------------------------------

    def get_progress_summary(self) -> dict:
        """Get progress summary for display."""
        stats = self.progress.get_completion_stats(self.total_lessons)
        tree = self.get_navigation_tree()

        # Unit-level stats
        unit_stats = []
        for nav_unit in tree:
            unit_stats.append({
                "id": nav_unit.unit.id,
                "title": nav_unit.unit.title,
                "completed": nav_unit.completed_count,
                "total": nav_unit.total_count,
            })

        return {
            **stats,
            "units": unit_stats,
            "current_lesson_id": self.progress.get_current_lesson_id(),
            "recommended_lesson_id": self.get_recommended_lesson_id(),
        }
