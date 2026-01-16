"""
WakaDecoder Classroom - Runtime components for loading and navigating lessons.

This module provides:
- ClassroomLoader: Load data from classroom.db
- ProgressTracker: Track student progress
- Navigator: Lesson sequencing and prerequisites
"""

from .loader import (
    ClassroomLoader,
    LiteraryLoader,
    UnitData,
    LessonSummary,
    PoemData,
    GrammarPointData,
)

from .progress import (
    ProgressTracker,
    DEFAULT_PROGRESS_DIR,
    DEFAULT_PROGRESS_DB,
)

from .navigator import (
    Navigator,
    LessonAvailability,
    NavigationLesson,
    NavigationUnit,
)

__all__ = [
    # Loader
    "ClassroomLoader",
    "LiteraryLoader",
    "UnitData",
    "LessonSummary",
    "PoemData",
    "GrammarPointData",
    # Progress
    "ProgressTracker",
    "DEFAULT_PROGRESS_DIR",
    "DEFAULT_PROGRESS_DB",
    # Navigator
    "Navigator",
    "LessonAvailability",
    "NavigationLesson",
    "NavigationUnit",
]
