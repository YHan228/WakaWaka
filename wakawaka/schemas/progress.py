"""
Progress tracking schemas for WakaDecoder.

Defines Pydantic models for student progress including:
- Lesson status tracking
- Student progress state
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class LessonStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class LessonProgress(BaseModel):
    lesson_id: str
    status: LessonStatus = LessonStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    quiz_score: Optional[float] = None


class StudentProgress(BaseModel):
    student_id: str = "default"  # single-user mode
    lessons: dict[str, LessonProgress]
    current_lesson_id: Optional[str] = None
    total_time_minutes: int = 0
