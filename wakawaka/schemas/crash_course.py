"""
Crash Course schemas for WakaDecoder.

Defines Pydantic models for the Japanese crash course for absolute beginners.
This is a separate curriculum from the main poetry lessons, designed to prepare
Chinese speakers with zero Japanese knowledge for the WakaWaka classical poetry curriculum.
"""

from pydantic import BaseModel, Field, Discriminator, Tag
from typing import Optional, Literal, Union, Annotated, Any
from datetime import datetime


# -----------------------------------------------------------------------------
# Step Types for Crash Course
# -----------------------------------------------------------------------------

class CrashCourseStepBase(BaseModel):
    """Base class for crash course teaching steps."""
    type: str


class TextStep(CrashCourseStepBase):
    """Simple markdown content step."""
    type: Literal["text"] = "text"
    title: Optional[str] = None
    content: str  # Markdown content


class KanaChartStep(CrashCourseStepBase):
    """Hiragana/katakana chart with interactive elements."""
    type: Literal["kana_chart"] = "kana_chart"
    title: str
    description: Optional[str] = None
    chart_type: Literal["hiragana", "katakana", "both"] = "hiragana"
    rows: Union[list[list[dict]], list[dict]]  # Can be 2D or flat list of kana cells
    highlight_row: Optional[int] = None  # Row to highlight for current lesson


class VocabIntroStep(CrashCourseStepBase):
    """Vocabulary introduction with Chinese comparison."""
    type: Literal["vocab_intro"] = "vocab_intro"
    title: str
    description: Optional[str] = None
    items: list[dict]  # [{japanese, reading, chinese, meaning, note?}, ...]


class PracticeStep(CrashCourseStepBase):
    """Interactive practice exercise."""
    type: Literal["practice"] = "practice"
    title: str
    instruction: str
    exercise_type: Optional[Literal["match", "fill_blank", "select", "order"]] = "select"
    items: list[Any] = []  # Format depends on exercise_type (can be strings or dicts)
    hint: Optional[str] = None
    answer: Optional[str] = None  # For some practice types


class ExampleSentenceStep(CrashCourseStepBase):
    """Sentence breakdown with parsing."""
    type: Literal["example_sentence"] = "example_sentence"
    title: Optional[str] = None
    sentence: str  # Japanese sentence
    reading: Optional[str] = None  # Hiragana reading
    translation: str  # Chinese translation
    breakdown: list[dict] = []  # [{part, reading, role, meaning}, ...]
    grammar_note: Optional[str] = None


class MiniPoemStep(CrashCourseStepBase):
    """Simple poem preview for crash course."""
    type: Literal["mini_poem"] = "mini_poem"
    title: Optional[str] = None
    poem_text: Optional[str] = None  # Japanese text
    reading: Optional[str] = None  # Hiragana/romaji
    translation: Optional[str] = None  # Chinese translation
    analysis: Optional[str] = None  # Simple analysis highlighting learned concepts
    vocabulary: list[dict] = []  # [{word, reading, meaning}, ...]
    # Alternative field names that LLM might use
    text: Optional[str] = None
    content: Optional[str] = None
    question: Optional[str] = None


class SummaryStep(CrashCourseStepBase):
    """Key takeaways summary."""
    type: Literal["summary"] = "summary"
    title: str = "本课要点"
    key_points: list[str]
    preview_next: Optional[str] = None  # What's coming next


class TipStep(CrashCourseStepBase):
    """Learning tip or cultural note."""
    type: Literal["tip"] = "tip"
    tip_type: Literal["learning", "cultural", "memory", "comparison"] = "learning"
    title: Optional[str] = None
    content: str


# Union of all step types with field discriminator
CrashCourseStep = Annotated[
    Union[
        Annotated[TextStep, Tag("text")],
        Annotated[KanaChartStep, Tag("kana_chart")],
        Annotated[VocabIntroStep, Tag("vocab_intro")],
        Annotated[PracticeStep, Tag("practice")],
        Annotated[ExampleSentenceStep, Tag("example_sentence")],
        Annotated[MiniPoemStep, Tag("mini_poem")],
        Annotated[SummaryStep, Tag("summary")],
        Annotated[TipStep, Tag("tip")],
    ],
    Discriminator("type"),
]


# -----------------------------------------------------------------------------
# Quiz for Crash Course
# -----------------------------------------------------------------------------

class CrashCourseQuizQuestion(BaseModel):
    """Single quiz question."""
    question: str
    question_type: Literal["multiple_choice", "fill_blank", "true_false"]
    options: Optional[list[str]] = None  # For multiple choice
    correct_answer: str
    explanation: str
    hint: Optional[str] = None


class CrashCourseQuiz(BaseModel):
    """Comprehension check quiz for a crash course lesson."""
    questions: list[CrashCourseQuizQuestion] = Field(..., min_length=1, max_length=5)


# -----------------------------------------------------------------------------
# Lesson and Course Models
# -----------------------------------------------------------------------------

class CrashCourseLesson(BaseModel):
    """Single crash course lesson."""
    lesson_id: str  # "crash_01", "crash_02", etc.
    lesson_number: int  # 1-7
    title: str  # Chinese title
    subtitle: str  # Hook/tagline
    estimated_minutes: int
    objectives: list[str]  # What student will learn
    prerequisites: list[str] = []  # What should be known before
    teaching_sequence: list[CrashCourseStep] = Field(..., min_length=1)
    comprehension_check: Optional[CrashCourseQuiz] = None


class CrashCourse(BaseModel):
    """Complete crash course curriculum."""
    title: str
    subtitle: str
    description: str
    target_audience: str
    lessons: list[CrashCourseLesson] = Field(..., min_length=1)
    version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# -----------------------------------------------------------------------------
# Progress Tracking for Crash Course
# -----------------------------------------------------------------------------

class CrashCourseProgress(BaseModel):
    """Track student progress through crash course."""
    completed_lessons: list[str] = []  # List of completed lesson_ids
    current_lesson: Optional[str] = None
    quiz_scores: dict[str, float] = {}  # lesson_id -> score (0.0-1.0)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def is_complete(self, total_lessons: int = 7) -> bool:
        """Check if all lessons are completed."""
        return len(self.completed_lessons) >= total_lessons

    def get_progress_percent(self, total_lessons: int = 7) -> float:
        """Get completion percentage."""
        return len(self.completed_lessons) / total_lessons * 100 if total_lessons > 0 else 0
