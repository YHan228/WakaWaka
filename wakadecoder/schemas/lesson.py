"""
Lesson content schemas for WakaDecoder.

Defines Pydantic models for lesson content including:
- Grammar explanations
- Teaching sequence steps
- Reference cards
- Forward references
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, Union


class GrammarExplanation(BaseModel):
    concept: str
    formation: Optional[str] = None
    variations: list[str] = []
    common_confusions: list[str] = []
    logic_analogy: Optional[str] = None


class PoemDisplay(BaseModel):
    text_with_furigana: str  # HTML with <ruby> tags
    romaji: str
    translation: str         # English translation


# -----------------------------------------------------------------------------
# Teaching step types
# -----------------------------------------------------------------------------

class TeachingStepBase(BaseModel):
    type: str


class IntroductionStep(TeachingStepBase):
    type: Literal["introduction"] = "introduction"
    content: str


class VocabularyItem(BaseModel):
    """Vocabulary item for lesson display (derived from VocabularyAnnotation)."""
    word: str              # Japanese word
    reading: str           # hiragana reading
    meaning: str           # meaning (may note Chinese cognate)
    span: Optional[list[int]] = None  # [start, end) in poem text for highlighting
    chinese_cognate_note: Optional[str] = None


class PoemPresentationStep(TeachingStepBase):
    """
    Presents a poem with furigana, translation, and vocabulary.
    Spans use 0-based, end-exclusive indexing (Python slice semantics).
    """
    type: Literal["poem_presentation"] = "poem_presentation"
    poem_id: str
    display: PoemDisplay
    vocabulary: list[VocabularyItem] = []  # key words for Chinese speakers
    focus_highlight: Optional[list[int]] = None  # [start, end) span for grammar point


class GrammarSpotlightStep(TeachingStepBase):
    type: Literal["grammar_spotlight"] = "grammar_spotlight"
    content: str
    evidence: str


class ContrastExampleStep(TeachingStepBase):
    type: Literal["contrast_example"] = "contrast_example"
    content: str


class ComprehensionCheckStep(TeachingStepBase):
    type: Literal["comprehension_check"] = "comprehension_check"
    question: str
    answer: str
    hint: Optional[str] = None


class SummaryStep(TeachingStepBase):
    type: Literal["summary"] = "summary"
    content: str


TeachingStep = Union[
    IntroductionStep,
    PoemPresentationStep,
    GrammarSpotlightStep,
    ContrastExampleStep,
    ComprehensionCheckStep,
    SummaryStep
]


# -----------------------------------------------------------------------------
# Reference card and forward references
# -----------------------------------------------------------------------------

class ReferenceCard(BaseModel):
    point: str
    one_liner: str
    example: str
    see_also: list[str] = []


class ForwardReference(BaseModel):
    point: str
    note: str


# -----------------------------------------------------------------------------
# Main lesson content schema
# -----------------------------------------------------------------------------

class LessonContent(BaseModel):
    lesson_id: str
    lesson_title: str
    lesson_summary: str
    grammar_point: str
    grammar_explanation: GrammarExplanation
    teaching_sequence: list[TeachingStep] = Field(..., min_length=3)
    reference_card: ReferenceCard
    forward_references: list[ForwardReference] = []
