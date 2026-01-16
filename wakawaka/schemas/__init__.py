"""
WakaDecoder Schemas - Pydantic models for the poetry learning platform.

This module exports all schema classes for:
- Annotation: poem tokenization, grammar points, vocabulary
- Curriculum: grammar index, prerequisite graph, lesson graph
- Lesson: lesson content, teaching steps, reference cards
- Progress: student progress tracking
"""

# Annotation schemas
from .annotation import (
    FugashiToken,
    TokenReading,
    GrammarPoint,
    VocabularyAnnotation,
    DifficultyFactor,
    PoemAnnotation,
    DIFFICULTY_WEIGHTS,
    compute_difficulty_score,
    validate_span,
)

# Curriculum schemas
from .curriculum import (
    SenseEntry,
    GrammarIndexEntry,
    GrammarIndex,
    PrerequisiteEdge,
    PrerequisiteGraph,
    LessonNode,
    Unit,
    LessonGraph,
    STOPLIST_CANONICAL_IDS,
)

# Lesson schemas
from .lesson import (
    GrammarExplanation,
    PoemDisplay,
    VocabularyItem,
    IntroductionStep,
    PoemPresentationStep,
    GrammarSpotlightStep,
    ContrastExampleStep,
    ComprehensionCheckStep,
    SummaryStep,
    TeachingStep,
    ReferenceCard,
    ForwardReference,
    LessonContent,
)

# Progress schemas
from .progress import (
    LessonStatus,
    LessonProgress,
    StudentProgress,
)

__all__ = [
    # Annotation
    'FugashiToken',
    'TokenReading',
    'GrammarPoint',
    'VocabularyAnnotation',
    'DifficultyFactor',
    'PoemAnnotation',
    'DIFFICULTY_WEIGHTS',
    'compute_difficulty_score',
    'validate_span',
    # Curriculum
    'SenseEntry',
    'GrammarIndexEntry',
    'GrammarIndex',
    'PrerequisiteEdge',
    'PrerequisiteGraph',
    'LessonNode',
    'Unit',
    'LessonGraph',
    'STOPLIST_CANONICAL_IDS',
    # Lesson
    'GrammarExplanation',
    'PoemDisplay',
    'VocabularyItem',
    'IntroductionStep',
    'PoemPresentationStep',
    'GrammarSpotlightStep',
    'ContrastExampleStep',
    'ComprehensionCheckStep',
    'SummaryStep',
    'TeachingStep',
    'ReferenceCard',
    'ForwardReference',
    'LessonContent',
    # Progress
    'LessonStatus',
    'LessonProgress',
    'StudentProgress',
]
