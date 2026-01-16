"""
WakaDecoder Viewer - Rendering components for lesson display.

This module provides:
- Lesson rendering with interactive vocabulary
- Quiz/comprehension check display
- Grammar reference card rendering
"""

from .lesson import (
    get_vocab_css,
    detect_word_type,
    render_poem_with_vocabulary,
    render_vocab_tooltip,
    render_vocab_list,
    render_poem_presentation,
    render_introduction,
    render_grammar_spotlight,
    render_contrast_example,
    render_summary,
    render_teaching_step,
    render_grammar_explanation,
    render_forward_references,
    render_lesson,
    get_comprehension_checks,
    WORD_TYPE_COLORS,
)

from .quiz import (
    get_quiz_css,
    QuizQuestion,
    extract_quiz_questions,
    render_quiz_question,
    render_all_quiz_questions,
    render_quiz_for_streamlit,
    calculate_quiz_score,
    render_quiz_score,
)

from .reference import (
    get_reference_css,
    render_reference_card,
    render_reference_card_from_lesson,
    render_grammar_index_item,
    render_grammar_index,
    render_learned_reference_cards,
    render_mini_reference_cards,
    filter_grammar_points,
)

__all__ = [
    # Lesson rendering
    "get_vocab_css",
    "detect_word_type",
    "render_poem_with_vocabulary",
    "render_vocab_tooltip",
    "render_vocab_list",
    "render_poem_presentation",
    "render_introduction",
    "render_grammar_spotlight",
    "render_contrast_example",
    "render_summary",
    "render_teaching_step",
    "render_grammar_explanation",
    "render_forward_references",
    "render_lesson",
    "get_comprehension_checks",
    "WORD_TYPE_COLORS",
    # Quiz
    "get_quiz_css",
    "QuizQuestion",
    "extract_quiz_questions",
    "render_quiz_question",
    "render_all_quiz_questions",
    "render_quiz_for_streamlit",
    "calculate_quiz_score",
    "render_quiz_score",
    # Reference
    "get_reference_css",
    "render_reference_card",
    "render_reference_card_from_lesson",
    "render_grammar_index_item",
    "render_grammar_index",
    "render_learned_reference_cards",
    "render_mini_reference_cards",
    "filter_grammar_points",
]
