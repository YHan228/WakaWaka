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
    render_literary_insight,
    render_teaching_step,
    render_grammar_explanation,
    render_forward_references,
    render_lesson,
    get_comprehension_checks,
    markdown_to_html,
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

from .audio import (
    get_audio_path,
    get_audio_for_lesson,
)

from .literary import (
    get_literary_css,
    render_literary_analysis,
)

from .crash_course import (
    get_crash_course_css,
    render_text_step,
    render_kana_chart_step,
    render_vocab_intro_step,
    render_practice_step,
    render_example_sentence_step,
    render_mini_poem_step,
    render_summary_step as render_crash_summary_step,
    render_tip_step,
    render_crash_course_step,
    render_crash_course_lesson_header,
    render_crash_course_quiz,
    render_crash_course_overview,
    load_crash_course,
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
    "render_literary_insight",
    "render_teaching_step",
    "render_grammar_explanation",
    "render_forward_references",
    "render_lesson",
    "get_comprehension_checks",
    "markdown_to_html",
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
    # Audio
    "get_audio_path",
    "get_audio_for_lesson",
    # Literary
    "get_literary_css",
    "render_literary_analysis",
    # Crash course
    "get_crash_course_css",
    "render_text_step",
    "render_kana_chart_step",
    "render_vocab_intro_step",
    "render_practice_step",
    "render_example_sentence_step",
    "render_mini_poem_step",
    "render_crash_summary_step",
    "render_tip_step",
    "render_crash_course_step",
    "render_crash_course_lesson_header",
    "render_crash_course_quiz",
    "render_crash_course_overview",
    "load_crash_course",
]
