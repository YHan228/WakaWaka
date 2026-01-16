"""
Quiz renderer - Comprehension check display and interaction.

Provides:
- Comprehension question rendering
- Show/hide answer functionality
- Quiz scoring support
"""

import html
from dataclasses import dataclass
from typing import Optional

from wakawaka.schemas import ComprehensionCheckStep, LessonContent


@dataclass
class QuizQuestion:
    """Quiz question with metadata."""
    index: int
    question: str
    answer: str
    hint: Optional[str]
    user_answer: Optional[str] = None
    revealed: bool = False


def get_quiz_css() -> str:
    """Get CSS styles for quiz display."""
    return """
    <style>
    .quiz-container {
        background: #e3f2fd;
        border-radius: 12px;
        padding: 1.5em;
        margin: 1.5em 0;
        border-left: 4px solid #1976D2;
    }
    .quiz-header {
        display: flex;
        align-items: center;
        gap: 0.5em;
        margin-bottom: 1em;
    }
    .quiz-icon {
        font-size: 1.5em;
    }
    .quiz-title {
        font-weight: 600;
        color: #1565C0;
        font-size: 1.1em;
    }
    .quiz-question {
        font-size: 1.05em;
        color: #333;
        margin-bottom: 1em;
        line-height: 1.6;
    }
    .quiz-hint {
        background: #fff3e0;
        padding: 0.8em 1em;
        border-radius: 8px;
        font-size: 0.95em;
        color: #e65100;
        margin-bottom: 1em;
    }
    .quiz-hint::before {
        content: "Hint: ";
        font-weight: 600;
    }
    .quiz-answer-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1em;
        margin-top: 1em;
    }
    .quiz-answer-label {
        font-weight: 600;
        color: #388E3C;
        margin-bottom: 0.5em;
    }
    .quiz-answer-content {
        color: #333;
        line-height: 1.6;
    }
    .quiz-reveal-btn {
        background: #1976D2;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.95em;
        transition: background 0.2s;
    }
    .quiz-reveal-btn:hover {
        background: #1565C0;
    }
    .quiz-score-box {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 1em;
        margin-top: 1.5em;
        text-align: center;
    }
    .quiz-score-value {
        font-size: 2em;
        font-weight: 700;
        color: #388E3C;
    }
    .quiz-score-label {
        color: #666;
        font-size: 0.9em;
    }
    </style>
    """


def extract_quiz_questions(lesson: LessonContent) -> list[QuizQuestion]:
    """Extract comprehension check questions from lesson."""
    questions = []
    idx = 0
    for step in lesson.teaching_sequence:
        if isinstance(step, ComprehensionCheckStep):
            questions.append(QuizQuestion(
                index=idx,
                question=step.question,
                answer=step.answer,
                hint=step.hint,
            ))
            idx += 1
    return questions


def render_quiz_question(question: QuizQuestion, show_answer: bool = False) -> str:
    """
    Render a single quiz question.

    Args:
        question: QuizQuestion object
        show_answer: Whether to show the answer

    Returns:
        HTML string for the question
    """
    parts = ['<div class="quiz-container">']

    # Header
    parts.append('<div class="quiz-header">')
    parts.append('<span class="quiz-icon">?</span>')
    parts.append(f'<span class="quiz-title">Comprehension Check {question.index + 1}</span>')
    parts.append('</div>')

    # Question
    parts.append(f'<div class="quiz-question">{html.escape(question.question)}</div>')

    # Hint (if available)
    if question.hint:
        parts.append(f'<div class="quiz-hint">{html.escape(question.hint)}</div>')

    # Answer (if revealed)
    if show_answer:
        parts.append('<div class="quiz-answer-box">')
        parts.append('<div class="quiz-answer-label">Answer:</div>')
        parts.append(f'<div class="quiz-answer-content">{html.escape(question.answer)}</div>')
        parts.append('</div>')

    parts.append('</div>')
    return ''.join(parts)


def render_all_quiz_questions(
    questions: list[QuizQuestion],
    revealed_indices: Optional[set[int]] = None,
) -> str:
    """
    Render all quiz questions for a lesson.

    Args:
        questions: List of QuizQuestion objects
        revealed_indices: Set of indices where answer is revealed

    Returns:
        HTML string for all questions
    """
    if not questions:
        return ""

    revealed = revealed_indices or set()
    parts = [get_quiz_css()]

    for q in questions:
        show = q.index in revealed
        parts.append(render_quiz_question(q, show_answer=show))

    return ''.join(parts)


def render_quiz_for_streamlit(question: QuizQuestion) -> dict:
    """
    Prepare quiz question for Streamlit interactive display.

    Returns a dict with components to render in Streamlit.
    """
    return {
        "index": question.index,
        "question_html": f'<div class="quiz-question">{html.escape(question.question)}</div>',
        "hint_html": f'<div class="quiz-hint">{html.escape(question.hint)}</div>' if question.hint else None,
        "answer_html": f'<div class="quiz-answer-content">{html.escape(question.answer)}</div>',
        "question_text": question.question,
        "answer_text": question.answer,
        "hint_text": question.hint,
    }


def calculate_quiz_score(questions: list[QuizQuestion], correct_count: int) -> dict:
    """
    Calculate quiz score.

    Args:
        questions: Total questions
        correct_count: Number answered correctly

    Returns:
        Dict with score info
    """
    total = len(questions)
    if total == 0:
        return {"score": 1.0, "percent": 100, "correct": 0, "total": 0}

    score = correct_count / total
    return {
        "score": round(score, 2),
        "percent": round(score * 100),
        "correct": correct_count,
        "total": total,
    }


def render_quiz_score(score_info: dict) -> str:
    """Render quiz score display."""
    return f"""
    <div class="quiz-score-box">
        <div class="quiz-score-value">{score_info['percent']}%</div>
        <div class="quiz-score-label">{score_info['correct']} of {score_info['total']} correct</div>
    </div>
    """
