"""
Lesson renderer - Generate HTML/Streamlit components for lesson display.

Features:
- Interactive vocabulary hover with color-coded word types
- Ruby (furigana) text rendering
- Teaching sequence step rendering
- Poem display with span-based vocabulary highlighting
"""

from typing import Optional
import html
import re

from wakadecoder.schemas import (
    LessonContent,
    VocabularyItem,
    PoemPresentationStep,
    IntroductionStep,
    GrammarSpotlightStep,
    ContrastExampleStep,
    ComprehensionCheckStep,
    SummaryStep,
    TeachingStep,
)


# Word type to CSS class mapping for color coding
WORD_TYPE_COLORS = {
    "particle": "vocab-particle",      # Blue
    "verb": "vocab-verb",              # Green
    "auxiliary": "vocab-auxiliary",    # Purple
    "adjective": "vocab-adjective",    # Orange
    "noun": "vocab-noun",              # Default
}


def get_vocab_css() -> str:
    """Get CSS styles for vocabulary display."""
    return """
    <style>
    .poem-text {
        font-size: 1.4em;
        line-height: 2.2em;
        margin: 1em 0;
        font-family: "Noto Serif JP", "Yu Mincho", serif;
    }
    .vocab-word {
        cursor: pointer;
        border-radius: 3px;
        padding: 0 2px;
        transition: background-color 0.2s;
        position: relative;
    }
    .vocab-word:hover {
        background-color: rgba(100, 100, 100, 0.15);
    }
    .vocab-particle {
        color: #1976D2;
        text-decoration: underline dotted #1976D2;
    }
    .vocab-verb {
        color: #388E3C;
        text-decoration: underline dotted #388E3C;
    }
    .vocab-auxiliary {
        color: #7B1FA2;
        text-decoration: underline dotted #7B1FA2;
    }
    .vocab-adjective {
        color: #F57C00;
        text-decoration: underline dotted #F57C00;
    }
    .vocab-noun {
        color: #455A64;
    }
    .vocab-tooltip {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: white;
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 8px 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        z-index: 1000;
        white-space: nowrap;
        display: none;
        font-size: 0.85em;
        line-height: 1.4;
    }
    .vocab-word:hover .vocab-tooltip {
        display: block;
    }
    .tooltip-reading {
        font-size: 1.1em;
        color: #333;
        font-weight: 500;
    }
    .tooltip-meaning {
        color: #666;
        margin-top: 2px;
    }
    .tooltip-cognate {
        color: #1976D2;
        font-size: 0.9em;
        margin-top: 4px;
        font-style: italic;
    }
    .focus-highlight {
        background-color: rgba(255, 235, 59, 0.4);
        border-radius: 3px;
        padding: 0 2px;
    }
    .poem-container {
        background: #fafafa;
        border-left: 4px solid #1976D2;
        padding: 1em 1.5em;
        margin: 1em 0;
        border-radius: 0 8px 8px 0;
    }
    .poem-romaji {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
        margin-top: 0.5em;
    }
    .poem-translation {
        font-size: 1em;
        color: #333;
        margin-top: 0.8em;
        padding-top: 0.8em;
        border-top: 1px dashed #ddd;
    }
    .vocab-list {
        margin-top: 1em;
        padding: 1em;
        background: #f5f5f5;
        border-radius: 8px;
    }
    .vocab-list-item {
        display: flex;
        margin: 0.5em 0;
        font-size: 0.95em;
    }
    .vocab-list-word {
        font-weight: 500;
        min-width: 80px;
    }
    .vocab-list-reading {
        color: #666;
        min-width: 80px;
    }
    .vocab-list-meaning {
        flex: 1;
    }
    .grammar-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 8px;
        padding: 1.2em;
        margin: 1em 0;
    }
    .grammar-title {
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 0.5em;
    }
    .step-container {
        margin: 1.5em 0;
        padding: 1em;
        border-radius: 8px;
    }
    .step-introduction {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
    }
    .step-grammar {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .step-contrast {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
    }
    .step-summary {
        background: #f3e5f5;
        border-left: 4px solid #9C27B0;
    }
    ruby {
        ruby-align: center;
    }
    rt {
        font-size: 0.6em;
        color: #888;
    }
    </style>
    """


def detect_word_type(vocab: VocabularyItem) -> str:
    """
    Detect word type from vocabulary item for color coding.

    Uses meaning hints and common patterns to classify words.
    """
    word = vocab.word.lower()
    meaning = vocab.meaning.lower()

    # Particle detection
    particles = {"の", "は", "が", "を", "に", "で", "と", "も", "か", "へ", "や", "より", "から", "まで", "ね", "よ", "ぞ", "ぜ", "な", "わ"}
    if word in particles or "particle" in meaning or "marker" in meaning:
        return "particle"

    # Auxiliary detection
    if "auxiliary" in meaning or word in {"けり", "たり", "なり", "べし", "らむ", "ける", "つ", "ぬ", "き", "む", "ず"}:
        return "auxiliary"

    # Verb detection
    if "to " in meaning and not "want to" in meaning[:10]:
        return "verb"

    # Adjective detection
    if meaning.endswith("y") or meaning.endswith("ful") or meaning.endswith("ive") or "adjective" in meaning:
        return "adjective"

    # Default to noun
    return "noun"


def render_poem_with_vocabulary(
    poem_text: str,
    vocabulary: list[VocabularyItem],
    focus_span: Optional[list[int]] = None,
) -> str:
    """
    Render poem text with interactive vocabulary spans.

    Args:
        poem_text: Raw poem text
        vocabulary: List of vocabulary items with spans
        focus_span: Optional [start, end) span to highlight grammar point

    Returns:
        HTML string with vocabulary spans and tooltips
    """
    if not vocabulary:
        # No vocabulary - just return escaped text
        return f'<div class="poem-text">{html.escape(poem_text)}</div>'

    # Sort vocabulary by span start position
    vocab_with_spans = [v for v in vocabulary if v.span]
    vocab_without_spans = [v for v in vocabulary if not v.span]
    vocab_with_spans.sort(key=lambda v: v.span[0])

    # Build HTML by processing text character by character
    result = []
    pos = 0
    vocab_idx = 0

    while pos < len(poem_text):
        # Check if we're at a vocabulary span
        if vocab_idx < len(vocab_with_spans):
            vocab = vocab_with_spans[vocab_idx]
            start, end = vocab.span

            if pos == start:
                # Render vocabulary span
                word_type = detect_word_type(vocab)
                css_class = WORD_TYPE_COLORS.get(word_type, "vocab-noun")

                # Check if this span overlaps with focus highlight
                has_focus = False
                if focus_span:
                    focus_start, focus_end = focus_span
                    if start < focus_end and end > focus_start:
                        has_focus = True

                span_text = poem_text[start:end]
                tooltip = render_vocab_tooltip(vocab)

                focus_class = " focus-highlight" if has_focus else ""
                result.append(
                    f'<span class="vocab-word {css_class}{focus_class}" data-idx="{vocab_idx}">'
                    f'{html.escape(span_text)}{tooltip}</span>'
                )
                pos = end
                vocab_idx += 1
                continue

        # Check for focus highlight without vocabulary
        if focus_span:
            focus_start, focus_end = focus_span
            if pos == focus_start:
                # Find end of focus or next vocab span
                end_pos = focus_end
                if vocab_idx < len(vocab_with_spans):
                    next_vocab_start = vocab_with_spans[vocab_idx].span[0]
                    end_pos = min(end_pos, next_vocab_start)
                result.append(f'<span class="focus-highlight">{html.escape(poem_text[pos:end_pos])}</span>')
                pos = end_pos
                continue

        # Regular character
        result.append(html.escape(poem_text[pos]))
        pos += 1

    html_text = ''.join(result)
    return f'<div class="poem-text">{html_text}</div>'


def render_vocab_tooltip(vocab: VocabularyItem) -> str:
    """Render tooltip HTML for a vocabulary item."""
    parts = [f'<div class="tooltip-reading">{html.escape(vocab.reading)}</div>']
    parts.append(f'<div class="tooltip-meaning">{html.escape(vocab.meaning)}</div>')

    if vocab.chinese_cognate_note:
        parts.append(f'<div class="tooltip-cognate">{html.escape(vocab.chinese_cognate_note)}</div>')

    return f'<div class="vocab-tooltip">{"".join(parts)}</div>'


def render_vocab_list(vocabulary: list[VocabularyItem]) -> str:
    """Render vocabulary as a fallback list (for items without spans)."""
    if not vocabulary:
        return ""

    items = []
    for vocab in vocabulary:
        cognate = f' <span style="color:#1976D2">({html.escape(vocab.chinese_cognate_note)})</span>' if vocab.chinese_cognate_note else ''
        items.append(
            f'<div class="vocab-list-item">'
            f'<span class="vocab-list-word">{html.escape(vocab.word)}</span>'
            f'<span class="vocab-list-reading">{html.escape(vocab.reading)}</span>'
            f'<span class="vocab-list-meaning">{html.escape(vocab.meaning)}{cognate}</span>'
            f'</div>'
        )

    return f'<div class="vocab-list"><strong>Vocabulary:</strong>{"".join(items)}</div>'


def render_poem_presentation(step: PoemPresentationStep) -> str:
    """Render a poem presentation step."""
    parts = ['<div class="poem-container">']

    # Furigana text (already has ruby tags)
    parts.append(f'<div class="poem-text">{step.display.text_with_furigana}</div>')

    # Interactive vocabulary version (plain text with spans)
    # Extract plain text from furigana HTML for vocabulary highlighting
    plain_text = re.sub(r'<ruby>([^<]+)<rt>[^<]+</rt></ruby>', r'\1', step.display.text_with_furigana)
    plain_text = re.sub(r'<[^>]+>', '', plain_text)

    # Only show vocabulary-highlighted version if we have spans
    vocab_with_spans = [v for v in step.vocabulary if v.span]
    if vocab_with_spans:
        interactive = render_poem_with_vocabulary(plain_text, step.vocabulary, step.focus_highlight)
        parts.append(f'<details><summary>Interactive vocabulary view</summary>{interactive}</details>')

    # Romaji
    parts.append(f'<div class="poem-romaji">{html.escape(step.display.romaji)}</div>')

    # Translation
    parts.append(f'<div class="poem-translation">{html.escape(step.display.translation)}</div>')

    parts.append('</div>')

    # Vocabulary list (always shown as reference)
    if step.vocabulary:
        parts.append(render_vocab_list(step.vocabulary))

    return ''.join(parts)


def render_introduction(step: IntroductionStep) -> str:
    """Render an introduction step."""
    content = step.content.replace('\n', '<br>')
    return f'<div class="step-container step-introduction">{content}</div>'


def render_grammar_spotlight(step: GrammarSpotlightStep) -> str:
    """Render a grammar spotlight step."""
    content = step.content.replace('\n', '<br>')
    evidence = f'<p style="font-size:0.9em;color:#666;margin-top:0.5em;"><em>{html.escape(step.evidence)}</em></p>' if step.evidence else ''
    return f'<div class="step-container step-grammar">{content}{evidence}</div>'


def render_contrast_example(step: ContrastExampleStep) -> str:
    """Render a contrast example step."""
    content = step.content.replace('\n', '<br>')
    return f'<div class="step-container step-contrast"><strong>Contrast:</strong><br>{content}</div>'


def render_summary(step: SummaryStep) -> str:
    """Render a summary step."""
    content = step.content.replace('\n', '<br>')
    return f'<div class="step-container step-summary"><strong>Summary:</strong><br>{content}</div>'


def render_teaching_step(step: TeachingStep) -> str:
    """Render any teaching step."""
    if isinstance(step, IntroductionStep):
        return render_introduction(step)
    elif isinstance(step, PoemPresentationStep):
        return render_poem_presentation(step)
    elif isinstance(step, GrammarSpotlightStep):
        return render_grammar_spotlight(step)
    elif isinstance(step, ContrastExampleStep):
        return render_contrast_example(step)
    elif isinstance(step, ComprehensionCheckStep):
        # Quiz handled separately
        return ""
    elif isinstance(step, SummaryStep):
        return render_summary(step)
    else:
        return f"<p>Unknown step type: {type(step)}</p>"


def render_grammar_explanation(lesson: LessonContent) -> str:
    """Render the grammar explanation section."""
    ge = lesson.grammar_explanation
    parts = ['<div class="grammar-box">']
    parts.append(f'<div class="grammar-title">{html.escape(lesson.grammar_point)}</div>')
    parts.append(f'<p><strong>Concept:</strong> {html.escape(ge.concept)}</p>')

    if ge.formation:
        parts.append(f'<p><strong>Formation:</strong> <code>{html.escape(ge.formation)}</code></p>')

    if ge.variations:
        parts.append('<p><strong>Variations:</strong></p><ul>')
        for v in ge.variations:
            parts.append(f'<li>{html.escape(v)}</li>')
        parts.append('</ul>')

    if ge.common_confusions:
        parts.append('<p><strong>Common Confusions:</strong></p><ul>')
        for c in ge.common_confusions:
            parts.append(f'<li>{html.escape(c)}</li>')
        parts.append('</ul>')

    parts.append('</div>')
    return ''.join(parts)


def render_forward_references(lesson: LessonContent) -> str:
    """Render forward references section."""
    if not lesson.forward_references:
        return ""

    parts = ['<div style="margin-top:1.5em;padding:1em;background:#fff8e1;border-radius:8px;">']
    parts.append('<strong>Coming Up:</strong><ul>')
    for ref in lesson.forward_references:
        parts.append(f'<li><strong>{html.escape(ref.point)}</strong>: {html.escape(ref.note)}</li>')
    parts.append('</ul></div>')
    return ''.join(parts)


def render_lesson(lesson: LessonContent) -> str:
    """
    Render complete lesson content as HTML.

    Args:
        lesson: LessonContent object

    Returns:
        Complete HTML string for the lesson
    """
    parts = [get_vocab_css()]

    # Title and summary
    parts.append(f'<h1>{html.escape(lesson.lesson_title)}</h1>')
    parts.append(f'<p style="color:#666;font-style:italic;">{html.escape(lesson.lesson_summary)}</p>')

    # Grammar explanation
    parts.append(render_grammar_explanation(lesson))

    # Teaching sequence
    for step in lesson.teaching_sequence:
        rendered = render_teaching_step(step)
        if rendered:
            parts.append(rendered)

    # Forward references
    parts.append(render_forward_references(lesson))

    return ''.join(parts)


def get_comprehension_checks(lesson: LessonContent) -> list[ComprehensionCheckStep]:
    """Extract comprehension check steps from lesson."""
    return [
        step for step in lesson.teaching_sequence
        if isinstance(step, ComprehensionCheckStep)
    ]
