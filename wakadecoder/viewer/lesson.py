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


def markdown_to_html(text: str) -> str:
    """
    Convert basic markdown to HTML.
    Handles: **bold**, *italic*, `code`, and newlines.
    """
    # Escape HTML first (but preserve intentional HTML like <ruby>)
    # Don't escape if it contains ruby tags
    if '<ruby>' not in text and '<span' not in text:
        text = html.escape(text)

    # Convert markdown
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)  # *italic*
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)  # `code`
    text = text.replace('\n', '<br>')  # newlines

    return text

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
    "particle": "vocab-particle",      # Indigo blue
    "verb": "vocab-verb",              # Pine green
    "auxiliary": "vocab-auxiliary",    # Plum purple
    "adjective": "vocab-adjective",    # Autumn orange
    "noun": "vocab-noun",              # Ink black
}


def get_vocab_css() -> str:
    """Get CSS styles for the 'Ink & Paper' theme."""
    return """
    <style>
    /* ============================================
       INK & PAPER THEME - Classical Japanese Aesthetic
       ============================================ */

    /* -- Color Variables -- */
    :root {
        --washi-cream: #FAF8F5;
        --washi-warm: #F5F2ED;
        --sumi-ink: #2D2D2D;
        --sumi-light: #4A4A4A;
        --vermillion: #C53D43;
        --indigo: #4A5568;
        --pine-green: #5B8A72;
        --plum: #8B687F;
        --autumn-orange: #C17F59;
        --gold-accent: #D4A84B;
        --soft-blue: #6B8CAE;
    }

    /* -- Typography -- */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;500;600;700&display=swap');

    /* -- Poem Display -- */
    .poem-text {
        font-size: 1.6em;
        line-height: 2.4em;
        margin: 1.5em 0;
        font-family: "Noto Serif JP", "Yu Mincho", "Hiragino Mincho Pro", serif;
        color: var(--sumi-ink);
        letter-spacing: 0.05em;
    }

    .poem-container {
        background: linear-gradient(145deg, var(--washi-cream) 0%, var(--washi-warm) 100%);
        border: 1px solid #E8E4DE;
        border-left: 3px solid var(--vermillion);
        padding: 2em 2.5em;
        margin: 1.5em 0;
        border-radius: 2px 8px 8px 2px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        position: relative;
    }

    .poem-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
        border-radius: inherit;
    }

    .poem-romaji {
        font-size: 0.95em;
        color: var(--sumi-light);
        font-style: italic;
        margin-top: 1em;
        padding-top: 1em;
        border-top: 1px solid #E8E4DE;
        letter-spacing: 0.02em;
    }

    .poem-translation {
        font-size: 1.05em;
        color: var(--sumi-ink);
        margin-top: 1em;
        padding: 1em;
        background: rgba(255,255,255,0.5);
        border-radius: 4px;
        line-height: 1.7;
    }

    /* -- Vocabulary Highlighting -- */
    .vocab-word {
        cursor: pointer;
        border-radius: 2px;
        padding: 2px 4px;
        margin: 0 1px;
        transition: all 0.2s ease;
        position: relative;
        display: inline-block;
    }

    .vocab-word:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .vocab-particle {
        color: var(--soft-blue);
        background: rgba(107, 140, 174, 0.1);
        border-bottom: 2px solid var(--soft-blue);
    }

    .vocab-verb {
        color: var(--pine-green);
        background: rgba(91, 138, 114, 0.1);
        border-bottom: 2px solid var(--pine-green);
    }

    .vocab-auxiliary {
        color: var(--plum);
        background: rgba(139, 104, 127, 0.1);
        border-bottom: 2px solid var(--plum);
    }

    .vocab-adjective {
        color: var(--autumn-orange);
        background: rgba(193, 127, 89, 0.1);
        border-bottom: 2px solid var(--autumn-orange);
    }

    .vocab-noun {
        color: var(--sumi-ink);
        background: rgba(45, 45, 45, 0.05);
        border-bottom: 2px dotted var(--sumi-light);
    }

    /* -- Vocabulary Tooltip -- */
    .vocab-tooltip {
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: white;
        border: 1px solid #E8E4DE;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        z-index: 1000;
        white-space: nowrap;
        display: none;
        font-size: 0.9em;
        line-height: 1.5;
        min-width: 150px;
    }

    .vocab-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 8px solid transparent;
        border-top-color: white;
    }

    .vocab-word:hover .vocab-tooltip {
        display: block;
        animation: tooltipFadeIn 0.2s ease;
    }

    @keyframes tooltipFadeIn {
        from { opacity: 0; transform: translateX(-50%) translateY(4px); }
        to { opacity: 1; transform: translateX(-50%) translateY(0); }
    }

    .tooltip-reading {
        font-size: 1.2em;
        color: var(--sumi-ink);
        font-weight: 600;
        font-family: "Noto Serif JP", serif;
        margin-bottom: 4px;
    }

    .tooltip-meaning {
        color: var(--sumi-light);
        font-size: 0.95em;
    }

    .tooltip-cognate {
        color: var(--vermillion);
        font-size: 0.9em;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px dashed #E8E4DE;
    }

    .focus-highlight {
        background: linear-gradient(180deg, transparent 60%, rgba(212, 168, 75, 0.3) 60%);
        padding: 0 2px;
    }

    /* -- Vocabulary List -- */
    .vocab-list {
        margin-top: 1.5em;
        padding: 1.2em 1.5em;
        background: var(--washi-warm);
        border-radius: 8px;
        border: 1px solid #E8E4DE;
    }

    .vocab-list strong {
        color: var(--sumi-ink);
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .vocab-list-item {
        display: flex;
        align-items: baseline;
        margin: 0.6em 0;
        padding: 0.4em 0;
        font-size: 0.95em;
        border-bottom: 1px dotted #E8E4DE;
    }

    .vocab-list-item:last-child {
        border-bottom: none;
    }

    .vocab-list-word {
        font-weight: 600;
        min-width: 80px;
        color: var(--sumi-ink);
        font-family: "Noto Serif JP", serif;
    }

    .vocab-list-reading {
        color: var(--sumi-light);
        min-width: 90px;
        font-size: 0.9em;
    }

    .vocab-list-meaning {
        flex: 1;
        color: var(--indigo);
    }

    /* -- Grammar Box -- */
    .grammar-box {
        background: var(--washi-cream);
        border: 1px solid #E8E4DE;
        border-radius: 8px;
        padding: 1.5em 2em;
        margin: 1.5em 0;
        position: relative;
    }

    .grammar-box::before {
        content: '文法';
        position: absolute;
        top: -10px;
        left: 20px;
        background: var(--vermillion);
        color: white;
        font-size: 0.75em;
        padding: 2px 12px;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    .grammar-title {
        font-weight: 700;
        color: var(--sumi-ink);
        font-size: 1.2em;
        margin-bottom: 0.8em;
        margin-top: 0.5em;
    }

    .grammar-box p {
        margin: 0.8em 0;
        line-height: 1.7;
    }

    .grammar-box code {
        background: rgba(107, 140, 174, 0.15);
        color: var(--indigo);
        padding: 2px 8px;
        border-radius: 4px;
        font-family: "Noto Serif JP", monospace;
        font-size: 1.1em;
    }

    .grammar-box ul, .grammar-box ol {
        margin: 0.8em 0;
        padding-left: 1.5em;
    }

    .grammar-box li {
        margin: 0.4em 0;
        line-height: 1.6;
    }

    /* -- Step Containers -- */
    .step-container {
        margin: 2em 0;
        padding: 1.5em 2em;
        border-radius: 8px;
        line-height: 1.8;
    }

    .step-container strong {
        color: var(--sumi-ink);
    }

    .step-container code {
        background: rgba(0,0,0,0.06);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.95em;
    }

    .step-introduction {
        background: linear-gradient(135deg, #F0F7F4 0%, #E8F4EC 100%);
        border-left: 4px solid var(--pine-green);
    }

    .step-grammar {
        background: linear-gradient(135deg, #F0F4F8 0%, #E8EEF4 100%);
        border-left: 4px solid var(--soft-blue);
    }

    .step-contrast {
        background: linear-gradient(135deg, #FDF8F4 0%, #FAF0E8 100%);
        border-left: 4px solid var(--autumn-orange);
    }

    .step-summary {
        background: linear-gradient(135deg, #F8F4F7 0%, #F2ECF0 100%);
        border-left: 4px solid var(--plum);
        position: relative;
    }

    .step-summary::before {
        content: '📝';
        position: absolute;
        top: 1em;
        right: 1em;
        font-size: 1.5em;
        opacity: 0.3;
    }

    /* -- Ruby (Furigana) -- */
    ruby {
        ruby-align: center;
    }

    rt {
        font-size: 0.55em;
        color: var(--sumi-light);
        font-weight: 400;
    }

    /* -- Forward References -- */
    .forward-refs {
        margin-top: 2em;
        padding: 1.2em 1.5em;
        background: linear-gradient(135deg, #FFFCF5 0%, #FFF8E8 100%);
        border: 1px solid #F0E6D0;
        border-radius: 8px;
    }

    .forward-refs strong {
        color: var(--gold-accent);
    }

    .forward-refs ul {
        margin: 0.8em 0;
        padding-left: 1.2em;
    }

    .forward-refs li {
        margin: 0.5em 0;
        color: var(--sumi-light);
    }

    /* -- Details/Summary -- */
    details {
        margin-top: 1em;
    }

    details summary {
        cursor: pointer;
        color: var(--indigo);
        font-size: 0.9em;
        padding: 0.5em 0;
        transition: color 0.2s;
    }

    details summary:hover {
        color: var(--vermillion);
    }

    details[open] summary {
        margin-bottom: 0.5em;
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
    """Render a poem presentation step with interactive colored vocabulary."""
    parts = ['<div class="poem-container">']

    # Extract plain text from furigana HTML
    plain_text = re.sub(r'<ruby>([^<]+)<rt>[^<]+</rt></ruby>', r'\1', step.display.text_with_furigana)
    plain_text = re.sub(r'<[^>]+>', '', plain_text)

    # Main display: colored poem with hover tooltips
    vocab_with_spans = [v for v in step.vocabulary if v.span]
    if vocab_with_spans:
        # Interactive colored poem as main display
        parts.append(render_poem_with_vocabulary(plain_text, step.vocabulary, step.focus_highlight))
    else:
        # Fallback to furigana if no spans
        parts.append(f'<div class="poem-text">{step.display.text_with_furigana}</div>')

    # Romaji
    parts.append(f'<div class="poem-romaji">{html.escape(step.display.romaji)}</div>')

    # Translation
    parts.append(f'<div class="poem-translation">{html.escape(step.display.translation)}</div>')

    parts.append('</div>')

    # Collapsible vocabulary list as reference
    if step.vocabulary:
        parts.append('<details><summary style="cursor:pointer;color:#666;">Show vocabulary list</summary>')
        parts.append(render_vocab_list(step.vocabulary))
        parts.append('</details>')

    return ''.join(parts)


def render_introduction(step: IntroductionStep) -> str:
    """Render an introduction step."""
    content = markdown_to_html(step.content)
    return f'<div class="step-container step-introduction">{content}</div>'


def render_grammar_spotlight(step: GrammarSpotlightStep) -> str:
    """Render a grammar spotlight step."""
    content = markdown_to_html(step.content)
    evidence = f'<p style="font-size:0.9em;color:#666;margin-top:0.5em;"><em>{html.escape(step.evidence)}</em></p>' if step.evidence else ''
    return f'<div class="step-container step-grammar">{content}{evidence}</div>'


def render_contrast_example(step: ContrastExampleStep) -> str:
    """Render a contrast example step."""
    content = markdown_to_html(step.content)
    return f'<div class="step-container step-contrast"><strong>Contrast:</strong><br>{content}</div>'


def render_summary(step: SummaryStep) -> str:
    """Render a summary step."""
    content = markdown_to_html(step.content)
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
