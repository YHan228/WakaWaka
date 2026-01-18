"""
Crash Course Viewer - Rendering components for the Japanese crash course.

This module provides:
- CSS styling for crash course pages
- Step rendering functions for each step type
- Quiz rendering for comprehension checks
"""

import json
import re
from pathlib import Path
from typing import Optional


def markdown_to_html(text: str) -> str:
    """Convert basic markdown to HTML."""
    if not text:
        return ""
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic: *text* or _text_ (but not in the middle of words)
    text = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'<em>\1</em>', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<em>\1</em>', text)
    # Code: `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Convert bullet points: "   - text" to list item style
    text = re.sub(r'^(\s*)- (.+)$', r'\1<span style="margin-left:1em;">• \2</span>', text, flags=re.MULTILINE)
    # Line breaks (preserve structure)
    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')
    return text

from wakawaka.schemas.crash_course import (
    CrashCourse,
    CrashCourseLesson,
    CrashCourseStep,
    TextStep,
    KanaChartStep,
    VocabIntroStep,
    PracticeStep,
    ExampleSentenceStep,
    MiniPoemStep,
    SummaryStep,
    TipStep,
    CrashCourseQuiz,
)


def get_crash_course_css() -> str:
    """Get CSS for crash course styling."""
    return """
    <style>
    /* ============================================
       CRASH COURSE STYLING
       ============================================ */

    /* -- Lesson Container -- */
    .crash-lesson {
        max-width: 800px;
        margin: 0 auto;
    }

    .crash-lesson-header {
        text-align: center;
        margin-bottom: 2em;
        padding-bottom: 1.5em;
        border-bottom: 2px solid var(--vermillion);
    }

    .crash-lesson-number {
        font-size: 0.9em;
        color: var(--vermillion);
        font-weight: 600;
        letter-spacing: 0.1em;
        margin-bottom: 0.5em;
    }

    .crash-lesson-title {
        font-family: 'Noto Serif SC', 'Noto Serif JP', serif;
        font-size: 2em;
        color: var(--sumi-ink);
        margin: 0.3em 0;
    }

    .crash-lesson-subtitle {
        font-size: 1.1em;
        color: var(--sumi-light);
        font-style: italic;
    }

    .crash-objectives {
        background: linear-gradient(135deg, #F8F6F3 0%, #F5F2ED 100%);
        border-radius: 12px;
        padding: 1.2em 1.5em;
        margin: 1.5em 0;
    }

    .crash-objectives-title {
        font-weight: 600;
        color: var(--pine-green);
        margin-bottom: 0.8em;
        font-size: 0.95em;
    }

    .crash-objectives ul {
        margin: 0;
        padding-left: 1.2em;
    }

    .crash-objectives li {
        margin: 0.4em 0;
        color: var(--sumi-ink);
    }

    /* -- Step Containers -- */
    .crash-step {
        margin: 2em 0;
        padding: 1.5em;
        background: var(--washi-cream);
        border-radius: 12px;
        border: 1px solid #E8E4DE;
    }

    .crash-step-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.3em;
        color: var(--sumi-ink);
        margin-bottom: 1em;
        padding-bottom: 0.5em;
        border-bottom: 1px solid #E8E4DE;
    }

    /* -- Text Step -- */
    .crash-text-content {
        line-height: 1.8;
        color: var(--sumi-ink);
    }

    .crash-text-content p {
        margin: 1em 0;
    }

    /* -- Kana Chart -- */
    .crash-kana-chart {
        overflow-x: auto;
    }

    .crash-kana-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 4px;
        margin: 1em 0;
    }

    .crash-kana-cell {
        background: white;
        border-radius: 8px;
        padding: 0.8em;
        text-align: center;
        transition: all 0.2s ease;
        cursor: default;
        min-width: 60px;
    }

    .crash-kana-cell:hover {
        background: #FFF8F0;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .crash-kana-cell.highlighted {
        background: rgba(197, 61, 67, 0.1);
        border: 2px solid var(--vermillion);
    }

    .crash-kana-main {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.8em;
        color: var(--sumi-ink);
        display: block;
    }

    .crash-kana-romaji {
        font-size: 0.85em;
        color: var(--sumi-light);
        margin-top: 0.3em;
        display: block;
    }

    .crash-kana-mnemonic {
        font-size: 0.75em;
        color: var(--pine-green);
        margin-top: 0.2em;
        display: none;
    }

    .crash-kana-cell:hover .crash-kana-mnemonic {
        display: block;
    }

    /* -- Vocab Intro -- */
    .crash-vocab-list {
        display: grid;
        gap: 1em;
    }

    .crash-vocab-item {
        background: white;
        border-radius: 10px;
        padding: 1em 1.2em;
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 1em;
        align-items: center;
        border: 1px solid #E8E4DE;
        transition: all 0.2s ease;
    }

    .crash-vocab-item:hover {
        border-color: var(--soft-blue);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .crash-vocab-japanese {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.4em;
        color: var(--sumi-ink);
    }

    .crash-vocab-reading {
        font-size: 0.9em;
        color: var(--vermillion);
    }

    .crash-vocab-details {
        flex: 1;
    }

    .crash-vocab-meaning {
        font-weight: 500;
        color: var(--sumi-ink);
    }

    .crash-vocab-note {
        font-size: 0.85em;
        color: var(--sumi-light);
        margin-top: 0.3em;
    }

    .crash-vocab-chinese {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.2em;
        color: var(--pine-green);
        padding: 0.3em 0.6em;
        background: rgba(91, 138, 114, 0.1);
        border-radius: 6px;
    }

    /* -- Practice Step -- */
    .crash-practice {
        background: linear-gradient(135deg, #F5F8F6 0%, #F0F5F2 100%);
    }

    .crash-practice-instruction {
        font-size: 1.05em;
        color: var(--sumi-ink);
        margin-bottom: 1.2em;
        padding: 0.8em;
        background: white;
        border-radius: 8px;
        border-left: 3px solid var(--pine-green);
    }

    .crash-practice-items {
        display: grid;
        gap: 0.8em;
    }

    .crash-practice-item {
        background: white;
        border-radius: 8px;
        padding: 1em;
        border: 1px solid #E8E4DE;
    }

    .crash-practice-hint {
        margin-top: 1em;
        padding: 0.8em;
        background: rgba(212, 168, 75, 0.15);
        border-left: 3px solid var(--gold-accent);
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
        color: var(--sumi-light);
    }

    /* Match items (left → right) */
    .crash-practice-match-item {
        display: flex;
        align-items: center;
        gap: 1em;
        padding: 0.8em 1em;
        background: white;
        border-radius: 8px;
        margin-bottom: 0.5em;
        border: 1px solid #E8E4DE;
    }

    .crash-match-left {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.1em;
        color: var(--sumi-ink);
        min-width: 120px;
    }

    .crash-match-arrow {
        color: var(--pine-green);
        font-weight: bold;
    }

    .crash-match-right {
        color: var(--sumi-light);
    }

    /* Question items */
    .crash-practice-question {
        background: white;
        border-radius: 10px;
        padding: 1.2em;
        margin-bottom: 1em;
        border: 1px solid #E8E4DE;
    }

    .crash-question-text {
        font-size: 1.05em;
        color: var(--sumi-ink);
        margin-bottom: 0.8em;
    }

    .crash-quiz-options {
        display: grid;
        gap: 0.5em;
        margin: 0.8em 0;
    }

    .crash-quiz-option {
        padding: 0.6em 1em;
        background: #F8F6F3;
        border-radius: 6px;
        border: 1px solid #E8E4DE;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .crash-quiz-option:hover {
        background: #F0EDE8;
        border-color: var(--soft-blue);
    }

    /* Answer reveal */
    .crash-practice-answer {
        margin-top: 0.8em;
    }

    .crash-practice-answer summary {
        cursor: pointer;
        color: var(--soft-blue);
        font-size: 0.9em;
        padding: 0.4em 0;
    }

    .crash-practice-answer summary:hover {
        color: var(--vermillion);
    }

    .crash-answer-content {
        background: rgba(91, 138, 114, 0.1);
        border-radius: 6px;
        padding: 0.8em 1em;
        margin-top: 0.5em;
        color: var(--sumi-ink);
    }

    .crash-answer-content em {
        color: var(--sumi-light);
        font-size: 0.9em;
    }

    /* Fill blank */
    .crash-practice-fill {
        background: white;
        border-radius: 8px;
        padding: 1em;
        margin-bottom: 0.8em;
        border: 1px solid #E8E4DE;
    }

    .crash-fill-sentence {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.1em;
        color: var(--sumi-ink);
    }

    /* -- Example Sentence -- */
    .crash-sentence-box {
        background: white;
        border-radius: 10px;
        padding: 1.5em;
        margin: 1em 0;
    }

    .crash-sentence-main {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.5em;
        color: var(--sumi-ink);
        text-align: center;
        margin-bottom: 0.5em;
    }

    .crash-sentence-reading {
        text-align: center;
        color: var(--vermillion);
        font-size: 1em;
        margin-bottom: 0.8em;
    }

    .crash-sentence-translation {
        text-align: center;
        color: var(--sumi-light);
        font-style: italic;
        padding-bottom: 1em;
        border-bottom: 1px dashed #E8E4DE;
    }

    .crash-sentence-breakdown {
        margin-top: 1.2em;
    }

    .crash-breakdown-title {
        font-weight: 600;
        color: var(--indigo);
        margin-bottom: 0.8em;
        font-size: 0.9em;
    }

    .crash-breakdown-parts {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8em;
        justify-content: center;
    }

    .crash-breakdown-part {
        background: #F8F6F3;
        border-radius: 8px;
        padding: 0.6em 1em;
        text-align: center;
    }

    .crash-breakdown-word {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.1em;
        color: var(--sumi-ink);
    }

    .crash-breakdown-role {
        font-size: 0.75em;
        color: var(--vermillion);
        font-weight: 500;
        margin-top: 0.2em;
    }

    .crash-breakdown-meaning {
        font-size: 0.8em;
        color: var(--sumi-light);
    }

    .crash-grammar-note {
        margin-top: 1em;
        padding: 0.8em 1em;
        background: rgba(107, 140, 174, 0.1);
        border-radius: 8px;
        font-size: 0.9em;
        color: var(--indigo);
    }

    /* -- Mini Poem -- */
    .crash-poem-box {
        background: linear-gradient(135deg, #FAF8F5 0%, #F8F4EF 100%);
        border: 2px solid #E8E4DE;
        border-radius: 12px;
        padding: 2em;
        text-align: center;
    }

    .crash-poem-text {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.6em;
        color: var(--sumi-ink);
        line-height: 2;
        margin-bottom: 1em;
    }

    .crash-poem-reading {
        font-size: 1em;
        color: var(--vermillion);
        margin-bottom: 0.8em;
    }

    .crash-poem-translation {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.1em;
        color: var(--sumi-light);
        font-style: italic;
        margin-bottom: 1.5em;
        padding-bottom: 1em;
        border-bottom: 1px dashed #E8E4DE;
    }

    .crash-poem-analysis {
        text-align: left;
        line-height: 1.7;
        color: var(--sumi-ink);
        background: white;
        padding: 1.2em;
        border-radius: 8px;
        margin-top: 1em;
    }

    /* -- Summary Step -- */
    .crash-summary {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFF5EA 100%);
        border: 2px solid var(--gold-accent);
    }

    .crash-summary-points {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .crash-summary-points li {
        padding: 0.8em 0;
        padding-left: 2em;
        position: relative;
        border-bottom: 1px dashed rgba(212, 168, 75, 0.3);
    }

    .crash-summary-points li:last-child {
        border-bottom: none;
    }

    .crash-summary-points li::before {
        content: "✓";
        position: absolute;
        left: 0;
        color: var(--pine-green);
        font-weight: bold;
    }

    .crash-summary-next {
        margin-top: 1.2em;
        padding-top: 1em;
        border-top: 2px solid var(--gold-accent);
        color: var(--sumi-light);
        font-style: italic;
    }

    /* -- Tip Step -- */
    .crash-tip {
        border-left: 4px solid;
        padding-left: 1.2em;
    }

    .crash-tip.learning {
        border-color: var(--soft-blue);
        background: rgba(107, 140, 174, 0.05);
    }

    .crash-tip.cultural {
        border-color: var(--plum);
        background: rgba(139, 104, 127, 0.05);
    }

    .crash-tip.memory {
        border-color: var(--gold-accent);
        background: rgba(212, 168, 75, 0.05);
    }

    .crash-tip.comparison {
        border-color: var(--pine-green);
        background: rgba(91, 138, 114, 0.05);
    }

    .crash-tip-icon {
        font-size: 1.2em;
        margin-right: 0.5em;
    }

    /* -- Quiz Styles -- */
    .crash-quiz {
        margin-top: 2em;
        padding: 1.5em;
        background: linear-gradient(135deg, #F8F6F3 0%, #F5F2ED 100%);
        border-radius: 12px;
        border: 2px solid #E8E4DE;
    }

    .crash-quiz-header {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.3em;
        color: var(--indigo);
        margin-bottom: 1em;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }

    .crash-quiz-question {
        background: white;
        border-radius: 10px;
        padding: 1.2em;
        margin-bottom: 1em;
    }

    .crash-quiz-q-text {
        font-size: 1.05em;
        color: var(--sumi-ink);
        margin-bottom: 1em;
    }

    .crash-quiz-options {
        display: grid;
        gap: 0.5em;
    }

    .crash-quiz-option {
        padding: 0.8em 1em;
        background: #F8F6F3;
        border-radius: 8px;
        border: 1px solid #E8E4DE;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .crash-quiz-option:hover {
        background: #F0EDE8;
        border-color: var(--soft-blue);
    }

    .crash-quiz-option.selected {
        background: rgba(107, 140, 174, 0.15);
        border-color: var(--soft-blue);
    }

    .crash-quiz-option.correct {
        background: rgba(91, 138, 114, 0.15);
        border-color: var(--pine-green);
    }

    .crash-quiz-option.incorrect {
        background: rgba(197, 61, 67, 0.1);
        border-color: var(--vermillion);
    }

    /* -- Navigation -- */
    .crash-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 2em;
        padding-top: 1.5em;
        border-top: 2px solid #E8E4DE;
    }

    .crash-nav-progress {
        color: var(--sumi-light);
        font-size: 0.9em;
    }

    /* -- Overview Page -- */
    .crash-overview {
        max-width: 900px;
        margin: 0 auto;
    }

    .crash-overview-header {
        text-align: center;
        margin-bottom: 2em;
    }

    .crash-overview-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.2em;
        color: var(--sumi-ink);
        margin-bottom: 0.3em;
    }

    .crash-overview-subtitle {
        font-size: 1.2em;
        color: var(--sumi-light);
    }

    .crash-overview-description {
        text-align: center;
        max-width: 600px;
        margin: 1.5em auto;
        line-height: 1.8;
        color: var(--sumi-ink);
    }

    .crash-lesson-cards {
        display: grid;
        gap: 1em;
        margin: 2em 0;
    }

    .crash-lesson-card {
        background: var(--washi-cream);
        border: 1px solid #E8E4DE;
        border-radius: 12px;
        padding: 1.2em 1.5em;
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 1em;
        align-items: center;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .crash-lesson-card:hover {
        border-color: var(--vermillion);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transform: translateY(-2px);
    }

    .crash-lesson-card.completed {
        border-color: var(--pine-green);
    }

    .crash-lesson-card.locked {
        opacity: 0.6;
        cursor: not-allowed;
    }

    .crash-card-number {
        width: 40px;
        height: 40px;
        background: var(--vermillion);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
    }

    .crash-card-number.completed {
        background: var(--pine-green);
    }

    .crash-card-info h4 {
        margin: 0;
        color: var(--sumi-ink);
        font-family: 'Noto Serif SC', serif;
    }

    .crash-card-info p {
        margin: 0.3em 0 0;
        font-size: 0.9em;
        color: var(--sumi-light);
    }

    .crash-card-duration {
        font-size: 0.85em;
        color: var(--sumi-light);
        white-space: nowrap;
    }

    </style>
    """


def render_text_step(step: TextStep) -> str:
    """Render a text content step."""
    title_html = f'<div class="crash-step-title">{markdown_to_html(step.title)}</div>' if step.title else ''
    content_html = markdown_to_html(step.content)
    return f'''
    <div class="crash-step crash-text">
        {title_html}
        <div class="crash-text-content">
            <p>{content_html}</p>
        </div>
    </div>
    '''


def render_kana_chart_step(step: KanaChartStep) -> str:
    """Render a kana chart step."""
    rows = step.rows
    cells_html = []

    # Detect format: flat list of cells or nested list of rows
    is_flat = rows and isinstance(rows[0], dict)

    if is_flat:
        # Flat format: each item in rows is a cell dict
        # Display as a grid of 5 columns
        for idx, cell in enumerate(rows):
            if cell:
                mnemonic = f'<span class="crash-kana-mnemonic">{cell.get("mnemonic", "")}</span>' if cell.get("mnemonic") else ''
                pitch = f'<span class="crash-kana-pitch" style="font-size:0.7em;color:var(--vermillion);">{cell.get("pitch", "")}</span>' if cell.get("pitch") and cell.get("pitch") != '-' else ''
                cells_html.append(f'''
                <td class="crash-kana-cell">
                    <span class="crash-kana-main">{cell.get("kana", "")}</span>
                    <span class="crash-kana-romaji">{cell.get("romaji", "")}</span>
                    {pitch}
                    {mnemonic}
                </td>
                ''')
            else:
                cells_html.append('<td class="crash-kana-cell"></td>')

        # Arrange into rows of 5 columns
        rows_html = []
        for i in range(0, len(cells_html), 5):
            row_cells = cells_html[i:i+5]
            # Pad if needed
            while len(row_cells) < 5:
                row_cells.append('<td class="crash-kana-cell"></td>')
            rows_html.append(f'<tr>{"".join(row_cells)}</tr>')
    else:
        # Nested format: each item in rows is a list of cells
        rows_html = []
        for row_idx, row in enumerate(rows):
            row_cells = []
            for cell in row:
                if cell:
                    highlighted = 'highlighted' if step.highlight_row == row_idx else ''
                    mnemonic = f'<span class="crash-kana-mnemonic">{cell.get("mnemonic", "")}</span>' if cell.get("mnemonic") else ''
                    row_cells.append(f'''
                    <td class="crash-kana-cell {highlighted}">
                        <span class="crash-kana-main">{cell.get("kana", "")}</span>
                        <span class="crash-kana-romaji">{cell.get("romaji", "")}</span>
                        {mnemonic}
                    </td>
                    ''')
                else:
                    row_cells.append('<td class="crash-kana-cell"></td>')
            rows_html.append(f'<tr>{"".join(row_cells)}</tr>')

    desc_html = f'<p>{markdown_to_html(step.description)}</p>' if step.description else ''

    return f'''
    <div class="crash-step crash-kana-chart">
        <div class="crash-step-title">{step.title}</div>
        {desc_html}
        <table class="crash-kana-table">
            {"".join(rows_html)}
        </table>
    </div>
    '''


def render_vocab_intro_step(step: VocabIntroStep) -> str:
    """Render a vocabulary introduction step."""
    items_html = []
    for item in step.items:
        note_html = f'<div class="crash-vocab-note">{item.get("note", "")}</div>' if item.get("note") else ''
        chinese_html = f'<div class="crash-vocab-chinese">{item.get("chinese", "")}</div>' if item.get("chinese") else ''

        items_html.append(f'''
        <div class="crash-vocab-item">
            <div>
                <span class="crash-vocab-japanese">{item.get("japanese", "")}</span>
                <span class="crash-vocab-reading">{item.get("reading", "")}</span>
            </div>
            <div class="crash-vocab-details">
                <div class="crash-vocab-meaning">{item.get("meaning", "")}</div>
                {note_html}
            </div>
            {chinese_html}
        </div>
        ''')

    desc_html = f'<p>{step.description}</p>' if step.description else ''

    return f'''
    <div class="crash-step crash-vocab-intro">
        <div class="crash-step-title">{step.title}</div>
        {desc_html}
        <div class="crash-vocab-list">
            {"".join(items_html)}
        </div>
    </div>
    '''


def render_practice_step(step: PracticeStep) -> str:
    """Render a practice exercise step."""
    items_html = []
    exercise_type = step.exercise_type or "select"

    for idx, item in enumerate(step.items):
        if isinstance(item, dict):
            # Match type: left-right pairs
            if "left" in item and "right" in item:
                items_html.append(f'''
                <div class="crash-practice-match-item">
                    <span class="crash-match-left">{item["left"]}</span>
                    <span class="crash-match-arrow">→</span>
                    <span class="crash-match-right">{item["right"]}</span>
                </div>
                ''')
            # Select/quiz type: question with options
            elif "question" in item:
                options_html = ""
                if item.get("options"):
                    opts = "".join(
                        f'<div class="crash-quiz-option">{opt}</div>'
                        for opt in item["options"]
                    )
                    options_html = f'<div class="crash-quiz-options">{opts}</div>'

                answer_html = ""
                if item.get("answer"):
                    explanation = item.get("explanation", "")
                    answer_html = f'''
                    <details class="crash-practice-answer">
                        <summary>显示答案</summary>
                        <div class="crash-answer-content">
                            <strong>答案：</strong>{item["answer"]}
                            {f'<br><em>{explanation}</em>' if explanation else ''}
                        </div>
                    </details>
                    '''

                items_html.append(f'''
                <div class="crash-practice-question">
                    <div class="crash-question-text"><strong>{idx + 1}.</strong> {item["question"]}</div>
                    {options_html}
                    {answer_html}
                </div>
                ''')
            # Fill blank type
            elif "sentence" in item:
                blank = item.get("blank", "___")
                items_html.append(f'''
                <div class="crash-practice-fill">
                    <div class="crash-fill-sentence">{item["sentence"]}</div>
                    <details class="crash-practice-answer">
                        <summary>显示答案</summary>
                        <div class="crash-answer-content">{blank}</div>
                    </details>
                </div>
                ''')
            # Text content
            elif "text" in item:
                items_html.append(f'<div class="crash-practice-item">{item["text"]}</div>')
            else:
                # Fallback for other dict formats
                content = "<br>".join(f"<strong>{k}:</strong> {v}" for k, v in item.items() if v)
                items_html.append(f'<div class="crash-practice-item">{content}</div>')
        else:
            # String items
            items_html.append(f'<div class="crash-practice-item">{item}</div>')

    hint_html = f'<div class="crash-practice-hint">💡 {step.hint}</div>' if step.hint else ''
    answer_html = f'<div class="crash-practice-hint">✓ 答案：{step.answer}</div>' if hasattr(step, 'answer') and step.answer else ''

    return f'''
    <div class="crash-step crash-practice">
        <div class="crash-step-title">{step.title}</div>
        <div class="crash-practice-instruction">{step.instruction}</div>
        <div class="crash-practice-items">
            {"".join(items_html)}
        </div>
        {hint_html}
        {answer_html}
    </div>
    '''


def render_example_sentence_step(step: ExampleSentenceStep) -> str:
    """Render an example sentence step with breakdown."""
    title_html = f'<div class="crash-step-title">{step.title}</div>' if step.title else ''

    breakdown_html = ''
    if step.breakdown:
        breakdown_parts = []
        for part in step.breakdown:
            breakdown_parts.append(f'''
            <div class="crash-breakdown-part">
                <div class="crash-breakdown-word">{part.get("part", "")}</div>
                <div class="crash-breakdown-role">{part.get("role", "")}</div>
                <div class="crash-breakdown-meaning">{part.get("meaning", "")}</div>
            </div>
            ''')
        breakdown_html = f'''
            <div class="crash-sentence-breakdown">
                <div class="crash-breakdown-title">句子分析</div>
                <div class="crash-breakdown-parts">
                    {"".join(breakdown_parts)}
                </div>
            </div>
        '''

    reading_html = f'<div class="crash-sentence-reading">{step.reading}</div>' if step.reading else ''
    grammar_html = f'<div class="crash-grammar-note">{step.grammar_note}</div>' if step.grammar_note else ''

    return f'''
    <div class="crash-step crash-example-sentence">
        {title_html}
        <div class="crash-sentence-box">
            <div class="crash-sentence-main">{step.sentence}</div>
            {reading_html}
            <div class="crash-sentence-translation">{step.translation}</div>
            {breakdown_html}
            {grammar_html}
        </div>
    </div>
    '''


def render_mini_poem_step(step: MiniPoemStep) -> str:
    """Render a mini poem step."""
    title_html = f'<div class="crash-step-title">{step.title}</div>' if step.title else '<div class="crash-step-title">诗歌赏析</div>'

    # Handle alternative field names
    poem_text = step.poem_text or step.text or step.content or ""
    reading = step.reading or ""
    translation = step.translation or ""
    analysis = step.analysis or step.question or ""

    vocab_html = ''
    if step.vocabulary:
        vocab_items = []
        for v in step.vocabulary:
            vocab_items.append(f'<span><strong>{v.get("word", "")}</strong>（{v.get("reading", "")}）: {v.get("meaning", "")}</span>')
        vocab_html = f'<div style="margin-top: 1em; font-size: 0.9em; color: var(--sumi-light);">词汇：{" | ".join(vocab_items)}</div>'

    reading_html = f'<div class="crash-poem-reading">{reading}</div>' if reading else ''
    translation_html = f'<div class="crash-poem-translation">{translation}</div>' if translation else ''
    # Convert markdown and preserve line breaks in analysis
    analysis_converted = markdown_to_html(analysis) if analysis else ''
    analysis_html = f'<div class="crash-poem-analysis">{analysis_converted}</div>' if analysis else ''

    return f'''
    <div class="crash-step">
        {title_html}
        <div class="crash-poem-box">
            <div class="crash-poem-text">{poem_text}</div>
            {reading_html}
            {translation_html}
            {analysis_html}
            {vocab_html}
        </div>
    </div>
    '''


def render_summary_step(step: SummaryStep) -> str:
    """Render a summary step."""
    points_html = ''.join(f'<li>{point}</li>' for point in step.key_points)
    next_html = f'<div class="crash-summary-next">📍 下一课预告：{step.preview_next}</div>' if step.preview_next else ''

    return f'''
    <div class="crash-step crash-summary">
        <div class="crash-step-title">{step.title}</div>
        <ul class="crash-summary-points">
            {points_html}
        </ul>
        {next_html}
    </div>
    '''


def render_tip_step(step: TipStep) -> str:
    """Render a tip/note step."""
    icons = {
        "learning": "📚",
        "cultural": "🎌",
        "memory": "💡",
        "comparison": "🔄",
    }
    icon = icons.get(step.tip_type, "💡")
    title_html = f'<strong>{step.title}</strong><br>' if step.title else ''

    return f'''
    <div class="crash-step crash-tip {step.tip_type}">
        <span class="crash-tip-icon">{icon}</span>
        {title_html}
        {step.content}
    </div>
    '''


def render_crash_course_step(step: dict) -> str:
    """Render a single crash course step based on its type."""
    step_type = step.get("type", "text")

    if step_type == "text":
        return render_text_step(TextStep(**step))
    elif step_type == "kana_chart":
        return render_kana_chart_step(KanaChartStep(**step))
    elif step_type == "vocab_intro":
        return render_vocab_intro_step(VocabIntroStep(**step))
    elif step_type == "practice":
        return render_practice_step(PracticeStep(**step))
    elif step_type == "example_sentence":
        return render_example_sentence_step(ExampleSentenceStep(**step))
    elif step_type == "mini_poem":
        return render_mini_poem_step(MiniPoemStep(**step))
    elif step_type == "summary":
        return render_summary_step(SummaryStep(**step))
    elif step_type == "tip":
        return render_tip_step(TipStep(**step))
    else:
        # Fallback for unknown types
        return f'<div class="crash-step"><pre>{json.dumps(step, ensure_ascii=False, indent=2)}</pre></div>'


def render_crash_course_lesson_header(lesson: dict, total_lessons: int = 7) -> str:
    """Render the lesson header."""
    objectives_html = ''.join(f'<li>{obj}</li>' for obj in lesson.get("objectives", []))

    return f'''
    <div class="crash-lesson-header">
        <div class="crash-lesson-number">第 {lesson.get("lesson_number", 0)} 课 / 共 {total_lessons} 课</div>
        <h1 class="crash-lesson-title">{lesson.get("title", "")}</h1>
        <div class="crash-lesson-subtitle">{lesson.get("subtitle", "")}</div>
    </div>
    <div class="crash-objectives">
        <div class="crash-objectives-title">🎯 本课目标</div>
        <ul>
            {objectives_html}
        </ul>
    </div>
    '''


def render_crash_course_quiz(quiz: dict, lesson_id: str) -> str:
    """Render the comprehension check quiz (static HTML - interactivity handled by Streamlit)."""
    if not quiz or not quiz.get("questions"):
        return ""

    questions_html = []
    for idx, q in enumerate(quiz.get("questions", [])):
        options_html = ""
        if q.get("options"):
            opts = ''.join(
                f'<div class="crash-quiz-option" data-q="{idx}" data-opt="{i}">{opt}</div>'
                for i, opt in enumerate(q["options"])
            )
            options_html = f'<div class="crash-quiz-options">{opts}</div>'

        questions_html.append(f'''
        <div class="crash-quiz-question" data-correct="{q.get("correct_answer", "")}">
            <div class="crash-quiz-q-text"><strong>问题 {idx + 1}:</strong> {q.get("question", "")}</div>
            {options_html}
        </div>
        ''')

    return f'''
    <div class="crash-quiz">
        <div class="crash-quiz-header">✍️ 理解检测</div>
        {"".join(questions_html)}
    </div>
    '''


def render_crash_course_overview(course: dict, completed_lessons: list[str] = None) -> str:
    """Render the crash course overview page."""
    if completed_lessons is None:
        completed_lessons = []

    lessons_html = []
    for lesson in course.get("lessons", []):
        lesson_id = lesson.get("lesson_id", "")
        is_completed = lesson_id in completed_lessons
        is_locked = False  # Could implement sequential unlocking

        status_class = "completed" if is_completed else ("locked" if is_locked else "")
        number_class = "completed" if is_completed else ""
        icon = "✓" if is_completed else lesson.get("lesson_number", 0)

        lessons_html.append(f'''
        <div class="crash-lesson-card {status_class}" data-lesson="{lesson_id}">
            <div class="crash-card-number {number_class}">{icon}</div>
            <div class="crash-card-info">
                <h4>{lesson.get("title", "")}</h4>
                <p>{lesson.get("subtitle", "")}</p>
            </div>
            <div class="crash-card-duration">~{lesson.get("estimated_minutes", 15)} 分钟</div>
        </div>
        ''')

    total_minutes = sum(l.get("estimated_minutes", 15) for l in course.get("lessons", []))
    completed_count = len(completed_lessons)
    total_count = len(course.get("lessons", []))

    return f'''
    <div class="crash-overview">
        <div class="crash-overview-header">
            <h1 class="crash-overview-title">{course.get("title", "日语速成课程")}</h1>
            <div class="crash-overview-subtitle">{course.get("subtitle", "")}</div>
        </div>
        <div class="crash-overview-description">
            {course.get("description", "")}
        </div>
        <div style="text-align: center; margin: 1.5em 0; color: var(--sumi-light);">
            📚 {total_count} 课 · ⏱️ 约 {total_minutes} 分钟 · ✓ 已完成 {completed_count}/{total_count}
        </div>
        <div class="crash-lesson-cards">
            {"".join(lessons_html)}
        </div>
    </div>
    '''


def load_crash_course(path: Path = None) -> Optional[dict]:
    """Load crash course data from JSON file."""
    if path is None:
        path = Path("data/crash_course.json")

    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)
