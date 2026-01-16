"""
Reference card renderer - Grammar reference cards and index display.

Provides:
- Reference card rendering for quick review
- Grammar index display
- Search/filter support for reference mode
"""

import html
from typing import Optional

from wakawaka.schemas import ReferenceCard, LessonContent
from wakawaka.classroom import GrammarPointData


def get_reference_css() -> str:
    """Get CSS styles for reference display."""
    return """
    <style>
    .reference-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2em;
        margin: 1em 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }
    .reference-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .reference-point {
        font-size: 1.2em;
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 0.5em;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }
    .reference-point-icon {
        background: #e3f2fd;
        color: #1976D2;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8em;
    }
    .reference-oneliner {
        color: #666;
        font-size: 1em;
        margin-bottom: 0.8em;
        line-height: 1.5;
    }
    .reference-example {
        background: #f5f5f5;
        padding: 0.8em 1em;
        border-radius: 8px;
        font-family: "Noto Serif JP", serif;
        margin-bottom: 0.8em;
    }
    .reference-example-label {
        font-size: 0.8em;
        color: #888;
        margin-bottom: 0.3em;
    }
    .reference-see-also {
        font-size: 0.9em;
        color: #888;
    }
    .reference-see-also a {
        color: #1976D2;
        text-decoration: none;
    }
    .reference-see-also a:hover {
        text-decoration: underline;
    }
    .grammar-index-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1em;
        padding: 1em 0;
    }
    .grammar-index-item {
        background: #fafafa;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 1em;
        cursor: pointer;
        transition: all 0.2s;
    }
    .grammar-index-item:hover {
        background: #e3f2fd;
        border-color: #1976D2;
    }
    .grammar-index-id {
        font-weight: 600;
        color: #333;
    }
    .grammar-index-category {
        font-size: 0.85em;
        color: #888;
        margin-top: 0.3em;
    }
    .grammar-index-stats {
        font-size: 0.8em;
        color: #666;
        margin-top: 0.5em;
    }
    .grammar-search-box {
        width: 100%;
        padding: 0.8em 1em;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        font-size: 1em;
        margin-bottom: 1em;
        transition: border-color 0.2s;
    }
    .grammar-search-box:focus {
        outline: none;
        border-color: #1976D2;
    }
    .category-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #1565C0;
        margin: 1.5em 0 0.5em 0;
        padding-bottom: 0.3em;
        border-bottom: 2px solid #e3f2fd;
    }
    .reference-mini-card {
        display: inline-block;
        background: #e3f2fd;
        padding: 0.3em 0.6em;
        border-radius: 4px;
        font-size: 0.9em;
        color: #1565C0;
        margin: 0.2em;
        cursor: pointer;
    }
    .reference-mini-card:hover {
        background: #bbdefb;
    }
    </style>
    """


def render_reference_card(card: ReferenceCard, lesson_id: Optional[str] = None) -> str:
    """
    Render a grammar reference card.

    Args:
        card: ReferenceCard object
        lesson_id: Optional lesson ID for linking

    Returns:
        HTML string for the card
    """
    parts = ['<div class="reference-card">']

    # Point title with icon
    parts.append('<div class="reference-point">')
    parts.append('<span class="reference-point-icon">G</span>')
    parts.append(f'<span>{html.escape(card.point)}</span>')
    parts.append('</div>')

    # One-liner description
    parts.append(f'<div class="reference-oneliner">{html.escape(card.one_liner)}</div>')

    # Example
    parts.append('<div class="reference-example">')
    parts.append('<div class="reference-example-label">Example:</div>')
    parts.append(f'{html.escape(card.example)}')
    parts.append('</div>')

    # See also
    if card.see_also:
        links = [f'<a href="#ref-{html.escape(s)}">{html.escape(s)}</a>' for s in card.see_also]
        parts.append(f'<div class="reference-see-also">See also: {", ".join(links)}</div>')

    parts.append('</div>')
    return ''.join(parts)


def render_reference_card_from_lesson(lesson: LessonContent) -> str:
    """Render reference card from a lesson."""
    return render_reference_card(lesson.reference_card, lesson.lesson_id)


def render_grammar_index_item(grammar: GrammarPointData) -> str:
    """
    Render a single grammar index item.

    Args:
        grammar: GrammarPointData object

    Returns:
        HTML string for the item
    """
    parts = ['<div class="grammar-index-item">']
    parts.append(f'<div class="grammar-index-id">{html.escape(grammar.id)}</div>')
    parts.append(f'<div class="grammar-index-category">{html.escape(grammar.category)}</div>')
    parts.append(f'<div class="grammar-index-stats">Frequency: {grammar.frequency} | Difficulty: {grammar.avg_difficulty:.2f}</div>')
    parts.append('</div>')
    return ''.join(parts)


def render_grammar_index(
    grammar_points: list[GrammarPointData],
    group_by_category: bool = True,
) -> str:
    """
    Render the full grammar index.

    Args:
        grammar_points: List of grammar point data
        group_by_category: Whether to group by category

    Returns:
        HTML string for the index
    """
    parts = [get_reference_css()]

    if group_by_category:
        # Group by category
        categories: dict[str, list[GrammarPointData]] = {}
        for gp in grammar_points:
            cat = gp.category or "Other"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(gp)

        # Sort categories
        for cat in sorted(categories.keys()):
            parts.append(f'<div class="category-header">{html.escape(cat)}</div>')
            parts.append('<div class="grammar-index-container">')
            for gp in sorted(categories[cat], key=lambda x: x.id):
                parts.append(render_grammar_index_item(gp))
            parts.append('</div>')
    else:
        parts.append('<div class="grammar-index-container">')
        for gp in sorted(grammar_points, key=lambda x: x.id):
            parts.append(render_grammar_index_item(gp))
        parts.append('</div>')

    return ''.join(parts)


def render_learned_reference_cards(lessons: list[LessonContent]) -> str:
    """
    Render all reference cards from completed lessons.

    Args:
        lessons: List of completed lesson content

    Returns:
        HTML string for all reference cards
    """
    if not lessons:
        return '<p style="color:#666;">Complete lessons to build your reference card collection.</p>'

    parts = [get_reference_css()]
    for lesson in lessons:
        parts.append(f'<div id="ref-{html.escape(lesson.grammar_point)}">')
        parts.append(render_reference_card(lesson.reference_card, lesson.lesson_id))
        parts.append('</div>')

    return ''.join(parts)


def render_mini_reference_cards(lessons: list[LessonContent]) -> str:
    """Render compact reference card tags for quick review."""
    if not lessons:
        return ""

    parts = ['<div style="margin: 1em 0;">']
    for lesson in lessons:
        parts.append(
            f'<span class="reference-mini-card" '
            f'data-lesson-id="{html.escape(lesson.lesson_id)}">'
            f'{html.escape(lesson.reference_card.point)}'
            f'</span>'
        )
    parts.append('</div>')
    return ''.join(parts)


def filter_grammar_points(
    grammar_points: list[GrammarPointData],
    search_query: str,
) -> list[GrammarPointData]:
    """
    Filter grammar points by search query.

    Searches in: id, category
    """
    if not search_query:
        return grammar_points

    query = search_query.lower()
    return [
        gp for gp in grammar_points
        if query in gp.id.lower() or query in (gp.category or "").lower()
    ]
