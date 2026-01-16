"""
Literary Analysis Viewer - Rendering components for poem literary analysis.

Displays literary analysis in a collapsible panel styled to match
the Ink & Paper theme.
"""

import html
from typing import Optional


def _is_nonempty_list(val) -> bool:
    """Check if value is a non-empty list-like object (handles numpy arrays)."""
    if val is None:
        return False
    try:
        return len(val) > 0
    except (TypeError, ValueError):
        return False


def _to_list(val) -> list:
    """Convert value to a regular Python list (handles numpy arrays)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        return list(val)
    except (TypeError, ValueError):
        return []


def get_literary_css() -> str:
    """Get CSS styles for literary analysis display."""
    return """
    <style>
    /* ============================================
       LITERARY ANALYSIS PANEL - Ink & Paper Theme
       ============================================ */

    .literary-panel {
        background: linear-gradient(145deg, #FDFCFA 0%, #F8F6F3 100%);
        border: 1px solid #E8E4DE;
        border-radius: 8px;
        margin: 1em 0;
        overflow: hidden;
    }

    .literary-header {
        background: linear-gradient(90deg, #4A5568 0%, #5A6578 100%);
        color: #FAF8F5;
        padding: 0.7em 1.2em;
        font-family: 'Noto Serif JP', serif;
        font-size: 0.95em;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        user-select: none;
    }

    .literary-header:hover {
        background: linear-gradient(90deg, #5A6578 0%, #6A7588 100%);
    }

    .literary-header-icon {
        font-size: 0.9em;
        transition: transform 0.2s;
    }

    .literary-content {
        padding: 1.2em 1.5em;
        font-size: 0.95em;
        line-height: 1.8;
        color: #2D2D2D;
    }

    .literary-section {
        margin-bottom: 1.2em;
        padding-bottom: 1em;
        border-bottom: 1px dashed #E8E4DE;
    }

    .literary-section:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }

    .literary-section-title {
        font-family: 'Noto Serif JP', serif;
        font-weight: 600;
        color: #4A5568;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5em;
        display: flex;
        align-items: center;
        gap: 0.4em;
    }

    .literary-text {
        color: #4A4A4A;
    }

    /* Interpretation - primary section */
    .literary-interpretation {
        font-size: 1.02em;
        color: #2D2D2D;
        padding: 0.8em 1em;
        background: rgba(197, 61, 67, 0.05);
        border-left: 3px solid #C53D43;
        border-radius: 0 4px 4px 0;
    }

    /* Emotional tone badge */
    .literary-tone {
        display: inline-block;
        background: linear-gradient(135deg, #8B687F 0%, #9B7890 100%);
        color: white;
        padding: 0.3em 0.8em;
        border-radius: 4px;
        font-size: 0.9em;
        font-weight: 500;
    }

    /* Poetic devices */
    .literary-device {
        background: #FDFCFA;
        border: 1px solid #E8E4DE;
        border-radius: 6px;
        padding: 0.8em 1em;
        margin: 0.5em 0;
    }

    .literary-device-name {
        font-weight: 600;
        color: #5B8A72;
        font-family: 'Noto Serif JP', serif;
    }

    .literary-device-location {
        color: #6B8CAE;
        font-style: italic;
        font-size: 0.95em;
    }

    .literary-device-explanation {
        margin-top: 0.4em;
        color: #4A4A4A;
    }

    .literary-device-effect {
        margin-top: 0.3em;
        color: #8B687F;
        font-size: 0.92em;
    }

    /* Imagery analysis */
    .literary-imagery {
        display: flex;
        gap: 0.5em;
        flex-wrap: wrap;
        margin-top: 0.3em;
    }

    .literary-image-item {
        background: rgba(107, 140, 174, 0.1);
        border: 1px solid rgba(107, 140, 174, 0.3);
        border-radius: 4px;
        padding: 0.4em 0.7em;
        font-size: 0.9em;
    }

    .literary-image-term {
        font-weight: 600;
        color: #4A5568;
    }

    .literary-image-meaning {
        color: #6B8CAE;
        margin-left: 0.3em;
    }

    /* Chinese poetry parallel */
    .literary-chinese {
        background: linear-gradient(135deg, #FFF8E8 0%, #FFF4DB 100%);
        border: 1px solid #F0E6D0;
        border-radius: 6px;
        padding: 0.8em 1em;
        font-family: 'Noto Serif JP', serif;
        color: #8B6914;
    }

    /* Cultural notes */
    .literary-cultural {
        background: rgba(91, 138, 114, 0.08);
        border-radius: 4px;
        padding: 0.6em 1em;
        color: #4A4A4A;
    }

    /* Critical notes */
    .literary-critical {
        font-style: italic;
        color: #666;
        padding-left: 1em;
        border-left: 2px solid #C17F59;
    }

    </style>
    """


def render_literary_analysis(
    analysis: dict,
    collapsed: bool = True,
) -> str:
    """
    Render literary analysis as an expandable panel.

    Args:
        analysis: Dictionary with literary analysis fields
        collapsed: Whether to start collapsed (default True)

    Returns:
        HTML string for the literary analysis panel
    """
    if not analysis:
        return ""

    parts = ['<div class="literary-panel">']

    # Header (clickable to expand)
    details_attr = "" if collapsed else "open"
    parts.append(f'<details {details_attr}>')
    parts.append('<summary class="literary-header">')
    parts.append('<span>文学赏析 Literary Analysis</span>')
    parts.append('<span class="literary-header-icon">▼</span>')
    parts.append('</summary>')

    parts.append('<div class="literary-content">')

    # === Interpretation (always show first) ===
    if analysis.get("interpretation"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">释义 Interpretation</div>')
        parts.append(f'<div class="literary-interpretation">{html.escape(analysis["interpretation"])}</div>')
        parts.append('</div>')

    # === Emotional Tone ===
    if analysis.get("emotional_tone"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">情感基调 Tone</div>')
        parts.append(f'<span class="literary-tone">{html.escape(analysis["emotional_tone"])}</span>')
        parts.append('</div>')

    # === Literary Techniques ===
    if analysis.get("literary_techniques"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">技法 Techniques</div>')
        parts.append(f'<div class="literary-text">{html.escape(analysis["literary_techniques"])}</div>')
        parts.append('</div>')

    # === Poetic Devices ===
    devices = _to_list(analysis.get("poetic_devices"))
    if _is_nonempty_list(devices):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">修辞手法 Poetic Devices</div>')
        for device in devices:
            if isinstance(device, dict):
                parts.append('<div class="literary-device">')
                name = device.get("name", "")
                location = device.get("location", "")
                explanation = device.get("explanation", "")
                effect = device.get("effect", "")

                parts.append(f'<div class="literary-device-name">{html.escape(str(name))}</div>')
                if location:
                    parts.append(f'<div class="literary-device-location">「{html.escape(str(location))}」</div>')
                if explanation:
                    parts.append(f'<div class="literary-device-explanation">{html.escape(str(explanation))}</div>')
                if effect:
                    parts.append(f'<div class="literary-device-effect">效果: {html.escape(str(effect))}</div>')
                parts.append('</div>')
        parts.append('</div>')

    # === Imagery Analysis ===
    imagery = _to_list(analysis.get("imagery_analysis"))
    if _is_nonempty_list(imagery):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">意象 Imagery</div>')
        parts.append('<div class="literary-imagery">')
        for img in imagery:
            if isinstance(img, dict):
                image = img.get("image", "")
                significance = img.get("significance", "")
                parts.append('<div class="literary-image-item">')
                parts.append(f'<span class="literary-image-term">{html.escape(str(image))}</span>')
                if significance:
                    parts.append(f'<span class="literary-image-meaning">— {html.escape(str(significance))}</span>')
                parts.append('</div>')
        parts.append('</div>')
        parts.append('</div>')

    # === Seasonal Context ===
    if analysis.get("seasonal_context"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">季节 Season</div>')
        parts.append(f'<div class="literary-text">{html.escape(analysis["seasonal_context"])}</div>')
        parts.append('</div>')

    # === Cultural Notes ===
    if analysis.get("cultural_notes"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">文化背景 Cultural Context</div>')
        parts.append(f'<div class="literary-cultural">{html.escape(analysis["cultural_notes"])}</div>')
        parts.append('</div>')

    # === Chinese Poetry Parallel ===
    if analysis.get("chinese_poetry_parallel"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">中国诗歌对照 Chinese Parallel</div>')
        parts.append(f'<div class="literary-chinese">{html.escape(analysis["chinese_poetry_parallel"])}</div>')
        parts.append('</div>')

    # === Critical Notes ===
    if analysis.get("critical_notes"):
        parts.append('<div class="literary-section">')
        parts.append('<div class="literary-section-title">评注 Critical Notes</div>')
        parts.append(f'<div class="literary-critical">{html.escape(analysis["critical_notes"])}</div>')
        parts.append('</div>')

    parts.append('</div>')  # literary-content
    parts.append('</details>')
    parts.append('</div>')  # literary-panel

    return ''.join(parts)
