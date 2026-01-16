"""
Shared utilities for analysis scripts.
"""

import matplotlib.pyplot as plt
import matplotlib

# Global variable to store the Japanese font name
JAPANESE_FONT = None


def setup_japanese_fonts():
    """Configure matplotlib to display Japanese characters correctly."""
    global JAPANESE_FONT

    # Try Japanese fonts in order of preference
    japanese_fonts = [
        'IPAexGothic',
        'IPAGothic',
        'Noto Sans CJK JP',
        'Hiragino Sans',
        'Yu Gothic',
        'Meiryo',
    ]

    # Get available fonts
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}

    # Find first available Japanese font
    for font in japanese_fonts:
        if font in available:
            plt.rcParams['font.family'] = font
            JAPANESE_FONT = font
            print(f"  Using font: {font}")
            return font

    # Fallback: try to add sans-serif with Japanese support
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = japanese_fonts + ['DejaVu Sans']
    JAPANESE_FONT = 'sans-serif'
    print("  Warning: No dedicated Japanese font found, using fallback")
    return None


def get_font():
    """Get the configured Japanese font name for use in networkx etc."""
    return JAPANESE_FONT or 'IPAexGothic'


def setup_plotting():
    """Standard plotting setup for analysis scripts."""
    setup_japanese_fonts()

    # Other matplotlib settings
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
