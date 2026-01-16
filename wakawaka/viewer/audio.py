"""
Audio viewer - Functions for displaying audio playback controls.

Provides utilities for rendering audio players for poems and vocabulary.
"""

from pathlib import Path
from typing import Optional
import json


def get_audio_path(poem_id: str, data_dir: Path) -> Optional[Path]:
    """
    Get the audio file path for a poem.

    Args:
        poem_id: Poem identifier
        data_dir: Base data directory (containing audio/ subdirectory)

    Returns:
        Path to audio file if it exists, None otherwise
    """
    # Sanitize poem_id for filename
    safe_id = poem_id.replace("/", "_").replace("\\", "_")
    audio_path = data_dir / "audio" / f"{safe_id}.mp3"

    if audio_path.exists():
        return audio_path
    return None


def get_poem_ids_from_lesson(lesson_content) -> list[str]:
    """
    Extract poem IDs from a lesson's teaching sequence.

    Args:
        lesson_content: LessonContent object

    Returns:
        List of poem IDs used in the lesson
    """
    poem_ids = []
    for step in lesson_content.teaching_sequence:
        if hasattr(step, 'poem_id') and step.poem_id:
            if step.poem_id not in poem_ids:
                poem_ids.append(step.poem_id)
    return poem_ids


def get_audio_for_lesson(lesson_content, data_dir: Path) -> dict[str, Path]:
    """
    Get all available audio files for poems in a lesson.

    Args:
        lesson_content: LessonContent object
        data_dir: Base data directory

    Returns:
        Dictionary mapping poem_id to audio file path
    """
    audio_files = {}
    poem_ids = get_poem_ids_from_lesson(lesson_content)

    for poem_id in poem_ids:
        audio_path = get_audio_path(poem_id, data_dir)
        if audio_path:
            audio_files[poem_id] = audio_path

    return audio_files


def has_any_audio(data_dir: Path) -> bool:
    """
    Check if any audio files have been generated.

    Args:
        data_dir: Base data directory

    Returns:
        True if audio directory exists and contains MP3 files
    """
    audio_dir = data_dir / "audio"
    if not audio_dir.exists():
        return False
    return len(list(audio_dir.glob("*.mp3"))) > 0
