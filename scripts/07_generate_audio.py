#!/usr/bin/env python3
"""
07_generate_audio.py - Generate TTS audio for poems using Google Cloud TTS.

Generates MP3 audio files for poem pronunciation using the reading_hiragana
field from the database. Audio files are stored in data/audio/ and can be
served by the Streamlit runtime.

Usage:
  python scripts/07_generate_audio.py
  python scripts/07_generate_audio.py --limit 10         # Generate for first 10 poems
  python scripts/07_generate_audio.py --voice ja-JP-Neural2-C  # Use male voice
  python scripts/07_generate_audio.py --skip-existing    # Skip existing audio files

Prerequisites:
  Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file path
  OR run `gcloud auth application-default login`
"""

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Google Cloud TTS
# -----------------------------------------------------------------------------

def get_tts_client():
    """Get Google Cloud TTS client."""
    try:
        from google.cloud import texttospeech
        return texttospeech.TextToSpeechClient()
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
        logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set or you're authenticated via gcloud")
        raise


def add_poetic_ssml(text: str) -> str:
    """
    Convert hiragana into Waka-structured SSML for poetic recitation.

    Strategy:
    1. If text already has space-delimited 5 parts, use them (most reliable)
    2. Otherwise, attempt character-based slicing for standard lengths
    3. Fallback to simple kami/shimo split

    Uses punctuation to guide prosody:
    - Full-width space (　) for subtle breath pauses within kami/shimo
    - Comma (、) before main break to maintain continuation intonation
    - Period (。) at end for natural closure
    """
    import re

    # Normalize: replace newlines with spaces, normalize space types
    normalized = text.replace('\n', ' ').replace('　', ' ').strip()

    # Split on whitespace to check for pre-segmented text
    raw_parts = [p for p in re.split(r'\s+', normalized) if p]

    # If we have exactly 5 parts, use them directly (most reliable)
    if len(raw_parts) == 5:
        parts = raw_parts
    # If we have 3 parts, might be kami(5-7-5) / middle / shimo format - try to use
    elif len(raw_parts) == 3:
        # Treat as: part1 = first ku, part2 = middle section, part3 = last section
        # Just use simple kami/shimo split
        kami = raw_parts[0] + '　' + raw_parts[1]
        shimo = raw_parts[2]
        return f"<speak>{kami}、<break time=\"600ms\"/>{shimo}。</speak>"
    else:
        # No useful segmentation - clean and try character-based slicing
        clean_text = ''.join(raw_parts)  # Remove all spaces
        length = len(clean_text)

        # Standard 31-mora waka
        if length == 31:
            parts = [
                clean_text[0:5],    # ku 1: 5 morae
                clean_text[5:12],   # ku 2: 7 morae
                clean_text[12:17],  # ku 3: 5 morae
                clean_text[17:24],  # ku 4: 7 morae
                clean_text[24:31],  # ku 5: 7 morae
            ]
        # Fallback for any other length: simple kami/shimo split at ~60% mark
        else:
            # For waka, kami is 5+7+5=17, shimo is 7+7=14, so split around 55%
            split_point = int(length * 0.55)
            kami = clean_text[:split_point]
            shimo = clean_text[split_point:]
            return f"<speak>{kami}、<break time=\"600ms\"/>{shimo}。</speak>"

    # Build structured SSML from 5 parts
    # Kami-no-ku (upper verse): ku1-ku2-ku3, connected by full-width spaces
    kami = f"{parts[0]}　{parts[1]}　{parts[2]}"

    # Shimo-no-ku (lower verse): ku4-ku5, connected by full-width space
    shimo = f"{parts[3]}　{parts[4]}"

    # Comma before break maintains "continuation" intonation (not finality)
    ssml = (
        f"<speak>"
        f"{kami}、"
        f'<break time="600ms"/>'  # Kugire: the major turning pause
        f"{shimo}。"
        f"</speak>"
    )
    return ssml


def synthesize_speech(
    client,
    text: str,
    output_path: Path,
    voice_name: str = "ja-JP-Neural2-B",
    speaking_rate: float = 0.85,
    pitch: float = -2.0,
) -> bool:
    """
    Synthesize speech using Google Cloud TTS with poetic SSML.

    Args:
        client: Google Cloud TTS client
        text: Text to synthesize (should be hiragana for best results)
        output_path: Path to save the MP3 file
        voice_name: Google Cloud TTS voice name
        speaking_rate: Speaking rate (0.85 = slightly slower for learning)
        pitch: Pitch shift in semitones (-2.0 = lower, more solemn)

    Returns:
        True if successful, False otherwise
    """
    from google.cloud import texttospeech

    try:
        # Build the synthesis input with waka-structured SSML
        ssml = add_poetic_ssml(text)
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name=voice_name,
        )

        # Audio config with lower pitch for gravitas
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=1.0,  # Slight boost to compensate for lower pitch
        )

        # Perform the TTS request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        # Write the audio content to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        return True

    except Exception as e:
        logger.error(f"TTS synthesis failed for {output_path.name}: {e}")
        return False


def list_available_voices(client):
    """List available Japanese voices."""
    from google.cloud import texttospeech

    response = client.list_voices(language_code="ja-JP")

    logger.info("Available Japanese voices:")
    for voice in response.voices:
        ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        logger.info(f"  {voice.name} ({ssml_gender})")


# -----------------------------------------------------------------------------
# Database Operations
# -----------------------------------------------------------------------------

def get_poems_for_audio(db_path: Path, limit: int | None = None) -> list[dict]:
    """Get poems that need audio generation."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT id, text, reading_hiragana
        FROM poems
        WHERE reading_hiragana IS NOT NULL AND reading_hiragana != ''
        ORDER BY difficulty_score ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    poems = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return poems


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio for poems using Google Cloud TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Authentication:
  Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file path
  OR run: gcloud auth application-default login

Voice options (Neural2 recommended for quality):
  ja-JP-Neural2-B  - Female, natural (default)
  ja-JP-Neural2-C  - Male, natural
  ja-JP-Neural2-D  - Female, warm

Example:
  python scripts/07_generate_audio.py --voice ja-JP-Neural2-C --limit 10
        """
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "classroom.db",
        help="Path to classroom.db"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "audio",
        help="Output directory for audio files"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="ja-JP-Neural2-B",
        help="Google Cloud TTS voice name (default: ja-JP-Neural2-B)"
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=0.85,
        help="Speaking rate, 0.5-2.0 (default: 0.85 for learners)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of poems to process (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip poems that already have audio files"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Japanese voices and exit"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )

    args = parser.parse_args()

    # Initialize TTS client
    logger.info("Initializing Google Cloud TTS client...")
    try:
        client = get_tts_client()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("")
        logger.error("Please set up authentication:")
        logger.error("  1. Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file")
        logger.error("  2. OR run: gcloud auth application-default login")
        sys.exit(1)

    # List voices and exit if requested
    if args.list_voices:
        list_available_voices(client)
        return

    # Check database exists
    if not args.database.exists():
        logger.error(f"Database not found: {args.database}")
        logger.error("Run 05_compile_classroom.py first to create the database")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load poems
    logger.info(f"Loading poems from {args.database}...")
    poems = get_poems_for_audio(args.database, args.limit)
    logger.info(f"Found {len(poems)} poems with reading data")

    if not poems:
        logger.warning("No poems found with reading_hiragana data")
        return

    # Load reading fixes for poems with kanji-contaminated readings
    reading_fixes = {}
    fixes_path = PROJECT_ROOT / "data" / "reading_fixes.json"
    if fixes_path.exists():
        with open(fixes_path, 'r', encoding='utf-8') as f:
            reading_fixes = json.load(f)
        logger.info(f"Loaded {len(reading_fixes)} reading fixes")

    # Generate audio with random voice selection (Neural2 + Wavenet, same price tier)
    voices = [
        "ja-JP-Neural2-B", "ja-JP-Neural2-C", "ja-JP-Neural2-D",
        "ja-JP-Wavenet-A", "ja-JP-Wavenet-B", "ja-JP-Wavenet-C", "ja-JP-Wavenet-D",
    ]
    logger.info(f"Generating audio with random voices: {voices}")
    logger.info(f"Speaking rate: {args.speaking_rate}")
    logger.info(f"Output directory: {args.output}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, poem in enumerate(poems, 1):
        poem_id = poem["id"]

        # Use clean reading if available, otherwise use original
        if poem_id in reading_fixes:
            reading = reading_fixes[poem_id]["clean"]
        else:
            reading = poem["reading_hiragana"]

        # Sanitize poem_id for filename
        safe_id = poem_id.replace("/", "_").replace("\\", "_")
        audio_path = args.output / f"{safe_id}.mp3"

        if args.skip_existing and audio_path.exists():
            skip_count += 1
            continue

        voice = random.choice(voices)
        logger.info(f"[{i}/{len(poems)}] Generating audio for {poem_id} ({voice})...")

        success = synthesize_speech(
            client=client,
            text=reading.strip(),
            output_path=audio_path,
            voice_name=voice,
            speaking_rate=args.speaking_rate,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting
        if args.delay > 0:
            time.sleep(args.delay)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("AUDIO GENERATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total poems: {len(poems)}")
    logger.info(f"Generated: {success_count}")
    logger.info(f"Skipped: {skip_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Output directory: {args.output}")

    # Estimate storage
    if success_count > 0:
        audio_files = list(args.output.glob("*.mp3"))
        total_size = sum(f.stat().st_size for f in audio_files)
        avg_size = total_size / len(audio_files) if audio_files else 0
        logger.info(f"Total audio size: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"Average file size: {avg_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
