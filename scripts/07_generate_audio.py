#!/usr/bin/env python3
"""
07_generate_audio.py - Generate TTS audio for poems.

Generates MP3 audio files for poem pronunciation using the reading_hiragana
field from the database. Audio files are stored in data/audio/ and can be
served by the Streamlit runtime.

Supports two TTS backends:
  1. Edge-TTS (default) - Free, no auth needed, good quality
  2. Google Cloud TTS - Requires service account auth

Usage:
  python scripts/07_generate_audio.py                    # Use Edge-TTS (default)
  python scripts/07_generate_audio.py --limit 10         # Generate for first 10 poems
  python scripts/07_generate_audio.py --voice ja-JP-NanamiNeural  # Specific Edge voice
  python scripts/07_generate_audio.py --backend google   # Use Google Cloud TTS
  python scripts/07_generate_audio.py --skip-existing    # Skip existing audio files
"""

import argparse
import asyncio
import base64
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests

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
# Edge-TTS (Free, no auth required)
# -----------------------------------------------------------------------------

async def synthesize_speech_edge_async(
    text: str,
    output_path: Path,
    voice_name: str = "ja-JP-NanamiNeural",
    speaking_rate: float = 0.85,
) -> bool:
    """
    Synthesize speech using Edge-TTS (Microsoft Edge's online TTS).

    Args:
        text: Text to synthesize
        output_path: Path to save the MP3 file
        voice_name: Edge-TTS voice name (ja-JP-NanamiNeural or ja-JP-KeitaNeural)
        speaking_rate: Speaking rate (0.85 = slightly slower)

    Returns:
        True if successful, False otherwise
    """
    try:
        import edge_tts

        # Convert rate to percentage format: 0.85 -> "-15%", 1.0 -> "+0%"
        rate_pct = int((speaking_rate - 1.0) * 100)
        rate_str = f"{rate_pct:+d}%"

        communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        await communicate.save(str(output_path))

        return True

    except Exception as e:
        logger.error(f"Edge-TTS synthesis failed for {output_path.name}: {e}")
        return False


def synthesize_speech_edge(
    text: str,
    output_path: Path,
    voice_name: str = "ja-JP-NanamiNeural",
    speaking_rate: float = 0.85,
) -> bool:
    """Synchronous wrapper for Edge-TTS."""
    return asyncio.run(synthesize_speech_edge_async(
        text, output_path, voice_name, speaking_rate
    ))


# -----------------------------------------------------------------------------
# Google Cloud TTS - REST API (requires OAuth, API keys no longer supported)
# -----------------------------------------------------------------------------

def synthesize_speech_rest(
    text: str,
    output_path: Path,
    api_key: str,
    voice_name: str = "ja-JP-Neural2-B",
    speaking_rate: float = 0.85,
) -> bool:
    """
    Synthesize speech using Google Cloud TTS REST API with API key.

    Args:
        text: Text to synthesize (should be hiragana for best results)
        output_path: Path to save the MP3 file
        api_key: Google Cloud API key
        voice_name: Google Cloud TTS voice name
        speaking_rate: Speaking rate (0.85 = slightly slower for learning)

    Returns:
        True if successful, False otherwise
    """
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": "ja-JP",
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": speaking_rate,
            "pitch": 0.0,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        # Decode base64 audio content
        audio_content = base64.b64decode(response.json()["audioContent"])

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(audio_content)

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"TTS REST API failed for {output_path.name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        return False


# -----------------------------------------------------------------------------
# Google Cloud TTS - Client Library (service account auth)
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


def add_poetic_pauses(text: str) -> str:
    """
    Convert poem text to SSML with poetic pauses.

    Waka follows 5-7-5-7-7 mora pattern. We add pauses:
    - Breath pause (800ms) between phrases within a line
    - Long pause (1.5s) at line breaks for contemplation
    """
    # Replace newlines with long pauses
    ssml_text = text.replace('\n', '<break time="1500ms"/>')

    # Replace spaces (phrase boundaries) with breath pauses
    ssml_text = ssml_text.replace(' ', '<break time="800ms"/>')
    ssml_text = ssml_text.replace('　', '<break time="800ms"/>')  # Full-width space

    # Wrap in SSML speak tags with slower, more contemplative prosody
    ssml = f'<speak><prosody rate="x-slow" pitch="-2st">{ssml_text}</prosody></speak>'

    return ssml


def synthesize_speech_client(
    client,
    text: str,
    output_path: Path,
    voice_name: str = "ja-JP-Neural2-B",
    speaking_rate: float = 0.85,
    poetic_mode: bool = True,
) -> bool:
    """
    Synthesize speech using Google Cloud TTS client library.

    Args:
        client: Google Cloud TTS client
        text: Text to synthesize (should be hiragana for best results)
        output_path: Path to save the MP3 file
        voice_name: Google Cloud TTS voice name
        speaking_rate: Speaking rate (0.85 = slightly slower for learning)
        poetic_mode: Add pauses and prosody for poetry reading

    Returns:
        True if successful, False otherwise
    """
    from google.cloud import texttospeech

    try:
        # Build the synthesis input - use SSML for poetic reading
        if poetic_mode:
            ssml = add_poetic_pauses(text)
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
        else:
            synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name=voice_name,
        )

        # Select the audio encoding
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=0.0,  # Normal pitch
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


def update_audio_metadata(db_path: Path, poem_id: str, audio_path: str):
    """Update poem record with audio path (stored in annotations JSON)."""
    import json

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT annotations FROM poems WHERE id = ?", (poem_id,)
    )
    row = cursor.fetchone()

    annotations = json.loads(row[0]) if row and row[0] else {}
    annotations["audio_path"] = audio_path

    conn.execute(
        "UPDATE poems SET annotations = ? WHERE id = ?",
        (json.dumps(annotations, ensure_ascii=False), poem_id)
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# Audio Generation
# -----------------------------------------------------------------------------

def generate_poem_audio(
    poem: dict,
    audio_dir: Path,
    voice_name: str,
    speaking_rate: float,
    api_key: str = None,
    client=None,
    skip_existing: bool = False,
) -> tuple[bool, str]:
    """
    Generate audio for a single poem.

    Args:
        poem: Poem dict with id and reading_hiragana
        audio_dir: Output directory for audio files
        voice_name: TTS voice name
        speaking_rate: Speech rate
        api_key: Google Cloud API key (for REST API)
        client: Google Cloud TTS client (for client library)
        skip_existing: Skip if audio file exists

    Returns:
        Tuple of (success, relative_audio_path)
    """
    poem_id = poem["id"]
    reading = poem["reading_hiragana"]

    # Sanitize poem_id for filename
    safe_id = poem_id.replace("/", "_").replace("\\", "_")
    audio_filename = f"{safe_id}.mp3"
    audio_path = audio_dir / audio_filename
    relative_path = f"audio/{audio_filename}"

    if skip_existing and audio_path.exists():
        logger.debug(f"Skipping {poem_id} (audio exists)")
        return True, relative_path

    # Clean up the reading text for TTS
    cleaned_reading = reading.strip()

    # Synthesize using API key (REST) or client library
    if api_key:
        success = synthesize_speech_rest(
            text=cleaned_reading,
            output_path=audio_path,
            api_key=api_key,
            voice_name=voice_name,
            speaking_rate=speaking_rate,
        )
    elif client:
        success = synthesize_speech_client(
            client=client,
            text=cleaned_reading,
            output_path=audio_path,
            voice_name=voice_name,
            speaking_rate=speaking_rate,
        )
    else:
        logger.error("No API key or client provided")
        return False, ""

    return success, relative_path if success else ""


def list_available_voices(client):
    """List available Japanese voices."""
    from google.cloud import texttospeech

    response = client.list_voices(language_code="ja-JP")

    logger.info("Available Japanese voices:")
    for voice in response.voices:
        ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        logger.info(f"  {voice.name} ({ssml_gender})")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio for poems using Google Cloud TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Authentication (in priority order):
  1. GOOGLE_TTS_API_KEY env var - Simple API key (recommended)
  2. --api-key argument - Pass API key directly
  3. GOOGLE_APPLICATION_CREDENTIALS - Service account JSON file path
  4. gcloud CLI default credentials

Voice options (Neural2 recommended for quality):
  ja-JP-Neural2-B  - Female, natural (default)
  ja-JP-Neural2-C  - Male, natural
  ja-JP-Neural2-D  - Female, warm
  ja-JP-Wavenet-A  - Female, Wavenet
  ja-JP-Wavenet-B  - Female, Wavenet
  ja-JP-Wavenet-C  - Male, Wavenet
  ja-JP-Wavenet-D  - Male, Wavenet
  ja-JP-Standard-A - Female, Standard (cheaper but lower quality)
  ja-JP-Standard-B - Female, Standard
  ja-JP-Standard-C - Male, Standard
  ja-JP-Standard-D - Male, Standard

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
        "--update-db",
        action="store_true",
        help="Update database with audio paths after generation"
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
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Cloud API key (alternative to env var)"
    )

    args = parser.parse_args()

    # Determine authentication method
    api_key = args.api_key or os.environ.get("GOOGLE_TTS_API_KEY")
    client = None

    if api_key:
        logger.info("Using API key authentication (REST API)")
    else:
        # Try service account / gcloud auth
        logger.info("No API key found, trying service account authentication...")
        try:
            client = get_tts_client()
            logger.info("Using service account authentication (client library)")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.error("")
            logger.error("Please set up authentication using one of these methods:")
            logger.error("  1. Set GOOGLE_TTS_API_KEY in your .env file")
            logger.error("  2. Pass --api-key YOUR_KEY")
            logger.error("  3. Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file")
            logger.error("  4. Run: gcloud auth application-default login")
            sys.exit(1)

    # List voices and exit if requested
    if args.list_voices:
        if client:
            list_available_voices(client)
        else:
            logger.info("Available Japanese Neural2 voices:")
            logger.info("  ja-JP-Neural2-B (Female, natural)")
            logger.info("  ja-JP-Neural2-C (Male, natural)")
            logger.info("  ja-JP-Neural2-D (Female, warm)")
            logger.info("Use --list-voices with service account auth to see all voices.")
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

    # Generate audio
    logger.info(f"Generating audio with voice: {args.voice}")
    logger.info(f"Speaking rate: {args.speaking_rate}")
    logger.info(f"Output directory: {args.output}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, poem in enumerate(poems, 1):
        poem_id = poem["id"]

        # Check if already exists
        safe_id = poem_id.replace("/", "_").replace("\\", "_")
        audio_path = args.output / f"{safe_id}.mp3"

        if args.skip_existing and audio_path.exists():
            skip_count += 1
            continue

        logger.info(f"[{i}/{len(poems)}] Generating audio for {poem_id}...")

        success, relative_path = generate_poem_audio(
            poem=poem,
            audio_dir=args.output,
            voice_name=args.voice,
            speaking_rate=args.speaking_rate,
            api_key=api_key,
            client=client,
            skip_existing=args.skip_existing,
        )

        if success:
            success_count += 1

            if args.update_db:
                update_audio_metadata(args.database, poem_id, relative_path)
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
