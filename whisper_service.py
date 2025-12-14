import os
from typing import Optional

from groq import Groq


def _get_client() -> Groq:
    """Initialize and return a Groq client using env var GROQ_API_KEY if set."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)
    return Groq()


def transcribe_audio_file(file_path: str, language: Optional[str] = "en") -> str:
    """
    Transcribe an audio file using Groq Whisper (whisper-large-v3).

    Args:
        file_path: Path to the audio file.
        language: Optional ISO-639-1 language code (e.g., 'en'). Defaults to 'en'.

    Returns:
        Transcribed text string (empty string if none).
    """
    client = _get_client()

    # Read file bytes and send as multipart via SDK
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    transcription = client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model="whisper-large-v3",
        response_format="json",
        language=language or "en",
        temperature=0.0,
    )

    # SDK returns an object with .text; be defensive in case of dict-like response
    text = getattr(transcription, "text", None)
    if not text and isinstance(transcription, dict):
        text = transcription.get("text")

    return (text or "").strip()