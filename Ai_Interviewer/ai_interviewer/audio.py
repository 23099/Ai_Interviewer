"""Voice recording persistence and speech-to-text helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import speech_recognition as sr

from .config import AUDIO_DIR, ensure_project_dirs


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    status: str
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def save_audio_file(uploaded_audio, session_id: str, question_id: str) -> Path | None:
    if uploaded_audio is None:
        return None

    ensure_project_dirs()
    suffix = _suffix_from_mime(getattr(uploaded_audio, "type", "audio/wav"))
    output_path = AUDIO_DIR / f"{session_id}_{question_id}{suffix}"
    output_path.write_bytes(uploaded_audio.getvalue())
    return output_path


def transcribe_audio(path: Path | None) -> TranscriptResult:
    if path is None:
        return TranscriptResult(text="", status="missing", error="No audio recording was submitted")

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(path)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return TranscriptResult(text="", status="unrecognized", error="Speech could not be recognized")
    except sr.RequestError as exc:
        return TranscriptResult(text="", status="service_error", error=f"Speech service error: {exc}")
    except (ValueError, OSError, EOFError) as exc:
        return TranscriptResult(text="", status="invalid_audio", error=f"Audio could not be processed: {exc}")

    return TranscriptResult(text=text, status="ok")


def _suffix_from_mime(mime_type: str) -> str:
    if "mpeg" in mime_type or "mp3" in mime_type:
        return ".mp3"
    if "ogg" in mime_type:
        return ".ogg"
    if "webm" in mime_type:
        return ".webm"
    return ".wav"
