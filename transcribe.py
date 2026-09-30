"""Speech-to-text front end (Whisper) used by the dashboard's audio tab.

``transformers``/``torch`` are imported lazily so importing this module (and the
test suite) never requires the heavy dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from config import get_settings
from utils import is_meaningful_text

__all__ = ["ASR_MODELS", "Transcript", "TranscriptionError", "transcribe"]

logger = logging.getLogger(__name__)

#: Whisper checkpoints exposed in the UI, smallest first.
ASR_MODELS: Mapping[str, str] = {
    "tiny (39M, fastest)": "openai/whisper-tiny",
    "base (74M, balanced)": "openai/whisper-base",
    "small (244M, best accuracy)": "openai/whisper-small",
}

DEFAULT_ASR_KEY = "base (74M, balanced)"

_CACHE: dict[str, Any] = {}


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@dataclass(frozen=True)
class Transcript:
    """Result of a speech-to-text call."""

    text: str
    model: str
    language: str | None = None
    duration_seconds: float | None = None
    skipped: bool = False
    reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return not self.skipped and bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def load_asr(model_id: str) -> Any:
    """Return (and memoise) an ``automatic-speech-recognition`` pipeline."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:  # pragma: no cover - requires transformers
        raise TranscriptionError(
            "transformers is required for audio transcription. "
            "Install it with: pip install -r requirements.txt"
        ) from exc
    logger.info("loading ASR pipeline %s", model_id)
    try:
        asr = hf_pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            cache_dir=str(get_settings().cache_dir),
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise TranscriptionError(f"Failed to load {model_id}: {exc}") from exc
    _CACHE[model_id] = asr
    return asr


def _duration_of(audio: bytes) -> float | None:
    """Best-effort duration in seconds using ``soundfile`` when installed."""
    try:
        import io

        import soundfile  # type: ignore[import-not-found]

        with soundfile.SoundFile(io.BytesIO(audio)) as handle:
            frames = len(handle)
            rate = handle.samplerate or 0
            return round(frames / rate, 2) if rate else None
    except Exception:
        logger.debug("duration unavailable (soundfile missing?)", exc_info=True)
        return None


def transcribe(
    audio: bytes,
    *,
    model_key: str = DEFAULT_ASR_KEY,
    loader: Callable[[str], Any] | None = None,
) -> Transcript:
    """Transcribe raw audio bytes into text.

    Returns a :class:`Transcript` with ``skipped=True`` for empty or silent
    recordings instead of raising, so the UI can render an inline warning.
    """
    model_id = ASR_MODELS.get(model_key, model_key)
    if not audio:
        return Transcript(text="", model=model_id, skipped=True, reason="empty_input")

    duration = _duration_of(audio)
    asr = (loader or load_asr)(model_id)
    try:
        output = asr(audio, return_language=True)
    except TypeError:  # pragma: no cover - older transformers
        output = asr(audio)
    except Exception as exc:  # pragma: no cover - audio/dependency dependent
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    if isinstance(output, list):  # pragma: no cover - chunked long-form output
        output = {"text": " ".join(str(chunk.get("text", "")) for chunk in output)}
    text = " ".join(str(output.get("text", "")).split())
    language = output.get("language") if isinstance(output, dict) else None

    if not is_meaningful_text(text):
        return Transcript(
            text=text,
            model=model_id,
            language=language,
            duration_seconds=duration,
            skipped=True,
            reason="no_speech_detected",
        )
    return Transcript(
        text=text,
        model=model_id,
        language=language,
        duration_seconds=duration,
    )


def reset_cache() -> None:
    """Drop cached ASR pipelines (frees memory)."""
    _CACHE.clear()
