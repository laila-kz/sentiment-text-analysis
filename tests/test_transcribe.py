"""Tests for the Whisper speech-to-text front end (no audio, no torch needed)."""

from __future__ import annotations

from typing import Any

import pytest

from transcribe import (
    ASR_MODELS,
    DEFAULT_ASR_KEY,
    Transcript,
    TranscriptionError,
    transcribe,
)


class FakeASR:
    """Minimal stand-in for the transformers ASR pipeline."""

    def __init__(self, text: str = "the battery lasts all day", **output: Any) -> None:
        self.text = text
        self.output = output
        self.calls: list[bytes] = []

    def __call__(self, audio: bytes, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(audio)
        return {"text": self.text, **self.output}


class TestTranscribe:
    def test_returns_transcript(self) -> None:
        asr = FakeASR(language="en")
        result = transcribe(b"audio-bytes", loader=lambda model_id: asr)
        assert result.is_usable
        assert result.text == "the battery lasts all day"
        assert result.language == "en"
        assert result.model == ASR_MODELS[DEFAULT_ASR_KEY]
        assert asr.calls == [b"audio-bytes"]

    def test_model_key_maps_to_a_checkpoint(self) -> None:
        seen: list[str] = []

        def loader(model_id: str) -> FakeASR:
            seen.append(model_id)
            return FakeASR()

        transcribe(b"a", model_key="tiny (39M, fastest)", loader=loader)
        assert seen == ["openai/whisper-tiny"]

    def test_explicit_model_id_is_passed_through(self) -> None:
        seen: list[str] = []

        def loader(model_id: str) -> FakeASR:
            seen.append(model_id)
            return FakeASR()

        transcribe(b"a", model_key="custom/checkpoint", loader=loader)
        assert seen == ["custom/checkpoint"]

    def test_empty_audio_is_skipped(self) -> None:
        result = transcribe(b"", loader=lambda model_id: FakeASR())
        assert result.skipped is True
        assert result.reason == "empty_input"

    def test_silence_is_skipped(self) -> None:
        result = transcribe(b"audio", loader=lambda model_id: FakeASR(text="  "))
        assert result.skipped is True
        assert result.reason == "no_speech_detected"

    def test_noise_only_transcript_is_skipped(self) -> None:
        result = transcribe(b"audio", loader=lambda model_id: FakeASR(text="\U0001f600"))
        assert result.skipped is True
        assert result.reason == "no_speech_detected"

    def test_pipeline_error_is_wrapped(self) -> None:
        class Boom:
            def __call__(self, audio: bytes, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("cuda oom")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe(b"audio", loader=lambda model_id: Boom())

    def test_load_failure_is_wrapped(self) -> None:
        def loader(model_id: str) -> Any:
            raise TranscriptionError(f"Failed to load {model_id}: offline")

        with pytest.raises(TranscriptionError, match="offline"):
            transcribe(b"audio", loader=loader)

    def test_legacy_pipeline_without_return_language(self) -> None:
        class Legacy:
            def __call__(self, audio: bytes) -> dict[str, Any]:
                return {"text": "legacy transcript"}

        result = transcribe(b"audio", loader=lambda model_id: Legacy())
        assert result.text == "legacy transcript"
        assert result.language is None

    def test_chunked_output_is_joined(self) -> None:
        class Chunked:
            def __call__(self, audio: bytes, **kwargs: Any) -> list[dict[str, str]]:
                return [{"text": "first part "}, {"text": "second part"}]

        result = transcribe(b"audio", loader=lambda model_id: Chunked())
        assert result.text == "first part second part"


class TestTranscript:
    def test_is_usable(self) -> None:
        assert Transcript(text="hello", model="m").is_usable
        assert not Transcript(text="", model="m").is_usable
        assert not Transcript(text="hello", model="m", skipped=True).is_usable

    def test_to_dict(self) -> None:
        payload = Transcript(
            text="hi", model="openai/whisper-base", language="en", duration_seconds=1.5
        ).to_dict()
        assert payload == {
            "text": "hi",
            "model": "openai/whisper-base",
            "language": "en",
            "duration_seconds": 1.5,
            "skipped": False,
            "reason": None,
        }

    def test_registry_is_ordered_smallest_first(self) -> None:
        keys = list(ASR_MODELS)
        assert keys[0].startswith("tiny")
        assert keys[-1].startswith("small")
        assert all(model_id.startswith("openai/whisper-") for model_id in ASR_MODELS.values())
