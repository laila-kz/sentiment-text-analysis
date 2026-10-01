"""Tests for the Pydantic request/response contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import sentiment
from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchSummaryResponse,
    HealthResponse,
    MemoryInfo,
    TextAnalysisResponse,
)
from sentiment import SentimentEngine


class TestAnalyzeRequest:
    def test_single_text_defaults(self) -> None:
        request = AnalyzeRequest(text="hello there")
        assert request.inputs == ["hello there"]
        assert request.is_batch is False
        assert request.model == "sentiment"
        assert request.temperature == 1.0
        assert request.include_logits is False

    def test_batch(self) -> None:
        request = AnalyzeRequest(texts=["a", "b", "c"], batch_size=4)
        assert request.inputs == ["a", "b", "c"]
        assert request.is_batch is True

    def test_requires_input(self) -> None:
        with pytest.raises(ValidationError, match="provide either"):
            AnalyzeRequest()

    def test_rejects_both_inputs(self) -> None:
        with pytest.raises(ValidationError, match="not both"):
            AnalyzeRequest(text="a", texts=["b"])

    def test_rejects_unknown_model(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="a", model="gpt-2")

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="a", nonsense=1)  # type: ignore[call-arg]

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="")

    def test_rejects_empty_batch(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(texts=[])

    def test_rejects_oversized_batch(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(texts=["a"] * 1001)

    @pytest.mark.parametrize("temperature", [0.0, -1.0, 11.0])
    def test_temperature_bounds(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="a", temperature=temperature)

    @pytest.mark.parametrize("batch_size", [0, 257])
    def test_batch_size_bounds(self, batch_size: int) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="a", batch_size=batch_size)

    def test_segments_require_single_text(self) -> None:
        with pytest.raises(ValidationError, match="single"):
            AnalyzeRequest(texts=["a"], include_segments=True)
        assert AnalyzeRequest(text="a", include_segments=True).include_segments is True

    def test_model_choices_are_constrained(self) -> None:
        assert AnalyzeRequest(text="a", model="emotion").model == "emotion"
        assert AnalyzeRequest(text="a", model="sentiment3").model == "sentiment3"


class TestResponses:
    def test_text_analysis_from_result(self, engine: SentimentEngine) -> None:
        result = engine.analyse("i love this", include_logits=True)
        response = TextAnalysisResponse.from_result(result, index=3)
        assert response.index == 3
        assert response.label == "positive"
        assert response.status == "ok"
        assert response.logits is not None
        assert response.uncertainty is not None
        assert set(response.distribution) == {"negative", "positive"}
        # internal engine fields are not leaked to clients
        assert not hasattr(response, "model_id")
        assert not hasattr(response, "device")

    def test_text_analysis_from_skipped_result(self, engine: SentimentEngine) -> None:
        response = TextAnalysisResponse.from_result(engine.analyse("   "), index=0)
        assert response.status == "skipped"
        assert response.reason == "empty_input"
        assert response.label is None
        assert response.confidence is None

    def test_batch_summary_from_result(self, engine: SentimentEngine) -> None:
        results = engine.analyse_batch(["good", "bad", ""])
        summary = BatchSummaryResponse.from_result(sentiment.summarise(results))
        assert summary.items == 3
        assert summary.analysed == 2
        assert summary.skipped == 1
        assert set(summary.label_counts) <= {"positive", "negative"}

    def test_response_requires_core_fields(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeResponse(request_id="x", model="sentiment")  # type: ignore[call-arg]

    def test_health_response_validates(self) -> None:
        payload = HealthResponse(
            status="ok",
            version="1.0.0",
            uptime_seconds=1.0,
            device="cpu",
            cuda_available=False,
            models_loaded=0,
            models_available=3,
            memory=MemoryInfo(rss_mb=10.0),
            checks={"registry_populated": True},
        )
        assert payload.memory.total_mb is None
        assert payload.model_dump()["checks"] == {"registry_populated": True}

    def test_health_status_is_constrained(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(
                status="exploded",  # type: ignore[arg-type]
                version="1",
                uptime_seconds=0.0,
                device="cpu",
                cuda_available=False,
                models_loaded=0,
                models_available=1,
                memory=MemoryInfo(rss_mb=1.0),
            )
