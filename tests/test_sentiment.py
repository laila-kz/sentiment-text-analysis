"""Tests for the inference engine: registry, batching, calibration, edge cases.

No model weights are downloaded: :class:`tests.conftest.FakeLoader` supplies
deterministic logits, so the suite runs in milliseconds.
"""

from __future__ import annotations

import math

import pytest
from conftest import FakeLoader, FakePipeline

import sentiment
from sentiment import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    AnalysisResult,
    EmptyTextError,
    ModelHandle,
    ModelKind,
    ModelSpec,
    SentimentEngine,
    UnknownModelError,
    available_models,
    get_model_spec,
    resolve_device,
    summarise,
)
from utils import softmax


class TestRegistry:
    def test_default_model_is_sentiment(self) -> None:
        assert DEFAULT_MODEL == "sentiment"
        assert MODEL_REGISTRY[DEFAULT_MODEL].default is True
        assert len([spec for spec in MODEL_REGISTRY.values() if spec.default]) == 1

    def test_emotion_model_has_six_labels(self) -> None:
        spec = get_model_spec("emotion")
        assert spec.kind is ModelKind.EMOTION
        assert spec.num_labels == 6
        assert "joy" in spec.labels and "anger" in spec.labels
        assert "emotion" in spec.hf_id

    def test_binary_sentiment_labels(self) -> None:
        spec = get_model_spec("sentiment")
        assert spec.kind is ModelKind.SENTIMENT
        assert spec.labels == ("negative", "positive")

    def test_available_models_puts_default_first(self) -> None:
        assert available_models()[0].key == DEFAULT_MODEL
        assert len(available_models()) == len(MODEL_REGISTRY)

    def test_get_model_spec_accepts_none_and_case(self) -> None:
        assert get_model_spec(None).key == DEFAULT_MODEL
        assert get_model_spec("SENTIMENT").key == "sentiment"
        assert get_model_spec(get_model_spec("emotion")).key == "emotion"

    def test_unknown_model_raises_with_helpful_message(self) -> None:
        with pytest.raises(UnknownModelError) as excinfo:
            get_model_spec("does-not-exist")
        message = str(excinfo.value)
        assert "does-not-exist" in message
        assert "emotion" in message  # lists the alternatives

    def test_spec_to_dict_is_json_friendly(self) -> None:
        payload = get_model_spec("emotion").to_dict()
        assert payload["num_labels"] == 6
        assert payload["kind"] == "emotion"
        import json

        assert json.loads(json.dumps(payload))["model_id"].endswith("emotion")


class TestDeviceResolution:
    def test_cpu_is_honoured(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_cuda_falls_back_when_unavailable(self) -> None:
        expected = "cuda" if sentiment.cuda_available() else "cpu"
        assert resolve_device("cuda") == expected

    def test_auto_matches_availability(self) -> None:
        expected = "cuda" if sentiment.cuda_available() else "cpu"
        assert resolve_device("auto") == expected
        assert resolve_device(None) == expected

    def test_cuda_available_is_boolean(self) -> None:
        assert isinstance(sentiment.cuda_available(), bool)


class TestSingleAnalysis:
    def test_returns_full_payload(self, engine: SentimentEngine) -> None:
        result = engine.analyse("I love this")
        assert result.is_ok
        assert result.label == "positive"
        assert result.confidence is not None and result.confidence > 0.9
        assert set(result.distribution) == {"negative", "positive"}
        assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-4)
        assert result.model == "sentiment"
        assert result.model_id.endswith("sst-2-english")
        assert result.processing_ms >= 0.0

    def test_confidence_matches_distribution(self, engine: SentimentEngine) -> None:
        result = engine.analyse("terrible and useless")
        assert result.label == "negative"
        assert result.confidence == pytest.approx(result.distribution["negative"], abs=1e-5)

    def test_uncertainty_metrics_are_consistent(self, engine: SentimentEngine) -> None:
        result = engine.analyse("i love this")
        uncertainty = result.uncertainty
        assert uncertainty is not None
        assert 0.0 <= uncertainty.normalized_entropy <= 1.0
        assert uncertainty.margin == pytest.approx(
            result.confidence - result.distribution["negative"], abs=1e-5
        )
        assert uncertainty.is_uncertain is False

    def test_ambiguous_text_is_flagged_uncertain(self, engine: SentimentEngine) -> None:
        result = engine.analyse("meh")
        assert result.uncertainty is not None
        assert result.uncertainty.normalized_entropy > 0.5
        assert result.uncertainty.is_uncertain is True

    def test_include_logits(self, engine: SentimentEngine) -> None:
        without = engine.analyse("i love this")
        with_logits = engine.analyse("i love this", include_logits=True)
        assert without.logits is None
        assert with_logits.logits is not None and len(with_logits.logits) == 2
        assert with_logits.ranked[0].logit == pytest.approx(max(with_logits.logits))
        # probabilities are recoverable from the logits
        assert with_logits.confidence == pytest.approx(max(softmax(with_logits.logits)), abs=1e-5)

    def test_to_dict_roundtrip_keys(self, engine: SentimentEngine) -> None:
        payload = engine.analyse("good", include_logits=True).to_dict()
        assert payload["status"] == "ok"
        assert payload["calibration"] == "temperature_scaling"
        assert payload["logits"] is not None
        assert payload["uncertainty"]["entropy_bits"] >= 0
        assert "distribution" in payload

    def test_emotion_model_switch(self, engine: SentimentEngine) -> None:
        result = engine.analyse("I am so happy today", model="emotion")
        assert result.model == "emotion"
        assert result.label == "love"
        assert len(result.distribution) == 6
        assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-4)

    def test_three_class_model(self, engine: SentimentEngine) -> None:
        result = engine.analyse("meh", model="sentiment3")
        assert len(result.distribution) == 3
        assert set(result.distribution) == {"negative", "neutral", "positive"}

    def test_unknown_model_propagates(self, engine: SentimentEngine) -> None:
        with pytest.raises(UnknownModelError):
            engine.analyse("hello", model="nope")


class TestCalibration:
    def test_high_temperature_flattens_confidence(self, engine: SentimentEngine) -> None:
        sharp = engine.analyse("i love this", temperature=0.5)
        soft = engine.analyse("i love this", temperature=5.0)
        assert sharp.confidence > soft.confidence
        assert soft.uncertainty.normalized_entropy > sharp.uncertainty.normalized_entropy
        assert soft.temperature == 5.0

    def test_temperature_keeps_distribution_normalised(self, engine: SentimentEngine) -> None:
        result = engine.analyse("i love this", temperature=3.0, model="emotion")
        assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-4)

    def test_probabilities_match_temperature_scaled_softmax(self, engine: SentimentEngine) -> None:
        result = engine.analyse("good", temperature=2.0, include_logits=True)
        expected = softmax(list(result.logits), temperature=2.0)
        assert sorted(result.distribution.values(), reverse=True)[0] == pytest.approx(
            max(expected), abs=1e-5
        )


class TestEdgeCases:
    @pytest.mark.parametrize("value", ["", "   ", "\n\t", "\u200b"])
    def test_blank_text_is_skipped(self, engine: SentimentEngine, value: str) -> None:
        result = engine.analyse(value)
        assert result.status == "skipped"
        assert result.reason in {"empty_input", "blank_input"}
        assert result.label is None and result.confidence is None
        assert result.distribution == {}

    @pytest.mark.parametrize("value", ["\U0001f600", "\U0001f44d\U0001f44d", "!!!", "???"])
    def test_emoji_only_text_is_skipped(self, engine: SentimentEngine, value: str) -> None:
        result = engine.analyse(value)
        assert result.status == "skipped"
        assert result.reason == "no_word_characters"

    def test_skipped_results_never_reach_the_model(self, engine: SentimentEngine) -> None:
        engine.analyse("")
        assert loader_calls(engine) == []

    def test_strict_mode_raises_on_bad_input(self, engine: SentimentEngine) -> None:
        with pytest.raises(EmptyTextError):
            engine.analyse_batch(["fine", "   "], strict=True)
        assert issubclass(EmptyTextError, ValueError)

    def test_very_long_text_is_truncated_with_warning(self, engine: SentimentEngine) -> None:
        limit = engine.settings.max_text_length
        result = engine.analyse("word " * (limit // 2))
        assert len(result.text) == limit
        assert any("truncated" in warning for warning in result.warnings)

    def test_non_string_input_is_coerced(self, engine: SentimentEngine) -> None:
        result = engine.analyse(42)  # type: ignore[arg-type]
        assert result.is_ok

    def test_batch_rejects_a_bare_string(self, engine: SentimentEngine) -> None:
        with pytest.raises(TypeError):
            engine.analyse_batch("just one string")  # type: ignore[arg-type]

    def test_empty_batch(self, engine: SentimentEngine) -> None:
        assert engine.analyse_batch([]) == []


class TestBatching:
    def test_results_keep_input_order(self, engine: SentimentEngine) -> None:
        texts = ["i love this", "terrible and useless", "meh", "good"]
        results = engine.analyse_batch(texts)
        assert [result.text for result in results] == texts
        assert [result.label for result in results] == [
            "positive",
            "negative",
            "positive",
            "positive",
        ]

    def test_vectorised_batching_chunks_by_batch_size(
        self, engine: SentimentEngine, loader: FakeLoader
    ) -> None:
        texts = ["good"] * 7
        engine.analyse_batch(texts, batch_size=3)
        pipeline = loader.pipeline("sentiment")
        assert pipeline.batch_sizes == [3, 3, 1]

    def test_skipped_items_do_not_consume_batch_capacity(
        self, engine: SentimentEngine, loader: FakeLoader
    ) -> None:
        engine.analyse_batch(["", "good", "   ", "bad"], batch_size=2)
        pipeline = loader.pipeline("sentiment")
        assert pipeline.batch_sizes == [2]

    def test_latency_is_shared_across_batch(self, engine: SentimentEngine) -> None:
        results = engine.analyse_batch(["good"] * 4, batch_size=4)
        assert all(result.processing_ms > 0 for result in results)
        total = sum(result.processing_ms for result in results)
        assert total < 1_000

    def test_models_are_loaded_once_and_cached(
        self, engine: SentimentEngine, loader: FakeLoader
    ) -> None:
        engine.analyse("one")
        engine.analyse("two")
        engine.analyse("three", model="emotion")
        assert loader.loads.count("sentiment") == 1
        assert loader.loads.count("emotion") == 1
        assert engine.loaded_models == ("emotion", "sentiment")

    def test_clear_drops_models(self, engine: SentimentEngine, loader: FakeLoader) -> None:
        engine.analyse("good")
        engine.clear()
        assert engine.loaded_models == ()
        engine.analyse("good")
        assert loader.loads.count("sentiment") == 2

    def test_warmup_all_loads_every_model(self, engine: SentimentEngine) -> None:
        engine.warmup_all()
        assert set(engine.loaded_models) == set(MODEL_REGISTRY)


class TestSegmentsAndSummary:
    def test_segments_align_with_sentences(self, engine: SentimentEngine) -> None:
        segments = engine.analyse_segments("I love it. It is awful.")
        assert [segment.text for segment in segments] == ["I love it.", "It is awful."]
        assert all(segment.label for segment in segments)
        assert all(segment.score is not None for segment in segments)

    def test_segments_on_blank_text(self, engine: SentimentEngine) -> None:
        assert engine.analyse_segments("   ") == []

    def test_summarise_counts_labels(self, engine: SentimentEngine) -> None:
        results = engine.analyse_batch(["i love this", "terrible and useless", "", "meh"])
        summary = summarise(results)
        assert summary.items == 4
        assert summary.analysed == 3
        assert summary.skipped == 1
        assert sum(summary.label_counts.values()) == 3
        assert 0.0 < summary.mean_confidence <= 1.0
        assert summary.to_dict()["label_counts"] == summary.label_counts

    def test_summarise_of_empty_batch(self) -> None:
        summary = summarise([])
        assert (summary.items, summary.analysed, summary.mean_confidence) == (0, 0, 0.0)


class TestHandleInternals:
    def test_pipeline_fallback_converts_scores_to_logits(self) -> None:
        spec = get_model_spec("sentiment")
        pipeline = FakePipeline(spec)
        handle = ModelHandle(spec=spec, pipeline_obj=pipeline, tokenizer=None, network=None)
        logits = handle.predict_logits(["i love this"])
        assert len(logits) == 1 and len(logits[0]) == 2
        expected = softmax(logits[0])
        assert max(expected) > 0.95

    def test_labels_follow_the_config_when_they_match(self) -> None:
        class Config:
            num_labels = 3
            id2label = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}

        class Network:
            config = Config()

        spec = ModelSpec(
            key="custom",
            hf_id="custom/model",
            kind=ModelKind.SENTIMENT,
            labels=("negative", "neutral", "positive"),
            description="test",
        )
        handle = ModelHandle(spec=spec, pipeline_obj=None, tokenizer=None, network=Network())
        assert handle.labels == ("negative", "positive", "neutral")
        assert handle.labels.index("negative") == 0  # config order == logit order

    def test_labels_fall_back_to_the_registry(self) -> None:
        class Config:
            num_labels = 2
            id2label = {0: "LABEL_0", 1: "LABEL_1"}

        class Network:
            config = Config()

        handle = ModelHandle(
            spec=get_model_spec("sentiment"), pipeline_obj=None, tokenizer=None, network=Network()
        )
        assert handle.labels == ("negative", "positive")

    def test_logit_width_mismatch_is_flagged(self, engine: SentimentEngine) -> None:
        class NarrowHandle(ModelHandle):
            """Pretends the head has fewer outputs than the label space."""

            def predict_logits(self, texts):  # type: ignore[no-untyped-def, override]
                return [[2.0] for _ in texts]

        spec = get_model_spec("sentiment")
        engine._handles[spec.key] = NarrowHandle(  # noqa: SLF001 - deliberate injection
            spec=spec, pipeline_obj=FakePipeline(spec), tokenizer=None, network=None
        )
        result = engine.analyse("good")
        assert result.is_ok
        assert len(result.distribution) == 1
        assert any("logit_width_mismatch" in warning for warning in result.warnings)

    def test_torch_path_is_used_when_a_network_is_present(self) -> None:
        pytest.importorskip("torch", reason="torch not installed; pipeline path is covered above")

        class FakeTensor:
            def to(self, *_: object, **__: object) -> FakeTensor:
                return self

            def detach(self) -> FakeTensor:
                return self

            def tolist(self) -> list[list[float]]:
                return [[0.25, 2.75]]

        class FakeOutputs:
            logits = FakeTensor()

        class FakeTokenizer:
            def __call__(self, texts, **_: object) -> dict[str, FakeTensor]:
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

        class FakeNetwork:
            def __call__(self, **_: object) -> FakeOutputs:
                return FakeOutputs()

        spec = get_model_spec("sentiment")
        handle = ModelHandle(
            spec=spec,
            pipeline_obj=None,
            tokenizer=FakeTokenizer(),
            network=FakeNetwork(),
            device="cpu",
        )
        assert handle.predict_logits(["anything"]) == [[0.25, 2.75]]

    def test_engine_info_lists_every_model(self, engine: SentimentEngine) -> None:
        engine.analyse("good")
        info = engine.info()
        assert len(info["models"]) == len(MODEL_REGISTRY)
        assert info["default_model"] == DEFAULT_MODEL
        loaded = {model["key"]: model["loaded"] for model in info["models"]}
        assert loaded[DEFAULT_MODEL] is True
        assert loaded["emotion"] is False

    def test_torch_path_requires_torch(self) -> None:
        pytest.importorskip("torch")
        assert sentiment.cuda_available() in {True, False}


class TestBenchmark:
    def test_benchmark_reports_throughput(self, engine: SentimentEngine) -> None:
        report = engine.benchmark(batch_size=8)
        payload = report.to_dict()
        assert payload["texts"] == 64
        assert payload["batch_size"] == 8
        assert payload["texts_per_second"] > 0
        assert payload["latency_ms_p95"] >= payload["latency_ms_p50"] >= 0
        assert payload["device"] in {"cpu", "cuda"}

    def test_benchmark_module_helper(self, engine: SentimentEngine) -> None:
        assert sentiment.benchmark(batch_size=4).texts == 64


class TestModuleLevelApi:
    def test_get_engine_is_a_singleton(self) -> None:
        first = sentiment.get_engine()
        assert sentiment.get_engine() is first

    def test_set_engine_installs_overrides(self, engine: SentimentEngine) -> None:
        assert sentiment.get_engine() is engine
        sentiment.set_engine(None)
        assert sentiment.get_engine() is not engine

    def test_analyse_uses_the_installed_engine(self, engine: SentimentEngine) -> None:
        assert sentiment.analyse("i love this").label == "positive"

    def test_analyse_batch_module_helper(self, engine: SentimentEngine) -> None:
        results = sentiment.analyse_batch(["good", ""], include_logits=True)
        assert [result.status for result in results] == ["ok", "skipped"]

    def test_get_classifier_returns_the_pipeline(self, engine: SentimentEngine) -> None:
        classifier = sentiment.get_classifier("sentiment")
        assert isinstance(classifier, FakePipeline)
        scores = {item["label"]: item["score"] for item in classifier("i love this")[0]}
        assert scores["positive"] > 0.9

    def test_clear_cache_module_helper(self, engine: SentimentEngine) -> None:
        engine.analyse("good")
        sentiment.clear_cache()
        assert engine.loaded_models == ()

    def test_module_analyse_segments(self, engine: SentimentEngine) -> None:
        assert len(sentiment.analyse_segments("Nice. Bad.")) == 2


class TestResultContainer:
    def test_skipped_constructor(self) -> None:
        result = AnalysisResult.skipped("  ", "blank_input", model="emotion")
        assert result.is_ok is False
        assert result.reason == "blank_input"
        assert result.model == "emotion"

    def test_to_dict_can_hide_logits(self) -> None:
        result = AnalysisResult(text="x", status="ok", label="positive", confidence=0.9)
        assert "logits" not in result.to_dict(include_logits=False)


def loader_calls(engine: SentimentEngine) -> list[list[str]]:
    """Total number of texts the fake pipeline has seen for any model."""
    return [
        text
        for handle in engine._handles.values()  # noqa: SLF001 - test introspection
        for call in getattr(handle.pipeline, "calls", [])
        for text in call
    ]


def test_logit_probabilities_match_distribution(engine: SentimentEngine) -> None:
    result = engine.analyse("i love this", include_logits=True)
    probabilities = softmax(list(result.logits))
    for label, probability in zip(get_model_spec("sentiment").labels, probabilities, strict=True):
        assert result.distribution[label] == pytest.approx(probability, abs=1e-5)


def test_entropy_of_uniform_distribution_is_one_bit(engine: SentimentEngine) -> None:
    result = engine.analyse("something the fake model has never seen")
    assert math.isclose(result.uncertainty.entropy_bits, 1.0, abs_tol=1e-3)
    assert result.uncertainty.normalized_entropy == pytest.approx(1.0, abs=1e-3)
