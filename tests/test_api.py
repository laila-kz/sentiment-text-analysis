"""Endpoint tests for the FastAPI service (fake engine, no model downloads)."""

from __future__ import annotations

import builtins
import io
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeLoader, make_app_client

import api
from sentiment import MODEL_REGISTRY, SentimentEngine


class TestServiceMetadata:
    def test_root_lists_endpoints(self, client: Any) -> None:
        payload = client.get("/").json()
        assert payload["docs"] == "/docs"
        assert payload["endpoints"]["analyze"] == "POST /v1/analyze"

    def test_openapi_is_documented(self, client: Any) -> None:
        spec = client.get("/openapi.json").json()
        assert "/v1/analyze" in spec["paths"]
        assert "AnalyzeRequest" in spec["components"]["schemas"]
        assert spec["info"]["version"] == "1.0.0"

    def test_unknown_route_is_json_404(self, client: Any) -> None:
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.json()["detail"] == "Not Found"

    def test_cors_headers_present(self, client: Any) -> None:
        response = client.get("/", headers={"Origin": "http://example.com"})
        assert response.headers.get("access-control-allow-origin") == "*"


class TestAnalyzeEndpoint:
    def test_single_text(self, client: Any) -> None:
        response = client.post("/v1/analyze", json={"text": "i love this"})
        assert response.status_code == 200
        body = response.json()
        assert body["batch"] is False
        assert body["model"] == "sentiment"
        assert body["model_id"].endswith("sst-2-english")
        assert body["device"] in {"cpu", "cuda"}
        assert body["summary"]["items"] == 1
        item = body["results"][0]
        assert item["index"] == 0
        assert item["label"] == "positive"
        assert item["confidence"] > 0.9
        assert item["distribution"]["negative"] < 0.1
        assert item["logits"] is None
        assert body["took_ms"] >= 0

    def test_single_text_with_logits(self, client: Any) -> None:
        body = client.post(
            "/v1/analyze", json={"text": "i love this", "include_logits": True}
        ).json()
        logits = body["results"][0]["logits"]
        assert logits is not None and len(logits) == 2
        assert max(logits) > min(logits)

    def test_batch(self, client: Any) -> None:
        texts = ["i love this", "terrible and useless", "good", "bad"]
        body = client.post("/v1/analyze", json={"texts": texts}).json()
        assert body["batch"] is True
        assert [item["text"] for item in body["results"]] == texts
        assert [item["index"] for item in body["results"]] == [0, 1, 2, 3]
        assert body["summary"]["analysed"] == 4
        assert body["summary"]["label_counts"] == {"negative": 2, "positive": 2}

    def test_batch_reports_skipped_items(self, client: Any) -> None:
        body = client.post("/v1/analyze", json={"texts": ["good", "", "\U0001f600"]}).json()
        statuses = [(item["status"], item["reason"]) for item in body["results"]]
        assert statuses == [
            ("ok", None),
            ("skipped", "empty_input"),
            ("skipped", "no_word_characters"),
        ]
        assert body["summary"]["skipped"] == 2
        assert body["summary"]["analysed"] == 1

    def test_batch_is_vectorised(self, client: Any, loader: FakeLoader) -> None:
        client.post("/v1/analyze", json={"texts": ["good"] * 6, "batch_size": 4})
        assert loader.pipeline("sentiment").batch_sizes == [4, 2]

    def test_emotion_model(self, client: Any) -> None:
        body = client.post("/v1/analyze", json={"text": "i love this", "model": "emotion"}).json()
        assert body["model"] == "emotion"
        assert len(body["results"][0]["distribution"]) == 6
        assert body["results"][0]["label"] == "love"

    def test_temperature_is_applied(self, client: Any) -> None:
        sharp = client.post("/v1/analyze", json={"text": "i love this", "temperature": 0.5}).json()
        soft = client.post("/v1/analyze", json={"text": "i love this", "temperature": 5.0}).json()
        assert sharp["results"][0]["confidence"] > soft["results"][0]["confidence"]
        assert soft["results"][0]["temperature"] == 5.0

    def test_segments(self, client: Any) -> None:
        body = client.post(
            "/v1/analyze",
            json={"text": "I love it. It is awful.", "include_segments": True},
        ).json()
        segments = body["results"][0]["segments"]
        assert [segment["text"] for segment in segments] == ["I love it.", "It is awful."]
        assert all(segment["label"] for segment in segments)

    def test_batch_too_large(self, engine: SentimentEngine) -> None:
        client = make_app_client(engine, max_batch_items=3)
        response = client.post("/v1/analyze", json={"texts": ["good"] * 4})
        assert response.status_code == 413
        assert "exceeds the limit" in response.json()["detail"]


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"text": "a", "texts": ["b"]},
            {"texts": []},
            {"text": ""},
            {"text": 42},
            {"text": "a", "model": "unknown"},
            {"text": "a", "temperature": 0},
            {"text": "a", "temperature": 99},
            {"text": "a", "batch_size": 0},
            {"text": "a", "unexpected": True},
            {"texts": ["a"], "include_segments": True},
            {"texts": "not-a-list"},
        ],
    )
    def test_rejects_bad_payloads(self, client: Any, payload: dict[str, Any]) -> None:
        response = client.post("/v1/analyze", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["detail"] == "Request validation failed."
        assert isinstance(body["errors"], list) and body["errors"]
        assert body["request_id"]

    def test_error_locations_are_actionable(self, client: Any) -> None:
        errors = client.post("/v1/analyze", json={"texts": ["a"], "include_segments": True}).json()
        assert errors["errors"][0]["location"] == "body"
        assert "single" in errors["errors"][0]["message"]


class TestModelsEndpoint:
    def test_lists_the_registry(self, client: Any) -> None:
        body = client.get("/v1/models").json()
        assert body["default_model"] == "sentiment"
        assert {model["key"] for model in body["models"]} == set(MODEL_REGISTRY)
        assert body["loaded_models"] == []

    def test_reflects_loaded_state(self, client: Any) -> None:
        client.post("/v1/analyze", json={"text": "good"})
        body = client.get("/v1/models").json()
        assert body["loaded_models"] == ["sentiment"]
        loaded = {model["key"]: model["loaded"] for model in body["models"]}
        assert loaded["sentiment"] is True
        assert loaded["emotion"] is False

    def test_benchmark(self, client: Any) -> None:
        body = client.get("/v1/benchmark?batch_size=8&repeats=1").json()
        assert body["model"] == "sentiment"
        assert body["texts_per_second"] > 0
        assert body["batch_size"] == 8

    def test_benchmark_unknown_model(self, client: Any) -> None:
        assert client.get("/v1/benchmark?model=nope").status_code == 404

    def test_benchmark_validates_query_params(self, client: Any) -> None:
        assert client.get("/v1/benchmark?batch_size=0").status_code == 422
        assert client.get("/v1/benchmark?repeats=99").status_code == 422


class TestHealth:
    def test_reports_models_and_memory(self, client: Any) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"
        assert body["models_available"] == len(MODEL_REGISTRY)
        assert body["models_loaded"] == 0
        assert body["uptime_seconds"] >= 0
        assert body["memory"]["rss_mb"] > 0
        assert body["checks"]["registry_populated"] is True
        assert body["checks"]["model_loaded"] is False

    def test_health_is_cacheable_and_never_rate_limited(self, engine: SentimentEngine) -> None:
        client = make_app_client(engine, rate_limit_requests=1)
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_health_reports_loaded_models(self, client: Any) -> None:
        client.post("/v1/analyze", json={"text": "good"})
        body = client.get("/health").json()
        assert body["models_loaded"] == 1
        assert body["checks"]["model_loaded"] is True

    def test_memory_falls_back_to_proc_without_psutil(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CI ships no psutil, so /proc has to carry the memory report."""
        statm = "1200 300 40 4 0 900 0\n"
        meminfo = (
            "MemTotal:       8192000 kB\nMemFree:         100 kB\nMemAvailable:    4096000 kB\n"
        )

        def fake_open(path, mode="r", *args, **kwargs):  # noqa: ANN001, ANN202
            name = str(path).replace("\\", "/")
            if name == "/proc/self/statm":
                return io.StringIO(statm)
            if name == "/proc/meminfo":
                return io.StringIO(meminfo)
            raise AssertionError(f"unexpected path: {path}")

        real_import = builtins.__import__

        def no_psutil(name, *args, **kwargs):  # noqa: ANN001, ANN202
            if name == "psutil":
                raise ImportError("psutil is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fake_open)
        monkeypatch.setattr(builtins, "__import__", no_psutil)
        monkeypatch.setattr(os, "sysconf", lambda name: 4096, raising=False)  # type: ignore[arg-type]

        info = api._memory_info()
        assert info.rss_mb == pytest.approx(300 * 4096 / (1024 * 1024), abs=0.01)
        assert info.total_mb == pytest.approx(8192000 / 1024, abs=0.01)
        assert info.available_mb == pytest.approx(4096000 / 1024, abs=0.01)
        assert info.percent_used == pytest.approx(50.0, abs=0.01)

        body = client.get("/health").json()
        assert body["checks"]["memory_readable"] is True
        assert body["memory"]["rss_mb"] > 0

    def test_memory_reports_zero_when_no_source_is_readable(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_open(path, mode="r", *args, **kwargs):  # noqa: ANN001, ANN202
            raise OSError(path)

        real_import = builtins.__import__

        def no_psutil(name, *args, **kwargs):  # noqa: ANN001, ANN202
            if name == "psutil":
                raise ImportError("psutil is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(Path, "open", no_open)
        monkeypatch.setattr(builtins, "__import__", no_psutil)
        monkeypatch.setattr(os, "sysconf", lambda name: 4096, raising=False)  # type: ignore[arg-type]

        assert api._memory_info().rss_mb == 0.0
        body = client.get("/health").json()
        assert body["checks"]["memory_readable"] is False
        assert body["status"] == "ok"


class TestMiddleware:
    def test_request_id_header_is_returned(self, client: Any) -> None:
        response = client.post("/v1/analyze", json={"text": "good"})
        assert response.headers["x-request-id"]
        assert response.json()["request_id"] == response.headers["x-request-id"]

    def test_client_request_id_is_propagated(self, client: Any) -> None:
        response = client.get("/", headers={"X-Request-ID": "trace-42"})
        assert response.headers["x-request-id"] == "trace-42"

    def test_duration_header_is_returned(self, client: Any) -> None:
        response = client.get("/")
        assert float(response.headers["x-process-time"]) >= 0

    def test_error_responses_carry_the_request_id(self, client: Any) -> None:
        response = client.get("/nope", headers={"X-Request-ID": "trace-err"})
        assert response.json()["request_id"] == "trace-err"

    def test_rate_limiting(self, engine: SentimentEngine) -> None:
        client = make_app_client(engine, rate_limit_requests=3, rate_limit_window=60.0)
        codes = [client.post("/v1/analyze", json={"text": "good"}).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]

    def test_rate_limit_response_shape(self, engine: SentimentEngine) -> None:
        client = make_app_client(engine, rate_limit_requests=1)
        client.post("/v1/analyze", json={"text": "good"})
        response = client.post("/v1/analyze", json={"text": "good"})
        assert response.status_code == 429
        body = response.json()
        assert body["detail"] == "Rate limit exceeded. Retry later."
        assert body["request_id"]
        assert response.headers["retry-after"]
        assert response.headers["x-ratelimit-limit"] == "1"
        assert response.headers["x-request-id"] == body["request_id"]

    def test_rate_limit_is_per_client(self, engine: SentimentEngine) -> None:
        client = make_app_client(engine, rate_limit_requests=1)
        first = client.post(
            "/v1/analyze", json={"text": "good"}, headers={"X-Forwarded-For": "1.1.1.1"}
        )
        second = client.post(
            "/v1/analyze", json={"text": "good"}, headers={"X-Forwarded-For": "2.2.2.2"}
        )
        assert first.status_code == 200
        assert second.status_code == 200


class TestMetrics:
    def test_counters_are_exposed_in_prometheus_format(self, client: Any) -> None:
        client.post("/v1/analyze", json={"texts": ["good", ""]})
        body = client.get("/metrics").text
        assert "http_requests_total" in body
        assert "analysis_requests_total" in body
        assert "texts_analyzed_total 1.0" in body
        assert "texts_skipped_total 1.0" in body
        assert 'analysis_requests_by_model{model="sentiment"}' in body

    def test_metrics_content_type(self, client: Any) -> None:
        response = client.get("/metrics")
        assert response.headers["content-type"].startswith("text/plain")


class TestDependencyInjection:
    def test_default_engine_dependency_is_the_singleton(self) -> None:
        import sentiment

        assert api.get_engine_dep() is sentiment.get_engine()

    def test_override_is_used(self, engine: SentimentEngine) -> None:
        client = make_app_client(engine)
        assert client.post("/v1/analyze", json={"text": "good"}).status_code == 200
        assert engine.loaded_models == ("sentiment",)

    def test_model_load_failure_returns_503(self, client: Any) -> None:
        def boom(spec):  # noqa: ARG001
            raise RuntimeError("checkpoint unavailable")

        client.app.dependency_overrides[api.get_engine_dep] = lambda: SentimentEngine(loader=boom)
        response = client.post("/v1/analyze", json={"text": "good"})
        assert response.status_code == 503
        assert "checkpoint unavailable" in response.json()["detail"]

    def test_unexpected_error_returns_500(self, client: Any, monkeypatch) -> None:
        def boom(*args, **kwargs):  # noqa: ARG001
            raise ZeroDivisionError("boom")

        import sentiment

        monkeypatch.setattr(sentiment.SentimentEngine, "analyse_batch", boom)
        response = client.post("/v1/analyze", json={"text": "good"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error."
