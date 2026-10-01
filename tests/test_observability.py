"""Tests for logging, metrics and the pure-ASGI middleware."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest

from observability import (
    ConsoleLogFormatter,
    JsonLogFormatter,
    Metrics,
    RateLimitMiddleware,
    RequestContextMiddleware,
    _normalise_path,
    configure_logging,
)


def make_record(**extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="api.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="request",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestLogging:
    def test_json_formatter_is_one_line_per_record(self) -> None:
        payload = json.loads(JsonLogFormatter().format(make_record(duration_ms=1.5)))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "api.access"
        assert payload["message"] == "request"
        assert payload["duration_ms"] == 1.5
        assert "ts" in payload

    def test_json_formatter_includes_exceptions(self) -> None:
        try:
            raise ValueError("kaboom")
        except ValueError:
            record = make_record()
            record.exc_info = __import__("sys").exc_info()  # noqa: SLF001
        payload = json.loads(JsonLogFormatter().format(record))
        assert "ValueError: kaboom" in payload["exception"]

    def test_json_formatter_survives_unserialisable_extras(self) -> None:
        payload = json.loads(JsonLogFormatter().format(make_record(obj=object())))
        assert "obj" in payload

    def test_console_formatter_renders_extras(self) -> None:
        line = ConsoleLogFormatter().format(make_record(status=200))
        assert "INFO" in line and "api.access" in line and "status=200" in line

    def test_configure_logging_installs_single_handler(self) -> None:
        root = logging.getLogger()
        original = list(root.handlers)
        try:
            configure_logging("DEBUG", json_format=True)
            assert len(root.handlers) == 1
            assert root.level == logging.DEBUG
            configure_logging("WARNING", json_format=False)
            assert root.level == logging.WARNING
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in original:
                root.addHandler(handler)

    def test_configure_logging_accepts_unknown_level(self) -> None:
        root = logging.getLogger()
        original = list(root.handlers)
        try:
            configure_logging("NOT_A_LEVEL")
            assert root.level == logging.INFO
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in original:
                root.addHandler(handler)


class TestMetrics:
    def test_counters_accumulate(self) -> None:
        counter = Metrics()
        counter.increment("requests")
        counter.increment("requests", 2)
        assert counter.snapshot()["requests"] == 3.0

    def test_labelled_counters(self) -> None:
        counter = Metrics()
        counter.increment_labeled("by_model", {"model": "sentiment"})
        counter.increment_labeled("by_model", {"model": "emotion"})
        snapshot = counter.snapshot()
        assert snapshot['by_model{model="sentiment"}'] == 1.0
        assert snapshot['by_model{model="emotion"}'] == 1.0

    def test_latency_statistics(self) -> None:
        counter = Metrics()
        for value in (10.0, 20.0, 30.0):
            counter.observe_latency(value)
        snapshot = counter.snapshot()
        assert snapshot["latency_ms_sum"] == 60.0
        assert snapshot["latency_ms_max"] == 30.0
        assert snapshot["latency_ms_avg"] == 20.0

    def test_render_is_prometheus_shaped(self) -> None:
        counter = Metrics()
        counter.increment("http_requests_total")
        counter.increment_labeled("by_model", {"model": "emotion"})
        rendered = counter.render()
        assert "# TYPE http_requests_total counter" in rendered
        assert "http_requests_total 1.0" in rendered
        assert 'by_model{model="emotion"} 1.0' in rendered
        assert rendered.endswith("\n")

    def test_latency_history_is_bounded(self) -> None:
        counter = Metrics()
        for index in range(10_050):
            counter.observe_latency(float(index))
        assert "latency_ms_avg" in counter.snapshot()

    def test_reset(self) -> None:
        counter = Metrics()
        counter.increment("requests")
        counter.reset()
        assert "requests" not in counter.snapshot()


class TestNormalisePath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/v1/analyze", "/v1/analyze"),
            ("/", "/"),
            ("", "/"),
            ("/items/12345", "/items/{12345}"),
            (
                "/items/3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "/items/{3fa85f64-5717-4562-b3fc-2c963f66afa6}",
            ),
        ],
    )
    def test_bounded_cardinality(self, path: str, expected: str) -> None:
        assert _normalise_path(path) == expected


class TestRateLimiterLogic:
    def build(self, requests: int = 2, window: float = 60.0) -> RateLimitMiddleware:
        return RateLimitMiddleware(app=None, requests=requests, window=window)  # type: ignore[arg-type]

    def test_allows_up_to_the_budget(self) -> None:
        limiter = self.build(requests=2)
        assert limiter._check("ip", 0.0)[0] is True
        allowed, remaining, _ = limiter._check("ip", 1.0)
        assert allowed is True and remaining == 0

    def test_blocks_beyond_the_budget(self) -> None:
        limiter = self.build(requests=1)
        limiter._check("ip", 0.0)
        allowed, remaining, retry_after = limiter._check("ip", 10.0)
        assert (allowed, remaining) == (False, 0)
        assert retry_after == pytest.approx(50.0)

    def test_window_slides(self) -> None:
        limiter = self.build(requests=1, window=10.0)
        limiter._check("ip", 0.0)
        assert limiter._check("ip", 5.0)[0] is False
        assert limiter._check("ip", 11.0)[0] is True

    def test_clients_are_independent(self) -> None:
        limiter = self.build(requests=1)
        assert limiter._check("a", 0.0)[0] is True
        assert limiter._check("b", 0.0)[0] is True
        assert limiter._check("a", 0.1)[0] is False

    def test_disabled_when_budget_is_zero(self) -> None:
        assert self.build(requests=0).enabled is False

    def test_idle_clients_are_evicted(self) -> None:
        limiter = self.build(requests=5, window=1.0)
        for index in range(6_000):
            limiter._check(f"ip-{index}", float(index))
        limiter._check("ip-0", time.monotonic() + 100)
        assert len(limiter._hits) < 6_000


class TestMiddlewareStack:
    def test_request_context_sets_state_and_headers(self) -> None:
        seen: dict[str, Any] = {}

        async def app(scope, receive, send):  # noqa: ANN001
            seen["request_id"] = scope["state"]["request_id"]
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        sent: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            sent.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "headers": [(b"x-request-id", b"given-id")],
            "client": ("1.2.3.4", 1234),
        }
        import asyncio

        asyncio.run(RequestContextMiddleware(app)(scope, None, send))
        assert seen["request_id"] == "given-id"
        headers = {key.decode(): value.decode() for key, value in sent[0]["headers"]}
        assert headers["x-request-id"] == "given-id"
        assert float(headers["x-process-time"]) >= 0

    def test_request_context_generates_an_id(self) -> None:
        async def app(scope, receive, send):  # noqa: ANN001
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        captured: list[str] = []

        async def send(message):  # noqa: ANN001
            if message["type"] == "http.response.start":
                captured.append(
                    {key.decode(): value.decode() for key, value in message["headers"]}[
                        "x-request-id"
                    ]
                )

        scope: dict[str, Any] = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "client": None,
        }
        import asyncio

        asyncio.run(RequestContextMiddleware(app)(scope, None, send))
        assert len(captured[0]) == 32

    def test_non_http_scopes_pass_through(self) -> None:
        called = []

        async def app(scope, receive, send):  # noqa: ANN001
            called.append(scope["type"])

        async def send(message):  # noqa: ANN001
            return None

        import asyncio

        asyncio.run(RequestContextMiddleware(app)({"type": "lifespan"}, None, send))
        assert called == ["lifespan"]

    def test_rate_limiter_exempts_paths(self) -> None:
        async def app(scope, receive, send):  # noqa: ANN001
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        limiter = RateLimitMiddleware(app, requests=1, window=60.0)
        statuses: list[int] = []

        async def send(message):  # noqa: ANN001
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        import asyncio

        for path in ("/health", "/health", "/metrics"):
            asyncio.run(
                limiter({"type": "http", "path": path, "headers": [], "client": None}, None, send)
            )
        assert statuses == [200, 200, 200]
