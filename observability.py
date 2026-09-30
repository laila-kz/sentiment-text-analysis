"""Cross-cutting concerns for the HTTP layer.

* :func:`configure_logging` – single-line JSON logs, easy to ship to any collector.
* :class:`RequestContextMiddleware` – request id, duration tracking, access logs.
* :class:`RateLimitMiddleware` – per-client sliding-window rate limiting.
* :class:`Metrics` – tiny in-process counters exposed on ``/metrics``.

Everything here is implemented as *pure ASGI* middleware (no Starlette version
coupling) and only uses the standard library.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "JsonLogFormatter",
    "Metrics",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    "configure_logging",
    "metrics",
]

_RESERVED_LOG_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonLogFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: MutableMapping[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleLogFormatter(logging.Formatter):
    """Human-friendly single line: ``LEVEL logger | key=value message``."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_FIELDS and not key.startswith("_")
        }
        rendered = " ".join(f"{key}={value}" for key, value in extras.items())
        base = f"{record.levelname:<7} {record.name:<28} | {record.getMessage()}"
        if rendered:
            base = f"{base} | {rendered}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str = "INFO", *, json_format: bool = True) -> None:
    """Install a single stream handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonLogFormatter() if json_format else ConsoleLogFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    # Uvicorn installs its own handlers; route them through ours too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class Metrics:
    """Minimal thread-safe counters (Prometheus text exposition compatible)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._labels: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._latency_ms: list[float] = []

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def observe_latency(self, milliseconds: float) -> None:
        with self._lock:
            self._latency_ms.append(float(milliseconds))
            if len(self._latency_ms) > 10_000:
                del self._latency_ms[:-10_000]

    def increment_labeled(self, name: str, labels: dict[str, str], value: float = 1.0) -> None:
        key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
        with self._lock:
            self._labels[key] = self._labels.get(key, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            data = dict(self._counters)
            latencies = list(self._latency_ms)
        if latencies:
            data["latency_ms_sum"] = round(sum(latencies), 3)
            data["latency_ms_max"] = round(max(latencies), 3)
            data["latency_ms_avg"] = round(sum(latencies) / len(latencies), 3)
        for (name, labels), value in sorted(self._labels.items()):
            rendered = ",".join(f'{key}="{value_}"' for key, value_ in labels)
            data[f"{name}{{{rendered}}}"] = value
        return data

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._labels.clear()
            self._latency_ms.clear()

    def render(self) -> str:
        """Render the current counters in Prometheus text exposition format."""
        lines: list[str] = []
        for name, value in sorted(self.snapshot().items()):
            base = name.split("{", 1)[0]
            lines.append(f"# TYPE {base} counter")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _request_id(headers: Sequence[tuple[bytes, bytes]]) -> str | None:
    for key, value in headers:
        if key.decode("latin-1").lower() == "x-request-id":
            candidate = value.decode("latin-1").strip()
            if candidate:
                return candidate[:128]
    return None


def _client_host(scope: MutableMapping[str, Any], headers: Sequence[tuple[bytes, bytes]]) -> str:
    for key, value in headers:
        if key.decode("latin-1").lower() == "x-forwarded-for":
            forwarded = value.decode("latin-1").split(",")[0].strip()
            if forwarded:
                return forwarded
    client = scope.get("client")
    if client:
        return str(client[0])
    return "unknown"


def _append_header(message: MutableMapping[str, Any], name: str, value: str) -> None:
    raw_name = name.lower().encode("latin-1")
    headers: list[tuple[bytes, bytes]] = list(message.get("headers") or [])
    headers = [(k, v) for k, v in headers if k.lower() != raw_name]
    headers.append((raw_name, value.encode("latin-1")))
    message["headers"] = headers


class RequestContextMiddleware:
    """Attach a request id, measure duration and emit an access log line."""

    def __init__(self, app: Any, *, slow_request_ms: float = 1_000.0) -> None:
        self.app = app
        self.slow_request_ms = slow_request_ms
        self.logger = logging.getLogger("api.access")

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers") or [])
        request_id = _request_id(headers) or uuid.uuid4().hex
        client = _client_host(scope, headers)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["client_host"] = client
        started = time.perf_counter()
        status_holder: dict[str, int] = {"status": 0}

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                status_holder["status"] = int(message.get("status", 0))
                _append_header(message, "x-request-id", request_id)
                _append_header(message, "x-process-time", f"{elapsed_ms:.2f}")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            self.logger.exception(
                "request failed",
                extra={"request_id": request_id, "path": scope.get("path"), "client": client},
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            status = status_holder["status"]
            metrics.increment("http_requests_total")
            if status >= 500:
                metrics.increment("http_server_errors_total")
            elif status >= 400:
                metrics.increment("http_client_errors_total")
            metrics.observe_latency(elapsed_ms)
            metrics.increment_labeled(
                "http_requests_by_endpoint",
                {"path": _normalise_path(scope.get("path", "")), "status": str(status)},
            )
            level = logging.WARNING if status >= 400 else logging.INFO
            self.logger.log(
                level,
                "%s %s -> %s",
                scope.get("method", "?"),
                scope.get("path", "?"),
                status,
                extra={
                    "request_id": request_id,
                    "status": status,
                    "duration_ms": round(elapsed_ms, 2),
                    "client": client,
                    "slow": elapsed_ms > self.slow_request_ms,
                },
            )


def _normalise_path(path: str) -> str:
    """Collapse ids in paths so metric cardinality stays bounded."""
    if not path:
        return "/"
    parts = path.split("/")
    normalised = [
        f"{{{part}}}" if part.isdigit() or (len(part) > 24 and "-" in part) else part
        for part in parts
    ]
    return "/".join(normalised) or "/"


class RateLimitMiddleware:
    """Per-client sliding-window rate limiter.

    In-memory by design (single-process service). ``exempt_paths`` keeps health
    checks and documentation reachable when a client exhausts its budget.
    """

    def __init__(
        self,
        app: Any,
        *,
        requests: int = 60,
        window: float = 60.0,
        exempt_paths: Iterable[str] = ("/health", "/metrics", "/docs", "/openapi.json", "/redoc"),
        enabled: bool = True,
    ) -> None:
        self.app = app
        self.requests = max(0, int(requests))
        self.window = float(window)
        self.exempt_paths = frozenset(exempt_paths)
        self.enabled = enabled and self.requests > 0
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger("api.rate_limit")

    def _check(self, key: str, now: float) -> tuple[bool, int, float]:
        """Return ``(allowed, remaining, retry_after_seconds)``."""
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.requests:
                retry_after = max(0.0, bucket[0] + self.window - now)
                return False, 0, retry_after
            bucket.append(now)
            remaining = self.requests - len(bucket)
            if len(self._hits) > 5_000:  # opportunistic cleanup of idle clients
                self._hits = {
                    client: hits
                    for client, hits in self._hits.items()
                    if hits and hits[-1] > cutoff
                }
            return True, remaining, 0.0

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if not self.enabled or scope["type"] != "http" or scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers") or [])
        client = _client_host(scope, headers)
        allowed, remaining, retry_after = self._check(client, time.monotonic())
        if allowed:
            await self.app(scope, receive, send)
            return

        metrics.increment("rate_limit_rejections_total")
        self.logger.warning(
            "rate limit exceeded", extra={"client": client, "path": scope.get("path")}
        )
        state = scope.get("state") or {}
        body = json.dumps(
            {
                "detail": "Rate limit exceeded. Retry later.",
                "request_id": state.get("request_id"),
            }
        ).encode("utf-8")
        retry_header = str(max(1, int(retry_after + 0.999)))
        headers_out: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("latin-1")),
            (b"retry-after", retry_header.encode("latin-1")),
            (b"x-ratelimit-limit", str(self.requests).encode("latin-1")),
            (b"x-ratelimit-remaining", b"0"),
        ]
        if state.get("request_id"):
            headers_out.append((b"x-request-id", str(state["request_id"]).encode("latin-1")))
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": headers_out,
            }
        )
        await send({"type": "http.response.body", "body": body})
