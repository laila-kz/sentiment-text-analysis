"""Shared pytest fixtures.

The whole suite runs **without** ``torch``/``transformers``/``streamlit``:
inference is faked through :class:`FakePipeline`, so CI stays fast and tests
never hit the network.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

import sentiment
from config import Settings, get_settings
from observability import metrics
from sentiment import MODEL_REGISTRY, ModelHandle, ModelSpec, SentimentEngine

# ---------------------------------------------------------------------------
# Deterministic fake models
# ---------------------------------------------------------------------------

#: text (lowercased) -> logits, chosen so predictions are unambiguous.
LOGIT_TABLE: dict[str, list[float]] = {
    "i love this": [0.1, 3.4],
    "this is great": [0.2, 2.8],
    "terrible and useless": [3.6, 0.1],
    "absolutely awful": [4.0, 0.0],
    "meh": [0.4, 0.5],
    "good": [0.1, 1.9],
    "bad": [2.1, 0.1],
}

EMOTION_LOGITS: list[float] = [0.1, 0.2, 3.5, 0.4, 0.05, 0.3]
NEUTRAL_LOGITS: list[float] = [0.4, 0.5, 0.6]


class FakePipeline:
    """Callable stand-in for a Hugging Face ``TextClassificationPipeline``.

    Records every batch it receives so tests can assert on batching behaviour
    and exercise the engine's chunking logic.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.calls: list[list[str]] = []

    def logits_for(self, text: str) -> list[float]:
        if self.spec.key == "emotion":
            return list(EMOTION_LOGITS)
        if self.spec.key == "sentiment3":
            return list(NEUTRAL_LOGITS)
        return LOGIT_TABLE.get(text.strip().lower(), [0.5, 0.5])

    def __call__(
        self,
        texts: Sequence[str] | str,
        top_k: int | None = None,
        batch_size: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append(list(items))
        rows: list[list[dict[str, Any]]] = []
        for text in items:
            logits = self.logits_for(text)
            peak = max(logits)
            exps = [math.exp(value - peak) for value in logits]
            total = sum(exps)
            rows.append(
                [
                    {"label": label, "score": value / total}
                    for label, value in zip(self.spec.labels, exps, strict=True)
                ]
            )
        return rows

    @property
    def batch_sizes(self) -> list[int]:
        return [len(call) for call in self.calls]


class FakeLoader:
    """Model loader that hands out :class:`FakePipeline` handles and counts loads."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.handles: dict[str, ModelHandle] = {}
        self.loads: list[str] = []

    def __call__(self, spec: ModelSpec) -> ModelHandle:
        self.loads.append(spec.key)
        handle = self.handles.get(spec.key)
        if handle is None:
            handle = ModelHandle(
                spec=spec,
                pipeline_obj=FakePipeline(spec),
                device=self.device,
                max_length=128,
            )
            self.handles[spec.key] = handle
        return handle

    def pipeline(self, key: str = "sentiment") -> FakePipeline:
        return self.handles[key].pipeline  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every test a pristine settings/engine/metrics state."""
    for name in [key for key in os.environ if key.startswith("SENTIMENT_")]:
        monkeypatch.delenv(name, raising=False)
    get_settings(refresh=True)
    sentiment.set_engine(None)
    metrics.reset()
    yield
    sentiment.set_engine(None)
    metrics.reset()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def loader() -> FakeLoader:
    return FakeLoader()


@pytest.fixture
def engine(loader: FakeLoader) -> SentimentEngine:
    """Engine wired to fake models, installed as the process-wide engine."""
    built = SentimentEngine(loader=loader)
    sentiment.set_engine(built)
    return built


def make_app_client(
    engine: SentimentEngine,
    *,
    settings: Settings | None = None,
    rate_limit_requests: int = 0,
    rate_limit_window: float = 60.0,
    max_batch_items: int = 1000,
) -> Any:
    """Build a TestClient over a freshly-created app wired to ``engine``."""
    from fastapi.testclient import TestClient

    import api

    resolved = dataclasses.replace(
        settings or get_settings(),
        rate_limit_requests=rate_limit_requests,
        rate_limit_window=rate_limit_window,
        max_batch_items=max_batch_items,
        cors_origins=("*",),
    )
    application = api.create_app(resolved)
    application.dependency_overrides[api.get_engine_dep] = lambda: engine
    return TestClient(application, raise_server_exceptions=False)


@pytest.fixture
def client(engine: SentimentEngine) -> Any:
    """TestClient with rate limiting disabled (see ``test_api.py`` for 429s)."""
    return make_app_client(engine)


@pytest.fixture
def registry() -> dict[str, ModelSpec]:
    return MODEL_REGISTRY
