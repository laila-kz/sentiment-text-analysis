"""Core inference engine: model registry, batched inference and calibration.

Design notes
------------
* Every model is described by a :class:`ModelSpec` in :data:`MODEL_REGISTRY`,
  which makes switching between 2-class sentiment and 6-class emotion models a
  pure runtime concern (no code changes, no re-download of shared vocabulary).
* Models are wrapped by :class:`ModelHandle`, which hides the difference between
  the fast torch path (real logits for calibration) and the plain pipeline path.
* Inference is *vectorised*: a batch of texts is tokenised once, padded once and
  pushed through the network in a single forward pass, on CUDA when available.
* Every prediction is returned as an :class:`AnalysisResult` containing the full
  label distribution, uncertainty metrics (entropy, margin) and, optionally, the
  raw logits together with the temperature used to calibrate them.

``torch``/``transformers`` are imported lazily so that the module (and the test
suite) can be imported without the heavy dependencies installed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from config import Settings, get_settings
from utils import (
    clamp,
    margin_of_confidence,
    normalized_entropy,
    rejection_reason,
    segment_text,
    softmax,
    top_k,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL",
    "AnalysisResult",
    "BatchResult",
    "BenchmarkResult",
    "EmptyTextError",
    "LabelScore",
    "ModelHandle",
    "ModelKind",
    "ModelLoadError",
    "ModelSpec",
    "MODEL_REGISTRY",
    "SegmentPrediction",
    "SentimentEngine",
    "SentimentError",
    "UncertaintyReport",
    "UnknownModelError",
    "analyse",
    "analyse_batch",
    "analyse_segments",
    "available_models",
    "benchmark",
    "clear_cache",
    "cuda_available",
    "get_classifier",
    "get_engine",
    "get_model_spec",
    "resolve_device",
    "set_engine",
    "summarise",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SentimentError(RuntimeError):
    """Base class for all errors raised by this package."""


class UnknownModelError(SentimentError, KeyError):
    """Raised when a model key is not present in :data:`MODEL_REGISTRY`."""

    def __init__(self, key: str, available: Iterable[str] = ()) -> None:
        options = ", ".join(sorted(available)) or "none"
        super().__init__(f"Unknown model {key!r}. Available models: {options}.")
        self.key = key


class ModelLoadError(SentimentError):
    """Raised when a model cannot be downloaded or instantiated."""


class EmptyTextError(SentimentError, ValueError):
    """Raised when the input text carries no analysable word characters."""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelKind(StrEnum):
    """The label space a model predicts."""

    SENTIMENT = "sentiment"
    EMOTION = "emotion"


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a supported fine-tuned checkpoint."""

    key: str
    hf_id: str
    kind: ModelKind
    labels: tuple[str, ...]
    description: str
    language: str = "en"
    default: bool = False

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model_id": self.hf_id,
            "kind": str(self.kind),
            "labels": list(self.labels),
            "num_labels": self.num_labels,
            "language": self.language,
            "default": self.default,
            "description": self.description,
        }


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="sentiment",
            hf_id="distilbert-base-uncased-finetuned-sst-2-english",
            kind=ModelKind.SENTIMENT,
            labels=("negative", "positive"),
            description="Binary sentiment classifier fine-tuned on SST-2 (default).",
            default=True,
        ),
        ModelSpec(
            key="emotion",
            hf_id="bhadresh-ps/distilbert-base-uncased-emotion",
            kind=ModelKind.EMOTION,
            labels=("sadness", "joy", "love", "anger", "fear", "surprise"),
            description="Fine-grained 6-way emotion classification (Dair-UI style).",
        ),
        ModelSpec(
            key="sentiment3",
            hf_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
            kind=ModelKind.SENTIMENT,
            labels=("negative", "neutral", "positive"),
            description="3-class sentiment tuned on tweets, including a neutral class.",
        ),
    )
}

DEFAULT_MODEL: str = next(key for key, spec in MODEL_REGISTRY.items() if spec.default)


def available_models() -> tuple[ModelSpec, ...]:
    """All registered model specifications, default first."""
    return tuple(sorted(MODEL_REGISTRY.values(), key=lambda spec: (not spec.default, spec.key)))


def get_model_spec(name: str | ModelSpec | None = None) -> ModelSpec:
    """Resolve a model key (or spec). ``None`` selects the default model."""
    if isinstance(name, ModelSpec):
        return name
    key = (name or DEFAULT_MODEL).strip()
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        lowered = {k.lower(): v for k, v in MODEL_REGISTRY.items()}.get(key.lower())
        if lowered is None:
            raise UnknownModelError(key, MODEL_REGISTRY.keys())
        return lowered
    return spec


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """``True`` when a CUDA device can actually be used by torch."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver level failures
        logger.warning("CUDA probe failed; falling back to CPU.", exc_info=True)
        return False


def resolve_device(preference: str | None = None) -> str:
    """Translate a device preference into ``"cuda"``/``"cpu"``.

    ``"auto"`` (the default) selects CUDA when available and CPU otherwise;
    ``"cpu"``/``"cuda"`` are honoured verbatim so operators can pin placement.
    """
    want = (preference or "auto").strip().lower()
    if want in {"cpu", "cuda"}:
        if want == "cuda" and not cuda_available():
            logger.warning("CUDA requested but unavailable; using CPU.")
            return "cpu"
        return want
    return "cuda" if cuda_available() else "cpu"


def _torch_dtype(device: str, fp16: bool) -> Any | None:
    if device != "cuda" or not fp16:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch always present on cuda
        return None
    return torch.float16


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelScore:
    """One entry of the (calibrated) label distribution."""

    label: str
    probability: float
    logit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label, "probability": self.probability}
        if self.logit is not None:
            payload["logit"] = self.logit
        return payload


@dataclass(frozen=True)
class UncertaintyReport:
    """How unsure the model was about a prediction."""

    entropy_bits: float
    normalized_entropy: float
    margin: float
    is_uncertain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy_bits": self.entropy_bits,
            "normalized_entropy": self.normalized_entropy,
            "margin": self.margin,
            "is_uncertain": self.is_uncertain,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Full analysis payload for a single text.

    ``status`` is ``"ok"`` for analysed texts and ``"skipped"`` when the input
    carried no analysable content (empty, whitespace-only or emoji-only); the
    remaining fields are then empty rather than fabricated.
    """

    text: str
    status: str = "skipped"
    reason: str | None = None
    model: str = DEFAULT_MODEL
    model_id: str = ""
    device: str = "cpu"
    label: str | None = None
    confidence: float | None = None
    distribution: dict[str, float] = field(default_factory=dict)
    ranked: tuple[LabelScore, ...] = ()
    uncertainty: UncertaintyReport | None = None
    logits: tuple[float, ...] | None = None
    temperature: float = 1.0
    calibration: str = "temperature_scaling"
    processing_ms: float = 0.0
    warnings: tuple[str, ...] = ()

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self, *, include_logits: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "status": self.status,
            "reason": self.reason,
            "model": self.model,
            "model_id": self.model_id,
            "device": self.device,
            "label": self.label,
            "confidence": self.confidence,
            "distribution": dict(self.distribution),
            "ranked": [item.to_dict() for item in self.ranked],
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "temperature": self.temperature,
            "calibration": self.calibration,
            "processing_ms": round(self.processing_ms, 3),
            "warnings": list(self.warnings),
        }
        if include_logits:
            payload["logits"] = list(self.logits) if self.logits is not None else None
        return payload

    @classmethod
    def skipped(cls, text: str, reason: str, **kwargs: Any) -> AnalysisResult:
        return cls(text=text, status="skipped", reason=reason, **kwargs)


@dataclass(frozen=True)
class SegmentPrediction:
    """Sentence-level prediction used for in-text highlighting."""

    text: str
    label: str | None
    score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "label": self.label, "score": self.score}


@dataclass(frozen=True)
class BatchResult:
    """Aggregate statistics for a batch run."""

    items: int
    analysed: int
    skipped: int
    label_counts: dict[str, int]
    mean_confidence: float
    mean_normalized_entropy: float
    total_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "analysed": self.analysed,
            "skipped": self.skipped,
            "label_counts": dict(self.label_counts),
            "mean_confidence": self.mean_confidence,
            "mean_normalized_entropy": self.mean_normalized_entropy,
            "total_ms": round(self.total_ms, 3),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Throughput measurement for a given text volume."""

    texts: int
    batch_size: int
    device: str
    total_seconds: float
    texts_per_second: float
    latency_ms_p50: float
    latency_ms_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "texts": self.texts,
            "batch_size": self.batch_size,
            "device": self.device,
            "total_seconds": round(self.total_seconds, 4),
            "texts_per_second": round(self.texts_per_second, 2),
            "latency_ms_p50": round(self.latency_ms_p50, 3),
            "latency_ms_p95": round(self.latency_ms_p95, 3),
        }


def summarise(results: Sequence[AnalysisResult], total_ms: float = 0.0) -> BatchResult:
    """Build a :class:`BatchResult` from individual results."""
    counts: dict[str, int] = {}
    confidences: list[float] = []
    entropies: list[float] = []
    skipped = 0
    for result in results:
        if not result.is_ok or result.label is None:
            skipped += 1
            continue
        counts[result.label] = counts.get(result.label, 0) + 1
        if result.confidence is not None:
            confidences.append(result.confidence)
        if result.uncertainty is not None:
            entropies.append(result.uncertainty.normalized_entropy)
    return BatchResult(
        items=len(results),
        analysed=len(results) - skipped,
        skipped=skipped,
        label_counts=dict(sorted(counts.items())),
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        mean_normalized_entropy=(sum(entropies) / len(entropies)) if entropies else 0.0,
        total_ms=total_ms or sum(result.processing_ms for result in results),
    )


# ---------------------------------------------------------------------------
# Model handle
# ---------------------------------------------------------------------------


class ModelHandle:
    """A loaded pipeline plus the metadata needed to post-process outputs."""

    def __init__(
        self,
        spec: ModelSpec,
        pipeline_obj: Any,
        *,
        tokenizer: Any = None,
        network: Any = None,
        device: str = "cpu",
        max_length: int = 256,
        temperature: float = 1.0,
        fp16: bool = False,
    ) -> None:
        self.spec = spec
        self.pipeline = pipeline_obj
        self.tokenizer = (
            tokenizer if tokenizer is not None else getattr(pipeline_obj, "tokenizer", None)
        )
        self.network = network if network is not None else getattr(pipeline_obj, "model", None)
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.fp16 = fp16
        self.labels = self._resolve_labels()
        self.loaded_at = time.time()

    # -- labels ---------------------------------------------------------
    def _resolve_labels(self) -> tuple[str, ...]:
        """Trust the config when it matches the head, else derive from the head."""
        config = getattr(self.network, "config", None)
        id2label = getattr(config, "id2label", None) if config is not None else None
        if isinstance(id2label, dict) and id2label:
            try:
                width = int(getattr(config, "num_labels", len(id2label)))
                derived = tuple(
                    str(id2label.get(index, id2label.get(str(index), f"label_{index}")))
                    .strip()
                    .lower()
                    for index in range(width)
                )
            except (TypeError, ValueError):  # pragma: no cover - odd configs
                derived = ()
            if derived and set(derived) == set(self.spec.labels):
                return derived
        return self.spec.labels

    # -- inference ------------------------------------------------------
    def predict_logits(self, texts: Sequence[str]) -> list[list[float]]:
        """Return raw logits for a batch of texts (one row per text)."""
        if not texts:
            return []
        if self.tokenizer is not None and self.network is not None:
            return self._predict_logits_torch(texts, self.tokenizer, self.network)
        return self._predict_logits_pipeline(texts)

    def _predict_logits_torch(
        self, texts: Sequence[str], tokenizer: Any, network: Any
    ) -> list[list[float]]:
        import torch

        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = network(**encoded)
        logits = output.logits
        return logits.detach().to("cpu", dtype=torch.float32).tolist()

    def _predict_logits_pipeline(self, texts: Sequence[str]) -> list[list[float]]:
        """Fallback path: read the scores straight off the pipeline.

        Probabilities are converted back to logits (``log p``) so that the rest
        of the pipeline can apply temperature scaling uniformly.
        """
        raw = self.pipeline(list(texts), top_k=None, batch_size=len(texts))
        rows: list[list[float]] = []
        for scores in raw:
            by_label = {str(entry["label"]).lower(): float(entry["score"]) for entry in scores}
            rows.append([math.log(max(by_label.get(label, 0.0), 1e-12)) for label in self.labels])
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "loaded": True,
            "labels_resolved": list(self.labels),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

HandleLoader = Callable[[ModelSpec], ModelHandle]


def _default_loader(spec: ModelSpec) -> ModelHandle:
    """Load a real Hugging Face pipeline for ``spec``."""
    settings = get_settings()
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ModelLoadError(
            "transformers is required. Install it with: pip install -r requirements.txt"
        ) from exc

    device = resolve_device(settings.device)
    logger.info(
        "Loading model '%s' (%s) on %s",
        spec.key,
        spec.hf_id,
        device,
    )
    kwargs: dict[str, Any] = {
        "model": spec.hf_id,
        "tokenizer": spec.hf_id,
        "task": "text-classification",
        "cache_dir": str(settings.cache_dir),
        "device": -1 if device == "cpu" else 0,
    }
    dtype = _torch_dtype(device, settings.fp16)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    try:
        pipe = hf_pipeline(**kwargs)
    except Exception as exc:  # pragma: no cover - network/dependency dependent
        raise ModelLoadError(f"Failed to load model {spec.hf_id!r}: {exc}") from exc
    return ModelHandle(
        spec=spec,
        pipeline_obj=pipe,
        device=device,
        max_length=settings.max_length,
        temperature=settings.temperature,
        fp16=settings.fp16,
    )


class SentimentEngine:
    """Holds loaded models and performs batched, calibrated inference.

    The engine is safe to share between threads: model loading and forward
    passes are serialised with a re-entrant lock, which also keeps peak GPU
    memory predictable under concurrent API traffic.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        loader: HandleLoader | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._loader: HandleLoader = loader or _default_loader
        self._handles: dict[str, ModelHandle] = {}
        self._lock = threading.RLock()

    # -- model management ---------------------------------------------
    def handle(self, model: str | ModelSpec | None = None) -> ModelHandle:
        """Return (loading on first use) the handle for ``model``."""
        spec = get_model_spec(model)
        with self._lock:
            handle = self._handles.get(spec.key)
            if handle is None:
                handle = self._loader(spec)
                self._handles[spec.key] = handle
                logger.info("Model '%s' ready (%s)", spec.key, handle.device)
            return handle

    def warmup(self, model: str | ModelSpec | None = None) -> ModelHandle:
        """Eagerly load a model (used at API startup and in benchmarks)."""
        return self.handle(model)

    def warmup_all(self) -> dict[str, ModelHandle]:
        """Load every model in the registry."""
        return {spec.key: self.handle(spec) for spec in MODEL_REGISTRY.values()}

    def clear(self) -> None:
        """Drop cached handles (frees GPU memory)."""
        with self._lock:
            self._handles.clear()

    @property
    def loaded_models(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._handles))

    def info(self) -> dict[str, Any]:
        with self._lock:
            keys = set(self._handles)
        models = []
        for spec in available_models():
            payload = spec.to_dict()
            payload["loaded"] = spec.key in keys
            if spec.key in keys:
                handle = self._handles[spec.key]
                payload["device"] = handle.device
                payload["loaded_at"] = handle.loaded_at
            models.append(payload)
        return {
            "models": models,
            "default_model": DEFAULT_MODEL,
            "device_preference": self.settings.device,
            "resolved_device": resolve_device(self.settings.device),
            "cuda_available": cuda_available(),
            "loaded_models": sorted(keys),
        }

    # -- inference ------------------------------------------------------
    def analyse(
        self,
        text: str,
        *,
        model: str | ModelSpec | None = None,
        temperature: float | None = None,
        include_logits: bool = False,
    ) -> AnalysisResult:
        """Analyse a single text and return the full result payload."""
        return self.analyse_batch(
            [text],
            model=model,
            temperature=temperature,
            include_logits=include_logits,
        )[0]

    def analyse_batch(
        self,
        texts: Sequence[str],
        *,
        model: str | ModelSpec | None = None,
        batch_size: int | None = None,
        temperature: float | None = None,
        include_logits: bool = False,
        strict: bool = False,
    ) -> list[AnalysisResult]:
        """Analyse many texts with vectorised, chunked inference.

        Texts that cannot be analysed (empty / whitespace / emoji-only) are
        reported as ``skipped`` results instead of failing the whole batch,
        unless ``strict=True`` in which case :class:`EmptyTextError` is raised.
        """
        if isinstance(texts, str):  # guard against a common caller mistake
            raise TypeError("analyse_batch expects a sequence of strings, not a single string")

        spec = get_model_spec(model)
        limit = self.settings.max_text_length
        temp = float(temperature) if temperature is not None else self.settings.temperature
        size = max(1, int(batch_size or self.settings.batch_size))
        started = time.perf_counter()

        results: list[AnalysisResult | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        warnings: dict[int, list[str]] = {}

        for index, raw in enumerate(texts):
            text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
            notes: list[str] = []
            if len(text) > limit:
                text = text[:limit]
                notes.append(f"truncated_to_{limit}_chars")
            reason = rejection_reason(text)
            if reason is not None:
                if strict:
                    raise EmptyTextError(f"Input at index {index} is not analysable ({reason}).")
                results[index] = AnalysisResult.skipped(
                    text=text,
                    reason=reason,
                    model=spec.key,
                    model_id=spec.hf_id,
                    temperature=temp,
                    warnings=tuple(notes),
                )
                continue
            pending.append((index, text))
            if notes:
                warnings[index] = notes

        if pending:
            with self._lock:
                handle = self.handle(spec)
                for start in range(0, len(pending), size):
                    chunk = pending[start : start + size]
                    rows = handle.predict_logits([text for _, text in chunk])
                    if len(rows) != len(chunk):  # pragma: no cover - defensive
                        raise SentimentError(
                            f"Model returned {len(rows)} rows for {len(chunk)} inputs."
                        )
                    for (index, text), logits in zip(chunk, rows, strict=True):
                        results[index] = self._build_result(
                            text=text,
                            logits=logits,
                            handle=handle,
                            temperature=temp,
                            include_logits=include_logits,
                            warnings=tuple(warnings.get(index, ())),
                        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if pending:
            # Attribute a fair share of the wall-clock time to each analysed text
            # so that per-item latencies still add up to the measured total.
            share = elapsed_ms / len(pending)
            results = [
                replace(result, processing_ms=share)
                if result is not None and result.is_ok
                else result
                for result in results
            ]

        return [result for result in results if result is not None]

    def analyse_segments(
        self,
        text: str,
        *,
        model: str | ModelSpec | None = None,
        max_chars: int = 160,
        include_logits: bool = False,
    ) -> list[SegmentPrediction]:
        """Analyse sentence by sentence (drives in-text highlighting)."""
        chunks = segment_text(text, max_chars=max_chars)
        if not chunks:
            return []
        results = self.analyse_batch(chunks, model=model, include_logits=include_logits)
        return [
            SegmentPrediction(text=chunk, label=result.label, score=result.confidence)
            for chunk, result in zip(chunks, results, strict=True)
        ]

    def benchmark(
        self,
        texts: Sequence[str] | None = None,
        *,
        model: str | ModelSpec | None = None,
        batch_size: int | None = None,
        repeats: int = 1,
    ) -> BenchmarkResult:
        """Measure throughput and latency percentiles for a text volume."""
        corpus = (
            list(texts) if texts else [f"Sample review sentence number {i}." for i in range(64)]
        )
        size = max(1, int(batch_size or self.settings.batch_size))
        self.handle(model)  # exclude cold start from the measurement
        latencies: list[float] = []
        started = time.perf_counter()
        for _ in range(max(1, repeats)):
            for start in range(0, len(corpus), size):
                chunk_start = time.perf_counter()
                self.analyse_batch(corpus[start : start + size], model=model, batch_size=size)
                latencies.append((time.perf_counter() - chunk_start) * 1000.0)
        total = time.perf_counter() - started
        ordered = sorted(latencies)
        return BenchmarkResult(
            texts=len(corpus) * max(1, repeats),
            batch_size=size,
            device=resolve_device(self.settings.device),
            total_seconds=total,
            texts_per_second=(len(corpus) * max(1, repeats)) / total if total > 0 else 0.0,
            latency_ms_p50=_percentile(ordered, 0.50),
            latency_ms_p95=_percentile(ordered, 0.95),
        )

    # -- internals ------------------------------------------------------
    def _build_result(
        self,
        *,
        text: str,
        logits: Sequence[float],
        handle: ModelHandle,
        temperature: float,
        include_logits: bool,
        warnings: tuple[str, ...] = (),
    ) -> AnalysisResult:
        labels = handle.labels
        width = min(len(labels), len(logits))
        if width == 0:
            raise SentimentError("Model returned no logits.")
        if width != len(labels):
            warnings = (*warnings, f"logit_width_mismatch_{width}_of_{len(labels)}")
        used_logits = [float(value) for value in logits[:width]]
        used_labels = list(labels[:width])

        probabilities = softmax(used_logits, temperature=temperature)
        ranked_idx = top_k(probabilities, k=min(3, width))
        best_idx, best_prob = ranked_idx[0]
        distribution = {
            label: round(probability, 6)
            for label, probability in sorted(
                zip(used_labels, probabilities, strict=True), key=lambda p: -p[1]
            )
        }
        uncertainty = UncertaintyReport(
            entropy_bits=round(normalized_entropy(probabilities) * math.log2(width), 6),
            normalized_entropy=round(normalized_entropy(probabilities), 6),
            margin=round(margin_of_confidence(probabilities), 6),
            is_uncertain=normalized_entropy(probabilities) > self.settings.uncertainty_threshold,
        )
        return AnalysisResult(
            text=text,
            status="ok",
            reason=None,
            model=handle.spec.key,
            model_id=handle.spec.hf_id,
            device=handle.device,
            label=used_labels[best_idx],
            confidence=round(clamp(best_prob), 6),
            distribution=distribution,
            ranked=tuple(
                LabelScore(
                    label=used_labels[index],
                    probability=round(probability, 6),
                    logit=round(used_logits[index], 6),
                )
                for index, probability in ranked_idx
            ),
            uncertainty=uncertainty,
            logits=tuple(round(value, 6) for value in used_logits) if include_logits else None,
            temperature=temperature,
            warnings=warnings,
        )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


# ---------------------------------------------------------------------------
# Module level convenience API
# ---------------------------------------------------------------------------

_ENGINE: SentimentEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> SentimentEngine:
    """Return the process-wide engine (models are cached inside it)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = SentimentEngine()
        return _ENGINE


def set_engine(engine: SentimentEngine | None) -> None:
    """Install (or clear) the process-wide engine.

    Dependency-injection seam for the API layer and for tests that must not
    download a real checkpoint.
    """
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = engine


def clear_cache() -> None:
    """Unload every cached model (frees memory, forces a reload next call)."""
    engine = _ENGINE
    if engine is not None:
        engine.clear()


def get_classifier(model: str | ModelSpec | None = None) -> Callable[..., Any]:
    """Return the raw Hugging Face pipeline for ``model`` (legacy helper)."""
    return get_engine().handle(model).pipeline


def analyse(
    text: str,
    *,
    model: str | ModelSpec | None = None,
    temperature: float | None = None,
    include_logits: bool = False,
) -> AnalysisResult:
    """Analyse one text with the default engine."""
    return get_engine().analyse(
        text, model=model, temperature=temperature, include_logits=include_logits
    )


def analyse_batch(
    texts: Sequence[str],
    *,
    model: str | ModelSpec | None = None,
    batch_size: int | None = None,
    temperature: float | None = None,
    include_logits: bool = False,
    strict: bool = False,
) -> list[AnalysisResult]:
    """Analyse many texts with the default engine."""
    return get_engine().analyse_batch(
        texts,
        model=model,
        batch_size=batch_size,
        temperature=temperature,
        include_logits=include_logits,
        strict=strict,
    )


def analyse_segments(
    text: str, *, model: str | ModelSpec | None = None, max_chars: int = 160
) -> list[SegmentPrediction]:
    """Analyse sentence by sentence with the default engine."""
    return get_engine().analyse_segments(text, model=model, max_chars=max_chars)


def benchmark(
    texts: Sequence[str] | None = None,
    *,
    model: str | ModelSpec | None = None,
    batch_size: int | None = None,
    repeats: int = 1,
) -> BenchmarkResult:
    """Measure throughput of the default engine."""
    return get_engine().benchmark(texts, model=model, batch_size=batch_size, repeats=repeats)
