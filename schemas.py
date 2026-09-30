"""Pydantic request/response models: the public contract of the HTTP API.

Validation is deliberately strict (unknown fields rejected, bounds enforced,
custom error messages) so that malformed payloads fail fast at the edge instead
of reaching the model.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentiment import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    AnalysisResult,
    BatchResult,
    get_model_spec,
)

__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "BatchSummaryResponse",
    "ErrorResponse",
    "HealthResponse",
    "LabelScoreResponse",
    "MemoryInfo",
    "ModelInfoResponse",
    "ModelsResponse",
    "TextAnalysisResponse",
    "UncertaintyResponse",
]

MODEL_KEYS = tuple(MODEL_REGISTRY)
ModelKey = Annotated[str, Field(pattern=f"^({'|'.join(MODEL_KEYS)})$")]

_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=False)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Payload for ``POST /v1/analyze``.

    Provide exactly one of ``text`` (single item) or ``texts`` (batch). All
    other fields are optional knobs for the inference pipeline.
    """

    model_config = _STRICT

    text: Annotated[str, Field(min_length=1, max_length=50_000)] | None = Field(
        default=None,
        description="A single text to analyse.",
        examples=["The battery lasts all day and the screen is gorgeous."],
    )
    texts: Annotated[list[str], Field(min_length=1, max_length=1_000)] | None = Field(
        default=None,
        description="A batch of texts to analyse in one vectorised pass.",
        examples=[["Great product", "Terrible support experience"]],
    )
    model: ModelKey = Field(
        default=DEFAULT_MODEL,
        description="Registry key of the model to use.",
    )
    temperature: Annotated[float, Field(gt=0.0, le=10.0)] = Field(
        default=1.0,
        description="Softmax temperature. >1 softens (less confident) probabilities.",
    )
    batch_size: Annotated[int, Field(ge=1, le=256)] = Field(
        default=16, description="Texts per forward pass."
    )
    include_logits: bool = Field(
        default=False, description="Return raw logits for calibration analysis."
    )
    include_segments: bool = Field(
        default=False,
        description="Also return per-sentence predictions (single-text mode only).",
    )

    @field_validator("texts")
    @classmethod
    def _reject_blank_items(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"texts[{index}] must be a string")
        return value

    @model_validator(mode="after")
    def _exactly_one_input(self) -> AnalyzeRequest:
        if self.text is None and self.texts is None:
            raise ValueError("provide either 'text' or 'texts'")
        if self.text is not None and self.texts is not None:
            raise ValueError("provide only one of 'text' or 'texts', not both")
        if self.include_segments and self.texts is not None:
            raise ValueError("include_segments is only supported with a single 'text'")
        return self

    @property
    def inputs(self) -> list[str]:
        """The texts to analyse, normalised to a list."""
        return [self.text] if self.text is not None else list(self.texts or [])

    @property
    def is_batch(self) -> bool:
        return self.texts is not None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class LabelScoreResponse(BaseModel):
    """One entry of the ranked label distribution."""

    model_config = ConfigDict(extra="forbid")

    label: str
    probability: float = Field(ge=0.0, le=1.0)
    logit: float | None = None


class UncertaintyResponse(BaseModel):
    """Uncertainty / calibration metrics for one prediction."""

    model_config = ConfigDict(extra="forbid")

    entropy_bits: float = Field(ge=0.0, description="Shannon entropy in bits (0 = certain).")
    normalized_entropy: float = Field(ge=0.0, le=1.0, description="1 = uniform over labels.")
    margin: float = Field(ge=0.0, le=1.0, description="Top-1 minus top-2 probability.")
    is_uncertain: bool = Field(description="True when normalized_entropy exceeds the threshold.")


class SegmentResponse(BaseModel):
    """Sentence-level prediction used for in-text highlighting."""

    model_config = ConfigDict(extra="forbid")

    text: str
    label: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class TextAnalysisResponse(BaseModel):
    """Analysis of one text."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Position in the submitted batch.")
    text: str
    status: Literal["ok", "skipped"] = "ok"
    reason: str | None = Field(default=None, description="Why the text was skipped.")
    model: str
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(
        default_factory=dict, description="Full calibrated label distribution."
    )
    ranked: list[LabelScoreResponse] = Field(default_factory=list)
    uncertainty: UncertaintyResponse | None = None
    logits: list[float] | None = None
    temperature: float = 1.0
    processing_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    segments: list[SegmentResponse] | None = None

    @classmethod
    def from_result(cls, result: AnalysisResult, index: int) -> TextAnalysisResponse:
        payload: dict[str, Any] = {
            "index": index,
            **result.to_dict(include_logits=True),
        }
        payload.pop("model_id", None)
        payload.pop("device", None)
        payload.pop("calibration", None)
        return cls.model_validate(payload)


class BatchSummaryResponse(BaseModel):
    """Aggregate statistics for a batch run."""

    model_config = ConfigDict(extra="forbid")

    items: int = Field(ge=0)
    analysed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    label_counts: dict[str, int] = Field(default_factory=dict)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    mean_normalized_entropy: float = Field(ge=0.0, le=1.0)
    total_ms: float = 0.0

    @classmethod
    def from_result(cls, summary: BatchResult) -> BatchSummaryResponse:
        return cls.model_validate(summary.to_dict())


class AnalyzeResponse(BaseModel):
    """Response body for ``POST /v1/analyze``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(description="Correlation id, also returned as X-Request-ID.")
    model: str
    model_id: str = Field(description="Hugging Face checkpoint id.")
    device: str = Field(description="Execution device actually used.")
    batch: bool = Field(description="True when the request contained more than one text.")
    results: list[TextAnalysisResponse]
    summary: BatchSummaryResponse
    took_ms: float = 0.0
    version: str = ""


class MemoryInfo(BaseModel):
    """Process/system memory snapshot reported by ``/health``."""

    model_config = ConfigDict(extra="forbid")

    rss_mb: float = Field(ge=0.0, description="Resident set size of the API process.")
    available_mb: float | None = Field(default=None, ge=0.0, description="System RAM still free.")
    total_mb: float | None = Field(default=None, ge=0.0, description="System RAM installed.")
    percent_used: float | None = Field(default=None, ge=0.0, le=100.0)


class ModelInfoResponse(BaseModel):
    """One entry of the model registry as exposed over HTTP."""

    model_config = ConfigDict(extra="forbid")

    key: str
    model_id: str
    kind: str
    labels: list[str]
    num_labels: int
    language: str = "en"
    default: bool = False
    description: str = ""
    loaded: bool = Field(default=False, description="True when weights are resident in memory.")
    device: str | None = None
    loaded_at: float | None = None


class ModelsResponse(BaseModel):
    """Response body for ``GET /v1/models``."""

    model_config = ConfigDict(extra="forbid")

    default_model: str
    device_preference: str
    resolved_device: str
    cuda_available: bool
    loaded_models: list[str]
    models: list[ModelInfoResponse]


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "error"] = "ok"
    version: str
    uptime_seconds: float = Field(ge=0.0)
    device: str
    cuda_available: bool
    models_loaded: int
    models_available: int
    memory: MemoryInfo
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    request_id: str | None = None
    errors: list[dict[str, Any]] | None = None


def validate_model_key(value: str) -> str:
    """Validate a model key outside of a Pydantic model (CLI / UI helpers)."""
    return get_model_spec(value).key
