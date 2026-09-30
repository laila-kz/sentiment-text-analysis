"""Environment-driven configuration shared by every entry point.

All settings can be overridden with environment variables (12-factor style) so
the same image can run in dev, CI and production without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = ["ENV_PREFIX", "Settings", "get_settings"]

ENV_PREFIX: Final[str] = "SENTIMENT_"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"{ENV_PREFIX}{name}", default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the process configuration."""

    # --- model runtime -------------------------------------------------
    device: str = "auto"
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", "./model_cache")))
    max_length: int = field(default_factory=lambda: _env_int("MAX_LENGTH", 256))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 16))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 1.0))
    uncertainty_threshold: float = field(
        default_factory=lambda: _env_float("UNCERTAINTY_THRESHOLD", 0.35)
    )
    fp16: bool = field(default_factory=lambda: _env_bool("FP16", False))
    preload_models: tuple[str, ...] = ()

    # --- API -----------------------------------------------------------
    api_title: str = "Sentiment Analysis API"
    api_version: str = "1.0.0"
    rate_limit_requests: int = field(default_factory=lambda: _env_int("RATE_LIMIT_REQUESTS", 60))
    rate_limit_window: float = field(default_factory=lambda: _env_float("RATE_LIMIT_WINDOW", 60.0))
    max_text_length: int = field(default_factory=lambda: _env_int("MAX_TEXT_LENGTH", 10_000))
    max_batch_items: int = field(default_factory=lambda: _env_int("MAX_BATCH_ITEMS", 256))
    cors_origins: tuple[str, ...] = ()

    # --- observability --------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper() or "INFO")
    log_json: bool = field(default_factory=lambda: _env_bool("LOG_JSON", True))

    @property
    def is_rate_limited(self) -> bool:
        return self.rate_limit_requests > 0 and self.rate_limit_window > 0


_CACHED: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process settings (cached; pass ``refresh=True`` to re-read)."""
    global _CACHED
    if _CACHED is None or refresh:
        raw_models = _env("PRELOAD_MODELS")
        raw_origins = _env("CORS_ORIGINS")
        _CACHED = Settings(
            preload_models=tuple(part.strip() for part in raw_models.split(",") if part.strip()),
            cors_origins=tuple(part.strip() for part in raw_origins.split(",") if part.strip()),
        )
    return _CACHED
