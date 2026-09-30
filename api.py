"""FastAPI microservice exposing the sentiment engine over HTTP.

Endpoints
---------
``POST /v1/analyze``   single **and** batch text analysis (vectorised)
``GET  /v1/models``    model registry with load state
``GET  /v1/benchmark`` throughput/latency probe for the active model
``GET  /health``       model availability + system memory checks
``GET  /metrics``      Prometheus text exposition

Run locally with::

    uvicorn api:app --reload
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import Settings, get_settings
from observability import (
    Metrics,
    RateLimitMiddleware,
    RequestContextMiddleware,
    configure_logging,
    metrics,
)
from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchSummaryResponse,
    ErrorResponse,
    HealthResponse,
    MemoryInfo,
    ModelsResponse,
    SegmentResponse,
    TextAnalysisResponse,
)
from sentiment import (
    DEFAULT_MODEL,
    SentimentEngine,
    UnknownModelError,
    get_engine,
    get_model_spec,
    resolve_device,
    summarise,
)

__all__ = ["app", "create_app", "get_settings_dep", "health_payload", "run"]

logger = logging.getLogger("api")

STARTED_AT = time.time()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_settings_dep(request: Request) -> Settings:
    """FastAPI dependency returning the settings bound to *this* app instance.

    Reading them from ``app.state`` (instead of the global singleton) is what
    lets tests spin up isolated apps with their own limits.
    """
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def get_engine_dep() -> SentimentEngine:
    """FastAPI dependency returning the shared inference engine.

    Tests override this with ``app.dependency_overrides[get_engine_dep]`` to
    inject an engine backed by a fake model, so no checkpoint is downloaded.
    """
    return get_engine()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
EngineDep = Annotated[SentimentEngine, Depends(get_engine_dep)]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and optionally warm models before serving traffic."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, json_format=settings.log_json)
    logger.info(
        "starting %s v%s (device=%s, models=%s)",
        settings.api_title,
        settings.api_version,
        settings.device,
        ",".join(settings.preload_models) or "lazy",
    )
    if settings.preload_models:
        engine = get_engine()
        for model in settings.preload_models:
            try:
                engine.warmup(model)
            except Exception as exc:  # pragma: no cover - depends on model download
                logger.error("preload of model %r failed: %s", model, exc)
    try:
        yield
    finally:
        logger.info("shutting down")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (factory keeps tests isolated)."""
    resolved = settings or get_settings()

    application = FastAPI(
        title=resolved.api_title,
        version=resolved.api_version,
        summary="Multi-model sentiment and emotion analysis with calibrated confidence.",
        description=__doc__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "analysis", "description": "Text sentiment/emotion endpoints."},
            {"name": "system", "description": "Health, metrics and service metadata."},
        ],
    )
    application.state.settings = resolved
    application.state.metrics = metrics

    if resolved.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Process-Time", "Retry-After"],
        )
    # NOTE: middleware runs in reverse registration order, so the rate limiter is
    # registered first and therefore sits *inside* the request-context layer,
    # which guarantees every response (including 429s) carries a request id.
    application.add_middleware(
        RateLimitMiddleware,
        requests=resolved.rate_limit_requests,
        window=resolved.rate_limit_window,
        enabled=resolved.is_rate_limited,
    )
    application.add_middleware(RequestContextMiddleware)

    _register_exception_handlers(application)
    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:  # noqa: C901 - flat route table
    @application.get("/", tags=["system"], summary="Service banner")
    async def root() -> dict[str, Any]:
        settings: Settings = application.state.settings
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "analyze": "POST /v1/analyze",
                "models": "GET /v1/models",
                "benchmark": "GET /v1/benchmark",
                "health": "GET /health",
                "metrics": "GET /metrics",
            },
        }

    @application.post(
        "/v1/analyze",
        tags=["analysis"],
        summary="Analyse a single text or a batch of texts",
        response_model=AnalyzeResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Request validation failed"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
            503: {"model": ErrorResponse, "description": "Model unavailable"},
        },
    )
    async def analyze(
        payload: AnalyzeRequest,
        request: Request,
        engine: EngineDep,
        settings: SettingsDep,
    ) -> AnalyzeResponse:
        started = time.perf_counter()
        request_id = getattr(request.state, "request_id", "")
        if len(payload.inputs) > settings.max_batch_items:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Batch of {len(payload.inputs)} exceeds the limit of "
                    f"{settings.max_batch_items} items."
                ),
            )

        spec = get_model_spec(payload.model)
        results = engine.analyse_batch(
            payload.inputs,
            model=spec,
            batch_size=payload.batch_size,
            temperature=payload.temperature,
            include_logits=payload.include_logits,
        )
        segments: list[SegmentResponse] | None = None
        if payload.include_segments and payload.text is not None:
            segments = [
                SegmentResponse.model_validate(segment.to_dict())
                for segment in engine.analyse_segments(payload.text, model=spec)
            ]

        analysed = [result for result in results if result.is_ok]
        device = analysed[0].device if analysed else resolve_device(settings.device)
        summary = summarise(results)
        took_ms = (time.perf_counter() - started) * 1000.0

        counter: Metrics = application.state.metrics
        counter.increment("analysis_requests_total")
        counter.increment_labeled("analysis_requests_by_model", {"model": spec.key})
        counter.increment("texts_analyzed_total", float(summary.analysed))
        counter.increment("texts_skipped_total", float(summary.skipped))

        items = [
            TextAnalysisResponse.from_result(result, index) for index, result in enumerate(results)
        ]
        if segments:
            for item in items:
                item.segments = segments

        return AnalyzeResponse(
            request_id=request_id,
            model=spec.key,
            model_id=spec.hf_id,
            device=device,
            batch=payload.is_batch,
            results=items,
            summary=BatchSummaryResponse.from_result(summary),
            took_ms=round(took_ms, 3),
            version=application.version,
        )

    @application.get(
        "/v1/models",
        tags=["system"],
        summary="List the model registry and its load state",
        response_model=ModelsResponse,
    )
    async def models(engine: EngineDep) -> ModelsResponse:
        return ModelsResponse.model_validate(engine.info())

    @application.get(
        "/v1/benchmark",
        tags=["system"],
        summary="Measure throughput and latency for the active model",
    )
    async def benchmark(
        engine: EngineDep,
        settings: SettingsDep,
        model: str = DEFAULT_MODEL,
        batch_size: Annotated[int | None, Query(ge=1, le=256)] = None,
        repeats: Annotated[int, Query(ge=1, le=20)] = 1,
    ) -> dict[str, Any]:
        try:
            spec = get_model_spec(model)
        except UnknownModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result = engine.benchmark(
            model=spec,
            batch_size=batch_size or settings.batch_size,
            repeats=repeats,
        )
        return {"model": spec.key, "model_id": spec.hf_id, **result.to_dict()}

    @application.get(
        "/health",
        tags=["system"],
        summary="Liveness/readiness probe with memory and model checks",
        response_model=HealthResponse,
    )
    async def health(engine: EngineDep, settings: SettingsDep) -> HealthResponse:
        return health_payload(engine=engine, settings=settings)

    @application.get(
        "/metrics", tags=["system"], summary="Prometheus metrics", response_class=PlainTextResponse
    )
    async def prometheus_metrics() -> Response:
        return PlainTextResponse(
            application.state.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _memory_info() -> MemoryInfo:
    """Best-effort memory snapshot (``psutil`` is optional)."""
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process(os.getpid())
        virtual = psutil.virtual_memory()
        return MemoryInfo(
            rss_mb=round(process.memory_info().rss / (1024 * 1024), 2),
            available_mb=round(virtual.available / (1024 * 1024), 2),
            total_mb=round(virtual.total / (1024 * 1024), 2),
            percent_used=round(float(virtual.percent), 2),
        )
    except Exception:  # pragma: no cover - psutil optional / restricted envs
        logger.debug("psutil unavailable; omitting memory metrics", exc_info=True)
        return MemoryInfo(rss_mb=0.0)


def health_payload(engine: SentimentEngine, settings: Settings) -> HealthResponse:
    """Build the health document (pure function, easy to assert on)."""
    info = engine.info()
    memory = _memory_info()
    checks = {
        "registry_populated": bool(info["models"]),
        "device_resolved": bool(info["resolved_device"]),
        "memory_readable": True,
        "model_loaded": bool(info["loaded_models"]),
    }
    pressure = (memory.percent_used or 0.0) > 95.0
    if not checks["registry_populated"]:
        state = "error"
    elif pressure:
        state = "degraded"
    else:
        state = "ok"
    return HealthResponse(
        status=state,  # type: ignore[arg-type]
        version=settings.api_version,
        uptime_seconds=round(time.time() - STARTED_AT, 3),
        device=str(info["resolved_device"]),
        cuda_available=bool(info["cuda_available"]),
        models_loaded=len(info["loaded_models"]),
        models_available=len(info["models"]),
        memory=memory,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _error(
    status_code: int, detail: str, request: Request, errors: list[dict[str, Any]] | None = None
) -> JSONResponse:
    payload: dict[str, Any] = {
        "detail": detail,
        "request_id": getattr(request.state, "request_id", None),
    }
    if errors:
        payload["errors"] = errors
    return JSONResponse(status_code=status_code, content=payload)


def _register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "location": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        logger.info(
            "validation failed", extra={"request_id": getattr(request.state, "request_id", None)}
        )
        return _error(422, "Request validation failed.", request, errors)

    @application.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), request)

    @application.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        logger.exception(
            "unhandled error", extra={"request_id": getattr(request.state, "request_id", None)}
        )
        return _error(
            500,
            "Internal server error.",
            request,
        )

    @application.exception_handler(UnknownModelError)
    async def _unknown_model(request: Request, exc: UnknownModelError) -> JSONResponse:
        return _error(404, str(exc), request)

    @application.exception_handler(ValueError)
    async def _bad_input(request: Request, exc: ValueError) -> JSONResponse:
        return _error(400, str(exc), request)

    @application.exception_handler(RuntimeError)
    async def _model_failure(request: Request, exc: RuntimeError) -> JSONResponse:
        return _error(503, str(exc), request)


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------


def run() -> None:  # pragma: no cover - manual invocation
    """Run the service with uvicorn (``python api.py``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
    )


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    run()
