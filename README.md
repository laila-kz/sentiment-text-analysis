# Sentiment Text Analysis

A production-ready sentiment and emotion analysis platform: lazy-loaded Hugging Face transformers, calibrated batch inference, a FastAPI REST service, a Streamlit dashboard, a CLI, comprehensive tests, container builds and GitHub Actions CI.

## 1. Key features

- **Multi-model support** – `sentiment` (distilbert-base-uncased-finetuned-sst-2-english), `emotion` (bhadresh-ps/distilbert-base-uncased-emotion) and `sentiment3` (cardiffnlp/twitter-roberta-base-sentiment-latest), with a registry that can be extended.
- **Calibrated & uncertainty-aware** – per-item uncertainty report (entropy, normalized entropy, top-2 margin), softmax with optional temperature scaling, full distribution and raw logits.
- **Robust batch inference** – automatic chunking by `batch_size`, CUDA/MPS auto-detection, half-precision on GPU, and strict empty-input handling for single-text runs.
- **Flat, import-safe layout** – top-level modules (`app.py`, `api.py`, `cli.py`, ...). `app.py` is the Streamlit entrypoint (no root `streamlit.py` that would shadow the installed package). Heavy dependencies are imported lazily so the test suite runs without torch/transformers/streamlit/plotly.
- **API-first** – FastAPI with Pydantic v2 schemas, OpenAPI examples, request IDs, structured JSON logging, in-memory rate limiting, Prometheus-style metrics at `/metrics`, CORS and global exception handling.
- **Interactive UI** – Streamlit dashboard with playground, batch CSV/TSV/JSON upload, Plotly distribution + radar (for 3+ classes), sentence-level HTML highlighting, and optional Whisper audio transcription.
- **CLI & batch I/O** – `cli.py` supports single text, file ingestion (CSV/TSV/JSON), column detection, calibrated output, threshold warnings and JSON export. `batchio.py` annotates records with predictions.
- **Quality gates** – 283 tests (2 skipped when torch missing), Ruff lint/format, mypy configuration, coverage reporting, Docker multi-stage build and a full CI pipeline.
- **Test doubles included** – `tests/conftest.py` provides `FakeLoader`/`FakePipeline` so the entire suite passes offline (no Hugging Face downloads, no GPU).

## 2. Project layout

```text
.
├── .dockerignore
├── .env.example
├── .github/workflows/ci.yml
├── Dockerfile
├── README.md
├── api.py              # FastAPI app + routes + dependencies
├── app.py              # Streamlit dashboard entrypoint
├── batchio.py          # CSV/TSV/JSON ingestion + annotated export
├── charts.py           # Plotly distribution/radar/histogram figures
├── cli.py              # CLI for single/batch inference
├── config.py           # Pydantic/dataclass settings + env parsing
├── docker-compose.yml  # API + dashboard with shared HF cache
├── observability.py    # JSON logging, middleware, rate limiting, metrics
├── pyproject.toml      # Ruff/mypy/pytest configuration
├── requirements.txt    # Core + API runtime deps
├── requirements-dev.txt# Tests/lint/typecheck
├── requirements-ui.txt # Streamlit + Plotly + pandas
├── schemas.py          # Pydantic request/response models
├── sentiment.py        # Model registry + inference engine (lazy imports)
├── transcribe.py       # Whisper ASR wrapper (lazy)
├── utils.py            # Probabilities, entropy, highlighting, validation
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_batchio.py
    ├── test_cli.py
    ├── test_observability.py
    ├── test_schemas.py
    ├── test_sentiment.py
    ├── test_transcribe.py
    └── test_utils.py
```

## 3. Quick start (local)

Prerequisites: Python 3.11+ (the code uses `enum.StrEnum`). Heavy ML deps (torch/transformers) are large; CPU-only is sufficient for the tests and for running with smaller models.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install runtime + API (CPU torch recommended on laptops)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# (Optional) install the UI + dashboard extras
pip install -r requirements-ui.txt

# (Optional) install dev tools
pip install -r requirements-dev.txt
```

CLI examples:

```bash
# Single text (sentiment model by default)
python cli.py --text "I absolutely love this product!"

# Emotion classification with JSON output
python cli.py --text "I'm so frustrated right now" --model emotion -o result.json

# Batch from CSV (auto-detects text column)
python cli.py --file reviews.csv -o annotated.json --batch-size 8

# Interactive REPL
python cli.py --interactive
```

FastAPI service:

```bash
# Start the API (default host/port 127.0.0.1:8000)
uvicorn api:app --reload

# Health and metadata
curl http://127.0.0.1:8000/health        # {"status":"ok",...}
curl http://127.0.0.1:8000/               # lists endpoints
curl http://127.0.0.1:8000/v1/models      # available models + specs
curl http://127.0.0.1:8000/metrics        # Prometheus-style counters/latency

# Analyse
curl -X POST http://127.0.0.1:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"inputs":["Great service","Awful experience"], "model":"sentiment"}'
```

Streamlit dashboard:

```bash
# Run the UI (entrypoint is app.py)
streamlit run app.py
```

## 4. Configuration

All configuration is read from environment variables with the `SENTIMENT_` prefix. See `.env.example` for the full list (defaults match `config.Settings`). Key variables:

| Variable | Default | Notes |
|---|---|---|
| `SENTIMENT_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `SENTIMENT_CACHE_DIR` | `./model_cache` | Hugging Face cache location |
| `SENTIMENT_BATCH_SIZE` | `16` | per forward pass when batching |
| `SENTIMENT_TEMPERATURE` | `1.0` | softmax temperature (>1 flattens, <1 sharpens) |
| `SENTIMENT_UNCERTAINTY_THRESHOLD` | `0.35` | normalized entropy above this flags "uncertain" |
| `SENTIMENT_FP16` | `false` | enable half precision on CUDA |
| `SENTIMENT_PRELOAD_MODELS` | `""` | comma-separated model keys to warm at startup |
| `SENTIMENT_RATE_LIMIT_REQUESTS` | `60` | per window; `0` disables |
| `SENTIMENT_RATE_LIMIT_WINDOW` | `60` | seconds |
| `SENTIMENT_MAX_BATCH_ITEMS` | `256` | max inputs per `/v1/analyze` request |
| `SENTIMENT_MAX_TEXT_LENGTH` | `10000` | character cap per input |
| `SENTIMENT_CORS_ORIGINS` | `http://localhost:8501` | comma-separated allowed origins |
| `SENTIMENT_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `SENTIMENT_LOG_JSON` | `true` | emit JSON access logs |

## 5. Docker & Compose

The repository ships a multi-stage `Dockerfile` (CPU torch by default) and `docker-compose.yml` (API + dashboard sharing an `hf-cache` volume).

Build and run with Docker:

```bash
# Build (CPU-only; use --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 for CUDA)
docker build -t sentiment-platform .

# Run API
docker run -p 8000:8000 -v hf-cache:/models sentiment-platform

# Run dashboard
docker run -p 8501:8501 -v hf-cache:/models sentiment-platform streamlit run app.py
```

With Docker Compose:

```bash
# Bring up both services (reads .env if present)
docker compose up --build

# API:       http://localhost:8000 (/docs, /metrics)
# Dashboard: http://localhost:8501
```

Notes:
- The default `TORCH_INDEX_URL` is `https://download.pytorch.org/whl/cpu` (much smaller). Override via build arg or `TORCH_INDEX_URL` env var for GPU images.
- Models download on first use into `/models` (mounted as `hf-cache` volume). Set `SENTIMENT_PRELOAD_MODELS=sentiment,emotion` to warm caches at startup (can be slow on cold pulls).
- The compose file includes an optional NVIDIA GPU reservation (ignored on CPU hosts). Set `SENTIMENT_GPU_COUNT` to control device count.

## 6. Testing & quality

Run the full offline test suite (no network/model downloads):

```bash
# All tests
pytest -q

# With coverage
pytest -q --cov=api --cov=sentiment --cov=utils --cov=batchio --cov=schemas --cov=observability --cov=cli --cov-report=term-missing

# Lint and format
ruff check .
ruff format --check .

# Type checking (mypy configured in pyproject.toml; install mypy if needed)
mypy .
```

What the tests cover: softmax/entropy/margin/validation/highlighting, model registry and calibration, batch chunking and temperature, Pydantic schemas, CSV/TSV/JSON parsing and column heuristics, FastAPI endpoints (including 413/422/429/503 paths and dependency overrides), CLI argument parsing and modes, observability (JSON formatter, metrics, middleware, rate limiter), Whisper transcription edge cases, and Streamlit helper contracts. Two tests are skipped only when `torch` is unavailable (the Torch pipeline path is exercised separately).

## 7. API reference (high level)

- `GET /` – service metadata and available endpoints.
- `GET /health` – liveness/readiness probe (includes uptime and settings snapshot).
- `GET /v1/models` – list of model keys, specs (labels, id, task) and defaults.
- `POST /v1/analyze` – batch analysis. Body: `{inputs: string[], model?: "sentiment"|"emotion"|"sentiment3", batch_size?: int, temperature?: number, include_logits?: bool}`. Returns per-item `label`, `confidence`, `distribution`, `logits?`, `uncertainty`, `processing_ms`, `device`, warnings and a `summary` (analysed/skipped/errors/avg_latency_ms).
- `GET /v1/benchmark` – quick latency/throughput estimate using synthetic or provided texts (model/batch_size/iterations configurable).
- `GET /metrics` – Prometheus text format: counters (`http_requests_total`, `requests_by_model{model=...}`, `errors_total{code=...}`) and latency histograms (`latency_ms_sum`, `latency_ms_max`, `latency_ms_avg`, plus recent samples).

Error responses follow a consistent shape (`{error: {code, message, request_id, details?}}`) with appropriate 4xx/5xx status codes (400/404/413/422/429/500/503). Request IDs are echoed via `X-Request-ID` (and generated if absent). Rate limiting returns `Retry-After` when exceeded.

## 8. Design notes

- **Lazy loading**: `transformers`, `torch`, `streamlit`, `plotly`, `pandas`, `whisper/faster-whisper` are only imported inside functions that actually use them. This keeps import time low and enables the pure-Python test doubles.
- **Deterministic tests**: `FakeLoader` produces logits via a tiny deterministic rule (token hash + label weights), so distributions are stable and assertions don't depend on external state.
- **Defensive validation**: `utils.is_meaningful_text` filters empty/whitespace/emoji-only/punctuation-only strings; single-text analysis uses `strict=True` (raises on unusable input) while batch files use `strict=False` (record is marked `skipped` with a reason).
- **Middleware order**: `RateLimitMiddleware` is registered before `RequestContextMiddleware` (Starlette executes the last-added middleware outermost), ensuring rate-limited responses still include request IDs and consistent error payloads.
- **Encoding hygiene**: `requirements*.txt` are UTF-8 (no BOM). The CLI and batch I/O treat CSV/TSV as UTF-8 with BOM stripping (`utf-8-sig`).
- **Security**: no secrets logged or committed. CORS origins are configurable. Rate limiting is in-memory (process-scoped) and intended for local/dev/proxy-fronted deployments.

## 9. Known limitations & next steps

- **External dependencies not installed locally**: torch, transformers, faster-whisper, streamlit, plotly/pandas are absent in this environment; real model inference, audio transcription and the full Streamlit runtime were not exercised here. The provided test suite fully validates the logic via fakes.
- **Benchmarks**: `GET /v1/benchmark` returns timing estimates but representative numbers against real checkpoints should be captured on target hardware (especially GPU).
- **Persistence**: rate limiting and metrics are in-memory only (single process). For multi-replica deployments consider Redis/token-bucket or a shared store.
- **Mypy strictness**: `disallow_untyped_defs`/`disallow_incomplete_defs` are relaxed in CI-friendly mode (kept enabled for core modules where practical). Tighten further as the codebase evolves.
- **GPU wheels**: switch `TORCH_INDEX_URL` to the appropriate CUDA index (e.g. `https://download.pytorch.org/whl/cu124`) in Docker builds for GPU acceleration.

## 10. CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes/PRs:
1. Checkout + Python 3.11 (pip cache)
2. Install dev deps (lightweight; core ML deps mocked in tests)
3. `ruff check` (lint)
4. `ruff format --check --diff` (formatting)
5. `mypy .` (type checks)
6. `pytest` with coverage (XML artifact uploaded)
7. Docker build + container smoke test (`/health` probe)

## 11. License

MIT. See individual model licenses on Hugging Face for the checkpoints you load.
