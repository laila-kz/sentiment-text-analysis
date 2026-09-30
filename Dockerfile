# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build for the sentiment platform.
#
#   docker build -t sentiment-platform .
#
# CPU-only images (much smaller, recommended for CI and laptops):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t sentiment-platform .
#
# The default runtime serves the FastAPI app. The dashboard and CLI are
# available in the same image:
#
#   docker run -p 8501:8501 sentiment-platform streamlit run app.py
#   docker run -it sentiment-platform python cli.py --text "great product"
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Stage 1: resolve dependencies into a self-contained virtualenv.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ARG TORCH_INDEX_URL
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build
COPY requirements.txt requirements-ui.txt ./

# CPU wheels first: --extra-index-url keeps the default PyPI resolution working
# for every other dependency while preferring the much smaller CPU torch build.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --extra-index-url "$TORCH_INDEX_URL" -r requirements.txt -r requirements-ui.txt

# ---------------------------------------------------------------------------
# Stage 2: slim runtime with the application code and the virtualenv only.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=warning \
    SENTIMENT_CACHE_DIR=/models \
    SENTIMENT_PRELOAD_MODELS=false

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app.py batchio.py charts.py cli.py config.py observability.py \
     schemas.py sentiment.py transcribe.py utils.py api.py ./
COPY pyproject.toml README.md ./

# Unprivileged runtime user; /models is the shared Hugging Face cache volume.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /models \
    && chown -R app:app /models /app
USER app

VOLUME ["/models"]
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
