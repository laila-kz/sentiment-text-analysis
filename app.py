"""Modern Streamlit dashboard for the sentiment platform.

Launch with::

    streamlit run app.py

Three workspaces are provided:

1. **Playground** – single text with confidence meters, in-text highlighting and
   interactive distribution / emotion-radar charts.
2. **Batch files** – drag-and-drop CSV/JSON with automatic column mapping,
   Plotly distributions and downloadable annotated results.
3. **Audio** – record or upload audio, transcribe with Whisper and analyse the
   transcript. Model switching happens in the sidebar and applies everywhere.

A fourth tab runs the built-in benchmark so the UI and the API can be compared
on the same hardware.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd
import streamlit as st

import batchio
import sentiment
from batchio import BatchDataset, UnsupportedFormatError
from charts import (
    batch_distribution_figure,
    confidence_histogram_figure,
    distribution_figure,
    emotion_radar_figure,
    entropy_histogram_figure,
)
from sentiment import AnalysisResult, ModelSpec, get_model_spec
from transcribe import ASR_MODELS, DEFAULT_ASR_KEY, TranscriptionError, transcribe
from utils import emoji_for_label, highlight_spans, rejection_reason, score_bar, truncate

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Sentiment Analysis Platform",
    page_icon="\N{SPEECH BALLOON}",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container { padding-top: 2.2rem; }
  .sentiment-pill { display:inline-block; padding:0.35rem 0.9rem; border-radius:999px;
                    font-weight:700; font-size:1.05rem; color:#fff; }
  .muted { color:#5f6368; font-size:0.85rem; }
  .kpi-label { font-size:0.8rem; text-transform:uppercase; letter-spacing:.04em; color:#5f6368; }
</style>
"""

EXAMPLE_TEXTS = [
    "The delivery was two days late but the product itself is fantastic.",
    "Support ignored three emails and then closed my ticket without a refund.",
    "Honestly not sure what to think about this update.",
    "I love the new design, it makes the app much easier to use!\N{GRINNING FACE}",
]


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def inject_css() -> None:
    """Inject the small stylesheet used by the KPI cards."""
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model weights...")
def load_engine() -> sentiment.SentimentEngine:
    """Return the shared engine (models stay resident between reruns)."""
    return sentiment.get_engine()


def model_choices() -> list[str]:
    return [spec.key for spec in sentiment.available_models()]


def spec_label(key: str) -> str:
    spec = get_model_spec(key)
    return f"{spec.key}  ·  {spec.num_labels}-class {spec.kind}  ·  {spec.description}"


def chart_key(prefix: str, *parts: str) -> str:
    """Stable Plotly element key so Streamlit does not re-create charts."""
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()[:10]
    return f"{prefix}_{digest}"


# ---------------------------------------------------------------------------
# Shared rendering
# ---------------------------------------------------------------------------


def render_kpis(result: AnalysisResult) -> None:
    """Top row: predicted label, confidence meter and uncertainty summary."""
    label = result.label or "n/a"
    emoji = emoji_for_label(label)
    confidence = float(result.confidence or 0.0)
    uncertainty = result.uncertainty

    col_label, col_conf, col_unc, col_time = st.columns(4)
    with col_label:
        st.metric("Prediction", f"{emoji} {label.title()}")
        st.caption(result.model_id)
    with col_conf:
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(min(1.0, confidence), text=score_bar(confidence))
    with col_unc:
        entropy = uncertainty.normalized_entropy if uncertainty else 0.0
        st.metric("Uncertainty", f"{entropy:.1%}", help="Normalised entropy (100% = uniform)")
        st.progress(
            min(1.0, entropy),
            text="high" if uncertainty and uncertainty.is_uncertain else "low",
        )
    with col_time:
        st.metric("Latency", f"{result.processing_ms:.1f} ms")
        st.caption(f"device={result.device} · T={result.temperature:g}")

    if uncertainty and uncertainty.is_uncertain:
        st.warning(
            "Low-confidence prediction: the label distribution is close to uniform "
            f"(margin {uncertainty.margin:.2f}).",
            icon="\N{WARNING SIGN}",
        )
    for warning in result.warnings:
        st.warning(warning)


def render_distribution(result: AnalysisResult, spec: ModelSpec) -> None:
    """Distribution bars plus a radar chart for multi-class models."""
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            distribution_figure(result.distribution, title=f"{spec.key} probabilities"),
            use_container_width=True,
            key=chart_key("dist", spec.key, result.text),
        )
    with right:
        if spec.num_labels >= 3:
            st.plotly_chart(
                emotion_radar_figure(result.distribution),
                use_container_width=True,
                key=chart_key("radar", spec.key, result.text),
            )
        else:
            st.info(
                f"{spec.key} is a binary classifier: a radar chart needs at least "
                "3 axes, so probabilities are shown as bars. Switch the model to "
                "`emotion` for the 6-axis radar view.",
                icon="\N{BAR CHART}",
            )
            ranked = pd.DataFrame(
                [
                    {"label": item.label, "probability": item.probability, "logit": item.logit}
                    for item in result.ranked
                ]
            )
            st.dataframe(ranked, hide_index=True, use_container_width=True)


def render_ranked_table(result: AnalysisResult) -> None:
    """Tabular view of the calibrated distribution with raw logits."""
    rows = [
        {
            "rank": position + 1,
            "label": item.label,
            "probability": item.probability,
            "logit": item.logit,
            "bar": score_bar(item.probability, 12),
        }
        for position, item in enumerate(result.ranked)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_highlighted(text: str, model_key: str) -> None:
    """Sentence-level highlighting of the sentiment of each clause."""
    segments = sentiment.get_engine().analyse_segments(text, model=model_key)
    if not segments:
        return
    markup = highlight_spans(
        text,
        [segment.label or "neutral" for segment in segments],
        [segment.score or 0.0 for segment in segments],
    )
    st.markdown(markup, unsafe_allow_html=True)
    legend = st.columns(len(segments))
    for column, segment in zip(legend, segments, strict=False):
        column.caption(
            f"{emoji_for_label(segment.label)} {segment.label}: {segment.score or 0:.0%}"
        )


def render_json(payload: dict[str, Any], filename: str) -> None:
    """Collapsible raw JSON view with a download button."""
    with st.expander("Raw response payload"):
        st.json(payload)
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=filename,
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Tab 1: single text playground
# ---------------------------------------------------------------------------


def render_playground(engine: sentiment.SentimentEngine, model_key: str, live: bool) -> None:
    spec = get_model_spec(model_key)
    st.subheader(f"Playground · {spec.key}")

    if st.session_state.get("example") and "example_text" not in st.session_state:
        st.session_state["example_text"] = st.session_state.pop("example")
    text = st.text_area(
        "Text to analyse",
        key="example_text",
        height=150,
        placeholder="Type or paste any text, e.g. 'The screen is gorgeous but the "
        "battery drains overnight.'",
        label_visibility="collapsed",
    )

    column_a, column_b = st.columns([1, 5])
    with column_a:
        if st.button("Analyse", type="primary", use_container_width=True):
            st.session_state["run_playground"] = True
        st.caption(f"{len(text.split())} words")
    with column_b:
        if st.toggle("Analyse as you type", value=live, key="live_mode"):
            st.caption("Every keystroke triggers inference; handy for short texts.")

    if not text.strip():
        st.info(
            "Enter some text to see calibrated sentiment, uncertainty and the "
            "full label distribution.",
            icon="\N{SPARKLES}",
        )
        st.caption("Examples:")
        for example in EXAMPLE_TEXTS:
            if st.button(example, key=f"ex_{hash(example) % 10_000}"):
                st.session_state["example_text"] = example
                st.rerun()
        return

    run = st.session_state.pop("run_playground", False) or st.session_state.get("live_mode", False)
    if not run:
        return

    result = engine.analyse(text, model=spec, include_logits=True)
    if not result.is_ok:
        st.error(f"Input rejected ({result.reason}): {rejection_reason(text)}")
        return

    render_kpis(result)
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        render_distribution(result, spec)
    with right:
        render_ranked_table(result)

    with st.expander("Per-sentence highlighting", expanded=True):
        render_highlighted(text, spec.key)

    render_json(result.to_dict(), f"{spec.key}-{truncate(text, 32, '')}.json")


# ---------------------------------------------------------------------------
# Tab 2: batch file analysis
# ---------------------------------------------------------------------------


def _dataset_from_upload(upload: Any) -> BatchDataset | None:
    """Parse the uploaded file once per (name, size) pair."""
    fingerprint = f"{upload.name}:{upload.size}"
    cached = st.session_state.get("dataset_fingerprint")
    if cached == fingerprint and "dataset" in st.session_state:
        return st.session_state["dataset"]
    try:
        dataset = batchio.load_dataset(upload.getvalue(), upload.name)
    except UnsupportedFormatError as exc:
        st.error(str(exc), icon="\N{CROSS MARK}")
        return None
    st.session_state["dataset"] = dataset
    st.session_state["dataset_fingerprint"] = fingerprint
    st.session_state.pop("batch_results", None)
    return dataset


def render_batch(engine: sentiment.SentimentEngine, model_key: str) -> None:
    spec = get_model_spec(model_key)
    st.subheader("Batch analysis")

    upload = st.file_uploader(
        "Drop a CSV or JSON file here",
        type=["csv", "tsv", "json"],
        help=(
            "CSV: one text column plus any metadata columns. "
            "JSON: an array of objects or an array of strings."
        ),
    )
    if upload is None:
        st.info(
            "Upload a file to analyse thousands of rows in vectorised batches. "
            "Column detection happens automatically and can be overridden.",
            icon="\N{OPEN FILE FOLDER}",
        )
        return

    dataset = _dataset_from_upload(upload)
    if dataset is None:
        return

    column = st.selectbox(
        "Text column",
        dataset.column_options(),
        index=dataset.column_options().index(dataset.text_column)
        if dataset.text_column in dataset.column_options()
        else 0,
        help="Auto-detected from column names and value shapes; override if needed.",
    )
    dataset.text_column = column

    texts = dataset.texts()
    preview = dataset.records[: min(25, len(dataset))]
    st.caption(
        f"{len(dataset):,} rows · {len(dataset.columns)} columns · "
        f"analyse limit {engine.settings.max_batch_items:,}"
    )
    st.dataframe(pd.DataFrame(preview), hide_index=True, use_container_width=True)

    if len(dataset) > engine.settings.max_batch_items:
        st.error(
            f"File has {len(dataset):,} rows but the limit is "
            f"{engine.settings.max_batch_items:,}. Trim the file or raise "
            "SENTIMENT_MAX_BATCH_ITEMS.",
            icon="\N{CROSS MARK}",
        )
        return

    batch_size = st.slider(
        "Batch size (texts per forward pass)", 1, 128, engine.settings.batch_size
    )
    if st.button("Analyse file", type="primary"):
        progress = st.progress(0.0, text="Analysing…")
        results: list[AnalysisResult] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            results.extend(engine.analyse_batch(chunk, model=spec, batch_size=batch_size))
            progress.progress(min(1.0, (start + len(chunk)) / len(texts)))
        st.session_state["batch_results"] = results
        progress.empty()

    results: Sequence[AnalysisResult] = st.session_state.get("batch_results") or []
    if not results:
        return

    summary = sentiment.summarise(results)
    annotated = batchio.annotate_records(dataset.records, results)
    frame = pd.DataFrame(annotated)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows analysed", f"{summary.analysed:,}")
    metric_cols[1].metric("Skipped", f"{summary.skipped:,}")
    metric_cols[2].metric("Mean confidence", f"{summary.mean_confidence:.1%}")
    metric_cols[3].metric(
        "Throughput", f"{len(results) / max(summary.total_ms, 1e-6) * 1000:,.0f} texts/s"
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(batch_distribution_figure(summary.label_counts), use_container_width=True)
    with right:
        st.plotly_chart(
            confidence_histogram_figure(
                [r.confidence for r in results if r.confidence is not None]
            ),
            use_container_width=True,
        )
    if any(r.uncertainty for r in results):
        st.plotly_chart(
            entropy_histogram_figure(
                [r.uncertainty.normalized_entropy for r in results if r.uncertainty]
            ),
            use_container_width=True,
        )

    st.dataframe(frame, hide_index=True, use_container_width=True, height=360)

    stem = upload.name.rsplit(".", 1)[0]
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download annotated CSV",
        data=batchio.to_csv(annotated),
        file_name=f"{stem}-annotated.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        "Download annotated JSON",
        data=batchio.to_json(annotated),
        file_name=f"{stem}-annotated.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 3: live audio
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Whisper…")
def load_asr_cached(model_id: str) -> Any:
    """Memoised ASR pipeline so reruns do not reload the checkpoint."""
    from transcribe import load_asr

    return load_asr(model_id)


def _audio_widgets() -> bytes | None:
    """Collect audio from the microphone or an upload, whichever is available."""
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Record audio", key="recorder")
        if recorded is not None:
            return recorded.getvalue()
    uploaded = st.file_uploader(
        "…or upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        key="audio_upload",
    )
    return uploaded.getvalue() if uploaded is not None else None


def render_audio(engine: sentiment.SentimentEngine, model_key: str) -> None:
    spec = get_model_spec(model_key)
    st.subheader("Audio & speech")

    whisper_key = st.selectbox(
        "Whisper model",
        list(ASR_MODELS),
        index=list(ASR_MODELS).index(DEFAULT_ASR_KEY),
        help="tiny is fastest, small is most accurate. The first run downloads weights.",
    )
    audio = _audio_widgets()
    if audio is None:
        st.info(
            "Record a sentence (or upload a clip) and it will be transcribed with "
            "Whisper, then analysed for sentiment.",
            icon="\N{MICROPHONE}",
        )
        return

    st.audio(audio)
    if not st.button("Transcribe & analyse", type="primary"):
        return

    try:
        with st.spinner("Transcribing…"):
            transcript = transcribe(audio, model_key=whisper_key, loader=load_asr_cached)
    except TranscriptionError as exc:
        st.error(f"{exc}", icon="\N{CROSS MARK}")
        return

    if transcript.skipped:
        st.warning(f"Nothing to analyse ({transcript.reason}).")
        return

    meta = [f"model={transcript.model}"]
    if transcript.language:
        meta.append(f"language={transcript.language}")
    if transcript.duration_seconds:
        meta.append(f"duration={transcript.duration_seconds:.1f}s")
        st.caption(" · ".join(meta))
        transcript_text = st.text_area("Transcript (editable)", value=transcript.text, height=100)
        if not transcript_text.strip():
            st.warning("The transcript is empty.")
            return

        result = engine.analyse(transcript_text, model=spec, include_logits=True)
        if not result.is_ok:
            st.error(f"Input rejected ({result.reason}).")
            return
        render_kpis(result)
        st.divider()
        left, right = st.columns([3, 2])
        with left:
            render_distribution(result, spec)
        with right:
            render_ranked_table(result)
        render_json(
            {"transcript": transcript.to_dict(), "analysis": result.to_dict()},
            f"speech-{spec.key}.json",
        )


# ---------------------------------------------------------------------------
# Tab 4: benchmarks
# ---------------------------------------------------------------------------


def render_benchmarks(engine: sentiment.SentimentEngine, model_key: str) -> None:
    spec = get_model_spec(model_key)
    st.subheader("Benchmarks")
    st.caption(
        "Measures the same code path the API uses: tokenise once per batch, one "
        "forward pass, temperature-scaled softmax."
    )
    size = st.slider("Sentences", 16, 512, 128, step=16)
    batch_size = st.slider("Batch size", 1, 128, engine.settings.batch_size)
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Benchmarking…"):
            corpus = [f"Product review sentence number {i}, quality varies." for i in range(size)]
            report = engine.benchmark(corpus, model=spec, batch_size=batch_size)
        st.session_state["benchmark"] = report.to_dict()
    report = st.session_state.get("benchmark")
    if not report:
        return
    columns = st.columns(4)
    columns[0].metric("Throughput", f"{report['texts_per_second']:,.0f} texts/s")
    columns[1].metric("p50 latency", f"{report['latency_ms_p50']:.1f} ms")
    columns[2].metric("p95 latency", f"{report['latency_ms_p95']:.1f} ms")
    columns[3].metric("Device", report["device"])
    st.json(report)


# ---------------------------------------------------------------------------
# Sidebar & entry point
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, bool]:
    """Sidebar controls shared by every tab."""
    st.sidebar.title("\N{ROBOT FACE} Sentiment Lab")
    st.sidebar.caption("Multi-model sentiment & emotion analysis")

    engine = load_engine()
    info = engine.info()
    keys = model_choices()
    default_index = keys.index(sentiment.DEFAULT_MODEL)
    model_key = st.sidebar.selectbox(
        "Model",
        keys,
        index=default_index,
        format_func=spec_label,
        help="Loaded lazily and cached; switching costs a one-off download.",
    )
    live = st.sidebar.toggle("Live analysis", value=False, help="Skip the button click.")

    st.sidebar.divider()
    st.sidebar.metric("Device", info["resolved_device"])
    st.sidebar.caption(
        f"CUDA available: {'yes' if info['cuda_available'] else 'no'}\n\n"
        f"Loaded: {', '.join(info['loaded_models']) or 'nothing yet'}\n\n"
        f"Batch size: {engine.settings.batch_size}\n"
        f"Temperature: {engine.settings.temperature:g}"
    )
    st.sidebar.divider()
    if st.sidebar.button("Unload models", help="Free RAM/VRAM"):
        engine.clear()
        st.sidebar.success("Models unloaded")
    st.sidebar.caption("Round-trip a text through the HTTP API from here to compare deployments.")
    with st.sidebar.expander("API smoke test"):
        host = st.text_input("API base URL", value="http://127.0.0.1:8000")
        probe = st.text_input("Probe text", value="I love this product")
        if st.button("POST /v1/analyze"):
            _api_probe(host, probe, model_key)
    return model_key, live


def _api_probe(host: str, text: str, model_key: str) -> None:
    try:
        import httpx
    except ImportError:
        st.warning("httpx is required for the API probe.")
        return
    try:
        response = httpx.post(
            f"{host.rstrip('/')}/v1/analyze",
            json={"text": text, "model": model_key},
            timeout=30.0,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(f"Request failed: {exc}", icon="\N{CROSS MARK}")
        return
    if response.status_code != 200:
        st.error(f"HTTP {response.status_code}: {response.text[:400]}")
        return
    payload = response.json()
    st.success(f"{response.status_code} in {payload['took_ms']:.1f} ms")
    st.json(payload)


def main() -> None:
    """Dashboard entry point."""
    inject_css()
    model_key, live = render_sidebar()
    engine = sentiment.get_engine()

    playground, batch_tab, audio_tab, bench_tab = st.tabs(
        ["Playground", "Batch files", "Audio", "Benchmarks"]
    )
    with playground:
        render_playground(engine, model_key, live)
    with batch_tab:
        render_batch(engine, model_key)
    with audio_tab:
        render_audio(engine, model_key)
    with bench_tab:
        render_benchmarks(engine, model_key)


if __name__ == "__main__":
    main()
