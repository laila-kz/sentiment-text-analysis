from __future__ import annotations

import logging
from typing import Any, Callable, Tuple

from utils import emoji_for_label, score_bar

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CACHE_DIR = "./model_cache"

# a pipeline bundles a model + preprocessing + postprocessing.
# pipeline = takes text → cleans it → sends to model → returns result.
classifier: Callable[[str], list[dict[str, Any]]] | None = None


def get_classifier() -> Callable[[str], list[dict[str, Any]]]:
    """Create or return the cached sentiment analysis pipeline."""
    global classifier
    if classifier is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        logger.info("Loading model '%s'...", MODEL_NAME)
        try:
            classifier = pipeline(
                "sentiment-analysis",
                model=MODEL_NAME,
                cache_dir=CACHE_DIR,
            )
        except Exception as exc:  # pragma: no cover - depends on external model download
            raise RuntimeError("Failed to load the sentiment model.") from exc
    return classifier


def analyse(text: str) -> Tuple[str, float]:
    """Analyze text and return label and score."""
    if not text or not text.strip():
        raise ValueError("Input text is empty.")
    classifier = get_classifier()  # loaded only once, even if called many times.
    result = classifier(text)[0]  # list of dicts, take the first
    label = result["label"].capitalize()  # 'POSITIVE' or 'NEGATIVE'
    score = float(result["score"])  # float between 0 and 1
    return label, score


def run_streamlit_app() -> None:
    """Run the Streamlit web UI."""
    import streamlit as st

    st.title("Sentiment Analysis with DistilBERT")
    text = st.text_area("Enter text to analyze sentiment:")
    if text:
        label, score = analyse(text)
        emoji = emoji_for_label(label)
        bar = score_bar(score)
        st.write(f"Sentiment: {label} {emoji} (score: {score:.2f})")
        st.write(f"Visualization: [{bar}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_streamlit_app()


