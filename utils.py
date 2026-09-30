"""Small, dependency-free helpers shared by the engine, the API and the UI.

Everything in this module is pure Python (stdlib only) so that it can be
imported and unit-tested without ``torch``/``transformers`` being installed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

__all__ = [
    "EMOJI_BY_LABEL",
    "LABEL_COLOURS",
    "clamp",
    "detect_language_hint",
    "emoji_for_label",
    "guess_text_column",
    "highlight_spans",
    "is_meaningful_text",
    "label_colour",
    "margin_of_confidence",
    "normalized_entropy",
    "rejection_reason",
    "score_bar",
    "segment_text",
    "shannon_entropy",
    "softmax",
    "top_k",
    "truncate",
]

# ---------------------------------------------------------------------------
# Labels & presentation
# ---------------------------------------------------------------------------

#: Emoji used for every label produced by the bundled model registry.
EMOJI_BY_LABEL: Final[Mapping[str, str]] = {
    "positive": "\N{GRINNING FACE WITH SMILING EYES}",
    "negative": "\N{ANGRY FACE}",
    "neutral": "\N{NEUTRAL FACE}",
    "label_0": "\N{NEUTRAL FACE}",
    "label_1": "\N{GRINNING FACE WITH SMILING EYES}",
    "label_2": "\N{SLIGHTLY FROWNING FACE}",
    "sadness": "\N{LOUDLY CRYING FACE}",
    "joy": "\N{GRINNING FACE WITH SMILING EYES}",
    "love": "\N{SMILING FACE WITH HEART-SHAPED EYES}",
    "anger": "\N{POUTING FACE}",
    "fear": "\N{FEARFUL FACE}",
    "surprise": "\N{FACE SCREAMING IN FEAR}",
    "optimism": "\N{SMILING FACE WITH SUNGLASSES}",
    "admiration": "\N{MILITARY MEDAL}",
    "annoyance": "\N{FACE WITH ROLLING EYES}",
    "confusion": "\N{CONFUSED FACE}",
}

DEFAULT_EMOJI: Final[str] = "\N{NEUTRAL FACE}"

#: HTML colours used by :func:`highlight_spans` and the dashboard charts.
LABEL_COLOURS: Final[Mapping[str, str]] = {
    "positive": "#1a7f37",
    "joy": "#1a7f37",
    "love": "#c2185b",
    "optimism": "#2e7d32",
    "admiration": "#00796b",
    "negative": "#b3261e",
    "anger": "#b3261e",
    "annoyance": "#b3261e",
    "sadness": "#1565c0",
    "fear": "#6a1b9a",
    "surprise": "#ef6c00",
    "neutral": "#5f6368",
    "confusion": "#5f6368",
}

DEFAULT_COLOUR: Final[str] = "#5f6368"


def label_colour(label: str | None) -> str:
    """Colour associated with a label (falls back to a neutral grey)."""
    return LABEL_COLOURS.get((label or "").strip().lower(), DEFAULT_COLOUR)


def emoji_for_label(label: str) -> str:
    """Return a single emoji that represents ``label`` (falls back to neutral)."""
    return EMOJI_BY_LABEL.get(label.strip().lower(), DEFAULT_EMOJI)


def score_bar(score: float, width: int = 10) -> str:
    """Build a text bar representing a score between 0 and 1."""
    if width <= 0:
        return ""
    clamped = clamp(score)
    filled = int(round(clamped * width))
    return "\N{FULL BLOCK}" * filled + "\N{LIGHT SHADE}" * (width - filled)


def truncate(text: str, limit: int = 280, suffix: str = "\N{HORIZONTAL ELLIPSIS}") -> str:
    """Shorten ``text`` to ``limit`` characters, appending ``suffix`` when cut."""
    if limit <= 0:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - len(suffix))].rstrip() + suffix


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Constrain ``value`` to the inclusive ``[low, high]`` range."""
    return max(low, min(high, float(value)))


# ---------------------------------------------------------------------------
# Probability maths: softmax, entropy, margin
# ---------------------------------------------------------------------------


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax, optionally temperature-scaled.

    ``temperature > 1`` softens the distribution (useful when a model is
    over-confident), ``temperature < 1`` sharpens it.
    """
    if not logits:
        raise ValueError("logits must not be empty")
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0:
        raise ValueError("temperature must be a positive finite number")

    scaled = [float(value) / temp for value in logits]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    if total <= 0:  # pragma: no cover - unreachable for finite logits
        raise ValueError("softmax overflow; check that logits are finite")
    return [value / total for value in exps]


def shannon_entropy(probabilities: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy of a distribution (bits by default).

    Entropy is ``0.0`` when the model is fully certain and ``log2(K)`` when it
    is uniform across ``K`` labels.
    """
    if base <= 1:
        raise ValueError("base must be greater than 1")
    total = 0.0
    for probability in probabilities:
        if probability <= 0.0:
            continue
        total -= probability * math.log(probability, base)
    return max(0.0, total)


def normalized_entropy(probabilities: Sequence[float], base: float = 2.0) -> float:
    """Entropy rescaled to ``[0, 1]`` where ``1.0`` means "maximally unsure"."""
    if not probabilities:
        return 0.0
    ceiling = math.log(len(probabilities), base)
    if ceiling <= 0:
        return 0.0
    return clamp(shannon_entropy(probabilities, base) / ceiling)


def margin_of_confidence(probabilities: Sequence[float]) -> float:
    """Gap between the best and the runner-up probability."""
    if len(probabilities) < 2:
        return float(probabilities[0]) if probabilities else 0.0
    ordered = sorted((float(p) for p in probabilities), reverse=True)
    return max(0.0, ordered[0] - ordered[1])


def top_k(probabilities: Sequence[float], k: int = 2) -> list[tuple[int, float]]:
    """Return up to ``k`` ``(index, probability)`` pairs, highest first."""
    if k <= 0:
        return []
    ordered = sorted(enumerate(probabilities), key=lambda pair: pair[1], reverse=True)
    return [(index, float(value)) for index, value in ordered[:k]]


# ---------------------------------------------------------------------------
# Text sanity checks
# ---------------------------------------------------------------------------

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\ufeff]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WORD = re.compile(r"\w+", re.UNICODE)


def is_meaningful_text(text: str | None) -> bool:
    """``True`` when ``text`` contains at least one word character.

    Rejects ``None``, empty/blank strings, punctuation-only and emoji-only
    strings, all of which produce meaningless model output.
    """
    if not text:
        return False
    cleaned = _ZERO_WIDTH.sub("", text)
    if not cleaned.strip():
        return False
    if _WORD.search(cleaned) is None:
        return False
    # At least one letter/number: digits only or symbols only are still usable,
    # but pure "variation selector"/combining-mark text is not.
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in cleaned)


def rejection_reason(text: str | None) -> str | None:
    """Explain why ``text`` cannot be analysed, or ``None`` when it is fine.

    Returns ``"empty_input"`` for ``None``/empty strings, ``"blank_input"`` for
    whitespace-only strings and ``"no_word_characters"`` when nothing analysable
    remains (punctuation or emoji only).
    """
    if text is None or not text.strip():
        return "empty_input"
    cleaned = _ZERO_WIDTH.sub("", text)
    if not cleaned.strip():
        return "blank_input"
    if _WORD.search(cleaned) is None:
        return "no_word_characters"
    if not any(unicodedata.category(char)[0] in {"L", "N"} for char in cleaned):
        return "no_word_characters"
    return None


def detect_language_hint(text: str) -> str:
    """Very rough script-based hint: ``latin``, ``cyrillic``, ``cjk`` ..."""
    if not text:
        return "unknown"
    seen_ascii_letter = False
    for char in text:
        code = ord(char)
        if 0x0400 <= code <= 0x04FF:
            return "cyrillic"
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
            return "cjk"
        if 0x0600 <= code <= 0x06FF:
            return "arabic"
        if 0x0370 <= code <= 0x03FF:
            return "greek"
        if char.isalpha():
            seen_ascii_letter = True
    return "latin" if seen_ascii_letter else "unknown"


# ---------------------------------------------------------------------------
# Presentation helpers used by the Streamlit playground
# ---------------------------------------------------------------------------


def segment_text(text: str, max_chars: int = 160) -> list[str]:
    """Split ``text`` into small chunks for per-span sentiment highlighting."""
    chunks: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def highlight_spans(text: str, labels: Sequence[str], scores: Sequence[float]) -> str:
    """Render annotated Markdown/HTML for ``text`` given per-segment results.

    ``labels``/``scores`` are aligned with :func:`segment_text`. Segments the
    model could not read are left untouched, and HTML is escaped so arbitrary
    input cannot inject markup.
    """
    import html

    chunks = segment_text(text)
    if not chunks:
        return html.escape(text)
    if len(labels) != len(chunks) or len(scores) != len(chunks):
        labels = list(labels) + ["neutral"] * (len(chunks) - len(labels))
        scores = list(scores) + [0.0] * (len(chunks) - len(scores))

    parts: list[str] = []
    for chunk, label, score in zip(chunks, labels, scores, strict=False):
        key = (label or "neutral").lower()
        colour = label_colour(key)
        safe = html.escape(chunk)
        parts.append(
            f'<span style="color:{colour};font-weight:600" '
            f'title="{html.escape(key)} {clamp(score):.0%}">{safe}</span>'
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Batch-file column heuristics (used for automatic column mapping)
# ---------------------------------------------------------------------------

_TEXT_COLUMN_HINTS: Final[tuple[str, ...]] = (
    "text",
    "review",
    "comment",
    "content",
    "body",
    "message",
    "feedback",
    "sentence",
    "tweet",
    "post",
    "description",
    "note",
    "answer",
    "response",
    "headline",
    "summary",
    "query",
    "prompt",
    "input",
)

_ID_COLUMN_HINTS: Final[tuple[str, ...]] = ("id", "index", "uuid", "key", "row", "order")


def guess_text_column(
    columns: Sequence[str],
    sample_rows: Iterable[Mapping[str, object]] = (),
    max_rows: int = 50,
) -> str | None:
    """Pick the column most likely to hold free text.

    Scoring combines (in order) name hints, average string length and the
    fraction of values that contain whitespace. Returns ``None`` when there is
    no candidate at all.
    """
    available = [column for column in columns if column is not None]
    if not available:
        return None

    rows = list(sample_rows)[:max_rows]
    best_column: str | None = None
    best_score = float("-inf")

    for column in available:
        lowered = str(column).strip().lower()
        score = 0.0
        if any(hint in lowered for hint in _ID_COLUMN_HINTS):
            score -= 5.0
        if any(hint == lowered for hint in _TEXT_COLUMN_HINTS):
            score += 6.0
        elif any(hint in lowered for hint in _TEXT_COLUMN_HINTS):
            score += 3.0

        values = [row.get(column) for row in rows]
        strings = [str(value) for value in values if value is not None and str(value).strip()]
        if strings:
            avg_len = sum(len(value) for value in strings) / len(strings)
            score += min(avg_len / 40.0, 3.0)
            spaced = sum(1 for value in strings if " " in value) / len(strings)
            score += spaced * 2.0
            score += sum(1 for value in strings if is_meaningful_text(value)) / len(strings)

        if score > best_score:
            best_score, best_column = score, column

    return best_column
