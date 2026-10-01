"""Unit tests for the pure helpers in :mod:`utils`."""

from __future__ import annotations

import math

import pytest

from utils import (
    clamp,
    detect_language_hint,
    emoji_for_label,
    guess_text_column,
    highlight_spans,
    is_meaningful_text,
    label_colour,
    margin_of_confidence,
    normalized_entropy,
    rejection_reason,
    score_bar,
    segment_text,
    shannon_entropy,
    softmax,
    top_k,
    truncate,
)


class TestPresentation:
    def test_emoji_for_sentiment_labels(self) -> None:
        assert emoji_for_label("positive") == "\N{GRINNING FACE WITH SMILING EYES}"
        assert emoji_for_label("Negative") == "\N{ANGRY FACE}"
        assert emoji_for_label("neutral") == "\N{NEUTRAL FACE}"
        assert emoji_for_label("totally unknown") == "\N{NEUTRAL FACE}"
        assert emoji_for_label("") == "\N{NEUTRAL FACE}"

    @pytest.mark.parametrize("label", ["sadness", "joy", "love", "anger", "fear", "surprise"])
    def test_emoji_for_every_emotion_label(self, label: str) -> None:
        emoji = emoji_for_label(label)
        assert emoji and emoji != "\N{NEUTRAL FACE}"

    def test_label_colour_fallback(self) -> None:
        assert label_colour("positive").startswith("#")
        assert label_colour("not-a-label") == label_colour(None)

    @pytest.mark.parametrize(
        ("score", "width", "expected"),
        [
            (1.0, 5, "\N{FULL BLOCK}" * 5),
            (0.0, 5, "\N{LIGHT SHADE}" * 5),
            (0.5, 4, "\N{FULL BLOCK}" * 2 + "\N{LIGHT SHADE}" * 2),
            (-1.0, 4, "\N{LIGHT SHADE}" * 4),
            (2.0, 4, "\N{FULL BLOCK}" * 4),
        ],
    )
    def test_score_bar(self, score: float, width: int, expected: str) -> None:
        assert score_bar(score, width) == expected

    def test_score_bar_zero_width(self) -> None:
        assert score_bar(0.5, 0) == ""
        assert score_bar(0.5, -3) == ""

    def test_clamp_and_truncate(self) -> None:
        assert clamp(1.5) == 1.0
        assert clamp(-0.2) == 0.0
        assert truncate("short", 10) == "short"
        assert truncate("a  b   c", 10) == "a b c"
        long = truncate("x" * 50, 10)
        assert len(long) == 10 and long.endswith("\N{HORIZONTAL ELLIPSIS}")
        assert truncate("abc", 0) == ""


class TestProbabilityMath:
    def test_softmax_sums_to_one(self) -> None:
        probabilities = softmax([1.0, 2.0, 3.0])
        assert pytest.approx(sum(probabilities)) == 1.0
        assert probabilities == sorted(probabilities)

    def test_softmax_is_stable_for_extreme_logits(self) -> None:
        probabilities = softmax([1000.0, -1000.0])
        assert pytest.approx(sum(probabilities)) == 1.0
        assert probabilities[0] == pytest.approx(1.0)

    def test_softmax_temperature_flattens_distribution(self) -> None:
        sharp = softmax([2.0, 1.0], temperature=0.5)
        soft = softmax([2.0, 1.0], temperature=3.0)
        assert sharp[0] > soft[0] > 0.5

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
    def test_softmax_rejects_invalid_temperature(self, temperature: float) -> None:
        with pytest.raises(ValueError):
            softmax([1.0, 2.0], temperature=temperature)

    def test_softmax_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            softmax([])

    def test_entropy_bounds(self) -> None:
        assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
        assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
        assert shannon_entropy([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(math.log2(3))

    def test_normalized_entropy_is_bounded(self) -> None:
        assert normalized_entropy([1.0, 0.0]) == 0.0
        assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
        assert normalized_entropy([0.9, 0.1]) == pytest.approx(0.469, abs=1e-3)
        assert normalized_entropy([]) == 0.0
        assert normalized_entropy([1.0]) == 0.0

    def test_entropy_rejects_bad_base(self) -> None:
        with pytest.raises(ValueError):
            shannon_entropy([0.5, 0.5], base=1.0)

    def test_margin_and_top_k(self) -> None:
        assert margin_of_confidence([0.8, 0.2, 0.0]) == pytest.approx(0.6)
        assert margin_of_confidence([0.5]) == pytest.approx(0.5)
        assert margin_of_confidence([]) == 0.0
        ranked = top_k([0.1, 0.7, 0.2], k=2)
        assert [index for index, _ in ranked] == [1, 2]
        assert [score for _, score in ranked] == pytest.approx([0.7, 0.2])
        assert top_k([0.1, 0.7], k=0) == []


class TestTextSanity:
    @pytest.mark.parametrize("value", [None, "", "   ", "\t\n", "\u200b"])
    def test_blank_text_is_not_meaningful(self, value: str | None) -> None:
        assert is_meaningful_text(value) is False

    @pytest.mark.parametrize(
        "value", ["\U0001f600", "\U0001f600\U0001f601", "!!!", "...???", "\u2764\ufe0f"]
    )
    def test_emoji_or_punctuation_only_is_not_meaningful(self, value: str) -> None:
        assert is_meaningful_text(value) is False

    @pytest.mark.parametrize(
        "value", ["hello", "gorgeous!", "5 stars", "caf\u00e9", "\u4f60\u597d"]
    )
    def test_real_text_is_meaningful(self, value: str) -> None:
        assert is_meaningful_text(value) is True

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            (None, "empty_input"),
            ("", "empty_input"),
            ("   ", "empty_input"),
            ("\u200b\ufeff", "blank_input"),
            ("\U0001f600", "no_word_characters"),
            ("???", "no_word_characters"),
            ("hello", None),
        ],
    )
    def test_rejection_reason(self, value: str | None, reason: str | None) -> None:
        assert rejection_reason(value) == reason

    def test_language_hint(self) -> None:
        assert detect_language_hint("hello") == "latin"
        assert detect_language_hint("\u043f\u0440\u0438\u0432\u0435\u0442") == "cyrillic"
        assert detect_language_hint("\u4f60\u597d") == "cjk"
        assert detect_language_hint("\u0645\u0631\u062d\u0628\u0627") == "arabic"
        assert detect_language_hint("123") == "unknown"


class TestSegmentationAndHighlighting:
    def test_segment_text_splits_sentences(self) -> None:
        chunks = segment_text("I love it. It is awful!  Maybe?")
        assert chunks == ["I love it.", "It is awful!", "Maybe?"]

    def test_segment_text_splits_long_paragraphs(self) -> None:
        text = "word " * 100
        chunks = segment_text(text, max_chars=40)
        assert len(chunks) > 1
        assert all(len(chunk) <= 40 for chunk in chunks)

    def test_highlight_spans_marks_each_segment(self) -> None:
        markup = highlight_spans("I love it. It is awful.", ["positive", "negative"], [0.9, 0.8])
        assert "<span" in markup
        assert "positive" in markup and "negative" in markup
        assert "I love it." in markup

    def test_highlight_spans_escapes_html(self) -> None:
        markup = highlight_spans("<script>alert(1)</script>", ["positive"], [0.9])
        assert "<script>" not in markup
        assert "&lt;script&gt;" in markup

    def test_highlight_spans_tolerates_length_mismatch(self) -> None:
        markup = highlight_spans("One. Two.", ["positive"], [0.9])
        assert "Two." in markup


class TestColumnDetection:
    def test_picks_column_with_name_hint(self) -> None:
        rows = [{"id": 1, "review": "this is a lovely long review", "stars": 5}]
        assert guess_text_column(["id", "review", "stars"], rows) == "review"

    def test_picks_longest_text_column_without_hint(self) -> None:
        rows = [{"a": "x", "b": "an extremely long sentence that is clearly free text"}]
        assert guess_text_column(["a", "b"], rows) == "b"

    def test_prefers_spaced_values(self) -> None:
        rows = [{"alpha": "one two three four", "beta": "abcdefghijkl"}]
        assert guess_text_column(["alpha", "beta"], rows) == "alpha"

    def test_penalises_id_columns(self) -> None:
        rows = [{"user_id": "12345", "text": "great"}]
        assert guess_text_column(["user_id", "text"], rows) == "text"

    def test_returns_none_without_columns(self) -> None:
        assert guess_text_column([]) is None

    def test_works_without_sample_rows(self) -> None:
        assert guess_text_column(["comment"]) == "comment"
