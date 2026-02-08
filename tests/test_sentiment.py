import pytest

import sentiment


def test_analyse_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_classifier(_: str):
        return [{"label": "POSITIVE", "score": 0.9}]

    monkeypatch.setattr(sentiment, "get_classifier", lambda: fake_classifier)
    label, score = sentiment.analyse("Great work")

    assert label == "Positive"
    assert score == 0.9


def test_analyse_empty_input() -> None:
    with pytest.raises(ValueError):
        sentiment.analyse("   ")
