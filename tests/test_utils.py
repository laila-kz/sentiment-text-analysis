from utils import emoji_for_label, score_bar


def test_emoji_for_label() -> None:
    assert emoji_for_label("positive") == "😊"
    assert emoji_for_label("negative") == "😠"
    assert emoji_for_label("neutral") == "😶"


def test_score_bar() -> None:
    assert score_bar(1.0, width=5) == "█████"
    assert score_bar(0.0, width=5) == "░░░░░"
    assert score_bar(0.5, width=4) == "██░░"
    assert score_bar(-1.0, width=4) == "░░░░"
    assert score_bar(2.0, width=4) == "████"
