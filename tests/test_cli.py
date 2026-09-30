"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cli
from sentiment import SentimentEngine


class TestArgumentParsing:
    def test_text_only(self) -> None:
        args = cli._parse_args(["--text", "hello"])
        assert args.text == "hello"
        assert args.model == "sentiment"
        assert args.output is None

    def test_short_flags(self) -> None:
        args = cli._parse_args(["-t", "hi", "-m", "emotion", "-o", "out.json"])
        assert (args.text, args.model, args.output) == ("hi", "emotion", "out.json")

    def test_sources_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_args(["--text", "a", "--file", "b.csv"])
        with pytest.raises(SystemExit):
            cli._parse_args(["--text", "a", "--interactive"])

    def test_unknown_model_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_args(["--text", "a", "--model", "nope"])

    def test_defaults(self) -> None:
        args = cli._parse_args([])
        assert args.threshold == 0.6
        assert args.temperature is None
        assert args.batch_size is None
        assert args.logits is False


class TestMainSingleText:
    def test_prints_label_and_score(
        self, engine: SentimentEngine, capsys: pytest.CaptureFixture
    ) -> None:
        assert cli.main(["--text", "i love this"]) == cli.EXIT_OK
        output = capsys.readouterr().out
        assert "Sentiment: Positive" in output
        assert "uncertainty:" in output
        assert "distribution:" in output

    def test_low_confidence_warns(
        self,
        engine: SentimentEngine,
        capsys: pytest.CaptureFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("WARNING"):
            assert cli.main(["--text", "meh", "--threshold", "0.99"]) == cli.EXIT_OK
        assert "below the confidence threshold" in caplog.text

    def test_invalid_threshold_is_corrected(
        self, engine: SentimentEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            assert cli.main(["--text", "good", "--threshold", "5"]) == cli.EXIT_OK
        assert "must be between 0 and 1" in caplog.text

    def test_emotion_model(self, engine: SentimentEngine, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["--text", "i love this", "--model", "emotion"]) == cli.EXIT_OK
        assert "love" in capsys.readouterr().out

    def test_writes_json_payload(
        self, engine: SentimentEngine, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        target = tmp_path / "result.json"
        assert cli.main(["--text", "i love this", "-o", str(target), "--logits"]) == cli.EXIT_OK
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["model"]["key"] == "sentiment"
        assert payload["results"][0]["label"] == "positive"
        assert payload["results"][0]["logits"] is not None
        assert payload["summary"]["analysed"] == 1
        assert "Wrote 1 result(s)" in capsys.readouterr().out

    def test_empty_text_is_an_error(self, engine: SentimentEngine) -> None:
        assert cli.main(["--text", "   "]) == cli.EXIT_ERROR

    def test_missing_input_is_an_error(self, engine: SentimentEngine) -> None:
        assert cli.main([]) == cli.EXIT_ERROR


class TestMainBatch:
    def test_batch_from_file(
        self, engine: SentimentEngine, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        source = tmp_path / "reviews.csv"
        source.write_text("review\ni love this\nterrible and useless\n", encoding="utf-8")
        target = tmp_path / "out.json"
        assert cli.main(["--file", str(source), "-o", str(target)]) == cli.EXIT_OK
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert [item["label"] for item in payload["results"]] == ["positive", "negative"]
        assert [item["review"] for item in payload["records"]] == [
            "i love this",
            "terrible and useless",
        ]
        assert payload["records"][0]["sentiment_label"] == "positive"

    def test_explicit_column(self, engine: SentimentEngine, tmp_path: Path) -> None:
        source = tmp_path / "reviews.tsv"
        source.write_text("body\tid\ni love this\t7\n", encoding="utf-8")
        assert cli.main(["--file", str(source), "--column", "body"]) == cli.EXIT_OK

    def test_missing_file(self, engine: SentimentEngine) -> None:
        assert cli.main(["--file", "does-not-exist.csv"]) == cli.EXIT_ERROR

    def test_unsupported_file(self, engine: SentimentEngine, tmp_path: Path) -> None:
        source = tmp_path / "data.xlsx"
        source.write_bytes(b"binary")
        assert cli.main(["--file", str(source)]) == cli.EXIT_ERROR

    def test_batch_size_is_respected(self, engine: SentimentEngine, tmp_path: Path, loader) -> None:
        source = tmp_path / "many.csv"
        source.write_text("text\n" + "good\n" * 5, encoding="utf-8")
        assert cli.main(["--file", str(source), "--batch-size", "2"]) == cli.EXIT_OK
        assert loader.pipeline("sentiment").batch_sizes == [2, 2, 1]


class TestInteractiveMode:
    def test_reads_until_exit(
        self,
        engine: SentimentEngine,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("i love this\nexit\n"))
        assert cli.main(["--interactive"]) == cli.EXIT_OK
        assert "Sentiment: Positive" in capsys.readouterr().out

    def test_empty_interactive_input(
        self, engine: SentimentEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("exit\n"))
        assert cli.main(["--interactive"]) == cli.EXIT_ERROR

    def test_keyboard_interrupt(
        self, engine: SentimentEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_interrupt() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_read_interactive_input", raise_interrupt)
        assert cli.main(["--interactive"]) == cli.EXIT_INTERRUPTED
