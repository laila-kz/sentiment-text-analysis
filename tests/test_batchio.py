"""Tests for CSV/JSON ingestion and annotation."""

from __future__ import annotations

import json
from typing import Any

import pytest

import batchio
from batchio import UnsupportedFormatError, load_dataset
from sentiment import SentimentEngine

CSV_SAMPLE = 'id,review,rating\n1,"good",5\n2,bad,1\n'

JSON_OBJECTS = '[{"comment": "great job"}, {"comment": "awful job"}]'
JSON_STRINGS = '["happy day", "sad day"]'


class TestLoadDataset:
    def test_csv_with_header(self) -> None:
        dataset = load_dataset(CSV_SAMPLE, "reviews.csv")
        assert len(dataset) == 2
        assert dataset.columns == ["id", "review", "rating"]
        assert dataset.text_column == "review"
        assert dataset.texts() == ["good", "bad"]
        assert dataset.source == "reviews.csv"

    def test_csv_from_bytes_strips_bom(self) -> None:
        dataset = load_dataset(CSV_SAMPLE.encode("utf-8-sig"), "reviews.csv")
        assert dataset.columns[0] == "id"

    def test_tsv_is_detected(self) -> None:
        dataset = load_dataset("title\ttext\nA\tloved it\nB\thated it\n", "data.tsv")
        assert dataset.columns == ["title", "text"]
        assert dataset.text_column == "text"

    def test_semicolon_delimited(self) -> None:
        dataset = load_dataset("review;score\nloved it;5\n", "euro.csv")
        assert dataset.columns == ["review", "score"]
        assert dataset.texts() == ["loved it"]

    def test_json_array_of_objects(self) -> None:
        dataset = load_dataset(JSON_OBJECTS, "reviews.json")
        assert dataset.text_column == "comment"
        assert dataset.texts() == ["great job", "awful job"]

    def test_json_array_of_strings(self) -> None:
        dataset = load_dataset(JSON_STRINGS, "list.json")
        assert dataset.texts() == ["happy day", "sad day"]
        assert dataset.text_column == "text"

    def test_json_wrapper_object(self) -> None:
        dataset = load_dataset('{"records": [{"text": "wrapped"}]}', "wrapper.json")
        assert dataset.texts() == ["wrapped"]

    def test_json_single_object(self) -> None:
        dataset = load_dataset('{"text": "lonely"}', "single.json")
        assert dataset.texts() == ["lonely"]

    def test_json_detected_without_extension(self) -> None:
        assert load_dataset(JSON_OBJECTS).text_column == "comment"

    def test_duplicate_headers_are_made_unique(self) -> None:
        dataset = load_dataset("a,a\n1,2\n", "dupes.csv")
        assert dataset.columns == ["a", "a_2"]

    def test_short_rows_are_padded(self) -> None:
        dataset = load_dataset("a,b,c\n1,2\n", "short.csv")
        assert dataset.records[0] == {"a": "1", "b": "2", "c": ""}

    def test_blank_lines_are_dropped(self) -> None:
        dataset = load_dataset("text\nhello\n\n\nworld\n", "blanks.csv")
        assert len(dataset) == 2

    def test_unsupported_extension(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="Unsupported file type"):
            load_dataset("binary", "data.xlsx")

    def test_empty_file(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="empty"):
            load_dataset(b"", "empty.csv")

    def test_header_only_csv(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="No data rows"):
            load_dataset("a,b\n", "header.csv")

    def test_invalid_json(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="Invalid JSON"):
            load_dataset("{not json", "broken.json")

    def test_non_object_array_values_are_stringified(self) -> None:
        dataset = load_dataset("[1, 2]", "numbers.json")
        assert dataset.texts() == ["1", "2"]

    def test_texts_without_a_column(self) -> None:
        dataset = load_dataset("x\nhello\n", "one.csv")
        dataset.text_column = None
        assert dataset.texts() == [""]


class TestAnnotation:
    def test_annotate_preserves_source_columns(self, engine: SentimentEngine) -> None:
        dataset = load_dataset(CSV_SAMPLE, "reviews.csv")
        results = engine.analyse_batch(dataset.texts())
        annotated = batchio.annotate_records(dataset.records, results)
        assert len(annotated) == 2
        first = annotated[0]
        assert first["id"] == "1"
        assert first["review"] == "good"
        assert first["sentiment_label"] == "positive"
        assert first["sentiment_status"] == "ok"
        assert first["sentiment_reason"] == ""
        assert json.loads(first["sentiment_scores"])
        assert 0.0 <= first["sentiment_entropy"] <= 1.0

    def test_annotate_handles_skipped_rows(self, engine: SentimentEngine) -> None:
        records = [{"text": ""}, {"text": "great"}]
        results = engine.analyse_batch([row["text"] for row in records])
        annotated = batchio.annotate_records(records, results)
        assert annotated[0]["sentiment_status"] == "skipped"
        assert annotated[0]["sentiment_reason"] == "empty_input"
        assert annotated[0]["sentiment_label"] is None

    def test_annotate_with_prefix(self, engine: SentimentEngine) -> None:
        results = engine.analyse_batch(["great"])
        annotated = batchio.annotate_records([{"text": "great"}], results, prefix="api")
        assert "api_sentiment_label" in annotated[0]

    def test_annotate_ignores_extra_results(self, engine: SentimentEngine) -> None:
        results = engine.analyse_batch(["great", "bad"])
        assert len(batchio.annotate_records([{"text": "great"}], results)) == 1

    def test_build_distribution_orders_by_frequency(self, engine: SentimentEngine) -> None:
        results = engine.analyse_batch(["good", "good", "bad", ""])
        counts = batchio.build_distribution(results)
        assert sum(counts.values()) == 3
        assert list(counts) == sorted(counts, key=lambda key: (-counts[key], key))


class TestSerialisation:
    def test_csv_roundtrip(self) -> None:
        records: list[dict[str, Any]] = [{"a": 1, "b": "x"}, {"c": [1, 2]}]
        rendered = batchio.to_csv(records)
        header, *rows = rendered.strip().splitlines()
        assert header == "a,b,c"
        assert '"[1, 2]"' in rows[1]

    def test_csv_of_empty_records(self) -> None:
        assert batchio.to_csv([]) == ""

    def test_json_roundtrip(self) -> None:
        records = [{"a": 1, "b": None}]
        assert json.loads(batchio.to_json(records)) == records

    def test_annotated_csv_is_reloadable(self, engine: SentimentEngine) -> None:
        dataset = load_dataset(CSV_SAMPLE, "reviews.csv")
        results = engine.analyse_batch(dataset.texts())
        annotated = batchio.annotate_records(dataset.records, results)
        reloaded = load_dataset(batchio.to_csv(annotated), "annotated.csv")
        assert len(reloaded) == 2
        assert "sentiment_label" in reloaded.columns
