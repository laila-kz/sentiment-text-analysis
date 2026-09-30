"""Batch file ingestion and annotation for the dashboard and CLI.

Pure stdlib on purpose: CSV/JSON parsing must work in environments without
``pandas`` (CI, minimal images) while the UI is free to render the resulting
records however it likes.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sentiment import AnalysisResult
from utils import emoji_for_label, guess_text_column

__all__ = [
    "BatchDataset",
    "UnsupportedFormatError",
    "annotate_records",
    "build_distribution",
    "load_dataset",
    "to_csv",
    "to_json",
]

MAX_ROWS = 100_000


class UnsupportedFormatError(ValueError):
    """Raised when an upload is neither CSV nor JSON (or is malformed)."""


@dataclass
class BatchDataset:
    """Rows loaded from an uploaded file plus the detected text column."""

    records: list[dict[str, Any]] = field(default_factory=list)
    text_column: str | None = None
    source: str = ""
    columns: list[str] = field(default_factory=list)
    skipped_rows: int = 0

    def __len__(self) -> int:
        return len(self.records)

    def texts(self) -> list[str]:
        """Text values of the detected column (empty string when missing)."""
        if self.text_column is None:
            return ["" for _ in self.records]
        return [_stringify(record.get(self.text_column)) for record in self.records]

    def column_options(self) -> list[str]:
        return list(self.columns)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_dataset(payload: bytes | str, filename: str = "") -> BatchDataset:
    """Parse an uploaded CSV/JSON payload into a :class:`BatchDataset`.

    JSON is accepted as a list of objects, a list of strings, or an object with a
    single list-of-objects/strings value.
    """
    text = payload.decode("utf-8-sig", errors="replace") if isinstance(payload, bytes) else payload
    if not text.strip():
        raise UnsupportedFormatError("The uploaded file is empty.")

    lowered = filename.lower()
    suffix = lowered.rsplit(".", 1)[-1] if "." in lowered else ""

    if suffix == "json" or text.lstrip()[:1] in "[{":
        records = _load_json(text, filename)
    elif suffix in {"csv", "tsv", "txt", ""}:
        records = _load_csv(text)
    else:
        raise UnsupportedFormatError(
            f"Unsupported file type {suffix!r}. Upload a .csv, .tsv or .json file."
        )

    if not records:
        raise UnsupportedFormatError("No data rows found in the uploaded file.")
    if len(records) > MAX_ROWS:
        records = records[:MAX_ROWS]

    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(str(key))

    return BatchDataset(
        records=records,
        text_column=guess_text_column(columns, records),
        source=filename or "upload",
        columns=columns,
    )


def _load_csv(text: str) -> list[dict[str, Any]]:
    sample = text[:4096]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else _sniff_delimiter(sample)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        return []
    header = [str(cell).strip() or f"column_{i}" for i, cell in enumerate(header)]
    if len(set(header)) != len(header):
        seen: dict[str, int] = {}
        unique: list[str] = []
        for cell in header:
            seen[cell] = seen.get(cell, 0) + 1
            unique.append(cell if seen[cell] == 1 else f"{cell}_{seen[cell]}")
        header = unique
    rows = [row for row in reader if row]
    width = len(header)
    records: list[dict[str, Any]] = []
    for row in rows:
        if not any(str(cell).strip() for cell in row):
            continue
        padded = list(row) + [""] * (width - len(row))
        records.append(dict(zip(header, padded[:width], strict=False)))
    return records


def _load_json(text: str, filename: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UnsupportedFormatError(f"Invalid JSON in {filename or 'upload'}: {exc}") from exc

    if isinstance(payload, dict):
        # A single wrapper object such as {"records": [...]} / {"data": [...]}
        list_values = [value for value in payload.values() if isinstance(value, list)]
        payload = list_values[0] if len(list_values) == 1 else [payload]
    if not isinstance(payload, list):
        raise UnsupportedFormatError("JSON must be an array of objects or strings.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if isinstance(item, dict):
            records.append({str(key): value for key, value in item.items()})
        elif isinstance(item, str):
            records.append({"text": item})
        else:
            records.append({"text": item, "_index": index})
    return records


# ---------------------------------------------------------------------------
# Annotation / export
# ---------------------------------------------------------------------------

#: Columns appended to the uploaded data by :func:`annotate_records`.
ANNOTATION_COLUMNS = (
    "sentiment_label",
    "sentiment_confidence",
    "sentiment_scores",
    "sentiment_entropy",
    "sentiment_status",
    "sentiment_reason",
)


def annotate_records(
    records: Sequence[dict[str, Any]],
    results: Iterable[AnalysisResult],
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Merge results into the original records, preserving source columns."""
    annotated: list[dict[str, Any]] = []
    for record, result in zip(records, results, strict=False):
        payload = dict(record)
        key = f"{prefix}_" if prefix else ""
        label = result.label
        payload[f"{key}sentiment_label"] = label
        payload[f"{key}sentiment_confidence"] = result.confidence
        payload[f"{key}sentiment_scores"] = json.dumps(result.distribution, ensure_ascii=False)
        payload[f"{key}sentiment_entropy"] = (
            result.uncertainty.normalized_entropy if result.uncertainty else None
        )
        payload[f"{key}sentiment_status"] = result.status
        payload[f"{key}sentiment_reason"] = result.reason or ""
        payload[f"{key}sentiment_emoji"] = emoji_for_label(label) if label else ""
        annotated.append(payload)
    return annotated


def build_distribution(results: Sequence[AnalysisResult]) -> dict[str, int]:
    """Count how often each label was predicted."""
    counts: dict[str, int] = {}
    for result in results:
        if not result.is_ok or result.label is None:
            continue
        counts[result.label] = counts.get(result.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def to_csv(records: Sequence[dict[str, Any]]) -> str:
    """Serialise records to CSV text (union of all keys, stable ordering)."""
    if not records:
        return ""
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(str(key))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({key: _csv_value(record.get(key)) for key in columns})
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "" if value is None else value


def to_json(records: Sequence[dict[str, Any]], *, indent: int = 2) -> str:
    """Serialise records to JSON text."""
    return json.dumps(records, indent=indent, ensure_ascii=False, default=str)
