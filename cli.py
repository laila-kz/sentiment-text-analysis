"""Command line interface for the sentiment platform.

Examples
--------
    python cli.py --text "I love this project"
    python cli.py --text "not sure" --model emotion --temperature 1.5
    python cli.py --file reviews.csv --column review --output results.json
    python cli.py --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import batchio
import sentiment
from sentiment import (
    DEFAULT_MODEL,
    SentimentEngine,
    UnknownModelError,
    available_models,
    get_engine,
    get_model_spec,
)
from utils import emoji_for_label, score_bar, truncate

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

logger = logging.getLogger("cli")


def _read_interactive_input() -> list[str]:
    """Read multi-line input until the user types 'exit'."""
    print("Enter text to analyze sentiment (type 'exit' on a new line to finish):")
    lines: list[str] = []
    for line in sys.stdin:
        if line.strip().lower() == "exit":
            break
        lines.append(line.rstrip("\n"))
    text = " ".join(part for part in lines if part.strip()).strip()
    return [text] if text else []


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Sentiment analysis CLI (multi-model, batch capable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-t", "--text", help="Text to analyze")
    source.add_argument("-f", "--file", help="CSV/JSON file to analyze in batch")
    source.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Read multi-line input until 'exit'",
    )
    parser.add_argument("--column", help="Text column for --file (auto-detected when omitted)")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        choices=[spec.key for spec in available_models()],
        help="Model registry key",
    )
    parser.add_argument(
        "-o", "--output", help="Path to write JSON output (default: stdout summary)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature (>1 softens probabilities)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Texts per forward pass",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for warnings (0-1)",
    )
    parser.add_argument("--logits", action="store_true", help="Include raw logits in the payload")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    return parser.parse_args(argv)


def _print_result(result: sentiment.AnalysisResult, index: int | None = None) -> None:
    label = result.label or "n/a"
    emoji = emoji_for_label(label)
    prefix = "" if index is None else f"[{index}] "
    if not result.is_ok:
        print(f"{prefix}skipped ({result.reason}): {truncate(result.text, 60)}")
        return
    confidence = float(result.confidence or 0.0)
    print(
        f"{prefix}Sentiment: {label.title()} {emoji} (score: {confidence:.2f}) "
        f"[{score_bar(confidence)}]"
    )
    uncertainty = result.uncertainty
    if uncertainty is not None:
        print(
            f"{prefix}  uncertainty: entropy={uncertainty.entropy_bits:.3f} bits "
            f"(norm {uncertainty.normalized_entropy:.2f}), margin={uncertainty.margin:.2f}"
            + ("  ** low confidence **" if uncertainty.is_uncertain else "")
        )
    print(f"{prefix}  distribution: {result.distribution}")


def _collect_texts(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(texts, source_records)`` for the requested input mode."""
    if args.file:
        dataset = batchio.load_dataset(Path(args.file).read_bytes(), args.file)
        column = args.column or dataset.text_column
        if not column:
            raise ValueError(
                f"Could not detect a text column in {args.file}; pass --column explicitly."
            )
        dataset.text_column = column
        logger.info("Loaded %d rows from %s (column=%s)", len(dataset), args.file, column)
        return dataset.texts(), dataset.records
    if args.interactive:
        return _read_interactive_input(), []
    if args.text is not None:
        return [args.text], []
    raise ValueError("Please provide --text, --file or --interactive.")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    try:
        spec = get_model_spec(args.model)
        texts, records = _collect_texts(args)
        if not texts:
            logging.error("No text to analyze.")
            return EXIT_ERROR

        engine: SentimentEngine = get_engine()
        results = engine.analyse_batch(
            texts,
            model=spec,
            batch_size=args.batch_size or engine.settings.batch_size,
            temperature=args.temperature,
            include_logits=args.logits,
            # Single-text runs should fail loudly on empty input; batch files
            # report unusable rows as "skipped" so one bad row is not fatal.
            strict=args.file is None,
        )

        if not 0.0 <= args.threshold <= 1.0:
            logging.warning("Threshold must be between 0 and 1; using 0.6.")
            args.threshold = 0.6
        low_confidence = [
            result
            for result in results
            if result.is_ok and (result.confidence or 0.0) < args.threshold
        ]

        for index, result in enumerate(results):
            _print_result(result, index if len(results) > 1 else None)
        if low_confidence:
            logging.warning(
                "%d/%d result(s) below the confidence threshold %.2f.",
                len(low_confidence),
                len(results),
                args.threshold,
            )

        payload: dict[str, Any] = {
            "model": spec.to_dict(),
            "summary": sentiment.summarise(results).to_dict(),
            "results": [result.to_dict(include_logits=args.logits) for result in results],
        }
        if records:
            payload["records"] = batchio.annotate_records(records, results)

        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nWrote {len(results)} result(s) to {args.output}")
        elif len(results) > 1:
            print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
        return EXIT_OK
    except (ValueError, UnknownModelError, batchio.UnsupportedFormatError) as exc:
        logging.error("%s", exc)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        logging.error("File not found: %s", exc.filename)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nBye...!")
        return EXIT_INTERRUPTED
    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
