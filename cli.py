from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List

from sentiment import analyse
from utils import emoji_for_label, score_bar


def _read_interactive_input() -> str:
	"""Read multi-line input until the user types 'exit'."""
	print("Enter text to analyze sentiment (type 'exit' on a new line to finish):")

	lines: List[str] = []
	for line in sys.stdin:
		if line.strip().lower() == "exit":
			break
		lines.append(line.rstrip("\n"))
	return " ".join([l for l in lines if l.strip()]).strip()


def _parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Sentiment analysis CLI")
	parser.add_argument("-t", "--text", help="Text to analyze")
	parser.add_argument(
		"-i",
		"--interactive",
		action="store_true",
		help="Read multi-line input until 'exit'",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="sentiment_result.json",
		help="Path to write JSON output",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.6,
		help="Confidence threshold for warnings (0-1)",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		help="Logging level (DEBUG, INFO, WARNING, ERROR)",
	)
	return parser.parse_args()


def main() -> int:
	"""CLI entry point."""
	args = _parse_args()
	logging.basicConfig(level=args.log_level.upper())

	try:
		text = _read_interactive_input() if args.interactive else (args.text or "")
		if not text.strip():
			logging.error("Please provide text via --text or --interactive.")
			return 1

		label, score = analyse(text)
		emoji = emoji_for_label(label)
		bar = score_bar(score)

		if not 0.0 <= args.threshold <= 1.0:
			logging.warning("Threshold should be between 0 and 1. Using default 0.6.")
			args.threshold = 0.6
		if score < args.threshold:
			logging.warning("Low confidence result (score=%.2f).", score)

		data = {
			"text": text,
			"label": label,
			"score": score,
			"emoji": emoji,
		}
		with open(args.output, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=4, ensure_ascii=False)

		print(f"Sentiment: {label} {emoji} (score: {score:.2f})")
		print(f"Visualization: [{bar}]")
		return 0
	except ValueError as exc:
		logging.error(str(exc))
		return 1
	except KeyboardInterrupt:
		print("\nBye...!")
		return 130
	except Exception as exc:
		logging.exception("Unexpected error: %s", exc)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
