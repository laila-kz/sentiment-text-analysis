from __future__ import annotations


def emoji_for_label(label: str) -> str:
	"""Return an emoji for a sentiment label."""
	normalized = label.strip().capitalize()
	return {"Positive": "😊", "Negative": "😠"}.get(normalized, "😶")


def score_bar(score: float, width: int = 10) -> str:
	"""Build a text bar representing a score between 0 and 1."""
	if width <= 0:
		return ""
	clamped = max(0.0, min(1.0, score))
	filled = int(clamped * width)
	return "█" * filled + "░" * (width - filled)
