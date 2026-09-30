"""Plotly figure builders for the Streamlit dashboard.

Kept separate from ``app.py`` so the plotting logic is small, importable and
unit-testable (tests skip this module when plotly is not installed).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.express as px
import plotly.graph_objects as go

from utils import label_colour

__all__ = [
    "batch_distribution_figure",
    "confidence_histogram_figure",
    "distribution_figure",
    "emotion_radar_figure",
    "entropy_histogram_figure",
    "score_table_figure",
]

_LAYOUT = {
    "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"size": 13},
    "showlegend": False,
}


def distribution_figure(
    distribution: Mapping[str, float], *, title: str = "Label distribution"
) -> go.Figure:
    """Horizontal bar chart of the full label probability distribution."""
    if not distribution:
        return go.Figure()
    labels = list(distribution)
    values = [float(distribution[label]) for label in labels]
    figure = px.bar(
        x=values,
        y=labels,
        orientation="h",
        color=labels,
        color_discrete_map={label: label_colour(label) for label in labels},
        labels={"x": "Probability", "y": ""},
        title=title,
    )
    figure.update_layout(**_LAYOUT)
    figure.update_yaxes(categoryorder="total ascending")
    figure.update_xaxes(tickformat=".0%")
    figure.update_traces(text=[f"{value:.1%}" for value in values], textposition="outside")
    return figure


def emotion_radar_figure(
    distribution: Mapping[str, float], *, title: str = "Emotion profile"
) -> go.Figure:
    """Radar (spider) chart of the label distribution.

    Radar charts need at least three axes to be readable, so binary sentiment
    models fall back to :func:`distribution_figure`.
    """
    labels = list(distribution)
    if len(labels) < 3:
        return distribution_figure(distribution, title=title)
    values = [float(distribution[label]) for label in labels]
    figure = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(66, 133, 244, 0.22)",
            line={"color": "#4285f4", "width": 2},
            hovertemplate="%{theta}: %{r:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        polar={
            "radialaxis": {"visible": True, "range": [0, max(values) or 1], "tickformat": ".0%"},
        },
        **_LAYOUT,
        showlegend=False,
    )
    return figure


def score_table_figure(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Bar chart comparing confidence across many items."""
    if not rows:
        return go.Figure()
    labels = [str(row.get("label", "unknown")) for row in rows]
    values = [float(row.get("confidence") or 0.0) for row in rows]
    figure = px.bar(
        x=values,
        y=[f"#{index}" for index in range(len(rows))],
        orientation="h",
        color=labels,
        color_discrete_map={label: label_colour(label) for label in set(labels)},
        labels={"x": "Confidence", "y": "Row"},
        title="Confidence per row",
    )
    figure.update_layout(**_LAYOUT)
    figure.update_xaxes(tickformat=".0%")
    return figure


def batch_distribution_figure(counts: Mapping[str, int]) -> go.Figure:
    """Donut chart of predicted label counts across a batch."""
    if not counts:
        return go.Figure()
    labels = list(counts)
    figure = px.pie(
        names=labels,
        values=[float(counts[label]) for label in labels],
        color=labels,
        color_discrete_map={label: label_colour(label) for label in labels},
        hole=0.45,
        title="Predicted label distribution",
    )
    figure.update_traces(textinfo="label+percent")
    figure.update_layout(**_LAYOUT, showlegend=True)
    return figure


def confidence_histogram_figure(confidences: Sequence[float]) -> go.Figure:
    """Histogram of per-row confidence values."""
    if not confidences:
        return go.Figure()
    figure = px.histogram(
        x=[float(value) for value in confidences],
        nbins=min(20, max(3, len(confidences) // 4 or 3)),
        labels={"x": "Confidence", "y": "Rows"},
        title="Confidence distribution",
        color_discrete_sequence=["#4285f4"],
    )
    figure.update_layout(**_LAYOUT)
    figure.update_xaxes(tickformat=".0%")
    return figure


def entropy_histogram_figure(entropies: Sequence[float]) -> go.Figure:
    """Histogram of normalised entropy (uncertainty) per row."""
    if not entropies:
        return go.Figure()
    figure = px.histogram(
        x=[float(value) for value in entropies],
        nbins=min(20, max(3, len(entropies) // 4 or 3)),
        labels={"x": "Normalized entropy", "y": "Rows"},
        title="Uncertainty distribution",
        color_discrete_sequence=["#ea4335"],
    )
    figure.update_layout(**_LAYOUT)
    figure.update_xaxes(tickformat=".0%")
    return figure
