"""Circuitsvis-backed interactive visualizations for marimo embedding."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from circuitsvis.attention import attention_patterns
from circuitsvis.tokens import colored_tokens, colored_tokens_multi


def colored_token_view(
    tokens: list[str],
    values: list[float],
    *,
    label: str = "",
    negative_color: str | None = "#c0392b",
    positive_color: str | None = "#2980b9",
) -> str:
    """Render tokens colored by scalar values using circuitsvis, returning embeddable HTML."""
    html = str(
        colored_tokens(
            tokens,
            values,
            negative_color=negative_color,
            positive_color=positive_color,
        )
    )
    if label:
        html = f"<h4 style='margin:4px 0;font-family:system-ui'>{label}</h4>" + html
    return html


def multi_feature_token_view(
    tokens: list[str],
    value_matrix: np.ndarray,
    feature_names: list[str],
) -> str:
    """Render multiple feature activation rows over the same token sequence."""
    return str(colored_tokens_multi(tokens, value_matrix, feature_names))


def attention_view(
    tokens: list[str],
    attention: np.ndarray,
) -> str:
    """Render attention-pattern heatmaps for all heads in one layer."""
    return str(attention_patterns(tokens, attention))
