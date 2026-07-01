"""Circuitsvis-backed interactive visualizations for marimo embedding."""

from __future__ import annotations

import html

import numpy as np
import torch
from circuitsvis.attention import attention_patterns
from circuitsvis.tokens import colored_tokens, colored_tokens_multi


def _safe_label(label: str) -> str:
    """Escape and normalize quotes so labels are safe inside notebook HTML wrappers."""
    return html.escape(label.replace('"', "'"))


def colored_token_view(
    tokens: list[str],
    values: list[float],
    *,
    label: str = "",
    negative_color: str | None = "#c0392b",
    positive_color: str | None = "#2980b9",
) -> str:
    """Render tokens colored by scalar values using circuitsvis, returning embeddable HTML."""
    rendered = str(
        colored_tokens(
            tokens,
            values,
            negative_color=negative_color,
            positive_color=positive_color,
        )
    )
    if label:
        rendered = (
            f"<h4 style='margin:4px 0;font-family:system-ui'>{_safe_label(label)}</h4>"
            + rendered
        )
    return rendered


def multi_feature_token_view(
    tokens: list[str],
    value_matrix: np.ndarray,
    feature_names: list[str],
) -> str:
    """Render multiple feature activation rows over the same token sequence."""
    arr = np.asarray(value_matrix, dtype=np.float32)
    n_tokens = len(tokens)
    n_features = len(feature_names)
    if arr.shape == (n_features, n_tokens):
        arr = arr.T
    elif arr.shape != (n_tokens, n_features):
        raise ValueError(
            f"value_matrix shape {arr.shape} must be "
            f"({n_tokens}, {n_features}) or ({n_features}, {n_tokens})"
        )
    tensor = torch.from_numpy(arr)
    return str(colored_tokens_multi(tokens, tensor, feature_names))


def attention_view(
    tokens: list[str],
    attention: np.ndarray,
) -> str:
    """Render attention-pattern heatmaps for all heads in one layer."""
    return str(attention_patterns(tokens, attention))
