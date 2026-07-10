"""Circuitsvis-backed interactive visualisations for marimo embedding.

Circuitsvis output embeds a ``<script type="module">`` that loads its renderer
from a CDN. Scripts inserted with bare ``mo.Html`` never execute, so callers
should wrap the returned HTML with :func:`embed_view_html` (a data-URI iframe).
"""

from __future__ import annotations

import base64
import html

import marimo as mo
import numpy as np
import torch
from circuitsvis.attention import attention_patterns
from circuitsvis.tokens import colored_tokens, colored_tokens_multi
from marimo._output.hypertext import Html

from gp_notebook.token_display import format_bpe_tokens

_IFRAME_STYLE = (
    "<style>body{margin:6px;background:#ffffff;color:#1c2833;"
    "font-family:system-ui,sans-serif;}</style>"
)


def _safe_label(label: str) -> str:
    """Escape and normalise quotes so labels are safe inside notebook HTML wrappers."""
    return html.escape(label.replace('"', "'"))


def embed_view_html(inner_html: str, *, height: str = "220px") -> Html:
    """Embed circuitsvis HTML in a data-URI iframe via ``mo.Html`` (no virtual files)."""
    payload = base64.b64encode(inner_html.encode("utf-8")).decode("ascii")
    return mo.Html(
        f'<iframe src="data:text/html;base64,{payload}" '
        f'style="width:100%;height:{height};border:0;border-radius:8px;'
        f'background:#fff;" loading="lazy" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def colored_token_view(
    tokens: list[str],
    values: list[float],
    *,
    label: str = "",
    negative_color: str | None = "#c0392b",
    positive_color: str | None = "#2980b9",
) -> str:
    """Render tokens coloured by scalar values using circuitsvis, returning embeddable HTML."""
    display_tokens = format_bpe_tokens(tokens)
    rendered = str(
        colored_tokens(
            display_tokens,
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
    return _IFRAME_STYLE + rendered


def multi_feature_token_view(
    tokens: list[str],
    value_matrix: np.ndarray,
    feature_names: list[str],
) -> str:
    """Render multiple feature activation rows over the same token sequence."""
    display_tokens = format_bpe_tokens(tokens)
    arr = np.asarray(value_matrix, dtype=np.float32)
    n_tokens = len(display_tokens)
    n_features = len(feature_names)
    if arr.shape == (n_features, n_tokens):
        arr = arr.T
    elif arr.shape != (n_tokens, n_features):
        raise ValueError(
            f"value_matrix shape {arr.shape} must be "
            f"({n_tokens}, {n_features}) or ({n_features}, {n_tokens})"
        )
    tensor = torch.from_numpy(arr)
    return _IFRAME_STYLE + str(colored_tokens_multi(display_tokens, tensor, feature_names))


def attention_view(
    tokens: list[str],
    attention: np.ndarray,
) -> str:
    """Render attention-pattern heatmaps for all heads in one layer."""
    return _IFRAME_STYLE + str(attention_patterns(format_bpe_tokens(tokens), attention))
