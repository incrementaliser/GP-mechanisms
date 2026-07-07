"""Neuronpedia URL helpers and feature-browser HTML for marimo cells."""

from __future__ import annotations

import html
from typing import Iterable
from urllib.parse import quote

import pandas as pd

from gp_notebook.token_display import format_bpe_tokens

NEURONPEDIA_BASE = "https://www.neuronpedia.org/pythia-70m-deduped"
MODEL_SLUG = "pythia-70m-deduped"


def feature_to_neuronpedia_url(feature_id: str) -> str:
    """Build a Neuronpedia feature page URL from a submodule/index id like resid_4/14907."""
    submodule, idx = feature_id.split("/")
    layer_part = submodule.replace("embed", "0-embed").replace("_", "-")
    if submodule == "embed":
        layer_part = "0-embed"
    elif submodule.startswith("resid_"):
        layer_part = f"{submodule.split('_')[1]}-res-sm"
    elif submodule.startswith("attn_"):
        layer_part = f"{submodule.split('_')[1]}-att-sm"
    elif submodule.startswith("mlp_"):
        layer_part = f"{submodule.split('_')[1]}-mlp-sm"
    return f"{NEURONPEDIA_BASE}/{layer_part}/{idx}"


def feature_card_html(row: pd.Series) -> str:
    """Render one annotated feature as a Neuronpedia-linked card."""
    feature_id = str(row.get("Feature", row.get("feature", "")))
    category = html.escape(str(row.get("Category", row.get("category", ""))))
    annotation = html.escape(str(row.get("Annotation", row.get("annotation", ""))))
    layer = row.get("layer", "")
    url = feature_to_neuronpedia_url(feature_id)
    side = row.get("reading_side", "other")
    border = "#c0392b" if side == "pro_gp" else "#2980b9" if side == "anti_gp" else "#95a5a6"
    return f"""
    <div style="border:2px solid {border}; border-radius:10px; padding:12px 14px;
                margin:8px 0; background:#fafafa; font-family:system-ui,sans-serif;">
      <div style="font-weight:700; font-size:0.95rem;">{html.escape(feature_id)}</div>
      <div style="color:#555; font-size:0.85rem; margin:4px 0;">Layer {layer} · {category}</div>
      <div style="font-size:0.9rem; margin-bottom:8px;">{annotation}</div>
      <a href="{url}" target="_blank" rel="noopener"
         style="color:#2563eb; font-weight:600; text-decoration:none;">
        Open in Neuronpedia ↗
      </a>
    </div>
    """


def feature_gallery_html(features: pd.DataFrame, *, max_cards: int = 6) -> str:
    """Render a grid of feature cards for the circuit explorer."""
    cards = [feature_card_html(row) for _, row in features.head(max_cards).iterrows()]
    return (
        "<div style='display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));"
        " gap:8px;'>" + "".join(cards) + "</div>"
    )


def spike_bar_html(
    tokens: Iterable[str],
    activations: Iterable[float],
    *,
    threshold: float = 0.1,
    label: str = "",
) -> str:
    """Render per-token SAE activation spikes as inline bars (Neuronpedia-style)."""
    tok_list = format_bpe_tokens(tokens)
    act_list = list(activations)
    max_act = max(act_list) if act_list else 1.0
    max_act = max(max_act, 1e-6)
    parts: list[str] = []
    if label:
        parts.append(f"<div style='font-weight:600;margin-bottom:6px;'>{html.escape(label)}</div>")
    parts.append("<div style='display:flex;flex-wrap:wrap;gap:4px;font-family:monospace;'>")
    for tok, act in zip(tok_list, act_list):
        safe = html.escape(tok)
        height = int(8 + 40 * (act / max_act))
        on = act >= threshold
        bg = "#e74c3c" if on else "#ecf0f1"
        color = "#fff" if on else "#333"
        parts.append(
            f"<div style='text-align:center;min-width:28px;'>"
            f"<div style='height:{height}px;width:22px;margin:0 auto;background:{bg};"
            f"border-radius:3px 3px 0 0;'></div>"
            f"<div style='font-size:10px;color:{color};'>{safe}</div></div>"
        )
    parts.append("</div>")
    return "".join(parts)
