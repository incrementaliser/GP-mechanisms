"""Neuronpedia URL helpers and feature-browser HTML for marimo cells."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd

from gp_notebook.token_display import format_bpe_tokens

NEURONPEDIA_BASE = "https://www.neuronpedia.org/pythia-70m-deduped"
MODEL_SLUG = "pythia-70m-deduped"


def feature_to_neuronpedia_url(feature_id: str) -> str:
    """Build a Neuronpedia feature page URL from a submodule/index id like resid_4/14907."""
    submodule, idx = feature_id.split("/")
    if submodule == "embed":
        layer_part = "0-embed"
    elif submodule.startswith("resid_"):
        layer_part = f"{submodule.split('_')[1]}-res-sm"
    elif submodule.startswith("attn_"):
        layer_part = f"{submodule.split('_')[1]}-att-sm"
    elif submodule.startswith("mlp_"):
        layer_part = f"{submodule.split('_')[1]}-mlp-sm"
    else:
        layer_part = submodule.replace("_", "-")
    return f"{NEURONPEDIA_BASE}/{layer_part}/{idx}"


def feature_card_html(row: pd.Series) -> str:
    """Render one annotated feature as a theme-aware Neuronpedia outbound card."""
    feature_id = str(row.get("Feature", row.get("feature", "")))
    category = html.escape(str(row.get("Category", row.get("category", ""))))
    annotation = html.escape(str(row.get("Annotation", row.get("annotation", ""))))
    layer = row.get("layer", "")
    url = feature_to_neuronpedia_url(feature_id)
    side = row.get("reading_side", "other")
    if side == "pro_gp":
        border = "var(--gp-color-gp)"
        soft = "var(--gp-color-gp-soft)"
    elif side == "anti_gp":
        border = "var(--gp-color-non-gp)"
        soft = "var(--gp-color-non-gp-soft)"
    else:
        border = "var(--gp-border)"
        soft = "var(--gp-surface)"
    return f"""
    <div style="border:2px solid {border}; border-radius:10px; padding:12px 14px;
                margin:8px 0; background:color-mix(in srgb, {soft} 45%, var(--gp-surface-card));
                color:var(--gp-ink); font-family:var(--gp-font-body);">
      <div style="font-weight:700; font-size:0.95rem; font-family:var(--gp-font-mono);">
        {html.escape(feature_id)}
      </div>
      <div style="color:var(--gp-muted); font-size:0.85rem; margin:4px 0;">
        Layer {html.escape(str(layer))} · {category}
      </div>
      <div style="font-size:0.9rem; margin-bottom:10px;">{annotation}</div>
      <a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener"
         style="color:var(--gp-accent); font-weight:700; text-decoration:none;
                border-bottom:1px solid var(--gp-accent);">
        Open in Neuronpedia atlas (new tab) ↗
      </a>
    </div>
    """


def feature_gallery_html(features: pd.DataFrame, *, max_cards: int = 6) -> str:
    """Render a grid of feature cards for the circuit explorer."""
    cards = [feature_card_html(row) for _, row in features.head(max_cards).iterrows()]
    header = (
        "<p style='margin:0 0 8px;color:var(--gp-muted);font-size:0.88rem;"
        "font-family:var(--gp-font-body);'>"
        "Outbound links to Neuronpedia’s feature atlas — activations below stay in-notebook."
        "</p>"
    )
    return (
        header
        + "<div style='display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));"
        " gap:8px;'>"
        + "".join(cards)
        + "</div>"
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
        parts.append(
            f"<div style='font-weight:600;margin-bottom:6px;color:var(--gp-ink);"
            f"font-family:var(--gp-font-body);'>{html.escape(label)}</div>"
        )
    parts.append(
        "<div style='display:flex;flex-wrap:wrap;gap:4px;font-family:var(--gp-font-mono);'>"
    )
    for tok, act in zip(tok_list, act_list):
        safe = html.escape(tok)
        height = int(8 + 40 * (act / max_act))
        on = act >= threshold
        bg = "var(--gp-color-gp)" if on else "var(--gp-border)"
        label_style = "font-weight:700;" if on else "opacity:0.65;"
        parts.append(
            f"<div style='text-align:center;min-width:28px;'>"
            f"<div style='height:{height}px;width:22px;margin:0 auto;background:{bg};"
            f"border-radius:3px 3px 0 0;'></div>"
            f"<div style='font-size:10px;color:inherit;{label_style}'>{safe}</div></div>"
        )
    parts.append("</div>")
    return "".join(parts)
