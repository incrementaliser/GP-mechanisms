"""Plotly and HTML visualization helpers for the marimo notebook."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def behavioral_figure(summary: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of mean p(GP)-p(non-GP) by structure and input type."""
    order = ["ambiguous", "gp", "post"]
    plot_df = summary.copy()
    plot_df["input_type"] = pd.Categorical(plot_df["input_type"], categories=order, ordered=True)
    fig = px.bar(
        plot_df,
        x="condition",
        y="mean_diff",
        color="input_type",
        barmode="group",
        error_y="sem_diff",
        labels={
            "condition": "Garden-path structure",
            "mean_diff": "p(GP) − p(non-GP)",
            "input_type": "Input type",
        },
        title="Garden-path continuation preferences (Pythia-70m)",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig


def tug_of_war_html(p_gp: float, p_non_gp: float) -> str:
    """Render a tug-of-war bar where GP pulls left and non-GP pulls right."""
    total = max(p_gp + p_non_gp, 1e-8)
    gp_share = p_gp / total
    non_gp_share = p_non_gp / total
    marker_left = int(gp_share * 100)
    return f"""
    <div style="font-family: system-ui, sans-serif; max-width: 720px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="color:#c0392b; font-weight:600;">GP reading</span>
        <span style="color:#2980b9; font-weight:600;">non-GP reading</span>
      </div>
      <div style="position:relative; height:28px; border-radius:14px; overflow:hidden;
                  background: linear-gradient(90deg, #f5b7b1 {marker_left}%, #aed6f1 {marker_left}%);">
        <div style="position:absolute; left:calc({marker_left}% - 10px); top:-4px; width:20px; height:36px;
                    background:#2c3e50; border-radius:4px;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:14px;">
        <span>p(GP) = {p_gp:.4f}</span>
        <span>p(non-GP) = {p_non_gp:.4f}</span>
      </div>
    </div>
    """


def token_reveal_html(
    tokens: Iterable[str],
    *,
    highlight_idx: int | None = None,
    reveal_count: int | None = None,
) -> str:
    """Render tokens with optional progressive reveal and highlight."""
    token_list = list(tokens)
    shown = token_list if reveal_count is None else token_list[:reveal_count]
    parts: list[str] = []
    for idx, token in enumerate(shown):
        safe = html.escape(token)
        style = "padding:4px 6px; margin:2px; border-radius:6px; display:inline-block;"
        if highlight_idx is not None and idx == highlight_idx:
            style += " background:#f9e79f; border:2px solid #f1c40f;"
        else:
            style += " background:#ecf0f1;"
        parts.append(f'<span style="{style}">{safe}</span>')
    return f'<div style="line-height:2.2; font-size:18px;">{"".join(parts)}</div>'


def reading_bubbles_html(gp_label: str, non_gp_label: str, *, winner: str | None = None) -> str:
    """Show competing parse hypotheses as floating callout bubbles."""
    gp_style = "border:2px solid #c0392b;"
    non_gp_style = "border:2px solid #2980b9;"
    if winner == "gp":
        gp_style += " background:#fadbd8; font-weight:700;"
    elif winner == "non_gp":
        non_gp_style += " background:#d6eaf8; font-weight:700;"
    return f"""
    <div style="display:flex; gap:16px; margin-top:12px; flex-wrap:wrap;">
      <div style="{gp_style} border-radius:12px; padding:12px 16px; max-width:280px;">
        <div style="font-size:12px; color:#922b21;">Garden-path reading</div>
        <div>{html.escape(gp_label)}</div>
      </div>
      <div style="{non_gp_style} border-radius:12px; padding:12px 16px; max-width:280px;">
        <div style="font-size:12px; color:#1f618d;">Non-GP reading</div>
        <div>{html.escape(non_gp_label)}</div>
      </div>
    </div>
    """


def circuit_svg(category_counts: pd.DataFrame) -> str:
    """Render a simplified layer-by-layer circuit diagram as inline SVG."""
    layers = sorted(category_counts["layer"].unique())
    width = 760
    height = 40 + 70 * len(layers)
    chunks: list[str] = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    for row_idx, layer in enumerate(layers):
        y = 30 + row_idx * 70
        chunks.append(
            f'<text x="10" y="{y}" font-size="14" fill="#555">Layer {layer}</text>'
        )
        subset = category_counts[category_counts["layer"] == layer].head(6)
        x = 90
        for _, item in subset.iterrows():
            color = "#c0392b" if item.reading_side == "pro_gp" else (
                "#2980b9" if item.reading_side == "anti_gp" else "#7f8c8d"
            )
            label = html.escape(f"{item.Category} ({item.count})")
            radius = 16 + min(int(item.count), 8) * 2
            chunks.append(
                f'<circle cx="{x}" cy="{y-8}" r="{radius}" fill="{color}" opacity="0.85" />'
            )
            chunks.append(
                f'<text x="{x-radius}" y="{y+18}" font-size="11" fill="#222">{label}</text>'
            )
            x += radius * 2 + 70
    chunks.append("</svg>")
    return "".join(chunks)


def activation_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of feature activations across token positions."""
    pivot = matrix.pivot_table(
        index="feature",
        columns="position",
        values="activation",
        aggfunc="mean",
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "Token position", "y": "Feature", "color": "Activation"},
        title="Pro- and anti-GP syntactic features both activate on ambiguous input",
    )
    return fig


def faithfulness_tradeoff() -> go.Figure:
    """Illustrate faithfulness vs number of features using paper-reported values."""
    data = pd.DataFrame(
        {
            "threshold": [0.20, 0.15, 0.12, 0.10, 0.08, 0.05],
            "n_features_npz": [25, 40, 55, 65, 95, 180],
            "n_features_nps": [60, 95, 120, 155, 220, 420],
            "faithfulness_npz": [5.2, 4.1, 3.8, 3.48, 2.4, 1.3],
            "faithfulness_nps": [0.08, 0.12, 0.16, 0.20, 0.55, 0.95],
        }
    )
    fig = go.Figure()
    for condition, color in (("npz", "#c0392b"), ("nps", "#2980b9")):
        fig.add_trace(
            go.Scatter(
                x=data[f"n_features_{condition}"],
                y=data[f"faithfulness_{condition}"],
                mode="lines+markers",
                name=condition.upper(),
                line=dict(color=color),
            )
        )
    fig.update_layout(
        title="Faithfulness budget: more features → closer to full model behavior",
        xaxis_title="Number of circuit features",
        yaxis_title="Faithfulness",
    )
    return fig
