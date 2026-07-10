"""Plotly and HTML visualisation helpers for the garden-path marimo notebook."""

from __future__ import annotations

import html
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gp_notebook.theme_tokens import GP_THEME, theme_color

_PAPER_LINKS_HTML = """
<div class="gp-hero__links">
  <a href="https://www.alphaxiv.org/abs/2412.05353">alphaXiv</a>
  · Hanna &amp; Mueller (NAACL 2025)
  · Notebook by: <a href="https://incrementaliser.github.io/">Arash Ashrafzadeh</a>
</div>
"""

_PAPER_LINKS_COMPACT_HTML = """
<div class="gp-header__links">
  <a href="https://www.alphaxiv.org/abs/2412.05353">alphaXiv</a>
  · Hanna &amp; Mueller (NAACL 2025)
  · Notebook by: <a href="https://incrementaliser.github.io/">Arash Ashrafzadeh</a>
</div>
"""


# Provenance badge kinds: label, tooltip, CSS class suffix.
_PROVENANCE_KINDS: dict[str, tuple[str, str]] = {
    "recomputed": (
        "Recomputed",
        "Regenerated from the model by this project's code (not copied from the paper).",
    ),
    "live": (
        "Live",
        "Computed on this machine just now, from the model and SAEs.",
    ),
    "paper": (
        "Paper-reported",
        "Values taken from the published paper, not independently regenerated here.",
    ),
    "schematic": (
        "Schematic",
        "Illustrative rendering to build intuition; magnitudes are not measurements.",
    ),
    "extension": (
        "Extension",
        "Analysis that goes beyond the paper, added by this notebook.",
    ),
}


def provenance_badge_html(kind: str, detail: str = "") -> str:
    """Render one provenance pill (Recomputed / Live / Paper-reported / Schematic / Extension)."""
    label, tooltip = _PROVENANCE_KINDS[kind]
    text = f"{label} · {html.escape(detail)}" if detail else label
    return (
        f'<span class="gp-badge gp-badge--{kind}" title="{html.escape(tooltip, quote=True)}">'
        f"{text}</span>"
    )


def provenance_row_html(badges: list[tuple[str, str]], note: str = "") -> str:
    """Render a row of provenance pills with an optional short note after them."""
    pills = "".join(provenance_badge_html(kind, detail) for kind, detail in badges)
    note_html = f'<span class="gp-badge-note">{note}</span>' if note else ""
    return f'<div class="gp-badge-row">{pills}{note_html}</div>'


def paper_figure_html(
    png_path: str,
    caption: str,
    *,
    max_width: str = "920px",
) -> str:
    """Embed a paper figure PNG with caption, or a graceful fallback if missing."""
    import base64
    from pathlib import Path

    path = Path(png_path)
    if not path.is_file():
        return f"""
<figure class="gp-paper-figure" style="max-width:{max_width};margin:0.75rem 0;">
  <div style="padding:1.25rem;border:1px dashed var(--gp-border);border-radius:10px;
              background:var(--gp-surface);color:var(--gp-muted);font-family:var(--gp-font-body);">
    Paper figure unavailable at <code>{html.escape(path.name)}</code>.
  </div>
  <figcaption style="font-size:0.82rem;color:var(--gp-muted);margin-top:0.35rem;font-family:var(--gp-font-body);">
    {html.escape(caption)}
  </figcaption>
</figure>
"""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
<figure class="gp-paper-figure" style="max-width:{max_width};margin:0.75rem 0;">
  <img src="data:image/png;base64,{data}" alt="{html.escape(caption, quote=True)}"
       style="width:100%;height:auto;border-radius:10px;border:1px solid var(--gp-border,var(--gp-status-border,#e2e8f0));background:#fff;" />
  <figcaption style="font-size:0.82rem;color:var(--gp-muted,var(--gp-status-muted,#64748b));margin-top:0.35rem;font-family:var(--gp-font-body);">
    {html.escape(caption)}
  </figcaption>
</figure>
"""


def plotly_template(theme: str) -> str:
    """Return the Plotly layout template name for the active marimo theme."""
    return "plotly_dark" if theme == "dark" else "plotly_white"


def apply_plotly_theme(fig: go.Figure, theme: str) -> go.Figure:
    """Apply the light or dark Plotly template and notebook fonts to a figure."""
    fig.update_layout(
        template=plotly_template(theme),
        font=dict(family=GP_THEME["font_body"].replace('"', "")),
        title_font=dict(family=GP_THEME["font_display"].replace('"', "")),
    )
    return fig


def notebook_header_compact_html() -> str:
    """Return a slim global header with title, paper links, and author names only."""
    return f"""
<div class="gp-header">
  <div class="gp-header__card">
    <div class="gp-header__row">
      <h1 class="gp-header__title">Garden Path Mechanisms in Language Models</h1>
      {_PAPER_LINKS_COMPACT_HTML}
    </div>
  </div>
</div>
"""


def notebook_hero_html() -> str:
    """Return a full-bleed intro hero: brand, one headline, one sentence, fork atmosphere."""
    gp = theme_color("gp")
    non_gp = theme_color("non_gp")
    highlight = theme_color("highlight")
    muted = theme_color("muted")
    return f"""
<div class="gp-hero">
  <div class="gp-hero__plane">
    <div class="gp-hero__diagram" aria-hidden="true">
      <svg viewBox="0 0 1100 340" preserveAspectRatio="xMaxYMid meet" role="presentation">
        <defs>
          <linearGradient id="gpForkGp" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="{gp}" stop-opacity="0.95"/>
            <stop offset="100%" stop-color="{gp}" stop-opacity="0.35"/>
          </linearGradient>
          <linearGradient id="gpForkNon" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="{non_gp}" stop-opacity="0.95"/>
            <stop offset="100%" stop-color="{non_gp}" stop-opacity="0.35"/>
          </linearGradient>
        </defs>
        <!-- Fork sits on the right, shifted slightly left of the far edge. -->
        <circle cx="820" cy="110" r="9" fill="{highlight}" opacity="0.9"/>
        <line x1="820" y1="36" x2="820" y2="110" stroke="{muted}" stroke-width="2.5" opacity="0.55"/>
        <path class="gp-hero__fork-branch" d="M820 110 C740 150, 680 190, 640 230"
              stroke="url(#gpForkGp)" stroke-width="4.5" fill="none" stroke-linecap="round"/>
        <path class="gp-hero__fork-branch gp-hero__fork-branch--delay"
              d="M820 110 C900 150, 955 190, 985 230"
              stroke="url(#gpForkNon)" stroke-width="4.5" fill="none" stroke-linecap="round"/>
        <text x="640" y="252" font-size="15" fill="{gp}" font-weight="700"
              font-family="var(--gp-font-body)" text-anchor="middle">GP · object → ,</text>
        <text x="985" y="252" font-size="15" fill="{non_gp}" font-weight="700"
              font-family="var(--gp-font-body)" text-anchor="middle">non-GP · subject → was</text>
      </svg>
    </div>
    <div class="gp-hero__grid">
      <div>
        <div class="gp-hero__badge">Hanna &amp; Mueller · NAACL 2025</div>
        <h1 class="gp-hero__title">Garden Path Mechanisms</h1>
        <p class="gp-hero__headline">
          Sparse features that tip an ambiguous parse — and a second circuit that barely reuses them.
        </p>
        <p class="gp-hero__subtitle">
          An interactive reading of how Pythia-70m maintains competing syntactic readings,
          then how Gemma-2-2b answers follow-ups without repairing the parse circuit.
        </p>
        {_PAPER_LINKS_HTML}
      </div>
    </div>
  </div>
</div>
"""


def fig1_walkthrough_html(step: str) -> str:
    """Render one panel of the paper Figure 1 pipeline (locate / annotate / intervene)."""
    gp = theme_color("gp")
    non_gp = theme_color("non_gp")
    accent = theme_color("accent")
    ink = theme_color("ink")
    muted = theme_color("muted")
    surface = theme_color("surface_card")
    border = theme_color("border")
    soft = theme_color("highlight_soft")

    panels: dict[str, tuple[str, str]] = {
        "locate": (
            f"""
<svg viewBox="0 0 640 200" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Locate features with AtP-IG">
  <rect width="640" height="200" rx="8" fill="{surface}" stroke="{border}"/>
  <text x="24" y="36" font-size="16" fill="{ink}" font-weight="700">1 · Locate</text>
  <text x="24" y="58" font-size="12" fill="{muted}">Score each SAE feature by its effect on m = p(GP) − p(non-GP)</text>
  <rect x="40" y="80" width="200" height="84" rx="8" fill="{soft}" stroke="{accent}"/>
  <text x="140" y="118" text-anchor="middle" font-size="14" fill="{ink}">ambiguous prefix</text>
  <text x="140" y="140" text-anchor="middle" font-size="12" fill="{muted}">… signed the bill</text>
  <path d="M255 122 L310 122" stroke="{accent}" stroke-width="2.5"/>
  <text x="400" y="100" font-size="13" fill="{gp}" font-weight="700">f_subject  ÎE = +0.52</text>
  <text x="400" y="124" font-size="13" fill="{non_gp}" font-weight="700">f_object   ÎE = −0.39</text>
  <text x="400" y="148" font-size="12" fill="{muted}">AtP-IG ranks the circuit</text>
</svg>
""",
            "Attribution patching with integrated gradients finds the sparse features "
            "that move the garden-path preference metric — cheaply approximating each "
            "feature's causal contribution.",
        ),
        "annotate": (
            f"""
<svg viewBox="0 0 640 200" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Annotate syntactic feature roles">
  <rect width="640" height="200" rx="8" fill="{surface}" stroke="{border}"/>
  <text x="24" y="36" font-size="16" fill="{ink}" font-weight="700">2 · Annotate</text>
  <text x="24" y="58" font-size="12" fill="{muted}">Inspect activations and name what each feature detects</text>
  <rect x="40" y="78" width="170" height="90" rx="8" fill="color-mix(in srgb, {gp} 18%, {surface})" stroke="{gp}"/>
  <text x="125" y="112" text-anchor="middle" font-size="13" fill="{ink}" font-weight="700">subject detector</text>
  <text x="125" y="134" text-anchor="middle" font-size="11" fill="{muted}">pro / anti reading</text>
  <rect x="235" y="78" width="170" height="90" rx="8" fill="color-mix(in srgb, {non_gp} 18%, {surface})" stroke="{non_gp}"/>
  <text x="320" y="112" text-anchor="middle" font-size="13" fill="{ink}" font-weight="700">object detector</text>
  <text x="320" y="134" text-anchor="middle" font-size="11" fill="{muted}">NP complement</text>
  <rect x="430" y="78" width="170" height="90" rx="8" fill="{soft}" stroke="{accent}"/>
  <text x="515" y="112" text-anchor="middle" font-size="13" fill="{ink}" font-weight="700">clause-end</text>
  <text x="515" y="134" text-anchor="middle" font-size="11" fill="{muted}">&amp; word detectors</text>
</svg>
""",
            "High-importance features are often syntactic (subjects, objects, clause ends), "
            "but word-level detectors and uninterpretable features also move m — the circuit "
            "mixes genuine structure with shallow heuristics.",
        ),
        "intervene": (
            f"""
<svg viewBox="0 0 640 200" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Clamp features to flip the reading">
  <rect width="640" height="200" rx="8" fill="{surface}" stroke="{border}"/>
  <text x="24" y="36" font-size="16" fill="{ink}" font-weight="700">3 · Intervene</text>
  <text x="24" y="58" font-size="12" fill="{muted}">Upweight / ablate annotated groups — random controls do nothing</text>
  <rect x="50" y="88" width="220" height="70" rx="8" fill="color-mix(in srgb, {gp} 22%, {surface})" stroke="{gp}"/>
  <text x="160" y="118" text-anchor="middle" font-size="14" fill="{ink}">before: prefers GP</text>
  <text x="160" y="140" text-anchor="middle" font-size="12" fill="{gp}" font-weight="700">p(,) &gt; p(was)</text>
  <path d="M285 123 L345 123" stroke="{accent}" stroke-width="3"/>
  <polygon points="345,117 360,123 345,129" fill="{accent}"/>
  <rect x="370" y="88" width="220" height="70" rx="8" fill="color-mix(in srgb, {non_gp} 22%, {surface})" stroke="{non_gp}"/>
  <text x="480" y="118" text-anchor="middle" font-size="14" fill="{ink}">after clamp: flips</text>
  <text x="480" y="140" text-anchor="middle" font-size="12" fill="{non_gp}" font-weight="700">p(was) &gt; p(,)</text>
</svg>
""",
            "Clamping the annotated syntactic groups flips the preferred continuation; "
            "clamping the same number of random features does not. That is the paper's "
            "causal verification of the circuit.",
        ),
    }
    key = step if step in panels else "locate"
    svg, caption = panels[key]
    return f"""
<div class="gp-fig1-panel">
  {svg}
  <p class="gp-fig1-panel__caption">{html.escape(caption)}</p>
</div>
"""


def sae_pipeline_svg() -> str:
    """Render a polished SAE encode/decode schematic using design tokens."""
    accent = theme_color("accent")
    accent_soft = theme_color("accent_soft")
    surface = theme_color("surface_card")
    ink = theme_color("ink")
    muted = theme_color("muted")
    highlight = theme_color("highlight_soft")
    border = theme_color("border")
    return f"""
<svg class="gp-fluid-svg gp-fluid-svg--lg" viewBox="0 0 640 130" width="100%" height="auto"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Sparse autoencoder maps activation x to sparse features f then reconstructs x-hat">
  <rect x="40" y="42" width="100" height="48" rx="10" fill="{surface}" stroke="{border}" stroke-width="1.5"/>
  <text x="90" y="32" font-size="13" text-anchor="middle" fill="{muted}">activation</text>
  <text x="90" y="72" font-size="18" text-anchor="middle" fill="{ink}" font-weight="700">x</text>
  <path d="M150 66 L210 66" stroke="{accent}" stroke-width="2.5" marker-end="url(#gpArrow)"/>
  <text x="180" y="56" font-size="11" text-anchor="middle" fill="{muted}">encode</text>
  <rect x="220" y="42" width="120" height="48" rx="10" fill="{highlight}" stroke="{theme_color('highlight')}" stroke-width="1.5"/>
  <text x="280" y="32" font-size="13" text-anchor="middle" fill="{muted}">sparse features</text>
  <text x="280" y="72" font-size="18" text-anchor="middle" fill="{ink}" font-weight="700">f</text>
  <path d="M350 66 L410 66" stroke="{accent}" stroke-width="2.5" marker-end="url(#gpArrow)"/>
  <text x="380" y="56" font-size="11" text-anchor="middle" fill="{muted}">decode</text>
  <rect x="420" y="42" width="100" height="48" rx="10" fill="{accent_soft}" stroke="{accent}" stroke-width="1.5"/>
  <text x="470" y="32" font-size="13" text-anchor="middle" fill="{muted}">reconstruction</text>
  <text x="470" y="72" font-size="18" text-anchor="middle" fill="{ink}" font-weight="700">x̂</text>
  <defs>
    <marker id="gpArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="{accent}"/>
    </marker>
  </defs>
</svg>
"""


def serial_parallel_svgs() -> tuple[str, str]:
    """Return schematic SVGs contrasting serial vs parallel parse maintenance."""
    gp = theme_color("gp")
    non_gp = theme_color("non_gp")
    ink = theme_color("ink")
    muted = theme_color("muted")
    border = theme_color("border")
    surface = theme_color("surface_card")
    serial = f"""
<svg class="gp-fluid-svg gp-fluid-svg--sm" viewBox="0 0 300 150" width="100%" height="auto"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Serial parser hypothesis: only one reading active">
  <rect width="300" height="150" rx="12" fill="{surface}" stroke="{border}"/>
  <text x="16" y="28" font-size="13" fill="{ink}" font-weight="700">Serial (hypothesis)</text>
  <text x="16" y="48" font-size="11" fill="{muted}">Only one reading stays active</text>
  <circle cx="90" cy="95" r="18" fill="{gp}"/>
  <text x="90" y="130" font-size="11" text-anchor="middle" fill="{muted}">one reading</text>
  <circle cx="200" cy="95" r="18" fill="none" stroke="{non_gp}" stroke-width="2" stroke-dasharray="4 3" opacity="0.45"/>
  <text x="200" y="130" font-size="11" text-anchor="middle" fill="{muted}" opacity="0.55">discarded</text>
</svg>
"""
    parallel = f"""
<svg class="gp-fluid-svg gp-fluid-svg--sm" viewBox="0 0 300 150" width="100%" height="auto"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Parallel finding: both readings remain active">
  <rect width="300" height="150" rx="12" fill="{surface}" stroke="{border}"/>
  <text x="16" y="28" font-size="13" fill="{ink}" font-weight="700">Parallel (paper finding)</text>
  <text x="16" y="48" font-size="11" fill="{muted}">Both feature camps fire together</text>
  <circle cx="110" cy="88" r="16" fill="{gp}"/>
  <circle cx="160" cy="108" r="16" fill="{non_gp}"/>
  <text x="135" y="140" font-size="11" text-anchor="middle" fill="{muted}">both readings active</text>
</svg>
"""
    return serial, parallel


def attention_to_last_token_scores(layer_attn: np.ndarray, n_tokens: int) -> list[float]:
    """Average head-wise attention from the last query position to each key token."""
    arr = np.asarray(layer_attn)
    if arr.ndim == 3:
        scores = arr[:, -1, :n_tokens].mean(axis=0)
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        scores = arr[-1, :n_tokens]
    else:
        scores = np.linalg.norm(arr[:n_tokens], axis=-1) if arr.shape[0] >= n_tokens else arr.mean(axis=-1)
    return scores.tolist()


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
        color_discrete_sequence=[
            theme_color("accent"),
            theme_color("gp"),
            theme_color("non_gp"),
        ],
    )
    fig.add_hline(y=0, line_dash="dot", line_color=theme_color("neutral"))
    return fig


def tug_of_war_html(p_gp: float, p_non_gp: float, *, animate: bool = True) -> str:
    """Render an animated tug-of-war bar where GP pulls left and non-GP pulls right."""
    total = max(p_gp + p_non_gp, 1e-8)
    gp_share = p_gp / total
    marker_left = int(gp_share * 100)
    transition = "transition:left 0.35s ease, background 0.35s ease;" if animate else ""
    gp = theme_color("gp")
    non_gp = theme_color("non_gp")
    gp_soft = theme_color("gp_soft")
    non_gp_soft = theme_color("non_gp_soft")
    ink = theme_color("ink")
    return f"""
    <div style="font-family:var(--gp-font-body);max-width:720px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="color:{gp};font-weight:600;">GP reading</span>
        <span style="color:{non_gp};font-weight:600;">non-GP reading</span>
      </div>
      <div style="position:relative;height:28px;border-radius:14px;overflow:hidden;
                  background:linear-gradient(90deg,{gp_soft} {marker_left}%,{non_gp_soft} {marker_left}%);">
        <div style="position:absolute;left:calc({marker_left}% - 10px);top:-4px;width:20px;
                    height:36px;background:{ink};border-radius:4px;{transition}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:14px;">
        <span>p(GP) = {p_gp:.4f}</span>
        <span>Δ = {p_gp - p_non_gp:+.4f}</span>
        <span>p(non-GP) = {p_non_gp:.4f}</span>
      </div>
    </div>
    """


def lookup_sweep_intervention(
    sweeps: pd.DataFrame,
    condition: str,
    *,
    subject_amp: float,
    object_amp: float,
    clause_amp: float,
) -> dict[str, float] | None:
    """Interpolate mean p(GP)/p(non-GP) from the precomputed intervention amplitude grid.

    The grid in ``intervention_sweeps.parquet`` holds real model runs: NP/Z is a
    (subject_amp x clause_amp) grid with object detectors clamped to 0; NP/S is a
    1-D object_amp sweep with subject and CP-verb detectors clamped to 0. Between
    grid points we interpolate linearly (bilinearly for NP/Z).
    """
    subset = sweeps[(sweeps["condition"] == condition) & (~sweeps["use_random"])]
    if subset.empty:
        return None

    def _interp_1d(rows: pd.DataFrame, col: str, target: float) -> dict[str, float]:
        rows = rows.sort_values(col)
        values = rows[col].to_numpy()
        target = float(np.clip(target, values.min(), values.max()))
        out: dict[str, float] = {}
        for metric in ("mean_p_gp", "mean_p_non_gp", "mean_diff"):
            out[metric] = float(np.interp(target, values, rows[metric].to_numpy()))
        return out

    if condition == "NPZ":
        clause_values = np.sort(subset["clause_amp"].unique())
        clause_amp = float(np.clip(clause_amp, clause_values.min(), clause_values.max()))
        lower = clause_values[clause_values <= clause_amp].max()
        upper = clause_values[clause_values >= clause_amp].min()
        at_lower = _interp_1d(subset[subset["clause_amp"] == lower], "subject_amp", subject_amp)
        if upper == lower:
            result = at_lower
        else:
            at_upper = _interp_1d(subset[subset["clause_amp"] == upper], "subject_amp", subject_amp)
            weight = (clause_amp - lower) / (upper - lower)
            result = {
                key: at_lower[key] * (1 - weight) + at_upper[key] * weight for key in at_lower
            }
    else:
        result = _interp_1d(subset, "object_amp", object_amp)
    result["source"] = "cached-sweep"
    return result


def token_reveal_html(
    tokens: Iterable[str],
    *,
    highlight_idx: int | None = None,
    reveal_count: int | None = None,
) -> str:
    """Render tokens with optional progressive reveal and highlight."""
    token_list = list(tokens)
    shown = token_list if reveal_count is None else token_list[:reveal_count]
    highlight = theme_color("highlight")
    highlight_soft = theme_color("highlight_soft")
    ink = theme_color("ink")
    surface = theme_color("surface")
    parts: list[str] = []
    for idx, token in enumerate(shown):
        safe = html.escape(token)
        style = (
            f"padding:4px 6px;margin:2px;border-radius:6px;display:inline-block;color:{ink};"
        )
        if highlight_idx is not None and idx == highlight_idx:
            style += f"background:{highlight_soft};border:2px solid {highlight};"
        else:
            style += f"background:{surface};"
        parts.append(f'<span style="{style}">{safe}</span>')
    return (
        f'<div style="line-height:2.2;font-size:18px;font-family:var(--gp-font-body);">'
        f'{"".join(parts)}</div>'
    )


def reading_bubbles_html(gp_label: str, non_gp_label: str, *, winner: str | None = None) -> str:
    """Show competing parse hypotheses as floating callout bubbles."""
    gp = theme_color("gp")
    non_gp = theme_color("non_gp")
    gp_soft = theme_color("gp_soft")
    non_gp_soft = theme_color("non_gp_soft")
    ink = theme_color("ink")
    gp_style = f"border:2px solid {gp};"
    non_gp_style = f"border:2px solid {non_gp};"
    if winner == "gp":
        gp_style += f"background:{gp_soft};color:{ink};font-weight:700;"
    elif winner == "non_gp":
        non_gp_style += f"background:{non_gp_soft};color:{ink};font-weight:700;"
    return f"""
    <div style="display:flex;gap:16px;margin-top:12px;flex-wrap:wrap;font-family:var(--gp-font-body);">
      <div style="{gp_style}border-radius:12px;padding:12px 16px;max-width:280px;">
        <div style="font-size:12px;color:{gp};">Garden-path reading</div>
        <div>{html.escape(gp_label)}</div>
      </div>
      <div style="{non_gp_style}border-radius:12px;padding:12px 16px;max-width:280px;">
        <div style="font-size:12px;color:{non_gp};">Non-GP reading</div>
        <div>{html.escape(non_gp_label)}</div>
      </div>
    </div>
    """


def circuit_svg(category_counts: pd.DataFrame) -> str:
    """Render a simplified layer-by-layer circuit diagram as inline SVG."""
    layers = sorted(category_counts["layer"].unique())
    width = 760
    height = 40 + 70 * len(layers)
    gp = theme_color("gp")
    non_gp = theme_color("non_gp")
    neutral = theme_color("neutral")
    chunks: list[str] = [
        f'<svg class="gp-fluid-svg gp-fluid-svg--xl" viewBox="0 0 {width} {height}" '
        f'width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">'
    ]
    for row_idx, layer in enumerate(layers):
        y = 30 + row_idx * 70
        chunks.append(
            f'<text x="10" y="{y}" font-size="14" fill="currentColor" opacity="0.75">'
            f"Layer {layer}</text>"
        )
        subset = category_counts[category_counts["layer"] == layer].head(6)
        x = 90
        for _, row in subset.iterrows():
            color = (
                gp
                if row["reading_side"] == "pro_gp"
                else (non_gp if row["reading_side"] == "anti_gp" else neutral)
            )
            count = int(row["count"])
            label = html.escape(f"{row['Category']} ({count})")
            radius = 16 + min(count, 8) * 2
            chunks.append(
                f'<circle cx="{x}" cy="{y-8}" r="{radius}" fill="{color}" opacity="0.85" />'
            )
            chunks.append(
                f'<text x="{x-radius}" y="{y+18}" font-size="11" fill="currentColor" '
                f'opacity="0.85">{label}</text>'
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


def probe_figure5_plot(probe_data: dict[str, list[dict[str, float | str]]], condition: str) -> go.Figure:
    """Line chart of structural probe action probabilities, digitized from paper Figure 5."""
    rows = probe_data.get(condition, [])
    df = pd.DataFrame(rows)
    df["layer"] = df["layer"].astype(str)
    fig = go.Figure()
    labels = {
        "LEFT-ARC": "LEFT-ARC (GP reading)",
        "GEN": "GEN (non-GP reading)",
        "RIGHT-ARC": "RIGHT-ARC (implausible)",
    }
    colors = {
        "LEFT-ARC": theme_color("gp"),
        "GEN": theme_color("non_gp"),
        "RIGHT-ARC": theme_color("neutral"),
    }
    for action in ("LEFT-ARC", "GEN", "RIGHT-ARC"):
        if action in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["layer"],
                    y=df[action],
                    mode="lines+markers",
                    name=labels[action],
                    line=dict(color=colors[action]),
                )
            )
    fig.update_layout(
        title=f"Structural probe action probabilities ({condition}) — digitized from Figure 5",
        xaxis_title="Layer",
        yaxis_title="Probability",
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(df["layer"])),
        yaxis_range=[0, 0.9],
    )
    return fig


def faithfulness_anchor_figure() -> go.Figure:
    """Plot the paper's two reported faithfulness values at the ÎE > 0.1 threshold.

    Only these two points are reported for Pythia-70m (Section 4.2 and Appendix C);
    the paper notes that faithfulness approaches 1 slowly and *non-monotonically*
    as the threshold is lowered, so no curve is drawn between or beyond them.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[65],
            y=[3.48],
            mode="markers+text",
            name="NP/Z",
            text=["NP/Z: 3.48 (65 features)"],
            textposition="middle right",
            marker=dict(color=theme_color("gp"), size=14),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[155],
            y=[0.20],
            mode="markers+text",
            name="NP/S",
            text=["NP/S: 0.20 (155 features)"],
            textposition="middle right",
            marker=dict(color=theme_color("non_gp"), size=14),
        )
    )
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color=theme_color("neutral"),
        annotation_text="faithfulness = 1 (circuit matches full model)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Circuit faithfulness at ÎE > 0.1 (paper-reported, Appendix C)",
        xaxis_title="Number of circuit features",
        yaxis_title="Faithfulness",
        xaxis_range=[0, 260],
        yaxis_range=[-0.4, 4.2],
        showlegend=False,
    )
    return fig


def group_ablation_figure(effects: pd.DataFrame) -> go.Figure:
    """Bar chart of measured Δm when zero-ablating each annotated feature group."""
    plot_df = effects.sort_values("delta_m", key=lambda s: s.abs(), ascending=False)
    colors = {
        "pro_gp": theme_color("gp"),
        "anti_gp": theme_color("non_gp"),
        "other": theme_color("neutral"),
    }
    fig = go.Figure(
        go.Bar(
            x=plot_df["group"],
            y=plot_df["delta_m"],
            marker_color=[colors[side] for side in plot_df["reading_side"]],
            customdata=plot_df[["n_features", "ablated_m", "baseline_m"]],
            hovertemplate=(
                "%{x}<br>Δm = %{y:+.4f}<br>features ablated: %{customdata[0]}"
                "<br>m after ablation: %{customdata[1]:.4f}"
                "<br>baseline m: %{customdata[2]:.4f}<extra></extra>"
            ),
        )
    )
    baseline = float(plot_df["baseline_m"].iloc[0])
    fig.add_hline(y=0, line_dash="dot", line_color=theme_color("neutral"))
    fig.update_layout(
        title=(
            "Measured effect of zero-ablating each feature group on "
            f"m = p(GP) − p(non-GP) (baseline m = {baseline:+.3f})"
        ),
        xaxis_title="Feature group (from the authors' annotations)",
        yaxis_title="Δm after ablation",
    )
    return fig
