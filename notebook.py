"""Marimo notebook: Garden Path Sentence Processing Mechanisms in LMs."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", css_file="assets/sidebar.css", html_head_file="assets/theme-init.html")

with app.setup:
    # Fully import torch before any other cell runs: marimo's formatter
    # registration can otherwise race a concurrent session's first torch
    # import in `marimo run` and crash with a circular-import error.
    import contextlib

    with contextlib.suppress(ImportError):
        import torch  # noqa: F401


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    from gp_notebook.sidebar_nav import initial_theme, theme_from_request

    _request = mo.app_meta().request
    get_theme, set_theme = mo.state(
        initial_theme(mo.app_meta().theme, theme_from_request(_request)),
        allow_self_loops=True,
    )
    return get_theme, set_theme


@app.cell(hide_code=True)
def _(get_theme, mo, set_theme):
    from gp_notebook.sidebar_nav import build_theme_toggle

    theme = get_theme()

    def _set_light(_value: object) -> int:
        set_theme("light")
        return 0

    def _set_dark(_value: object) -> int:
        set_theme("dark")
        return 0

    _sun = '<iconify-icon icon="lucide:sun" width="18" height="18"></iconify-icon>'
    _moon = '<iconify-icon icon="lucide:moon" width="18" height="18"></iconify-icon>'
    light_button = mo.ui.button(
        label=_sun,
        on_click=_set_light,
        tooltip="Light mode",
    )
    dark_button = mo.ui.button(
        label=_moon,
        on_click=_set_dark,
        tooltip="Dark mode",
    )
    theme_toggle = build_theme_toggle(theme, light_button, dark_button)
    return theme, theme_toggle


@app.cell(hide_code=True)
def _(mo):
    from gp_notebook.sidebar_nav import MODULE_NAV_DEFAULT_KEY, module_nav_full_options

    live_switch = mo.ui.switch(label="Live mode (model + SAEs)", value=False)
    module_nav = mo.ui.radio(
        options=module_nav_full_options(),
        value=MODULE_NAV_DEFAULT_KEY,
        label="Section",
    )
    return live_switch, module_nav


@app.cell(hide_code=True)
def _(module_nav):
    show_intro = module_nav.value == "intro"
    return (show_intro,)


@app.cell(hide_code=True)
def _(mo, show_intro):
    mo.stop(not show_intro)
    from gp_notebook.sidebar_nav import wrap_with_class as _wrap_with_class
    from gp_notebook.viz import notebook_hero_html as _hero_html

    hero = mo.Html(_hero_html())
    tldr = mo.md(
        """
### TL;DR

- **Problem:** Autoregressive LMs handle syntactic ambiguities word-by-word, but we do not know
  *which internal features* drive their preferred reading of garden-path sentences.
- **Approach:** Sparse autoencoders (SAEs) decompose activations into interpretable features;
  attribution patching (AtP-IG) locates causally relevant circuits; targeted clamping verifies them.
- **Findings:** In **Pythia-70m**, garden-path preferences are causally driven by genuine syntactic
  detectors (subjects, objects, clause boundaries) *working alongside* shallow lexical heuristics,
  and the model keeps **both readings active at once**. In **Gemma-2-2b** — the model in the paper
  large enough to answer follow-up questions — those parse features are *not* reused when answering
  questions about the sentence: neither human-style repair nor reanalysis.
"""
    )
    lay_summary = mo.callout(
        mo.md(
            """
**In plain language:** Language models read sentences incrementally — one word at a time —
like humans listening to speech. A *garden-path* sentence tricks you into one interpretation
until a later word reveals another. This notebook walks through how the paper
**reverse-engineers** what happens inside a small LM (Pythia-70m) at the ambiguous noun: which
internal "detectors" fire, and whether both readings coexist. For the final question — does the
model *revise* its parse when asked about it afterwards? — the paper switches to the larger
Gemma-2-2b, because Pythia-70m cannot answer follow-up questions above chance (paper Table 3).
"""
        ),
        kind="info",
    )
    _wrap_with_class(
        mo.vstack(
            [
                hero,
                tldr,
                lay_summary,
            ]
        ),
        "gp-intro-page",
    )
    return


@app.cell(hide_code=True)
def _(mo, show_intro):
    mo.stop(show_intro)
    from gp_notebook.viz import notebook_header_compact_html

    mo.Html(notebook_header_compact_html())
    return


@app.cell(hide_code=True)
def _(live_switch, module_nav, mo, theme_toggle):
    from gp_notebook.cache_status import cache_status_markdown, missing_required_caches
    from gp_notebook.device import device_status_message, live_mode_available as _live_mode_available
    from gp_notebook.paths import saes_available as _sidebar_saes_available
    from gp_notebook.runtime import last_timed_call
    from gp_notebook.sidebar_nav import system_status_html, wrap_with_class

    gpu_ok = _live_mode_available()
    data_ok = not missing_required_caches()
    saes_ok = _sidebar_saes_available()
    data_tooltip = cache_status_markdown().replace("**", "").replace("`", "")
    sae_tooltip = (
        "SAE checkpoints found"
        if saes_ok
        else "SAE checkpoints missing — one-click download in Module 4"
    )
    status_widget = mo.Html(
        system_status_html(
            gpu_ok=gpu_ok,
            gpu_tooltip=device_status_message(),
            data_ok=data_ok,
            data_tooltip=data_tooltip,
            saes_ok=saes_ok,
            saes_tooltip=sae_tooltip,
            last_call=last_timed_call(),
        )
    )
    sidebar = mo.sidebar(
        wrap_with_class(
            mo.vstack(
                [
                    wrap_with_class(
                        mo.vstack(
                            [
                                theme_toggle,
                                wrap_with_class(module_nav, "gp-sidebar-nav"),
                                mo.md("### Execution"),
                                live_switch,
                            ]
                        ),
                        "gp-sidebar-main",
                    ),
                    wrap_with_class(status_widget, "gp-status-box-wrapper"),
                ]
            ),
            "gp-sidebar-inner",
        ),
        width="280px",
    )
    sidebar
    return


@app.cell(hide_code=True)
def _(mo, show_intro):
    mo.stop(show_intro)
    import numpy as np
    import plotly.express as px

    from gp_notebook.ablation import load_group_effects
    from gp_notebook.behavior import score_sentence, top_next_tokens
    from gp_notebook.device import live_mode_available
    from gp_notebook.features import (
        category_counts,
        enrich_feature_table,
        layer_narrative,
    )
    from gp_notebook.gprc import CIRCUIT_IOU, gprc_table_df, sample_gprc_items
    from gp_notebook.interp_views import (
        attention_view,
        colored_token_view,
        multi_feature_token_view,
    )
    from gp_notebook.interventions import run_intervention_suite, saes_available
    from gp_notebook.neuronpedia import feature_gallery_html, spike_bar_html
    from gp_notebook.paths import (
        ASSETS_DIR,
        load_gp_dataset,
        load_json_cache,
        load_parquet_cache,
    )
    from gp_notebook.probes import load_probe_cache
    from gp_notebook.runtime import get_hf_model, timed_call
    from gp_notebook.sae_fetch import download_saes, missing_sae_dirs
    from gp_notebook.viz import (
        activation_heatmap,
        apply_plotly_theme,
        attention_to_last_token_scores,
        behavioral_figure,
        circuit_svg,
        faithfulness_anchor_figure,
        group_ablation_figure,
        lookup_sweep_intervention,
        paper_figure_html,
        probe_figure5_plot,
        reading_bubbles_html,
        token_reveal_html,
        tug_of_war_html,
    )
    from gp_notebook.widgets import GardenPathScrubber

    gp_df = load_gp_dataset()
    behavioral_summary = load_parquet_cache("behavioral_summary.parquet")
    behavioral_scored = load_parquet_cache("behavioral_scored.parquet")
    interventions_df = load_parquet_cache("interventions.parquet")
    intervention_sweeps = load_parquet_cache("intervention_sweeps.parquet")
    top_next_cache = load_parquet_cache("top_next_tokens.parquet")
    probe_cache = load_probe_cache()
    return (
        ASSETS_DIR,
        CIRCUIT_IOU,
        GardenPathScrubber,
        activation_heatmap,
        apply_plotly_theme,
        attention_to_last_token_scores,
        attention_view,
        behavioral_figure,
        behavioral_scored,
        behavioral_summary,
        category_counts,
        circuit_svg,
        colored_token_view,
        download_saes,
        enrich_feature_table,
        faithfulness_anchor_figure,
        feature_gallery_html,
        get_hf_model,
        gp_df,
        gprc_table_df,
        group_ablation_figure,
        intervention_sweeps,
        interventions_df,
        layer_narrative,
        live_mode_available,
        load_group_effects,
        load_json_cache,
        load_parquet_cache,
        lookup_sweep_intervention,
        missing_sae_dirs,
        multi_feature_token_view,
        np,
        paper_figure_html,
        probe_cache,
        probe_figure5_plot,
        px,
        reading_bubbles_html,
        run_intervention_suite,
        saes_available,
        sample_gprc_items,
        score_sentence,
        spike_bar_html,
        timed_call,
        token_reveal_html,
        top_next_cache,
        top_next_tokens,
        tug_of_war_html,
    )


# ═══════════════════════════════════════════════════════════════════
# CHAPTER 0 — Primer
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m0 = module_nav.value == "m0"
    return (show_m0,)


@app.cell(hide_code=True)
def _(load_json_cache, mo, show_m0):
    mo.stop(not show_m0)
    m0_intro = mo.md(
        r"""
## 0. Primer

### What is mechanistic interpretability?

**Mechanistic interpretability** aims to reverse-engineer what neural networks
*actually compute*, not just what they get right.

| Concept | One-liner |
|---------|-----------|
| **Neuron** | A single unit — often *polysemantic* (fires for unrelated things). |
| **Feature** | A *monosemantic* direction found via sparse autoencoders (SAEs). |
| **Circuit** | Minimal subgraph of features that reproduces a behavior. |
| **Ablation** | Clamp a feature to zero and measure the effect — the core causal test. |
| **AtP-IG** | Gradient-based estimate of each feature's causal contribution, used to find circuit features cheaply. |
| **Faithfulness** | How well the small circuit alone matches the full model (ideally $\approx 1.0$). |

We apply these tools to **garden-path sentences** — inputs with two competing syntactic readings.
"""
    )
    glossary = mo.accordion(
        {
            "Sparse Autoencoder (SAE)": (
                r"$f = \mathrm{ReLU}(W_e(x - b_d) + b_e)$, $\hat{x} = W_d f + b_d$. "
                r"Each dimension of $f$ is an interpretable feature. Because of the ReLU, "
                r"feature activations are non-negative."
            ),
            "Garden-path metric": (
                r"$m = p(\text{GP}) - p(\text{non-GP})$. "
                r"$\text{GP}$ = comma/period continuation; $\text{non-GP}$ = $\texttt{was}$ continuation."
            ),
            "Neuronpedia": (
                "Interactive atlas for inspecting SAE features on real text — "
                "[pythia-70m-deduped](https://www.neuronpedia.org/pythia-70m-deduped)."
            ),
        }
    )
    attn_cache = load_json_cache("attention_npz.json")
    layer_select = (
        mo.ui.dropdown(
            options={f"Layer {i}": str(i) for i in range(6)},
            value="Layer 0",
            label="Attention layer",
        )
        if attn_cache is not None
        else None
    )
    mo.vstack([m0_intro, mo.md("### Key concepts"), glossary])
    return attn_cache, layer_select


@app.cell(hide_code=True)
def _(mo, show_m0):
    mo.stop(not show_m0)
    sparsity = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Sparsity threshold")
    return (sparsity,)


@app.cell(hide_code=True)
def _(load_parquet_cache, mo, np, show_m0, sparsity, spike_bar_html):
    mo.stop(not show_m0)
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, 32)
    sparse_features = np.maximum(raw - sparsity.value, 0)
    active = int((sparse_features > 0).sum())
    sae_svg = """
    <svg width="640" height="120" xmlns="http://www.w3.org/2000/svg">
      <text x="80" y="30" font-size="14" text-anchor="middle">x</text>
      <text x="300" y="30" font-size="14" text-anchor="middle">f</text>
      <text x="500" y="30" font-size="14" text-anchor="middle">x̂</text>
      <rect x="60" y="50" width="80" height="40" fill="#d5dbdb" />
      <polygon points="160,70 220,70 240,50 240,90 220,70" fill="#566573"/>
      <rect x="260" y="50" width="80" height="40" fill="#f9e79f" />
      <polygon points="360,70 420,70 440,50 440,90 420,70" fill="#566573"/>
      <rect x="460" y="50" width="80" height="40" fill="#d5dbdb" />
    </svg>
    """
    _content = [
        mo.md("### SAE primer"),
        mo.md(
            r"$x$ (activation) $\rightarrow$ $f = \mathrm{ReLU}(W_e(x - b_d) + b_e)$ $\rightarrow$ $\hat{x} = W_d f + b_d$"
        ),
        mo.Html(sae_svg),
        sparsity,
        mo.md(
            f"Illustration with random numbers: **{active}** of 32 features stay active at "
            f"threshold {sparsity.value:.2f} — sparsity is what makes individual features "
            "interpretable."
        ),
    ]
    _tok_act = load_parquet_cache("token_activations_npz.parquet")
    if _tok_act is not None and not _tok_act.empty and "sentence" in _tok_act.columns:
        _first_sentence = _tok_act["sentence"].iloc[0]
        _sent = _tok_act[
            (_tok_act["sentence"] == _first_sentence)
            & (_tok_act["reading_side"].isin({"pro_gp", "anti_gp"}))
        ]
        for feat_name in _sent["feature"].unique()[:2]:
            feat_rows = _sent[_sent["feature"] == feat_name].sort_values("position")
            _content.append(
                mo.Html(
                    spike_bar_html(
                        feat_rows["token"].tolist(),
                        feat_rows["activation"].tolist(),
                        label=f"Measured SAE activations — {feat_name}",
                    )
                )
            )
    mo.vstack(_content)
    return


@app.cell(hide_code=True)
def _(attn_cache, attention_view, layer_select, mo, np, show_m0):
    mo.stop(not show_m0)
    _content = [mo.md("### Attention patterns")]
    _attn_panel = mo.md("_Attention cache not loaded. Run `uv run python precompute.py`._")
    if attn_cache is not None and layer_select is not None:
        layer_idx = layer_select.value
        attn_data = np.array(attn_cache["layers"][layer_idx])
        _tokens = attn_cache["tokens"]
        attn_html = attention_view(_tokens, attn_data)
        _content.append(layer_select)
        _attn_panel = mo.vstack([
            mo.md(f"**Layer {layer_idx}** — {attn_data.shape[0]} heads"),
            mo.iframe(attn_html),
        ])
    _content.append(_attn_panel)
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — Feel the Garden Path
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m1 = module_nav.value == "m1"
    return (show_m1,)


@app.cell(hide_code=True)
def _(mo, show_m1):
    mo.stop(not show_m1)
    structure = mo.ui.dropdown(
        options={"NP/Z (subordinate clause)": "NPZ", "NP/S (sentential complement)": "NPS"},
        value="NP/Z (subordinate clause)",
        label="Structure",
    )
    verb_type = mo.ui.radio(
        options={"Ambiguous verb": "ambiguous", "GP-forcing verb": "gp", "Non-GP verb": "post"},
        value="Ambiguous verb",
        label="Verb type",
        inline=True,
    )
    return structure, verb_type


@app.cell(hide_code=True)
def _(GardenPathScrubber, load_parquet_cache, mo, show_m1, structure):
    mo.stop(not show_m1)
    prefix_cache = load_parquet_cache(f"prefix_probs_{structure.value.lower()}.parquet")
    scrubber = None
    if prefix_cache is not None and not prefix_cache.empty:
        _rows = prefix_cache.sort_values("n_tokens")
        _words = str(_rows["prefix"].iloc[-1]).split()
        if structure.value == "NPZ":
            _gp_label, _non_gp_label = "GP: clause ends (',')", "non-GP: subject ('was')"
        else:
            _gp_label, _non_gp_label = "GP: sentence ends ('.')", "non-GP: subject ('was')"
        scrubber = mo.ui.anywidget(
            GardenPathScrubber(
                tokens=_words,
                p_gp=[float(v) for v in _rows["p_gp"]],
                p_non_gp=[float(v) for v in _rows["p_non_gp"]],
                gp_label=_gp_label,
                non_gp_label=_non_gp_label,
                ambiguous_index=len(_words) - 1,
                revealed=2,
            )
        )
    return prefix_cache, scrubber


@app.cell(hide_code=True)
def _(
    apply_plotly_theme,
    attention_to_last_token_scores,
    colored_token_view,
    gp_df,
    load_json_cache,
    mo,
    np,
    prefix_cache,
    px,
    reading_bubbles_html,
    scrubber,
    show_m1,
    structure,
    theme,
    token_reveal_html,
    verb_type,
):
    mo.stop(not show_m1)
    col = f"sentence_{verb_type.value}"
    subset = gp_df[gp_df["condition"] == structure.value].iloc[0]
    gp_sentence = subset[col]
    _tokens = gp_sentence.split()
    if structure.value == "NPZ":
        gp_read = "Final noun = object of subordinate verb (clause can end → comma)."
        non_gp_read = "Final noun = matrix subject (continuation → was …)."
    else:
        gp_read = "Final noun = object of matrix verb (sentence can end → period)."
        non_gp_read = "Final noun = sentential subject (continuation → was …)."

    winner = None
    if prefix_cache is not None and verb_type.value == "ambiguous" and not prefix_cache.empty:
        _last = prefix_cache.sort_values("n_tokens").iloc[-1]
        winner = "gp" if _last["p_gp"] > _last["p_non_gp"] else "non_gp"

    _content = [
        mo.md("## 1. Feel the garden path"),
        mo.md(
            "The dataset (Sathe et al. 2024, same-length subset) varies only the verb: an "
            "*ambiguous* verb licenses both readings, a *GP-forcing* verb makes the garden-path "
            "reading correct, and a *non-GP* verb blocks it."
        ),
        mo.hstack([structure, verb_type], justify="start", gap=1.5, wrap=True),
    ]

    if verb_type.value == "ambiguous" and scrubber is not None:
        _content.append(scrubber)
    else:
        _content.extend(
            [
                mo.md(f"**Item {int(subset['item'])}** — {verb_type.value} verb variant:"),
                mo.Html(
                    token_reveal_html(_tokens, highlight_idx=len(_tokens) - 1, reveal_count=None)
                ),
            ]
        )
    _content.append(mo.Html(reading_bubbles_html(gp_read, non_gp_read, winner=winner)))

    if prefix_cache is not None and not prefix_cache.empty:
        prefix_fig = px.line(
            prefix_cache,
            x="n_tokens",
            y=["p_gp", "p_non_gp"],
            labels={"n_tokens": "Tokens seen", "value": "Probability", "variable": "Reading"},
            title="Commitment timeline: Pythia's preference as tokens arrive",
        )
        prefix_fig.update_traces(selector=dict(name="p_gp"), line_color="#c0392b", name="p(GP)")
        prefix_fig.update_traces(
            selector=dict(name="p_non_gp"), line_color="#2980b9", name="p(non-GP)"
        )
        apply_plotly_theme(prefix_fig, theme)
        _content.extend([mo.md("### Model commitment timeline"), prefix_fig])

    attn_path = f"attention_{structure.value.lower()}.json"
    tok_act_cache = load_json_cache(attn_path)
    if tok_act_cache is not None and verb_type.value == "ambiguous":
        tok_strs = tok_act_cache["tokens"]
        layer0 = np.array(tok_act_cache["layers"]["0"])
        attn_to_last = attention_to_last_token_scores(layer0, len(tok_strs))
        _content.extend([
            mo.md("### Attention at the ambiguous noun (layer 0, mean over heads)"),
            mo.iframe(
                colored_token_view(
                    tok_strs,
                    attn_to_last,
                    label="Attention into the ambiguous noun",
                    positive_color="#e67e22",
                ),
                height="150px",
            ),
        ])

    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — Behavioral Lab (Fig 2)
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m2 = module_nav.value == "m2"
    return (show_m2,)


@app.cell(hide_code=True)
def _(behavioral_scored, mo, show_m2):
    mo.stop(not show_m2)
    drill_df = None
    drill_table = None
    if behavioral_scored is not None:
        drill_df = behavioral_scored[behavioral_scored["input_type"] == "ambiguous"].copy()
        drill_table = mo.ui.table(
            drill_df[["item", "condition", "sentence", "p_gp", "p_non_gp", "diff"]],
            selection="single",
        )
    return drill_df, drill_table


@app.cell(hide_code=True)
def _(
    ASSETS_DIR,
    apply_plotly_theme,
    behavioral_figure,
    behavioral_summary,
    colored_token_view,
    drill_df,
    drill_table,
    load_json_cache,
    mo,
    paper_figure_html,
    px,
    show_m2,
    theme,
    top_next_cache,
):
    mo.stop(not show_m2)
    _content: list = [
        mo.md("## 2. Behavioral lab — does Pythia get garden-pathed?"),
        mo.md(
            "Reproduction of the paper's **Figure 2**: mean m = p(GP) − p(non-GP) per structure "
            "and verb type. MV/RR is shown but excluded from the mechanistic analyses because "
            "Pythia barely garden-paths on it (paper §4.1)."
        ),
    ]
    if behavioral_summary is None or drill_table is None:
        _content.append(
            mo.callout(
                mo.md(
                    "Behavioral caches are missing. Run `uv run python precompute.py` once to "
                    "regenerate them from the model."
                ),
                kind="warn",
            )
        )
    else:
        behavior_fig = apply_plotly_theme(behavioral_figure(behavioral_summary), theme)
        table_selection = drill_table.value
        if table_selection is not None and len(table_selection) > 0:
            selected = table_selection.iloc[0].to_dict()
        else:
            selected = drill_df.iloc[0].to_dict()
        token_bar = px.bar(
            x=["p(GP)", "p(non-GP)"],
            y=[selected["p_gp"], selected["p_non_gp"]],
            labels={"x": "", "y": "probability"},
            title=f"Item {selected['item']} ({selected['condition']})",
        )
        apply_plotly_theme(token_bar, theme)
        _content.extend([
            behavior_fig,
            mo.accordion(
                {
                    "Compare with the paper's original Figure 2": mo.Html(
                        paper_figure_html(
                            str(ASSETS_DIR / "paper_figures" / "fig2_behavioral.png"),
                            "Paper Figure 2 (Pythia-70m): the pattern reproduced above — "
                            "positive m for NP/Z and MV/RR, negative for NP/S; GP-forcing "
                            "verbs amplify, non-GP verbs suppress.",
                        )
                    )
                }
            ),
            mo.md("### Sentence drill-down"),
            drill_table,
            token_bar,
        ])

        if top_next_cache is not None:
            _tops = top_next_cache[
                (top_next_cache["item"] == selected["item"])
                & (top_next_cache["condition"] == selected["condition"])
            ]
            if not _tops.empty:
                next_fig = px.bar(
                    _tops,
                    x="token",
                    y="probability",
                    title="Top next-token probabilities at the ambiguous noun",
                )
                apply_plotly_theme(next_fig, theme)
                _content.append(next_fig)

        cond_key = selected["condition"].lower()
        attr_cache = load_json_cache(f"attributions_{cond_key}.json")
        if attr_cache is not None:
            _content.append(
                mo.iframe(
                    colored_token_view(
                        attr_cache["tokens"],
                        attr_cache["scores"],
                        label="Which tokens push m = p(GP) − p(non-GP)?",
                    ),
                    height="150px",
                ),
            )
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — Feature Microscope (Fig 3)
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m3 = module_nav.value == "m3"
    return (show_m3,)


@app.cell(hide_code=True)
def _(mo, show_m3):
    mo.stop(not show_m3)
    circuit_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Circuit",
    )
    category_filter = mo.ui.dropdown(
        options={
            "All categories": "all",
            "Syntactic only": "syntactic",
            "Pro-GP only": "pro_gp",
            "Anti-GP only": "anti_gp",
        },
        value="All categories",
        label="Filter",
    )
    layer_slider = mo.ui.slider(0, 5, value=5, label="Show up to layer", show_value=True)
    return category_filter, circuit_condition, layer_slider


@app.cell(hide_code=True)
def _(circuit_condition, enrich_feature_table, mo, show_m3):
    mo.stop(not show_m3)
    feat_catalog = enrich_feature_table(circuit_condition.value)
    _syntactic_first = feat_catalog.sort_values(
        ["is_syntactic", "layer"], ascending=[False, True]
    )
    feature_options = {
        f"{row.Feature} — {row.Category}": row.Feature
        for _, row in _syntactic_first.head(24).iterrows()
    }
    feature_pick = mo.ui.dropdown(
        options=feature_options,
        value=next(iter(feature_options)),
        label="Inspect feature",
    )
    return feat_catalog, feature_options, feature_pick


@app.cell(hide_code=True)
def _(
    ASSETS_DIR,
    apply_plotly_theme,
    category_counts,
    category_filter,
    circuit_condition,
    circuit_svg,
    feat_catalog,
    feature_gallery_html,
    feature_options,
    feature_pick,
    group_ablation_figure,
    layer_narrative,
    layer_slider,
    load_group_effects,
    load_parquet_cache,
    mo,
    paper_figure_html,
    show_m3,
    spike_bar_html,
    theme,
):
    mo.stop(not show_m3)
    counts = category_counts(circuit_condition.value)
    if category_filter.value == "syntactic":
        filtered_feats = feat_catalog[feat_catalog["is_syntactic"]]
        counts = counts[counts["Category"].str.contains("detector|clause", case=False, na=False)]
    elif category_filter.value == "pro_gp":
        filtered_feats = feat_catalog[feat_catalog["reading_side"] == "pro_gp"]
        counts = counts[counts["reading_side"] == "pro_gp"]
    elif category_filter.value == "anti_gp":
        filtered_feats = feat_catalog[feat_catalog["reading_side"] == "anti_gp"]
        counts = counts[counts["reading_side"] == "anti_gp"]
    else:
        filtered_feats = feat_catalog
    counts = counts[counts["layer"] <= layer_slider.value]

    group_effects = load_group_effects(circuit_condition.value)
    if group_effects is not None:
        ablation_fig = apply_plotly_theme(group_ablation_figure(group_effects), theme)
        ablation_panel = mo.vstack(
            [
                ablation_fig,
                mo.md(
                    "Word-detector heuristics move m at least as much as the genuinely syntactic "
                    "detectors — direct evidence for the paper's claim that the mechanism mixes "
                    "**real syntax with shallow heuristics**."
                ),
            ]
        )
    else:
        ablation_panel = mo.callout(
            mo.md(
                "Group-ablation cache missing — run `uv run python precompute.py` (needs SAE "
                "checkpoints) to measure these effects."
            ),
            kind="warn",
        )

    gallery_feats = (
        filtered_feats.sort_values(["is_syntactic", "layer"], ascending=[False, True]).head(6)
    )
    narrative = layer_narrative(circuit_condition.value)

    _tok_act = load_parquet_cache(f"token_activations_{circuit_condition.value.lower()}.parquet")
    selected_feature = feature_pick.value
    if selected_feature in feature_options:
        selected_feature = feature_options[selected_feature]
    detail_html = mo.md("_Select a feature above for token-level spikes._")
    if selected_feature:
        feat_row = feat_catalog[feat_catalog["Feature"] == selected_feature]
        if not feat_row.empty and _tok_act is not None and not _tok_act.empty:
            fr = feat_row.iloc[0]
            ann = fr["Annotation"]
            fd = _tok_act[
                (_tok_act["feature"] == ann) | (_tok_act["category"] == fr["Category"])
            ]
            if not fd.empty:
                first = fd["sentence"].iloc[0]
                fd = fd[fd["sentence"] == first].sort_values("position")
                detail_html = mo.Html(
                    spike_bar_html(
                        fd["token"].tolist(),
                        fd["activation"].tolist(),
                        label=f"{selected_feature}: {ann}",
                    )
                )

    _content = [
        mo.md("## 3. Feature microscope — what is inside the circuit?"),
        mo.md(
            "The paper finds the circuit with AtP-IG (ÎE > 0.1), then hand-annotates every "
            "feature by inspecting its activating contexts. The catalogue below is the authors' "
            "own annotation table; the ablation chart then *verifies causally* what each family "
            "of features contributes. Red = pro-GP, blue = anti-GP."
        ),
        mo.hstack([circuit_condition, category_filter, layer_slider, feature_pick]),
        mo.Html(circuit_svg(counts)),
        ablation_panel,
        mo.accordion(
            {
                "The paper's own circuit sketch (Figure 3)": mo.Html(
                    paper_figure_html(
                        str(ASSETS_DIR / "paper_figures" / "fig3_circuit.png"),
                        "Paper Figure 3: the NP/Z circuit for “After the politician signed the "
                        "bill …”. Numbers are feature counts per node; red nodes push the GP "
                        "reading, blue nodes the non-GP reading.",
                    )
                )
            }
        ),
        mo.Html(feature_gallery_html(gallery_feats)),
        mo.accordion({f"Layer {layer}": text for layer, text in narrative if layer <= layer_slider.value}),
        detail_html,
    ]
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 4 — Intervention Sandbox (Fig 4) — CENTERPIECE
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m4 = module_nav.value == "m4"
    return (show_m4,)


@app.cell(hide_code=True)
def _(mo, saes_available, show_m4):
    mo.stop(not show_m4)
    sandbox_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure",
    )
    subject_slider = mo.ui.slider(0.0, 3.0, value=2.0, step=0.25, label="Subject-detector amp")
    object_slider = mo.ui.slider(0.0, 3.0, value=2.0, step=0.25, label="Object-detector amp")
    clause_slider = mo.ui.slider(0.0, 3.0, value=2.0, step=0.25, label="Clause-end amp")
    preset = mo.ui.radio(
        options={
            "Custom sliders": "custom",
            "Paper syntactic flip": "syntactic",
            "Baseline (no edit)": "baseline",
            "Random control": "random",
        },
        value="Paper syntactic flip",
        label="Preset",
        inline=True,
    )
    live_run = mo.ui.run_button(label="Re-run this setting live on the model")
    sae_download = (
        mo.ui.run_button(label="Download SAE checkpoints (~2.3 GB)")
        if not saes_available()
        else None
    )
    return clause_slider, live_run, object_slider, preset, sae_download, sandbox_condition, subject_slider


@app.cell(hide_code=True)
def _(
    ASSETS_DIR,
    apply_plotly_theme,
    clause_slider,
    colored_token_view,
    download_saes,
    gp_df,
    intervention_sweeps,
    interventions_df,
    live_mode_available,
    live_run,
    live_switch,
    load_parquet_cache,
    lookup_sweep_intervention,
    missing_sae_dirs,
    mo,
    object_slider,
    paper_figure_html,
    preset,
    px,
    run_intervention_suite,
    sae_download,
    saes_available,
    sandbox_condition,
    show_m4,
    subject_slider,
    theme,
    timed_call,
    tug_of_war_html,
):
    mo.stop(not show_m4)

    cond = sandbox_condition.value

    # One-click SAE fetch so live mode works on a fresh (e.g. molab) machine.
    _sae_note = None
    if sae_download is not None and sae_download.value and not saes_available():
        with mo.status.progress_bar(
            total=100,
            title="Fetching SAE checkpoints (~2.3 GB)",
            subtitle="from huggingface.co/saprmarks/pythia-70m-deduped-saes",
        ) as _bar:
            _progress = {"pct": 0}

            def _on_progress(phase: str, done: int, total: int) -> None:
                """Map download (0-90%) and extraction (90-100%) onto one bar."""
                if phase == "downloading":
                    pct = int(90 * done / max(total, 1))
                    subtitle = f"downloading {done} / {total} MB"
                else:
                    pct = 90 + int(10 * done / max(total, 1))
                    subtitle = f"extracting {done} / {total} files"
                if pct > _progress["pct"]:
                    _bar.update(increment=pct - _progress["pct"], subtitle=subtitle)
                    _progress["pct"] = pct

            download_saes(_on_progress)
        _sae_note = mo.callout(
            mo.md("SAE checkpoints installed — live interventions are now available."),
            kind="success",
        )

    if cond == "NPZ":
        _relevant_sliders = [subject_slider, clause_slider]
        sub_amp, obj_amp, cl_amp = subject_slider.value, 0.0, clause_slider.value
        _protocol_note = (
            "NP/Z protocol (paper §4.2): **upweight subject detectors** at the ambiguous noun, "
            "**amplify clause-end detectors** at the verb (and zero them afterwards), and always "
            "**clamp object detectors to 0** — pushing the net toward the non-GP reading."
        )
        if preset.value == "syntactic":
            sub_amp, cl_amp = 2.0, 2.0
    else:
        _relevant_sliders = [object_slider]
        sub_amp, obj_amp, cl_amp = 0.0, object_slider.value, 0.0
        _protocol_note = (
            "NP/S protocol (paper §4.2): **upweight object and end-of-sentence detectors** and "
            "always **clamp subject and CP-verb detectors to 0** — pushing the net toward the GP "
            "reading. (The direction is opposite to NP/Z because NP/S sentences start non-GP.)"
        )
        if preset.value == "syntactic":
            obj_amp = 2.0

    result = None
    if preset.value == "baseline" and interventions_df is not None:
        _row = interventions_df[
            (interventions_df["condition"] == cond)
            & (interventions_df["intervention"] == "baseline")
        ].iloc[0]
        result = {k: float(_row[k]) for k in ("mean_p_gp", "mean_p_non_gp", "mean_diff")}
    elif preset.value == "random" and interventions_df is not None:
        _row = interventions_df[
            (interventions_df["condition"] == cond)
            & (interventions_df["intervention"] == "random")
        ].iloc[0]
        result = {k: float(_row[k]) for k in ("mean_p_gp", "mean_p_non_gp", "mean_diff")}
    elif preset.value in {"custom", "syntactic"} and intervention_sweeps is not None:
        result = lookup_sweep_intervention(
            intervention_sweeps,
            cond,
            subject_amp=sub_amp,
            object_amp=obj_amp,
            clause_amp=cl_amp,
        )

    if live_run.value and preset.value in {"custom", "syntactic"}:
        if saes_available():
            try:
                result = timed_call(
                    "intervention",
                    run_intervention_suite,
                    gp_df,
                    cond,
                    subject_amp=sub_amp,
                    object_amp=obj_amp,
                    clause_amp=cl_amp,
                )
            except Exception as exc:  # noqa: BLE001
                mo.output.append(mo.callout(f"Live intervention failed: {exc}", kind="warn"))
        else:
            mo.output.append(
                mo.callout(
                    mo.md("Live interventions need the SAE checkpoints — download them above."),
                    kind="warn",
                )
            )

    example = gp_df[gp_df["condition"] == cond].iloc[0]["sentence_ambiguous"]

    if result is not None:
        result_panel = mo.Html(
            tug_of_war_html(result["mean_p_gp"], result["mean_p_non_gp"])
        )
    else:
        result_panel = mo.callout(
            mo.md(
                "No intervention caches found. Run `uv run python precompute.py` with SAE "
                "checkpoints present, or use the live button below."
            ),
            kind="warn",
        )

    sweep_fig = None
    if intervention_sweeps is not None:
        _grid = intervention_sweeps[
            (intervention_sweeps["condition"] == cond) & (~intervention_sweeps["use_random"])
        ]
        if cond == "NPZ" and not _grid.empty:
            _clause_values = sorted(_grid["clause_amp"].unique())
            _nearest_clause = min(_clause_values, key=lambda v: abs(v - cl_amp))
            _line = _grid[_grid["clause_amp"] == _nearest_clause].sort_values("subject_amp")
            _x_col, _x_now = "subject_amp", sub_amp
            _sweep_title = (
                f"Measured dose-response: subject-detector amp → m (clause amp = {_nearest_clause:g})"
            )
            _x_title = "Subject-detector amplitude"
        elif not _grid.empty:
            _line = _grid.sort_values("object_amp")
            _x_col, _x_now = "object_amp", obj_amp
            _sweep_title = "Measured dose-response: object-detector amp → m"
            _x_title = "Object-detector amplitude"
        else:
            _line = None
        if _line is not None:
            sweep_fig = px.line(
                _line,
                x=_x_col,
                y="mean_diff",
                markers=True,
                title=_sweep_title,
                labels={_x_col: _x_title, "mean_diff": "m = p(GP) − p(non-GP)"},
            )
            sweep_fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
            if result is not None:
                sweep_fig.add_scatter(
                    x=[_x_now],
                    y=[result["mean_diff"]],
                    mode="markers",
                    marker=dict(size=14, color="#f39c12", symbol="diamond"),
                    name="current setting",
                )
            apply_plotly_theme(sweep_fig, theme)

    paper_chart = None
    if interventions_df is not None:
        paper_chart = px.bar(
            interventions_df[interventions_df["condition"] == cond],
            x="intervention",
            y="mean_diff",
            color="intervention",
            title=f"Figure 4 reproduction ({cond}): syntactic edits flip m, random edits do not",
            labels={"mean_diff": "p(GP) − p(non-GP)"},
        )
        apply_plotly_theme(paper_chart, theme)

    _tok_act = load_parquet_cache(f"token_activations_{cond.lower()}.parquet")
    feat_panels: list = []
    if _tok_act is not None and not _tok_act.empty:
        _sent_data = _tok_act[_tok_act["sentence"] == _tok_act["sentence"].iloc[0]]
        for _side, _color in [("pro_gp", "#c0392b"), ("anti_gp", "#2980b9")]:
            _side_data = _sent_data[_sent_data["reading_side"] == _side]
            if _side_data.empty:
                continue
            _feat = _side_data["feature"].iloc[0]
            _fd = _side_data[_side_data["feature"] == _feat].sort_values("position")
            feat_panels.append(
                mo.iframe(
                    colored_token_view(
                        _fd["token"].tolist(),
                        _fd["activation"].tolist(),
                        label=f"Baseline activation — {_feat}",
                        positive_color=_color,
                    ),
                    height="150px",
                )
            )

    _live_hint = (
        "Live runs take ~1–2 s per setting on a GPU"
        if live_mode_available()
        else "No GPU detected — live runs take roughly 15–60 s per setting on CPU"
    )

    _content = [
        mo.md("## 4. Intervention sandbox — **flip the model's reading**"),
        mo.md(
            "The paper's headline causal test (Figure 4): clamp the annotated syntactic features "
            "and the preferred reading flips; clamp the same number of *random* features and "
            "nothing happens. The sliders below move through a grid of **real intervention runs** "
            "(one full model run per grid point, all 24 sentences each)."
        ),
        mo.md(_protocol_note),
        mo.md(f"**Example sentence:** _{example}_"),
        mo.hstack([sandbox_condition, preset]),
        mo.hstack(_relevant_sliders) if preset.value == "custom" else mo.md(""),
        result_panel,
    ]
    if sweep_fig is not None:
        _content.append(sweep_fig)
    if paper_chart is not None:
        _content.extend(
            [
                paper_chart,
                mo.accordion(
                    {
                        "Compare with the paper's original Figure 4": mo.Html(
                            paper_figure_html(
                                str(ASSETS_DIR / "paper_figures" / "fig4_causal.png"),
                                "Paper Figure 4 (Pythia-70m). Same qualitative result: the "
                                "syntactic intervention flips the sign of m for both structures; "
                                "the random-feature control does not.",
                            )
                        )
                    }
                ),
            ]
        )
    _content.append(mo.md(f"### Verify live · {_live_hint}"))
    if sae_download is not None:
        _content.append(sae_download)
    if _sae_note is not None:
        _content.append(_sae_note)
    _content.append(live_run)
    if feat_panels:
        _content.append(mo.md("### The features being clamped (baseline activations)"))
        _content.extend(feat_panels)
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 5 — One or Many Readings? (RQ2)
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m5 = module_nav.value == "m5"
    return (show_m5,)


@app.cell(hide_code=True)
def _(mo, show_m5):
    mo.stop(not show_m5)
    rq2_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure",
    )
    return (rq2_condition,)


@app.cell(hide_code=True)
def _(
    ASSETS_DIR,
    activation_heatmap,
    apply_plotly_theme,
    load_parquet_cache,
    mo,
    multi_feature_token_view,
    np,
    paper_figure_html,
    probe_cache,
    probe_figure5_plot,
    rq2_condition,
    show_m5,
    theme,
):
    mo.stop(not show_m5)
    _tok_act = load_parquet_cache(f"token_activations_{rq2_condition.value.lower()}.parquet")

    serial_svg = """<svg width="300" height="140"><text x="8" y="16" font-size="12" fill="currentColor">Serial parser (hypothesis)</text>
    <circle cx="80" cy="70" r="10" fill="#c0392b"/><text x="50" y="100" font-size="10" fill="currentColor">one reading</text></svg>"""
    parallel_svg = """<svg width="300" height="140"><text x="8" y="16" font-size="12" fill="currentColor">Parallel (what the paper finds)</text>
    <circle cx="70" cy="60" r="8" fill="#c0392b"/><circle cx="70" cy="90" r="8" fill="#2980b9"/>
    <text x="100" y="78" font-size="10" fill="currentColor">both readings active</text></svg>"""

    _content = [
        mo.md("## 5. One or many readings? (RQ2)"),
        mo.md(
            "Does the model commit to a single parse, or hedge? The paper measures the annotated "
            "features of *both* readings on ambiguous inputs: mean activations stay in the "
            "0.27–0.41 band for every feature group, and **more than 50% of each group's features "
            "stay active** — the model maintains both readings in parallel."
        ),
        rq2_condition,
    ]

    if _tok_act is not None and not _tok_act.empty:
        activation_fig = apply_plotly_theme(activation_heatmap(_tok_act), theme)
        _content.extend(
            [
                activation_fig,
            ]
        )
    else:
        _content.append(
            mo.callout(
                mo.md("Token-activation caches missing — run `uv run python precompute.py`."),
                kind="warn",
            )
        )

    probe_fig = apply_plotly_theme(probe_figure5_plot(probe_cache, rq2_condition.value), theme)
    _content.extend(
        [
            mo.md("### The structural-probe view"),
            mo.md(
                "A structural probe reads the *parse* out of hidden states: LEFT-ARC = the "
                "garden-path dependency, GEN = the non-GP alternative. Both actions keep "
                "non-trivial probability through the middle layers — the probe agrees that "
                "neither reading is discarded (except in the final layer, where probe quality "
                "collapses; paper §5)."
            ),
            probe_fig,
            mo.accordion(
                {
                    "The paper's original Figure 5": mo.Html(
                        paper_figure_html(
                            str(ASSETS_DIR / "paper_figures" / "fig5_probe.png"),
                            "Paper Figure 5: probe action probabilities at the ambiguous noun, "
                            "averaged over items, per layer of Pythia-70m.",
                        )
                    )
                }
            ),
            mo.hstack([mo.Html(serial_svg), mo.Html(parallel_svg)]),
        ]
    )

    if _tok_act is not None and not _tok_act.empty:
        _sent = _tok_act["sentence"].iloc[0]
        _data = _tok_act[_tok_act["sentence"] == _sent]
        pro = _data[_data["reading_side"] == "pro_gp"].groupby("position")["activation"].max()
        anti = _data[_data["reading_side"] == "anti_gp"].groupby("position")["activation"].max()
        tokens = (
            _data.sort_values("position")
            .drop_duplicates("position")["token"]
            .tolist()
        )
        if pro.size and anti.size:
            n = min(len(tokens), len(pro), len(anti))
            mat = np.vstack([pro.values[:n], anti.values[:n]])
            _content.extend(
                [
                    mo.md("### Both camps, token by token"),
                    mo.iframe(
                        multi_feature_token_view(
                            tokens[:n],
                            mat,
                            ["pro-GP features", "anti-GP features"],
                        ),
                        height="190px",
                    ),
                ]
            )
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 6 — Repair vs Reanalysis (RQ3)
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m6 = module_nav.value == "m6"
    return (show_m6,)


@app.cell(hide_code=True)
def _(mo, show_m6):
    mo.stop(not show_m6)
    gprc_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="GPRC structure",
    )
    return (gprc_condition,)


@app.cell(hide_code=True)
def _(
    CIRCUIT_IOU,
    gprc_condition,
    gprc_table_df,
    mo,
    sample_gprc_items,
    show_m6,
):
    mo.stop(not show_m6)
    qa_table = mo.ui.table(gprc_table_df())
    samples = sample_gprc_items(gprc_condition.value, n=3)
    iou = CIRCUIT_IOU.get(gprc_condition.value, 0.0)
    overlap_svg = f"""
    <svg width="520" height="160" xmlns="http://www.w3.org/2000/svg">
      <circle cx="150" cy="80" r="60" fill="#fadbd8" opacity="0.8"/>
      <circle cx="280" cy="80" r="60" fill="#d6eaf8" opacity="0.8"/>
      <text x="85" y="85" font-size="12" fill="#1c2833">GP circuit C₁</text>
      <text x="295" y="85" font-size="12" fill="#1c2833">GPRC C₂</text>
      <text x="200" y="85" font-size="11" fill="#922b21">IoU ≈ {iou:.1%}</text>
    </svg>
    """
    mo.vstack([
        mo.md("## 6. Repair vs reanalysis? (RQ3)"),
        mo.callout(
            mo.md(
                "**Model switch:** this module is about **Gemma-2-2b**. Pythia-70m answers "
                "garden-path comprehension questions at chance (50%, Table 3), so the paper can "
                "only study *question answering about* garden paths in the larger model."
            ),
            kind="info",
        ),
        mo.md(
            "After the garden path is disambiguated, does the model **repair** its "
            "representation (like humans do) or **reanalyze** from scratch? The paper's test: "
            "find the circuit Gemma uses to *answer questions* about garden-path sentences "
            "(GPRC), and check how much it overlaps the circuit that *parses* them."
        ),
        qa_table,
        mo.Html(overlap_svg),
        mo.md("### Sample GPRC questions (from the paper's dataset, shipped in this repo)"),
        gprc_condition,
        mo.ui.table(
            samples[
                ["condition", "Sentence_GP", "Comp_Question_Yes", "Comp_Question_No"]
            ]
        ),
        mo.callout(
            mo.md(
                "The paper's evidence points to **neither** repair nor reanalysis: the QA "
                "circuit shares almost no features with the parse circuit (IoU ≤ 0.2%) and "
                "instead leans on shallow yes/no features (\u201cCertainly\u201d, \u201cOf "
                "course\u201d). Comprehension behaves like a separate, heuristic pathway."
            ),
            kind="warn",
        ),
    ])
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 7 — Your Garden-Path Sentence
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m7 = module_nav.value == "m7"
    return (show_m7,)


@app.cell(hide_code=True)
def _(gp_df, mo, show_m7):
    mo.stop(not show_m7)
    example_rows = gp_df[gp_df["condition"].isin(["NPZ", "NPS"])].head(8)
    example_lookup: dict[str, tuple[str, str]] = {}
    for _, r in example_rows.iterrows():
        label = f"Item {int(r['item'])} ({r['condition']})"
        example_lookup[label] = (r["sentence_ambiguous"], r["condition"])
    example_pick = mo.ui.dropdown(
        options=list(example_lookup.keys()),
        value=list(example_lookup.keys())[1],
        label="Curated examples (instant)",
    )
    custom_sentence = mo.ui.text_area(
        label="Or type your own sentence",
        value="After the politician signed the bill",
    )
    custom_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure type",
    )
    use_custom = mo.ui.switch(label="Use custom text (needs live mode)", value=False)
    score_button = mo.ui.run_button(label="Score sentence")
    intervene_button = mo.ui.run_button(label="Try causal flip")
    return (
        custom_condition,
        custom_sentence,
        example_lookup,
        example_pick,
        intervene_button,
        score_button,
        use_custom,
    )


@app.cell(hide_code=True)
def _(
    behavioral_scored,
    colored_token_view,
    custom_condition,
    custom_sentence,
    example_lookup,
    example_pick,
    get_hf_model,
    gp_df,
    intervene_button,
    live_mode_available,
    live_switch,
    mo,
    run_intervention_suite,
    saes_available,
    score_button,
    score_sentence,
    show_m7,
    timed_call,
    top_next_cache,
    top_next_tokens,
    tug_of_war_html,
    use_custom,
):
    mo.stop(not show_m7)

    if use_custom.value:
        user_sentence, condition = custom_sentence.value, custom_condition.value
    else:
        user_sentence, condition = example_lookup[example_pick.value]

    live_scores = None
    tops_df = None
    attr_html = None
    intervention_result = None

    if score_button.value or (not use_custom.value and not live_switch.value):
        try:
            if live_switch.value:
                model, tokenizer = get_hf_model()
                from gp_notebook.device import get_torch_device

                device = get_torch_device()
                live_scores = timed_call(
                    "score",
                    score_sentence,
                    model,
                    tokenizer,
                    user_sentence,
                    condition,
                    device=device,
                )
                tops_df = top_next_tokens(model, tokenizer, user_sentence, k=8, device=device)
                from gp_notebook.attribution import token_attributions

                attr = token_attributions(model, tokenizer, user_sentence, condition, device=device, n_steps=12)
                attr_html = colored_token_view(attr["tokens"], attr["scores"], label="Token IG")
            else:
                cached = gp_df[
                    (gp_df["sentence_ambiguous"] == user_sentence) & (gp_df["condition"] == condition)
                ]
                if not cached.empty:
                    scored = None
                    if behavioral_scored is not None:
                        scored = behavioral_scored[
                            (behavioral_scored["sentence"] == user_sentence)
                            & (behavioral_scored["condition"] == condition)
                            & (behavioral_scored["input_type"] == "ambiguous")
                        ]
                    if scored is not None and not scored.empty:
                        score_row = scored.iloc[0]
                        live_scores = {
                            "p_gp": float(score_row["p_gp"]),
                            "p_non_gp": float(score_row["p_non_gp"]),
                            "diff": float(score_row["diff"]),
                        }
                    if top_next_cache is not None:
                        _cached_tops = top_next_cache[
                            (top_next_cache["item"] == int(cached.iloc[0]["item"]))
                            & (top_next_cache["condition"] == condition)
                        ]
                        if not _cached_tops.empty:
                            tops_df = _cached_tops[["token", "probability"]].copy()
        except Exception as exc:  # noqa: BLE001
            mo.output.append(mo.callout(f"Scoring failed: {exc}", kind="warn"))

    if intervene_button.value:
        if live_switch.value and saes_available():
            try:
                mini_df = gp_df.head(1).copy()
                mini_df.loc[0, "sentence_ambiguous"] = user_sentence
                mini_df.loc[0, "condition"] = condition
                intervention_result = timed_call(
                    "intervention",
                    run_intervention_suite,
                    mini_df,
                    condition,
                    subject_amp=2.0 if condition == "NPZ" else 0.0,
                    object_amp=0.0 if condition == "NPZ" else 2.0,
                    clause_amp=2.0 if condition == "NPZ" else 0.0,
                )
            except Exception as exc:  # noqa: BLE001
                mo.output.append(mo.callout(f"Intervention failed: {exc}", kind="warn"))
        else:
            mo.output.append(
                mo.callout(
                    mo.md(
                        "The causal flip needs **Live mode** (sidebar) and the SAE checkpoints "
                        "(one-click download in Module 4)."
                    ),
                    kind="warn",
                )
            )

    score_md = (
        mo.md(
            f"**{user_sentence}** ({condition})\n\n"
            f"p(GP)={live_scores['p_gp']:.4f}, p(non-GP)={live_scores['p_non_gp']:.4f}, "
            f"Δ={live_scores['diff']:+.4f}"
        )
        if live_scores
        else mo.md("_Click **Score sentence** or pick a curated example._")
    )

    _live_note = (
        "Live scoring runs on any device — a few seconds even on CPU. "
        + (
            "GPU detected, so everything here is fast."
            if live_mode_available()
            else "No GPU detected: scoring stays quick, but the causal flip takes ~15–60 s."
        )
    )

    _content = [
        mo.md("## 7. Build your own garden-path sentence"),
        mo.md(
            "**Extension:** apply the paper's metric and causal tools to novel input. "
            "Curated examples work instantly from caches; custom text runs the model live. "
            + _live_note
        ),
        example_pick,
        use_custom,
        custom_sentence if use_custom.value else mo.md(""),
        custom_condition,
        mo.hstack([score_button, intervene_button]),
        score_md,
    ]
    if tops_df is not None:
        _content.append(mo.ui.table(tops_df))
    if attr_html:
        _content.append(mo.iframe(attr_html, height="150px"))
    if intervention_result:
        _content.append(mo.Html(
            tug_of_war_html(intervention_result["mean_p_gp"], intervention_result["mean_p_non_gp"])
        ))
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 8 — Outro
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_m8 = module_nav.value == "m8"
    return (show_m8,)


@app.cell(hide_code=True)
def _(
    apply_plotly_theme,
    faithfulness_anchor_figure,
    mo,
    show_m8,
    theme,
):
    mo.stop(not show_m8)
    _content: list = [
        mo.md("## 8. Outro"),
        mo.md(
            """
### What this notebook recomputed vs. reports
"""
        ),
    ]

    _content.extend(
        [
            mo.md("### How much of the behavior do these circuits capture?"),
            apply_plotly_theme(faithfulness_anchor_figure(), theme),
            mo.md(
                """
### Closing takeaways

1. **RQ1 (Pythia-70m):** Ablating or clamping the annotated features reliably moves — and can
   flip — the model's preferred reading, while random-feature controls do nothing. The circuits
   mix genuine syntactic detectors with shallow lexical heuristics, and their limited
   faithfulness means they are part of the mechanism, not all of it.
2. **RQ2 (Pythia-70m):** Features for *both* readings stay active on ambiguous input (mean
   activations 0.27–0.41, over half of each group firing) — parallel maintenance rather than a
   single committed parse.
3. **RQ3 (Gemma-2-2b):** Question-answering circuits share almost no features with parsing
   circuits (IoU ≤ 0.2%) and lean on yes/no heuristics — evidence for neither human-style
   repair nor reanalysis.

*You just reverse-engineered incremental parsing in a 70M-parameter LM — one ambiguous noun at
a time.*
"""
            ),
            mo.md(
                """
### Where this could go next

- **Edge semantics:** the discovered circuits treat features as independent nodes; formalizing
  AND/OR interactions between detector families (a limitation the paper flags) would sharpen
  claims about *how* heuristics and syntax combine.
- **Scale and training data:** repeat the pipeline on models trained with cognitively plausible
  data budgets, and on larger models where QA is reliable, to map where heuristic detectors give
  way to robust syntax.
- **Close the probe gap:** Figure 5's final-layer collapse leaves the end of the parse pipeline
  uncertain; better-calibrated structural probes (or unambiguous control prefixes) could resolve
  whether readings are truly discarded there.
"""
            )
        ]
    )
    mo.vstack(_content)
    return


if __name__ == "__main__":
    app.run()
