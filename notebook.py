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

    # === DESIGN TOKENS (edit gp_notebook/theme_tokens.py :: GP_THEME) ===
    # Colours, fonts, and Google Fonts URL live in one dict so the look can be
    # retuned without hunting through CSS / Plotly / SVG helpers.
    from gp_notebook.theme_tokens import GP_THEME  # noqa: F401


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    """Inject CSS variables and font rules from GP_THEME into the page."""
    from gp_notebook.theme_tokens import css_variables_block

    mo.Html(css_variables_block())
    return


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

- **Problem:** Autoregressive LMs handle temporary syntactic ambiguities incrementally, but
  behavioural surprisal alone does not reveal *which internal features* drive the preferred reading.
- **Approach:** Sparse autoencoders (SAEs) yield monosemantic features; AtP-IG finds the circuit
  for $m = p(\\text{GP}) - p(\\text{non-GP})$; targeted clamping verifies causal role.
"""
    )
    findings = mo.md(
        """
- **Findings:** Pythia-70m mixes genuine syntactic detectors with shallow lexical heuristics, and
  keeps **both readings active in parallel**. Gemma-2-2b answers follow-up questions above chance,
  yet reuses almost none of the parse circuit (IoU ≤ 0.2%) — neither repair nor reanalysis.
"""
    )
    gist = mo.callout(
        mo.md(
            """
**Why incremental processing matters?** Humans process dialogue incrementally, which means that we understand language word by word as it is generated. This incremental processing also gives use the ability to predict the speaker's intent and revision of our interpretation of what is being said on the fly if necessary. This feature can make conversational AI, and as a result verbal communication with robots, more human-like and robust to phenonmena like pauses, interruptions, corrections, etc. that make current dialogue systems brittle. In modern AI agents, it can allow for real-time tool calling before an instruction is finished, and in a healthcare setting, it can provide with a less frustrating communication with invidivuals with dementia or other cognitive impairments.
"""
        ),
        kind="neutral",
    )
    lay_summary = mo.callout(
        mo.md(
            """
**In plain language:** Language models read sentences one word at a time. A *garden-path*
sentence invites one parse until a later word forces another. This notebook reverse-engineers
what fires inside **Pythia-70m** at the ambiguous noun, then switches to **Gemma-2-2b** for the
comprehension question — Pythia cannot answer follow-ups above chance (paper Table 3).
"""
        ),
        kind="info",
    )
    pipeline = mo.mermaid(
        """
flowchart LR
  locate[Locate features with AtP-IG]
  annotate[Annotate syntactic roles]
  intervene[Clamp to flip the reading]
  locate --> annotate --> intervene
"""
    )
    callouts = _wrap_with_class(
        mo.vstack([gist, lay_summary], gap=0.25),
        "gp-intro-callouts",
    )
    _wrap_with_class(
        mo.vstack(
            [
                hero,
                tldr,
                _wrap_with_class(pipeline, "gp-intro-pipeline"),
                findings,
                callouts,
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
        else "SAE checkpoints missing — one-click download in RQ1 → Intervene"
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
    import importlib

    import gp_notebook.theme_tokens as _theme_tokens
    import gp_notebook.viz as _viz

    # Marimo keeps sibling packages in sys.modules across edits; reload so new
    # helpers (sae_pipeline_svg, serial_parallel_svgs, …) are always visible.
    importlib.reload(_theme_tokens)
    importlib.reload(_viz)

    theme_color = _theme_tokens.theme_color
    activation_heatmap = _viz.activation_heatmap
    apply_plotly_theme = _viz.apply_plotly_theme
    attention_to_last_token_scores = _viz.attention_to_last_token_scores
    behavioral_figure = _viz.behavioral_figure
    circuit_svg = _viz.circuit_svg
    faithfulness_anchor_figure = _viz.faithfulness_anchor_figure
    group_ablation_figure = _viz.group_ablation_figure
    lookup_sweep_intervention = _viz.lookup_sweep_intervention
    paper_figure_html = _viz.paper_figure_html
    probe_figure5_plot = _viz.probe_figure5_plot
    reading_bubbles_html = _viz.reading_bubbles_html
    sae_pipeline_svg = _viz.sae_pipeline_svg
    serial_parallel_svgs = _viz.serial_parallel_svgs
    token_reveal_html = _viz.token_reveal_html
    tug_of_war_html = _viz.tug_of_war_html
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
        sae_pipeline_svg,
        saes_available,
        sample_gprc_items,
        score_sentence,
        serial_parallel_svgs,
        spike_bar_html,
        theme_color,
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
| **Circuit** | Minimal subgraph of features that reproduces a behaviour. |
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
def _(load_parquet_cache, mo, np, sae_pipeline_svg, show_m0, sparsity, spike_bar_html):
    mo.stop(not show_m0)
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, 32)
    sparse_features = np.maximum(raw - sparsity.value, 0)
    active = int((sparse_features > 0).sum())
    _content = [
        mo.md("### SAE primer"),
        mo.md(
            r"$x$ (activation) $\rightarrow$ $f = \mathrm{ReLU}(W_e(x - b_d) + b_e)$ $\rightarrow$ $\hat{x} = W_d f + b_d$"
        ),
        mo.Html(sae_pipeline_svg()),
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
    theme_color,
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
            "The stimuli are an adaptation of Arehalli et al. (2022), forced to equal token "
            "length so feature importance can be compared at fixed positions. Only the verb "
            "changes: *ambiguous* licenses both readings, *GP-forcing* makes the garden-path "
            "reading correct, and *non-GP* blocks it."
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
        prefix_fig.update_traces(
            selector=dict(name="p_gp"), line_color=theme_color("gp"), name="p(GP)"
        )
        prefix_fig.update_traces(
            selector=dict(name="p_non_gp"), line_color=theme_color("non_gp"), name="p(non-GP)"
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
                    positive_color=theme_color("highlight"),
                ),
                height="150px",
            ),
        ])

    _content.append(
        mo.callout(
            mo.md(
                "**Bridge to RQ1.** Behaviour alone shows *that* Pythia prefers one continuation; "
                "the next page asks *which features* cause that preference — syntactic detectors, "
                "shallow heuristics, or both."
            ),
            kind="info",
        )
    )
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# RQ1 — Syntactic features or heuristics? (Figs 2–4, one page)
# ═══════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(module_nav):
    show_rq1 = module_nav.value == "rq1"
    return (show_rq1,)


@app.cell(hide_code=True)
def _(mo, show_rq1):
    mo.stop(not show_rq1)
    rq1_panel = mo.ui.radio(
        options={
            "1 · Behaviour (Fig 2)": "behaviour",
            "2 · Circuit (Fig 3)": "circuit",
            "3 · Intervene (Fig 4)": "intervene",
        },
        value="1 · Behaviour (Fig 2)",
        label="RQ1 step",
        inline=True,
    )
    return (rq1_panel,)


@app.cell(hide_code=True)
def _(behavioral_scored, mo, rq1_panel, show_rq1):
    mo.stop(not show_rq1)
    show_rq1_behavior = rq1_panel.value == "behaviour"
    show_rq1_circuit = rq1_panel.value == "circuit"
    show_rq1_intervene = rq1_panel.value == "intervene"
    drill_df = None
    drill_table = None
    if show_rq1_behavior and behavioral_scored is not None:
        drill_df = behavioral_scored[behavioral_scored["input_type"] == "ambiguous"].copy()
        drill_table = mo.ui.table(
            drill_df[["item", "condition", "sentence", "p_gp", "p_non_gp", "diff"]],
            selection="single",
        )
    return drill_df, drill_table, show_rq1_behavior, show_rq1_circuit, show_rq1_intervene


@app.cell(hide_code=True)
def _(mo, show_rq1, show_rq1_circuit):
    mo.stop(not (show_rq1 and show_rq1_circuit))
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
def _(circuit_condition, enrich_feature_table, mo, show_rq1, show_rq1_circuit):
    mo.stop(not (show_rq1 and show_rq1_circuit))
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
def _(mo, saes_available, show_rq1, show_rq1_intervene):
    mo.stop(not (show_rq1 and show_rq1_intervene))
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
    return (
        clause_slider,
        live_run,
        object_slider,
        preset,
        sae_download,
        sandbox_condition,
        subject_slider,
    )


@app.cell(hide_code=True)
def _(mo, rq1_panel, show_rq1):
    mo.stop(not show_rq1)
    mo.vstack(
        [
            mo.md("## 2. RQ1 — Do LMs use syntactic features or shallow heuristics?"),
            mo.callout(
                mo.md(
                    "**Gist.** Many high-importance features are interpretable and syntax-related "
                    "(subjects, objects, clause ends), yet word-level detectors and uninterpretable "
                    "features also move $m$. Causal clamps on the syntactic groups flip the preferred "
                    "reading; random controls do not."
                ),
                kind="neutral",
            ),
            mo.md(
                "Walk the paper's RQ1 arc on one page: measure garden-path preferences "
                "(Fig 2), inspect the annotated circuit (Fig 3), then clamp features to flip "
                "the reading (Fig 4)."
            ),
            rq1_panel,
        ]
    )
    return


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
    show_rq1,
    show_rq1_behavior,
    theme,
    top_next_cache,
):
    mo.stop(not (show_rq1 and show_rq1_behavior))
    _content: list = [
        mo.md("### Behaviour — does Pythia get garden-pathed?"),
        mo.md(
            "Reproduction of the paper's **Figure 2**: mean $m = p(\\text{GP}) - p(\\text{non-GP})$ "
            "per structure and verb type. MV/RR is shown but excluded from the mechanistic "
            "analyses because Pythia barely garden-paths on it (paper §4.1)."
        ),
    ]
    if behavioral_summary is None or drill_table is None:
        _content.append(
            mo.callout(
                mo.md(
                    "Behavioural caches are missing. Run `uv run python precompute.py` once to "
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
        _content.extend(
            [
                behavior_fig,
                mo.Html(
                    paper_figure_html(
                        str(ASSETS_DIR / "paper_figures" / "fig2_behavioral.png"),
                        "Paper Figure 2 (Pythia-70m): positive m for NP/Z and MV/RR, negative "
                        "for NP/S; GP-forcing verbs amplify, non-GP verbs suppress.",
                    )
                ),
                mo.md("### Sentence drill-down"),
                drill_table,
                token_bar,
            ]
        )

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
    show_rq1,
    show_rq1_circuit,
    spike_bar_html,
    theme,
):
    mo.stop(not (show_rq1 and show_rq1_circuit))
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
                    "Word-detector heuristics move $m$ at least as much as the genuinely syntactic "
                    "detectors — direct evidence that the mechanism mixes **real syntax with "
                    "shallow heuristics**."
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
        mo.md("### Circuit — what is inside the feature set?"),
        mo.md(
            "AtP-IG keeps features with $\\hat{\\mathrm{IE}} > 0.1$; the authors then hand-annotate "
            "each one. Lower layers are mostly word detectors; upper layers encode subjects, "
            "objects, and clause boundaries. Red = pro-GP, blue = anti-GP."
        ),
        mo.hstack(
            [circuit_condition, category_filter, layer_slider, feature_pick],
            justify="start",
            gap=1.5,
            wrap=True,
        ),
        mo.Html(circuit_svg(counts)),
        ablation_panel,
        mo.Html(
            paper_figure_html(
                str(ASSETS_DIR / "paper_figures" / "fig3_circuit.png"),
                "Paper Figure 3: NP/Z circuit for “After the politician signed the bill …”. "
                "Numbers are feature counts; red nodes push GP, blue push non-GP.",
            )
        ),
        mo.Html(feature_gallery_html(gallery_feats)),
        mo.accordion(
            {f"Layer {layer}": text for layer, text in narrative if layer <= layer_slider.value}
        ),
        detail_html,
    ]
    mo.vstack(_content)
    return


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
    load_parquet_cache,
    lookup_sweep_intervention,
    mo,
    object_slider,
    paper_figure_html,
    preset,
    px,
    run_intervention_suite,
    sae_download,
    saes_available,
    sandbox_condition,
    show_rq1,
    show_rq1_intervene,
    subject_slider,
    theme,
    theme_color,
    timed_call,
    tug_of_war_html,
):
    mo.stop(not (show_rq1 and show_rq1_intervene))

    cond = sandbox_condition.value

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
            "NP/Z protocol (paper §4.3): **upweight subject detectors** at the ambiguous noun, "
            "**amplify clause-end detectors** at the verb (and zero them afterwards), and always "
            "**clamp object detectors to 0** — pushing the net toward the non-GP reading."
        )
        if preset.value == "syntactic":
            sub_amp, cl_amp = 2.0, 2.0
    else:
        _relevant_sliders = [object_slider]
        sub_amp, obj_amp, cl_amp = 0.0, object_slider.value, 0.0
        _protocol_note = (
            "NP/S protocol (paper §4.3): **upweight object detectors** and always **clamp "
            "subject and CP-verb detectors to 0** — pushing the net toward the GP reading "
            "(NP/S starts non-GP, so the flip direction is opposite to NP/Z)."
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
                    marker=dict(size=14, color=theme_color("highlight"), symbol="diamond"),
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
        for _side, _color in [
            ("pro_gp", theme_color("gp")),
            ("anti_gp", theme_color("non_gp")),
        ]:
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
        mo.md("### Intervene — flip the model's reading"),
        mo.md(
            "The paper's causal test (Figure 4): clamp annotated syntactic features and the "
            "preferred reading flips; clamp the same number of *random* features and nothing "
            "happens. Sliders move through a grid of **real intervention runs** (all 24 "
            "sentences per grid point)."
        ),
        mo.md(_protocol_note),
        mo.md(f"**Example sentence:** _{example}_"),
        mo.hstack([sandbox_condition, preset], justify="start", gap=1.5, wrap=True),
        mo.hstack(_relevant_sliders, justify="start", gap=1.5, wrap=True)
        if preset.value == "custom"
        else mo.md(""),
        result_panel,
    ]
    if sweep_fig is not None:
        _content.append(sweep_fig)
    if paper_chart is not None:
        _content.extend(
            [
                paper_chart,
                mo.Html(
                    paper_figure_html(
                        str(ASSETS_DIR / "paper_figures" / "fig4_causal.png"),
                        "Paper Figure 4 (Pythia-70m): syntactic intervention flips the sign of m; "
                        "the random-feature control does not.",
                    )
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
    serial_parallel_svgs,
    show_m5,
    theme,
):
    mo.stop(not show_m5)
    _tok_act = load_parquet_cache(f"token_activations_{rq2_condition.value.lower()}.parquet")
    serial_svg, parallel_svg = serial_parallel_svgs()

    _content = [
        mo.md("## 3. One or many readings? (RQ2)"),
        mo.callout(
            mo.md(
                "**Gist.** On ambiguous prefixes, pro-GP and anti-GP syntactic features both "
                "fire (mean activations 0.27–0.41; >50% of each group active). Structural probes "
                "agree: LEFT-ARC and GEN both keep non-trivial probability — parallel maintenance, "
                "not a single committed parse."
            ),
            kind="neutral",
        ),
        mo.md(
            "Does the model commit to one reading, or keep both? The paper checks annotated "
            "feature activations on ambiguous inputs and reads parse distributions out of "
            "hidden states with MLP action probes."
        ),
        rq2_condition,
    ]

    if _tok_act is not None and not _tok_act.empty:
        activation_fig = apply_plotly_theme(activation_heatmap(_tok_act), theme)
        _content.append(activation_fig)
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
                "LEFT-ARC = garden-path dependency; GEN = non-GP alternative. Both stay "
                "non-trivial through the middle layers (final-layer probe quality collapses; "
                "paper §5)."
            ),
            probe_fig,
            mo.Html(
                paper_figure_html(
                    str(ASSETS_DIR / "paper_figures" / "fig5_probe.png"),
                    "Paper Figure 5: probe action probabilities at the ambiguous noun, "
                    "averaged over items, per layer of Pythia-70m.",
                )
            ),
            mo.hstack(
                [mo.Html(serial_svg), mo.Html(parallel_svg)],
                justify="start",
                gap=1.5,
                wrap=True,
            ),
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
    theme_color,
):
    mo.stop(not show_m6)
    qa_table = mo.ui.table(gprc_table_df())
    samples = sample_gprc_items(gprc_condition.value, n=3)
    iou = CIRCUIT_IOU.get(gprc_condition.value, 0.0)
    gp_soft = theme_color("gp_soft")
    non_gp_soft = theme_color("non_gp_soft")
    gp = theme_color("gp")
    ink = theme_color("ink")
    overlap_svg = f"""
    <svg class="gp-fluid-svg gp-fluid-svg--md" viewBox="0 0 520 160" width="100%" height="auto"
         xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Near-zero overlap between parse and GPRC circuits">
      <circle cx="150" cy="80" r="60" fill="{gp_soft}" opacity="0.9"/>
      <circle cx="280" cy="80" r="60" fill="{non_gp_soft}" opacity="0.9"/>
      <text x="85" y="85" font-size="12" fill="{ink}">GP circuit C₁</text>
      <text x="295" y="85" font-size="12" fill="{ink}">GPRC C₂</text>
      <text x="200" y="85" font-size="11" fill="{gp}" font-weight="700">IoU ≈ {iou:.1%}</text>
    </svg>
    """
    mo.vstack([
        mo.md("## 4. Repair vs reanalysis? (RQ3)"),
        mo.callout(
            mo.md(
                "**Model switch:** this section is about **Gemma-2-2b**. Pythia-70m answers "
                "garden-path comprehension questions at chance (50%, Table 3), so the paper can "
                "only study *question answering about* garden paths in the larger model."
            ),
            kind="info",
        ),
        mo.callout(
            mo.md(
                "**Gist.** After disambiguation, Gemma's QA circuit shares almost no features "
                "with the parse circuit (IoU ≤ 0.2%) and leans on shallow yes/no heuristics — "
                "evidence for **neither** human-style repair nor reanalysis."
            ),
            kind="neutral",
        ),
        mo.md(
            "Operationally: *repair* would reuse reading-specific syntactic features after the "
            "disambiguating token; *reanalysis* would rebuild from reading-agnostic features. "
            "The paper compares the GPRC answering circuit to the initial parse circuit."
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
                "Comprehension here behaves like a separate, heuristic pathway — not a "
                "continuation of the syntactic circuit that set the preferred reading."
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
                        "(one-click download under RQ1 → Intervene)."
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
        mo.md("## 5. Build your own garden-path sentence"),
        mo.callout(
            mo.md(
                "**Extension.** Score novel prefixes with $m = p(\\text{GP}) - p(\\text{non-GP})$ "
                "and optionally clamp the same syntactic feature groups used in RQ1 — a direct "
                "check that the paper's causal story transfers beyond the curated set."
            ),
            kind="neutral",
        ),
        mo.md(
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
        mo.md("## 6. Outro"),
        mo.callout(
            mo.md(
                "**Gist, restated.** Sparse feature circuits show that incremental garden-path "
                "preferences in Pythia mix syntax with heuristics and keep both parses alive; "
                "Gemma's follow-up answers barely reuse that parse circuit."
            ),
            kind="neutral",
        ),
        mo.md("### How much of the behaviour do these circuits capture?"),
        apply_plotly_theme(faithfulness_anchor_figure(), theme),
        mo.md(
            """
Faithfulness far from 1.0 (e.g. 3.48 for NP/Z, 0.20 for NP/S) means these circuits are
*influential* but incomplete — negative-effect features and non-linear interactions remain
outside the annotated set (paper Appendix C). That caveat does not erase the causal evidence
for the features we *do* have.
"""
        ),
        mo.md(
            """
### Closing takeaways

1. **RQ1 (Pythia-70m):** Clamping annotated syntactic features flips the preferred reading;
   random controls do not. Circuits mix genuine syntactic detectors with shallow lexical
   heuristics, and limited faithfulness means they are part of the mechanism, not all of it.
2. **RQ2 (Pythia-70m):** Features for *both* readings stay active on ambiguous input (mean
   activations 0.27–0.41, over half of each group firing) — parallel maintenance rather than a
   single committed parse.
3. **RQ3 (Gemma-2-2b):** Question-answering circuits share almost no features with parsing
   circuits (IoU ≤ 0.2%) and lean on yes/no heuristics — neither human-style repair nor
   reanalysis.
"""
        ),
        mo.md(
            """
### Where this could go next

- **Evaluating Recurrent Models:** The same pipeline can be applied to recurrent models like xLSTMs ([Beck et al. 2025](https://www.alphaxiv.org/abs/2405.04517)) that have shown superior performance compared to both Transformers and State Space Models across many benchmarks, to study the mechanisms of incremental parsing in these models, and how model compression techniques like pruning and quantisation affect the incremental performance (for Edge AI applications).
- **Extending Evaluation Pipeline:** A standardised incrementality evaluation pipeline (model in, scores out) can be created to evaluate the incremental performance of other models, which could as well include the diachronic metrics of [Baumann and Schlangen, 2011](https://aclanthology.org/2011.dnd-2.10/) and "Triangular Structures" of [Madureira et al. 2024](https://www.alphaxiv.org/abs/2402.13113). The correlation between these metrics can then be studied, which in turn could shed light on the possibility of using the diachronic metrics over mechanistic explanations due to simplicity and efficiency.
- **Guaranteed Circuit Validity:** Current circuit discovery methods are known to not generalise out of distribution robustly. Further work can follow "Certified Circuits" ([Anani et al. 2026](https://www.alphaxiv.org/abs/2602.22968)) to ensure provably stable circuit discovery.
"""
        ),
    ]
    mo.vstack(_content)
    return


if __name__ == "__main__":
    app.run()
