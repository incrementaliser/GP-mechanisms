"""Marimo notebook: Garden Path Sentence Processing Mechanisms in LMs."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from gp_notebook.viz import notebook_hero_html as _hero_html

    hero = mo.Html(_hero_html())
    tldr = mo.md(
        """
### TL;DR

- **Problem:** Autoregressive LMs handle syntactic ambiguities word-by-word, but we do not know
  *which internal features* drive their preferred reading of garden-path sentences.
- **Approach:** Sparse autoencoders (SAEs) decompose activations into interpretable features;
  attribution patching (AtP-IG) finds causally relevant circuits; targeted clamping verifies them.
- **Finding:** Pythia-70m uses real syntactic detectors (subjects, objects, clause ends) *and*
  spurious heuristics; it represents **both readings at once**, but does not reuse those features
  when answering follow-up questions.
"""
    )
    lay_summary = mo.callout(
        mo.md(
            """
**In plain language:** Language models read sentences incrementally — one word at a time —
like humans listening to speech. A *garden-path* sentence tricks you into one interpretation
until a later word reveals another. This paper **reverse-engineers** (mechanistic interpretability)
what happens inside Pythia-70m at the ambiguous noun: which internal "detectors" fire, whether both
readings coexist, and whether the model revises its parse when disambiguated.
"""
        ),
        kind="info",
    )
    reader_note = mo.md(
        "_Best experienced with **code hidden** and **vertical layout**. "
        "Run all cells once. Default mode uses precomputed caches; enable **Live mode** in the "
        "sidebar for GPU-backed interventions and custom sentences._"
    )
    recommended = mo.callout(
        "**Recommended path:** M1 Feel the garden path → M3 Behavioral lab → "
        "M4 Feature microscope → **M5 Flip the reading** → M6 Multiple readings → "
        "M7 Repair vs reanalysis → M8 Your sentence",
        kind="success",
    )
    mo.vstack([hero, tldr, lay_summary, reader_note, recommended])
    return


@app.cell
def _(mo):
    from gp_notebook.cache_status import cache_status_markdown, missing_required_caches
    from gp_notebook.device import device_status_message, live_mode_available as _live_mode_available
    from gp_notebook.interventions import saes_available as _sidebar_saes_available
    from gp_notebook.runtime import runtime_status_line as _sidebar_runtime_status

    live_switch = mo.ui.switch(label="Live mode (GPU model + SAEs)", value=False)
    module_nav = mo.ui.radio(
        options={
            "0 — Mech-interp primer": "m0",
            "1 — Feel the garden path": "m1",
            "2 — SAE primer": "m2",
            "3 — Behavioral lab (Fig 2)": "m3",
            "4 — Feature microscope (Fig 3)": "m4",
            "5 — Intervention sandbox (Fig 4)": "m5",
            "6 — Multiple readings? (RQ2)": "m6",
            "7 — Repair vs reanalysis (RQ3)": "m7",
            "8 — Your garden-path sentence": "m8",
        },
        value="1 — Feel the garden path",
        label="Section",
    )
    device_line = device_status_message()
    cache_line = cache_status_markdown()
    runtime_line = _sidebar_runtime_status()
    sae_line = "SAE checkpoints: available" if _sidebar_saes_available() else "SAE checkpoints: missing"
    missing = missing_required_caches()
    banner_kind = "info" if not missing else "warn"
    banner = mo.callout(
        mo.vstack(
            [
                mo.md(device_line),
                mo.md(cache_line),
                mo.md(sae_line),
                mo.md(runtime_line),
            ]
        ),
        kind=banner_kind if _live_mode_available() or not missing else "warn",
    )
    sidebar = mo.sidebar(
        mo.vstack(
            [
                mo.md("### Navigation"),
                module_nav,
                mo.md("### Execution"),
                live_switch,
                banner,
            ]
        )
    )
    sidebar
    return live_switch, module_nav


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from gp_notebook.device import live_mode_available
    from gp_notebook.atp_ig import load_atp_ig_cache, rank_features_by_ie
    from gp_notebook.behavior import (
        MODEL_NAME,
        continuation_tokens_for_condition,
        score_sentence,
        top_next_tokens,
    )
    from gp_notebook.features import (
        category_counts,
        enrich_feature_table,
        layer_narrative,
        representative_activation_matrix,
    )
    from gp_notebook.gprc import (
        CIRCUIT_IOU,
        gprc_table_df,
        load_gprc_dataset,
        sample_gprc_items,
    )
    from gp_notebook.interp_views import (
        attention_view,
        colored_token_view,
        multi_feature_token_view,
    )
    from gp_notebook.interventions import (
        build_feature_edits,
        feature_token_activations,
        run_intervention_suite,
        saes_available,
    )
    from gp_notebook.neuronpedia import feature_card_html, feature_gallery_html, spike_bar_html
    from gp_notebook.paths import ASSETS_DIR, load_gp_dataset, load_json_cache, load_parquet_cache
    from gp_notebook.probes import load_probe_cache
    from gp_notebook.runtime import get_hf_model, runtime_status_line, timed_call
    from gp_notebook.viz import (
        activation_heatmap,
        attention_to_last_token_scores,
        behavioral_figure,
        circuit_svg,
        faithfulness_tradeoff,
        interpolate_intervention,
        probe_figure5_plot,
        reading_bubbles_html,
        token_reveal_html,
        tug_of_war_html,
    )

    gp_df = load_gp_dataset()
    behavioral_summary = load_parquet_cache("behavioral_summary.parquet")
    behavioral_scored = load_parquet_cache("behavioral_scored.parquet")
    interventions_df = load_parquet_cache("interventions.parquet")
    intervention_sweeps = load_parquet_cache("intervention_sweeps.parquet")
    top_next_cache = load_parquet_cache("top_next_tokens.parquet")
    atp_ig_cache = load_atp_ig_cache()
    probe_cache = load_probe_cache()
    gprc_summary = load_json_cache("gprc_summary.json") or {}
    metadata = load_json_cache("metadata.json") or {}
    return (
        ASSETS_DIR,
        MODEL_NAME,
        activation_heatmap,
        atp_ig_cache,
        attention_to_last_token_scores,
        attention_view,
        behavioral_figure,
        behavioral_scored,
        behavioral_summary,
        build_feature_edits,
        category_counts,
        circuit_svg,
        colored_token_view,
        continuation_tokens_for_condition,
        enrich_feature_table,
        faithfulness_tradeoff,
        feature_card_html,
        feature_gallery_html,
        feature_token_activations,
        go,
        gp_df,
        gprc_summary,
        gprc_table_df,
        interpolate_intervention,
        intervention_sweeps,
        interventions_df,
        layer_narrative,
        live_mode_available,
        load_gprc_dataset,
        load_json_cache,
        load_parquet_cache,
        load_probe_cache,
        metadata,
        multi_feature_token_view,
        np,
        probe_cache,
        probe_figure5_plot,
        px,
        rank_features_by_ie,
        reading_bubbles_html,
        representative_activation_matrix,
        run_intervention_suite,
        runtime_status_line,
        saes_available,
        sample_gprc_items,
        score_sentence,
        spike_bar_html,
        timed_call,
        token_reveal_html,
        top_next_cache,
        top_next_tokens,
        tug_of_war_html,
        get_hf_model,
        CIRCUIT_IOU,
    )


# ═══════════════════════════════════════════════════════════════════
# CHAPTER 0 — Mechanistic Interpretability Primer
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m0 = module_nav.value == "m0"
    return (show_m0,)


@app.cell
def _(attention_view, load_json_cache, mo, show_m0):
    mo.stop(not show_m0)
    m0_intro = mo.md(
        r"""
## 0. What is mechanistic interpretability?

**Mechanistic interpretability** aims to reverse-engineer what neural networks
*actually compute*, not just what they get right.

| Concept | One-liner |
|---------|-----------|
| **Neuron** | A single unit — often *polysemantic* (fires for unrelated things). |
| **Feature** | A *monosemantic* direction found via sparse autoencoders (SAEs). |
| **Circuit** | Minimal subgraph of features that reproduces a behavior. |
| **Ablation** | Zero a feature and measure the effect — the core causal test. |
| **AtP-IG** | Gradient-based estimate of each feature's causal contribution. |
| **Faithfulness** | How well a circuit matches the full model (ideally ≈ 1.0). |

We apply these tools to **garden-path sentences** — inputs with two competing syntactic readings.
"""
    )
    glossary = mo.accordion(
        {
            "Sparse Autoencoder (SAE)": (
                r"$f = \mathrm{ReLU}(W_e(x - b_d) + b_e)$, $\hat{x} = W_d f + b_d$. "
                "Each dimension of **f** is an interpretable feature."
            ),
            "Garden-path metric": (
                "m = p(GP) − p(non-GP). GP = comma/period continuation; non-GP = ' was'."
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
    roadmap = mo.md(
        """
### Notebook roadmap

| Ch. | Title | Paper | RQ |
|-----|-------|-------|-----|
| 1 | Feel the garden path | §4.1 setup | — |
| 3 | Behavioral lab | Figure 2 | RQ1 |
| 4 | Feature microscope | Figures 1–3, Table 2 | RQ1 |
| 5 | Intervention sandbox | Figure 4 | RQ1 |
| 6 | Multiple readings | Figure 5 | RQ2 |
| 7 | Repair vs reanalysis | Table 3, §6 | RQ3 |
| 8 | Your sentence | Extension | — |
"""
    )
    _content = [m0_intro, mo.md("### Key concepts"), glossary]
    if layer_select is not None:
        _content.append(mo.md("### Attention patterns"))
        _content.append(layer_select)
    _content.append(roadmap)
    mo.vstack(_content)
    return (attn_cache, layer_select)


@app.cell
def _(attn_cache, attention_view, layer_select, mo, np, show_m0):
    mo.stop(not show_m0)
    _attn_panel = mo.md("_Attention cache not loaded. Run `uv run python precompute.py`._")
    if attn_cache is not None and layer_select is not None:
        layer_idx = layer_select.value
        attn_data = np.array(attn_cache["layers"][layer_idx])
        _tokens = attn_cache["tokens"]
        attn_html = attention_view(_tokens, attn_data)
        _attn_panel = mo.vstack([
            mo.md(f"**Layer {layer_idx}** — {attn_data.shape[0]} heads"),
            mo.iframe(attn_html),
        ])
    _attn_panel
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — Feel the Garden Path
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m1 = module_nav.value == "m1"
    return (show_m1,)


@app.cell
def _(gp_df, mo, show_m1):
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
    reveal = mo.ui.slider(1, 8, value=6, label="Tokens revealed", show_value=True)
    return reveal, structure, verb_type


@app.cell
def _(
    attention_to_last_token_scores,
    colored_token_view,
    gp_df,
    load_json_cache,
    load_parquet_cache,
    mo,
    px,
    reading_bubbles_html,
    reveal,
    show_m1,
    structure,
    token_reveal_html,
    verb_type,
    np,
):
    mo.stop(not show_m1)
    col = f"sentence_{verb_type.value}"
    subset = gp_df[gp_df["condition"] == structure.value].iloc[0]
    gp_sentence = subset[col]
    _tokens = gp_sentence.split()
    highlight = len(_tokens) - 1
    if structure.value == "NPZ":
        gp_read = "Final noun = object of subordinate verb (clause can end → comma)."
        non_gp_read = "Final noun = matrix subject (continuation → was …)."
    else:
        gp_read = "Final noun = object of matrix verb (sentence can end → period)."
        non_gp_read = "Final noun = sentential subject (continuation → was …)."

    winner = None
    prefix_cache = load_parquet_cache(f"prefix_probs_{structure.value.lower()}.parquet")
    if prefix_cache is not None and verb_type.value == "ambiguous":
        row = prefix_cache[prefix_cache["n_tokens"] == min(reveal.value, len(_tokens))]
        if not row.empty:
            winner = "gp" if row.iloc[-1]["p_gp"] > row.iloc[-1]["p_non_gp"] else "non_gp"

    _content = [
        mo.md("## 1. Feel the garden path"),
        mo.md(f"**Item {int(subset['item'])}** — read token by token:"),
        mo.Html(token_reveal_html(_tokens, highlight_idx=highlight, reveal_count=reveal.value)),
        mo.Html(reading_bubbles_html(gp_read, non_gp_read, winner=winner)),
    ]

    if prefix_cache is not None:
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
        _content.extend([mo.md("### Model commitment timeline"), prefix_fig])

    attn_path = f"attention_{structure.value.lower()}.json"
    tok_act_cache = load_json_cache(attn_path)
    if tok_act_cache is not None and verb_type.value == "ambiguous":
        tok_strs = tok_act_cache["tokens"]
        n_shown = min(reveal.value, len(tok_strs))
        layer0 = np.array(tok_act_cache["layers"]["0"])
        attn_to_last = attention_to_last_token_scores(layer0, n_shown)
        _content.extend([
            mo.md("### Attention at ambiguous noun (layer 0, mean over heads)"),
            mo.Html(
                colored_token_view(
                    tok_strs[:n_shown],
                    attn_to_last,
                    label="Darker = more attention flowing to final token",
                    positive_color="#e67e22",
                )
            ),
        ])

    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — SAE Primer
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m2 = module_nav.value == "m2"
    return (show_m2,)


@app.cell
def _(mo, show_m2):
    mo.stop(not show_m2)
    sparsity = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Sparsity threshold")
    return (sparsity,)


@app.cell
def _(colored_token_view, load_parquet_cache, mo, np, show_m2, sparsity, spike_bar_html):
    mo.stop(not show_m2)
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, 32)
    sparse_features = np.maximum(raw - sparsity.value, 0)
    active = int((sparse_features > 0).sum())
    sae_svg = """
    <svg width="640" height="120" xmlns="http://www.w3.org/2000/svg">
      <text x="10" y="30" font-size="14">x (activation)</text>
      <text x="280" y="30" font-size="14">f = ReLU(W_e(x - b_d) + b_e)</text>
      <text x="520" y="30" font-size="14">x̂ = W_d f + b_d</text>
      <rect x="60" y="50" width="80" height="40" fill="#d5dbdb" />
      <polygon points="160,70 220,70 240,50 240,90 220,70" fill="#566573"/>
      <rect x="260" y="50" width="80" height="40" fill="#f9e79f" />
      <polygon points="360,70 420,70 440,50 440,90 420,70" fill="#566573"/>
      <rect x="460" y="50" width="80" height="40" fill="#d5dbdb" />
    </svg>
    """
    _content = [
        mo.md("## 2. SAE primer"),
        mo.Html(sae_svg),
        mo.md(f"Toy demo: **{active}** of 32 features active at threshold {sparsity.value:.2f}."),
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
                        label=f"Real SAE spike: {feat_name}",
                    )
                )
            )
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — Behavioral Lab (Fig 2)
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m3 = module_nav.value == "m3"
    return (show_m3,)


@app.cell
def _(behavioral_scored, behavioral_summary, mo, show_m3):
    mo.stop(not show_m3)
    drill_df = None
    drill_table = None
    if behavioral_summary is not None and behavioral_scored is not None:
        drill_df = behavioral_scored[behavioral_scored["input_type"] == "ambiguous"].copy()
        drill_table = mo.ui.table(
            drill_df[["item", "condition", "sentence", "p_gp", "p_non_gp", "diff"]],
            selection="single",
        )
    return drill_df, drill_table


@app.cell
def _(
    behavioral_figure,
    behavioral_summary,
    colored_token_view,
    drill_df,
    drill_table,
    load_json_cache,
    mo,
    px,
    show_m3,
    top_next_cache,
):
    mo.stop(not show_m3)
    _content: list = [
        mo.md("## 3. Behavioral lab — does Pythia get garden-pathed?"),
        mo.md("Reproduction of **Figure 2**. MV/RR is shown but excluded from later analyses (paper §4.1)."),
    ]
    if behavioral_summary is None:
        _content.append(mo.callout("Run `uv run python precompute.py`.", kind="warn"))
    else:
        behavior_fig = behavioral_figure(behavioral_summary)
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
        _content.extend([behavior_fig, mo.md("### Sentence drill-down"), drill_table, token_bar])

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
                    title="Top next-token probabilities at critical position",
                )
                _content.append(next_fig)

        cond_key = selected["condition"].lower()
        attr_cache = load_json_cache(f"attributions_{cond_key}.json")
        if attr_cache is not None:
            _content.append(
                mo.Html(
                    colored_token_view(
                        attr_cache["tokens"],
                        attr_cache["scores"],
                        label="Token IG: m = p(GP) − p(non-GP)",
                    )
                )
            )
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 4 — Feature Microscope (Fig 3)
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m4 = module_nav.value == "m4"
    return (show_m4,)


@app.cell
def _(mo, show_m4):
    mo.stop(not show_m4)
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
    ie_threshold = mo.ui.slider(0.05, 0.5, value=0.1, step=0.05, label="AtP-IG ÎE threshold")
    return category_filter, circuit_condition, ie_threshold, layer_slider


@app.cell
def _(
    category_filter,
    circuit_condition,
    enrich_feature_table,
    ie_threshold,
    layer_slider,
    mo,
    rank_features_by_ie,
    show_m4,
):
    mo.stop(not show_m4)
    feat_catalog = enrich_feature_table(circuit_condition.value)
    ranked = rank_features_by_ie(circuit_condition.value, threshold=ie_threshold.value)
    ranked = ranked[ranked["layer"] <= layer_slider.value]
    options = {f"{row.Feature} ({row.Category})": row.Feature for _, row in ranked.head(20).iterrows()}
    if not options:
        options = {f"{row.Feature}": row.Feature for _, row in feat_catalog.head(10).iterrows()}
    option_labels = list(options.keys())
    feature_pick = mo.ui.dropdown(
        options=options,
        value=option_labels[0],
        label="Inspect feature",
    )
    return feature_pick, feat_catalog, ranked


@app.cell
def _(
    atp_ig_cache,
    category_counts,
    category_filter,
    circuit_condition,
    circuit_svg,
    colored_token_view,
    feat_catalog,
    feature_gallery_html,
    feature_pick,
    ie_threshold,
    layer_narrative,
    layer_slider,
    load_parquet_cache,
    mo,
    px,
    ranked,
    show_m4,
    spike_bar_html,
):
    mo.stop(not show_m4)
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
    in_circuit = ranked[ranked["in_circuit"]]

    ie_fig = px.bar(
        ranked.head(12),
        x="Feature",
        y="ie_hat",
        color="reading_side",
        title="AtP-IG demo: top features by estimated indirect effect",
        labels={"ie_hat": "ÎE (demo)", "Feature": "Feature id"},
    )

    gallery_feats = in_circuit.head(6) if not in_circuit.empty else ranked.head(6)
    narrative = layer_narrative(circuit_condition.value)

    _tok_act = load_parquet_cache(f"token_activations_{circuit_condition.value.lower()}.parquet")
    selected_feature = feature_pick.value
    if selected_feature in options:
        selected_feature = options[selected_feature]
    detail_html = mo.md("_Select a feature above for token-level spikes._")
    if selected_feature and selected_feature != "0":
        feat_row = filtered_feats[filtered_feats["Feature"] == selected_feature]
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
        mo.md("## 4. Feature microscope — SAE circuits (Figure 3)"),
        mo.md(
            "Browse annotated features like [Neuronpedia](https://www.neuronpedia.org/pythia-70m-deduped). "
            "Red = pro-GP; blue = anti-GP."
        ),
        mo.hstack([circuit_condition, category_filter, layer_slider, ie_threshold, feature_pick]),
        mo.Html(circuit_svg(counts)),
        ie_fig,
        mo.Html(feature_gallery_html(gallery_feats)),
        mo.accordion({f"Layer {layer}": text for layer, text in narrative if layer <= layer_slider.value}),
        detail_html,
    ]
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 5 — Intervention Sandbox (Fig 4) — CENTERPIECE
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m5 = module_nav.value == "m5"
    return (show_m5,)


@app.cell
def _(interventions_df, live_switch, mo, show_m5):
    mo.stop(not show_m5)
    sandbox_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure",
    )
    subject_slider = mo.ui.slider(-3.0, 3.0, value=2.0, step=0.1, label="Subject detector amp")
    object_slider = mo.ui.slider(-3.0, 3.0, value=0.0, step=0.1, label="Object detector amp")
    clause_slider = mo.ui.slider(-3.0, 3.0, value=2.0, step=0.1, label="Clause-end detector amp")
    random_toggle = mo.ui.switch(label="Random features (null control)", value=False)
    preset = mo.ui.radio(
        options={
            "Custom sliders": "custom",
            "Paper syntactic flip": "syntactic",
            "Baseline (no edit)": "baseline",
            "Random control": "random",
        },
        value="Custom sliders",
        label="Preset",
        inline=True,
    )
    sentence_pick = mo.ui.slider(1, 24, value=2, label="Example item", show_value=True)
    return (
        clause_slider,
        interventions_df,
        live_switch,
        object_slider,
        preset,
        random_toggle,
        sandbox_condition,
        sentence_pick,
        subject_slider,
    )


@app.cell
def _(
    clause_slider,
    colored_token_view,
    gp_df,
    interpolate_intervention,
    interventions_df,
    live_switch,
    load_parquet_cache,
    live_mode_available,
    mo,
    object_slider,
    preset,
    px,
    random_toggle,
    run_intervention_suite,
    saes_available,
    sandbox_condition,
    sentence_pick,
    show_m5,
    subject_slider,
    timed_call,
    tug_of_war_html,
):
    mo.stop(not show_m5)

    cond = sandbox_condition.value
    sub_amp = subject_slider.value
    obj_amp = object_slider.value
    cl_amp = clause_slider.value
    use_random = random_toggle.value

    if preset.value == "syntactic":
        sub_amp, obj_amp, cl_amp, use_random = 2.0, 0.0 if cond == "NPZ" else 2.0, 2.0 if cond == "NPZ" else 0.0, False
    elif preset.value == "baseline":
        sub_amp, obj_amp, cl_amp, use_random = 0.0, 0.0, 0.0, False
    elif preset.value == "random":
        use_random = True

    baseline_row = syntactic_row = None
    if interventions_df is not None:
        baseline_row = interventions_df[
            (interventions_df["condition"] == cond) & (interventions_df["intervention"] == "baseline")
        ].iloc[0]
        syntactic_row = interventions_df[
            (interventions_df["condition"] == cond) & (interventions_df["intervention"] == "syntactic")
        ].iloc[0]

    result = {"mean_p_gp": 0.1, "mean_p_non_gp": 0.05, "mean_diff": 0.05, "source": "fallback"}
    if baseline_row is not None and syntactic_row is not None:
        result = interpolate_intervention(
            baseline_row.to_dict(),
            syntactic_row.to_dict(),
            subject_amp=sub_amp,
            object_amp=obj_amp,
            clause_amp=cl_amp,
            use_random=use_random,
        )

    if live_switch.value and live_mode_available() and saes_available() and preset.value in {"custom", "syntactic"}:
        try:
            live = timed_call(
                "intervention",
                run_intervention_suite,
                gp_df,
                cond,
                subject_amp=sub_amp,
                object_amp=obj_amp,
                clause_amp=cl_amp,
                use_random=use_random,
            )
            result.update(live)
            result["source"] = "live-gpu"
        except Exception as exc:  # noqa: BLE001
            mo.output.append(mo.callout(f"Live intervention failed: {exc}", kind="warn"))

    example = gp_df[
        (gp_df["condition"] == cond) & (gp_df["item"] == sentence_pick.value)
    ].iloc[0]["sentence_ambiguous"]

    paper_chart = None
    if interventions_df is not None:
        paper_chart = px.bar(
            interventions_df[interventions_df["condition"] == cond],
            x="intervention",
            y="mean_diff",
            color="intervention",
            title=f"Paper Figure 4 ({cond})",
            labels={"mean_diff": "p(GP) − p(non-GP)"},
        )

    _tok_act = load_parquet_cache(f"token_activations_{cond.lower()}.parquet")
    feat_panels: list = []
    if _tok_act is not None and not _tok_act.empty:
        _sent_data = _tok_act[_tok_act["sentence"].str.contains(example.split()[-1], na=False)]
        if _sent_data.empty:
            _sent_data = _tok_act[_tok_act["sentence"] == _tok_act["sentence"].iloc[0]]
        else:
            _sent_data = _sent_data[_sent_data["sentence"] == _sent_data["sentence"].iloc[0]]
        for _side, _color in [("pro_gp", "#c0392b"), ("anti_gp", "#2980b9")]:
            _side_data = _sent_data[_sent_data["reading_side"] == _side]
            if _side_data.empty:
                continue
            _feat = _side_data["feature"].iloc[0]
            _fd = _side_data[_side_data["feature"] == _feat].sort_values("position")
            feat_panels.append(
                mo.Html(
                    colored_token_view(
                        _fd["token"].tolist(),
                        _fd["activation"].tolist(),
                        label=f"{'Clamped' if preset.value == 'syntactic' else 'Baseline'}: {_feat}",
                        positive_color=_color,
                    )
                )
            )

    mo.vstack([
        mo.md("## 5. Intervention sandbox — **flip the model's reading**"),
        mo.md(f"**Sentence:** _{example}_"),
        mo.hstack([sandbox_condition, sentence_pick, preset]),
        mo.hstack([subject_slider, object_slider, clause_slider, random_toggle]),
        mo.Html(tug_of_war_html(result["mean_p_gp"], result["mean_p_non_gp"])),
        mo.md(f"**Δ = {result['mean_diff']:+.4f}** · source: `{result.get('source', 'unknown')}`"),
        paper_chart,
        mo.md("### SAE feature activations on example sentence"),
        *feat_panels,
    ])
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 6 — One or Many Readings? (RQ2)
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m6 = module_nav.value == "m6"
    return (show_m6,)


@app.cell
def _(mo, probe_cache, show_m6):
    mo.stop(not show_m6)
    rq2_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure",
    )
    input_type = mo.ui.radio(
        options={"Ambiguous": "ambiguous", "GP": "gp", "Non-GP": "post"},
        value="Ambiguous",
        label="Input type",
        inline=True,
    )
    return input_type, probe_cache, rq2_condition


@app.cell
def _(
    activation_heatmap,
    colored_token_view,
    input_type,
    load_parquet_cache,
    mo,
    multi_feature_token_view,
    np,
    probe_cache,
    probe_figure5_plot,
    representative_activation_matrix,
    rq2_condition,
    show_m6,
):
    mo.stop(not show_m6)
    matrix = representative_activation_matrix(rq2_condition.value)
    activation_fig = activation_heatmap(matrix)
    probe_fig = probe_figure5_plot(probe_cache, rq2_condition.value)

    serial_svg = """<svg width="300" height="140"><text x="8" y="16" font-size="12">Serial parser</text>
    <circle cx="80" cy="70" r="10" fill="#c0392b"/><text x="50" y="100" font-size="10">one reading</text></svg>"""
    parallel_svg = """<svg width="300" height="140"><text x="8" y="16" font-size="12">Parallel (paper)</text>
    <circle cx="70" cy="60" r="8" fill="#c0392b"/><circle cx="70" cy="90" r="8" fill="#2980b9"/>
    <text x="100" y="78" font-size="10">both active</text></svg>"""

    _content = [
        mo.md("## 6. One or many readings? (RQ2)"),
        mo.md("Both pro-GP and anti-GP features activate — the model **hedges** between parses."),
        activation_fig,
        probe_fig,
        mo.hstack([mo.Html(serial_svg), mo.Html(parallel_svg)]),
    ]

    _tok_act = load_parquet_cache(f"token_activations_{rq2_condition.value.lower()}.parquet")
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
            _content.append(
                mo.Html(
                    multi_feature_token_view(
                        tokens[:n],
                        mat,
                        ["pro-GP features", "anti-GP features"],
                    )
                )
            )
    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 7 — Repair vs Reanalysis (RQ3)
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m7 = module_nav.value == "m7"
    return (show_m7,)


@app.cell
def _(CIRCUIT_IOU, gprc_table_df, mo, sample_gprc_items, show_m7):
    mo.stop(not show_m7)
    gprc_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="GPRC structure",
    )
    return (gprc_condition,)


@app.cell
def _(CIRCUIT_IOU, gprc_condition, gprc_table_df, mo, sample_gprc_items, show_m7):
    mo.stop(not show_m7)
    qa_table = mo.ui.table(gprc_table_df())
    samples = sample_gprc_items(gprc_condition.value, n=3)
    iou = CIRCUIT_IOU.get(gprc_condition.value, 0.0)
    overlap_svg = f"""
    <svg width="520" height="160" xmlns="http://www.w3.org/2000/svg">
      <circle cx="150" cy="80" r="60" fill="#fadbd8" opacity="0.8"/>
      <circle cx="280" cy="80" r="60" fill="#d6eaf8" opacity="0.8"/>
      <text x="85" y="85" font-size="12">GP circuit C₁</text>
      <text x="295" y="85" font-size="12">GPRC C₂</text>
      <text x="200" y="85" font-size="11" fill="#922b21">IoU ≈ {iou:.1%}</text>
    </svg>
    """
    mo.vstack([
        mo.md("## 7. Repair vs reanalysis? (RQ3)"),
        mo.md("**Table 3** — only Gemma-2-2b answers follow-ups above chance."),
        qa_table,
        mo.Html(overlap_svg),
        mo.md("### Sample GPRC questions"),
        mo.ui.table(
            samples[
                ["condition", "Sentence_GP", "Comp_Question_Yes", "Comp_Question_No"]
            ]
        ),
        mo.callout(
            "Gemma 2 does **neither** human-style repair nor syntactic reanalysis: "
            "GPRC circuits barely overlap garden-path circuits and rely on spurious yes/no features.",
            kind="warn",
        ),
    ])
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 8 — Your Garden-Path Sentence
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m8 = module_nav.value == "m8"
    return (show_m8,)


@app.cell
def _(gp_df, mo, show_m8):
    mo.stop(not show_m8)
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
    use_custom = mo.ui.switch(label="Use custom text (live mode)", value=False)
    score_button = mo.ui.run_button(label="Score sentence")
    intervene_button = mo.ui.run_button(label="Try causal flip")
    return (
        custom_condition,
        custom_sentence,
        example_lookup,
        example_pick,
        intervene_button,
        live_switch,
        score_button,
        use_custom,
    )


@app.cell
def _(
    MODEL_NAME,
    behavioral_scored,
    colored_token_view,
    custom_condition,
    custom_sentence,
    example_lookup,
    example_pick,
    faithfulness_tradeoff,
    get_hf_model,
    gp_df,
    intervene_button,
    live_switch,
    live_mode_available,
    mo,
    run_intervention_suite,
    saes_available,
    score_button,
    score_sentence,
    show_m8,
    timed_call,
    top_next_cache,
    top_next_tokens,
    tug_of_war_html,
    use_custom,
):
    mo.stop(not show_m8)

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

    if intervene_button.value and live_switch.value and live_mode_available() and saes_available():
        try:
            mini_df = gp_df.head(1).copy()
            mini_df.loc[0, "sentence_ambiguous"] = user_sentence
            mini_df.loc[0, "condition"] = condition
            intervention_result = timed_call(
                "intervention",
                run_intervention_suite,
                mini_df,
                condition,
                subject_amp=2.0,
                object_amp=0.0 if condition == "NPZ" else 2.0,
                clause_amp=2.0 if condition == "NPZ" else 0.0,
            )
        except Exception as exc:  # noqa: BLE001
            mo.output.append(mo.callout(f"Intervention failed: {exc}", kind="warn"))

    score_md = (
        mo.md(
            f"**{user_sentence}** ({condition})\n\n"
            f"p(GP)={live_scores['p_gp']:.4f}, p(non-GP)={live_scores['p_non_gp']:.4f}, "
            f"Δ={live_scores['diff']:+.4f}"
        )
        if live_scores
        else mo.md("_Click **Score sentence** or pick a curated example._")
    )

    _content = [
        mo.md("## 8. Build your own garden-path sentence"),
        mo.md(
            "**Extension:** apply the paper's metric and causal tools to novel input. "
            "Curated examples work instantly; custom text needs live mode."
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
        _content.append(mo.Html(attr_html))
    if intervention_result:
        _content.append(mo.Html(
            tug_of_war_html(intervention_result["mean_p_gp"], intervention_result["mean_p_non_gp"])
        ))
        _content.append(mo.md("_After syntactic clamp: reading preference should flip._"))
    _content.extend([
        mo.md("### Faithfulness budget (Appendix C)"),
        faithfulness_tradeoff(),
        mo.md(
            """
### Closing takeaways

1. **RQ1:** Syntactic SAE features causally drive garden-path preferences — alongside spurious detectors.
2. **RQ2:** Pro- and anti-GP features co-activate → multiple readings represented in parallel.
3. **RQ3:** Follow-up QA circuits barely reuse parse features → neither repair nor reanalysis.

*You just reverse-engineered incremental parsing in a 70M-parameter LM — one ambiguous noun at a time.*
"""
        ),
    ])
    mo.vstack(_content)
    return


if __name__ == "__main__":
    app.run()
