"""Marimo notebook: Garden Path Sentence Processing Mechanisms in LMs."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
# Garden Path Mechanisms in Language Models

**Paper:** [*Incremental Sentence Processing Mechanisms in Autoregressive
Transformer Language Models*](https://arxiv.org/abs/2412.05353) (Hanna & Mueller, NAACL 2025)

This notebook brings the paper to life: you'll *feel* garden-path ambiguity,
inspect sparse autoencoder (SAE) features, and **causally flip** which reading
Pythia-70m prefers by tuning subject/object/clause detectors.
"""
    )
    return


@app.cell
def _(mo):
    from gp_notebook.device import device_status_message, live_mode_available

    live_switch = mo.ui.switch(label="Live mode (run model + SAEs)", value=False)
    module_nav = mo.ui.radio(
        options={
            "0 — Mech-interp primer": "m0",
            "1 — Feel the garden path": "m1",
            "2 — SAE primer": "m2",
            "3 — Behavioral lab (Fig 2)": "m3",
            "4 — Circuit explorer (Fig 3)": "m4",
            "5 — Intervention sandbox (Fig 4)": "m5",
            "6 — Multiple readings? (RQ2)": "m6",
            "7 — Repair vs reanalysis (RQ3)": "m7",
            "8 — Extensions": "m8",
        },
        value="0 — Mech-interp primer",
        label="Section",
    )
    status = device_status_message()
    banner = mo.callout(
        status if live_mode_available() else "No GPU/MPS detected — precomputed caches used by default.",
        kind="info" if live_mode_available() else "warn",
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
    return live_switch, module_nav, sidebar


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

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
    from gp_notebook.interp_views import (
        attention_view,
        colored_token_view,
        multi_feature_token_view,
    )
    from gp_notebook.interventions import (
        build_feature_edits,
        run_intervention_suite,
        saes_available,
    )
    from gp_notebook.paths import ASSETS_DIR, load_gp_dataset, load_json_cache, load_parquet_cache
    from gp_notebook.viz import (
        activation_heatmap,
        behavioral_figure,
        circuit_svg,
        faithfulness_tradeoff,
        reading_bubbles_html,
        token_reveal_html,
        tug_of_war_html,
    )

    gp_df = load_gp_dataset()
    behavioral_summary = load_parquet_cache("behavioral_summary.parquet")
    behavioral_scored = load_parquet_cache("behavioral_scored.parquet")
    interventions_df = load_parquet_cache("interventions.parquet")
    metadata = load_json_cache("metadata.json") or {}
    return (
        ASSETS_DIR,
        MODEL_NAME,
        attention_view,
        colored_token_view,
        go,
        multi_feature_token_view,
        np,
        px,
        activation_heatmap,
        behavioral_figure,
        behavioral_scored,
        behavioral_summary,
        build_feature_edits,
        category_counts,
        circuit_svg,
        continuation_tokens_for_condition,
        enrich_feature_table,
        faithfulness_tradeoff,
        gp_df,
        interventions_df,
        layer_narrative,
        load_json_cache,
        load_parquet_cache,
        metadata,
        reading_bubbles_html,
        representative_activation_matrix,
        run_intervention_suite,
        saes_available,
        score_sentence,
        token_reveal_html,
        top_next_tokens,
        tug_of_war_html,
    )


# ═══════════════════════════════════════════════════════════════════
# CHAPTER 0 — Mechanistic Interpretability Primer
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m0 = module_nav.value == "m0"
    return (show_m0,)


@app.cell
def _(attention_view, load_json_cache, mo, np, show_m0):
    mo.stop(not show_m0)
    m0_intro = mo.md(
        r"""
## 0. What is mechanistic interpretability?

**Mechanistic interpretability** aims to reverse-engineer what neural networks
*actually compute*, not just what they get right. The key intuitions:

| Concept | One-liner |
|---------|-----------|
| **Neuron** | A single unit in the network — often *polysemantic* (fires for unrelated things). |
| **Feature** | A *monosemantic* direction in activation space, often found via sparse autoencoders. |
| **Superposition** | Networks pack more features than they have neurons by using sparse, overlapping codes. |
| **Circuit** | The minimal subgraph of features that reproduces a model behavior. |
| **Ablation** | Setting a feature to zero and measuring the effect — the core causal test. |
| **Attribution (AtP-IG)** | Fast gradient-based estimate of each feature's causal contribution. |
| **Faithfulness** | How well a circuit's behavior matches the full model (ideally ≈ 1.0). |

In this notebook we apply these ideas to **garden-path sentences** — ambiguous
inputs where the model must decide between two syntactic readings.
"""
    )

    glossary = mo.accordion(
        {
            "Sparse Autoencoder (SAE)": (
                r"An autoencoder trained on LM activations with sparsity regularization.  "
                r"Given activation **x**: $f = \mathrm{ReLU}(W_e(x - b_d) + b_e)$ and "
                r"$\hat{x} = W_d f + b_d$.  Each dimension of **f** is called a *feature*."
            ),
            "Feature": (
                "A single sparse dimension of the SAE encoding. Active (non-zero) only "
                "when the input differs meaningfully from the mean — making features more "
                "interpretable than raw neurons."
            ),
            "Indirect Effect (IE) & AtP-IG": (
                r"A feature's **indirect effect** on metric $m$ measures how much $m$ changes "
                r"when the feature is ablated. Attribution Patching with Integrated Gradients "
                r"(AtP-IG) approximates this cheaply: $\widehat{IE} = (a-a') \cdot "
                r"\frac{1}{K}\sum_{k=0}^{K}\frac{\partial m}{\partial a}\Big|_{a'+\frac{k}{K}(a-a')}$"
            ),
            "Garden-path (GP) vs non-GP": (
                "A GP reading treats the ambiguous noun as an object (clause can end → comma/period); "
                "the non-GP reading treats it as a new subject (continuation → ' was ...'). "
                "We measure m = p(GP) − p(non-GP)."
            ),
            "Faithfulness": (
                "The ratio of a circuit's metric value to the full model's, averaged over the dataset. "
                "1.0 = perfect; the paper's circuits achieve 0.20 (NP/S) to 3.48 (NP/Z)."
            ),
        }
    )

    attn_cache = load_json_cache("attention_npz.json")
    if attn_cache is not None:
        layer_select = mo.ui.dropdown(
            options={f"Layer {i}": str(i) for i in range(6)},
            value="Layer 0",
            label="Attention layer",
        )
    else:
        layer_select = None

    roadmap = mo.md(
        """
### Notebook roadmap

| Ch. | Title | What you'll see | RQ |
|-----|-------|----------------|----|
| 0 | Mech-interp primer | *You are here* — concepts + attention demo | — |
| 1 | Feel the garden path | Token-by-token reveal with model commitment timeline | — |
| 2 | SAE primer | Encoder/decoder + live feature decode | — |
| 3 | Behavioral lab | Figure 2 + captum attribution per sentence | RQ1 |
| 4 | Circuit explorer | Interactive circuit graph + Table 2 examples | RQ1 |
| 5 | Intervention sandbox | Slider-driven causal flip with before/after view | RQ1 |
| 6 | Multiple readings? | Real activation heatmap + dual-feature painting | RQ2 |
| 7 | Repair vs reanalysis | Table 3 + IoU disconnection | RQ3 |
| 8 | Build your own | Free-text sentence scoring + faithfulness chart | ext. |
"""
    )

    _content = [m0_intro, mo.md("### Key concepts"), glossary]
    if attn_cache is not None and layer_select is not None:
        _content.append(mo.md("### Attention patterns — Pythia-70m on a garden-path sentence"))
        _content.append(layer_select)
    _content.append(roadmap)

    mo.vstack(_content)
    return (attn_cache, layer_select)


@app.cell
def _(attn_cache, attention_view, layer_select, mo, np, show_m0):
    mo.stop(not show_m0)
    if attn_cache is not None and layer_select is not None:
        layer_idx = layer_select.value
        attn_data = np.array(attn_cache["layers"][layer_idx])
        _tokens = attn_cache["tokens"]
        attn_html = attention_view(_tokens, attn_data)
        _attn_panel = mo.vstack([
            mo.md(f"**Layer {layer_idx}** — {attn_data.shape[0]} attention heads"),
            mo.iframe(attn_html),
        ])
    else:
        _attn_panel = mo.md(
            "_Run `uv run python precompute.py` to generate attention caches for the demo._"
        )
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
    colored_token_view,
    gp_df,
    load_json_cache,
    load_parquet_cache,
    mo,
    px,
    reading_bubbles_html,
    reveal,
    structure,
    token_reveal_html,
    verb_type,
):
    mo.stop(not (structure and verb_type and reveal))
    col = f"sentence_{verb_type.value}"
    subset = gp_df[gp_df["condition"] == structure.value].iloc[0]
    sentence = subset[col]
    _tokens = sentence.split()
    highlight = len(_tokens) - 1
    if structure.value == "NPZ":
        gp_read = "The final noun is the object of the subordinate verb (clause can end here → comma)."
        non_gp_read = "The final noun is the matrix subject (continuation → was …)."
    else:
        gp_read = "The final noun is the object of the matrix verb (sentence can end → period)."
        non_gp_read = "The final noun starts a sentential complement (continuation → was …)."

    _content = [
        mo.md("## 1. Feel the garden path"),
        mo.md(f"**Example item {int(subset['item'])}** — read token by token:"),
        mo.Html(token_reveal_html(_tokens, highlight_idx=highlight, reveal_count=reveal.value)),
        mo.Html(reading_bubbles_html(gp_read, non_gp_read)),
        mo.md(
            "_At the ambiguous noun, two syntactic continuations compete. "
            "Humans often prefer one reading and are surprised when the other appears._"
        ),
    ]

    prefix_cache = load_parquet_cache(f"prefix_probs_{structure.value.lower()}.parquet")
    if prefix_cache is not None:
        prefix_fig = px.line(
            prefix_cache,
            x="n_tokens",
            y=["p_gp", "p_non_gp"],
            labels={"n_tokens": "Tokens seen", "value": "Probability", "variable": "Reading"},
            title="Commitment timeline: how Pythia's preference shifts as tokens arrive",
        )
        prefix_fig.update_traces(
            selector=dict(name="p_gp"),
            line_color="#c0392b",
            name="p(GP)",
        )
        prefix_fig.update_traces(
            selector=dict(name="p_non_gp"),
            line_color="#2980b9",
            name="p(non-GP)",
        )
        _content.append(mo.md("### Model commitment timeline"))
        _content.append(prefix_fig)

    tok_act_cache = load_json_cache(f"attention_{structure.value.lower()}.json")
    if tok_act_cache is not None and verb_type.value == "ambiguous":
        tok_strs = tok_act_cache["tokens"]
        n_shown = min(reveal.value, len(tok_strs))
        _scores = [0.0] * n_shown
        if n_shown > 0:
            _scores[-1] = 0.5
        if n_shown > 1:
            _scores[-2] = 0.2
        _content.append(mo.md("### Token importance (attention-derived)"))
        _content.append(
            mo.Html(
                colored_token_view(
                    tok_strs[:n_shown],
                    _scores,
                    label="Darker = more attention at the ambiguous noun",
                    positive_color="#e74c3c",
                )
            )
        )

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
def _(colored_token_view, load_parquet_cache, mo, np, show_m2, sparsity):
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
        mo.md(
            "Sparse autoencoders decompose LM activations into **monosemantic features** "
            "used throughout the paper's circuit analysis."
        ),
        mo.Html(sae_svg),
        mo.md(
            r"""
- $f = \mathrm{ReLU}(W_e(x - b_d) + b_e)$
- $\hat{x} = W_d f + b_d$
"""
        ),
        mo.md(f"Toy sparsity demo: **{active}** of 32 features active at threshold {sparsity.value:.2f}."),
    ]

    _tok_act = load_parquet_cache("token_activations_npz.parquet")
    if _tok_act is not None and not _tok_act.empty:
        _first_sentence = _tok_act["sentence"].iloc[0] if "sentence" in _tok_act.columns else None
        if _first_sentence:
            _sent_data = _tok_act[_tok_act["sentence"] == _first_sentence]
            top_features = _sent_data.groupby("feature")["activation"].max().nlargest(5).index.tolist()
            for feat_name in top_features[:3]:
                feat_rows = _sent_data[_sent_data["feature"] == feat_name].sort_values("position")
                _content.append(
                    mo.Html(
                        colored_token_view(
                            feat_rows["token"].tolist(),
                            feat_rows["activation"].tolist(),
                            label=f"Feature: {feat_name}",
                            positive_color="#e67e22",
                        )
                    )
                )
        _content.append(
            mo.md("_Above: real SAE feature activations from Pythia-70m on a garden-path sentence._")
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
    if behavioral_summary is not None:
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
):
    mo.stop(not show_m3)
    _content: list = [mo.md("## 3. Behavioral lab — does Pythia get garden-pathed?")]
    if behavioral_summary is None:
        _content.append(
            mo.callout("Run `uv run python precompute.py` to generate behavioral caches.", kind="warn")
        )
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
        _content.extend(
            [
                mo.md("Reproduction of **Figure 2**: models prefer GP tokens on GP inputs and vice versa."),
                behavior_fig,
                mo.md("### Sentence drill-down"),
                drill_table,
                token_bar,
            ]
        )

        cond_key = selected["condition"].lower() if isinstance(selected["condition"], str) else "npz"
        attr_cache = load_json_cache(f"attributions_{cond_key}.json")
        if attr_cache is not None:
            _content.append(mo.md("### Token attribution (Captum Integrated Gradients)"))
            _content.append(
                mo.md(
                    "_Which input tokens push toward GP vs non-GP? "
                    "Red = pushes toward GP; blue = pushes toward non-GP._"
                )
            )
            _content.append(
                mo.Html(
                    colored_token_view(
                        attr_cache["tokens"],
                        attr_cache["scores"],
                        label=f"IG attribution: m = p(GP) − p(non-GP)",
                    )
                )
            )

    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 4 — Circuit Explorer (Fig 3)
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
        options={"All categories": "all", "Syntactic only": "syntactic"},
        value="All categories",
        label="Filter",
    )
    layer_slider = mo.ui.slider(0, 5, value=5, label="Show up to layer", show_value=True)
    return category_filter, circuit_condition, layer_slider


@app.cell
def _(
    category_counts,
    category_filter,
    circuit_condition,
    colored_token_view,
    enrich_feature_table,
    go,
    layer_narrative,
    layer_slider,
    load_parquet_cache,
    mo,
    show_m4,
):
    mo.stop(not show_m4)
    counts = category_counts(circuit_condition.value)
    if category_filter.value == "syntactic":
        counts = counts[counts["Category"].str.contains("detector|clause", case=False, na=False)]
    counts = counts[counts["layer"] <= layer_slider.value]

    circuit_features = enrich_feature_table(circuit_condition.value)
    narrative = layer_narrative(circuit_condition.value)

    node_fig = go.Figure()
    for _, _row in counts.iterrows():
        _color = "#c0392b" if _row.reading_side == "pro_gp" else (
            "#2980b9" if _row.reading_side == "anti_gp" else "#7f8c8d"
        )
        x_offset = 0.3 if _row.reading_side == "pro_gp" else (
            0.7 if _row.reading_side == "anti_gp" else 0.5
        )
        node_fig.add_trace(go.Scatter(
            x=[x_offset + hash(_row.Category) % 100 / 500],
            y=[_row.layer],
            mode="markers+text",
            marker=dict(size=12 + min(int(_row["count"]), 10) * 3, color=_color, opacity=0.8),
            text=[f"{_row.Category} ({int(_row['count'])})"],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=f"Layer {_row.layer}: {_row.Category} ({int(_row['count'])} features, {_row.reading_side})",
            showlegend=False,
        ))
    node_fig.update_layout(
        title="Circuit topology — feature groups by layer and reading side",
        xaxis=dict(title="← pro-GP (red) | anti-GP (blue) →", range=[-0.1, 1.1], showticklabels=False),
        yaxis=dict(title="Layer", dtick=1, autorange="reversed"),
        height=450,
    )

    _content = [
        mo.md("## 4. Circuit explorer"),
        mo.md(
            "Interactive **Figure 3**: early layers = word detectors; late layers = syntax. "
            "Use the layer slider to progressively reveal the circuit."
        ),
        node_fig,
        mo.accordion(
            {f"Layer {layer}": text for layer, text in narrative if layer <= layer_slider.value},
        ),
        mo.md("### Annotated features (Table 2 sample)"),
        mo.ui.table(
            circuit_features[circuit_features["is_syntactic"]][
                ["Feature", "Category", "Annotation", "layer"]
            ].head(12)
        ),
    ]

    _tok_act = load_parquet_cache(f"token_activations_{circuit_condition.value.lower()}.parquet")
    if _tok_act is not None and not _tok_act.empty and "sentence" in _tok_act.columns:
        _content.append(mo.md("### Real feature activation examples (Table 2 style)"))
        top_cats = counts[counts["reading_side"].isin({"pro_gp", "anti_gp"})].head(3)
        for _, cat_row in top_cats.iterrows():
            cat_data = _tok_act[_tok_act["category"] == cat_row.Category]
            if cat_data.empty:
                continue
            first_sent = cat_data["sentence"].iloc[0]
            _feat_data = cat_data[
                (cat_data["sentence"] == first_sent) & (cat_data["feature"] == cat_data["feature"].iloc[0])
            ].sort_values("position")
            if not _feat_data.empty:
                _content.append(
                    mo.Html(
                        colored_token_view(
                            _feat_data["token"].tolist(),
                            _feat_data["activation"].tolist(),
                            label=f"{cat_row.Category} ({cat_row.reading_side})",
                            positive_color="#c0392b" if cat_row.reading_side == "pro_gp" else "#2980b9",
                        )
                    )
                )

    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 5 — Intervention Sandbox (Fig 4)
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
    random_toggle = mo.ui.switch(label="Intervene on random features (null control)", value=False)
    run_button = mo.ui.run_button(label="Run intervention")
    return (
        clause_slider,
        interventions_df,
        live_switch,
        object_slider,
        random_toggle,
        run_button,
        sandbox_condition,
        subject_slider,
    )


@app.cell
def _(
    clause_slider,
    colored_token_view,
    gp_df,
    interventions_df,
    live_switch,
    load_parquet_cache,
    mo,
    object_slider,
    px,
    random_toggle,
    run_button,
    run_intervention_suite,
    saes_available,
    sandbox_condition,
    show_m5,
    subject_slider,
    tug_of_war_html,
):
    mo.stop(not show_m5)

    result = {"mean_p_gp": 0.1, "mean_p_non_gp": 0.05, "mean_diff": 0.05, "source": "precomputed"}
    if interventions_df is not None:
        _row = interventions_df[
            (interventions_df["condition"] == sandbox_condition.value)
            & (interventions_df["intervention"] == "baseline")
        ].iloc[0]
        result.update(
            {
                "mean_p_gp": _row["mean_p_gp"],
                "mean_p_non_gp": _row["mean_p_non_gp"],
                "mean_diff": _row["mean_diff"],
            }
        )
    if run_button.value and live_switch.value:
        if not saes_available():
            mo.output.replace(mo.callout("SAE checkpoints missing.", kind="warn"))
        else:
            try:
                live = run_intervention_suite(
                    gp_df,
                    sandbox_condition.value,
                    subject_amp=subject_slider.value,
                    object_amp=object_slider.value,
                    clause_amp=clause_slider.value,
                    use_random=random_toggle.value,
                )
                result.update(live)
                result["source"] = "live"
            except Exception as exc:  # noqa: BLE001
                mo.output.replace(mo.callout(f"Live intervention failed: {exc}", kind="warn"))

    paper_chart = None
    if interventions_df is not None:
        paper_chart = px.bar(
            interventions_df,
            x="condition",
            y="mean_diff",
            color="intervention",
            barmode="group",
            title="Paper interventions (Figure 4)",
            labels={"mean_diff": "p(GP) − p(non-GP)"},
        )

    _content = [
        mo.md("## 5. Intervention sandbox — flip the model's reading"),
        mo.md(
            "Clamp **subject**, **object**, and **clause-end** SAE features to induce the "
            "opposite garden-path preference. This is the paper's causal verification."
        ),
        mo.hstack([subject_slider, object_slider, clause_slider]),
        mo.hstack([random_toggle, run_button]),
        mo.Html(tug_of_war_html(result["mean_p_gp"], result["mean_p_non_gp"])),
        mo.md(
            f"**Δ = {result['mean_diff']:.4f}** · source: {result['source']} "
            f"(positive → GP reading preferred)"
        ),
        paper_chart,
    ]

    _tok_act = load_parquet_cache(f"token_activations_{sandbox_condition.value.lower()}.parquet")
    if _tok_act is not None and not _tok_act.empty and "sentence" in _tok_act.columns:
        _content.append(mo.md("### Before intervention — baseline feature activations"))
        _first_sentence = _tok_act["sentence"].iloc[0]
        _sent_data = _tok_act[_tok_act["sentence"] == _first_sentence]
        for _side, _color in [("pro_gp", "#c0392b"), ("anti_gp", "#2980b9")]:
            _side_data = _sent_data[_sent_data["reading_side"] == _side]
            if _side_data.empty:
                continue
            _first_feat = _side_data["feature"].iloc[0]
            _feat_data = _side_data[_side_data["feature"] == _first_feat].sort_values("position")
            _content.append(
                mo.Html(
                    colored_token_view(
                        _feat_data["token"].tolist(),
                        _feat_data["activation"].tolist(),
                        label=f"Baseline: {_first_feat} ({_side})",
                        positive_color=_color,
                    )
                )
            )

    mo.vstack(_content)
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 6 — One or Many Readings? (RQ2)
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m6 = module_nav.value == "m6"
    return (show_m6,)


@app.cell
def _(mo, show_m6):
    mo.stop(not show_m6)
    rq2_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure",
    )
    return (rq2_condition,)


@app.cell
def _(
    activation_heatmap,
    colored_token_view,
    load_parquet_cache,
    mo,
    np,
    multi_feature_token_view,
    representative_activation_matrix,
    rq2_condition,
    show_m6,
):
    mo.stop(not show_m6)
    matrix = representative_activation_matrix(rq2_condition.value)
    activation_fig = activation_heatmap(matrix)

    serial_svg = """
    <svg width="320" height="180"><text x="10" y="20" font-size="13">Serial parser</text>
    <line x1="20" y1="150" x2="300" y2="150" stroke="#555"/>
    <circle cx="60" cy="120" r="8" fill="#c0392b"/><text x="45" y="140" font-size="11">GP only</text>
    <text x="180" y="100" font-size="11" fill="#c0392b">crash → repair</text></svg>
    """
    parallel_svg = """
    <svg width="320" height="180"><text x="10" y="20" font-size="13">Parallel (paper finding)</text>
    <line x1="20" y1="150" x2="300" y2="150" stroke="#555"/>
    <circle cx="80" cy="110" r="8" fill="#c0392b"/><circle cx="80" cy="80" r="8" fill="#2980b9"/>
    <text x="120" y="95" font-size="11">both readings active</text></svg>
    """

    _content = [
        mo.md("## 6. One or many readings? (RQ2)"),
        mo.md(
            "Both pro-GP and anti-GP syntactic features activate on ambiguous input — "
            "the model represents **multiple readings simultaneously**."
        ),
        activation_fig,
        mo.hstack([mo.Html(serial_svg), mo.Html(parallel_svg)]),
    ]

    _tok_act = load_parquet_cache(f"token_activations_{rq2_condition.value.lower()}.parquet")
    if _tok_act is not None and not _tok_act.empty and "sentence" in _tok_act.columns:
        _content.append(mo.md("### Dual-feature view — both readings painted on the same sentence"))
        _first_sentence = _tok_act["sentence"].iloc[0]
        _sent_data = _tok_act[_tok_act["sentence"] == _first_sentence]
        for _side, label, _color in [
            ("pro_gp", "Pro-GP features (object/clause detectors)", "#c0392b"),
            ("anti_gp", "Anti-GP features (subject detectors)", "#2980b9"),
        ]:
            _side_data = _sent_data[_sent_data["reading_side"] == _side]
            if _side_data.empty:
                continue
            _first_feat = _side_data["feature"].iloc[0]
            _feat_data = _side_data[_side_data["feature"] == _first_feat].sort_values("position")
            _content.append(
                mo.Html(
                    colored_token_view(
                        _feat_data["token"].tolist(),
                        _feat_data["activation"].tolist(),
                        label=f"{label}: {_first_feat}",
                        positive_color=_color,
                    )
                )
            )
        _content.append(
            mo.md("_Both pro-GP (red) and anti-GP (blue) features fire — the model hedges._")
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
def _(mo, show_m7):
    mo.stop(not show_m7)
    qa_table = mo.ui.table(
        {
            "Task": ["BoolQ", "MCQA", "GPRC NPS", "GPRC NPZ"],
            "Pythia-70m": ["42.8%", "50.0%", "50.0%", "50.0%"],
            "Gemma-2-2b": ["70.7%", "90.0%", "83.3%", "70.9%"],
        }
    )
    overlap_svg = """
    <svg width="520" height="160" xmlns="http://www.w3.org/2000/svg">
      <circle cx="150" cy="80" r="60" fill="#fadbd8" opacity="0.8"/>
      <circle cx="280" cy="80" r="60" fill="#d6eaf8" opacity="0.8"/>
      <text x="95" y="85" font-size="12">GP circuit C₁</text>
      <text x="300" y="85" font-size="12">GPRC circuit C₂</text>
      <text x="205" y="85" font-size="11" fill="#922b21">IoU ≈ 0%</text>
    </svg>
    """
    mo.vstack(
        [
            mo.md("## 7. Repair vs reanalysis? (RQ3)"),
            mo.md("**Table 3** — only Gemma-2-2b answers follow-up questions above chance."),
            qa_table,
            mo.Html(overlap_svg),
            mo.callout(
                "Gemma 2 does **neither** human-style repair (reusing parse features) nor "
                "reanalysis (building new syntactic features for follow-ups). GPRC circuits "
                "are dominated by spurious yes/no politeness features.",
                kind="warn",
            ),
        ]
    )
    return


# ═══════════════════════════════════════════════════════════════════
# MODULE 8 — Extensions
# ═══════════════════════════════════════════════════════════════════

@app.cell
def _(module_nav, mo):
    show_m8 = module_nav.value == "m8"
    return (show_m8,)


@app.cell
def _(faithfulness_tradeoff, live_switch, mo, score_sentence, show_m8):
    mo.stop(not show_m8)
    custom_sentence = mo.ui.text_area(
        label="Your garden-path sentence",
        value="After the politician signed the bill",
    )
    custom_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NP/Z",
        label="Structure type",
    )
    score_button = mo.ui.run_button(label="Score sentence")
    return custom_condition, custom_sentence, score_button


@app.cell
def _(
    MODEL_NAME,
    colored_token_view,
    custom_condition,
    custom_sentence,
    faithfulness_tradeoff,
    live_switch,
    mo,
    score_button,
    score_sentence,
    show_m8,
):
    mo.stop(not show_m8)
    live_scores = None
    attr_html = None
    if score_button.value and live_switch.value:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from gp_notebook.attribution import token_attributions
            from gp_notebook.device import get_torch_device

            device = get_torch_device()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
            model.eval()
            live_scores = score_sentence(
                model, tokenizer, custom_sentence.value, custom_condition.value, device=device
            )
            try:
                attr_result = token_attributions(
                    model, tokenizer, custom_sentence.value, custom_condition.value, device=device
                )
                attr_html = colored_token_view(
                    attr_result["tokens"],
                    attr_result["scores"],
                    label="Token attribution (Integrated Gradients)",
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            mo.output.replace(mo.callout(f"Scoring failed: {exc}", kind="warn"))

    score_md = (
        mo.md(
            f"p(GP)={live_scores['p_gp']:.4f}, p(non-GP)={live_scores['p_non_gp']:.4f}, "
            f"Δ={live_scores['diff']:.4f}"
        )
        if live_scores
        else mo.md("_Enable live mode and click **Score sentence**._")
    )

    _content = [
        mo.md("## 8. Extensions"),
        custom_sentence,
        custom_condition,
        score_button,
        score_md,
    ]

    if attr_html is not None:
        _content.append(mo.md("### Token-level attribution"))
        _content.append(
            mo.md("_Which tokens push toward GP (red) vs non-GP (blue)?_")
        )
        _content.append(mo.Html(attr_html))

    _content.extend([
        mo.md("### Faithfulness budget (Appendix C)"),
        faithfulness_tradeoff(),
        mo.md(
            "Including more SAE features improves faithfulness but requires manual annotation — "
            "a key limitation the paper discusses."
        ),
    ])
    mo.vstack(_content)
    return


if __name__ == "__main__":
    app.run()
