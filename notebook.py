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
            "1 — Feel the garden path": "m1",
            "2 — SAE primer": "m2",
            "3 — Behavioral lab (Fig 2)": "m3",
            "4 — Circuit explorer (Fig 3)": "m4",
            "5 — Intervention sandbox (Fig 4)": "m5",
            "6 — Multiple readings? (RQ2)": "m6",
            "7 — Repair vs reanalysis (RQ3)": "m7",
            "8 — Extensions": "m8",
        },
        value="m1",
        label="Section",
    )
    status = device_status_message()
    banner = mo.callout(
        status if live_mode_available() else "CPU / no CUDA — precomputed caches used by default.",
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
    import pandas as pd
    import plotly.express as px

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


@app.cell
def _(module_nav, mo):
    show_m1 = module_nav.value == "m1"
    return (show_m1,)


@app.cell
def _(gp_df, mo, reading_bubbles_html, show_m1, token_reveal_html):
    mo.stop(not show_m1)
    structure = mo.ui.dropdown(
        options={"NP/Z (subordinate clause)": "NPZ", "NP/S (sentential complement)": "NPS"},
        value="NPZ",
        label="Structure",
    )
    verb_type = mo.ui.radio(
        options={"Ambiguous verb": "ambiguous", "GP-forcing verb": "gp", "Non-GP verb": "post"},
        value="ambiguous",
        label="Verb type",
        inline=True,
    )
    reveal = mo.ui.slider(1, 8, value=6, label="Tokens revealed", show_value=True)
    return reveal, structure, verb_type


@app.cell
def _(gp_df, mo, reading_bubbles_html, reveal, structure, token_reveal_html, verb_type):
    col = f"sentence_{verb_type.value}"
    subset = gp_df[gp_df["condition"] == structure.value].iloc[0]
    sentence = subset[col]
    tokens = sentence.split()
    highlight = len(tokens) - 1
    if structure.value == "NPZ":
        gp_read = "The final noun is the object of the subordinate verb (clause can end here → comma)."
        non_gp_read = "The final noun is the matrix subject (continuation → was …)."
    else:
        gp_read = "The final noun is the object of the matrix verb (sentence can end → period)."
        non_gp_read = "The final noun starts a sentential complement (continuation → was …)."
    mo.vstack(
        [
            mo.md("## 1. Feel the garden path"),
            mo.md(f"**Example item {int(subset['item'])}** — read token by token:"),
            mo.Html(token_reveal_html(tokens, highlight_idx=highlight, reveal_count=reveal.value)),
            mo.Html(reading_bubbles_html(gp_read, non_gp_read)),
            mo.md(
                "_At the ambiguous noun, two syntactic continuations compete. "
                "Humans often prefer one reading and are surprised when the other appears._"
            ),
        ]
    )
    return


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
def _(mo, show_m2, sparsity):
    mo.stop(not show_m2)
    import numpy as np

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
    mo.vstack(
        [
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
    )
    return


@app.cell
def _(module_nav, mo):
    show_m3 = module_nav.value == "m3"
    return (show_m3,)


@app.cell
def _(behavioral_figure, behavioral_scored, behavioral_summary, mo, px, show_m3):
    mo.stop(not show_m3)
    content: list = [mo.md("## 3. Behavioral lab — does Pythia get garden-pathed?")]
    if behavioral_summary is None:
        content.append(
            mo.callout("Run `uv run python precompute.py` to generate behavioral caches.", kind="warn")
        )
    else:
        behavior_fig = behavioral_figure(behavioral_summary)
        drill = behavioral_scored[behavioral_scored["input_type"] == "ambiguous"].copy()
        table = mo.ui.table(
            drill[["item", "condition", "sentence", "p_gp", "p_non_gp", "diff"]],
            selection="single",
        )
        selected = table.value[0] if table.value else drill.iloc[0].to_dict()
        token_bar = px.bar(
            x=["p(GP)", "p(non-GP)"],
            y=[selected["p_gp"], selected["p_non_gp"]],
            labels={"x": "", "y": "probability"},
            title=f"Item {selected['item']} ({selected['condition']})",
        )
        content.extend(
            [
                mo.md("Reproduction of **Figure 2**: models prefer GP tokens on GP inputs and vice versa."),
                behavior_fig,
                mo.md("### Sentence drill-down"),
                table,
                token_bar,
            ]
        )
    mo.vstack(content)
    return


@app.cell
def _(module_nav, mo):
    show_m4 = module_nav.value == "m4"
    return (show_m4,)


@app.cell
def _(category_counts, circuit_svg, enrich_feature_table, layer_narrative, mo, show_m4):
    mo.stop(not show_m4)
    circuit_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NPZ",
        label="Circuit",
    )
    category_filter = mo.ui.dropdown(
        options={"All categories": "all", "Syntactic only": "syntactic"},
        value="all",
        label="Filter",
    )
    return category_filter, circuit_condition


@app.cell
def _(
    category_counts,
    category_filter,
    circuit_condition,
    circuit_svg,
    enrich_feature_table,
    layer_narrative,
    mo,
    show_m4,
):
    mo.stop(not show_m4)
    counts = category_counts(circuit_condition.value)
    if category_filter.value == "syntactic":
        counts = counts[counts["Category"].str.contains("detector|clause", case=False, na=False)]
    circuit_features = enrich_feature_table(circuit_condition.value)
    narrative = layer_narrative(circuit_condition.value)
    mo.vstack(
        [
            mo.md("## 4. Circuit explorer"),
            mo.md("Simplified **Figure 3**: early layers = word detectors; late layers = syntax."),
            mo.Html(circuit_svg(counts)),
            mo.accordion(
                {f"Layer {layer}": text for layer, text in narrative},
            ),
            mo.md("### Annotated features (Table 2 sample)"),
            mo.ui.table(
                circuit_features[circuit_features["is_syntactic"]][
                    ["Feature", "Category", "Annotation", "layer"]
                ].head(12)
            ),
        ]
    )
    return


@app.cell
def _(module_nav, mo):
    show_m5 = module_nav.value == "m5"
    return (show_m5,)


@app.cell
def _(interventions_df, live_switch, mo, show_m5):
    mo.stop(not show_m5)
    sandbox_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NPZ",
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
    gp_df,
    interventions_df,
    live_switch,
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
        row = interventions_df[
            (interventions_df["condition"] == sandbox_condition.value)
            & (interventions_df["intervention"] == "baseline")
        ].iloc[0]
        result.update(
            {
                "mean_p_gp": row["mean_p_gp"],
                "mean_p_non_gp": row["mean_p_non_gp"],
                "mean_diff": row["mean_diff"],
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

    mo.vstack(
        [
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
    )
    return


@app.cell
def _(module_nav, mo):
    show_m6 = module_nav.value == "m6"
    return (show_m6,)


@app.cell
def _(activation_heatmap, mo, representative_activation_matrix, show_m6):
    mo.stop(not show_m6)
    rq2_condition = mo.ui.dropdown(
        options={"NP/Z": "NPZ", "NP/S": "NPS"},
        value="NPZ",
        label="Structure",
    )
    return (rq2_condition,)


@app.cell
def _(activation_heatmap, mo, representative_activation_matrix, rq2_condition, show_m6):
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
    mo.vstack(
        [
            mo.md("## 6. One or many readings? (RQ2)"),
            mo.md(
                "Both pro-GP and anti-GP syntactic features activate on ambiguous input — "
                "the model represents **multiple readings simultaneously**."
            ),
            activation_fig,
            mo.hstack([mo.Html(serial_svg), mo.Html(parallel_svg)]),
        ]
    )
    return


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
        value="NPZ",
        label="Structure type",
    )
    score_button = mo.ui.run_button(label="Score sentence")
    return custom_condition, custom_sentence, score_button


@app.cell
def _(
    MODEL_NAME,
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
    scores = None
    if score_button.value and live_switch.value:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from gp_notebook.device import get_torch_device

            device = get_torch_device()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
            model.eval()
            scores = score_sentence(
                model, tokenizer, custom_sentence.value, custom_condition.value, device=device
            )
        except Exception as exc:  # noqa: BLE001
            mo.output.replace(mo.callout(f"Scoring failed: {exc}", kind="warn"))

    score_md = (
        mo.md(
            f"p(GP)={scores['p_gp']:.4f}, p(non-GP)={scores['p_non_gp']:.4f}, "
            f"Δ={scores['diff']:.4f}"
        )
        if scores
        else mo.md("_Enable live mode and click **Score sentence**._")
    )
    mo.vstack(
        [
            mo.md("## 8. Extensions"),
            custom_sentence,
            custom_condition,
            score_button,
            score_md,
            mo.md("### Faithfulness budget (Appendix C)"),
            faithfulness_tradeoff(),
            mo.md(
                "Including more SAE features improves faithfulness but requires manual annotation — "
                "a key limitation the paper discusses."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
