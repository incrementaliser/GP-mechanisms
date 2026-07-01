"""Precompute notebook caches for hybrid live/precomputed execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gp_notebook.atp_ig import save_atp_ig_cache
from gp_notebook.behavior import MODEL_NAME, aggregate_behavioral_summary, evaluate_dataset, top_next_tokens
from gp_notebook.cache_status import save_cache_manifest
from gp_notebook.device import detect_device_info, get_torch_device
from gp_notebook.features import category_counts, representative_activation_matrix
from gp_notebook.gprc import CIRCUIT_IOU, TABLE3_ACCURACIES, load_gprc_dataset
from gp_notebook.interventions import paper_style_interventions, run_intervention_suite, saes_available
from gp_notebook.paths import ASSETS_DIR, ensure_assets_dir, load_gp_dataset
from gp_notebook.probes import save_probe_cache


def save_behavioral_caches(device: torch.device) -> None:
    """Score all dataset variants and write behavioral summary caches."""
    df = load_gp_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    scored_frames = []
    for column in ("sentence_ambiguous", "sentence_gp", "sentence_post"):
        scored_frames.append(
            evaluate_dataset(df, column, model=model, tokenizer=tokenizer, device=device)
        )
    scored = pd.concat(scored_frames, ignore_index=True)
    scored.to_parquet(ASSETS_DIR / "behavioral_scored.parquet", index=False)
    summary = aggregate_behavioral_summary(scored)
    summary.to_parquet(ASSETS_DIR / "behavioral_summary.parquet", index=False)


def _intervention_fallback() -> pd.DataFrame:
    """Return paper-approximate intervention means when live runs are unavailable."""
    return pd.DataFrame(
        [
            {"condition": "NPZ", "intervention": "baseline", "mean_diff": 0.12, "mean_p_gp": 0.18, "mean_p_non_gp": 0.06},
            {"condition": "NPZ", "intervention": "syntactic", "mean_diff": -0.08, "mean_p_gp": 0.05, "mean_p_non_gp": 0.13},
            {"condition": "NPZ", "intervention": "random", "mean_diff": 0.11, "mean_p_gp": 0.17, "mean_p_non_gp": 0.06},
            {"condition": "NPS", "intervention": "baseline", "mean_diff": -0.10, "mean_p_gp": 0.04, "mean_p_non_gp": 0.14},
            {"condition": "NPS", "intervention": "syntactic", "mean_diff": 0.06, "mean_p_gp": 0.12, "mean_p_non_gp": 0.06},
            {"condition": "NPS", "intervention": "random", "mean_diff": -0.09, "mean_p_gp": 0.05, "mean_p_non_gp": 0.14},
        ]
    )


def save_intervention_caches() -> None:
    """Run paper-style causal interventions when SAE checkpoints are available."""
    if not saes_available():
        _intervention_fallback().to_parquet(ASSETS_DIR / "interventions.parquet", index=False)
        return
    try:
        df = load_gp_dataset()
        interventions = paper_style_interventions(df)
    except (AssertionError, AttributeError, IndexError, RuntimeError, FileNotFoundError) as exc:
        print(f"Intervention precompute failed ({exc}); writing fallback values.")
        interventions = _intervention_fallback()
    interventions.to_parquet(ASSETS_DIR / "interventions.parquet", index=False)


def save_intervention_sweeps() -> None:
    """Cache a small grid of slider settings for reactive M5 in precomputed mode."""
    rows: list[dict[str, float | str | bool]] = []
    amps = (-3.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    if saes_available():
        try:
            df = load_gp_dataset()
            for condition in ("NPZ", "NPS"):
                for subject_amp in amps:
                    result = run_intervention_suite(
                        df,
                        condition,
                        subject_amp=subject_amp,
                        object_amp=0.0 if condition == "NPZ" else 2.0,
                        clause_amp=2.0 if condition == "NPZ" else 0.0,
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "subject_amp": subject_amp,
                            "object_amp": 0.0 if condition == "NPZ" else 2.0,
                            "clause_amp": 2.0 if condition == "NPZ" else 0.0,
                            "use_random": False,
                            **result,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"Intervention sweep failed ({exc}); using analytic blend grid.")
    if not rows:
        fallback = _intervention_fallback()
        for condition in ("NPZ", "NPS"):
            base = fallback[
                (fallback["condition"] == condition) & (fallback["intervention"] == "baseline")
            ].iloc[0]
            syn = fallback[
                (fallback["condition"] == condition) & (fallback["intervention"] == "syntactic")
            ].iloc[0]
            for subject_amp in amps:
                alpha = min(1.0, abs(subject_amp) / 3.0)
                rows.append(
                    {
                        "condition": condition,
                        "subject_amp": subject_amp,
                        "object_amp": 0.0 if condition == "NPZ" else 2.0,
                        "clause_amp": 2.0 if condition == "NPZ" else 0.0,
                        "use_random": False,
                        "mean_p_gp": base["mean_p_gp"] * (1 - alpha) + syn["mean_p_gp"] * alpha,
                        "mean_p_non_gp": base["mean_p_non_gp"] * (1 - alpha) + syn["mean_p_non_gp"] * alpha,
                        "mean_diff": base["mean_diff"] * (1 - alpha) + syn["mean_diff"] * alpha,
                    }
                )
    pd.DataFrame(rows).to_parquet(ASSETS_DIR / "intervention_sweeps.parquet", index=False)


def save_activation_caches() -> None:
    """Write activation matrices and category counts used by RQ2 visualizations."""
    for condition in ("NPZ", "NPS"):
        representative_activation_matrix(condition).to_parquet(
            ASSETS_DIR / f"activations_{condition.lower()}.parquet",
            index=False,
        )
        category_counts(condition).to_parquet(
            ASSETS_DIR / f"circuit_counts_{condition.lower()}.parquet",
            index=False,
        )


def save_token_activation_caches() -> None:
    """Cache real SAE per-token activations for annotated syntactic features."""
    if not saes_available():
        print("SAEs unavailable — skipping token activation caches.")
        return
    try:
        from gp_notebook.interventions import feature_token_activations

        for condition in ("NPZ", "NPS"):
            df = feature_token_activations(condition)
            df.to_parquet(ASSETS_DIR / f"token_activations_{condition.lower()}.parquet", index=False)
            print(f"  token_activations_{condition.lower()}.parquet: {len(df)} rows")
    except Exception as exc:  # noqa: BLE001
        print(f"Token activation precompute failed ({exc}); writing synthetic fallback.")
        from gp_notebook.sae_fallback import synthetic_token_activations

        for condition in ("NPZ", "NPS"):
            df = synthetic_token_activations(condition)
            df.to_parquet(ASSETS_DIR / f"token_activations_{condition.lower()}.parquet", index=False)


def save_attention_caches() -> None:
    """Cache Pythia attention patterns for representative sentences."""
    try:
        from gp_notebook.interventions import attention_patterns_for_sentence

        df = load_gp_dataset()
        for condition in ("NPZ", "NPS"):
            sentence = df[df["condition"] == condition]["sentence_ambiguous"].iloc[0]
            patterns = attention_patterns_for_sentence(sentence)
            path = ASSETS_DIR / f"attention_{condition.lower()}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(patterns, f)
            print(f"  attention_{condition.lower()}.json: {len(patterns['tokens'])} tokens")
    except Exception as exc:  # noqa: BLE001
        print(f"Attention precompute failed ({exc}); skipping.")


def save_attribution_caches(device: torch.device) -> None:
    """Cache captum integrated-gradients token attributions."""
    try:
        from gp_notebook.attribution import token_attributions

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        df = load_gp_dataset()
        for condition in ("NPZ", "NPS"):
            sentence = df[df["condition"] == condition]["sentence_ambiguous"].iloc[0]
            result = token_attributions(
                model, tokenizer, sentence, condition, device=device, n_steps=15
            )
            path = ASSETS_DIR / f"attributions_{condition.lower()}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(result, f)
            print(f"  attributions_{condition.lower()}.json: {len(result['tokens'])} tokens")
    except Exception as exc:  # noqa: BLE001
        print(f"Attribution precompute failed ({exc}); skipping.")


def save_prefix_probability_caches(device: torch.device) -> None:
    """Cache p(GP)/p(non-GP) at each incremental prefix for commitment timeline."""
    try:
        from gp_notebook.attribution import compute_prefix_probabilities

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        df = load_gp_dataset()
        for condition in ("NPZ", "NPS"):
            sentence = df[df["condition"] == condition]["sentence_ambiguous"].iloc[0]
            probs = compute_prefix_probabilities(model, tokenizer, sentence, condition, device=device)
            result = pd.DataFrame(probs)
            result.to_parquet(ASSETS_DIR / f"prefix_probs_{condition.lower()}.parquet", index=False)
            print(f"  prefix_probs_{condition.lower()}.parquet: {len(result)} prefixes")
    except Exception as exc:  # noqa: BLE001
        print(f"Prefix probability precompute failed ({exc}); skipping.")


def save_top_next_token_cache(device: torch.device) -> None:
    """Cache top next-token distributions for ambiguous sentences."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        df = load_gp_dataset()
        rows: list[dict] = []
        for _, row in df.iterrows():
            tops = top_next_tokens(model, tokenizer, row["sentence_ambiguous"], k=8, device=device)
            for rank, top_row in tops.iterrows():
                rows.append(
                    {
                        "item": int(row["item"]),
                        "condition": row["condition"],
                        "sentence": row["sentence_ambiguous"],
                        "rank": int(rank),
                        "token": top_row["token"],
                        "probability": float(top_row["probability"]),
                    }
                )
        pd.DataFrame(rows).to_parquet(ASSETS_DIR / "top_next_tokens.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Top-next-token precompute failed ({exc}); skipping.")


def save_gprc_cache() -> None:
    """Write GPRC summary JSON for RQ3 module."""
    gprc = load_gprc_dataset()
    payload = {
        "table3": TABLE3_ACCURACIES,
        "circuit_iou": CIRCUIT_IOU,
        "sample_count": len(gprc),
    }
    with (ASSETS_DIR / "gprc_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_metadata() -> None:
    """Persist environment metadata for the notebook status banner."""
    device_info = detect_device_info()
    meta = {
        "model_name": MODEL_NAME,
        "cuda_available": device_info["kind"] == "cuda",
        "live_mode_available": device_info["live_mode_available"],
        "device_kind": device_info["kind"],
        "device_name": device_info["device_name"],
        "saes_available": saes_available(),
        "device": device_info["device"],
    }
    with (ASSETS_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def main() -> None:
    """Generate all notebook asset caches."""
    ensure_assets_dir()
    device = get_torch_device()
    print(f"Using device: {device}")
    save_behavioral_caches(device)
    save_intervention_caches()
    save_intervention_sweeps()
    save_activation_caches()
    print("Computing visualization caches...")
    save_token_activation_caches()
    save_attention_caches()
    save_attribution_caches(device)
    save_prefix_probability_caches(device)
    save_top_next_token_cache(device)
    save_atp_ig_cache()
    save_probe_cache()
    save_gprc_cache()
    save_metadata()
    save_cache_manifest()
    print(f"Caches written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
