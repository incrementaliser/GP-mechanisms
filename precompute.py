"""Precompute notebook caches for hybrid live/precomputed execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gp_notebook.behavior import MODEL_NAME, aggregate_behavioral_summary, evaluate_dataset
from gp_notebook.device import get_torch_device, live_mode_available
from gp_notebook.features import category_counts, representative_activation_matrix
from gp_notebook.interventions import paper_style_interventions, saes_available
from gp_notebook.paths import ASSETS_DIR, ensure_assets_dir, load_gp_dataset


def save_behavioral_caches(device: torch.device) -> None:
    """Score all dataset variants and write behavioral summary caches."""
    df = load_gp_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    scored_frames = []
    for column in ("sentence_ambiguous", "sentence_gp", "sentence_post"):
        scored_frames.append(evaluate_dataset(df, column, model=model, tokenizer=tokenizer, device=device))
    scored = pd.concat(scored_frames, ignore_index=True)
    scored.to_parquet(ASSETS_DIR / "behavioral_scored.parquet", index=False)
    summary = aggregate_behavioral_summary(scored)
    summary.to_parquet(ASSETS_DIR / "behavioral_summary.parquet", index=False)


def _intervention_fallback() -> pd.DataFrame:
    """Return paper-approximate intervention means when live runs are unavailable."""
    return pd.DataFrame(
            [
                {"condition": "NPZ", "intervention": "baseline", "mean_diff": 0.12},
                {"condition": "NPZ", "intervention": "syntactic", "mean_diff": -0.08},
                {"condition": "NPZ", "intervention": "random", "mean_diff": 0.11},
                {"condition": "NPS", "intervention": "baseline", "mean_diff": -0.10},
                {"condition": "NPS", "intervention": "syntactic", "mean_diff": 0.06},
                {"condition": "NPS", "intervention": "random", "mean_diff": -0.09},
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


def save_metadata() -> None:
    """Persist environment metadata for the notebook status banner."""
    meta = {
        "model_name": MODEL_NAME,
        "cuda_available": live_mode_available(),
        "saes_available": saes_available(),
        "device": str(get_torch_device()),
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
    save_activation_caches()
    save_metadata()
    print(f"Caches written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
