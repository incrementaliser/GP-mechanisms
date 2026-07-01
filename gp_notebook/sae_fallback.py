"""Fallback synthetic SAE token activations when live nnsight traces fail."""

from __future__ import annotations

import pandas as pd

from gp_notebook.features import enrich_feature_table
from gp_notebook.paths import load_gp_dataset


def synthetic_token_activations(condition: str) -> pd.DataFrame:
    """Build demo token activations from annotated features and dataset sentences."""
    df = enrich_feature_table(condition)
    syntactic = df[df["reading_side"].isin({"pro_gp", "anti_gp"})].copy()
    gp_df = load_gp_dataset()
    prompts = gp_df[gp_df["condition"] == condition]["sentence_ambiguous"].tolist()[:3]
    rows: list[dict] = []
    for prompt in prompts:
        tokens = prompt.split()
        for _, feat in syntactic.iterrows():
            pos = int(feat["Position"])
            for idx, tok in enumerate(tokens):
                base = 0.4 if idx == pos else 0.08
                if feat["reading_side"] == "anti_gp" and idx == len(tokens) - 1:
                    base = 0.35
                rows.append(
                    {
                        "feature": feat["Annotation"],
                        "category": feat["Category"],
                        "reading_side": feat["reading_side"],
                        "position": idx,
                        "token": tok,
                        "activation": base,
                        "sentence": prompt,
                        "Feature": feat["Feature"],
                    }
                )
    return pd.DataFrame(rows)
