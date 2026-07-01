"""Feature-table helpers for circuit exploration and activation views."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from gp_notebook.paths import load_feature_table

SYNTACTIC_CATEGORIES = {
    "subject detector",
    "object detector",
    "end-clause detector",
    "clause detector",
    "end of clause detector",
    "CP verb detector",
    "end of sentence detector",
}

PRO_GP_CATEGORIES = {
    "object detector",
    "end-clause detector",
    "clause detector",
    "end of clause detector",
}

ANTI_GP_CATEGORIES = {
    "subject detector",
    "CP verb detector",
}


def parse_submodule(feature_id: str) -> tuple[str, int | None]:
    """Split a feature id like resid_4/14907 into submodule name and index."""
    submodule, idx = feature_id.split("/")
    return submodule, int(idx)


def layer_from_submodule(submodule: str) -> int:
    """Extract transformer layer index from a submodule label."""
    if submodule == "embed":
        return 0
    match = re.search(r"_(\d+)$", submodule)
    return int(match.group(1)) if match else 0


def enrich_feature_table(condition: str) -> pd.DataFrame:
    """Add parsed submodule, layer, and reading-side columns to a feature table."""
    df = load_feature_table(condition).copy()
    parsed = df["Feature"].map(parse_submodule)
    df["submodule"] = parsed.map(lambda x: x[0])
    df["feature_idx"] = parsed.map(lambda x: x[1])
    df["layer"] = df["submodule"].map(layer_from_submodule)
    df["reading_side"] = df["Category"].map(
        lambda cat: "pro_gp"
        if cat in PRO_GP_CATEGORIES
        else ("anti_gp" if cat in ANTI_GP_CATEGORIES else "other")
    )
    df["is_syntactic"] = df["Category"].isin(SYNTACTIC_CATEGORIES)
    return df


def category_counts(condition: str) -> pd.DataFrame:
    """Count features per category for circuit overview charts."""
    df = enrich_feature_table(condition)
    counts = (
        df.groupby(["Category", "reading_side", "layer"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["layer", "count"], ascending=[True, False])
    )
    return counts


def layer_narrative(condition: str) -> list[tuple[int, str]]:
    """Short prose snippets describing how feature types change across layers."""
    counts = category_counts(condition)
    layers = sorted(counts["layer"].unique())
    narrative: list[tuple[int, str]] = []
    for layer in layers:
        subset = counts[counts["layer"] == layer].sort_values("count", ascending=False)
        top = ", ".join(
            f"{row['Category']} ({int(row['count'])})" for _, row in subset.head(4).iterrows()
        )
        if layer <= 1:
            text = (
                f"Layer {layer}: mostly lexical detectors — {top}. "
                "These features track specific words rather than syntactic roles."
            )
        else:
            text = (
                f"Layer {layer}: syntactic structure emerges — {top}. "
                "Subject/object/clause features here directly encode competing readings."
            )
        narrative.append((layer, text))
    return narrative


def representative_activation_matrix(condition: str) -> pd.DataFrame:
    """Load real precomputed activations when available, else build synthetic approximation."""
    from gp_notebook.paths import load_parquet_cache

    cached = load_parquet_cache(f"token_activations_{condition.lower()}.parquet")
    if cached is not None and not cached.empty:
        return cached

    return _synthetic_activation_matrix(condition)


def _synthetic_activation_matrix(condition: str) -> pd.DataFrame:
    """Build a synthetic activation matrix for ambiguous inputs using paper-reported ranges."""
    df = enrich_feature_table(condition)
    syntactic = df[df["reading_side"].isin({"pro_gp", "anti_gp"})].copy()
    positions = sorted(syntactic["Position"].unique())
    rows: list[dict[str, float | str | int]] = []
    for _, feature in syntactic.iterrows():
        base = 0.35 if feature["reading_side"] == "pro_gp" else 0.30
        for pos in positions:
            activation = base if pos == feature["Position"] else base * 0.45
            rows.append(
                {
                    "feature": feature["Annotation"],
                    "category": feature["Category"],
                    "reading_side": feature["reading_side"],
                    "position": pos,
                    "activation": activation,
                }
            )
    return pd.DataFrame(rows)
