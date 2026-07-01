"""Lightweight AtP-IG style feature ranking for circuit discovery demo."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from gp_notebook.features import enrich_feature_table
from gp_notebook.paths import ASSETS_DIR, ensure_assets_dir

# Representative indirect-effect ranks for syntactic features (demo ordering).
SYNTACTIC_IE_PRIORS: dict[str, float] = {
    "subject detector": 0.52,
    "object detector": 0.39,
    "end-clause detector": 0.48,
    "clause detector": 0.45,
    "end of clause detector": 0.44,
    "CP verb detector": 0.35,
    "end of sentence detector": 0.33,
}


def rank_features_by_ie(condition: str, *, threshold: float = 0.1) -> pd.DataFrame:
    """Rank annotated features by a demo indirect-effect score for AtP-IG visualization."""
    df = enrich_feature_table(condition)
    df = df.copy()
    def _ie_prior(category: str) -> float:
        if category in SYNTACTIC_IE_PRIORS:
            return SYNTACTIC_IE_PRIORS[category]
        return 0.05 if category in {
            "subject detector", "object detector", "end-clause detector",
            "clause detector", "end of clause detector", "CP verb detector",
            "end of sentence detector",
        } else 0.01

    df["ie_hat"] = df["Category"].map(_ie_prior)
    df.loc[df["reading_side"] == "pro_gp", "ie_hat"] *= -1
    df["abs_ie"] = df["ie_hat"].abs()
    df = df.sort_values("abs_ie", ascending=False)
    df["in_circuit"] = df["abs_ie"] >= threshold
    return df


def atp_ig_payload(condition: str) -> dict[str, Any]:
    """Serialize top features for JSON cache used by the notebook."""
    ranked = rank_features_by_ie(condition, threshold=0.1)
    top = ranked.head(20)
    return {
        "condition": condition,
        "threshold": 0.1,
        "features": [
            {
                "Feature": row.Feature,
                "Category": row.Category,
                "Annotation": row.Annotation,
                "layer": int(row.layer),
                "reading_side": row.reading_side,
                "ie_hat": float(row.ie_hat),
                "in_circuit": bool(row.in_circuit),
            }
            for _, row in top.iterrows()
        ],
    }


def save_atp_ig_cache() -> None:
    """Write atp_ig_features.json for NPZ and NPS conditions."""
    ensure_assets_dir()
    payload = {cond: atp_ig_payload(cond) for cond in ("NPZ", "NPS")}
    path = ASSETS_DIR / "atp_ig_features.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_atp_ig_cache() -> dict[str, Any] | None:
    """Load cached AtP-IG demo data if present."""
    path = ASSETS_DIR / "atp_ig_features.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
