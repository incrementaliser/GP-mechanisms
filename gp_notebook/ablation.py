"""Exact zero-ablation effects for annotated feature groups (RQ1 causal evidence).

Instead of approximating each feature's indirect effect with AtP-IG (paper
section 2.3), this module measures the *exact* effect of zero-ablating every
annotated feature group on m = p(GP) - p(non-GP), averaged over the dataset.
The results are cached to ``assets/group_effects_{npz,nps}.parquet`` so the
notebook can display real, recomputed causal evidence in cached mode.
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Tuple

import pandas as pd
import torch

from gp_notebook.behavior import MODEL_NAME, continuation_tokens_for_condition
from gp_notebook.features import ANTI_GP_CATEGORIES, PRO_GP_CATEGORIES, enrich_feature_table
from gp_notebook.paths import ASSETS_DIR, load_gp_dataset, load_parquet_cache

FeatureEdit = Tuple[int, int, float]

# Map raw annotation categories onto the functional families used in the
# paper's Figure 3 (subject / object / clause-end detectors, plus heuristics).
GROUP_LABELS: dict[str, str] = {
    "subject detector": "Subject detectors",
    "object detector": "Object detectors",
    "end-clause detector": "Clause-end detectors",
    "clause detector": "Clause-end detectors",
    "end of clause detector": "Clause-end detectors",
    "in-clause detector": "Clause-end detectors",
    "end of sentence detector": "Clause-end detectors",
    "CP verb detector": "CP-verb detectors",
    "word detector": "Word detectors (heuristic)",
    "noun detector": "Noun detectors (heuristic)",
    "verb detector": "Verb detectors (heuristic)",
}

FALLBACK_GROUP = "Uninterpretable / other"

DICT_SIZE = 32768


def group_label(category: str) -> str:
    """Return the functional family label for one raw annotation category."""
    return GROUP_LABELS.get(category, FALLBACK_GROUP)


def group_reading_side(categories: pd.Series) -> str:
    """Classify a feature group as pro-GP, anti-GP, or other by its member categories."""
    pro = categories.isin(PRO_GP_CATEGORIES).sum()
    anti = categories.isin(ANTI_GP_CATEGORIES).sum()
    if pro > anti:
        return "pro_gp"
    if anti > pro:
        return "anti_gp"
    return "other"


def zero_ablation_edits(features: pd.DataFrame) -> dict[str, List[FeatureEdit]]:
    """Build edits that clamp every listed feature to zero at its annotated position."""
    edits: DefaultDict[str, List[FeatureEdit]] = defaultdict(list)
    for _, row in features.iterrows():
        submodule, idx = str(row["Feature"]).split("/")
        edits[submodule].append((int(row["Position"]), int(idx), 0.0))
    return dict(edits)


def random_control_edits(features: pd.DataFrame, *, seed: int = 0) -> dict[str, List[FeatureEdit]]:
    """Build zero-clamp edits for count-matched random features at the same positions."""
    rng = torch.Generator().manual_seed(seed)
    annotated: DefaultDict[str, set[int]] = defaultdict(set)
    for _, row in features.iterrows():
        submodule, idx = str(row["Feature"]).split("/")
        annotated[submodule].add(int(idx))

    edits: DefaultDict[str, List[FeatureEdit]] = defaultdict(list)
    for _, row in features.iterrows():
        submodule, idx = str(row["Feature"]).split("/")
        candidate = int(torch.randint(0, DICT_SIZE, (1,), generator=rng).item())
        while candidate in annotated[submodule]:
            candidate = int(torch.randint(0, DICT_SIZE, (1,), generator=rng).item())
        edits[submodule].append((int(row["Position"]), candidate, 0.0))
    return dict(edits)


def _continuation_token_ids(tokenizer, condition: str) -> tuple[list[int], list[int]]:
    """Return (GP, non-GP) continuation token id lists for one structure."""
    tokens = continuation_tokens_for_condition(condition)
    gp_ids = [tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.gp_tokens]
    non_gp_ids = [
        tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.non_gp_tokens
    ]
    return gp_ids, non_gp_ids


def measure_group_effects(condition: str, *, seed: int = 0) -> pd.DataFrame:
    """Zero-ablate each annotated feature group and measure the change in m.

    Returns one row per group with the ablated mean m, the no-edit baseline m,
    and their difference (the group's exact indirect effect on the metric).
    """
    from transformers import AutoTokenizer

    from gp_notebook.interventions import get_performance
    from gp_notebook.runtime import get_nnsight_bundle

    model, dictionaries = get_nnsight_bundle()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gp_ids, non_gp_ids = _continuation_token_ids(tokenizer, condition)

    gp_df = load_gp_dataset()
    prompts = gp_df[gp_df["condition"] == condition]["sentence_ambiguous"].tolist()

    features = enrich_feature_table(condition)
    features = features.copy()
    features["group"] = features["Category"].map(group_label)

    baseline_gp, baseline_non_gp = get_performance(
        model, prompts, (gp_ids, non_gp_ids), dictionaries, {}
    )
    baseline_m = float((baseline_gp - baseline_non_gp).mean())

    rows: list[dict[str, float | int | str]] = []
    for group, members in features.groupby("group"):
        edits = zero_ablation_edits(members)
        gp, non_gp = get_performance(model, prompts, (gp_ids, non_gp_ids), dictionaries, edits)
        ablated_m = float((gp - non_gp).mean())
        rows.append(
            {
                "condition": condition,
                "group": group,
                "n_features": int(len(members)),
                "reading_side": group_reading_side(members["Category"]),
                "baseline_m": baseline_m,
                "ablated_m": ablated_m,
                "delta_m": ablated_m - baseline_m,
                "kind": "annotated",
            }
        )

    random_edits = random_control_edits(features, seed=seed)
    gp, non_gp = get_performance(model, prompts, (gp_ids, non_gp_ids), dictionaries, random_edits)
    random_m = float((gp - non_gp).mean())
    rows.append(
        {
            "condition": condition,
            "group": "Random features (control)",
            "n_features": int(len(features)),
            "reading_side": "other",
            "baseline_m": baseline_m,
            "ablated_m": random_m,
            "delta_m": random_m - baseline_m,
            "kind": "random_control",
        }
    )
    return pd.DataFrame(rows)


def save_group_effect_caches() -> None:
    """Compute and cache exact group ablation effects for both structures."""
    for condition in ("NPZ", "NPS"):
        df = measure_group_effects(condition)
        path = ASSETS_DIR / f"group_effects_{condition.lower()}.parquet"
        df.to_parquet(path, index=False)
        print(f"  group_effects_{condition.lower()}.parquet: {len(df)} groups")


def load_group_effects(condition: str) -> pd.DataFrame | None:
    """Load cached group ablation effects for one structure, if present."""
    return load_parquet_cache(f"group_effects_{condition.lower()}.parquet")
