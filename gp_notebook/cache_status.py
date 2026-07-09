"""Cache manifest validation for hybrid notebook execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gp_notebook.paths import ASSETS_DIR, ensure_assets_dir

CacheTier = Literal["required", "optional", "live_only"]


@dataclass(frozen=True)
class CacheSpec:
    """One expected asset file and how critical it is for the notebook."""

    name: str
    tier: CacheTier
    description: str


CACHE_SPECS: tuple[CacheSpec, ...] = (
    CacheSpec("behavioral_summary.parquet", "required", "Figure 2 aggregate"),
    CacheSpec("behavioral_scored.parquet", "required", "Per-sentence behavioural scores"),
    CacheSpec("interventions.parquet", "required", "Figure 4 intervention bars"),
    CacheSpec("intervention_sweeps.parquet", "optional", "M4 slider amplitude grid"),
    CacheSpec("prefix_probs_npz.parquet", "optional", "M1 commitment timeline NPZ"),
    CacheSpec("prefix_probs_nps.parquet", "optional", "M1 commitment timeline NPS"),
    CacheSpec("token_activations_npz.parquet", "optional", "SAE activations NPZ"),
    CacheSpec("token_activations_nps.parquet", "optional", "SAE activations NPS"),
    CacheSpec("attributions_npz.json", "optional", "Token IG NPZ"),
    CacheSpec("attributions_nps.json", "optional", "Token IG NPS"),
    CacheSpec("attention_npz.json", "optional", "Attention patterns NPZ"),
    CacheSpec("attention_nps.json", "optional", "Attention patterns NPS"),
    CacheSpec("top_next_tokens.parquet", "optional", "M3 drill-down next tokens"),
    CacheSpec("group_effects_npz.parquet", "optional", "Measured group ablation effects NPZ"),
    CacheSpec("group_effects_nps.parquet", "optional", "Measured group ablation effects NPS"),
    CacheSpec("probe_figure5.json", "optional", "Structural probe Figure 5 (digitized)"),
    CacheSpec("gprc_summary.json", "optional", "RQ3 GPRC summary"),
    CacheSpec("metadata.json", "required", "Environment metadata"),
)


def cache_manifest() -> dict[str, bool]:
    """Return a map of cache filename to whether it exists on disk."""
    return {spec.name: (ASSETS_DIR / spec.name).exists() for spec in CACHE_SPECS}


def missing_required_caches() -> list[str]:
    """List required cache files that are absent."""
    return [
        spec.name
        for spec in CACHE_SPECS
        if spec.tier == "required" and not (ASSETS_DIR / spec.name).exists()
    ]


def cache_ready_fraction() -> float:
    """Fraction of all listed caches that are present (0–1)."""
    manifest = cache_manifest()
    if not manifest:
        return 0.0
    return sum(manifest.values()) / len(manifest)


def save_cache_manifest() -> Path:
    """Write cache_manifest.json summarising asset availability."""
    ensure_assets_dir()
    payload = {
        "files": cache_manifest(),
        "missing_required": missing_required_caches(),
        "ready_fraction": cache_ready_fraction(),
    }
    path = ASSETS_DIR / "cache_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def cache_status_markdown() -> str:
    """Markdown summary of cache health for the notebook banner."""
    missing = missing_required_caches()
    fraction = cache_ready_fraction()
    if not missing:
        return (
            f"**Data status:** {fraction:.0%} of visualisation caches loaded. "
            "Default mode is fully interactive."
        )
    return (
        f"**Data status:** {fraction:.0%} caches loaded. "
        f"Missing required: {', '.join(missing)}. "
        "Run `uv run python precompute.py` or enable live mode."
    )
