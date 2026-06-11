"""Project paths and lightweight asset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
DATA_CSV: Path = PROJECT_ROOT / "data_csv" / "gp_same_len.csv"
FEATURE_RESULTS: Path = PROJECT_ROOT / "results" / "pythia-70m-deduped"
DICTIONARIES_DIR: Path = PROJECT_ROOT / "feature-circuits-gp" / "dictionaries"


def ensure_assets_dir() -> Path:
    """Create the assets directory if it does not already exist."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return ASSETS_DIR


def load_gp_dataset() -> pd.DataFrame:
    """Load the garden-path sentence dataset used throughout the paper."""
    return pd.read_csv(DATA_CSV)


def load_feature_table(condition: str) -> pd.DataFrame:
    """Load annotated SAE features for NPZ or NPS circuits."""
    filename = "npz_features.csv" if condition.upper() == "NPZ" else "nps_features.csv"
    return pd.read_csv(FEATURE_RESULTS / filename)


def load_json_cache(name: str) -> dict[str, Any] | list[Any] | None:
    """Load a JSON cache from assets/ if present."""
    path = ASSETS_DIR / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_parquet_cache(name: str) -> pd.DataFrame | None:
    """Load a parquet cache from assets/ if present."""
    path = ASSETS_DIR / name
    if not path.exists():
        return None
    return pd.read_parquet(path)
