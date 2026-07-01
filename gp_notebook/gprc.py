"""Garden-path reading comprehension (GPRC) helpers for RQ3 module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gp_notebook.paths import PROJECT_ROOT

GPRC_CSV = PROJECT_ROOT / "data_csv" / "garden_path_samelen_readingcomp_contrastive.csv"

# Paper Table 3 accuracies (percent)
TABLE3_ACCURACIES: dict[str, dict[str, float]] = {
    "Pythia-70m": {"BoolQ": 42.8, "MCQA": 50.0, "GPRC NPS": 50.0, "GPRC NPZ": 50.0},
    "Gemma-2-2b": {"BoolQ": 70.7, "MCQA": 90.0, "GPRC NPS": 83.3, "GPRC NPZ": 70.9},
}

CIRCUIT_IOU: dict[str, float] = {"NPZ": 0.002, "NPS": 0.0}

SPURIOUS_GPRC_FEATURES: tuple[str, ...] = (
    "Certainly",
    "Of course",
    "Yes",
    "agreement phrases",
)


def load_gprc_dataset() -> pd.DataFrame:
    """Load contrastive GPRC question pairs from the project CSV."""
    return pd.read_csv(GPRC_CSV)


def gprc_table_df() -> pd.DataFrame:
    """Return Table 3 as a tidy dataframe for mo.ui.table."""
    rows: list[dict[str, str | float]] = []
    for model, tasks in TABLE3_ACCURACIES.items():
        for task, acc in tasks.items():
            rows.append({"Model": model, "Task": task, "Accuracy (%)": acc})
    return pd.DataFrame(rows)


def sample_gprc_items(condition: str | None = None, n: int = 3) -> pd.DataFrame:
    """Return a small sample of GPRC items for interactive display."""
    df = load_gprc_dataset()
    if condition:
        df = df[df["condition"] == condition.upper()]
    return df.head(n)
