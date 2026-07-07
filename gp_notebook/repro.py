"""Notebook helpers for displaying reproducibility-harness status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gp_notebook.paths import PROJECT_ROOT

REPRO_OUTPUTS_DIR: Path = PROJECT_ROOT / "repro" / "outputs"
STATUS_PATH: Path = REPRO_OUTPUTS_DIR / "status.json"
MATRIX_PATH: Path = REPRO_OUTPUTS_DIR / "reproducibility_matrix.json"
COMPARISON_PATH: Path = REPRO_OUTPUTS_DIR / "comparison_summary.json"


def load_repro_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Load one reproducibility-harness JSON file if it exists."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def repro_status_df() -> pd.DataFrame:
    """Return a table of paper targets and whether they were regenerated."""
    payload = load_repro_json(STATUS_PATH)
    if not isinstance(payload, dict):
        return pd.DataFrame(
            [
                {
                    "result_id": "not_run",
                    "paper_target": "Repro harness",
                    "state": "not_run",
                    "reproducibility": "not checked",
                    "command": "repro/run_repro.sh core",
                    "message": "Run the harness to generate status.json.",
                }
            ]
        )
    return pd.DataFrame(payload.get("results", []))


def reproducibility_matrix_df() -> pd.DataFrame:
    """Return the detailed result-to-code reproducibility matrix."""
    payload = load_repro_json(MATRIX_PATH)
    if not isinstance(payload, list):
        return pd.DataFrame()
    return pd.DataFrame(payload)


def comparison_summary() -> dict[str, Any]:
    """Return generated-vs-cache comparison details when available."""
    payload = load_repro_json(COMPARISON_PATH)
    return payload if isinstance(payload, dict) else {}


def reproducibility_summary_markdown() -> str:
    """Summarize the current harness status in a short Markdown block."""
    status = repro_status_df()
    if status.empty or (status["result_id"] == "not_run").all():
        return (
            "**Reproducibility status:** not run yet. "
            "Use `repro/run_repro.sh core` to regenerate feasible author-script outputs."
        )
    generated = int((status["reproducibility"] == "code-generated").sum())
    blockers = status[
        ~status["reproducibility"].isin({"code-generated", "not checked"})
    ]
    return (
        f"**Reproducibility status:** {generated}/{len(status)} reported targets currently "
        f"have code-generated outputs. {len(blockers)} target(s) remain blocked, illustrative, "
        "or require external artifacts."
    )
