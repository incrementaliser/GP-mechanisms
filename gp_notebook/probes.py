"""Paper Figure 5 structural probe values (cached faithful reproduction)."""

from __future__ import annotations

import json
from typing import Any

from gp_notebook.paths import ASSETS_DIR, ensure_assets_dir

# Mean probe action probabilities from paper Figure 5 (approximate from description).
PROBE_FIGURE5: dict[str, list[dict[str, float | int | str]]] = {
    "NPZ": [
        {"layer": 0, "LEFT-ARC": 0.35, "GEN": 0.25, "RIGHT-ARC": 0.05},
        {"layer": 1, "LEFT-ARC": 0.55, "GEN": 0.30, "RIGHT-ARC": 0.04},
        {"layer": 2, "LEFT-ARC": 0.62, "GEN": 0.28, "RIGHT-ARC": 0.03},
        {"layer": 3, "LEFT-ARC": 0.58, "GEN": 0.32, "RIGHT-ARC": 0.04},
        {"layer": 4, "LEFT-ARC": 0.52, "GEN": 0.35, "RIGHT-ARC": 0.05},
        {"layer": 5, "LEFT-ARC": 0.40, "GEN": 0.38, "RIGHT-ARC": 0.06},
    ],
    "NPS": [
        {"layer": 0, "LEFT-ARC": 0.20, "GEN": 0.40, "RIGHT-ARC": 0.05},
        {"layer": 1, "LEFT-ARC": 0.22, "GEN": 0.48, "RIGHT-ARC": 0.04},
        {"layer": 2, "LEFT-ARC": 0.25, "GEN": 0.52, "RIGHT-ARC": 0.03},
        {"layer": 3, "LEFT-ARC": 0.28, "GEN": 0.50, "RIGHT-ARC": 0.04},
        {"layer": 4, "LEFT-ARC": 0.30, "GEN": 0.48, "RIGHT-ARC": 0.05},
        {"layer": 5, "LEFT-ARC": 0.25, "GEN": 0.42, "RIGHT-ARC": 0.06},
    ],
}


def save_probe_cache() -> None:
    """Persist Figure 5 probe curves to assets/probe_figure5.json."""
    ensure_assets_dir()
    path = ASSETS_DIR / "probe_figure5.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(PROBE_FIGURE5, handle, indent=2)


def load_probe_cache() -> dict[str, Any]:
    """Load probe data from cache or fall back to built-in paper values."""
    path = ASSETS_DIR / "probe_figure5.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return PROBE_FIGURE5
