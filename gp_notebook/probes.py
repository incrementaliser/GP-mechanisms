"""Structural-probe action probabilities digitized from the paper's Figure 5.

Training the probes from scratch requires the Penn Treebank (not freely
redistributable) plus the authors' probe checkpoints, so this module ships the
paper's reported curves instead: values are digitized directly from the
Figure 5 panel in the arXiv source (also embedded in the notebook for
side-by-side comparison). Provenance: paper-reported, not regenerated here.
"""

from __future__ import annotations

import json
from typing import Any

from gp_notebook.paths import ASSETS_DIR, ensure_assets_dir

PROBE_LAYERS: tuple[str, ...] = ("embeds", "0", "1", "2", "3", "4", "5")

# Mean probe action probabilities, digitized from paper Figure 5.
# LEFT-ARC = garden-path reading (final noun depends on the verb);
# GEN = non-GP reading (no dependency); RIGHT-ARC is implausible.
PROBE_FIGURE5: dict[str, list[dict[str, float | str]]] = {
    "NPZ": [
        {"layer": "embeds", "LEFT-ARC": 0.58, "GEN": 0.42, "RIGHT-ARC": 0.01},
        {"layer": "0", "LEFT-ARC": 0.72, "GEN": 0.27, "RIGHT-ARC": 0.01},
        {"layer": "1", "LEFT-ARC": 0.77, "GEN": 0.23, "RIGHT-ARC": 0.00},
        {"layer": "2", "LEFT-ARC": 0.80, "GEN": 0.21, "RIGHT-ARC": 0.00},
        {"layer": "3", "LEFT-ARC": 0.82, "GEN": 0.18, "RIGHT-ARC": 0.00},
        {"layer": "4", "LEFT-ARC": 0.81, "GEN": 0.18, "RIGHT-ARC": 0.00},
        {"layer": "5", "LEFT-ARC": 0.28, "GEN": 0.45, "RIGHT-ARC": 0.28},
    ],
    "NPS": [
        {"layer": "embeds", "LEFT-ARC": 0.39, "GEN": 0.57, "RIGHT-ARC": 0.04},
        {"layer": "0", "LEFT-ARC": 0.37, "GEN": 0.63, "RIGHT-ARC": 0.00},
        {"layer": "1", "LEFT-ARC": 0.45, "GEN": 0.55, "RIGHT-ARC": 0.00},
        {"layer": "2", "LEFT-ARC": 0.42, "GEN": 0.58, "RIGHT-ARC": 0.00},
        {"layer": "3", "LEFT-ARC": 0.41, "GEN": 0.59, "RIGHT-ARC": 0.00},
        {"layer": "4", "LEFT-ARC": 0.36, "GEN": 0.64, "RIGHT-ARC": 0.00},
        {"layer": "5", "LEFT-ARC": 0.28, "GEN": 0.45, "RIGHT-ARC": 0.28},
    ],
}


def save_probe_cache() -> None:
    """Persist the digitized Figure 5 probe curves to assets/probe_figure5.json."""
    ensure_assets_dir()
    path = ASSETS_DIR / "probe_figure5.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(PROBE_FIGURE5, handle, indent=2)


def load_probe_cache() -> dict[str, Any]:
    """Load probe curves from cache or fall back to the built-in digitized values."""
    path = ASSETS_DIR / "probe_figure5.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        # Older caches lacked the embeds layer; prefer the corrected values.
        first = cached.get("NPZ", [{}])[0]
        if str(first.get("layer")) == "embeds":
            return cached
    return PROBE_FIGURE5
