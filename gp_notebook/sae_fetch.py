"""Download the Pythia-70m SAE checkpoints needed for live interventions.

The paper uses Marks et al.'s (2024) sparse autoencoders. They are distributed
as one ~2.3 GB zip on Hugging Face (repo ``saprmarks/pythia-70m-deduped-saes``)
containing 19 checkpoints: the embedding plus attention/MLP/residual outputs of
all six layers. This module streams the zip and unpacks it into
``feature-circuits-gp/dictionaries/`` so that live mode works on a fresh
machine (for example a molab GPU instance) without manual setup.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from gp_notebook.paths import DICTIONARIES_DIR

SAE_ZIP_URL = (
    "https://huggingface.co/saprmarks/pythia-70m-deduped-saes/resolve/main/"
    "dictionaries_pythia-70m-deduped_10.zip"
)

SAE_SUBMODULE_DIRS: tuple[str, ...] = (
    "embed",
    *(f"{module}_out_layer{i}" for i in range(6) for module in ("attn", "mlp", "resid")),
)

APPROX_ZIP_MB = 2260

ProgressCallback = Callable[[str, int, int], None]


def missing_sae_dirs() -> list[str]:
    """List submodule directories whose ae.pt checkpoint is not on disk."""
    root = DICTIONARIES_DIR / "pythia-70m-deduped"
    return [
        name
        for name in SAE_SUBMODULE_DIRS
        if not (root / name / "10_32768" / "ae.pt").exists()
    ]


def _download_zip(destination: Path, progress_callback: ProgressCallback | None) -> None:
    """Stream the SAE zip to ``destination``, reporting progress in MB."""
    tmp_path = destination.with_suffix(".part")
    request = urllib.request.Request(SAE_ZIP_URL, headers={"User-Agent": "gp-notebook/0.2"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp_path.open("wb") as out:
        total_bytes = int(response.headers.get("Content-Length") or APPROX_ZIP_MB * 2**20)
        done_bytes = 0
        while True:
            chunk = response.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
            done_bytes += len(chunk)
            if progress_callback is not None:
                progress_callback("downloading", done_bytes >> 20, total_bytes >> 20)
    tmp_path.rename(destination)


def download_saes(progress_callback: ProgressCallback | None = None) -> int:
    """Fetch and unpack all missing SAE checkpoints, returning the number installed.

    ``progress_callback(phase, done, total)`` is called with MB counts while
    downloading and file counts while extracting, so callers (for example a
    marimo progress bar) can display progress.
    """
    if not missing_sae_dirs():
        return 0
    DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DICTIONARIES_DIR / "dictionaries_pythia-70m-deduped_10.zip"
    if not zip_path.exists():
        _download_zip(zip_path, progress_callback)

    # Zip members are rooted at dictionaries/, so extract into its parent.
    extract_root = DICTIONARIES_DIR.parent
    installed = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        for index, member in enumerate(members, start=1):
            target = extract_root / member
            if not target.exists():
                archive.extract(member, extract_root)
                installed += 1
            if progress_callback is not None:
                progress_callback("extracting", index, len(members))
    zip_path.unlink(missing_ok=True)
    return installed
