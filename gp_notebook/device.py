"""Device detection helpers for hybrid live/precomputed execution."""

from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def live_mode_available() -> bool:
    """Return True when live model inference can run on an accelerator."""
    return torch.cuda.is_available()


def device_status_message() -> str:
    """Human-readable summary of the current compute device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"Live mode available on GPU: {name}"
    return "Live mode unavailable — using precomputed caches (CPU only)."
