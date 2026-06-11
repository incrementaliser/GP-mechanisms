"""Device detection helpers for hybrid live/precomputed execution."""

from __future__ import annotations

from typing import Literal, TypedDict

import torch

DeviceKind = Literal["cuda", "mps", "cpu"]


class DeviceInfo(TypedDict):
    """Structured result from PyTorch-backed device auto-detection."""

    device: str
    kind: DeviceKind
    live_mode_available: bool
    device_name: str | None


def detect_device_kind() -> DeviceKind:
    """Pick the best available PyTorch backend: CUDA, then MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def get_torch_device() -> torch.device:
    """Return the auto-detected PyTorch device for model and tensor placement."""
    return torch.device(detect_device_kind())


def _device_name(kind: DeviceKind) -> str | None:
    """Return a human-readable accelerator name when one is available."""
    if kind == "cuda":
        return torch.cuda.get_device_name(torch.cuda.current_device())
    if kind == "mps":
        return "Apple Metal (MPS)"
    return None


def detect_device_info() -> DeviceInfo:
    """Auto-detect compute device using standard PyTorch availability checks."""
    kind = detect_device_kind()
    live = kind != "cpu"
    return DeviceInfo(
        device=str(torch.device(kind)),
        kind=kind,
        live_mode_available=live,
        device_name=_device_name(kind),
    )


def live_mode_available() -> bool:
    """Return True when live inference can run on an accelerator (CUDA or MPS)."""
    return detect_device_kind() != "cpu"


def device_status_message() -> str:
    """Human-readable summary of the auto-detected compute device."""
    info = detect_device_info()
    if info["device_name"] is not None:
        return f"Live mode available on {info['kind'].upper()}: {info['device_name']}"
    return "Live mode unavailable — using precomputed caches (CPU only)."
