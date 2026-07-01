"""Singleton model/SAE loaders and timing helpers for live notebook cells."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from gp_notebook.behavior import MODEL_NAME
from gp_notebook.device import get_torch_device
from gp_notebook.paths import saes_available


@dataclass
class RuntimeState:
    """Holds lazily loaded models and the last live-run timing in seconds."""

    hf_model: Any | None = None
    hf_tokenizer: Any | None = None
    nnsight_model: Any | None = None
    dictionaries: dict[str, Any] | None = None
    last_runtime_s: float | None = None
    last_operation: str | None = None


_STATE = RuntimeState()


def get_hf_model() -> tuple[Any, Any]:
    """Load Pythia once and reuse across live scoring cells."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_torch_device()
    if _STATE.hf_model is None or _STATE.hf_tokenizer is None:
        _STATE.hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _STATE.hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        _STATE.hf_model.eval()
    return _STATE.hf_model, _STATE.hf_tokenizer


def get_nnsight_bundle() -> tuple[Any, dict[str, Any]]:
    """Load nnsight model and SAE dictionaries once for intervention cells."""
    import torch
    from nnsight import LanguageModel

    from gp_notebook.interventions import load_dictionaries

    if not saes_available():
        raise FileNotFoundError("Pythia SAE checkpoints are not available locally.")
    device = get_torch_device()
    if _STATE.nnsight_model is None:
        _STATE.nnsight_model = LanguageModel(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=str(device),
            dispatch=True,
        )
    if _STATE.dictionaries is None:
        _STATE.dictionaries = load_dictionaries(MODEL_NAME, _STATE.nnsight_model)
    return _STATE.nnsight_model, _STATE.dictionaries


def timed_call(operation: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` and record elapsed wall time on the shared runtime state."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _STATE.last_runtime_s = time.perf_counter() - start
    _STATE.last_operation = operation
    return result


def runtime_status_line() -> str:
    """Human-readable line for sidebar device/timing card."""
    device = get_torch_device()
    sae = "SAEs ready" if saes_available() else "SAEs missing"
    parts = [f"Device: `{device}`", sae]
    if _STATE.last_runtime_s is not None and _STATE.last_operation:
        parts.append(f"Last {_STATE.last_operation}: {_STATE.last_runtime_s:.2f}s")
    return " · ".join(parts)


def clear_runtime() -> None:
    """Drop cached models to free GPU memory."""
    _STATE.hf_model = None
    _STATE.hf_tokenizer = None
    _STATE.nnsight_model = None
    _STATE.dictionaries = None
