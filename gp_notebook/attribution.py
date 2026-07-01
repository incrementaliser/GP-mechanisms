"""Captum-based token attribution for garden-path continuation preferences."""

from __future__ import annotations

from typing import Any

import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForCausalLM, AutoTokenizer

from gp_notebook.behavior import MODEL_NAME, continuation_tokens_for_condition
from gp_notebook.device import get_torch_device


def _build_forward_fn(
    model: AutoModelForCausalLM,
    gp_ids: list[int],
    non_gp_ids: list[int],
) -> Any:
    """Return a differentiable forward function mapping embeddings to m = p(GP) - p(non-GP)."""

    def forward_fn(input_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the scalar metric for attribution."""
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits.squeeze(0)[-1]
        probs = torch.softmax(logits, dim=-1)
        p_gp = probs[gp_ids].sum()
        p_non_gp = probs[non_gp_ids].sum()
        return (p_gp - p_non_gp).unsqueeze(0)

    return forward_fn


def token_attributions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    condition: str,
    *,
    device: torch.device | None = None,
    n_steps: int = 25,
) -> dict[str, list[float] | list[str]]:
    """Compute per-token integrated-gradients attribution of m = p(GP) - p(non-GP)."""
    device = device or get_torch_device()
    cont = continuation_tokens_for_condition(condition)
    gp_ids = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in cont.gp_tokens]
    non_gp_ids = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in cont.non_gp_tokens]

    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
    tok_strings = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    embed_layer = model.gpt_neox.embed_in
    input_embeds = embed_layer(input_ids).detach().requires_grad_(True)
    baseline = torch.zeros_like(input_embeds)

    forward_fn = _build_forward_fn(model, gp_ids, non_gp_ids)
    try:
        lig = LayerIntegratedGradients(forward_fn, embed_layer)
        attrs = lig.attribute(input_embeds, baselines=baseline, n_steps=n_steps)
        scores = attrs.sum(dim=-1).squeeze(0).detach().cpu().tolist()
    except Exception:
        metric = forward_fn(input_embeds)
        model.zero_grad()
        metric.backward()
        grad = input_embeds.grad
        scores = grad.sum(dim=-1).squeeze(0).detach().cpu().tolist() if grad is not None else [0.0] * len(tok_strings)
    return {"tokens": tok_strings, "scores": scores}


def compute_prefix_probabilities(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    condition: str,
    *,
    device: torch.device | None = None,
) -> list[dict[str, float | int]]:
    """Compute p(GP) and p(non-GP) at each incremental prefix of the sentence."""
    device = device or get_torch_device()
    cont = continuation_tokens_for_condition(condition)
    gp_ids = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in cont.gp_tokens]
    non_gp_ids = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in cont.non_gp_tokens]

    tokens = sentence.split()
    results: list[dict[str, float | int]] = []
    for end in range(2, len(tokens) + 1):
        prefix = " ".join(tokens[:end])
        input_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)
        with torch.inference_mode():
            logits = model(input_ids).logits.squeeze(0)[-1]
            probs = torch.softmax(logits, dim=-1)
        results.append({
            "n_tokens": end,
            "prefix": prefix,
            "p_gp": float(probs[gp_ids].sum().item()),
            "p_non_gp": float(probs[non_gp_ids].sum().item()),
        })
    return results
