"""Behavioural evaluation: p(GP) and p(non-GP) for garden-path sentences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gp_notebook.device import get_torch_device

MODEL_NAME = "EleutherAI/pythia-70m-deduped"


@dataclass(frozen=True)
class ContinuationTokens:
    """GP and non-GP continuation token strings for a structure type."""

    gp_tokens: tuple[str, ...]
    non_gp_tokens: tuple[str, ...]


def continuation_tokens_for_condition(condition: str) -> ContinuationTokens:
    """Map NPZ/NPS/MVRR to the paper's GP and non-GP continuation tokens."""
    if condition == "NPZ":
        return ContinuationTokens(gp_tokens=(",",), non_gp_tokens=(" was",))
    if condition in {"NPS", "MVRR"}:
        return ContinuationTokens(gp_tokens=(".",), non_gp_tokens=(" was",))
    raise ValueError(f"Unknown condition: {condition}")


def score_sentence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    condition: str,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute p(GP), p(non-GP), and their difference for one sentence."""
    device = device or get_torch_device()
    tokens = continuation_tokens_for_condition(condition)
    gp_ids = [tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.gp_tokens]
    non_gp_ids = [
        tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.non_gp_tokens
    ]
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        logits = model(input_ids).logits.squeeze(0)[-1]
        probs = torch.softmax(logits, dim=-1)
    p_gp = float(probs[gp_ids].sum().item())
    p_non_gp = float(probs[non_gp_ids].sum().item())
    return {"p_gp": p_gp, "p_non_gp": p_non_gp, "diff": p_gp - p_non_gp}


def evaluate_dataset(
    df: pd.DataFrame,
    sentence_col: str,
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Score every row in the dataset for one sentence column."""
    device = device or get_torch_device()
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
    rows: list[dict[str, float | str | int]] = []
    for _, row in df.iterrows():
        scores = score_sentence(
            model, tokenizer, row[sentence_col], row["condition"], device=device
        )
        rows.append(
            {
                "item": int(row["item"]),
                "condition": row["condition"],
                "sentence": row[sentence_col],
                "input_type": sentence_col.replace("sentence_", ""),
                **scores,
            }
        )
    return pd.DataFrame(rows)


def aggregate_behavioral_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean p(GP)-p(non-GP) by condition and input type (Figure 2)."""
    summary = (
        scored.groupby(["condition", "input_type"], as_index=False)["diff"]
        .agg(mean_diff="mean", sem_diff="sem")
        .sort_values(["condition", "input_type"])
    )
    return summary


def top_next_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    k: int = 8,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Return the top-k next-token probabilities for drill-down views."""
    device = device or get_torch_device()
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        logits = model(input_ids).logits.squeeze(0)[-1]
        probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=k)
    return pd.DataFrame(
        {
            "token": [tokenizer.decode([idx]) for idx in indices.tolist()],
            "probability": values.tolist(),
        }
    )
