"""Human-readable formatting for GPT-NeoX BPE token strings."""

from __future__ import annotations

from collections.abc import Iterable

GPT_NEOX_SPACE = "\u0120"  # Ġ — leading-space marker in Pythia/GPT-NeoX BPE vocab


def format_bpe_token(token: str) -> str:
    """Strip GPT-NeoX BPE leading-space marker for notebook display."""
    return token[len(GPT_NEOX_SPACE) :] if token.startswith(GPT_NEOX_SPACE) else token


def format_bpe_tokens(tokens: Iterable[str]) -> list[str]:
    """Return display-friendly copies of BPE token strings."""
    return [format_bpe_token(token) for token in tokens]
