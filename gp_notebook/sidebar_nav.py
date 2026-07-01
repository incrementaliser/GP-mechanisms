"""Sidebar navigation helpers for the garden-path marimo notebook."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Final

from marimo._output.formatting import as_html
from marimo._output.hypertext import Html


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """One notebook section: internal value, rail label, and expanded title."""

    value: str
    rail_label: str
    full_title: str


SECTIONS: Final[tuple[SectionSpec, ...]] = (
    SectionSpec("intro", "home", "Introduction"),
    SectionSpec("m0", "0", "0 — Primer"),
    SectionSpec("m1", "1", "1 — Feel the garden path"),
    SectionSpec("m2", "2", "2 — Behavioral lab (Fig 2)"),
    SectionSpec("m3", "3", "3 — Feature microscope (Fig 3)"),
    SectionSpec("m4", "4", "4 — Intervention sandbox (Fig 4)"),
    SectionSpec("m5", "5", "5 — Multiple readings? (RQ2)"),
    SectionSpec("m6", "6", "6 — Repair vs reanalysis (RQ3)"),
    SectionSpec("m7", "7", "7 — Your garden-path sentence"),
)

MODULE_NAV_DEFAULT_KEY: Final[str] = "Introduction"


def wrap_with_class(item: object, class_name: str) -> Html:
    """Wrap a marimo output element in a div with the given CSS class."""
    inner = as_html(item).text
    return Html(f'<div class="{class_name}">{inner}</div>')


def module_nav_full_options() -> dict[str, str]:
    """Return full section labels for the open sidebar navigation radio."""
    return {spec.full_title: spec.value for spec in SECTIONS}


def _status_row(label: str, ok: bool, tooltip: str) -> str:
    """Render one status indicator row with a hover tooltip."""
    dot_class = "gp-dot gp-dot-green" if ok else "gp-dot gp-dot-red"
    safe_tooltip = html.escape(tooltip, quote=True)
    safe_label = html.escape(label)
    return (
        f'<div class="gp-status-row">'
        f'<span class="gp-status-indicator" data-tooltip="{safe_tooltip}">'
        f'<span class="{dot_class}" aria-hidden="true"></span>'
        f'<span class="gp-status-label">{safe_label}</span>'
        f"</span></div>"
    )


def system_status_html(
    gpu_ok: bool,
    gpu_tooltip: str,
    data_ok: bool,
    data_tooltip: str,
    saes_ok: bool,
    saes_tooltip: str,
    last_call: tuple[str, float] | None,
) -> str:
    """Render the compact three-light status widget as raw HTML."""
    rows = [
        _status_row("GPU", gpu_ok, gpu_tooltip),
        _status_row("Data", data_ok, data_tooltip),
        _status_row("SAEs", saes_ok, saes_tooltip),
    ]
    timing_html = ""
    if last_call is not None:
        operation, elapsed_s = last_call
        timing_html = (
            f'<div class="gp-status-timing">'
            f"last: {html.escape(operation)} {elapsed_s:.2f} s"
            f"</div>"
        )
    return (
        '<div class="gp-status-box">'
        + "".join(rows)
        + timing_html
        + "</div>"
    )
