"""Sidebar navigation helpers for the garden-path marimo notebook."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Final, Literal

import marimo as mo
from marimo._output.formatting import as_html
from marimo._output.hypertext import Html

ThemeMode = Literal["light", "dark"]


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
    SectionSpec("rq1", "2", "2 — Syntax or heuristics? (RQ1)"),
    SectionSpec("m5", "3", "3 — Multiple readings? (RQ2)"),
    SectionSpec("m6", "4", "4 — Repair vs reanalysis (RQ3)"),
    SectionSpec("m7", "5", "5 — Your garden-path sentence"),
    SectionSpec("m8", "6", "6 — Outro"),
)

MODULE_NAV_DEFAULT_KEY: Final[str] = "Introduction"


def wrap_with_class(item: object, class_name: str) -> Html:
    """Wrap a marimo output element in a div with the given CSS class."""
    inner = as_html(item).text
    return Html(f'<div class="{class_name}">{inner}</div>')


def module_nav_full_options() -> dict[str, str]:
    """Return full section labels for the open sidebar navigation radio."""
    return {spec.full_title: spec.value for spec in SECTIONS}


def initial_theme(app_meta_theme: str, request_theme: str | None = None) -> ThemeMode:
    """Normalize marimo app-meta or request cookie theme to light or dark."""
    if request_theme in ("light", "dark"):
        return request_theme  # type: ignore[return-value]
    return "dark" if app_meta_theme == "dark" else "light"


def theme_from_request(request: object | None) -> ThemeMode | None:
    """Read the persisted notebook theme cookie from the active HTTP request."""
    if request is None:
        return None
    cookies = getattr(request, "cookies", None)
    if not isinstance(cookies, dict):
        return None
    value = cookies.get("gp-notebook-theme")
    return value if value in ("light", "dark") else None


def build_theme_toggle(theme: str, light_button: object, dark_button: object) -> Html:
    """Wrap reactive sun/moon marimo buttons in the sidebar theme-toggle chrome."""
    safe_theme: ThemeMode = "dark" if theme == "dark" else "light"
    state_class = "gp-theme-is-dark" if safe_theme == "dark" else "gp-theme-is-light"
    buttons_row = mo.hstack(
        [
            wrap_with_class(light_button, "gp-theme-btn gp-theme-btn-light"),
            wrap_with_class(dark_button, "gp-theme-btn gp-theme-btn-dark"),
        ],
        gap=0.15,
        align="center",
        justify="center",
    )
    return wrap_with_class(buttons_row, f"gp-theme-toggle {state_class}")


def _status_row(label: str, ok: bool, tooltip: str) -> str:
    """Render one status indicator row with a hover tooltip."""
    dot_class = "gp-dot gp-dot-green" if ok else "gp-dot gp-dot-red"
    safe_tooltip = html.escape(tooltip, quote=True)
    safe_label = html.escape(label)
    return (
        f'<div class="gp-status-row">'
        # Use data-gp-tip (not data-tooltip) so marimo does not add a second Radix tooltip.
        f'<span class="gp-status-indicator" data-gp-tip="{safe_tooltip}">'
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
