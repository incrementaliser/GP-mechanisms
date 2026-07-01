"""Sidebar navigation helpers for the garden-path marimo notebook."""

from __future__ import annotations

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
