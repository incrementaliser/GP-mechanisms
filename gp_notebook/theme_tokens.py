"""Editable design tokens for the garden-path notebook look and feel.

Change values in ``GP_THEME`` to retune colours and fonts without hunting
through CSS, Plotly charts, or inline SVG/HTML helpers.
"""

from __future__ import annotations

from typing import Final

# === DESIGN TOKENS (edit me) ===
GP_THEME: Final[dict[str, str]] = {
    # Typography — loaded via Google Fonts in assets/theme-init.html
    "font_display": '"Fraunces", "Iowan Old Style", Georgia, serif',
    "font_body": '"Source Sans 3", "Source Sans Pro", system-ui, sans-serif',
    "font_mono": '"IBM Plex Mono", ui-monospace, monospace',
    "google_fonts_href": (
        "https://fonts.googleapis.com/css2?"
        "family=Fraunces:opsz,wght@9..144,500;9..144,700&"
        "family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&"
        "family=IBM+Plex+Mono:wght@400;500&display=swap"
    ),
    # Semantic reading colours
    "gp": "#B42318",
    "gp_soft": "#F9D4CF",
    "non_gp": "#1D4ED8",
    "non_gp_soft": "#DBEAFE",
    "neutral": "#78716C",
    "highlight": "#F59E0B",
    "highlight_soft": "#FDE68A",
    # Light chrome
    "accent": "#0F766E",
    "accent_soft": "#CCFBF1",
    "surface": "#F4F1EB",
    "surface_card": "#FFFcf7",
    "ink": "#1C1917",
    "muted": "#57534E",
    "border": "#E7E5E4",
    # Dark chrome
    "accent_dark": "#2DD4BF",
    "accent_soft_dark": "#134E4A",
    "surface_dark": "#1C1917",
    "surface_card_dark": "#292524",
    "ink_dark": "#F5F5F4",
    "muted_dark": "#A8A29E",
    "border_dark": "#44403C",
    "gp_dark": "#F87171",
    "non_gp_dark": "#93C5FD",
}


def theme_color(key: str, fallback: str = "#78716C") -> str:
    """Return one colour/font token from ``GP_THEME``."""
    return GP_THEME.get(key, fallback)


def css_variables_block() -> str:
    """Build a ``<style>`` block that maps ``GP_THEME`` onto CSS custom properties."""
    t = GP_THEME
    return f"""
<style id="gp-design-tokens">
:root {{
  --gp-font-display: {t["font_display"]};
  --gp-font-body: {t["font_body"]};
  --gp-font-mono: {t["font_mono"]};
  --gp-color-gp: {t["gp"]};
  --gp-color-gp-soft: {t["gp_soft"]};
  --gp-color-non-gp: {t["non_gp"]};
  --gp-color-non-gp-soft: {t["non_gp_soft"]};
  --gp-color-neutral: {t["neutral"]};
  --gp-color-highlight: {t["highlight"]};
  --gp-color-highlight-soft: {t["highlight_soft"]};
  --gp-accent: {t["accent"]};
  --gp-accent-soft: {t["accent_soft"]};
  --gp-surface: {t["surface"]};
  --gp-surface-card: {t["surface_card"]};
  --gp-ink: {t["ink"]};
  --gp-muted: {t["muted"]};
  --gp-border: {t["border"]};
  --gp-accent-bg: {t["accent_soft"]};
  --gp-accent-fg: {t["accent"]};
  --gp-muted-fg: {t["muted"]};
  --gp-status-bg: {t["surface_card"]};
  --gp-status-border: {t["border"]};
  --gp-status-muted: {t["muted"]};
  --gp-tooltip-bg: {t["ink"]};
  --gp-tooltip-fg: {t["surface_card"]};
}}
html[data-theme="dark"],
html.dark,
html.dark-theme,
body[data-theme="dark"],
body.dark,
body.dark-theme {{
  --gp-color-gp: {t["gp_dark"]};
  --gp-color-non-gp: {t["non_gp_dark"]};
  --gp-accent: {t["accent_dark"]};
  --gp-accent-soft: {t["accent_soft_dark"]};
  --gp-surface: {t["surface_dark"]};
  --gp-surface-card: {t["surface_card_dark"]};
  --gp-ink: {t["ink_dark"]};
  --gp-muted: {t["muted_dark"]};
  --gp-border: {t["border_dark"]};
  --gp-accent-bg: {t["accent_soft_dark"]};
  --gp-accent-fg: {t["accent_dark"]};
  --gp-muted-fg: {t["muted_dark"]};
  --gp-status-bg: {t["surface_card_dark"]};
  --gp-status-border: {t["border_dark"]};
  --gp-status-muted: {t["muted_dark"]};
  --gp-tooltip-bg: {t["surface_card_dark"]};
  --gp-tooltip-fg: {t["ink_dark"]};
}}
body, .markdown, .mo-markdown, .cm-editor {{
  font-family: var(--gp-font-body) !important;
}}
.gp-hero__title,
.gp-header__title,
h1, h2, h3 {{
  font-family: var(--gp-font-display) !important;
}}
code, pre, .mo-code {{
  font-family: var(--gp-font-mono) !important;
}}
</style>
"""


def google_fonts_link_html() -> str:
    """Return a stylesheet link tag for the fonts declared in ``GP_THEME``."""
    href = GP_THEME["google_fonts_href"]
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{href}" rel="stylesheet">'
    )
