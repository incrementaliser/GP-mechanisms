"""Editable design tokens for the garden-path notebook look and feel.

Change values in ``GP_THEME`` to retune colours and fonts without hunting
through CSS, Plotly charts, or inline SVG/HTML helpers.
"""

from __future__ import annotations

from typing import Final

# === DESIGN TOKENS (edit me) ===
# Ink-and-signal: cool paper, charcoal dark, vermillion GP vs slate-cyan non-GP.
GP_THEME: Final[dict[str, str]] = {
    # Typography — loaded via Google Fonts in assets/theme-init.html
    "font_display": '"Newsreader", "Iowan Old Style", Georgia, serif',
    "font_body": '"IBM Plex Sans", system-ui, sans-serif',
    "font_mono": '"JetBrains Mono", ui-monospace, monospace',
    "google_fonts_href": (
        "https://fonts.googleapis.com/css2?"
        "family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,700;1,6..72,500&"
        "family=IBM+Plex+Sans:ital,wght@0,400;0,600;0,700;1,400&"
        "family=JetBrains+Mono:wght@400;500&display=swap"
    ),
    # Semantic reading colours (the brand)
    "gp": "#C23B22",
    "gp_soft": "#F8D6CF",
    "non_gp": "#0E7490",
    "non_gp_soft": "#CFF4FC",
    "neutral": "#64748B",
    "highlight": "#D97706",
    "highlight_soft": "#FDE68A",
    # Light chrome — cool paper white, not warm cream
    "accent": "#0F766E",
    "accent_soft": "#CCFBF1",
    "surface": "#F4F7FA",
    "surface_card": "#FFFFFF",
    "ink": "#0F172A",
    "muted": "#475569",
    "border": "#CBD5E1",
    # Dark chrome — charcoal
    "accent_dark": "#2DD4BF",
    "accent_soft_dark": "#134E4A",
    "surface_dark": "#0B1220",
    "surface_card_dark": "#162032",
    "ink_dark": "#F1F5F9",
    "muted_dark": "#94A3B8",
    "border_dark": "#334155",
    "gp_dark": "#F87171",
    "non_gp_dark": "#22D3EE",
}


def theme_color(key: str, fallback: str = "#64748B") -> str:
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
  --gp-color-gp-soft: #7F1D1D;
  --gp-color-non-gp: {t["non_gp_dark"]};
  --gp-color-non-gp-soft: #164E63;
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
