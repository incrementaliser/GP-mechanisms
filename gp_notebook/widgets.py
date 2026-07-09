"""Custom anywidget components for the garden-path notebook.

The centerpiece is :class:`GardenPathScrubber`: readers drag through a sentence
token by token and watch the model's preference for the two syntactic readings
(measured p(GP) and p(non-GP) at every prefix) swing in real time, with an
animated parse fork. All probabilities are passed in from real model runs; the
widget does no modelling of its own.
"""

from __future__ import annotations

import anywidget
import traitlets

_SCRUBBER_ESM = r"""
function render({ model, el }) {
  const GP_COLOR = "#c0392b";
  const NON_GP_COLOR = "#2980b9";

  const root = document.createElement("div");
  root.className = "gp-scrubber";
  root.style.cssText = `
    font-family: system-ui, sans-serif;
    border: 1px solid rgba(148, 163, 184, 0.45);
    border-radius: 14px;
    padding: 18px 20px 14px 20px;
    max-width: 860px;
    background: rgba(148, 163, 184, 0.07);
  `;

  const hint = document.createElement("div");
  hint.style.cssText = "font-size: 12.5px; opacity: 0.75; margin-bottom: 10px;";
  hint.textContent =
    "Drag across the sentence (or click a word) to reveal it token by token \u2014 " +
    "the fork shows which reading the model prefers after each word.";
  root.appendChild(hint);

  const tokenRow = document.createElement("div");
  tokenRow.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; cursor: ew-resize; user-select: none; touch-action: none;";
  root.appendChild(tokenRow);

  const svgWrap = document.createElement("div");
  svgWrap.style.cssText = "display: flex; justify-content: center;";
  const SVG_W = 460, SVG_H = 170;
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${SVG_W} ${SVG_H}`);
  svg.setAttribute("width", "100%");
  svg.style.maxWidth = `${SVG_W}px`;
  svgWrap.appendChild(svg);
  root.appendChild(svgWrap);

  function svgEl(tag, attrs) {
    const node = document.createElementNS(svgNS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    return node;
  }

  const START = { x: SVG_W / 2, y: 22 };
  const FORK = { x: SVG_W / 2, y: 62 };
  const GP_END = { x: 92, y: 132 };
  const NON_GP_END = { x: SVG_W - 92, y: 132 };

  svg.appendChild(svgEl("line", {
    x1: START.x, y1: START.y, x2: FORK.x, y2: FORK.y,
    stroke: "#94a3b8", "stroke-width": 3, "stroke-linecap": "round",
  }));
  const gpBranch = svgEl("path", {
    d: `M ${FORK.x} ${FORK.y} L ${GP_END.x} ${GP_END.y}`,
    stroke: GP_COLOR, "stroke-width": 3, fill: "none",
    "stroke-linecap": "round", opacity: 0.55,
  });
  const nonGpBranch = svgEl("path", {
    d: `M ${FORK.x} ${FORK.y} L ${NON_GP_END.x} ${NON_GP_END.y}`,
    stroke: NON_GP_COLOR, "stroke-width": 3, fill: "none",
    "stroke-linecap": "round", opacity: 0.55,
  });
  svg.appendChild(gpBranch);
  svg.appendChild(nonGpBranch);

  const gpText = svgEl("text", {
    x: GP_END.x, y: GP_END.y + 22, "text-anchor": "middle",
    "font-size": 12, fill: GP_COLOR, "font-weight": 600,
  });
  const nonGpText = svgEl("text", {
    x: NON_GP_END.x, y: NON_GP_END.y + 22, "text-anchor": "middle",
    "font-size": 12, fill: NON_GP_COLOR, "font-weight": 600,
  });
  svg.appendChild(gpText);
  svg.appendChild(nonGpText);

  const ball = svgEl("circle", {
    cx: START.x, cy: START.y, r: 9,
    fill: "#f59e0b", stroke: "#b45309", "stroke-width": 2,
  });
  ball.style.transition = "cx 0.3s ease, cy 0.3s ease";
  svg.appendChild(ball);

  const barsWrap = document.createElement("div");
  barsWrap.style.cssText = "margin-top: 8px;";
  root.appendChild(barsWrap);

  function makeBar(color) {
    const row = document.createElement("div");
    row.style.cssText = "display: flex; align-items: center; gap: 10px; margin: 4px 0;";
    const label = document.createElement("span");
    label.style.cssText = `width: 96px; font-size: 12.5px; font-weight: 600; color: ${color}; text-align: right;`;
    const track = document.createElement("div");
    track.style.cssText =
      "flex: 1; height: 12px; border-radius: 6px; background: rgba(148,163,184,0.25); overflow: hidden;";
    const fill = document.createElement("div");
    fill.style.cssText = `height: 100%; width: 0%; background: ${color}; transition: width 0.3s ease;`;
    track.appendChild(fill);
    const value = document.createElement("span");
    value.style.cssText = "width: 74px; font-size: 12px; font-variant-numeric: tabular-nums;";
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    barsWrap.appendChild(row);
    return { label, fill, value };
  }
  const gpBar = makeBar(GP_COLOR);
  const nonGpBar = makeBar(NON_GP_COLOR);

  const readout = document.createElement("div");
  readout.style.cssText = "font-size: 12.5px; opacity: 0.8; margin-top: 6px; min-height: 1.2em;";
  root.appendChild(readout);

  el.appendChild(root);

  function setRevealedFromEvent(event) {
    const chips = Array.from(tokenRow.children);
    if (!chips.length) return;
    const x = event.clientX;
    let revealed = 1;
    chips.forEach((chip, index) => {
      const rect = chip.getBoundingClientRect();
      if (x >= rect.left + rect.width * 0.35) revealed = index + 1;
    });
    model.set("revealed", Math.max(1, revealed));
    model.save_changes();
  }

  let dragging = false;
  tokenRow.addEventListener("pointerdown", (event) => {
    dragging = true;
    tokenRow.setPointerCapture(event.pointerId);
    setRevealedFromEvent(event);
  });
  tokenRow.addEventListener("pointermove", (event) => {
    if (dragging) setRevealedFromEvent(event);
  });
  tokenRow.addEventListener("pointerup", () => { dragging = false; });
  tokenRow.addEventListener("pointercancel", () => { dragging = false; });

  function refresh() {
    const tokens = model.get("tokens");
    const pGp = model.get("p_gp");
    const pNonGp = model.get("p_non_gp");
    const revealed = Math.min(Math.max(model.get("revealed"), 1), tokens.length);
    const ambiguousIdx = model.get("ambiguous_index");

    gpText.textContent = model.get("gp_label");
    nonGpText.textContent = model.get("non_gp_label");

    tokenRow.replaceChildren();
    tokens.forEach((token, index) => {
      const chip = document.createElement("span");
      const shown = index < revealed;
      const isAmbiguous = index === ambiguousIdx;
      chip.textContent = shown ? token : "\u2022\u2022\u2022";
      chip.style.cssText = `
        padding: 5px 9px; border-radius: 8px; font-size: 16px;
        border: 2px solid ${isAmbiguous && shown ? "#f1c40f" : "transparent"};
        background: ${shown ? (isAmbiguous ? "rgba(241,196,64,0.25)" : "rgba(148,163,184,0.22)") : "rgba(148,163,184,0.10)"};
        opacity: ${shown ? 1 : 0.45};
        transition: background 0.2s ease, opacity 0.2s ease;
      `;
      tokenRow.appendChild(chip);
    });

    // Probability arrays are aligned with prefixes of length 2..n tokens.
    const probIdx = Math.min(Math.max(revealed - 2, -1), pGp.length - 1);
    if (probIdx < 0) {
      ball.setAttribute("cx", START.x);
      ball.setAttribute("cy", START.y);
      gpBar.fill.style.width = "0%";
      nonGpBar.fill.style.width = "0%";
      gpBar.value.textContent = "\u2014";
      nonGpBar.value.textContent = "\u2014";
      gpBar.label.textContent = "p(GP)";
      nonGpBar.label.textContent = "p(non-GP)";
      readout.textContent = "Reveal at least two words to query the model.";
      return;
    }

    const gp = pGp[probIdx];
    const nonGp = pNonGp[probIdx];
    const total = gp + nonGp;
    const share = total > 1e-9 ? gp / total : 0.5;

    // Ball rolls down the fork: progress = how far along, side = which branch.
    const side = share >= 0.5 ? "gp" : "non_gp";
    const lean = Math.abs(share - 0.5) * 2; // 0 = undecided, 1 = fully committed
    const target = side === "gp" ? GP_END : NON_GP_END;
    const cx = FORK.x + (target.x - FORK.x) * lean;
    const cy = FORK.y + (target.y - FORK.y) * lean;
    ball.setAttribute("cx", cx);
    ball.setAttribute("cy", cy);
    gpBranch.setAttribute("opacity", side === "gp" ? 0.95 : 0.35);
    nonGpBranch.setAttribute("opacity", side === "non_gp" ? 0.95 : 0.35);

    const scale = Math.max(gp, nonGp, 1e-9);
    gpBar.fill.style.width = `${(gp / scale) * 100}%`;
    nonGpBar.fill.style.width = `${(nonGp / scale) * 100}%`;
    gpBar.label.textContent = "p(GP)";
    nonGpBar.label.textContent = "p(non-GP)";
    gpBar.value.textContent = gp.toFixed(4);
    nonGpBar.value.textContent = nonGp.toFixed(4);

    const prefix = tokens.slice(0, revealed).join(" ");
    const delta = gp - nonGp;
    readout.textContent =
      `"${prefix}" \u2192 m = p(GP) \u2212 p(non-GP) = ${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`;
  }

  model.on("change:revealed", refresh);
  model.on("change:tokens", refresh);
  model.on("change:p_gp", refresh);
  refresh();
}

export default { render };
"""


class GardenPathScrubber(anywidget.AnyWidget):
    """Token scrubber with an animated parse fork driven by measured prefix probabilities.

    ``p_gp[i]`` / ``p_non_gp[i]`` hold the model's continuation probabilities
    after revealing ``i + 2`` tokens (prefixes shorter than two words are not
    scored). ``revealed`` syncs both ways so marimo cells can react to it.
    """

    _esm = _SCRUBBER_ESM

    tokens = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    p_gp = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    p_non_gp = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    gp_label = traitlets.Unicode("GP reading").tag(sync=True)
    non_gp_label = traitlets.Unicode("non-GP reading").tag(sync=True)
    ambiguous_index = traitlets.Int(-1).tag(sync=True)
    revealed = traitlets.Int(2).tag(sync=True)
