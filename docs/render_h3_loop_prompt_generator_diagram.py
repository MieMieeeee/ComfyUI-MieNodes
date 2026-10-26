"""Render the MiniMaxH3LoopPromptGenerator excalidraw JSON to SVG.

Pure Python (no browser, no Node, no Playwright). Reconstructs the same
shapes the build_h3_loop_prompt_generator_diagram.py emits: rectangles
with bound labels, free text, and arrows. Used to ship a visual PNG/SVG
of the parameter reference card without needing the Excalidraw app.
"""
import json
import re
from pathlib import Path
import html

SRC = Path(
    r"C:/Users/administered/PycharmProjects/ComfyUI-MieNodes/docs/"
    "h3_loop_prompt_generator_diagram.excalidraw"
)
OUT_SVG = Path(
    r"C:/Users/administered/PycharmProjects/ComfyUI-MieNodes/docs/"
    "h3_loop_prompt_generator_diagram.svg"
)

doc = json.loads(SRC.read_text(encoding="utf-8"))

# Map of rect id -> text element (bound label)
labels = {}
texts = []
rects = []
arrows = []
for el in doc["elements"]:
    t = el.get("type")
    if t == "text":
        if el.get("containerId"):
            labels[el["containerId"]] = el
        else:
            texts.append(el)
    elif t == "rectangle":
        rects.append(el)
    elif t == "arrow":
        arrows.append(el)

# Compute canvas bbox from all elements
xs0 = [e["x"] for e in doc["elements"]]
ys0 = [e["y"] for e in doc["elements"]]
xs1 = [e["x"] + e["width"] for e in doc["elements"]]
ys1 = [e["y"] + e["height"] for e in doc["elements"]]
x0, y0 = min(xs0) - 20, min(ys0) - 20
x1, y1 = max(xs1) + 20, max(ys1) + 20
w, h = x1 - x0, y1 - y0

svg_parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
    f'width="{w}" height="{h}" font-family="Virgil, Segoe UI, sans-serif">',
    f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>',
]

# Rectangles (with optional bound label)
for r in rects:
    rx = r["x"] - x0
    ry = r["y"] - y0
    rw = r["width"]
    rh = r["height"]
    stroke = r["strokeColor"]
    bg = r["backgroundColor"] if r["backgroundColor"] != "transparent" else "#ffffff"
    rxr = 8  # rounded corners
    svg_parts.append(
        f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="{rxr}" ry="{rxr}" '
        f'fill="{bg}" stroke="{stroke}" stroke-width="2"/>'
    )
    label = labels.get(r["id"])
    if label:
        text = label["text"]
        fs = label["fontSize"]
        # multi-line: split, draw tspans
        lines = text.split("\n")
        # vertical center, multi-line stacked from top
        line_h = fs * 1.25
        total = len(lines) * line_h
        start_y = ry + (rh - total) / 2 + fs
        tspans = []
        for i, line in enumerate(lines):
            tline = html.escape(line)
            tspans.append(
                f'<tspan x="{rx + rw/2:.1f}" y="{start_y + i * line_h:.1f}">{tline}</tspan>'
            )
        svg_parts.append(
            f'<text x="0" y="0" font-size="{fs}" text-anchor="middle" '
            f'fill="#1e1e1e">{"".join(tspans)}</text>'
        )

# Free text (no container)
for t in texts:
    tx = t["x"] - x0
    ty = t["y"] - y0 + t["fontSize"]
    fs = t["fontSize"]
    color = t.get("strokeColor", "#1e1e1e")
    align = t.get("textAlign", "left")
    anchor = {"left": "start", "center": "middle", "right": "end"}.get(align, "start")
    text = html.escape(t["text"])
    svg_parts.append(
        f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{fs}" text-anchor="{anchor}" '
        f'fill="{color}">{text}</text>'
    )

# Arrows
for a in arrows:
    pts = a["points"]
    if len(pts) < 2:
        continue
    ax = a["x"] - x0
    ay = a["y"] - y0
    x_start = ax + pts[0][0]
    y_start = ay + pts[0][1]
    x_end = ax + pts[-1][0]
    y_end = ay + pts[-1][1]
    stroke = a["strokeColor"]
    width = a["strokeWidth"]
    svg_parts.append(
        f'<line x1="{x_start:.1f}" y1="{y_start:.1f}" x2="{x_end:.1f}" y2="{y_end:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" marker-end="url(#arrow)"/>'
    )

# Arrowhead marker
svg_parts.insert(2,
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1e1e1e"/>'
    '</marker></defs>'
)

svg_parts.append("</svg>")
OUT_SVG.write_text("\n".join(svg_parts), encoding="utf-8")
print(f"wrote {OUT_SVG} ({w}x{h})")