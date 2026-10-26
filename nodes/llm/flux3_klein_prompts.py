"""FLUX.3 Image Klein prompt-upsampler helpers (loader map + reply splitter).

Companion module to ``flux3_klein_prompt_generator.py``. Holds:

* ``MODES``: the four official upsampler modes, in dropdown order. The codes
  are the keys of Black Forest Labs' own prompt packs, so the node and the
  tests branch on exactly the upstream vocabulary.
* ``PROMPT_PACKS``: which official pack to send. ``short`` is the
  ``system_prompts.json`` released with the PUP weights -- upstream's model
  card states any other LLM can reproduce the behaviour with it, so it is
  the default. ``long`` is ``pup_prompts_external_vlm.json`` from inference
  repo PR #5, intended for external VLMs and carrying many more few-shot
  examples. Same four keys, materially different text -- they are two
  separately published official sets, not one file with two spellings.
* ``load_system_prompt``: verbatim accessor. Both packs are bundled
  byte-for-byte and must never be edited; see ``prompts/flux3_klein/UPSTREAM.md``.
* ``build_user_message``: the user-side scaffold per mode. The framing lines
  and the closing instruction are copied from the official template for that
  mode, including the fact that ``i2i``/``i2i_bbox`` close with
  "Output the caption." while ``t2i_bbox`` closes with
  "Output the caption and the table."
* ``strip_think``: removes a leaked ``<think>...</think>`` wrapper.
* ``split_bbox_output``: deterministic split of the bbox modes' single-line
  contract ("global caption paragraph, one space, then ONE JSON array").

This module is deliberately connector-free and torch-free so the node layer
and the tests can both import it cheaply.
"""
from __future__ import annotations

import json
import re

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_dict
except ImportError:
    from .prompts.loader import load_prompt_dict


# Ordered dropdown entries shown in the node menu, bilingual
# ``code - 中文`` like the rest of the Prompt Generator family. The code is
# the upstream key; the parenthetical is purely for the menu.
MODES: tuple[str, ...] = (
    "t2i - 文生图(默认)",
    "i2i - 图生图编辑(需参考图)",
    "t2i_bbox - 文生图+布局框",
    "i2i_bbox - 编辑+布局框(需参考图)",
)

# Which official prompt pack to load. The short pack is what upstream ships
# with the weights and what its "any other LLM can do the job" sentence refers
# to; the long pack is the external-VLM variant from PR #5.
PROMPT_PACKS: tuple[str, ...] = (
    "short - 短版(HF 权重同源,默认)",
    "long - 长版(PR#5 外部VLM,示例更多)",
)

_PACK_FILES: dict[str, str] = {
    "short": "flux3_klein/system_prompts",
    "long": "flux3_klein/pup_prompts_external_vlm",
}

# Modes whose official contract is "caption paragraph + JSON element table".
BBOX_MODES: frozenset[str] = frozenset({"t2i_bbox", "i2i_bbox"})

# Modes that must be handed actual pixels -- their system prompt opens with
# "You see the reference image(s), attached in order", so a text-only reply
# would be the upsampler hallucinating about images it never saw.
VISION_MODES: frozenset[str] = frozenset({"i2i", "i2i_bbox"})

# Official ceiling for reference images (upstream multi-reference guide).
MAX_REFERENCE_IMAGES: int = 10

# Framing lines copied from the official user-message template. Kept as module
# constants so tests can assert the node emits them verbatim.
_I2I_HEADER = "Reference images follow in order (the first image, the second image, ...)."
# Upstream words the element-table header differently per mode, so it is not
# one shared constant.
_BBOX_HEADER_T2I = "Bounding boxes (optional, keep exactly if provided):"
_BBOX_HEADER_I2I = "Bounding boxes (optional):"
_OUT_CAPTION = "Output the caption."
_OUT_CAPTION_AND_TABLE = "Output the caption and the table."
_CANVAS_PREFIX = "TARGET CANVAS:"

# Upstream enables thinking off for PUP; external APIs cannot, so reasoning
# models may still leak a wrapper.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_mode_label(label: str) -> str:
    """Strip the `` - 中文`` suffix off a mode dropdown label and return the
    bare upstream key. Unknown / empty input returns ``""`` so callers can
    raise one loud error instead of silently upsampling in the wrong mode.
    """
    code = (label or "").split(" - ", 1)[0].strip()
    return code if code in _MODE_KEYS else ""


def parse_pack_label(label: str) -> str:
    """Strip the `` - 中文`` suffix off a prompt-pack dropdown label and
    return the bare key (``short`` / ``long``). Unknown input returns ``""``.
    """
    code = (label or "").split(" - ", 1)[0].strip()
    return code if code in _PACK_FILES else ""


_MODE_KEYS: frozenset[str] = frozenset(
    label.split(" - ", 1)[0].strip() for label in MODES
)


def load_system_prompt(mode: str, pack: str) -> str:
    """Return the bundled official system prompt for ``mode`` from ``pack``.

    Raises ``ValueError`` on an unknown mode or pack, and ``KeyError`` if the
    pack file is missing the mode's key -- both are loud on purpose: a silent
    fallback would upsample against the wrong training distribution.
    """
    pack_file = _PACK_FILES.get(pack or "")
    if not pack_file:
        raise ValueError(
            f"Flux3Klein: unknown prompt_pack {pack!r}; expected one of "
            f"{sorted(_PACK_FILES)}"
        )
    data = load_prompt_dict(pack_file)
    if mode not in data:
        raise ValueError(
            f"Flux3Klein: prompt pack {pack_file!r} has no {mode!r} entry "
            f"(found {list(data)})"
        )
    return data[mode]


def normalize_canvas(target_canvas: str) -> str:
    """Normalize an optional canvas hint to bare ``W:H``, or ``""`` to omit.

    Accepts ``16:9``, ``16 : 9``, a float like ``1.7778`` (converted to the
    nearest common ratio), and ``auto`` (treated as absent -- upstream reads
    a missing canvas line as "compose freely"). Anything else raises, because
    a malformed ratio would otherwise reach the model as a silent no-op.
    """
    raw = (target_canvas or "").strip()
    if not raw or raw.lower() == "auto":
        return ""
    if ":" in raw:
        w, _, h = raw.partition(":")
        try:
            wi, hi = int(w.strip()), int(h.strip())
        except ValueError as exc:
            raise ValueError(
                f"Flux3Klein: target_canvas {raw!r} is not a W:H ratio / "
                f"target_canvas 应为 W:H 比例或 auto"
            ) from exc
        if wi <= 0 or hi <= 0:
            raise ValueError(
                f"Flux3Klein: target_canvas {raw!r} has a non-positive side / "
                f"target_canvas 宽高必须为正数"
            )
        return f"{wi}:{hi}"
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Flux3Klein: target_canvas {raw!r} is neither W:H nor a number / "
            f"target_canvas 需为 W:H 或数值"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Flux3Klein: target_canvas {raw!r} must be positive / "
            f"target_canvas 必须为正数"
        )
    for w, h in _COMMON_RATIOS:
        if abs(value - w / h) < 0.01:
            return f"{w}:{h}"
    raise ValueError(
        f"Flux3Klein: cannot express numeric target_canvas {raw!r} as a W:H "
        f"ratio; use e.g. 16:9 / 请改用 W:H 形式"
    )


_COMMON_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16),
    (21, 9), (5, 4), (4, 5), (7, 5), (2, 1), (1, 2),
)


def build_user_message(
    mode: str,
    prompt: str,
    *,
    target_canvas: str = "",
    rows: list | None = None,
) -> str:
    """Assemble the user-side text for one upsampling call.

    Mirrors the official scaffolds: ``t2i``/``t2i_bbox`` take the short
    request directly (optionally preceded by a ``TARGET CANVAS: W:H`` line,
    which steers composition and must never be echoed back), ``i2i``/
    ``i2i_bbox`` add the reference-image framing line and close with
    "Output the caption.", and the ``*_bbox`` modes add the element table the
    user marked on the canvas.

    In the bbox modes an empty ``prompt`` is legal: a supplied element table
    already describes the whole scene, so the request is composed from the
    rows' own ``desc`` fields rather than rejected. The text modes have no
    such fallback and still require a request.

    ``rows`` are the canvas rows, already parsed; the table is the only
    structured input, and it always reaches the model.
    """
    request = (prompt or "").strip()
    canvas = normalize_canvas(target_canvas)
    rows = rows or []

    if not request:
        # A supplied table already describes the scene, so its own `desc`
        # fields can stand in for a prose request. If it has none, there is
        # genuinely nothing to work from.
        request = _request_from_rows(rows) if mode in BBOX_MODES else ""
        if not request:
            raise ValueError(
                "Flux3Klein: prompt is empty; there is nothing to upsample / "
                "prompt 为空，无内容可扩写"
            )

    table = _table_text(rows)

    lines: list[str] = []
    if mode in VISION_MODES:
        lines.append(_I2I_HEADER)
        lines.append(f"User request: {request}")
        if table:
            lines.append(_BBOX_HEADER_I2I)
            lines.append(table)
        lines.append(_OUT_CAPTION)
    else:
        if canvas:
            lines.append(f"{_CANVAS_PREFIX} {canvas}")
        if mode == "t2i_bbox":
            lines.append(f"User request: {request}")
            if table:
                lines.append(_BBOX_HEADER_T2I)
                lines.append(table)
            lines.append(_OUT_CAPTION_AND_TABLE)
        else:
            lines.append(request)
    return "\n".join(lines)


def supplied_boxes(table: list) -> set:
    """Every coordinate the user supplied, in any row shape.

    Collects ``bbox`` (generation shape), ``src_bbox`` and ``tgt_bbox``
    (edit shape) from every row, so one set covers both conventions.

    Returns a set of 4-tuples. A ``None`` box (a New row's ``src_bbox``, a
    Remove row's ``tgt_bbox``) contributes nothing -- there is no coordinate
    to have come from.
    """
    boxes: set = set()
    if not isinstance(table, list):
        return boxes
    for row in table:
        if not isinstance(row, dict):
            continue
        for key in ("bbox", "src_bbox", "tgt_bbox"):
            value = row.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                try:
                    boxes.add(tuple(int(v) for v in value))
                except (TypeError, ValueError):
                    continue
    return boxes


def check_box_provenance(returned: list, supplied: set) -> list:
    """Return the supplied boxes that the reply does not account for.

    A supplied box is *accounted for* when some returned box overlaps it.
    Overlap rather than equality, because the official annotation system
    tightens a marked region onto the object inside it: a hand-drawn box is
    usually larger than the thing it marks, and upstream's own behaviour is
    to refine it (a marked ``[438, 31, 688, 188]`` comes back as
    ``src_bbox [572, 34, 681, 181]``). Holding the model to byte equality
    would forbid exactly the refinement that makes an edit land on the
    object instead of near it.

    Rows the model adds are allowed -- upstream expects background and anchor
    rows whose boxes the model read off the image -- so the check runs in one
    direction only: every region the user marked has to be represented.

    Returns the unaccounted boxes; empty means the reply honoured every
    marked region.
    """
    if not supplied:
        return []
    returned_boxes: list = []
    for row in returned if isinstance(returned, list) else []:
        if not isinstance(row, dict):
            continue
        for key in ("bbox", "src_bbox", "tgt_bbox"):
            value = row.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                try:
                    returned_boxes.append(tuple(int(v) for v in value))
                except (TypeError, ValueError):
                    continue
    return [
        box for box in supplied
        if not any(_box_iou(box, other) >= MIN_SUPPLIED_BOX_IOU
                   for other in returned_boxes)
    ]


# How much a returned box must overlap a supplied box to count as "the same
# region". 10% is loose enough to accept a box tightened onto the object
# inside a larger marked region, and tight enough to catch a reply that
# ignored the marked region entirely.
MIN_SUPPLIED_BOX_IOU: float = 0.1


def _box_iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two (y_min, x_min, y_max, x_max) boxes."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    inter_h = min(ay1, by1) - max(ay0, by0)
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    if inter_h <= 0 or inter_w <= 0:
        return 0.0
    inter = inter_h * inter_w
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def is_edit_shaped(table: list) -> bool:
    """True when the table already carries source/target boxes.

    ``CreateBoundingBoxes`` emits generation-shaped rows (``bbox`` only);
    a hand-written editing table carries ``src_bbox`` / ``tgt_bbox``. The two
    shapes carry different promises about what the model may change, so the
    callers need to tell them apart.
    """
    if not isinstance(table, list):
        return False
    return any(
        isinstance(row, dict) and ("src_bbox" in row or "tgt_bbox" in row)
        for row in table
    )


def elements_to_table(elements) -> list:
    """Convert a ``CreateBoundingBoxes`` elements array into official rows.

    ``CreateBoundingBoxes.elements`` is already in the official coordinate
    convention -- ``build_elements`` in ComfyUI normalizes every box to
    ``[y_min, x_min, y_max, x_max]`` as integers clamped to a 0-1000 grid, y
    first -- so the only work is assigning ids and quoting text rows.

    This mirrors ``Flux3LayoutPrompt.execute`` exactly, including its
    desc-plus-quoted-text rule and its dropping of colour palettes (upstream
    says palettes are not used; colours belong in the description). Doing it
    here rather than hand-writing the JSON means the boxes drawn on the canvas
    reach the model in the exact form upstream asks for.

    Raises ``ValueError`` when the array is empty or a row has no usable box,
    so a mis-wired socket fails before any tokens are spent.
    """
    rows: list[dict] = []
    items = elements if isinstance(elements, list) else []
    for i, el in enumerate(items):
        if not isinstance(el, dict):
            continue
        bbox = el.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"Flux3Klein: element {i + 1} has no usable 'bbox' as "
                f"[y_min, x_min, y_max, x_max] on a 0-1000 grid; it came from "
                f"{el!r} / 元素 {i + 1} 的 bbox 缺失或格式不对"
            )
        desc = str(el.get("desc") or "")
        if el.get("type") == "text" and el.get("text"):
            # Upstream requires rendered text verbatim inside double quotes.
            desc = f'{desc} "{el["text"]}"'.strip()
        rows.append({"id": f"obj_{i + 1}", "bbox": list(bbox), "desc": desc})
    if not rows:
        raise ValueError(
            "Flux3Klein: the elements input is empty; draw at least one box in "
            "Create Bounding Boxes, or leave it unconnected and let the model "
            "plan the layout / elements 为空：请在 Create Bounding Boxes "
            "里至少画一个框，或断开该输入让模型自行规划布局"
        )
    return rows


def _request_from_rows(rows: list) -> str:
    """Compose a scene request from the rows' own ``desc`` fields.

    Used when a bbox mode runs on a supplied table with no prose request. The
    table is the authoritative description of the scene, so this reads the
    descriptions back out rather than inventing any new content.
    """
    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        desc = str(row.get("desc") or "").strip()
        if desc:
            parts.append(desc)
    if not parts:
        return ""
    return "a scene containing: " + "; ".join(parts)


def _table_text(rows: list) -> str:
    return json.dumps(rows, ensure_ascii=False) if rows else ""


def strip_think(raw_text: str) -> str:
    """Remove any ``<think>...</think>`` wrapper and return the visible text."""
    if not raw_text:
        return ""
    return _THINK_BLOCK_RE.sub("", raw_text).strip()


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping ``` or ```json fence around the whole reply.

    Upstream asks for a bare caption line, but models habitually fence JSON
    when they emit a table. The fence breaks the "one space then the array"
    contract without changing the content, so it is removed here rather than
    being rejected.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = re.sub(r"^```[A-Za-z0-9_+-]*\s*", "", stripped)
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    return body.strip()


def split_bbox_output(raw_text: str, *, mode: str = "") -> tuple[str, str, list | None]:
    """Split a bbox-mode reply into ``(caption, bbox_json, table)``.

    Upstream's bbox contract is a single line: the global caption paragraph,
    one space, then ONE JSON array. The array boundary is found by trying
    candidate ``[`` positions from the right and keeping the first whose tail
    parses -- not by splitting on the first bracket, which would cut a
    caption that quotes bracketed text (or a nested array inside a desc).

    Non-bbox modes return ``(text, "", None)``. A bbox-mode reply with no
    parseable array raises, because feeding the raw line to the image model
    would hand it a JSON blob inside the prompt.
    """
    text = strip_think(raw_text)
    if mode not in BBOX_MODES:
        return text, "", None
    text = _strip_code_fence(text)

    # Official contract puts one space between the paragraph and the array, so
    # try that boundary first; then fall back to scanning candidate brackets
    # right-to-left until a tail parses as an array.
    candidates: list[int] = []
    head, sep, tail = text.partition(" ")
    if sep and tail.lstrip().startswith("["):
        candidates.append(len(head) + 1 + len(tail) - len(tail.lstrip()))
    candidates.extend(i for i in range(len(text) - 1, -1, -1) if text[i] == "[")

    for start in candidates:
        chunk = text[start:]
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        caption = text[:start].strip()
        if not caption:
            continue
        return caption, chunk.strip(), parsed

    raise ValueError(
        f"Flux3Klein: mode {mode or 'bbox'} requires the caption paragraph "
        f"followed by one JSON element table, but no parseable array was "
        f"found in the reply. Reply tail: {text[-240:]!r} / "
        f"{mode or 'bbox'} 模式要求「caption 段落 + 一个 "
        f"JSON 元素表」，回复里没有可解析的数组"
    )


__all__ = [
    "MODES",
    "PROMPT_PACKS",
    "BBOX_MODES",
    "VISION_MODES",
    "MAX_REFERENCE_IMAGES",
    "parse_mode_label",
    "parse_pack_label",
    "load_system_prompt",
    "normalize_canvas",
    "build_user_message",
    "strip_think",
    "split_bbox_output",
]
