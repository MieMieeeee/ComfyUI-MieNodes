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
    bbox_json: str = "",
) -> str:
    """Assemble the user-side text for one upsampling call.

    Mirrors the official scaffolds: ``t2i``/``t2i_bbox`` take the short
    request directly (optionally preceded by a ``TARGET CANVAS: W:H`` line,
    which steers composition and must never be echoed back), ``i2i``/
    ``i2i_bbox`` add the reference-image framing line and close with
    "Output the caption.", and the ``*_bbox`` modes accept a user-supplied
    element table the model must preserve exactly.

    ``bbox_json`` is validated here rather than at the node, so a malformed
    table is rejected before any tokens are spent.
    """
    request = (prompt or "").strip()
    if not request:
        raise ValueError(
            "Flux3Klein: prompt is empty; there is nothing to upsample / "
            "prompt 为空，无内容可扩写"
        )
    canvas = normalize_canvas(target_canvas)
    table = _validated_bbox_json(bbox_json)

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


def _validated_bbox_json(bbox_json: str) -> str:
    """Return the element table verbatim after checking it parses.

    Upstream says a supplied table must be kept exactly, so the node passes
    the user's own string through untouched -- but it still has to be a JSON
    array, or the model is being handed a table it cannot reproduce.
    """
    table = (bbox_json or "").strip()
    if not table:
        return ""
    try:
        parsed = json.loads(table)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Flux3Klein: bboxes_json is not valid JSON ({exc}); fix it or "
            f"leave it empty to let the model plan the layout / "
            f"bboxes_json 不是合法 JSON，可留空让模型自拟布局"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(
            f"Flux3Klein: bboxes_json must be a JSON array of element rows, "
            f"got {type(parsed).__name__} / bboxes_json 必须是元素行组成的 JSON 数组"
        )
    return table


def strip_think(raw_text: str) -> str:
    """Remove any ``<think>...</think>`` wrapper and return the visible text."""
    if not raw_text:
        return ""
    return _THINK_BLOCK_RE.sub("", raw_text).strip()


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
        f"found in the reply / {mode or 'bbox'} 模式要求「caption 段落 + 一个 "
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
