"""MOSS-Image prompt-enhancer helpers (modes, box spaces, layout compiler).

Companion module to ``moss_image_prompt_generator.py``. It holds everything
that is deterministic, so the node layer only wires sockets and the LLM call:

* ``MODES`` -- MOSS-Image has no image-to-image mode, so there are exactly two:
  ``t2i`` (caption upsampling) and ``t2i_layout`` (spatial Layout).
* ``load_system_prompt`` -- verbatim accessor for the bundled prompts under
  ``nodes/llm/prompts/moss_image/``. The upsampler prompt is an official
  artifact and must never be edited; see ``prompts/moss_image/UPSTREAM.md``.
* ``elements_to_boxes`` -- converts ``CreateBoundingBoxes`` elements into
  MOSS' *draft* box space (absolute pixels, ``xyxy``, top-left origin).
* ``compile_layout_draft`` -- a faithful port of the official
  ``compile_prompt.py``: the draft space becomes the one-line compact JSON the
  image model consumes (``abs_centralized`` / ``yxyx``).
* ``parse_layout_reply`` -- pulls the JSON draft out of a model reply.
* ``enforce_supplied_boxes`` -- the "keep the boxes verbatim" guarantee: the
  user's pixels win over the model's, because the official upsampler is
  documented to rewrite layouts.

The module is connector-free and torch-free so the node layer and the tests
can both import it cheaply.
"""
from __future__ import annotations

import json
import math
import re

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text


# Ordered dropdown entries shown in the node menu, bilingual ``code - 中文``
# like the rest of the Prompt Generator family. The code is the upstream key.
MODES: tuple[str, ...] = (
    "t2i - 文生图(只扩写描述)",
    "t2i_layout - 文生图+布局(生成Layout JSON)",
)

# The two modes upstream supports. MOSS-Image has no image-to-image mode at
# all: a reference image is only ever shown to the LLM.
CAPTION_MODE = "t2i"
LAYOUT_MODE = "t2i_layout"

_MODE_KEYS: frozenset[str] = frozenset(
    label.split(" - ", 1)[0].strip() for label in MODES
)

# Which bundled system prompt each mode uses. The upsampler prompt is the
# official artifact shared by all four official templates; the layout prompt
# encodes the official Layout *skill* contract (a JSON draft), which the
# upsampler prompt cannot express -- it explicitly forbids JSON.
_SYSTEM_PROMPT_FILES: dict[str, str] = {
    CAPTION_MODE: "moss_image/upsampler_system_prompt",
    LAYOUT_MODE: "moss_image/layout_system_prompt",
}

# The official Layout draft schema. The compiler rejects unknown fields
# instead of silently dropping creative intent, so the whitelists here are
# exactly the official ones.
_LAYOUT_ROOT_FIELDS: frozenset[str] = frozenset(
    {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
)
_SUMMARY_FIELDS: frozenset[str] = frozenset({"caption", "meta", "canvas", "palette"})
_ITEM_FIELDS: frozenset[str] = frozenset(
    {"label", "caption", "bbox", "text", "tags", "palette"}
)
_ITEM_LABELS: frozenset[str] = frozenset({"object", "character", "text"})
_CANVAS_KINDS: frozenset[str] = frozenset({"single", "multi_panel", "collage"})

# How much a model's box must overlap a user's box to count as "the same
# region". 0.5 is the IoU of two boxes that agree on most of their area; it
# accepts a model that nudged a box by a pixel or two and rejects one that
# moved an element somewhere else entirely.
MIN_BOX_IOU: float = 0.5

# Upstream enables thinking off for the upsampler; external APIs cannot, so a
# reasoning model may still leak a wrapper.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# JSON decoding for a reply that may be wrapped in a code fence or trailed by
# prose. ``raw_decode`` returns the object and the offset it ended at.
_JSON_DECODER = json.JSONDecoder()


def parse_mode_label(label: str) -> str:
    """Strip the `` - 中文`` suffix off a mode dropdown label and return the
    bare upstream key. Unknown / empty input returns ``""`` so callers can
    raise one loud error instead of silently running in the wrong mode.
    """
    code = (label or "").split(" - ", 1)[0].strip()
    return code if code in _MODE_KEYS else ""


def load_system_prompt(mode: str) -> str:
    """Return the bundled system prompt for ``mode``.

    Raises ``ValueError`` on an unknown mode and ``FileNotFoundError`` when
    the bundled file is missing -- both are loud on purpose: a silent
    fallback would enhance against the wrong training distribution.
    """
    logical = _SYSTEM_PROMPT_FILES.get(mode or "")
    if not logical:
        raise ValueError(
            f"MossImage: unknown mode {mode!r}; expected one of "
            f"{sorted(_SYSTEM_PROMPT_FILES)}"
        )
    return load_prompt_text(logical)


def strip_think(raw_text: str) -> str:
    """Remove any ``<think>...</think>`` wrapper and return the visible text."""
    if not raw_text:
        return ""
    return _THINK_BLOCK_RE.sub("", raw_text).strip()


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping ``` or ```json fence around the whole reply.

    Models habitually fence JSON. The fence breaks the "one JSON object"
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


# --------------------------------------------------------------------------- #
# Box spaces
# --------------------------------------------------------------------------- #
def grid_to_pixels(value: float, extent: int, *, upper: bool) -> int:
    """Map one axis of the 0-1000 canvas grid onto real pixels.

    ``CreateBoundingBoxes`` normalizes each axis independently onto a 0-1000
    grid (y by canvas height, x by canvas width), origin at the canvas
    top-left. The official MOSS conversion encloses the marked region in
    whole pixels -- floor on the near edge, ceil on the far edge -- so the
    drawn region is never shaved off.
    """
    pixels = round(float(value) * int(extent) / 1000, 9)
    return math.ceil(pixels) if upper else math.floor(pixels)


def _parse_character_boxes(spec: str) -> set[int]:
    """Parse the ``"1, 3"`` character-box spec into 1-based element numbers.

    Mirrors the official ``MOSSImageLayoutPrompt``: the numbers are the
    positions of the *element* boxes (1-based, in canvas order) that should
    be labelled ``character`` instead of ``object``.
    """
    return {int(n) for n in re.findall(r"\d+", spec or "")}


def elements_to_boxes(
    elements,
    width: int,
    height: int,
    character_boxes: str = "",
) -> list[dict]:
    """Convert ``CreateBoundingBoxes`` elements into MOSS draft-space items.

    The draft space is the one the official Layout skill uses: absolute
    pixels, ``[x_min, y_min, x_max, y_max]``, origin at the canvas top-left.
    That is *not* the space ``CreateBoundingBoxes`` emits (a 0-1000 grid in
    ``[y_min, x_min, y_max, x_max]`` order) and *not* the space the image
    model consumes (``abs_centralized`` / ``yxyx``). Converting here is what
    keeps the model from reading a 0-1000 grid as pixels.

    Returns a list of draft items -- ``{"label", "caption", "bbox"}`` plus
    ``"text"`` for text elements -- ready to be handed to the model or
    compiled directly.

    Raises ``ValueError`` on a missing / malformed box, on a box that
    collapses to zero area, or on a text element with no copy, so a
    mis-wired socket fails before any tokens are spent.
    """
    if elements is None:
        return []
    if not isinstance(elements, (list, tuple)):
        raise ValueError(
            f"MossImage: elements must be a list from Create Bounding Boxes, "
            f"got {type(elements).__name__} / elements 必须是 Create Bounding "
            f"Boxes 的 elements 列表"
        )
    characters = _parse_character_boxes(character_boxes)
    items: list[dict] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        bbox = element.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"MossImage: element {index + 1} has no usable 'bbox' as "
                f"[y_min, x_min, y_max, x_max] on a 0-1000 grid; it came from "
                f"{element!r} / 元素 {index + 1} 的 bbox 缺失或格式不对"
            )
        try:
            y0, x0, y1, x1 = (float(v) for v in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MossImage: element {index + 1} has a non-numeric bbox "
                f"{bbox!r} / 元素 {index + 1} 的 bbox 不是数字"
            ) from exc

        x_min = max(0, min(width, grid_to_pixels(x0, width, upper=False)))
        x_max = max(0, min(width, grid_to_pixels(x1, width, upper=True)))
        y_min = max(0, min(height, grid_to_pixels(y0, height, upper=False)))
        y_max = max(0, min(height, grid_to_pixels(y1, height, upper=True)))
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"MossImage: element {index + 1} collapses to an empty box on a "
                f"{width}x{height} canvas (grid {bbox!r}) / 元素 {index + 1} "
                f"在 {width}x{height} 画布上退化为空框，请重新画一个更大的框"
            )

        is_text = element.get("type") == "text"
        item: dict = {
            "label": "text" if is_text else (
                "character" if (index + 1) in characters else "object"
            ),
            "caption": str(element.get("desc") or "").strip(),
            "bbox": [x_min, y_min, x_max, y_max],
        }
        if is_text:
            copy = str(element.get("text") or "").strip()
            if not copy:
                raise ValueError(
                    f"MossImage: element {index + 1} is a text box with no copy; "
                    f"the Layout schema requires the exact visible characters / "
                    f"元素 {index + 1} 是文本框但没有文字内容，Layout 要求给出"
                    f"准确的可见字符"
                )
            item["text"] = copy
        items.append(item)

    if not items:
        raise ValueError(
            "MossImage: the elements input is empty; draw at least one box in "
            "Create Bounding Boxes, or leave it unconnected and let the model "
            "plan the layout / elements 为空：请在 Create Bounding Boxes 里"
            "至少画一个框，或断开该输入让模型自行规划布局"
        )
    return items


def box_iou(a: list, b: list) -> float:
    """Intersection over union of two ``[x_min, y_min, x_max, y_max]`` boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Layout draft: parse / enforce / compile
# --------------------------------------------------------------------------- #
def parse_layout_reply(raw_text: str) -> dict:
    """Pull the JSON Layout draft out of a model reply.

    Accepts a bare object, a fenced object and an object followed by stray
    prose. Raises ``ValueError`` when no JSON object can be recovered --
    handing the raw text to the compiler would produce nonsense.
    """
    text = _strip_code_fence(strip_think(raw_text))
    if not text:
        raise ValueError(
            "MossImage: the model returned an empty reply in layout mode / "
            "布局模式下模型返回了空回复"
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    start = text.find("{")
    while start != -1:
        try:
            candidate, _ = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            return candidate
        start = text.find("{", start + 1)

    raise ValueError(
        f"MossImage: could not find a JSON Layout draft in the model reply. "
        f"Reply tail: {text[-240:]!r} / 回复里没有可解析的 JSON 布局草稿"
    )


def enforce_supplied_boxes(draft: dict, supplied: list[dict]) -> dict:
    """Rebuild a draft's items from the user's boxes, keeping the model's text.

    The official docs are explicit that the prompt upsampler rewrites the
    layout, and that the boxes must survive verbatim if they are to mean
    anything. So when the user drew boxes, the user's pixels win: each item
    keeps the user's box and takes its caption from the model's matching item
    (matched by overlap), falling back to the box's own description.

    The model's own extra items are dropped -- the layout becomes exactly what
    the user drew. ``summary`` is left alone, so the global description still
    comes from the model.
    """
    if not supplied:
        return draft
    model_items = draft.get("items")
    model_items = model_items if isinstance(model_items, list) else []
    used: set[int] = set()

    items: list[dict] = []
    for box in supplied:
        best_index = -1
        best_score = 0.0
        for position, model_item in enumerate(model_items):
            if position in used or not isinstance(model_item, dict):
                continue
            model_bbox = model_item.get("bbox")
            if not isinstance(model_bbox, (list, tuple)) or len(model_bbox) != 4:
                continue
            try:
                score = box_iou(box["bbox"], [float(v) for v in model_bbox])
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_score = score
                best_index = position

        item: dict = {
            "label": box["label"],
            "caption": "",
            "bbox": list(box["bbox"]),
        }
        if best_index != -1 and best_score >= MIN_BOX_IOU:
            used.add(best_index)
            model_item = model_items[best_index]
            caption = str(model_item.get("caption") or "").strip()
            if caption:
                item["caption"] = caption
            model_text = model_item.get("text")
            if isinstance(model_text, str) and model_text.strip():
                item["text"] = model_text
        if not item["caption"]:
            item["caption"] = box.get("caption") or ""
        if item["label"] == "text" and "text" not in item:
            item["text"] = box.get("text") or ""
        if not item["caption"]:
            raise ValueError(
                f"MossImage: the model dropped the box {box['bbox']} and that "
                f"box has no description of its own, so the layout would ship "
                f"an empty caption / 模型丢弃了该框且该框本身没有描述，"
                f"布局会带有空caption，请给框写上描述或改用让模型自行规划布局"
            )
        items.append(item)

    result = dict(draft)
    result["items"] = items
    return result


def compile_layout_draft(draft: dict) -> str:
    """Compile a Layout draft into the one-line prompt MOSS-Image consumes.

    A faithful port of the official ``compile_prompt.py``:
    each ``[x1, y1, x2, y2]`` box is enclosed in whole pixels (floor on the
    near edge, ceil on the far edge), then the canvas centre is subtracted
    and the box is written in ``yxyx`` order. The result is a single compact
    JSON line with ``bbox_coord_space="abs_centralized"`` and
    ``bbox_format="yxyx"`` and no canvas metadata -- the canvas size is
    passed to the image workflow separately.

    Unknown fields are rejected rather than silently dropped, and an
    out-of-bounds box raises instead of being clipped, exactly as upstream
    does.
    """
    require(isinstance(draft, dict), "Layout draft must be a JSON object")
    _check_fields(
        draft, _LAYOUT_ROOT_FIELDS, _LAYOUT_ROOT_FIELDS, "Layout draft"
    )

    width, height = draft["width"], draft["height"]
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or value <= 0 or value % 16 != 0:
            raise ValueError(
                f"MossImage: {name} must be a positive integer multiple of 16, "
                f"got {value!r} / {name} 必须是 16 的正整数倍"
            )
    if draft["bbox_coord_space"] != "abs":
        raise ValueError(
            "MossImage: bbox_coord_space must be 'abs' / bbox_coord_space 必须为 abs"
        )
    if draft["bbox_format"] != "xyxy":
        raise ValueError(
            "MossImage: bbox_format must be 'xyxy' / bbox_format 必须为 xyxy"
        )

    summary = draft["summary"]
    _check_fields(summary, _SUMMARY_FIELDS, {"caption"}, "summary")
    _require_text(summary["caption"], "summary.caption")
    if "canvas" in summary and summary["canvas"] not in _CANVAS_KINDS:
        raise ValueError(
            "MossImage: summary.canvas must be single, multi_panel or collage / "
            "summary.canvas 只能是 single / multi_panel / collage"
        )
    for key in ("meta", "palette"):
        if key in summary:
            _require_strings(summary[key], f"summary.{key}")

    items = draft["items"]
    if not isinstance(items, list) or not items:
        raise ValueError(
            "MossImage: items must be a nonempty flat array / items 必须是非空数组"
        )
    compiled_items: list[dict] = []
    for index, item in enumerate(items):
        location = f"items[{index}]"
        _check_fields(item, _ITEM_FIELDS, {"label", "caption", "bbox"}, location)
        if item["label"] not in _ITEM_LABELS:
            raise ValueError(
                f"MossImage: {location}.label must be object, character or text / "
                f"{location}.label 只能是 object / character / text"
            )
        _require_text(item["caption"], f"{location}.caption")
        if item["label"] == "text" and "text" not in item:
            raise ValueError(
                f"MossImage: {location}: text items require literal text / "
                f"{location} 是文本项，必须带 text 字段"
            )
        if "text" in item:
            _require_text(item["text"], f"{location}.text")
        for key in ("tags", "palette"):
            if key in item:
                _require_strings(item[key], f"{location}.{key}")
        compiled_items.append(
            {
                key: (
                    _centered_box(value, width, height, f"{location}.bbox")
                    if key == "bbox"
                    else value
                )
                for key, value in item.items()
            }
        )

    result = {
        "summary": summary,
        "items": compiled_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(
        result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _centered_box(box, width: int, height: int, location: str) -> list[int]:
    """Turn a draft-space box into the centered ``yxyx`` box the model reads."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(
            f"MossImage: {location} must be [x1, y1, x2, y2] / {location} 必须是 "
            f"[x1, y1, x2, y2]"
        )
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in box
    ):
        raise ValueError(
            f"MossImage: {location} must contain four finite numbers / "
            f"{location} 必须包含四个有限数字"
        )
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        raise ValueError(
            f"MossImage: {location} must have positive width and height / "
            f"{location} 的宽高必须为正"
        )
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"MossImage: {location} exceeds the {width}x{height} canvas / "
            f"{location} 超出了 {width}x{height} 画布"
        )
    # Enclose in pixel edges before centering, as the official compiler does.
    result = [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise ValueError(
            f"MossImage: {location} is too small at export precision / "
            f"{location} 在导出精度下太小了"
        )
    return result


def _check_fields(value, allowed, required, location: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"MossImage: {location} must be an object")
    unknown = set(value.keys()) - set(allowed)
    missing = set(required) - set(value.keys())
    if unknown:
        raise ValueError(
            f"MossImage: {location}: unknown fields {sorted(unknown)} / "
            f"{location} 含有未知字段 {sorted(unknown)}"
        )
    if missing:
        raise ValueError(
            f"MossImage: {location}: missing fields {sorted(missing)} / "
            f"{location} 缺少字段 {sorted(missing)}"
        )


def _require_text(value, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"MossImage: {location} must be nonempty text / {location} 必须是非空文本"
        )


def _require_strings(value, location: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"MossImage: {location} must be an array")
    for index, item in enumerate(value):
        _require_text(item, f"{location}[{index}]")


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when ``condition`` is false."""
    if not condition:
        raise ValueError(message)


# --------------------------------------------------------------------------- #
# User message
# --------------------------------------------------------------------------- #
def build_user_message(
    mode: str,
    prompt: str,
    *,
    width: int = 0,
    height: int = 0,
    boxes: list[dict] | None = None,
) -> str:
    """Assemble the user-side text for one MOSS enhancement call.

    The caption mode sends the request as it is: the official upsampler
    prompt already says how to treat the user's copy, and the upstream
    templates feed the request straight in.

    The layout mode adds the canvas -- the model has to know the pixel size
    to place anything -- and, when the user drew boxes, the boxes
    themselves in the official draft space (absolute pixels, ``xyxy``,
    top-left origin). Sending them in that space matters: the official
    prompt says to use the given values and never convert them, so a 0-1000
    grid handed over raw would be read as pixels.
    """
    request = (prompt or "").strip()
    if not request:
        raise ValueError(
            "MossImage: prompt is empty; there is nothing to enhance / "
            "prompt 为空，无内容可增强"
        )
    if mode == CAPTION_MODE:
        return request

    lines = [request, ""]
    lines.append(
        f"Canvas: {int(width)} x {int(height)} pixels (width x height). "
        f"The canvas origin is the top-left corner and both sides are "
        f"multiples of 16."
    )
    if boxes:
        lines.append("")
        lines.append(
            "Placement boxes (absolute pixels, xyxy, origin at the top-left "
            "corner). Use every box exactly as given, in this order:"
        )
        for index, box in enumerate(boxes, start=1):
            copy = box.get("text")
            suffix = f" | text: {copy}" if copy else ""
            lines.append(
                f"{index}. {box['label']} | [{box['bbox'][0]}, {box['bbox'][1]}, "
                f"{box['bbox'][2]}, {box['bbox'][3]}] | "
                f"{box.get('caption') or ''}{suffix}"
            )
    return "\n".join(lines)


__all__ = [
    "MODES",
    "CAPTION_MODE",
    "LAYOUT_MODE",
    "MIN_BOX_IOU",
    "parse_mode_label",
    "load_system_prompt",
    "strip_think",
    "grid_to_pixels",
    "elements_to_boxes",
    "box_iou",
    "parse_layout_reply",
    "enforce_supplied_boxes",
    "compile_layout_draft",
    "build_user_message",
]
