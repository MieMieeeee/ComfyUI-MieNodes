# -*- coding: utf-8 -*-
"""Tests for the MOSS-Image prompt-enhancement ComfyUI node.

The node reproduces OpenMOSS' official prompt expander over HTTP. The
official upsampler system prompt is an upstream artifact that must never be
rewritten, so the first tests guard its verbatim fidelity: character count
and content digest, checked against the snapshot published in the upstream
research doc.

Covered here:

* Fidelity: the bundled official prompt is 8398 characters with the
  documented sha1 prefix, and the layout prompt carries no ``{placeholder}``
  tokens (so it stays usable as a generic system prompt).
* Coordinate spaces: the 0-1000 ``[y_min, x_min, y_max, x_max]`` canvas grid
  becomes absolute-pixel ``xyxy`` for the model, and the compiler produces
  the centred ``yxyx`` the official script produces.
* The compiler: the official worked example, the field whitelists, and the
  loud rejections (unknown field, non-multiple-of-16 canvas, out-of-bounds
  box, text item without copy).
* Reply handling: a bare JSON draft, a fenced one, one followed by prose,
  and the retry when no draft can be found.
* Verbatim boxes: the user's pixels survive a model that rewrote the layout.
* Loud failures: empty prompt, bad canvas, empty reply, unparseable reply.
* Surface: widget shape, RETURN_NAMES, CATEGORY, plugin registration,
  ``is_changed`` stability and content-sensitivity.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
LLM_DIR = PROJECT_DIR / "nodes" / "llm"
MOSS_DIR = PROMPTS_DIR / "moss_image"

sys.path.insert(0, str(PROJECT_DIR / "tests"))

# Upstream snapshot figures (official Comfy-Org template, 2026-10-25).
OFFICIAL_PROMPT_CHARS = 8398
OFFICIAL_PROMPT_SHA1_PREFIX = "ab424ab68233"
OFFICIAL_PROMPT_SHA256 = (
    "87d45638f0c17ad25d38c24a63c7aa7a71316388eb40065c09850e8801b55f85"
)


# --------------------------------------------------------------------------- #
# Module bootstrap (the package is importable both as ``nodes.*`` and as the
# ``_mienodes_internal`` alias the plugin registers at runtime)
# --------------------------------------------------------------------------- #
def _ensure_pkg(fqn: str, path: Path | None = None):
    if fqn in sys.modules:
        return sys.modules[fqn]
    mod = types.ModuleType(fqn)
    if path is not None:
        mod.__path__ = [str(path)]
    mod.__package__ = fqn
    sys.modules[fqn] = mod
    return mod


def _load_file(fqn: str, path: Path):
    if fqn in sys.modules:
        del sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"
    )


@pytest.fixture(scope="module")
def helpers():
    _bootstrap()
    return _load_file(
        "_mienodes_internal.nodes.llm.moss_image_prompts",
        LLM_DIR / "moss_image_prompts.py",
    )


@pytest.fixture(scope="module")
def mod(helpers):
    # Loaded last so its relative import of the helpers resolves to the
    # shimmed module above.
    return _load_file(
        "_mienodes_internal.nodes.llm.moss_image_prompt_generator",
        LLM_DIR / "moss_image_prompt_generator.py",
    )


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeConnector:
    """Minimal LLMServiceConnector stand-in recording every call."""

    model = "fake-model"

    def __init__(self, replies=None):
        self.timeout = 120
        self.calls: list[dict] = []
        self._replies = list(replies or ["a caption"])

    def get_state(self) -> str:
        return "fake-state"

    def invoke(self, messages, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "seed": kwargs.get("seed"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "timeout": self.timeout,
            }
        )
        return self._replies.pop(0) if self._replies else ""


class _FakeFlat:
    """The tail of ``detach().cpu().reshape(-1)`` that ``is_changed`` consumes."""

    def __init__(self, arr):
        self._a = arr

    def numpy(self):
        return self._a

    def tobytes(self):
        return self._a.tobytes()


class _FakeImageBatch:
    """Stands in for a ComfyUI IMAGE batch so the real encoder path runs."""

    def __init__(self, n: int = 1, fill: float = 0.0):
        self._t = np.full((n, 8, 8, 3), fill, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        return _FakeFlat(self._t.reshape(*shape))

    def __getattr__(self, item):
        return getattr(self._t, item)

    def __getitem__(self, item):
        return self._t[item]

    @property
    def shape(self):
        return self._t.shape


def _element(bbox, desc="", etype="obj", text=""):
    """One ``CreateBoundingBoxes`` element in the official 0-1000 grid."""
    el = {"type": etype, "bbox": list(bbox), "desc": desc}
    if etype == "text":
        el["text"] = text
    return el


def _draft(width=1536, height=1024, items=None, caption="A warm studio campaign."):
    return {
        "width": width,
        "height": height,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": {"caption": caption},
        "items": items
        if items is not None
        else [
            {
                "label": "object",
                "caption": "An ivory ceramic cup.",
                "bbox": [176, 224, 800, 864],
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Fidelity of the bundled official prompt
# --------------------------------------------------------------------------- #
def test_official_upsampler_prompt_is_verbatim():
    path = MOSS_DIR / "upsampler_system_prompt.txt"
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert len(text) == OFFICIAL_PROMPT_CHARS
    assert hashlib.sha1(raw).hexdigest().startswith(OFFICIAL_PROMPT_SHA1_PREFIX)
    assert hashlib.sha256(raw).hexdigest() == OFFICIAL_PROMPT_SHA256
    # No CRLF and no trailing newline: the official file has none, and the
    # digest above only matches if the bytes are exactly as published.
    assert b"\r" not in raw
    assert not raw.endswith(b"\n")
    assert text.startswith("You are a prompt engineer for an image generation model.")


def test_layout_prompt_has_no_placeholder_tokens():
    """A ``{placeholder}`` would make the loader treat the file as a template."""
    import re

    pattern = re.compile(r"\{[a-zA-Z_]*\}")
    for name in ("upsampler_system_prompt.txt", "layout_system_prompt.txt"):
        text = (MOSS_DIR / name).read_text(encoding="utf-8")
        assert not pattern.search(text), name


def test_both_modes_load_a_system_prompt(helpers):
    for mode in helpers.MODES:
        code = helpers.parse_mode_label(mode)
        assert code
        assert helpers.load_system_prompt(code).strip()


def test_unknown_mode_is_loud(helpers):
    assert helpers.parse_mode_label("nope - x") == ""
    assert helpers.parse_mode_label("") == ""
    with pytest.raises(ValueError):
        helpers.load_system_prompt("i2i")


# --------------------------------------------------------------------------- #
# Coordinate spaces
# --------------------------------------------------------------------------- #
def test_grid_to_pixels_encloses_the_marked_region(helpers):
    # Near edge floors, far edge ceils: the drawn region is never shaved.
    assert helpers.grid_to_pixels(0, 1536, upper=False) == 0
    assert helpers.grid_to_pixels(500, 1536, upper=True) == 768
    assert helpers.grid_to_pixels(500, 1024, upper=True) == 512
    assert helpers.grid_to_pixels(1000, 1536, upper=True) == 1536


def test_elements_to_boxes_converts_grid_to_draft_space(helpers):
    elements = [_element([0, 0, 500, 500], desc="a cup")]
    boxes = helpers.elements_to_boxes(elements, 1536, 1024)
    assert len(boxes) == 1
    assert boxes[0]["bbox"] == [0, 0, 768, 512]
    assert boxes[0]["label"] == "object"
    assert boxes[0]["caption"] == "a cup"
    assert "text" not in boxes[0]


def test_elements_to_boxes_labels_text_and_characters(helpers):
    elements = [
        _element([0, 0, 100, 100], desc="背景", etype="text", text="把日常，过成诗。"),
        _element([0, 0, 100, 100], desc="a person"),
    ]
    boxes = helpers.elements_to_boxes(elements, 1536, 1024, character_boxes="2")
    assert boxes[0]["label"] == "text"
    assert boxes[0]["text"] == "把日常，过成诗。"
    assert boxes[1]["label"] == "character"


def test_elements_to_boxes_rejects_bad_input(helpers):
    with pytest.raises(ValueError):
        helpers.elements_to_boxes([], 1024, 1024)
    with pytest.raises(ValueError):
        helpers.elements_to_boxes([{"type": "obj", "desc": "x"}], 1024, 1024)
    with pytest.raises(ValueError):
        helpers.elements_to_boxes([{"type": "obj", "bbox": [0, 0, 0, 0]}], 1024, 1024)
    with pytest.raises(ValueError):
        helpers.elements_to_boxes(
            [{"type": "text", "bbox": [0, 0, 100, 100], "text": ""}], 1024, 1024
        )


# --------------------------------------------------------------------------- #
# The compiler (port of the official compile_prompt.py)
# --------------------------------------------------------------------------- #
def test_compiler_matches_the_official_worked_example(helpers):
    compiled = helpers.compile_layout_draft(_draft())
    parsed = json.loads(compiled)
    assert parsed["items"][0]["bbox"] == [-288, -592, 352, 32]
    assert parsed["bbox_coord_space"] == "abs_centralized"
    assert parsed["bbox_format"] == "yxyx"
    # One physical line, compact separators, no canvas metadata.
    assert "\n" not in compiled
    assert ", " not in compiled
    assert "width" not in parsed
    assert "height" not in parsed


def test_compiler_keeps_text_and_palette_fields(helpers):
    draft = _draft(
        items=[
            {
                "label": "text",
                "caption": "Large dark-green Chinese title.",
                "text": "把日常，过成诗。",
                "bbox": [928, 272, 1408, 448],
                "palette": ["dark green"],
            }
        ]
    )
    parsed = json.loads(helpers.compile_layout_draft(draft))
    item = parsed["items"][0]
    assert item["text"] == "把日常，过成诗。"
    assert item["palette"] == ["dark green"]
    assert "把日常" in helpers.compile_layout_draft(draft)  # not ASCII-escaped


def test_compiler_rejects_bad_drafts(helpers):
    with pytest.raises(ValueError):
        helpers.compile_layout_draft({"caption": "not a layout"})
    with pytest.raises(ValueError):
        helpers.compile_layout_draft(_draft(width=1000))
    with pytest.raises(ValueError):
        helpers.compile_layout_draft(_draft(items=[]))
    with pytest.raises(ValueError):
        helpers.compile_layout_draft(_draft(items=[{"label": "object"}]))
    # A text item must carry the exact characters.
    with pytest.raises(ValueError):
        helpers.compile_layout_draft(
            _draft(items=[{"label": "text", "caption": "title", "bbox": [0, 0, 10, 10]}])
        )
    # Unknown fields are rejected, not silently dropped.
    bad = _draft()
    bad["children"] = []
    with pytest.raises(ValueError):
        helpers.compile_layout_draft(bad)
    # Out of bounds.
    with pytest.raises(ValueError):
        helpers.compile_layout_draft(
            _draft(items=[{"label": "object", "caption": "x", "bbox": [0, 0, 2000, 2000]}])
        )


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #
def test_parse_layout_reply_accepts_the_usual_shapes(helpers):
    draft = _draft()
    raw = json.dumps(draft, ensure_ascii=False)
    assert helpers.parse_layout_reply(raw)["width"] == 1536
    assert helpers.parse_layout_reply(f"```json\n{raw}\n```")["width"] == 1536
    assert helpers.parse_layout_reply(f"Here you go:\n{raw}\nDone.")["width"] == 1536


def test_parse_layout_reply_is_loud_without_json(helpers):
    with pytest.raises(ValueError):
        helpers.parse_layout_reply("I could not do that.")
    with pytest.raises(ValueError):
        helpers.parse_layout_reply("")


# --------------------------------------------------------------------------- #
# Verbatim boxes
# --------------------------------------------------------------------------- #
def test_enforce_supplied_boxes_keeps_the_users_pixels(helpers):
    supplied = [
        {"label": "object", "caption": "cup", "bbox": [100, 100, 400, 400]},
        {"label": "text", "caption": "title", "text": "标题", "bbox": [1000, 100, 1400, 300]},
    ]
    draft = _draft(
        items=[
            # The model moved the cup and invented an extra item.
            {"label": "object", "caption": "A blue ceramic cup.", "bbox": [120, 120, 420, 420]},
            {"label": "object", "caption": "A background wall.", "bbox": [0, 0, 1536, 1024]},
        ]
    )
    result = helpers.enforce_supplied_boxes(draft, supplied)
    assert len(result["items"]) == 2
    assert result["items"][0]["bbox"] == [100, 100, 400, 400]
    assert result["items"][0]["caption"] == "A blue ceramic cup."
    assert result["items"][1]["bbox"] == [1000, 100, 1400, 300]
    assert result["items"][1]["text"] == "标题"
    # The model's own extra item is gone: the layout is what the user drew.
    assert all("wall" not in i["caption"] for i in result["items"])


def test_enforce_supplied_boxes_falls_back_to_the_elements_own_text(helpers):
    supplied = [
        {"label": "object", "caption": "a plain cup", "bbox": [100, 100, 400, 400]},
    ]
    draft = _draft(items=[])
    result = helpers.enforce_supplied_boxes(draft, supplied)
    assert result["items"][0]["caption"] == "a plain cup"


def test_enforce_supplied_boxes_is_loud_when_nothing_describes_the_box(helpers):
    supplied = [{"label": "object", "caption": "", "bbox": [100, 100, 400, 400]}]
    with pytest.raises(ValueError):
        helpers.enforce_supplied_boxes(_draft(items=[]), supplied)


# --------------------------------------------------------------------------- #
# User message
# --------------------------------------------------------------------------- #
def test_caption_user_message_is_the_request(helpers):
    assert helpers.build_user_message("t2i", "一只猫") == "一只猫"


def test_layout_user_message_carries_the_canvas_and_the_boxes(helpers):
    boxes = [
        {"label": "object", "caption": "a cup", "bbox": [176, 224, 800, 864]},
        {"label": "text", "caption": "title", "text": "把日常，过成诗。", "bbox": [928, 272, 1408, 448]},
    ]
    text = helpers.build_user_message(
        "t2i_layout", "海报", width=1536, height=1024, boxes=boxes
    )
    assert "Canvas: 1536 x 1024 pixels" in text
    assert "[176, 224, 800, 864]" in text
    assert "把日常，过成诗。" in text
    assert text.startswith("海报")


def test_empty_prompt_is_loud(helpers):
    with pytest.raises(ValueError):
        helpers.build_user_message("t2i", "   ")


# --------------------------------------------------------------------------- #
# Node: caption mode
# --------------------------------------------------------------------------- #
def test_caption_mode_returns_the_caption(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(["Anime-style digital illustration of a cat."])
    prompt, caption, layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        prompt="一只猫",
        canvas_width=1024,
        canvas_height=1024,
    )
    assert prompt == "Anime-style digital illustration of a cat."
    assert caption == prompt
    assert layout == ""
    call = connector.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"].startswith(
        "You are a prompt engineer for an image generation model."
    )
    assert call["messages"][1] == {"role": "user", "content": "一只猫"}
    assert call["max_tokens"] == mod._MAX_TOKENS_DEFAULT


def test_empty_reply_is_loud(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(["   "])
    with pytest.raises(RuntimeError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[0],
            prompt="一只猫",
            canvas_width=1024,
            canvas_height=1024,
        )


# --------------------------------------------------------------------------- #
# Node: layout mode
# --------------------------------------------------------------------------- #
def test_layout_mode_compiles_the_draft(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector([json.dumps(_draft(), ensure_ascii=False)])
    prompt, caption, layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1536,
        canvas_height=1024,
    )
    parsed = json.loads(prompt)
    assert parsed["bbox_format"] == "yxyx"
    assert parsed["items"][0]["bbox"] == [-288, -592, 352, 32]
    assert caption == "A warm studio campaign."
    # The readable draft is kept for editing.
    assert json.loads(layout)["items"][0]["bbox"] == [176, 224, 800, 864]


def test_layout_mode_honours_the_requested_canvas(mod):
    """A model that picks its own canvas would misplace every box."""
    node = mod.MossImagePromptUpsampler()
    draft = _draft(width=1024, height=1024)
    connector = _FakeConnector([json.dumps(draft, ensure_ascii=False)])
    prompt, _caption, _layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1536,
        canvas_height=1024,
    )
    assert json.loads(prompt)["items"][0]["bbox"] == [-288, -592, 352, 32]


def test_layout_mode_keeps_the_users_boxes(mod):
    node = mod.MossImagePromptUpsampler()
    elements = [_element([0, 0, 260, 500], desc="a cup")]
    draft = _draft(
        items=[
            {
                "label": "object",
                "caption": "A cobalt blue ceramic cup.",
                "bbox": [10, 10, 700, 250],
            }
        ]
    )
    connector = _FakeConnector([json.dumps(draft, ensure_ascii=False)])
    prompt, _caption, _layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1536,
        canvas_height=1024,
        elements=elements,
    )
    parsed = json.loads(prompt)
    # The model nudged the cup; the user's pixels win.
    assert parsed["items"][0]["bbox"] == [-512, -768, -245, 0]
    assert parsed["items"][0]["caption"] == "A cobalt blue ceramic cup."


def test_layout_mode_can_let_the_model_plan(mod):
    node = mod.MossImagePromptUpsampler()
    elements = [_element([0, 0, 260, 500], desc="a cup")]
    draft = _draft(
        items=[
            {
                "label": "object",
                "caption": "A cobalt blue ceramic cup.",
                "bbox": [400, 400, 900, 900],
            }
        ]
    )
    connector = _FakeConnector([json.dumps(draft, ensure_ascii=False)])
    prompt, _caption, _layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1536,
        canvas_height=1024,
        elements=elements,
        keep_layout_verbatim=False,
    )
    assert json.loads(prompt)["items"][0]["bbox"] == [-112, -368, 388, 132]


def test_layout_mode_retries_once_then_fails_loudly(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(["not json", "still not json"])
    with pytest.raises(ValueError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[1],
            prompt="海报",
            canvas_width=1024,
            canvas_height=1024,
        )
    assert len(connector.calls) == 2


def test_layout_mode_retries_an_invalid_draft(mod):
    """A draft that does not compile is worth one more try, not a dead run."""
    node = mod.MossImagePromptUpsampler()
    bad = _draft(
        width=1024,
        height=1024,
        items=[{"label": "object", "caption": "x", "bbox": [0, 0, 4000, 4000]}],
    )
    good = _draft(width=1024, height=1024)
    connector = _FakeConnector([json.dumps(bad, ensure_ascii=False),
                                json.dumps(good, ensure_ascii=False)])
    prompt, _caption, _layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1024,
        canvas_height=1024,
    )
    assert json.loads(prompt)["bbox_format"] == "yxyx"
    assert len(connector.calls) == 2
    # The retry tells the model what went wrong.
    correction = connector.calls[1]["messages"][-1]["content"]
    assert "canvas" in correction


def test_layout_mode_retries_and_recovers(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(
        ["sorry", json.dumps(_draft(width=1024, height=1024), ensure_ascii=False)]
    )
    prompt, _caption, _layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1024,
        canvas_height=1024,
    )
    assert json.loads(prompt)["bbox_format"] == "yxyx"
    assert len(connector.calls) == 2


def test_bad_canvas_is_loud(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(["x"])
    with pytest.raises(ValueError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[1],
            prompt="海报",
            canvas_width=1000,
            canvas_height=1024,
        )
    with pytest.raises(ValueError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[1],
            prompt="海报",
            canvas_width=1024,
            canvas_height=0,
        )


# --------------------------------------------------------------------------- #
# Node: skipping the expansion
# --------------------------------------------------------------------------- #
def test_skip_enhance_in_caption_mode_never_calls_the_llm(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(["should not be used"])
    prompt, caption, layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        prompt="一只猫",
        canvas_width=1024,
        canvas_height=1024,
        skip_enhance=True,
    )
    assert prompt == "一只猫"
    assert caption == "一只猫"
    assert layout == ""
    assert connector.calls == []


def test_skip_enhance_in_layout_mode_compiles_the_boxes(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector(["should not be used"])
    elements = [_element([0, 0, 500, 500], desc="a cup")]
    prompt, caption, layout = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        prompt="海报",
        canvas_width=1536,
        canvas_height=1024,
        elements=elements,
        skip_enhance=True,
    )
    parsed = json.loads(prompt)
    assert parsed["bbox_format"] == "yxyx"
    assert parsed["items"][0]["bbox"] == [-512, -768, 0, 0]
    assert caption == "海报"
    assert json.loads(layout)["items"][0]["caption"] == "a cup"
    assert connector.calls == []


def test_skip_enhance_in_layout_mode_needs_boxes(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector([])
    with pytest.raises(ValueError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[1],
            prompt="海报",
            canvas_width=1024,
            canvas_height=1024,
            skip_enhance=True,
        )


# --------------------------------------------------------------------------- #
# Surface
# --------------------------------------------------------------------------- #
def test_widget_shape_and_returns(mod):
    inputs = mod.MossImagePromptUpsampler.INPUT_TYPES()
    required = set(inputs["required"])
    assert required == {
        "llm_service_connector",
        "mode",
        "prompt",
        "canvas_width",
        "canvas_height",
        "seed",
    }
    optional = set(inputs["optional"])
    assert {
        "ref_image",
        "elements",
        "character_boxes",
        "keep_layout_verbatim",
        "skip_enhance",
    } <= optional
    assert mod.MossImagePromptUpsampler.RETURN_NAMES == (
        "upsampled_prompt",
        "caption",
        "layout_json",
    )
    assert mod.MossImagePromptUpsampler.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert mod.MossImagePromptUpsampler.FUNCTION == "enhance"
    assert "Prompt Generator" in mod.MossImagePromptUpsampler.CATEGORY


def test_plugin_registration():
    from test_plugin_imports import load_plugin_module

    plugin = load_plugin_module()
    registered = [
        key
        for key in plugin.NODE_CLASS_MAPPINGS
        if key.split("|")[0] == "MossImagePromptUpsampler"
    ]
    assert registered, "the node is not registered in NODE_CLASS_MAPPINGS"
    assert any(
        name.split("|")[0] == "MossImagePromptUpsampler"
        for name in plugin.NODE_DISPLAY_NAME_MAPPINGS
    )


def test_is_changed_is_stable_and_content_sensitive(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector()
    kwargs = dict(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        prompt="一只猫",
        canvas_width=1024,
        canvas_height=1024,
        seed=7,
    )
    first = node.is_changed(**kwargs)
    assert first == node.is_changed(**kwargs)
    assert first != node.is_changed(**{**kwargs, "prompt": "一只狗"})
    assert first != node.is_changed(**{**kwargs, "mode": mod.MODES[1]})
    assert first != node.is_changed(**{**kwargs, "seed": 8})
    # A canvas edit changes the boxes without touching any widget.
    elements = [_element([0, 0, 500, 500], desc="a cup")]
    with_elements = node.is_changed(**kwargs, elements=elements)
    other = [_element([0, 0, 500, 500], desc="a teapot")]
    assert with_elements != node.is_changed(**kwargs, elements=other)


def test_is_changed_survives_a_bad_element_array(mod):
    node = mod.MossImagePromptUpsampler()
    connector = _FakeConnector()
    value = node.is_changed(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        prompt="x",
        canvas_width=1024,
        canvas_height=1024,
        seed=0,
        elements=[{"type": "obj"}],
    )
    assert isinstance(value, str) and value
