# -*- coding: utf-8 -*-
"""Tests for the FLUX.3 Klein prompt-upsampler ComfyUI node.

The node reproduces Black Forest Labs' official Prompt Upsampling by sending a
bundled official system prompt (``nodes/llm/prompts/flux3_klein/*.json``) to the
configured LLM. Because those prompts are upstream artifacts that must never be
rewritten, the bulk of this module guards verbatim fidelity: key order,
character counts, and a content hash per prompt, checked against the snapshot
counts published upstream.

Covered here:

* Fidelity: both packs carry the four expected keys in order, at the exact
  upstream character counts, and the bundled files hash to the snapshot digests
  (guards against an accidental edit of an official prompt).
* Placeholder hygiene: the ``i2i_bbox`` prompt contains ``{"id", "from", …}``
  row-key prose that must not be mistaken for a ``{placeholder}`` token.
* User scaffold: per-mode framing lines, canvas placement, element-table
  passthrough, and the verbatim closing instruction of each mode.
* Reply splitting: the ``*_bbox`` "caption paragraph + one JSON array" contract,
  including descs containing brackets and nested arrays.
* Loud failures: empty prompt, editing mode without images, malformed
  ``bboxes_json`` (rejected *before* any LLM call), empty LLM reply, bbox mode
  whose reply has no array.
* Surface: widget shape, RETURN_NAMES, CATEGORY, plugin registration,
  ``is_changed`` stability and content-sensitivity.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
LLM_DIR = PROJECT_DIR / "nodes" / "llm"
FLUX3_DIR = PROMPTS_DIR / "flux3_klein"

sys.path.insert(0, str(PROJECT_DIR / "tests"))

# Upstream self-check figures (short pack documented in the PUP model card,
# long pack measured from inference repo PR #5). Snapshot date 2026-10-19.
EXPECTED_LENGTHS: dict[str, dict[str, int]] = {
    "short": {"t2i": 3404, "i2i": 2677, "t2i_bbox": 3302, "i2i_bbox": 3034},
    "long": {"t2i": 46615, "i2i": 6194, "t2i_bbox": 27258, "i2i_bbox": 21614},
}

# sha256 of each bundled file as vendored from upstream. If an official prompt
# is ever edited, this fails loudly instead of silently upsampling against
# modified training-distribution text.
EXPECTED_FILE_DIGESTS: dict[str, str] = {
    "system_prompts.json":
        "b5da3ed8bc04ab59dbdb747930f7dbff847e868903fed8028f85ee1a4ac82fb4",
    "pup_prompts_external_vlm.json":
        "361bb6e081d50e54a3dee9abf26327436a9386c83465f881c7848989232028be",
}


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
        "_mienodes_internal.nodes.llm.flux3_klein_prompts",
        LLM_DIR / "flux3_klein_prompts.py",
    )


@pytest.fixture(scope="module")
def mod(helpers):
    # Loaded last so its relative import of the helpers resolves to the
    # shimmed module above.
    return _load_file(
        "_mienodes_internal.nodes.llm.flux3_klein_prompt_generator",
        LLM_DIR / "flux3_klein_prompt_generator.py",
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

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
    """Stands in for a ComfyUI IMAGE batch so the real encoder path runs.

    Exposes just enough of the torch surface (``ndim`` / ``shape`` /
    ``detach`` / ``cpu`` / ``reshape`` / ``numpy``) that both the JPEG encoder
    and the ``is_changed`` pixel sampler execute for real instead of bailing
    out to a shape-only fingerprint.
    """

    def __init__(self, n: int, fill: float = 0.0):
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

    @property
    def ndim(self):
        return self._t.ndim

    @property
    def dtype(self):
        return self._t.dtype


# --------------------------------------------------------------------------- #
# 1. Verbatim fidelity of the bundled official prompts
# --------------------------------------------------------------------------- #
def test_both_packs_carry_four_keys_in_upstream_order():
    """Dropdown order must match upstream's own key order."""
    expected = ["t2i", "i2i", "t2i_bbox", "i2i_bbox"]
    for name in ("system_prompts.json", "pup_prompts_external_vlm.json"):
        data = json.loads((FLUX3_DIR / name).read_text(encoding="utf-8"))
        assert list(data) == expected, f"{name} key order drifted from upstream"


@pytest.mark.parametrize("pack", ["short", "long"])
def test_every_prompt_matches_upstream_character_count(helpers, pack):
    for mode, expected in EXPECTED_LENGTHS[pack].items():
        text = helpers.load_system_prompt(mode, pack)
        assert len(text) == expected, (
            f"{pack}/{mode} is {len(text)} chars, upstream snapshot says "
            f"{expected} — an official prompt was edited"
        )


def test_bundled_files_are_byte_identical_to_snapshot(helpers):
    """Guard against edits to the official text itself.

    The digests are of the files as vendored from upstream. Character counts
    alone would not catch, say, a swapped adjective of identical length.
    """
    for name in EXPECTED_FILE_DIGESTS:
        path = FLUX3_DIR / name
        assert path.is_file(), f"bundled prompt file missing: {name}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == EXPECTED_FILE_DIGESTS[name], (
            f"{name} no longer matches the upstream snapshot copy — official "
            f"prompt text must never be edited"
        )


def test_prompts_are_not_rewritten_with_placeholders(helpers):
    """The row-key prose ``{"id", "from", …}`` is not a format placeholder.

    ``loader.list_usable_builtin_prompts`` drops any prompt containing a
    ``{word}`` token; the official bbox prompts quote their row keys as prose
    and must stay intact.
    """
    placeholders = re.compile(r"\{[a-zA-Z_]*\}")
    for pack in ("short", "long"):
        for mode in EXPECTED_LENGTHS[pack]:
            text = helpers.load_system_prompt(mode, pack)
            assert not placeholders.search(text), (
                f"{pack}/{mode} contains a {{placeholder}} token — official "
                f"prompts must not carry format placeholders"
            )
    assert '{"id", "from", "src_bbox", "tgt_bbox", "desc"}' in helpers.load_system_prompt(
        "i2i_bbox", "short"
    )


def test_packs_differ_materially(helpers):
    """The two packs are separate official publications, not one text twice."""
    for mode in ("t2i", "i2i", "t2i_bbox", "i2i_bbox"):
        assert helpers.load_system_prompt(mode, "short") != helpers.load_system_prompt(
            mode, "long"
        )


def test_unknown_mode_or_pack_raises_loudly(helpers):
    with pytest.raises(ValueError):
        helpers.load_system_prompt("t2i", "medium")
    with pytest.raises(ValueError):
        helpers.load_system_prompt("t2v", "short")


# --------------------------------------------------------------------------- #
# 2. Label parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,code", [
    ("t2i - 文生图(默认)", "t2i"),
    ("i2i - 图生图编辑(需参考图)", "i2i"),
    ("t2i_bbox - 文生图+布局框", "t2i_bbox"),
    ("i2i_bbox - 编辑+布局框(需参考图)", "i2i_bbox"),
])
def test_parse_mode_label(helpers, label, code):
    assert helpers.parse_mode_label(label) == code


def test_parse_mode_label_rejects_junk(helpers):
    for junk in ("", "nope - x", None, "T2I - 文生图"):
        assert helpers.parse_mode_label(junk) == ""


def test_parse_pack_label(helpers):
    for label in helpers.PROMPT_PACKS:
        assert helpers.parse_pack_label(label) in ("short", "long")
    assert helpers.parse_pack_label("") == ""
    assert helpers.parse_pack_label("gpt - x") == ""


# --------------------------------------------------------------------------- #
# 3. User-message scaffold
# --------------------------------------------------------------------------- #
def test_t2i_user_message_is_the_bare_request(helpers):
    assert helpers.build_user_message("t2i", "A man walking with his dog") == (
        "A man walking with his dog"
    )


def test_t2i_canvas_line_leads_and_is_the_only_difference(helpers):
    got = helpers.build_user_message("t2i", "a cat", target_canvas="16:9")
    assert got == "TARGET CANVAS: 16:9\na cat"


def test_i2i_user_message_uses_official_three_line_scaffold(helpers):
    got = helpers.build_user_message("i2i", "put a red bandana around the dog's neck")
    assert got.splitlines() == [
        "Reference images follow in order (the first image, the second image, ...).",
        "User request: put a red bandana around the dog's neck",
        "Output the caption.",
    ]


def test_i2i_ignores_canvas(helpers):
    """Upstream only teaches the canvas line to the t2i prompts."""
    assert helpers.build_user_message("i2i", "x", target_canvas="16:9") == (
        helpers.build_user_message("i2i", "x")
    )


def test_bbox_modes_close_with_their_own_instruction(helpers):
    t2i = helpers.build_user_message("t2i_bbox", "silhouette")
    assert t2i.splitlines()[-1] == "Output the caption and the table."
    i2i = helpers.build_user_message("i2i_bbox", "move the cup")
    assert i2i.splitlines()[-1] == "Output the caption."


def test_bbox_modes_pass_the_table_through_verbatim(helpers):
    table = '[{"id":"cup_1","from":"ref_image_0","src_bbox":[1,2,3,4],"tgt_bbox":[5,6,7,8],"desc":"A white ceramic cup."}]'
    got = helpers.build_user_message("i2i_bbox", "move the cup", bbox_json=table)
    lines = got.splitlines()
    assert lines[2] == "Bounding boxes (optional):"
    assert lines[3] == table, "user-supplied table must survive untouched"
    assert lines[4] == "Output the caption."


def test_t2i_bbox_uses_its_own_table_header(helpers):
    got = helpers.build_user_message("t2i_bbox", "x", bbox_json="[{}]")
    assert "Bounding boxes (optional, keep exactly if provided):" in got


def test_bbox_modes_accept_a_canvas(helpers):
    got = helpers.build_user_message("t2i_bbox", "silhouette", target_canvas="1:1")
    assert got.splitlines()[0] == "TARGET CANVAS: 1:1"


def test_empty_prompt_raises(helpers):
    for blank in ("", "   ", None):
        with pytest.raises(ValueError):
            helpers.build_user_message("t2i", blank)


@pytest.mark.parametrize("bad", ['{"id":"a"}', "not json at all", "[1, 2"])
def test_malformed_bbox_json_rejected(helpers, bad):
    with pytest.raises(ValueError):
        helpers.build_user_message("t2i_bbox", "x", bbox_json=bad)


def test_empty_bbox_json_omits_the_table_section(helpers):
    got = helpers.build_user_message("t2i_bbox", "x", bbox_json="   ")
    assert "Bounding boxes" not in got


def test_bbox_json_must_be_an_array(helpers):
    with pytest.raises(ValueError):
        helpers.build_user_message("t2i_bbox", "x", bbox_json='{"id": "a_1"}')


# --------------------------------------------------------------------------- #
# 4. Canvas normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("16:9", "16:9"),
    ("16 : 9", "16:9"),
    ("1:1", "1:1"),
    ("1.7778", "16:9"),
    ("1.0", "1:1"),
    ("", ""),
    ("auto", ""),
    ("AUTO", ""),
])
def test_normalize_canvas(helpers, raw, expected):
    assert helpers.normalize_canvas(raw) == expected


@pytest.mark.parametrize("bad", ["16/9", "wide", "0:1", "-1:1", "2.5"])
def test_normalize_canvas_rejects_junk(helpers, bad):
    with pytest.raises(ValueError):
        helpers.normalize_canvas(bad)


# --------------------------------------------------------------------------- #
# 5. <think> stripping
# --------------------------------------------------------------------------- #
def test_strip_think(helpers):
    assert helpers.strip_think("<think>hmm</think>caption") == "caption"
    assert helpers.strip_think("caption<think>trailing</think>") == "caption"
    assert helpers.strip_think("plain caption") == "plain caption"
    assert helpers.strip_think("") == ""
    # An unterminated block is left alone rather than eating the caption.
    assert helpers.strip_think("<think>unclosed caption") == "<think>unclosed caption"


# --------------------------------------------------------------------------- #
# 6. bbox reply splitting
# --------------------------------------------------------------------------- #
def test_split_bbox_output_simple(helpers):
    raw = 'A caption paragraph. [{"id":"a_1","bbox":[0,0,10,10],"desc":"x"}]'
    caption, table_json, table = helpers.split_bbox_output(raw, mode="t2i_bbox")
    assert caption == "A caption paragraph."
    assert json.loads(table_json) == [
        {"id": "a_1", "bbox": [0, 0, 10, 10], "desc": "x"}
    ]
    assert table[0]["id"] == "a_1"


def test_split_bbox_output_survives_brackets_in_desc(helpers):
    """The array boundary is found by parsing, not by the first bracket."""
    raw = (
        'A sign reads "[SALE]" in red. '
        '[{"id":"sign_1","bbox":[0,0,10,10],'
        '"desc":"A red sign reading \\"[SALE]\\" over a nested list [1, 2]."}]'
    )
    caption, table_json, _table = helpers.split_bbox_output(raw, mode="t2i_bbox")
    assert caption == 'A sign reads "[SALE]" in red.'
    assert json.loads(table_json)[0]["id"] == "sign_1"


def test_split_bbox_output_is_a_noop_for_non_bbox_modes(helpers):
    assert helpers.split_bbox_output("just a caption", mode="t2i") == (
        "just a caption", "", None
    )
    assert helpers.split_bbox_output("just a caption", mode="i2i") == (
        "just a caption", "", None
    )


def test_split_bbox_output_requires_an_array(helpers):
    with pytest.raises(ValueError):
        helpers.split_bbox_output("A caption with no table at all.", mode="i2i_bbox")


def test_split_bbox_output_requires_a_caption(helpers):
    """A bare array with no leading paragraph is not the official contract."""
    with pytest.raises(ValueError):
        helpers.split_bbox_output('[{"id":"a_1"}]', mode="t2i_bbox")


def test_split_bbox_output_strips_think_first(helpers):
    raw = '<think>plan</think>Caption here. [{"id":"a_1","bbox":[0,0,1,1],"desc":"d"}]'
    caption, _json_out, _table = helpers.split_bbox_output(raw, mode="t2i_bbox")
    assert caption == "Caption here."


# --------------------------------------------------------------------------- #
# 7. Node behaviour
# --------------------------------------------------------------------------- #
def test_t2i_end_to_end_sends_the_official_system_prompt(helpers, mod):
    conn = _FakeConnector(["Full-color outdoor photograph, eye-level medium-wide shot."])
    node = mod.Flux3KleinPromptUpsampler()
    out = node.upsample(conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "a man and his dog")
    assert len(out) == 3
    assert out[0] == "Full-color outdoor photograph, eye-level medium-wide shot."
    # non-bbox modes leave the split outputs empty
    assert out[1] == out[0]
    assert out[2] == ""

    call = conn.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"] == helpers.load_system_prompt("t2i", "short")
    assert call["messages"][1] == {"role": "user", "content": "a man and his dog"}
    assert call["temperature"] == mod._DEFAULT_TEMPERATURE
    assert call["max_tokens"] == mod._MAX_TOKENS_DEFAULT


def test_bbox_mode_splits_into_three_outputs(helpers, mod):
    raw = 'Minimalist graphic illustration. [{"id":"background_1","bbox":[0,0,1000,1000],"desc":"A flat field."}]'
    conn = _FakeConnector([raw])
    node = mod.Flux3KleinPromptUpsampler()
    full, caption, table_json = node.upsample(
        conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "black silhouette on chartreuse"
    )
    assert full == raw
    assert caption == "Minimalist graphic illustration."
    assert json.loads(table_json)[0]["id"] == "background_1"
    assert conn.calls[0]["messages"][0]["content"] == helpers.load_system_prompt(
        "t2i_bbox", "short"
    )


def test_prompt_pack_selects_a_different_system_prompt(helpers, mod):
    conn = _FakeConnector(["a", "b"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "x")
    node.upsample(conn, helpers.MODES[0], helpers.PROMPT_PACKS[1], "x")
    short_sys = conn.calls[0]["messages"][0]["content"]
    long_sys = conn.calls[1]["messages"][0]["content"]
    assert short_sys != long_sys
    assert short_sys == helpers.load_system_prompt("t2i", "short")
    assert long_sys == helpers.load_system_prompt("t2i", "long")


def test_vision_mode_attaches_images_in_order_before_the_text(helpers, mod):
    conn = _FakeConnector(["Change the ... to red. Keep the rest unchanged."])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[1], helpers.PROMPT_PACKS[0],
        "change the dress color to red", reference_images=_FakeImageBatch(2),
    )
    parts = conn.calls[0]["messages"][1]["content"]
    assert isinstance(parts, list)
    images = [p for p in parts if p["type"] == "image_url"]
    texts = [p for p in parts if p["type"] == "text"]
    assert len(images) == 2, "both references must be attached, in order"
    assert len(texts) == 1
    # images precede the request text, matching the official i2i scaffold
    assert parts.index(images[0]) < parts.index(texts[0])
    assert texts[0]["text"].splitlines()[0] == (
        "Reference images follow in order (the first image, the second image, ...)."
    )


@pytest.mark.parametrize("mode_label", ["i2i - 图生图编辑(需参考图)", "i2i_bbox - 编辑+布局框(需参考图)"])
def test_vision_modes_require_images(helpers, mod, mode_label):
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(ValueError, match="reference"):
        node.upsample(_FakeConnector(), mode_label, helpers.PROMPT_PACKS[0], "x")


def test_text_modes_ignore_stray_images_with_a_warning(helpers, mod):
    conn = _FakeConnector(["a caption"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "x",
        reference_images=_FakeImageBatch(3),
    )
    assert conn.calls[0]["messages"][1]["content"] == "x"


def test_more_than_ten_references_are_truncated(helpers, mod):
    conn = _FakeConnector(["a caption"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[1], helpers.PROMPT_PACKS[0], "x",
        reference_images=_FakeImageBatch(12),
    )
    parts = conn.calls[0]["messages"][1]["content"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 10


def test_empty_llm_reply_raises_with_a_token_hint(helpers, mod):
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(RuntimeError, match="max_tokens"):
        node.upsample(_FakeConnector([""]), helpers.MODES[0], helpers.PROMPT_PACKS[0], "x")
    with pytest.raises(RuntimeError, match="max_tokens"):
        node.upsample(
            _FakeConnector(["   "]), helpers.MODES[0], helpers.PROMPT_PACKS[0], "x"
        )


def test_bad_bbox_json_is_rejected_before_any_llm_call(helpers, mod):
    conn = _FakeConnector()
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(ValueError):
        node.upsample(
            conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "x", bboxes_json="{oops"
        )
    assert conn.calls == [], "no tokens may be spent on an invalid table"


def test_bbox_table_only_reaches_bbox_modes(helpers, mod):
    """A stray table in t2i mode is ignored rather than smuggled into the prompt."""
    conn = _FakeConnector(["a caption"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "x",
        bboxes_json='[{"id":"a_1"}]',
    )
    assert conn.calls[0]["messages"][1]["content"] == "x"


def test_timeout_override_is_applied_and_restored(helpers, mod):
    conn = _FakeConnector(["a caption"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "x", timeout=600
    )
    assert conn.calls[0]["timeout"] == 600
    assert conn.timeout == 120, "the connector's own timeout must be restored"


# --------------------------------------------------------------------------- #
# 8. Surface / registration
# --------------------------------------------------------------------------- #
def test_input_types_shape(helpers, mod):
    spec = mod.Flux3KleinPromptUpsampler.INPUT_TYPES()
    assert set(spec["required"]) == {
        "llm_service_connector", "mode", "prompt_pack", "prompt", "seed",
    }
    assert set(spec["optional"]) == {
        "reference_images", "target_canvas", "bboxes_json",
        "temperature", "max_tokens", "timeout",
    }
    assert spec["required"]["llm_service_connector"] == ("LLMServiceConnector",)
    assert spec["optional"]["reference_images"][0] == "IMAGE"
    assert spec["required"]["mode"][0] == list(helpers.MODES)
    assert spec["required"]["prompt_pack"][0] == list(helpers.PROMPT_PACKS)
    assert spec["required"]["prompt"][1]["multiline"] is True


def test_return_and_category(mod):
    node = mod.Flux3KleinPromptUpsampler()
    assert node.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert node.RETURN_NAMES == ("upsampled_prompt", "caption", "bbox_json")
    assert node.FUNCTION == "upsample"
    assert node.CATEGORY == "\U0001F411 MieNodes/\U0001F411 Prompt Generator"


def test_node_is_registered_in_the_plugin():
    from test_plugin_imports import load_plugin_module  # noqa: PLC0415

    plugin = load_plugin_module()
    assert "Flux3KleinPromptUpsampler|Mie" in plugin.NODE_CLASS_MAPPINGS
    assert "Flux3KleinPromptUpsampler|Mie" in plugin.NODE_DISPLAY_NAME_MAPPINGS


# --------------------------------------------------------------------------- #
# 9. is_changed
# --------------------------------------------------------------------------- #
def _changed(node, conn, **overrides):
    kwargs = dict(
        llm_service_connector=conn, mode="t2i", prompt_pack="short",
        prompt="a cat", seed=0, reference_images=None, target_canvas="",
        bboxes_json="", temperature=0.2, max_tokens=16384, timeout=300,
    )
    kwargs.update(overrides)
    return node.is_changed(**kwargs)


def test_is_changed_is_stable_and_sensitive(helpers, mod):
    node = mod.Flux3KleinPromptUpsampler()
    conn = _FakeConnector()
    base = _changed(node, conn)
    assert base == _changed(node, conn)
    for field, value in [
        ("prompt", "a dog"),
        ("mode", "i2i - 图生图编辑(需参考图)"),
        ("prompt_pack", "long"),
        ("target_canvas", "16:9"),
        ("seed", 7),
        ("temperature", 0.9),
        ("max_tokens", 4096),
        ("timeout", 60),
        ("bboxes_json", '[{"id":"a_1"}]'),
    ]:
        assert _changed(node, conn, **{field: value}) != base, f"{field} is ignored"


def test_is_changed_tracks_the_reference_images(helpers, mod):
    node = mod.Flux3KleinPromptUpsampler()
    conn = _FakeConnector()
    one = _changed(node, conn, reference_images=_FakeImageBatch(1))
    two = _changed(node, conn, reference_images=_FakeImageBatch(2))
    assert one != two, "a different reference count must invalidate the cache"

    # Same shape, different pixels: the pixel sampler has to be doing work,
    # otherwise swapping in a different photo would silently reuse a caption.
    dark = _changed(node, conn, reference_images=_FakeImageBatch(1, fill=0.0))
    bright = _changed(node, conn, reference_images=_FakeImageBatch(1, fill=1.0))
    assert dark != bright, "swapped reference pixels must invalidate the cache"
    assert dark == _changed(node, conn, reference_images=_FakeImageBatch(1, fill=0.0))
