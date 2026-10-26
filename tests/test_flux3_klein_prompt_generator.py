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

    def invoke(self, messages, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "seed": kwargs.get("seed"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "timeout": self.timeout,
                "kwargs": kwargs,
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


def test_canvas_ratio_is_read_from_the_preview(helpers, mod):
    """The aspect ratio comes from the canvas, never from a typed field."""
    class _Canvas:
        def __init__(self, h, w):
            self.shape = (1, h, w, 3)

    assert mod._canvas_ratio(_Canvas(576, 1024)) == "1024:576"
    assert mod._canvas_ratio(_Canvas(1024, 1024)) == "1024:1024"
    assert mod._canvas_ratio(None) == ""
    assert mod._canvas_ratio(object()) == ""


def test_t2i_sends_the_canvas_line_from_the_preview(helpers, mod):
    """The official t2i prompts steer composition by the canvas ratio."""
    class _Canvas:
        shape = (1, 576, 1024, 3)

    conn = _FakeConnector(
        ['A caption. [{"id":"obj_1","bbox":[300,400,700,600],"desc":"a bench"}]']
    )
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "a bench",
        elements=[{"type": "obj", "bbox": [300, 400, 700, 600], "desc": "a bench"}],
        canvas_preview=_Canvas(),
    )
    content = conn.calls[0]["messages"][1]["content"]
    assert content.splitlines()[0] == "TARGET CANVAS: 1024:576"


def test_bbox_modes_close_with_their_own_instruction(helpers):
    t2i = helpers.build_user_message("t2i_bbox", "silhouette")
    assert t2i.splitlines()[-1] == "Output the caption and the table."
    i2i = helpers.build_user_message("i2i_bbox", "move the cup")
    assert i2i.splitlines()[-1] == "Output the caption."


def test_bbox_modes_pass_the_canvas_rows_through(helpers):
    """The marked rows reach the model as the table."""
    rows = [{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": "A white ceramic cup."}]
    got = helpers.build_user_message("i2i_bbox", "move the cup", rows=rows)
    lines = got.splitlines()
    assert lines[2] == "Bounding boxes (optional):"
    assert json.loads(lines[3]) == rows
    assert lines[4] == "Output the caption."


def test_t2i_bbox_uses_its_own_table_header(helpers):
    got = helpers.build_user_message("t2i_bbox", "x", rows=[{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": ""}])
    assert "Bounding boxes (optional, keep exactly if provided):" in got


def test_bbox_modes_run_on_the_table_alone(helpers):
    """A marked region already describes the scene, so no prose is needed."""
    rows = [{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": "A white ceramic cup."}]
    got = helpers.build_user_message("i2i_bbox", "", rows=rows)
    lines = got.splitlines()
    assert lines[1].startswith("User request: ")
    assert "A white ceramic cup." in lines[1]


def test_bbox_modes_still_need_either_prompt_or_table(helpers):
    """No prose and no marked region is nothing to upsample."""
    for mode in ("t2i_bbox", "i2i_bbox"):
        with pytest.raises(ValueError):
            helpers.build_user_message(mode, "")
    # A row with no desc yields no usable request either.
    with pytest.raises(ValueError):
        helpers.build_user_message("t2i_bbox", "", rows=[{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": ""}])


def test_table_only_request_is_composed_from_descs_not_invented(helpers):
    rows = [{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": 'A sign reading "Nice Day".'},
            {"id": "obj_2", "bbox": [5, 6, 7, 8], "desc": "A tabby cat."}]
    request = helpers.build_user_message("t2i_bbox", "", rows=rows).splitlines()[0]
    assert request == (
        'User request: a scene containing: A sign reading "Nice Day".; A tabby cat.'
    )


# --------------------------------------------------------------------------- #
# 3b. CreateBoundingBoxes elements -> official table
# --------------------------------------------------------------------------- #
def test_elements_to_table_matches_flux3_layout_prompt(helpers):
    """The adapter must reproduce ComfyUI's own Flux3LayoutPrompt row build.

    ``Flux3LayoutPrompt.execute`` emits ``{"id": f"obj_{i+1}", "bbox":
    el["bbox"], "desc": desc}`` with the text quoted into the desc. Matching it
    exactly is the point: the boxes drawn on the canvas then reach the model in
    the same form the native node would have produced.
    """
    elements = [
        {"type": "obj", "bbox": [416, 64, 960, 352], "desc": "a cat"},
        {"type": "text", "bbox": [94, 594, 281, 1000], "desc": "", "text": "Nice Day"},
    ]
    assert helpers.elements_to_table(elements) == [
        {"id": "obj_1", "bbox": [416, 64, 960, 352], "desc": "a cat"},
        {"id": "obj_2", "bbox": [94, 594, 281, 1000], "desc": '"Nice Day"'},
    ]


def test_elements_to_table_appends_quoted_text_to_a_desc(helpers):
    rows = helpers.elements_to_table([
        {"type": "text", "bbox": [0, 0, 100, 100], "desc": "A shop sign", "text": "OPEN"},
    ])
    assert rows[0]["desc"] == 'A shop sign "OPEN"'


def test_elements_to_table_preserves_the_official_grid(helpers):
    """Boxes pass through untouched — the canvas already normalized them."""
    bbox = [64, 416, 608, 352]
    rows = helpers.elements_to_table([{"type": "obj", "bbox": bbox, "desc": "x"}])
    assert rows[0]["bbox"] == bbox


def test_elements_to_table_drops_palettes(helpers):
    """Upstream says colour palettes are not used; colours belong in desc."""
    rows = helpers.elements_to_table([
        {"type": "obj", "bbox": [0, 0, 10, 10], "desc": "a cat", "color_palette": ["#fff"]},
    ])
    assert "color_palette" not in rows[0]


@pytest.mark.parametrize("bad", [[], None, "not a list", [42]])
def test_elements_to_table_rejects_unusable_input(helpers, bad):
    with pytest.raises(ValueError):
        helpers.elements_to_table(bad)


def test_elements_to_table_rejects_a_row_without_a_box(helpers):
    with pytest.raises(ValueError):
        helpers.elements_to_table([{"type": "obj", "desc": "a cat"}])
    with pytest.raises(ValueError):
        helpers.elements_to_table([{"type": "obj", "bbox": [1, 2], "desc": "x"}])


def test_node_accepts_elements_instead_of_hand_written_json(helpers, mod):
    conn = _FakeConnector(['A caption. [{"id":"obj_1","bbox":[1,2,3,4],"desc":"a cat"}]'])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "in the forest",
        elements=[{"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cat"}],
    )
    sent = conn.calls[0]["messages"][1]["content"]
    assert '{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": "a cat"}' in sent


def test_elements_are_accepted_in_i2i_bbox(helpers, mod):
    """Canvas rows are transported as drawn; the model converts them.

    The official i2i_bbox system prompt is the annotation system that turns
    "a marked region plus an instruction" into src/tgt rows, and the native
    ComfyUI flow does exactly this -- Flux3LayoutPrompt hands the
    generation-shaped rows over and the model produces the edit-shaped table.
    Refusing the canvas would make that path impossible.
    """
    conn = _FakeConnector([
        'Move the cup. [{"id": "cup_1", "from": "ref_image_0", '
        '"src_bbox": [1, 2, 3, 4], "tgt_bbox": [1, 2, 3, 4], "desc": "a cup"}]'
    ])
    node = mod.Flux3KleinPromptUpsampler()
    full, caption, table_json = node.upsample(
        conn, helpers.MODES[3], helpers.PROMPT_PACKS[0], "move the cup",
        elements=[{"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cup"}],
            reference_images=_FakeImageBatch(1),
    )
    content = conn.calls[0]["messages"][1]["content"]
    text = content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content if part.get("type") == "text"
    )
    assert '"bbox": [1, 2, 3, 4]' in text, "the marked region must reach the model"
    assert json.loads(table_json)[0]["id"] == "cup_1"


def test_generation_shaped_table_allows_the_model_to_add_rows(helpers, mod):
    """The row shape must change; only the coordinates are held."""
    conn = _FakeConnector([
        'Changed. [{"id": "cup_1", "from": "ref_image_0", "src_bbox": [1, 2, 3, 4], '
        '"tgt_bbox": [1, 2, 3, 4], "desc": "a cup"}, '
        '{"id": "table_1", "from": "ref_image_0", "src_bbox": [5, 6, 7, 8], '
        '"tgt_bbox": [5, 6, 7, 8], "desc": "a table"}]'
    ])
    node = mod.Flux3KleinPromptUpsampler()
    # the model added a second row for the table -- but every box is one the
    # user marked, so it must be accepted.
    node.upsample(
        conn, helpers.MODES[3], helpers.PROMPT_PACKS[0], "x",
        elements=[
            {"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cup"},
            {"type": "obj", "bbox": [5, 6, 7, 8], "desc": "a table"},
        ],
        reference_images=_FakeImageBatch(1),
    )


def test_generation_shaped_table_rejects_a_dropped_region(helpers, mod):
    """A reply that ignores the marked region is the failure worth catching."""
    bad = ('Changed. [{"id": "cup_1", "from": "target", "src_bbox": null, '
           '"tgt_bbox": [900, 900, 950, 950], "desc": "a cup"}]')
    conn = _FakeConnector([bad, bad])
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(ValueError) as err:
        node.upsample(
            conn, helpers.MODES[3], helpers.PROMPT_PACKS[0], "x",
            elements=[{"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cup"}],
            reference_images=_FakeImageBatch(1),
        )
    assert "does not account for the marked regions" in str(err.value)


def test_generation_shaped_table_allows_the_model_to_tighten_a_box(helpers, mod):
    """The official annotation system tightens a marked box onto the object.

    A hand-drawn region is usually larger than what it marks; upstream's own
    behaviour is to refine it (a marked [438, 31, 688, 188] comes back as
    [572, 34, 681, 181]). Holding the reply to byte equality would forbid
    exactly the refinement that makes an edit land on the object.
    """
    refined = ('Changed. [{"id": "cup_1", "from": "ref_image_0", '
               '"src_bbox": [572, 34, 681, 181], "tgt_bbox": [550, 42, 688, 181], '
               '"desc": "a cup"}]')
    conn = _FakeConnector([refined])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[3], helpers.PROMPT_PACKS[0], "x",
        elements=[{"type": "obj", "bbox": [438, 31, 688, 188], "desc": "a cup"}],
        reference_images=_FakeImageBatch(1),
    )


def test_there_is_only_one_table_source(helpers, mod):
    """The canvas is the only source; a hand-written JSON slot no longer exists."""
    spec = mod.Flux3KleinPromptUpsampler.INPUT_TYPES()
    assert "bboxes_json" not in spec["optional"]
    assert "bboxes_json" not in spec["required"]
    assert "elements" in spec["optional"]


def test_connected_but_empty_elements_is_an_error(helpers, mod):
    """A connected-but-empty socket must not silently plan an unasked layout."""
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(ValueError) as err:
        node.upsample(
            _FakeConnector(), helpers.MODES[2], helpers.PROMPT_PACKS[0], "x",
            elements=[],
        )
    assert "silently ignored" in str(err.value)


def test_elements_none_means_no_table(helpers, mod):
    """An absent socket means the model plans the layout itself."""
    conn = _FakeConnector(['A caption. [{"id":"a_1","bbox":[1,2,3,4],"desc":"d"}]'])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "x", elements=None,
    )
    content = conn.calls[0]["messages"][1]["content"]
    assert "Bounding boxes" not in content, "no table means no table section"
    assert "a_1" not in content, "no table means the model plans the layout"


def test_elements_are_ignored_outside_bbox_modes(helpers, mod):
    conn = _FakeConnector(["a caption"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "x",
        elements=[{"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cat"}],
    )
    assert conn.calls[0]["messages"][1]["content"] == "x"


def test_is_changed_tracks_canvas_edits(helpers, mod):
    """Moving a box on the canvas changes no widget, so it must still bust."""
    node = mod.Flux3KleinPromptUpsampler()
    conn = _FakeConnector()
    moved = [
        {"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cat"},
        {"type": "obj", "bbox": [5, 6, 7, 8], "desc": "an apple"},
    ]
    edited = list(moved)
    edited[1] = {"type": "obj", "bbox": [50, 60, 70, 80], "desc": "an apple"}
    a = _changed(node, conn, elements=moved)
    b = _changed(node, conn, elements=edited)
    assert a != b, "a canvas edit must invalidate the cached caption"


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


def test_split_bbox_output_survives_a_markdown_fence(helpers):
    """Models habitually fence JSON; the fence must not break the contract."""
    raw = (
        '```json\nCaption here. [{"id":"a_1","bbox":[0,0,1,1],"desc":"d"}]\n```'
    )
    caption, table_json, _table = helpers.split_bbox_output(raw, mode="t2i_bbox")
    assert caption == "Caption here."
    assert json.loads(table_json)[0]["id"] == "a_1"


def test_split_bbox_failure_reports_the_reply_tail(helpers):
    """An opaque failure is unreproducible from the UI; name the tail."""
    with pytest.raises(ValueError) as err:
        helpers.split_bbox_output("no table here at all", mode="t2i_bbox")
    assert "no table here at all" in str(err.value)


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


def test_node_refuses_the_reasoning_fallback(helpers, mod):
    """A provider that returns only reasoning must not pass as a caption."""
    seen = {}

    class _Recorder(_FakeConnector):
        def invoke(self, messages, **kwargs):
            seen.update(kwargs)
            return super().invoke(messages, **kwargs)

    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(_Recorder(["a caption"]), helpers.MODES[0], helpers.PROMPT_PACKS[0], "x")
    assert seen.get("allow_reasoning_fallback") is False


def test_node_logs_the_reply_when_the_bbox_split_fails(helpers, mod):
    """Without the log line this failure is undiagnosable from the UI."""
    logged: list[str] = []
    original = mod.mie_log
    mod.mie_log = lambda msg: logged.append(msg)
    try:
        node = mod.Flux3KleinPromptUpsampler()
        with pytest.raises(ValueError):
            node.upsample(
                _FakeConnector(["a caption with no table", "still no table"]),
                helpers.MODES[2], helpers.PROMPT_PACKS[0], "x",
            )
    finally:
        mod.mie_log = original
    assert any("no parseable" in m and "a caption with no table" in m for m in logged)


def test_bbox_split_is_resampled_once_before_failing(helpers, mod):
    """A loose ``id [coords] desc`` row is resampled, not fatal on first sight."""
    good = 'A caption. [{"id":"obj_1","bbox":[1,2,3,4],"desc":"a cat"}]'
    conn = _FakeConnector(['En_Text_1 [525, 295, 690, 705] "HELLO!" set in bold.', good])
    node = mod.Flux3KleinPromptUpsampler()
    full, caption, table_json = node.upsample(
        conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "a poster",
    )
    assert len(conn.calls) == 2, "the malformed reply must trigger one resample"
    assert caption == "A caption."
    assert json.loads(table_json)[0]["id"] == "obj_1"
    assert full == good


def test_bbox_split_gives_up_after_the_configured_attempts(helpers, mod):
    conn = _FakeConnector(["no table here", "still no table"])
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(ValueError):
        node.upsample(conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "x")
    assert len(conn.calls) == mod._TABLE_PARSE_ATTEMPTS


def test_empty_canvas_is_rejected_before_any_llm_call(helpers, mod):
    """A connected-but-empty canvas must fail loudly, not spend tokens."""
    conn = _FakeConnector()
    node = mod.Flux3KleinPromptUpsampler()
    with pytest.raises(ValueError, match="empty"):
        node.upsample(
            conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "x", elements=[],
        )
    assert conn.calls == [], "no tokens may be spent on an empty table"


def test_bbox_table_only_reaches_bbox_modes(helpers, mod):
    """A stray canvas in t2i mode is ignored rather than smuggled into the prompt."""
    conn = _FakeConnector(["a caption"])
    node = mod.Flux3KleinPromptUpsampler()
    node.upsample(
        conn, helpers.MODES[0], helpers.PROMPT_PACKS[0], "x",
        elements=[{"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cat"}],
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
        "reference_images", "elements", "canvas_preview",
        "temperature", "max_tokens", "timeout",
    }
    assert spec["required"]["llm_service_connector"] == ("LLMServiceConnector",)
    assert spec["optional"]["reference_images"][0] == "IMAGE"
    assert spec["optional"]["elements"][0] == "ARRAY"
    assert spec["optional"]["canvas_preview"][0] == "IMAGE"
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
        prompt="a cat", seed=0, reference_images=None, elements=None,
        canvas_preview=None, temperature=0.2, max_tokens=16384, timeout=300,
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
        ("canvas_preview", _FakeImageBatch(1)),
        ("seed", 7),
        ("temperature", 0.9),
        ("max_tokens", 4096),
        ("timeout", 60),
        ("elements", [{"type": "obj", "bbox": [1, 2, 3, 4], "desc": "a cat"}]),
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

@pytest.mark.parametrize("mode_index", [2, 3])
def test_supplied_table_drift_is_retried_before_returning(helpers, mod, mode_index):
    """A reply that drops the marked region is retried once."""
    canvas = [{"type": "obj", "bbox": [100, 200, 300, 400], "desc": "a cup"}]
    bad = ("Move the cup <cup_1>. "
           + json.dumps([{"id": "cup_1", "from": "target", "src_bbox": None,
                          "tgt_bbox": [900, 900, 950, 950], "desc": "a cup"}]))
    good = ("Move the cup <cup_1>. "
            + json.dumps([{"id": "cup_1", "from": "target", "src_bbox": None,
                           "tgt_bbox": [120, 220, 290, 380], "desc": "a cup"}]))
    conn = _FakeConnector([bad, good])
    full, caption, table = mod.Flux3KleinPromptUpsampler().upsample(
        conn, helpers.MODES[mode_index], helpers.PROMPT_PACKS[0], "move the cup",
        elements=canvas,
        reference_images=_FakeImageBatch(1),
    )
    assert json.loads(table)[0]["id"] == "cup_1"
    assert full == good
    assert len(conn.calls) == 2


def test_repeated_supplied_table_drift_is_an_error(helpers, mod):
    """A reply that keeps dropping the marked region fails the run."""
    canvas = [{"type": "obj", "bbox": [100, 200, 300, 400], "desc": "a cup"}]
    bad = ("A cup <cup_1>. "
           + json.dumps([{"id": "cup_1", "from": "target", "src_bbox": None,
                          "tgt_bbox": [900, 900, 950, 950], "desc": "a cup"}]))
    conn = _FakeConnector([bad, bad])
    with pytest.raises(ValueError, match="does not account for the marked regions"):
        mod.Flux3KleinPromptUpsampler().upsample(
            conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "a cup", elements=canvas,
        )
    assert len(conn.calls) == mod._TABLE_PARSE_ATTEMPTS


def test_generation_shaped_drift_is_an_error(helpers, mod):
    """Canvas rows may be reshaped, but not dropped entirely."""
    canvas = [{"type": "obj", "bbox": [100, 200, 300, 400], "desc": "a cup"}]
    bad = ("A cup <cup_1>. "
           + json.dumps([{"id": "cup_1", "bbox": [1, 2, 3, 4], "desc": "a cup"}]))
    conn = _FakeConnector([bad, bad])
    with pytest.raises(ValueError, match="does not account for the marked regions"):
        mod.Flux3KleinPromptUpsampler().upsample(
            conn, helpers.MODES[2], helpers.PROMPT_PACKS[0], "a cup", elements=canvas,
        )
    assert len(conn.calls) == mod._TABLE_PARSE_ATTEMPTS


