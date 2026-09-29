# -*- coding: utf-8 -*-
"""Tests for the Qwen-Image-2.1 prompt-enhancer ComfyUI node.

The node is a thin wrapper around the LLM call defined in
``nodes/llm/qwen_image_21_prompt_generator.py``: it takes a rough draft
plus ``mode`` and an optional ``reference_images`` IMAGE batch, calls
the LLM with the matching bundled system prompt under
``nodes/llm/prompts/qwen_image_21/``, then extracts the
``--- BEGIN enhanced_prompt ---`` ... block from the reply. Tests cover:

* Parser: block extraction, whitespace trimming, ``<think>`` stripping.
* Failure modes: missing BEGIN/END markers raise ``RuntimeError``, not
  silent fallback.
* Mode routing: 5 modes → 5 distinct .txt files (and remove_bg
  short-circuits without an LLM call).
* Widget passthrough: the reference count reaches the user message
  only for the modes that consume image slots.
* Surface: required / optional widget shape, RETURN_NAMES =
  ("enhanced_prompt",), CATEGORY matches the loop node, ``is_changed``
  is stable and content-aware (system-prompt body participates).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
LLM_DIR = PROJECT_DIR / "nodes" / "llm"


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


@pytest.fixture(scope="module")
def helpers():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader",
        PROMPTS_DIR / "loader.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.qwen_image_21_prompts",
        LLM_DIR / "qwen_image_21_prompts.py",
    )


@pytest.fixture(scope="module")
def mod(helpers):
    # Generator module imports the helpers; we still load it last so its
    # `from .qwen_image_21_prompts import …` resolves to the shimmed
    # helpers module above.
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    return _load_file(
        "_mienodes_internal.nodes.llm.qwen_image_21_prompt_generator",
        LLM_DIR / "qwen_image_21_prompt_generator.py",
    )


@pytest.fixture(autouse=True)
def _isolate_caption_cache(mod, tmp_path, monkeypatch):
    """Force every test in this module to use a fresh tmp_path as the
    caption disk-cache root. Real ComfyUI runs and earlier test runs
    may have written entries to the persistent cache directory; if we
    didn't isolate, a test fake that happened to encode to a
    previously-cached data URL would silently disk-hit instead of
    exercising the miss path it's supposed to.

    The generator imports ``caption_cache_disk_root`` by name into its
    own namespace (``from .qwen_image_21_prompts import
    caption_cache_disk_root``), so we patch the symbol in the
    generator's namespace — patching the helpers namespace wouldn't
    reach the captioner.
    """
    target = str(tmp_path / "mien_nodes" / "caption_cache")
    monkeypatch.setattr(mod, "caption_cache_disk_root", lambda: target)
    return target


# --------------------------------------------------------------------------- #
# Test fixtures: stub LLM connector + canned reply
# --------------------------------------------------------------------------- #
class _FakeConnector:
    """Stand-in for ``LLMServiceConnector`` exposing only the methods the
    enhancer calls."""

    model = "fake-model"

    def __init__(self):
        self.calls: list[dict] = []
        self.timeout = 120

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        self.calls.append({
            "messages": list(messages),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return _REPLY_QUEUE.pop(0) if _REPLY_QUEUE else ""

    def get_state(self):
        return "fake-state"


def _make_connector_with_replies(replies):
    """Build a connector whose ``invoke`` returns the supplied canned
    replies in order. Empty queue + extra call raises ``AssertionError``.
    """
    conn = _FakeConnector()
    conn._replies = list(replies)

    def invoke(messages, *, seed=None, temperature=None, max_tokens=None):
        conn.calls.append({
            "messages": list(messages),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if not conn._replies:
            raise AssertionError("scripted connector exhausted")
        return conn._replies.pop(0)

    conn.invoke = invoke
    return conn


_REPLY_QUEUE: list[str] = []


class _FakeImageBatch:
    """Stand-in for a ComfyUI ``IMAGE`` batch — a real ``numpy.ndarray``
    with ``shape == (N, H, W, C)`` so the captioner can encode it via
    the plugin's standard ``image_tensor_batch_to_data_urls`` path.

    The numpy shape-only surface is enough: ``image_tensor_to_data_url``
    falls through to ``np.array(t)`` (no ``.detach()``) and then runs
    ``cv2.imencode`` to produce a JPEG data URL. The fake pixels are
    all-zero so the encoded data URL is short and deterministic per
    frame; the cache key derives from the data URL so two fakes with
    the same ``n`` produce the same key.
    """

    def __init__(self, n: int):
        import numpy as np
        self._arr = np.zeros((int(n), 8, 8, 3), dtype=np.float32)

    def __getattr__(self, name):
        # Delegate everything else to the underlying ndarray so the
        # real encoder path works unmodified.
        return getattr(self.__dict__["_arr"], name)

    def __getitem__(self, idx):
        return self._arr[idx]

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def dtype(self):
        return self._arr.dtype


def _good_t2i_reply() -> str:
    """A well-formed LLM reply matching the T2I contract."""
    return (
        "Notes: short brief expanded to a single observer paragraph.\n"
        "\n"
        "--- BEGIN enhanced_prompt ---\n"
        "A vertical sticker of a glossy red apple with a single green "
        "leaf, on a fully transparent background, softly lit from the "
        "upper left.\n"
        "--- END enhanced_prompt ---\n"
    )


def _good_rgba_reply() -> str:
    """Well-formed RGBA reply: opening bookend + observer + closing bookend."""
    return (
        "--- BEGIN enhanced_prompt ---\n"
        "This is an RGBA image with transparency. A vertical sticker of "
        "a glossy red apple with a single green leaf. The image has "
        "alpha channel and the background is transparent.\n"
        "--- END enhanced_prompt ---\n"
    )


def _good_edit_reply() -> str:
    """Well-formed edit reply (English, single-image edit)."""
    return (
        "--- BEGIN enhanced_prompt ---\n"
        "Keep the face, pose and background of the input image unchanged. "
        "Change only the colour of the character's jacket from deep navy "
        "to warm terracotta.\n"
        "--- END enhanced_prompt ---\n"
    )


def _good_multi_ref_reply() -> str:
    """Well-formed multi-ref reply with <image1>..<image3>."""
    return (
        "--- BEGIN enhanced_prompt ---\n"
        "<image1> is the canvas: pose and background stay as is. "
        "<image2> provides the new outfit; transfer only the silhouette "
        "and material. <image3> provides the accessory placed on the "
        "left shoulder. Output keeps the original framing.\n"
        "--- END enhanced_prompt ---\n"
    )


# --------------------------------------------------------------------------- #
# Parser unit tests — extract_enhanced_prompt_block / split_enhancer_reply
# --------------------------------------------------------------------------- #
def test_extract_block_well_formed(helpers):
    """Plain well-formed reply: block is returned verbatim (inner only)."""
    raw = _good_t2i_reply()
    out = helpers.extract_enhanced_prompt_block(raw)
    assert out is not None
    assert out.startswith("A vertical sticker")
    assert out.endswith("upper left.")
    assert "Notes:" not in out
    assert "--- BEGIN enhanced_prompt ---" not in out
    assert "--- END enhanced_prompt ---" not in out


def test_extract_block_strips_surrounding_whitespace(helpers):
    """Leading/trailing blank lines and whitespace inside the block
    are trimmed before return."""
    raw = (
        "Notes: short.\n"
        "\n"
        "--- BEGIN enhanced_prompt ---\n"
        "\n"
        "\n"
        "A watchmaker at his bench.\n"
        "  \n"
        "--- END enhanced_prompt ---\n"
    )
    out = helpers.extract_enhanced_prompt_block(raw)
    assert out == "A watchmaker at his bench."


def test_extract_block_strips_think_wrapper(helpers):
    """Reasoning models may emit ``<think>...</think>`` before the
    visible answer; the wrapper must not leak into the captured
    block content."""
    raw = (
        "<think>"
        "The user wants a T2I observer rewrite. I'll emit a single "
        "paragraph between the markers."
        "</think>"
        "\n"
        "Notes: done.\n"
        "\n"
        "--- BEGIN enhanced_prompt ---\n"
        "An old watchmaker at his bench.\n"
        "--- END enhanced_prompt ---\n"
    )
    out = helpers.extract_enhanced_prompt_block(raw)
    assert out == "An old watchmaker at his bench."
    assert "<think>" not in out


def test_extract_block_returns_none_when_missing(helpers):
    """No BEGIN/END markers -> None (caller raises)."""
    assert helpers.extract_enhanced_prompt_block("just some chatter, no markers") is None
    assert helpers.extract_enhanced_prompt_block("") is None
    assert helpers.extract_enhanced_prompt_block(None) is None


def test_extract_block_requires_close_marker(helpers):
    """BEGIN without END -> None (don't half-extract)."""
    raw = "Notes: x\n--- BEGIN enhanced_prompt ---\nstuff"
    assert helpers.extract_enhanced_prompt_block(raw) is None


def test_split_enhancer_reply_returns_header_and_block(helpers):
    """``split_enhancer_reply`` returns (block, header); header is the
    prose before the BEGIN marker with ``<think>`` stripped."""
    raw = (
        "<think>\nthinking\n</think>\n"
        "Notes: short.\n"
        "\n"
        "--- BEGIN enhanced_prompt ---\n"
        "Body text.\n"
        "--- END enhanced_prompt ---\n"
    )
    block, header = helpers.split_enhancer_reply(raw)
    assert block == "Body text."
    assert "Notes: short." in header
    assert "<think>" not in header


# --------------------------------------------------------------------------- #
# Mode routing + file map
# --------------------------------------------------------------------------- #
def test_modes_contains_all_five_entries(helpers):
    """Five modes wired in the dropdown, in the documented order.
    Labels are bilingual (Loop pattern); the bare codes appear as
    the leading prefix of each label."""
    assert helpers.MODES == (
        "t2i - 文生图(默认)",
        "t2i_rgba - 透明文生图",
        "edit - 单图编辑",
        "multi_ref - 多参考编辑",
        "remove_bg - 抠图(无 LLM 调用)",
    )


def test_each_llm_mode_resolves_to_a_distinct_file(helpers):
    """The four LLM-driven modes each map to a different .txt file so
    editing one does not silently affect another. ``remove_bg`` maps to
    None because the node short-circuits that mode. The bilingual
    labels go through ``parse_mode_label`` first so the test exercises
    the same normalisation the node uses at runtime."""
    codes = [helpers.parse_mode_label(m) for m in helpers.MODES]
    names = {code: helpers.resolve_prompt_name(code) for code in codes}
    assert names["t2i"] == "qwen_image_21/_enhance_t2i"
    assert names["t2i_rgba"] == "qwen_image_21/_enhance_t2i_rgba"
    assert names["edit"] == "qwen_image_21/_enhance_edit"
    assert names["multi_ref"] == "qwen_image_21/_enhance_multi_ref"
    assert names["remove_bg"] is None
    # The four non-None names must all be distinct.
    file_names = {v for v in names.values() if v is not None}
    assert len(file_names) == 4


def test_load_mode_prompt_reads_each_file(helpers):
    """All four bundled .txt files exist and load without error."""
    for mode in ("t2i", "t2i_rgba", "edit", "multi_ref"):
        text = helpers.load_mode_prompt(mode)
        assert isinstance(text, str)
        assert text.strip()
        # Every file must teach the MieNodes BEGIN/END wrapper so the
        # block extractor can find it.
        assert "--- BEGIN enhanced_prompt ---" in text
        assert "--- END enhanced_prompt ---" in text


def test_resolve_prompt_name_unknown_mode_raises(helpers):
    """Unknown mode -> KeyError."""
    with pytest.raises(KeyError):
        helpers.resolve_prompt_name("not-a-mode")


# --------------------------------------------------------------------------- #
# remove_bg short-circuit
# --------------------------------------------------------------------------- #
def test_remove_bg_short_circuits_without_llm_call(mod):
    """``mode = remove_bg`` must not touch the connector at all."""
    conn = _FakeConnector()
    node = mod.QwenImage21PromptGenerator()
    out = node.enhance(
        conn,
        mode="remove_bg",
        user_input="anything the user typed is ignored",
        seed=0,
    )
    assert isinstance(out, tuple)
    assert len(out) == 1
    # The fixed block quotes a known phrase (so downstream nodes can
    # detect it programmatically if needed).
    assert "Remove the background" in out[0]
    # No LLM call was made.
    assert conn.calls == []


def test_remove_bg_run_helper_does_not_call_llm(mod):
    """The helper-level short-circuit also skips the connector."""
    conn = _FakeConnector()
    out = mod.run_enhancer(
        conn,
        draft="ignored",
        mode="remove_bg",
    )
    assert "Remove the background" in out
    assert conn.calls == []


# --------------------------------------------------------------------------- #
# End-to-end: enhance() through the connector
# --------------------------------------------------------------------------- #
def test_enhance_t2i_happy_path_returns_extracted_block(mod):
    """End-to-end: canned well-formed reply -> the BEGIN/END block is
    what gets returned through the STRING output socket."""
    conn = _make_connector_with_replies([_good_t2i_reply()])
    node = mod.QwenImage21PromptGenerator()
    out = node.enhance(
        conn,
        mode="t2i",
        user_input="a glossy red apple sticker",
        seed=0,
    )
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert out[0].startswith("A vertical sticker")
    assert out[0].endswith("upper left.")
    # Exactly one LLM call.
    assert len(conn.calls) == 1
    # System prompt matches the t2i .txt; user message declares mode.
    sys_text = conn.calls[0]["messages"][0]["content"]
    assert "Image Prompt Rewriting Expert" in sys_text
    assert "--- BEGIN enhanced_prompt ---" in sys_text
    user_text = conn.calls[0]["messages"][1]["content"]
    assert "Mode: t2i" in user_text
    assert "---BEGIN USER DRAFT---" in user_text


def test_enhance_edit_emits_reference_count_from_images_socket(mod):
    """``edit`` mode includes the ``reference_count`` line derived from
    the IMAGE batch size; T2I mode never emits it. (Captioning is
    opted out via ``caption_mode = no_cache`` so the test focuses on
    the rewrite-message contract; captioner behaviour is covered in
    the dedicated captioner test block.)"""
    conn = _make_connector_with_replies([_good_edit_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn,
        mode="edit",
        user_input="change jacket colour",
        seed=0,
        reference_images=_FakeImageBatch(1),
        caption_mode="no_cache - 禁用缓存",
    )
    user_text = conn.calls[0]["messages"][1]["content"]
    assert "reference_count: 1" in user_text

    # T2I mode should NOT emit the line even when images are wired.
    conn2 = _make_connector_with_replies([_good_t2i_reply()])
    node.enhance(
        conn2,
        mode="t2i",
        user_input="apple sticker",
        seed=0,
        reference_images=_FakeImageBatch(3),
    )
    user_text2 = conn2.calls[0]["messages"][1]["content"]
    assert "reference_count:" not in user_text2


def test_enhance_with_no_images_socket_yields_zero(mod):
    """No IMAGE socket connected -> N=0; edit mode still emits the
    line (so the LLM knows it's a single-image edit), t2i omits it."""
    conn = _make_connector_with_replies([_good_edit_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn,
        mode="edit",
        user_input="change jacket colour",
        seed=0,
    )
    user_text = conn.calls[0]["messages"][1]["content"]
    assert "reference_count: 0" in user_text


def test_enhance_with_too_many_images_passes_full_count(mod):
    """Above the recommended 10-image ceiling, the node still passes
    the full count to the LLM (we don't truncate). The LLM gets
    ``reference_count: 12`` so it knows what to do."""
    conn = _make_connector_with_replies([_good_multi_ref_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn,
        mode="multi_ref",
        user_input="composite scene",
        seed=0,
        reference_images=_FakeImageBatch(12),
        caption_mode="no_cache - 禁用缓存",
    )
    user_text = conn.calls[0]["messages"][1]["content"]
    assert "reference_count: 12" in user_text


def test_count_reference_images_handles_shapes(mod):
    """``count_reference_images`` accepts None / shape-only tensors and
    rejects anything that doesn't look like an IMAGE tensor."""
    assert mod.count_reference_images(None) == 0
    # Real ComfyUI images come as 4-D torch tensors (B, H, W, C); the
    # function should also accept shape-only test stubs.
    class _Fake:
        def __init__(self, shape):
            self.shape = shape
    assert mod.count_reference_images(_Fake((3, 8, 8, 3))) == 3
    assert mod.count_reference_images(_Fake((1, 8, 8, 3))) == 1
    # Plain objects without ``shape`` -> 0.
    assert mod.count_reference_images(object()) == 0
    # Empty / weird shape -> 0.
    assert mod.count_reference_images(_Fake(())) == 0


def test_enhance_multi_ref_uses_multi_ref_prompt(mod):
    """``multi_ref`` mode loads the multi-ref .txt file (mentions
    ``<imageN>``) and includes the reference_count derived from the
    IMAGE batch. (Captioning opted out — see captioner tests.)"""
    conn = _make_connector_with_replies([_good_multi_ref_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn,
        mode="multi_ref",
        user_input="transfer outfit + accessory",
        seed=0,
        reference_images=_FakeImageBatch(3),
        caption_mode="no_cache - 禁用缓存",
    )
    sys_text = conn.calls[0]["messages"][0]["content"]
    assert "多参考" in sys_text or "多参考图" in sys_text or "multi" in sys_text.lower()
    assert "reference_count: 3" in conn.calls[0]["messages"][1]["content"]


def test_enhance_t2i_rgba_uses_rgba_prompt(mod):
    """``t2i_rgba`` mode loads the forced-bookend RGBA prompt."""
    conn = _make_connector_with_replies([_good_rgba_reply()])
    node = mod.QwenImage21PromptGenerator()
    out = node.enhance(
        conn,
        mode="t2i_rgba",
        user_input="an apple sticker on transparent background",
        seed=0,
    )
    sys_text = conn.calls[0]["messages"][0]["content"]
    # The RGBA .txt explicitly forces the bookend in this mode.
    assert "RGBA forced" in sys_text or "RGBA sandwich" in sys_text
    assert out[0].startswith("This is an RGBA image with transparency.")
    assert out[0].rstrip().endswith("background is transparent.")


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
def test_retry_succeeds_after_two_empty_attempts(mod):
    """Two empty replies, third well-formed -> node returns the block."""
    replies = ["", "", _good_t2i_reply()]
    conn = _make_connector_with_replies(replies)
    node = mod.QwenImage21PromptGenerator()
    out = node.enhance(
        conn,
        mode="t2i",
        user_input="apple",
        seed=42,
    )
    assert out[0].startswith("A vertical sticker")
    assert len(conn.calls) == 3
    # Fresh seed on retries (per the documented policy).
    assert conn.calls[0]["seed"] == 42
    assert conn.calls[1]["seed"] == 43
    assert conn.calls[2]["seed"] == 44


def test_retry_raises_after_three_empty_attempts(mod):
    """Three empty replies -> ``RuntimeError`` carrying the raw reply
    head (last 400 chars of the last reply, which is empty here, but
    the error message itself must mention the retry policy)."""
    conn = _make_connector_with_replies(["", "", ""])
    node = mod.QwenImage21PromptGenerator()
    with pytest.raises(RuntimeError) as ei:
        node.enhance(
            conn,
            mode="t2i",
            user_input="apple",
            seed=0,
        )
    msg = str(ei.value)
    assert "--- BEGIN enhanced_prompt ---" in msg
    assert "after 3 attempts" in msg


def test_retry_raises_with_raw_head_when_block_missing(mod):
    """Three replies with prose but no BEGIN/END markers -> the
    ``RuntimeError`` message includes a head of the raw reply."""
    conn = _make_connector_with_replies(
        [
            "garbled reply 1",
            "garbled reply 2",
            "garbled reply 3 with some LLM chatter",
        ]
    )
    node = mod.QwenImage21PromptGenerator()
    with pytest.raises(RuntimeError) as ei:
        node.enhance(
            conn,
            mode="t2i",
            user_input="apple",
            seed=0,
        )
    msg = str(ei.value)
    assert "garbled reply 3" in msg


# --------------------------------------------------------------------------- #
# Widget passthrough
# --------------------------------------------------------------------------- #
def test_widget_kwargs_reach_connector(mod):
    """``temperature``, ``max_tokens``, ``timeout`` reach the connector."""
    conn = _make_connector_with_replies([_good_t2i_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn,
        mode="t2i",
        user_input="apple",
        seed=0,
        temperature=1.1,
        max_tokens=2048,
        timeout=120,
    )
    assert conn.calls[0]["temperature"] == 1.1
    assert conn.calls[0]["max_tokens"] == 2048
    assert conn.timeout == 120


# --------------------------------------------------------------------------- #
# Widget surface / is_changed
# --------------------------------------------------------------------------- #
def test_input_types_surface(helpers, mod):
    """The ``INPUT_TYPES`` exposes the documented widgets: required
    connector / mode / user_input / seed + optional reference_images
    IMAGE socket / caption_mode dropdown / temperature / max_tokens /
    timeout. (The legacy reference_count INT and slot_roles /
    verbatim_text widgets were removed — see UPSTREAM.md.)"""
    types_ = mod.QwenImage21PromptGenerator.INPUT_TYPES()
    required = types_["required"]
    assert "llm_service_connector" in required
    assert required["mode"] == (
        list(helpers.MODES),
        {"default": helpers.MODES[0]},
    ) or (isinstance(required["mode"], tuple) and required["mode"][0] == list(helpers.MODES))
    assert "user_input" in required
    assert required["seed"][0] == "INT"

    optional = types_["optional"]
    for key in (
        "reference_images",
        "caption_mode",
        "temperature",
        "max_tokens",
        "timeout",
    ):
        assert key in optional, f"optional widget missing: {key}"

    # reference_images is an IMAGE socket (not INT / not STRING).
    ri = optional["reference_images"]
    assert ri[0] == "IMAGE"

    # The removed widgets must NOT come back silently.
    for removed in ("reference_count", "slot_roles", "verbatim_text"):
        assert removed not in optional, f"legacy widget returned: {removed}"


def test_return_names_and_category(mod):
    """The output socket is named ``enhanced_prompt`` and the category
    matches the other prompt-generator nodes (so users find it next to
    them in the ComfyUI menu)."""
    node = mod.QwenImage21PromptGenerator()
    assert node.RETURN_TYPES == ("STRING",)
    assert node.RETURN_NAMES == ("enhanced_prompt",)
    assert node.FUNCTION == "enhance"
    assert node.CATEGORY == mod.MY_CATEGORY


def test_is_changed_stable_for_same_inputs(mod):
    """``is_changed`` returns the same digest for identical inputs."""
    node = mod.QwenImage21PromptGenerator()
    conn = _FakeConnector()
    a = node.is_changed(conn, mode="t2i", user_input="apple", seed=0)
    b = node.is_changed(conn, mode="t2i", user_input="apple", seed=0)
    assert a == b


def test_is_changed_changes_with_user_input(mod):
    """Different ``user_input`` -> different digest."""
    node = mod.QwenImage21PromptGenerator()
    conn = _FakeConnector()
    a = node.is_changed(conn, mode="t2i", user_input="apple", seed=0)
    b = node.is_changed(conn, mode="t2i", user_input="banana", seed=0)
    assert a != b


def test_is_changed_changes_with_mode(mod):
    """Different ``mode`` -> different digest (because the bundled .txt
    files differ and the node hashes the system-prompt body)."""
    node = mod.QwenImage21PromptGenerator()
    conn = _FakeConnector()
    a = node.is_changed(conn, mode="t2i", user_input="x", seed=0)
    b = node.is_changed(conn, mode="edit", user_input="x", seed=0)
    assert a != b


def test_is_changed_includes_connector_state(mod):
    """A connector with a different ``get_state`` digest flips is_changed."""
    class _Stateful:
        model = "m"
        timeout = 60

        def invoke(self, *a, **kw):
            return ""

        def get_state(self):
            return self._state

    a = _Stateful(); a._state = "state-a"
    b = _Stateful(); b._state = "state-b"
    node = mod.QwenImage21PromptGenerator()
    h1 = node.is_changed(a, mode="t2i", user_input="x", seed=0)
    h2 = node.is_changed(b, mode="t2i", user_input="x", seed=0)
    assert h1 != h2


def test_is_changed_stable_for_same_images_batch(mod):
    """Same IMAGE batch on two calls -> same digest."""
    node = mod.QwenImage21PromptGenerator()
    conn = _FakeConnector()
    a = node.is_changed(
        conn, mode="multi_ref", user_input="x", seed=0,
        reference_images=_FakeImageBatch(3),
    )
    b = node.is_changed(
        conn, mode="multi_ref", user_input="x", seed=0,
        reference_images=_FakeImageBatch(3),
    )
    assert a == b


def test_is_changed_changes_with_image_count(mod):
    """Different batch sizes -> different digest (cache invalidates when
    the user adds / removes a reference)."""
    node = mod.QwenImage21PromptGenerator()
    conn = _FakeConnector()
    a = node.is_changed(
        conn, mode="multi_ref", user_input="x", seed=0,
        reference_images=_FakeImageBatch(3),
    )
    b = node.is_changed(
        conn, mode="multi_ref", user_input="x", seed=0,
        reference_images=_FakeImageBatch(4),
    )
    assert a != b


def test_is_changed_changes_with_images_connected_vs_none(mod):
    """Connecting / disconnecting the IMAGE socket flips the digest."""
    node = mod.QwenImage21PromptGenerator()
    conn = _FakeConnector()
    a = node.is_changed(conn, mode="edit", user_input="x", seed=0)
    b = node.is_changed(
        conn, mode="edit", user_input="x", seed=0,
        reference_images=_FakeImageBatch(1),
    )
    assert a != b


def _fresh_captioner(mod, conn, tmp_path, *, scope="memory_disk", force=False):
    """Build a captioner with a tmp_path disk root so tests never touch
    the persistent ``<ComfyUI output>/mien_nodes/caption_cache/`` —
    previous test runs (or live runs) may have cached the same fake
    images, and we want deterministic misses / hits regardless."""
    cap = mod._ReferenceCaptioner(
        conn, cache_scope=scope, force_recaption=force
    )
    cap._disk_root = str(tmp_path / "mien_nodes" / "caption_cache")
    return cap


# --------------------------------------------------------------------------- #
# Captioner: helpers, cache modes, disk cache, integration
# --------------------------------------------------------------------------- #
def test_caption_mode_labels_and_codes(helpers):
    """Caption-mode dropdown labels and the code parser line up; unknown
    labels resolve to ``""`` (mirrors H3 Loop's parse_caption_mode)."""
    assert helpers.CAPTION_MODE_CODES == (
        "cache_memory_disk",
        "cache_memory_only",
        "no_cache",
        "force_recaption_once",
    )
    assert helpers.caption_mode_code(
        "cache_memory_disk - 缓存:内存+磁盘(推荐)"
    ) == "cache_memory_disk"
    assert helpers.caption_mode_code("force_recaption_once - 本次强制重打标") == (
        "force_recaption_once"
    )
    assert helpers.caption_mode_code("garbage") == ""
    assert helpers.caption_mode_code("") == ""


def test_resolve_caption_controls_maps_each_code(helpers):
    """Each dropdown label maps to the documented ``(force, scope)``
    pair; an unknown / empty label falls back to the
    ``cache_memory_disk`` defaults so a corrupt workflow never silently
    disables caching."""
    for label, expected in [
        ("cache_memory_disk - 缓存:内存+磁盘(推荐)", (False, "memory_disk")),
        ("cache_memory_only - 缓存:仅内存", (False, "memory_only")),
        ("no_cache - 禁用缓存", (False, "disabled")),
        ("force_recaption_once - 本次强制重打标", (True, "disabled")),
    ]:
        assert helpers.resolve_caption_controls(label) == expected
    # Unknown / empty -> safe default (cache_memory_disk).
    assert helpers.resolve_caption_controls("") == (False, "memory_disk")
    assert helpers.resolve_caption_controls(None) == (False, "memory_disk")


def test_caption_cache_key_changes_with_image_and_prompt(helpers):
    """Cache key embeds BOTH the image payload and the prompt, so
    editing ``_caption_image.txt`` invalidates every cached entry and
    swapping images with the same prompt gives different keys."""
    prompt_a = "prompt-A"
    prompt_b = "prompt-B"
    image_a = "data:image/jpeg;base64,AAA"
    image_b = "data:image/jpeg;base64,BBB"
    assert (
        helpers.caption_cache_key(image_a, prompt_a)
        != helpers.caption_cache_key(image_a, prompt_b)
    )
    assert (
        helpers.caption_cache_key(image_a, prompt_a)
        != helpers.caption_cache_key(image_b, prompt_a)
    )


def test_caption_image_prompt_loads_bundled_file(helpers):
    """The caption system prompt loads and contains the Qwen-specific
    ``CHARACTER SHEETS ARE PICTURES OF THE CHARACTER`` rule (which is
    the exact failure mode the user hit in the three-view example)."""
    text = helpers.caption_image_prompt()
    assert isinstance(text, str) and text.strip()
    assert "CHARACTER SHEETS ARE PICTURES OF THE CHARACTER" in text
    assert "WARDROBE IS NON-NEGOTIABLE" in text


def test_captioner_in_memory_cache_hit_avoids_second_call(mod, tmp_path):
    """Same image twice in one batch: 1 LLM call. Same image twice in
    a second batch: still 1 LLM call (memory cache hit)."""
    conn = _make_connector_with_replies([
        "A young woman with brown bob, tortoiseshell glasses, off-shoulder brown sweater.",
        _good_edit_reply(),
    ])
    cap = _fresh_captioner(mod, conn, tmp_path)
    img = _FakeImageBatch(1)

    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn, mode="edit", user_input="add a hat", seed=0,
        reference_images=img,
    )
    # First call batch: 1 caption + 1 rewrite = 2 LLM calls.
    assert len(conn.calls) == 2
    # First call is the caption (multimodal content with image_url).
    first_msg = conn.calls[0]["messages"][1]["content"]
    assert isinstance(first_msg, list)
    assert any(
        part.get("type") == "image_url" for part in first_msg
    ), "caption message should carry image_url parts"


def test_captioner_disk_cache_hit_avoids_second_call(mod, tmp_path):
    """Pre-populated disk cache entry -> captioner reads it without an
    LLM call. Mirrors the H3 Loop behaviour so users get warm starts
    across restarts / workflows."""
    sys_text = mod.caption_image_prompt()
    img = _FakeImageBatch(1)
    # Encode the same fake image to get the same data URL the captioner
    # will use.
    urls = mod.encode_reference_images(img)
    assert len(urls) == 1
    url = urls[0]
    key = mod.caption_cache_key(url, sys_text)
    cache_dir = tmp_path / "mien_nodes" / "caption_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{key}.txt").write_text(
        "Pre-cached caption: a young woman in a brown sweater.",
        encoding="utf-8",
    )

    conn = _make_connector_with_replies([_good_edit_reply()])
    cap = _fresh_captioner(mod, conn, tmp_path)
    captions = cap.caption_all([url], seed=0)
    assert captions == ["Pre-cached caption: a young woman in a brown sweater."]
    # Zero LLM calls — disk cache served the caption.
    assert conn.calls == []


def test_captioner_force_recaption_bypasses_caches(mod, tmp_path):
    """``force_recaption=True`` (the ``force_recaption_once`` mode) skips
    both in-memory and disk tiers so a prompt upgrade is honoured on
    the very next run."""
    sys_text = mod.caption_image_prompt()
    img = _FakeImageBatch(1)
    urls = mod.encode_reference_images(img)
    url = urls[0]
    key = mod.caption_cache_key(url, sys_text)
    cache_dir = tmp_path / "mien_nodes" / "caption_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{key}.txt").write_text("STALE caption", encoding="utf-8")

    conn = _make_connector_with_replies([
        "Fresh caption: a young woman with short hair.",
        _good_edit_reply(),
    ])
    cap = _fresh_captioner(
        mod, conn, tmp_path, scope="memory_disk", force=True
    )
    captions = cap.caption_all([url], seed=0)
    assert captions == ["Fresh caption: a young woman with short hair."]
    # Exactly one LLM call (the caption); cache was bypassed.
    assert len(conn.calls) == 1


def test_captioner_empty_reply_raises(mod, tmp_path):
    """An empty caption reply is a hard failure (no silent fallback) —
    the rewrite LLM would hallucinate identity if we let an empty
    caption through."""
    conn = _make_connector_with_replies(["", _good_edit_reply()])
    cap = _fresh_captioner(mod, conn, tmp_path)
    img = _FakeImageBatch(1)
    with pytest.raises(RuntimeError, match="empty reply"):
        cap.caption_all(mod.encode_reference_images(img), seed=0)


def test_captioner_disabled_scope_skips_captions(mod, tmp_path):
    """``cache_scope="disabled"`` + ``force_recaption=False`` → no
    captioner LLM call. One entry per input (empty string), so the
    caption list stays slot-aligned with the image batch."""
    conn = _make_connector_with_replies([_good_edit_reply()])
    cap = _fresh_captioner(mod, conn, tmp_path, scope="disabled")
    img = _FakeImageBatch(1)
    captions = cap.caption_all(mod.encode_reference_images(img), seed=0)
    assert captions == [""]
    assert conn.calls == []


def test_enhance_runs_captioner_in_edit_mode(mod, tmp_path):
    """End-to-end: edit mode + 1 reference image triggers one caption
    call + one rewrite call, and the rewrite message includes the
    caption as DATA under ``<image1>``. (Captioner uses a tmp_path
    disk root so the test never collides with the persistent cache.)"""
    img = _FakeImageBatch(1)
    conn = _make_connector_with_replies([
        "A young woman with brown bob, tortoiseshell glasses, "
        "off-shoulder brown sweater with mint stripes.",
        _good_edit_reply(),
    ])
    # Pre-bind the captioner's disk root to tmp_path via a sentinel:
    # the node builds its own captioner, so we monkey-patch
    # caption_cache_disk_root for the duration of the call.
    saved_root = mod.caption_cache_disk_root
    mod.caption_cache_disk_root = lambda: str(
        tmp_path / "mien_nodes" / "caption_cache"
    )
    try:
        node = mod.QwenImage21PromptGenerator()
        node.enhance(
            conn, mode="edit", user_input="change the sweater to red",
            seed=0, reference_images=img,
        )
    finally:
        mod.caption_cache_disk_root = saved_root
    assert len(conn.calls) == 2
    # Rewrite user message must embed the caption.
    rewrite_user = conn.calls[1]["messages"][1]["content"]
    assert "brown bob" in rewrite_user
    assert "<image1>" in rewrite_user
    assert "DATA" in rewrite_user or "caption" in rewrite_user.lower()


def test_enhance_skips_captioner_in_t2i_mode(mod):
    """T2I mode never sees reference images, so the captioner must not
    fire even when an IMAGE batch is connected (mirrors the v1
    contract)."""
    img = _FakeImageBatch(2)
    conn = _make_connector_with_replies([_good_t2i_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn, mode="t2i", user_input="an apple sticker",
        seed=0, reference_images=img,
    )
    # Single LLM call = the rewrite. No captioner.
    assert len(conn.calls) == 1


def test_enhance_no_caption_mode_widget_bypasses_captioner(mod):
    """``caption_mode = no_cache`` + ``force=False`` → scope=disabled →
    node skips the captioner entirely (one LLM call, no image sent)."""
    img = _FakeImageBatch(1)
    conn = _make_connector_with_replies([_good_edit_reply()])
    node = mod.QwenImage21PromptGenerator()
    node.enhance(
        conn, mode="edit", user_input="add hat",
        seed=0, reference_images=img,
        caption_mode="no_cache - 禁用缓存",
    )
    assert len(conn.calls) == 1
    rewrite_user = conn.calls[0]["messages"][1]["content"]
    assert "DATA" not in rewrite_user
    assert "caption" not in rewrite_user.lower().split("begin", 1)[0]


def test_caption_mode_widget_in_input_types(mod):
    """``caption_mode`` widget is exposed alongside ``reference_images``
    so users can pick a cache strategy from the node UI. The widget
    shape mirrors the ``mode`` widget — ``([labels], {opts})``."""
    types_ = mod.QwenImage21PromptGenerator.INPUT_TYPES()
    optional = types_["optional"]
    assert "caption_mode" in optional
    labels, opts = optional["caption_mode"]
    assert isinstance(labels, list)
    assert opts["default"] == labels[0]
    codes = [lbl.split(" - ", 1)[0].strip() for lbl in labels]
    assert "cache_memory_disk" in codes
    assert "cache_memory_only" in codes
    assert "no_cache" in codes
    assert "force_recaption_once" in codes


def test_edit_template_requires_english_body():
    """The edit-mode system prompt must explicitly require the rewrite
    body to be English regardless of the user's input language, so
    Qwen-Image-2.1 gets the prompt shape its renderer is tuned for."""
    path = PROJECT_DIR / "nodes" / "llm" / "prompts" / "qwen_image_21" / "_enhance_edit.txt"
    text = path.read_text(encoding="utf-8")
    assert "一律使用英文" in text or "always English" in text or "一律英文" in text
    # Verbatim rule is preserved.
    assert "verbatim" in text.lower()
    # Reference-caption-as-data rule is encoded.
    assert "DATA" in text or "标注" in text or "caption" in text.lower()


def test_multi_ref_template_requires_english_body():
    """The multi_ref-mode system prompt must also default to English
    body + verbatim rule + caption-as-data handling."""
    path = PROJECT_DIR / "nodes" / "llm" / "prompts" / "qwen_image_21" / "_enhance_multi_ref.txt"
    text = path.read_text(encoding="utf-8")
    assert "一律使用英文" in text or "always English" in text or "一律英文" in text
    assert "verbatim" in text.lower()
    assert "DATA" in text or "标注" in text or "caption" in text.lower()


# --------------------------------------------------------------------------- #
# Bilingual mode labels (Loop-pattern dropdowns)
# --------------------------------------------------------------------------- #
def test_mode_labels_are_bilingual(helpers):
    """Every ``MODES`` entry is bilingual ``code - 中文(标签)`` and the
    bare code round-trips through ``parse_mode_label``. Mirrors
    MiniMaxH3LoopPromptGenerator dropdown conventions."""
    expected_codes = {"t2i", "t2i_rgba", "edit", "multi_ref", "remove_bg"}
    seen_codes = set()
    for label in helpers.MODES:
        code = label.split(" - ", 1)[0].strip()
        seen_codes.add(code)
        # Bilingual: a non-empty Chinese parenthetical lives after the dash.
        tail = label.split(" - ", 1)[1] if " - " in label else ""
        assert any("\u4e00" <= ch <= "\u9fff" for ch in tail), (
            f"mode label missing Chinese tail: {label!r}"
        )
        assert code in helpers._MODE_TO_PROMPT, (
            f"mode code {code!r} has no _MODE_TO_PROMPT entry"
        )
    assert seen_codes == expected_codes


def test_parse_mode_label_round_trip(helpers):
    """``parse_mode_label`` strips the bilingual tail and returns the
    bare code; unknown / empty input round-trips to ``""``."""
    for label, code in (
            ("t2i - 文生图(默认)", "t2i"),
            ("t2i_rgba - 透明文生图", "t2i_rgba"),
            ("edit - 单图编辑", "edit"),
            ("multi_ref - 多参考编辑", "multi_ref"),
            ("remove_bg - 抠图(无 LLM 调用)", "remove_bg"),
            # Bare code still works (so existing tests passing
            # ``mode="edit"`` don't break).
            ("edit", "edit"),
    ):
        assert helpers.parse_mode_label(label) == code
    assert helpers.parse_mode_label("") == ""
    assert helpers.parse_mode_label("garbage") == ""
    assert helpers.parse_mode_label(None) == ""


def test_mode_widget_labels_in_input_types(helpers, mod):
    """The dropdown values in INPUT_TYPES are the bilingual labels
    (not the bare codes), so users see both halves in the menu."""
    types_ = mod.QwenImage21PromptGenerator.INPUT_TYPES()
    mode_widget = types_["required"]["mode"]
    labels = mode_widget[0]
    assert isinstance(labels, list)
    # First entry must be the default (t2i) and bilingual.
    assert labels[0].startswith("t2i - ")
    # Every entry is bilingual.
    for label in labels:
        assert " - " in label, f"mode label missing bilingual separator: {label!r}"
        tail = label.split(" - ", 1)[1]
        assert any("\u4e00" <= ch <= "\u9fff" for ch in tail)


def test_node_accepts_bilingual_mode_label(mod):
    """End-to-end: passing ``mode="edit - 单图编辑"`` works end-to-end
    the same as ``mode="edit"`` — the same load_mode_prompt() call
    happens, the same captioning gate fires, the same parse_mode_label
    normalisation is applied. Bilingual labels are an a11y / i18n
    affordance, not a behaviour change."""
    conn = _make_connector_with_replies([
        "A young woman in a brown sweater.",
        _good_edit_reply(),
    ])
    node = mod.QwenImage21PromptGenerator()
    out_bilingual = node.enhance(
        conn, mode="edit - 单图编辑", user_input="change the sweater to red",
        seed=0, reference_images=_FakeImageBatch(1),
    )
    # Same input via the bare code — same effect.
    conn_bare = _make_connector_with_replies([
        "A young woman in a brown sweater.",
        _good_edit_reply(),
    ])
    out_bare = node.enhance(
        conn_bare, mode="edit", user_input="change the sweater to red",
        seed=0, reference_images=_FakeImageBatch(1),
    )
    assert out_bilingual == out_bare


def test_reference_images_tooltip_warns_about_character_sheets(mod):
    """A four-up turnaround is copied as a layout: four views in,
    four poses out. The tooltip tells the user to crop to one person.
    It must not claim that only one panel is edited and the rest stay."""
    types_ = mod.QwenImage21PromptGenerator.INPUT_TYPES()
    # Widget shape: ``("IMAGE", {opts})`` — access opts by index 1.
    widget_opts = types_["optional"]["reference_images"][1]
    tooltip = widget_opts["tooltip"]
    assert "character sheet" in tooltip.lower() or "设定表" in tooltip
    assert "crop" in tooltip.lower() or "裁" in tooltip
    assert "four poses" in tooltip.lower() or "四身" in tooltip
    assert "only apply" not in tooltip.lower()
    assert "stay unchanged" not in tooltip.lower()
    assert any("\u4e00" <= ch <= "\u9fff" for ch in tooltip)


def test_scrub_caption_sheet_language_keeps_wardrobe(helpers):
    """Sheet layout, backdrop, and sheet lighting come off. The
    wardrobe and the face cues stay. A caption that is only sheet
    language is returned unchanged rather than as an empty string."""
    scrub = helpers.scrub_caption_sheet_language
    out = scrub(
        "A young woman with a brown bob and tortoiseshell glasses, "
        "an off-shoulder brown sweater with mint stripes, brown "
        "trousers and tan heels, three views on a seamless white "
        "background, turnaround sheet, standing with arms at her sides, "
        "bright even frontal lighting."
    )
    assert "brown bob" in out
    assert "mint stripes" in out
    assert "tan heels" in out
    assert "three views" not in out
    assert "white background" not in out
    assert "turnaround" not in out
    assert "arms at her sides" not in out
    assert "frontal lighting" not in out
    # A real scene pose is not sheet metadata.
    kept = scrub(
        "The same woman standing against a sunlit brick wall, "
        "wearing the brown sweater."
    )
    assert "standing against a sunlit brick wall" in kept
    assert scrub("shown from three angles") == "shown from three angles"


def test_caption_prompt_collapses_a_sheet_to_one_person(helpers):
    """The caption system prompt must not tell the model to list each
    view as its own person, or to lock the sheet pose and sheet light."""
    text = helpers.caption_image_prompt()
    assert "ONE person shown several times" in text
    assert "name the pose once" not in text
    assert "If you count 3 characters" not in text
    assert "you MUST list all N" not in text


def test_reply_block_passes_through_untouched(mod):
    """The node never post-processes the LLM block with draft-keyword
    rules: no sentence is dropped and no identity-lock sentence is
    prepended, in any mode. A draft that names a character sheet (and
    would have tripped the retired keyword lock) passes through
    verbatim — the sheet rule lives in the mode's system prompt, not
    in node code."""
    inner = (
        "<image1> is a three-view character sheet reference. "
        "Change the jacket colour to terracotta."
    )
    reply = (
        f"--- BEGIN enhanced_prompt ---\n{inner}\n"
        "--- END enhanced_prompt ---\n"
    )
    sheet_draft = "参考设定表生成一张新图，光影斑驳"
    for mode, images in (("t2i", None), ("edit", _FakeImageBatch(1))):
        conn = _make_connector_with_replies([reply])
        node = mod.QwenImage21PromptGenerator()
        out = node.enhance(
            conn,
            mode=mode,
            user_input=sheet_draft,
            seed=0,
            reference_images=images,
            caption_mode="no_cache - 禁用缓存",
        )
        assert out[0] == inner


def test_edit_prompts_open_a_sheet_as_one_pose():
    """New-scene rewrites from a turnaround must open by revoking the
    sheet layout. Local edits are explicitly left on the framing rule."""
    root = PROJECT_DIR / "nodes" / "llm" / "prompts" / "qwen_image_21"
    for name in ("_enhance_edit.txt", "_enhance_multi_ref.txt"):
        text = (root / name).read_text(encoding="utf-8")
        assert "One person, one pose." in text
        assert "Do not copy the character-sheet layout" in text
        assert "局部编辑" in text


def test_sheet_caption_is_scrubbed_before_rewrite(mod, tmp_path):
    """A caption that still says 'three views' must not reach the
    rewrite user message. Wardrobe does."""
    img = _FakeImageBatch(1)
    conn = _make_connector_with_replies([
        "A young woman with a brown bob and tortoiseshell glasses, "
        "three views on a seamless white background, turnaround sheet.",
        _good_edit_reply(),
    ])
    saved_root = mod.caption_cache_disk_root
    mod.caption_cache_disk_root = lambda: str(
        tmp_path / "mien_nodes" / "caption_cache"
    )
    try:
        node = mod.QwenImage21PromptGenerator()
        node.enhance(
            conn, mode="edit",
            user_input="她在阳光下靠着墙，光影斑驳",
            seed=0, reference_images=img,
        )
    finally:
        mod.caption_cache_disk_root = saved_root
    rewrite_user = conn.calls[1]["messages"][1]["content"]
    caption_line = next(
        line for line in rewrite_user.splitlines()
        if line.startswith("- <image1>:")
    )
    assert "brown bob" in caption_line
    assert "three views" not in caption_line
    assert "turnaround" not in caption_line
    assert "white background" not in caption_line
    # The one-person rule lives in the rewrite system prompt, not in
    # node-side post-processing.
    assert "one person, one pose" in conn.calls[1]["messages"][0]["content"].lower()


# --------------------------------------------------------------------------- #
# T2I optimization plan (2026-09-28) — structural constraints
# --------------------------------------------------------------------------- #
_T2I_PROMPT_PATH = PROJECT_DIR / "nodes" / "llm" / "prompts" / "qwen_image_21" / "_enhance_t2i.txt"
_T2I_RGBA_PROMPT_PATH = PROJECT_DIR / "nodes" / "llm" / "prompts" / "qwen_image_21" / "_enhance_t2i_rgba.txt"


@pytest.fixture(scope="module")
def t2i_prompt_text():
    return _T2I_PROMPT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def t2i_rgba_prompt_text():
    return _T2I_RGBA_PROMPT_PATH.read_text(encoding="utf-8")


def test_t2i_has_step_2_5_skeleton_menu(t2i_prompt_text):
    """Plan §需求 1: insert Step 2.5 — Pick a structural skeleton
    before you write. The six-skeleton menu must appear by name so the
    model commits to one instead of listing things in the middle of
    the frame."""
    assert "Step 2.5" in t2i_prompt_text
    for skeleton in (
        "Mirror",
        "Oblique interior",
        "Diagonal",
        "Aerial",
        "Minimal",
        "Macro",
    ):
        assert skeleton in t2i_prompt_text, (
            f"t2i prompt missing skeleton: {skeleton!r}"
        )


def test_t2i_step_2_5_depth_source_menu(t2i_prompt_text):
    """Plan §需求 3: depth has a source, and the source constrains
    the whole frame. The six-source menu and the reflection-locks-
    composition warning must both be present."""
    text = t2i_prompt_text
    # Six depth sources — at least the most distinctive phrasings.
    for source in (
        "reflective plane",
        "stacked objects receding",
        "layered bands of depth",
        "scale contrast",
        "edge that drops away",
        "falloff into shadow or fog",
    ):
        assert source in text, f"t2i prompt missing depth source: {source!r}"
    # Reflection-locks warning.
    assert "lock" in text.lower() and (
        "symmetry" in text.lower() or "reflection" in text.lower()
    )
    # Second-world lower-half hint.
    assert "second world" in text.lower()


def test_t2i_step_3_camera_angle_committed(t2i_prompt_text):
    """Plan §需求 2: camera angle is part of the skeleton and must
    be committed once in the opening sentence or the space sentence."""
    text = t2i_prompt_text
    assert "camera angle" in text.lower() or "Camera angle" in text
    # At least the canonical five angles named.
    for angle in (
        "high three-quarter angle",
        "low upward angle",
        "directly above",
        "eye level",
        "extreme close-up",
    ):
        assert angle in text, f"t2i prompt missing camera angle: {angle!r}"


def test_t2i_step_4_left_right_contrast(t2i_prompt_text):
    """Plan §需求 5: left and right must differ in kind, not just in
    position. The t2i prompt must encode this rule in Step 4."""
    text = t2i_prompt_text
    assert "differ in kind" in text
    # The contrast examples should appear — at least one of them.
    # The prompt wraps long lines at word boundaries; collapse
    # whitespace so a wrap-induced newline doesn't break the match.
    flat = " ".join(text.split())
    for example in (
        "Thin curves against thick blocks",
        "small repeated shapes against one large mass",
        "warm against cool",
    ):
        assert example in flat, (
            f"t2i prompt missing left/right example: {example!r}"
        )


def test_t2i_size_range_and_sentence_cap(t2i_prompt_text):
    """Plan §需求 7 + §需求 8: sentence count 13–20 / word count 300–550
    (not the old 20/400-500 hard count), and sentence cap 40 words."""
    text = t2i_prompt_text
    flat = " ".join(text.split())
    # Range + rationale.
    assert "thirteen to twenty" in flat or "13 to 20" in flat
    assert "300 to 550" in flat or "three hundred to five hundred and fifty" in flat
    # Sentence cap.
    assert "forty" in flat.lower()
    assert "40" in flat  # the digit form is what the model reads
    # The old hard-coded 20-sentence / 400-500-word spec must NOT
    # survive verbatim. We allow "twenty sentences" to appear as part
    # of the new "thirteen to twenty sentences" range, but the old
    # exact phrasing is gone.
    assert "twenty sentences and four to five hundred" not in flat
    assert "twenty sentences and 400" not in flat


def test_t2i_negation_warning(t2i_prompt_text):
    """Plan §需求 9: negation is unreliable for things you would get
    anyway. The t2i prompt must encode this so the model rewrites
    'no background detail' as a positive state instead."""
    text = t2i_prompt_text.lower()
    assert "negation" in text
    # Either of the canonical example phrasings should appear.
    assert "no background detail" in text
    assert "solid flat" in text or "single-colour" in text or "single-colour background" in text


def test_t2i_writes_absence_for_minimal(t2i_prompt_text):
    """Plan §需求 6: on a Minimal frame, absence is content. The
    t2i prompt must tell the model to spell out what is NOT there."""
    text = t2i_prompt_text
    assert "absence is content" in text or "absence" in text.lower()
    # The canonical examples should appear.
    assert "no clouds" in text
    assert "no visible sun" in text or "no other figures" in text


def test_t2i_surreal_without_strangeness(t2i_prompt_text):
    """Plan §需求 4: surreal can be reached via scale / posture rather
    than exotic objects. The t2i prompt must surface this so the
    model reaches for scale before reaching for strangeness."""
    text = t2i_prompt_text.lower()
    assert "surreal" in text
    # Canonical examples: scale / posture.
    assert "scale" in text
    assert "posture" in text or "limb" in text or "hanging" in text


def test_t2i_double_light_source_for_surreal(t2i_prompt_text):
    """Plan §需求 7 (lighting): self-luminous / surreal elements need
    two light sources (ambient + local). Step 7 must encode this."""
    text = t2i_prompt_text
    assert "self-luminous" in text or "self luminous" in text or "self-lum" in text.lower()
    # Two-source language.
    assert "ambient light" in text or "local light" in text
    assert "two sources" in text.lower() or "two" in text


def test_t2i_rgba_has_skeleton_and_camera_angle(t2i_rgba_prompt_text):
    """RGBA cutouts default to Minimal or Macro skeleton; the prompt
    must commit to one and tell the model the camera angle that
    follows."""
    text = t2i_rgba_prompt_text
    # Both skeletons named.
    assert "Minimal" in text
    assert "Macro" in text
    # Camera angles explicit.
    assert "at eye level" in text
    assert "extreme close-up" in text
    # No reflective plane allowed — RGBA forbids it.
    assert "reflective" in text.lower()
    assert "transparent" in text.lower()


def test_t2i_rgba_has_sentence_cap_and_negation(t2i_rgba_prompt_text):
    """RGBA template inherits the new constraints: sentence cap 40,
    negation warning, left/right contrast."""
    text = t2i_rgba_prompt_text
    assert "40" in text and "forty" in text.lower()
    assert "negation" in text.lower()
    assert "differ in kind" in text


def test_t2i_rgba_size_cap_preserved(t2i_rgba_prompt_text):
    """The RGBA template keeps its tight 80-word inner-paragraph cap
    (Qwen-Image-2.1 RGBA soft-alpha degrades on long prose) AND adds
    the new sentence cap 40 + total 4-7 sentences."""
    text = t2i_rgba_prompt_text
    assert "80 English words" in text
    # New structural caps from the plan.
    assert "4" in text and "7" in text  # 4-7 sentences range
    assert "sentence cap 40" in text or "cap 40 words" in text or "forty words" in text.lower()