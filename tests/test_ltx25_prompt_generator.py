# -*- coding: utf-8 -*-
"""Tests for the LTX25PromptEnhancer behavior and the node's contract.

Mirrors the patterns from ``test_krea2_prompts.py`` (import via the
``PACKAGE_NAME`` namespace) and ``test_h3_prompt_generator.py``
(monkeypatched ``image_tensor_batch_to_data_urls``).

Coverage:
* ComfyUI node contract: INPUT_TYPES / RETURN_TYPES / CATEGORY and
  plugin-root registration as ``LTX25PromptGenerator|Mie``. No
  encoder_family input -- LTX-2.5 is gemma4-only.
* ``postprocess_caption`` strips a leading ``<think>`` block.
* t2v: single invoke, gemma4 t2v system prompt, official
  ``"user prompt: ..."`` user turn, temperature forwarded.
* i2v: single multimodal invoke with the first frame attached (same
  single-stage shape as Bernini / H3 i2v) and the official
  ``"User Raw Input Prompt: ...."`` user turn.
* Graceful degradation: missing image / unknown mode / empty prompt.
* ``is_changed`` stability and sensitivity, including the i2v
  first-frame media signature (tensor shape).
"""
import importlib
import re
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "tests"))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME  # noqa: E402


@pytest.fixture(scope="module")
def ltx25():
    load_plugin_module()
    prompts = importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ltx25_prompts")
    gen = importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ltx25_prompt_generator")
    return prompts, gen


class FakeConnector:
    """Minimal connector double: exposes ``get_state`` + ``model`` so
    ``is_changed`` and ``mie_log`` work without a real LLM backend."""

    model = "fake-model"

    def get_state(self):
        return "fake-state"


class _FakeTensor:
    """Stand-in for a ComfyUI IMAGE tensor; the data-URL conversion is
    monkeypatched in the tests that need it."""


def _image_parts(user_content):
    """Return the image_url parts of a multimodal user turn."""
    return [
        p for p in user_content
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]


def _text_part(user_content):
    """Concatenate the text parts of a multimodal user turn, skipping
    the small ``[Image N]:`` labels that precede image parts."""
    return "".join(
        p.get("text", "")
        for p in user_content
        if isinstance(p, dict)
        and p.get("type") == "text"
        and not re.fullmatch(r"\s*\[Image \d+\]:", p.get("text", ""))
    )


# --------------------------------------------------------------------------- #
# ComfyUI node contract
# --------------------------------------------------------------------------- #
def test_comfyui_node_input_types(ltx25):
    _, gen = ltx25
    spec = gen.LTX25PromptGenerator.INPUT_TYPES()
    required = spec["required"]
    assert "llm_service_connector" in required
    assert "mode" in required
    assert "user_prompt" in required
    assert "seed" in required
    # No encoder_family: LTX-2.5 is gemma4-only.
    assert "encoder_family" not in required
    # Dropdown default: t2v.
    assert required["mode"][1]["default"].startswith("t2v")
    # user_prompt is a multiline STRING widget.
    up = required["user_prompt"]
    assert up[0] == "STRING"
    assert up[1].get("multiline") is True
    optional = spec["optional"]
    assert optional["image"][0] == "IMAGE"
    assert optional["image_detail"][0] == ["auto", "low", "high"]
    # Plain temperature widget: default 0.8 (LTX2 sibling), min 0.0.
    temp = optional["temperature"]
    assert temp[0] == "FLOAT"
    assert temp[1]["default"] == 0.8
    assert temp[1]["min"] == 0.0
    assert "max_tokens" in optional
    assert "timeout" in optional


def test_comfyui_node_return_types(ltx25):
    _, gen = ltx25
    assert gen.LTX25PromptGenerator.RETURN_TYPES == ("STRING",)
    assert gen.LTX25PromptGenerator.RETURN_NAMES == ("ltx25_prompt",)
    assert "Prompt Generator" in gen.LTX25PromptGenerator.CATEGORY


def test_node_registered_in_plugin(ltx25):
    plugin = load_plugin_module()
    assert "LTX25PromptGenerator|Mie" in plugin.NODE_CLASS_MAPPINGS
    display = plugin.NODE_DISPLAY_NAME_MAPPINGS
    assert "LTX25PromptGenerator|Mie" in display
    assert "LTX2.5" in display["LTX25PromptGenerator|Mie"]


# --------------------------------------------------------------------------- #
# postprocess
# --------------------------------------------------------------------------- #
def test_postprocess_caption_strips_think_block(ltx25):
    _, gen = ltx25
    assert gen.postprocess_caption("") == ""
    assert gen.postprocess_caption(None) == ""
    assert (
        gen.postprocess_caption("<think>plan...</think>\nA caption.")
        == "A caption."
    )
    assert gen.postprocess_caption("  A caption.  ") == "A caption."
    # A think block that is NOT leading is left alone.
    assert "<think>" in gen.postprocess_caption("text <think>x</think> more")


# --------------------------------------------------------------------------- #
# t2v pipeline
# --------------------------------------------------------------------------- #
def test_t2v_single_invoke_with_official_format(ltx25, monkeypatch):
    prompts, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
            return "A single-paragraph LTX-2.5 caption."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls", lambda t: [] if t is None else ["data:image/jpeg;base64,AAAA"]
    )
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    out = enhancer("t2v - 文生视频", "a neon city", seed=7)
    assert out == "A single-paragraph LTX-2.5 caption."
    assert len(captured) == 1

    messages = captured[0]["messages"]
    assert len(messages) == 2
    # System: official gemma4 t2v prompt.
    assert messages[0]["content"] is prompts.SYSTEM_PROMPT_T2V
    # User turn: official "user prompt: ..." format, no image part.
    assert _text_part(messages[1]["content"]) == "user prompt: a neon city"
    assert _image_parts(messages[1]["content"]) == []
    # Default temperature (0.8) and token budget are forwarded.
    assert captured[0]["temperature"] == 0.8
    assert captured[0]["max_tokens"] == gen._DEFAULT_MAX_TOKENS


def test_t2v_explicit_temperature_overrides(ltx25, monkeypatch):
    _, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append(temperature)
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    enhancer = gen.LTX25PromptEnhancer(_Conn(), temperature=0.0)
    enhancer("t2v", "x")
    assert captured == [0.0]


def test_empty_user_prompt_uses_default_idea(ltx25, monkeypatch):
    _, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append(messages)
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    enhancer("t2v", "   ")
    user_text = _text_part(captured[0][1]["content"])
    assert user_text.startswith("user prompt: Create a visually striking")


def test_unknown_mode_falls_back_to_t2v(ltx25, monkeypatch):
    _, gen = ltx25
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(1)
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    out = enhancer("v2v", "a neon city")
    assert out == "ok"
    assert len(invocations) == 1  # ran as single-stage t2v


# --------------------------------------------------------------------------- #
# i2v pipeline (single multimodal call, image attached directly)
# --------------------------------------------------------------------------- #
def test_i2v_single_invoke_image_attached(ltx25, monkeypatch):
    prompts, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append({"messages": messages, "max_tokens": max_tokens})
            return "A single-paragraph i2v caption."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,AAAA"] if t is not None else [],
    )
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    out = enhancer(
        "i2v - 图生视频", "she walks away",
        image=_FakeTensor(),
        seed=3,
    )
    assert out == "A single-paragraph i2v caption."
    # Single-stage: exactly one multimodal call.
    assert len(captured) == 1

    messages = captured[0]["messages"]
    # System: official gemma4 i2v prompt.
    assert messages[0]["content"] is prompts.SYSTEM_PROMPT_I2V
    # User turn: first frame attached + official raw-input line.
    parts = _image_parts(messages[1]["content"])
    assert len(parts) == 1
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"
    assert _text_part(messages[1]["content"]) == "User Raw Input Prompt: she walks away."
    assert captured[0]["max_tokens"] == gen._DEFAULT_MAX_TOKENS


def test_i2v_missing_image_returns_original(ltx25, monkeypatch):
    _, gen = ltx25
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(1)
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    out = enhancer("i2v", "she walks away", image=None)
    assert out == "she walks away"
    assert invocations == []


def test_i2v_batch_uses_only_first_frame(ltx25, monkeypatch):
    _, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append(messages)
            return "ok"

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,A", "data:image/jpeg;base64,B", "data:image/jpeg;base64,C"],
    )
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    enhancer("i2v", "she walks away", image=_FakeTensor())
    assert len(captured) == 1
    parts = _image_parts(captured[0][1]["content"])
    assert len(parts) == 1
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,A"


def test_i2v_strips_think_block(ltx25, monkeypatch):
    _, gen = ltx25

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            return "<think>plan</think>\nFinal i2v caption."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,AAAA"] if t is not None else [],
    )
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    out = enhancer("i2v", "she walks away", image=_FakeTensor())
    assert out == "Final i2v caption."
    assert "<think>" not in out


def test_timeout_override_restores_connector(ltx25, monkeypatch):
    _, gen = ltx25

    class _Conn(FakeConnector):
        timeout = 30

        def invoke(self, messages, *, seed, temperature, max_tokens):
            assert self.timeout == 120
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    connector = _Conn()
    enhancer = gen.LTX25PromptEnhancer(connector, timeout=120)
    enhancer("t2v", "hello")
    assert connector.timeout == 30, "timeout was not restored after invoke"


# --------------------------------------------------------------------------- #
# is_changed
# --------------------------------------------------------------------------- #
def _is_changed_kwargs(**overrides):
    kwargs = dict(
        user_prompt="a cat",
        mode="t2v - 文生视频",
        seed=0,
        image_detail="auto",
        temperature=0.8,
        max_tokens=8192,
        timeout=120,
    )
    kwargs.update(overrides)
    return kwargs


def test_is_changed_stable_for_identical_inputs(ltx25):
    _, gen = ltx25
    node = gen.LTX25PromptGenerator()
    a = node.is_changed(FakeConnector(), **_is_changed_kwargs())
    b = node.is_changed(FakeConnector(), **_is_changed_kwargs())
    assert a == b


def test_is_changed_varies_with_mode(ltx25):
    _, gen = ltx25
    node = gen.LTX25PromptGenerator()
    base = node.is_changed(FakeConnector(), **_is_changed_kwargs())
    assert base != node.is_changed(
        FakeConnector(), **_is_changed_kwargs(mode="i2v - 图生视频")
    )


def test_is_changed_varies_with_user_prompt_and_temperature(ltx25):
    _, gen = ltx25
    node = gen.LTX25PromptGenerator()
    base = node.is_changed(FakeConnector(), **_is_changed_kwargs())
    assert base != node.is_changed(FakeConnector(), **_is_changed_kwargs(user_prompt="a dog"))
    assert base != node.is_changed(FakeConnector(), **_is_changed_kwargs(temperature=0.0))


def test_is_changed_varies_with_image_shape(ltx25):
    """The i2v first frame participates in the cache key via its tensor
    shape (SCAIL-2 / H3 media-signature pattern): attaching a frame or
    changing its resolution must trigger a re-run; same-shape pixel
    swaps intentionally do not."""
    _, gen = ltx25
    node = gen.LTX25PromptGenerator()

    class _ShapedTensor:
        def __init__(self, shape):
            self.shape = shape

    h_none = node.is_changed(FakeConnector(), image=None, **_is_changed_kwargs())
    h_512 = node.is_changed(
        FakeConnector(), image=_ShapedTensor((1, 512, 512, 3)), **_is_changed_kwargs()
    )
    h_720 = node.is_changed(
        FakeConnector(), image=_ShapedTensor((1, 720, 1280, 3)), **_is_changed_kwargs()
    )
    assert h_512 != h_none, "attaching a first frame must change the hash"
    assert h_720 != h_512, "changing the frame's resolution must change the hash"
    # Same shape -> same hash.
    assert (
        node.is_changed(
            FakeConnector(), image=_ShapedTensor((1, 512, 512, 3)), **_is_changed_kwargs()
        )
        == h_512
    )


# --------------------------------------------------------------------------- #
# Multishot widget integration (LTX-2.5 native multi-cut caption support)
#
# The multishot widget is OPTIONAL with default False; when True, the
# generator must append the multishot directive to the user turn in
# both t2v and i2v modes. is_changed must hash multishot so toggling
# the bool forces a re-run.
# --------------------------------------------------------------------------- #
def test_multishot_widget_schema(ltx25):
    """multishot is OPTIONAL with default False (BOOLEAN)."""
    _, gen = ltx25
    spec = gen.LTX25PromptGenerator.INPUT_TYPES()
    ms = spec["optional"]["multishot"]
    assert ms[0] == "BOOLEAN"
    assert ms[1]["default"] is False


def test_t2v_default_multishot_false_no_directive(ltx25, monkeypatch):
    """Default behavior (multishot omitted or False) must NOT inject
    any multishot markers into the user turn -- preserves the
    pre-multishot contract."""
    _, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append(messages)
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    enhancer("t2v - 文生视频", "a neon city", seed=7)
    user_text = _text_part(captured[0][1]["content"])
    assert user_text == "user prompt: a neon city"
    # No directive markers leaked.
    assert "hard cut transitions" not in user_text
    assert "C1" not in user_text


def test_t2v_multishot_true_appends_directive(ltx25, monkeypatch):
    """multishot=True on t2v must append the C1-C4 + prose-transitions
    directive after the user's idea (system prompt byte-lock honored)."""
    _, gen = ltx25
    captured = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append(messages)
            return "ok"

    monkeypatch.setattr(gen, "image_tensor_batch_to_data_urls", lambda t: [])
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    enhancer("t2v", "a neon city", seed=7, multishot=True)
    user_text = _text_part(captured[0][1]["content"])
    # Original user_text prefix is intact.
    assert user_text.startswith("user prompt: a neon city")
    # Directive markers present.
    for marker in ("C1", "C2", "C3", "C4", "A hard cut transitions to", "2-4"):
        assert marker in user_text, f"missing directive marker: {marker!r}"
    # No reference-frame note on t2v.
    assert "OPENING shot" not in user_text
    assert "opening shot" not in user_text


def test_i2v_multishot_true_appends_directive_with_opening_shot_note(ltx25, monkeypatch):
    """multishot=True on i2v must include the spec sec 4.5 OPENING-shot
    note (reference frame is the opening shot of the multi-cut sequence)."""
    _, gen = ltx25
    captured = []
    fake_image = _FakeTensor()

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            captured.append(messages)
            return "ok"

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda im: ["data:image/jpeg;base64,AAAA"] if im is fake_image else []
    )
    enhancer = gen.LTX25PromptEnhancer(_Conn())
    enhancer("i2v - 图生视频", "a cat", image=fake_image, seed=7, multishot=True)
    user_text = _text_part(captured[0][1]["content"])
    assert user_text.startswith("User Raw Input Prompt: a cat.")
    # i2v-specific OPENING-shot note.
    assert "OPENING shot" in user_text or "opening shot" in user_text
    # Base C1-C4 still applies.
    assert "C1" in user_text and "C4" in user_text


def test_multishot_change_invalidates_is_changed(ltx25):
    """Toggling multishot must invalidate is_changed -- otherwise
    ComfyUI will cache and silently skip re-running the LLM when the
    user opts into multi-cut mode."""
    _, gen = ltx25
    node = gen.LTX25PromptGenerator()
    conn = FakeConnector()
    h_off = node.is_changed(
        conn, "t2v - 文生视频", "a cat", seed=1, multishot=False
    )
    h_on = node.is_changed(
        conn, "t2v - 文生视频", "a cat", seed=1, multishot=True
    )
    assert h_off != h_on, "is_changed must be sensitive to multishot toggle"
