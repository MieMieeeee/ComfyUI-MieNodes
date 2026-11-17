# -*- coding: utf-8 -*-
"""Tests for the MiniMax-H3-Image prompt-enhancement ComfyUI node.

The node is a thin shell: the mode table, the official-text loaders and
the reply normalizer live in ``mie_prompt.h3_image_prompts`` (tested
there). What is tested here is the shell's own behaviour:

* the widget surface and the plugin registration;
* the message each mode puts on the wire -- the t2i task prompt with the
  draft substituted, the edit mode's interleaved reference-media parts and
  the instruction last, the guide modes' system prompt;
* the reference-image metadata: the exact ratio comes from the pixels, and
  the edit modes refuse to run without images;
* the aspect-ratio policy: the user's choice wins, and t2i stays inside the
  ten official ratios;
* the reply handling: the JSON contract, the corrective retry, the verbatim
  rule for quoted text;
* ``is_changed`` stability and content-sensitivity.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_DIR / "nodes" / "llm"

sys.path.insert(0, str(PROJECT_DIR / "tests"))


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


@pytest.fixture(scope="module")
def helpers():
    _bootstrap()
    return _load_file(
        "_mienodes_internal.nodes.llm.h3_image_prompts",
        LLM_DIR / "h3_image_prompts.py",
    )


@pytest.fixture(scope="module")
def mod(helpers):
    # Loaded last so its relative imports resolve to the shimmed modules.
    return _load_file(
        "_mienodes_internal.nodes.llm.h3_image_prompt_generator",
        LLM_DIR / "h3_image_prompt_generator.py",
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
        self._replies = list(replies or [])

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

    def __init__(self, n: int = 1, h: int = 8, w: int = 8):
        self._t = np.zeros((n, h, w, 3), dtype=np.float32)

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


def _json_reply(caption="A red cup on a wooden table.", ratio="4:3"):
    return (
        '{"expanded_caption": ' + repr(caption).replace("'", '"') + ','
        ' "aspect_ratio": "' + ratio + '"}'
    )


# --------------------------------------------------------------------------- #
# Surface
# --------------------------------------------------------------------------- #
def test_widget_surface(mod):
    node = mod.H3ImagePromptGenerator
    types_ = node.INPUT_TYPES()
    assert "llm_service_connector" in types_["required"]
    assert "mode" in types_["required"]
    assert "user_input" in types_["required"]
    assert "seed" in types_["required"]
    assert "reference_images" in types_["optional"]
    assert "aspect_ratio" in types_["optional"]
    assert node.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert node.RETURN_NAMES == ("prompt", "aspect_ratio", "raw_reply")
    assert node.FUNCTION == "enhance"
    assert "Prompt Generator" in node.CATEGORY


def test_the_four_modes_are_offered(mod):
    labels = list(mod.MODES)
    assert len(labels) == 4
    codes = [mod.parse_mode(label) for label in labels]
    assert codes == ["t2i", "edit", "t2i_guide", "edit_guide"]


def test_plugin_registration():
    from test_plugin_imports import load_plugin_module

    plugin = load_plugin_module()
    registered = [
        key
        for key in plugin.NODE_CLASS_MAPPINGS
        if key.split("|")[0] == "H3ImagePromptGenerator"
    ]
    assert registered, "the node is not registered in NODE_CLASS_MAPPINGS"
    assert any(
        name.split("|")[0] == "H3ImagePromptGenerator"
        for name in plugin.NODE_DISPLAY_NAME_MAPPINGS
    )


# --------------------------------------------------------------------------- #
# What goes on the wire
# --------------------------------------------------------------------------- #
def test_t2i_sends_the_official_task_prompt_with_the_draft(mod):
    connector = _FakeConnector([_json_reply()])
    node = mod.H3ImagePromptGenerator()
    node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="一只红杯子",
        seed=1,
    )
    messages = connector.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "expanded_caption" in messages[0]["content"]
    user = messages[1]["content"]
    assert "input_text: 一只红杯子" in user
    assert "{{" not in user


def test_edit_sends_the_reference_media_parts_and_the_instruction_last(mod):
    connector = _FakeConnector([_json_reply(caption="A blue cup.")])
    node = mod.H3ImagePromptGenerator()
    node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        user_input="把杯子变成蓝色",
        seed=1,
        reference_images=_FakeImageBatch(n=2, h=64, w=32),
        aspect_ratio="16:9",
    )
    content = connector.calls[0]["messages"][1]["content"]
    assert isinstance(content, list)
    kinds = [part["type"] for part in content]
    # The official edit task prompt, then a text/image pair per reference,
    # then the instruction.
    assert kinds == ["text", "text", "image_url", "text", "image_url", "text"]
    assert content[0]["text"].startswith("You are an expert at enhancing")
    assert content[1]["text"] == "Reference Media: <Picture 1> aspect_ratio 1:2"
    assert content[3]["text"] == "Reference Media: <Picture 2> aspect_ratio 1:2"
    last = content[5]["text"]
    assert "把杯子变成蓝色" in last
    assert "User Aspect Ratio Selection: 16:9" in last


def test_the_guide_modes_use_the_official_skill(mod):
    connector = _FakeConnector(["```text\nA red cup on a wooden table.\n```\n\nAspect ratio: 3:4"])
    node = mod.H3ImagePromptGenerator()
    prompt, ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[2],
        user_input="a product shot of a red cup",
        seed=1,
    )
    system = connector.calls[0]["messages"][0]["content"]
    assert "Text-to-image prompt writing guide" in system
    assert "Aspect ratio:" in system
    assert prompt == "A red cup on a wooden table."
    assert ratio == "3:4"


# --------------------------------------------------------------------------- #
# The reference images
# --------------------------------------------------------------------------- #
def test_the_edit_modes_refuse_to_run_without_images(mod):
    node = mod.H3ImagePromptGenerator()
    with pytest.raises(ValueError):
        node.enhance(
            llm_service_connector=_FakeConnector([_json_reply()]),
            mode=mod.MODES[1],
            user_input="make it blue",
            seed=1,
        )


def test_the_text_modes_ignore_the_images(mod):
    connector = _FakeConnector([_json_reply()])
    node = mod.H3ImagePromptGenerator()
    node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
        reference_images=_FakeImageBatch(n=1),
    )
    assert isinstance(connector.calls[0]["messages"][1]["content"], str)


def test_the_ratio_comes_from_the_pixels(mod):
    """The model must not guess the reference ratio."""
    connector = _FakeConnector([_json_reply()])
    node = mod.H3ImagePromptGenerator()
    node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        user_input="make it blue",
        seed=1,
        reference_images=_FakeImageBatch(n=1, h=1080, w=1920),
    )
    content = connector.calls[0]["messages"][1]["content"]
    assert content[1]["text"] == "Reference Media: <Picture 1> aspect_ratio 16:9"


# --------------------------------------------------------------------------- #
# The aspect-ratio policy
# --------------------------------------------------------------------------- #
def test_the_users_ratio_wins_over_the_models(mod):
    connector = _FakeConnector([_json_reply(ratio="1:1")])
    node = mod.H3ImagePromptGenerator()
    _prompt, ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
        aspect_ratio="16:9",
    )
    assert ratio == "16:9"


def test_t2i_refuses_a_ratio_outside_the_official_ten(mod):
    connector = _FakeConnector([_json_reply(ratio="4:3")])
    node = mod.H3ImagePromptGenerator()
    _prompt, ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
        aspect_ratio="7:3",
    )
    # The invalid request falls back to adaptive, so the model's own
    # (official) ratio stands.
    assert ratio == "4:3"
    assert "User Aspect Ratio Selection" not in str(
        connector.calls[0]["messages"][1]["content"]
    )


def test_edit_accepts_any_wh_ratio(mod):
    connector = _FakeConnector([_json_reply(ratio="7:5")])
    node = mod.H3ImagePromptGenerator()
    _prompt, ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        user_input="make it blue",
        seed=1,
        reference_images=_FakeImageBatch(n=1),
        aspect_ratio="7:5",
    )
    assert ratio == "7:5"


def test_an_adaptive_edit_reply_falls_back_to_the_reference_ratio(mod):
    """In adaptive mode the official edit prompt keeps the reference ratio."""
    connector = _FakeConnector([_json_reply(ratio="adaptive")])
    node = mod.H3ImagePromptGenerator()
    _prompt, ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[1],
        user_input="make it blue",
        seed=1,
        reference_images=_FakeImageBatch(n=1, h=1080, w=1920),
    )
    assert ratio == "16:9"


# --------------------------------------------------------------------------- #
# The reply
# --------------------------------------------------------------------------- #
def test_a_fenced_reply_is_accepted(mod):
    connector = _FakeConnector(
        ["```json\n" + _json_reply(caption="A red cup.") + "\n```"]
    )
    node = mod.H3ImagePromptGenerator()
    prompt, _ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
    )
    assert prompt == "A red cup."


def test_a_thinking_wrapper_is_accepted(mod):
    connector = _FakeConnector(
        ["<think>hmm</think>\n" + _json_reply(caption="A red cup.")]
    )
    node = mod.H3ImagePromptGenerator()
    prompt, _ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
    )
    assert prompt == "A red cup."


def test_an_unusable_reply_is_retried_once_then_fails_loudly(mod):
    connector = _FakeConnector(["not json at all", "still not json"])
    node = mod.H3ImagePromptGenerator()
    with pytest.raises(RuntimeError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[0],
            user_input="a red cup",
            seed=1,
        )
    assert len(connector.calls) == 2
    # The corrective turn carries the assistant reply and a correction.
    retry = connector.calls[1]["messages"]
    assert retry[-2]["role"] == "assistant"
    assert "ONE JSON object" in retry[-1]["content"]


def test_a_reply_that_dropped_a_quoted_literal_fails_loudly(mod):
    connector = _FakeConnector(
        [
            _json_reply(caption="A sales dashboard."),
            _json_reply(caption="A sales dashboard."),
        ]
    )
    node = mod.H3ImagePromptGenerator()
    with pytest.raises(RuntimeError):
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[0],
            user_input='a dashboard titled "Quarterly sales"',
            seed=1,
        )


def test_an_empty_reply_names_the_real_cause(mod):
    connector = _FakeConnector([""])
    node = mod.H3ImagePromptGenerator()
    with pytest.raises(RuntimeError) as excinfo:
        node.enhance(
            llm_service_connector=connector,
            mode=mod.MODES[0],
            user_input="a red cup",
            seed=1,
        )
    assert "max_tokens" in str(excinfo.value)


def test_an_empty_reply_is_retried_before_the_run_dies(mod):
    """A reasoning model that thinks itself out of the answer gets one more call."""
    connector = _FakeConnector(["", _json_reply(caption="A red cup.")])
    node = mod.H3ImagePromptGenerator()
    prompt, _ratio, _raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
    )
    assert prompt == "A red cup."
    assert len(connector.calls) == 2


def test_the_raw_reply_is_returned_for_debugging(mod):
    raw = _json_reply(caption="A red cup.")
    connector = _FakeConnector([raw])
    node = mod.H3ImagePromptGenerator()
    _prompt, _ratio, out_raw = node.enhance(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="a red cup",
        seed=1,
    )
    assert out_raw == raw


# --------------------------------------------------------------------------- #
# is_changed
# --------------------------------------------------------------------------- #
def test_is_changed_is_stable_and_content_sensitive(mod):
    node = mod.H3ImagePromptGenerator()
    connector = _FakeConnector()
    kwargs = dict(
        llm_service_connector=connector,
        mode=mod.MODES[0],
        user_input="一只猫",
        seed=7,
    )
    first = node.is_changed(**kwargs)
    assert first == node.is_changed(**kwargs)
    assert first != node.is_changed(**{**kwargs, "user_input": "一只狗"})
    assert first != node.is_changed(**{**kwargs, "mode": mod.MODES[2]})
