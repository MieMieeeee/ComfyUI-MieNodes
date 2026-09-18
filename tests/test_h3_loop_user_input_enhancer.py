# -*- coding: utf-8 -*-
"""Tests for the MiniMax H3 Loop user-input enhancer node.

The node is a thin wrapper around the LLM call defined in
``nodes/llm/minimax_h3_loop_user_input_enhancer.py``: it takes a rough
draft + the loop node's ``category`` / ``reference_mode`` widgets,
calls the LLM with the bundled ``user_input_enhancer.txt`` system
prompt, then extracts the ``--- BEGIN user_input ---`` ... block from
the reply. Tests cover:

* Parser: block extraction, whitespace trimming, ``<think>`` stripping.
* Failure modes: missing BEGIN/END markers raise RuntimeError, not
  silent fallback.
* Widget passthrough: category / reference_mode reach the LLM user
  message; seed reaches ``connector.invoke``.
* Surface: required / optional widget shape, RETURN_NAMES = ("user_input",)
  matching the loop node's widget name so the wire patches without
  renaming, CATEGORY matches the loop node, ``is_changed`` is stable.
"""
from __future__ import annotations

import importlib.util
import json
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
def enh():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"
    )
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_user_input_enhancer",
        LLM_DIR / "minimax_h3_loop_user_input_enhancer.py",
    )


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
        # Snapshot the messages + kwargs the enhancer actually sent.
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


# Module-level reply queue used by the simplest helper variant.
_REPLY_QUEUE: list[str] = []


def _good_reply_body() -> str:
    """A well-formed LLM reply matching the contract."""
    return (
        "Classification: Dialogue\n"
        "Notes for the user: Speaker names stable; setting paragraph "
        "carries character descriptions to the CAST sheet.\n"
        "\n"
        "--- BEGIN user_input ---\n"
        "场景设定：咖啡店室内。白色长毛母猫 = 莎莉猫（坐画面右边）。\n"
        "\n"
        "莎莉猫：今天天气真好。\n"
        "--- END user_input ---\n"
    )


# --------------------------------------------------------------------------- #
# Parser unit tests — extract_user_input_block
# --------------------------------------------------------------------------- #
def test_extract_block_well_formed(enh):
    """Plain well-formed reply: block is returned verbatim (inner only)."""
    raw = _good_reply_body()
    out = enh.extract_user_input_block(raw)
    assert out is not None
    assert out.startswith("场景设定：咖啡店室内")
    assert out.endswith("莎莉猫：今天天气真好。")
    # The Classification / Notes / markers themselves are stripped.
    assert "Classification:" not in out
    assert "--- BEGIN user_input ---" not in out
    assert "--- END user_input ---" not in out


def test_extract_block_strips_surrounding_whitespace(enh):
    """Leading/trailing blank lines and whitespace inside the block
    are trimmed before return."""
    raw = (
        "Classification: Narration\n"
        "\n"
        "--- BEGIN user_input ---\n"
        "\n"
        "\n"
        "A watchmaker at his bench.\n"
        "  \n"
        "--- END user_input ---\n"
    )
    out = enh.extract_user_input_block(raw)
    assert out == "A watchmaker at his bench."


def test_extract_block_strips_think_wrapper(enh):
    """Reasoning models may emit ``<think>...</think>`` before the
    visible answer; the wrapper must not leak into the captured
    block content."""
    raw = (
        "<think>"
        "The user wants a narration rewrite. I'll classify it and emit a "
        "single paragraph between the markers."
        "</think>"
        "\n"
        "Classification: Narration\n"
        "Notes: single paragraph.\n"
        "\n"
        "--- BEGIN user_input ---\n"
        "An old watchmaker at his bench.\n"
        "--- END user_input ---\n"
    )
    out = enh.extract_user_input_block(raw)
    assert out == "An old watchmaker at his bench."
    assert "<think>" not in out


def test_extract_block_returns_none_when_missing(enh):
    """No BEGIN/END markers -> None (caller raises)."""
    assert enh.extract_user_input_block("just some chatter, no markers") is None
    assert enh.extract_user_input_block("") is None
    assert enh.extract_user_input_block(None) is None


def test_extract_block_requires_close_marker(enh):
    """BEGIN without END -> None (don't half-extract)."""
    raw = "Classification: X\n--- BEGIN user_input ---\nstuff"
    assert enh.extract_user_input_block(raw) is None


# --------------------------------------------------------------------------- #
# End-to-end: enhance() through the connector
# --------------------------------------------------------------------------- #
def test_enhance_happy_path_returns_extracted_block(enh):
    """End-to-end: canned well-formed reply -> the BEGIN/END block
    is what gets returned through the STRING output socket."""
    conn = _make_connector_with_replies([_good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    out = node.enhance(
        conn,
        draft="two cats in a cafe",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        seed=0,
    )
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert out[0].startswith("场景设定")
    assert out[0].endswith("莎莉猫：今天天气真好。")
    # Exactly one LLM call.
    assert len(conn.calls) == 1


def test_enhance_raises_when_block_missing(enh):
    """No BEGIN/END in ANY reply -> RuntimeError after 3 attempts
    (silent fallback is forbidden by design — user choice)."""
    conn = _make_connector_with_replies([
        "Sure! Here is the rewritten user_input: it starts now...",
        "Still no markers, trust me.",
        "Third time, no markers either.",
    ])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    with pytest.raises(RuntimeError) as exc:
        node.enhance(
            conn,
            draft="x",
            category="none - 不指定",
            reference_mode="t2va - 文生视频链(默认)",
        )
    # All three attempts were spent before raising.
    assert len(conn.calls) == 3
    # Error message must include the attempt count + raw head.
    assert "after 3 attempts" in str(exc.value)
    assert "Third time" in str(exc.value)


def test_enhance_raises_when_reply_is_always_empty(enh):
    """Connector returns '' every time -> 3 attempts, then RuntimeError
    (never a silent fallback to the raw draft)."""
    conn = _make_connector_with_replies(["", "", ""])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    with pytest.raises(RuntimeError):
        node.enhance(
            conn,
            draft="x",
            category="none - 不指定",
            reference_mode="t2va - 文生视频链(默认)",
        )
    assert len(conn.calls) == 3


def test_enhance_retries_empty_reply_then_succeeds(enh):
    """Regression (MiniMax-M3 empty-200 flake): the FIRST attempt may
    return an empty string; the second attempt (fresh seed) succeeds —
    the node must return the block instead of killing the run."""
    conn = _make_connector_with_replies(["", _good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    out = node.enhance(
        conn,
        draft="two cats in a cafe",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        seed=7,
    )
    assert len(conn.calls) == 2
    assert out[0].startswith("场景设定")
    # The retry used a FRESH seed (same seed could deterministically
    # reproduce the same empty answer).
    assert conn.calls[0]["seed"] == 7
    assert conn.calls[1]["seed"] == 8


def test_enhance_passes_category_and_reference_mode_to_user_message(enh):
    """The category and reference_mode widget values reach the LLM's
    user message so the preprocessor picks the right (A/B/C/D)
    branch."""
    conn = _make_connector_with_replies([_good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    node.enhance(
        conn,
        draft="an alley fight with a hero and a villain",
        category="action - 动作戏/打斗/飙车",
        reference_mode="ref2va - 参考图(N张/全场景)",
        seed=7,
    )
    user_msg = conn.calls[0]["messages"][1]["content"]
    assert "an alley fight" in user_msg
    assert "category: action - 动作戏/打斗/飙车" in user_msg
    assert "reference_mode: ref2va - 参考图(N张/全场景)" in user_msg


def test_enhance_forwards_seed_to_connector(enh):
    """``seed`` is passed straight to ``connector.invoke`` as the
    ``seed`` keyword — it is NOT embedded in the user message."""
    conn = _make_connector_with_replies([_good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    node.enhance(
        conn,
        draft="x",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        seed=42,
    )
    assert conn.calls[0]["seed"] == 42
    user_msg = conn.calls[0]["messages"][1]["content"]
    assert "seed: 42" not in user_msg  # seed is connector-side, not prompt-side


def test_enhance_forwards_temperature_and_max_tokens(enh):
    """``temperature`` and ``max_tokens`` reach the connector."""
    conn = _make_connector_with_replies([_good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    node.enhance(
        conn,
        draft="x",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        temperature=0.7,
        max_tokens=2048,
    )
    assert conn.calls[0]["temperature"] == 0.7
    assert conn.calls[0]["max_tokens"] == 2048


def test_enhance_system_prompt_contains_four_task_type_branches(enh):
    """Sanity check: the bundled system prompt really covers all four
    task types — without this, the LLM has nothing to dispatch on."""
    conn = _make_connector_with_replies([_good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    node.enhance(
        conn,
        draft="x",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
    )
    sys_text = conn.calls[0]["messages"][0]["content"]
    for marker in ("## (A) Dialogue", "## (B) Action", "## (C) Pure narration",
                   "## (D) Reference-image driven", "--- BEGIN user_input ---"):
        assert marker in sys_text, f"system prompt missing marker: {marker!r}"


# --------------------------------------------------------------------------- #
# Surface: widget shape, return names, category, is_changed
# --------------------------------------------------------------------------- #
def test_input_types_required_and_optional_widgets(enh):
    """The 5 required widgets + 3 optional knobs match the plan."""
    types = enh.MiniMaxH3LoopUserInputEnhancer.INPUT_TYPES()
    required = set(types["required"].keys())
    assert required == {
        "llm_service_connector", "draft", "category", "reference_mode",
        "pacing", "seed",
    }
    optional = set(types["optional"].keys())
    assert optional == {"temperature", "max_tokens", "timeout"}


def test_return_types_and_names_match_loop_node_widget(enh):
    """``RETURN_NAMES = ("user_input",)`` so the wire patches straight
    into the H3 Loop node's ``user_input`` socket without renaming."""
    assert enh.MiniMaxH3LoopUserInputEnhancer.RETURN_TYPES == ("STRING",)
    assert enh.MiniMaxH3LoopUserInputEnhancer.RETURN_NAMES == ("user_input",)


def test_category_matches_loop_node_category(enh):
    """The new node lives in the same Prompt Generator subtree as the
    loop node — same MY_CATEGORY literal (including the escape form)."""
    assert enh.MY_CATEGORY == "\U0001F411 MieNodes/\U0001F411 Prompt Generator"
    assert (
        enh.MiniMaxH3LoopUserInputEnhancer.CATEGORY == enh.MY_CATEGORY
    )


def test_is_changed_stable_when_inputs_unchanged(enh):
    """Same connector + same inputs -> same hash. Different draft ->
    different hash."""
    conn = _FakeConnector()
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    h1 = node.is_changed(conn, "draft A", "none - 不指定", "t2va - 文生视频链(默认)",
                         seed=0, temperature=0.4, max_tokens=4096, timeout=300)
    h2 = node.is_changed(conn, "draft A", "none - 不指定", "t2va - 文生视频链(默认)",
                         seed=0, temperature=0.4, max_tokens=4096, timeout=300)
    h3 = node.is_changed(conn, "draft B", "none - 不指定", "t2va - 文生视频链(默认)",
                         seed=0, temperature=0.4, max_tokens=4096, timeout=300)
    assert h1 == h2
    assert h1 != h3


# --------------------------------------------------------------------------- #
# Timeout override mechanics
# --------------------------------------------------------------------------- #
def test_timeout_override_works_without_connector_attr(enh):
    """Regression: the per-call timeout override used to silently no-op
    when the connector had no ``timeout`` attribute of its own. It must
    apply for the duration of the call AND not leave a stale attribute
    behind on the connector afterwards."""
    class _BareConnector:
        model = "bare"

        def __init__(self):
            self.seen_timeout = "never-set"

        def invoke(self, messages, *, seed=None, temperature=None,
                   max_tokens=None):
            self.seen_timeout = getattr(self, "timeout", "not-set")
            return _good_reply_body()

    conn = _BareConnector()
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    out = node.enhance(
        conn,
        draft="two cats in a cafe",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        seed=0,
        timeout=300,
    )
    assert out[0].startswith("场景设定")
    # The override was visible while invoke() ran...
    assert conn.seen_timeout == 300
    # ...and the attribute we created is cleaned up afterwards.
    assert not hasattr(conn, "timeout")


def test_timeout_override_restores_previous_value(enh):
    """With an existing ``timeout`` attribute, the connector's own value
    is restored after the call."""
    conn = _FakeConnector()
    conn.timeout = 120
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    _REPLY_QUEUE.append(_good_reply_body())
    node.enhance(
        conn,
        draft="two cats",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        seed=0,
        timeout=600,
    )
    assert conn.timeout == 120


# --------------------------------------------------------------------------- #
# split_enhancer_reply — block + advice header (inline-enhance support)
# --------------------------------------------------------------------------- #
def test_split_reply_returns_block_and_header(enh):
    raw = (
        "<think>chain of thought</think>\n"
        "Classification: Dialogue\n"
        "Notes for the user: Speaker names stable; positions declared.\n"
        "\n"
        "--- BEGIN user_input ---\n"
        "场景设定：咖啡店。\n"
        "莎莉猫：今天天气真好。\n"
        "--- END user_input ---"
    )
    block, header = enh.split_enhancer_reply(raw)
    assert block == "场景设定：咖啡店。\n莎莉猫：今天天气真好。"
    assert "Classification: Dialogue" in header
    assert "Notes for the user" in header
    assert "<think>" not in header


def test_split_reply_missing_markers(enh):
    assert enh.split_enhancer_reply("no markers at all") == (None, "")
    assert enh.split_enhancer_reply("") == (None, "")
    # BEGIN without END: no block (the caller raises on None), and the
    # header stays whatever preceded the dangling BEGIN marker.
    block, header = enh.split_enhancer_reply(
        "Classification: X\n--- BEGIN user_input ---\nstuff"
    )
    assert block is None
    assert "Classification: X" in header


def test_enhance_passes_pacing_to_user_message(enh):
    """The pacing widget value reaches the rewrite prompt's context block
    so the LLM shapes content density to match (fast -> more/shorter
    beats, slow -> fewer/longer)."""
    conn = _make_connector_with_replies([_good_reply_body()])
    node = enh.MiniMaxH3LoopUserInputEnhancer()
    node.enhance(
        conn,
        draft="two cats in a cafe",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        pacing="fast - 快",
        seed=0,
    )
    user_msg = conn.calls[0]["messages"][1]["content"]
    assert "pacing: fast - 快" in user_msg
    # Without a pacing value the context omits the line entirely.
    conn2 = _make_connector_with_replies([_good_reply_body()])
    node.enhance(
        conn2,
        draft="two cats",
        category="none - 不指定",
        reference_mode="t2va - 文生视频链(默认)",
        pacing="",
        seed=0,
    )
    assert "pacing:" not in conn2.calls[0]["messages"][1]["content"]
