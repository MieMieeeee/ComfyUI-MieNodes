# -*- coding: utf-8 -*-
"""Unit tests for ``validate_dialogue_invariantity`` + ``extract_d_blocks``.

Covers the post-LLM contract that every input dialogue line must land in
exactly one ``<d>[Language]...</d>`` block in the produced shot prompt,
with verbatim text.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
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
def lp():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.dialogue_segmenter",
        LLM_DIR / "dialogue_segmenter.py",
    )
    return sys.modules["_mienodes_internal.nodes.llm.minimax_h3_loop_prompts"]


def _turn(speaker: str, *lines: str):
    from _mienodes_internal.nodes.llm.dialogue_segmenter import DialogueTurn

    return DialogueTurn(speaker=speaker, lines=list(lines))


# --------------------------------------------------------------------------- #
# extract_d_blocks / count_d_blocks
# --------------------------------------------------------------------------- #
def test_extract_d_blocks_strips_inner_text(lp):
    text = "前文 <d>[Chinese] 你好。</d> 后文 <d>[English] hi.</d>"
    assert lp.extract_d_blocks(text) == ["你好。", "hi."]


def test_count_d_blocks_matches_extract(lp):
    text = "<d>[Chinese] a</d> blah <d>[English] b</d> <d>[Chinese] c</d>"
    assert lp.count_d_blocks(text) == 3
    assert len(lp.extract_d_blocks(text)) == 3


def test_lowercase_language_tag_is_not_counted(lp):
    """The contract is full-name tags only: [Chinese] / [English]."""
    text = "<d>[zh] 你好。</d> <d>[en] hi.</d>"
    assert lp.count_d_blocks(text) == 0


def test_case_insensitive_full_name_language_tag_is_counted(lp):
    text = "<d>[chinese] 你好。</d> <d>[ENGLISH] hi.</d>"
    assert lp.extract_d_blocks(text) == ["你好。", "hi."]


# --------------------------------------------------------------------------- #
# validate_dialogue_invariantity
# --------------------------------------------------------------------------- #
def test_invariantity_passes_when_blocks_match(lp):
    shot_prompt = (
        "integrated_multimodal_description:\n"
        "[Shot 1] Beat.\n"
        "<d>[Chinese] 你好。</d>\n"
        "overall_soundscape:\nquiet.\n"
        "non_diegetic_music:\nNo non-diegetic music."
    )
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt.splitlines()}],
        [_turn("Sahli", "你好。")],
    )
    assert errors == []


def test_invariantity_flags_split_block(lp):
    """If the LLM splits one input line into setup + punchline two
    <d> blocks, the per-shot count exceeds the turn's line count."""
    shot_prompt = (
        "<d>[Chinese] 你好，</d> blabla <d>[Chinese] 怎么？</d>"
    )
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}],
        [_turn("Sahli", "你好，怎么？")],
    )
    assert any("expected 1 <d> blocks" in e for e in errors)


def test_invariantity_flags_missing_block(lp):
    shot_prompt = "<d>[Chinese] 第一句。</d>"
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}],
        [_turn("Sahli", "第一句。", "第二句。")],
    )
    assert any("expected 2 <d> blocks" in e for e in errors)


def test_invariantity_flags_paraphrased_text(lp):
    """Verbatim text mismatch -> error."""
    shot_prompt = "<d>[Chinese] 你好世界。</d>"
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}],
        [_turn("Sahli", "你好。")],
    )
    assert any("verbatim mismatch" in e for e in errors)


def test_invariantity_flags_shot_count_mismatch(lp):
    shot_prompt = "<d>[Chinese] hi</d>"
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}, {"prompt": "<d>[Chinese] bye</d>"}],
        [_turn("Sahli", "hi")],
    )
    assert any("shot count 2 != turn count 1" in e for e in errors)


def test_invariantity_handles_prompt_as_string(lp):
    shot_prompt = (
        "integrated_multimodal_description:\n[Shot 1] beat.\n"
        "<d>[Chinese] 你好。</d>\n"
        "overall_soundscape:\nquiet.\n"
        "non_diegetic_music:\nNo music."
    )
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}],
        [_turn("Sahli", "你好。")],
    )
    assert errors == []


def test_invariantity_handles_multiline_turn(lp):
    """Same-speaker consecutive lines land as N <d> blocks in one shot."""
    shot_prompt = (
        "<d>[Chinese] 第一句。</d>\n"
        "<d>[Chinese] 第二句。</d>"
    )
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}],
        [_turn("Harry", "第一句。", "第二句。")],
    )
    assert errors == []


def test_invariantity_strips_whitespace_inside_block(lp):
    shot_prompt = "<d>[Chinese]   你好。   </d>"
    errors = lp.validate_dialogue_invariantity(
        [{"prompt": shot_prompt}],
        [_turn("Sahli", "你好。")],
    )
    assert errors == []