# -*- coding: utf-8 -*-
"""Tests for the LTX-2.5 prompt templates module (``ltx25_prompts``).

Mirrors the loader-stubbing pattern from ``test_krea2_prompts.py``.

Coverage:
* The two bundled official gemma4 system prompts are on disk,
  byte-sized to match the Lightricks/LTX-2 main-repo originals, and
  carry their 2.5-era signature phrases. LTX-2.5 is gemma4-only, so no
  gemma3 files ship.
* t2v is the default mode.
* ``parse_mode`` honors the ``"<code> - <label>"`` separator contract.
* ``load_system_prompt`` dispatches t2v/i2v and falls back to t2v on
  unknown values.
* User-turn templates match the official ``base_encoder.py`` formats
  (``"user prompt: ..."`` / ``"User Raw Input Prompt: ...."``).
"""
import importlib
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "tests"))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME  # noqa: E402

LTX25_PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts" / "ltx25"

# Byte sizes of the upstream files on Lightricks/LTX-2 main
# (packages/ltx-core/.../gemma/encoders/prompts/), verified 2026-08-17.
UPSTREAM_SIZES = {
    "system_t2v_gemma4.txt": 3769,
    "system_i2v_gemma4.txt": 4708,
}


@pytest.fixture(scope="module")
def ltx25():
    """Import the helper module under the ``PACKAGE_NAME`` namespace,
    matching the runtime layout the project uses inside ComfyUI.
    """
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ltx25_prompts")


# --------------------------------------------------------------------------- #
# Bundled prompt-file tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,size", sorted(UPSTREAM_SIZES.items()))
def test_official_prompt_file_size(name, size):
    """Each official prompt file exists and byte-matches the upstream
    size (guards against accidental edits / re-encoding)."""
    path = LTX25_PROMPTS_DIR / name
    assert path.is_file(), f"missing prompt file: {path}"
    data = path.read_bytes()
    assert len(data) == size, f"{name}: {len(data)} bytes, expected {size}"
    assert b"\r" not in data, f"{name}: must keep LF-only line endings"


def test_prompt_dir_is_gemma4_only():
    """LTX-2.5 is gemma4-only: exactly the two gemma4 prompts ship, no
    gemma3 leftovers and no MieNodes-authored caption prompt (i2v is
    single-stage, image attached directly)."""
    assert sorted(p.name for p in LTX25_PROMPTS_DIR.glob("*.txt")) == sorted(
        UPSTREAM_SIZES.keys()
    )


def test_gemma4_t2v_signature_phrases(ltx25):
    """The gemma4 t2v prompt carries its 2.5-era signatures: caption-style
    role, framing triple, 150-220 word band, AESTHETIC QUALITY section."""
    text = ltx25.SYSTEM_PROMPT_T2V
    for phrase in (
        "Write a single, highly detailed audio-visual caption",
        "Shot type",
        "Camera motion",
        "Camera viewpoint",
        "150–220 words",
        "AESTHETIC QUALITY",
        "Output ONLY the caption text",
        "Do not infer ethnicity, nationality, religion, or culture",
    ):
        assert phrase in text, f"missing phrase: {phrase!r}"


def test_gemma4_i2v_signature_phrases(ltx25):
    """The gemma4 i2v prompt adds the FIRST-FRAME / IMAGE GROUNDING
    block on top of the t2v caption style."""
    text = ltx25.SYSTEM_PROMPT_I2V
    for phrase in (
        "FIRST-FRAME / IMAGE GROUNDING",
        "the exact first frame of the video",
        "Single continuous take — no hard cuts",
        "150–220 words",
        "AESTHETIC QUALITY",
    ):
        assert phrase in text, f"missing phrase: {phrase!r}"


# --------------------------------------------------------------------------- #
# Defaults + dropdown contract
# --------------------------------------------------------------------------- #
def test_t2v_is_default_mode(ltx25):
    assert ltx25.DEFAULT_MODE == "t2v"
    assert ltx25.MODES[0].startswith("t2v")


def test_parse_mode(ltx25):
    assert ltx25.parse_mode("t2v - 文生视频") == "t2v"
    assert ltx25.parse_mode("i2v - 图生视频") == "i2v"
    assert ltx25.parse_mode("i2v") == "i2v"
    assert ltx25.parse_mode("") == ""
    assert ltx25.parse_mode(None) is None


# --------------------------------------------------------------------------- #
# System-prompt dispatch
# --------------------------------------------------------------------------- #
def test_load_system_prompt_dispatch(ltx25):
    assert ltx25.load_system_prompt("t2v") is ltx25.SYSTEM_PROMPT_T2V
    assert ltx25.load_system_prompt("i2v") is ltx25.SYSTEM_PROMPT_I2V
    # Display strings (what the widget sends) resolve the same as bare
    # codes.
    assert (
        ltx25.load_system_prompt("i2v - 图生视频") is ltx25.SYSTEM_PROMPT_I2V
    )


def test_load_system_prompt_unknown_falls_back(ltx25):
    """Unknown mode values fall back to t2v instead of erroring
    (hand-edited workflows must still produce a valid prompt)."""
    assert ltx25.load_system_prompt("v2v") is ltx25.SYSTEM_PROMPT_T2V
    assert ltx25.load_system_prompt("") is ltx25.SYSTEM_PROMPT_T2V


# --------------------------------------------------------------------------- #
# User-turn templates (official base_encoder.py formats)
# --------------------------------------------------------------------------- #
def test_build_t2v_user_text(ltx25):
    assert ltx25.build_t2v_user_text("a cat surfing") == "user prompt: a cat surfing"
    assert ltx25.build_t2v_user_text("  padded  ") == "user prompt: padded"
    assert ltx25.build_t2v_user_text("") == "user prompt: "


def test_build_i2v_user_text(ltx25):
    assert (
        ltx25.build_i2v_user_text("she walks away")
        == "User Raw Input Prompt: she walks away."
    )
    assert ltx25.build_i2v_user_text("  padded  ") == "User Raw Input Prompt: padded."


# --------------------------------------------------------------------------- #
# Multishot directive (LTX-2.5 native multi-cut caption support)
#
# The two official system prompts are byte-locked (cannot be modified
# without forking the upstream caption style), so multishot guidance
# is appended to the user turn as a directive rather than baked into
# the system prompt. Default off -- the bool widget on the node stays
# False unless the user explicitly opts in.
# --------------------------------------------------------------------------- #
def test_append_multishot_directive_t2v_marks_c1_through_c4(ltx25):
    """The t2v directive must reference the C1-C4 cut checklist
    (per official sec 4.2) and name the prose-form transitions so the
    LLM follows template E rather than the slugline fallback."""
    out = ltx25.append_multishot_directive("user prompt: a neon city", "t2v")
    # User-text prefix is preserved (system-prompt byte-lock is honored).
    assert out.startswith("user prompt: a neon city")
    # C1-C4 markers.
    assert "C1" in out and "C2" in out and "C3" in out and "C4" in out
    # Prose transition names (template E primary form).
    for phrase in ("A hard cut transitions to", "match cut", "dissolves into"):
        assert phrase in out, f"t2v directive missing transition phrase: {phrase!r}"
    # Cut-count guidance.
    assert "2-4" in out


def test_append_multishot_directive_i2v_adds_opening_shot_note(ltx25):
    """i2v must add the spec sec 4.5 caveat: the reference frame is the
    OPENING shot of the multi-cut sequence. Without this note the LLM
    tends to treat i2v as single-shot (the system prompt default)."""
    out = ltx25.append_multishot_directive("User Raw Input Prompt: a cat.", "i2v")
    assert "OPENING shot" in out or "opening shot" in out
    assert "reference" in out.lower()
    # Base C1-C4 still applies.
    assert "C1" in out and "C4" in out


def test_append_multishot_directive_unknown_mode_falls_back_to_t2v(ltx25):
    """Unknown mode code reuses the t2v directive shape (no OPENING-shot
    note); same defensive fallback as load_system_prompt."""
    out = ltx25.append_multishot_directive("user prompt: x", "what")
    assert out.startswith("user prompt: x")
    assert "OPENING shot" not in out
    assert "C1" in out


def test_append_multishot_directive_empty_user_text_is_safe(ltx25):
    """A blank user_prompt should still produce a valid directive
    wrapper (used when _default_idea fills in later)."""
    out = ltx25.append_multishot_directive("", "t2v")
    assert "C1" in out
