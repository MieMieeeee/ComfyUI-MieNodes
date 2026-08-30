"""LTX-2.5 prompt-enhancement templates and helpers for MieNodes.

Bundles the official LTX-2.5 system prompts shipped by the
Lightricks/LTX-2 main repo
(``packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/``)
and adds the user-turn templates.

LTX-2.5 is gemma4-only (verified 2026-08-17 against the official
repos): HF ``Lightricks/LTX-2.5`` ships only
``gemma4-12b-with-proj-ltx-2.5-*`` text encoders; the 2.5 checkpoint
config declares ``gemma_source_checkpoint`` "LTX 2.5 / gemma4" and the
loader enforces the match; upstream ``base_encoder._default_system_prompt``
loads ``gemma4_{t2v,i2v}_system_prompt.txt`` for ``model_type ==
"gemma4"``. There is no encoder-family switch -- the node is 100%
gemma4, so the only dimension is t2v vs i2v.

User-turn formats mirror upstream ``base_encoder.py`` verbatim:
t2v sends ``"user prompt: {p}"``; i2v attaches the reference image to
the multimodal user turn alongside ``"User Raw Input Prompt: {p}."``
(the generator attaches the image, as Bernini / H3 i2v do).
"""
from __future__ import annotations

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text


# --------------------------------------------------------------------------- #
# Dropdown (display strings for the ComfyUI widget).
#
# Display strings use the literal separator " - " (space-hyphen-space,
# ASCII U+002D) so ``parse_mode`` can split them back into the short
# code. Every entry MUST follow "<code> - <label>" exactly; using a
# different separator (em-dash, colon, no spaces) silently breaks the
# split.
# --------------------------------------------------------------------------- #
MODES = (
    "t2v - 文生视频",
    "i2v - 图生视频",
)

MODE_CODES = (
    "t2v",
    "i2v",
)

DEFAULT_MODE = MODE_CODES[0]


def parse_mode(mode: str) -> str:
    """Extract the short mode code from a display string.

    Accepts ``"t2v - 文生视频"``-style display strings, bare codes
    (saved workflows), and None / empty (returned unchanged; the
    enhancer falls back to ``DEFAULT_MODE`` on unknown values).
    """
    if not mode:
        return mode
    return mode.split(" - ", 1)[0].strip()


# --------------------------------------------------------------------------- #
# System prompts (verbatim from Lightricks/LTX-2 main repo, LF endings)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT_T2V = load_prompt_text("ltx25/system_t2v_gemma4")
SYSTEM_PROMPT_I2V = load_prompt_text("ltx25/system_i2v_gemma4")

_SYSTEM_PROMPTS = {
    "t2v": SYSTEM_PROMPT_T2V,
    "i2v": SYSTEM_PROMPT_I2V,
}


def load_system_prompt(mode: str) -> str:
    """Return the official gemma4 system prompt for a mode.

    Unknown mode values fall back to ``t2v`` rather than erroring, so
    a typo'd value from a hand-edited workflow still produces a valid
    prompt.
    """
    mode_code = parse_mode(mode)
    if mode_code not in MODE_CODES:
        mode_code = DEFAULT_MODE
    return _SYSTEM_PROMPTS[mode_code]


# --------------------------------------------------------------------------- #
# User-turn templates (formats mirror upstream base_encoder.py)
# --------------------------------------------------------------------------- #
def build_t2v_user_text(user_prompt: str) -> str:
    """Build the t2v user turn, matching upstream ``enhance_t2v``
    (``f"user prompt: {prompt}"``)."""
    return f"user prompt: {(user_prompt or '').strip()}"


def build_i2v_user_text(user_prompt: str) -> str:
    """Build the i2v user turn text.

    Matches upstream ``enhance_i2v``
    (``"User Raw Input Prompt: {prompt}."``). The reference image
    itself is attached to the same user turn by the generator, as
    upstream and the Bernini / H3 i2v paths do.
    """
    return f"User Raw Input Prompt: {(user_prompt or '').strip()}."
