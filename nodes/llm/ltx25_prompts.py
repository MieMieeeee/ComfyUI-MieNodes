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


# --------------------------------------------------------------------------- #
# Multishot directive (LTX-2.5 native multi-cut caption support)
#
# The two official system prompts are byte-locked (cannot be modified
# without forking the upstream caption style), so multishot guidance is
# injected via the user turn rather than the system prompt. The directive
# is appended AFTER the user's idea, separated by a blank line, so the
# upstream "user prompt: ..." / "User Raw Input Prompt: ...." format
# stays intact at the start of the user content.
#
# The directive encodes spec.md sec 4.2 / 4.3 + template E:
#  * 2-4 cuts per generation
#  * timeline prose (NOT [Shot N] 0-5s: ... sluglines) as the primary
#    form, with explicit transition phrases
#    "A hard cut transitions to..." / "A match cut connects..." /
#    "The image dissolves into..."
#  * per-cut C1-C4 checklist:
#    C1 name the transition in prose
#    C2 re-establish the new shot (size + angle + subject + lighting)
#    C3 re-anchor identity (visual markers carry across cuts)
#    C4 state audio continuity (carries over or drops)
#  * present tense, time-ordered transitions
#
# i2v adds the spec.md sec 4.5 caveat: the reference first frame is the
# OPENING shot of the multi-cut sequence (without this note the LLM
# tends to treat i2v as single-shot -- the default behavior of the
# upstream system prompt).
# --------------------------------------------------------------------------- #
_MULTISHOT_DIRECTIVE_BASE = '''

[Multishot directive -- expand the above as a 2-4 cut multi-shot
caption in the official LTX-2.5 multi-shot style (spec sec 4.3 +
template E, NOT the gemma3-era slugline format):

- Primary form: timeline prose. Use one of these prose transitions to
  mark each cut -- "A hard cut transitions to ..." / "A match cut
  connects ..." / "The image dissolves into ...". Avoid "[Shot N]
  0-5s: ..." sluglines unless the user explicitly asks for shot-list
  format.
- Each cut must cover all of C1-C4:
  C1 -- name the transition (one of the prose forms above).
  C2 -- re-establish the new shot (size + angle + subject in frame +
        lighting if it changed).
  C3 -- re-anchor identity across the cut ("the woman in the red coat,
        earlier at the table, now ...").
  C4 -- state audio continuity ("the synth score continues across the
        cut, traffic muffled" or "the dialogue drops; only wind
        remains").
- 2-4 cuts per generation. Each cut should carry a clear function
  (establish -> detail -> reaction, or wide -> medium -> close-up).
- Present tense, time-ordered actions ("Initially ..." / "A moment
  later ..." / "Simultaneously ...").
- Keep the rest of the official caption rules (camera language, audio
  paragraph, ~150-220 words per cut's worth of prose).'''

_MULTISHOT_DIRECTIVE_I2V_NOTE = '''
- The provided reference first frame is the OPENING shot of the
  multi-cut sequence; subsequent cuts diverge from it. (i2v default
  is single-shot per spec sec 4.5; this directive explicitly chooses
  multi-cut from frame 1.)'''


def append_multishot_directive(user_text: str, mode_code: str) -> str:
    """Append the multishot directive to a user turn (t2v or i2v).

    The upstream `user prompt: ...` / `User Raw Input Prompt: ....`
    prefix stays intact at the start of the returned string; the
    directive follows after a blank line. Unknown mode codes reuse the
    t2v shape (no OPENING-shot note), matching the defensive fallback
    in `load_system_prompt`.
    """
    code = (mode_code or "").strip()
    if code == "i2v":
        return f"{user_text}\n{_MULTISHOT_DIRECTIVE_BASE}{_MULTISHOT_DIRECTIVE_I2V_NOTE}"
    return f"{user_text}\n{_MULTISHOT_DIRECTIVE_BASE}"
