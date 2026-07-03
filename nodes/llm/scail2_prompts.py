"""SCAIL-2 prompt templates and message-building helpers.

Two tasks are supported:

- ``character_replacement`` -- verbatim port of upstream's
  https://github.com/zai-org/SCAIL-2/blob/wan-scail2/prompt_enhancer.py
  (two-stage: caption source video, then rewrite as a post-replacement
  positive prompt).

- ``motion_transfer`` -- MieNodes original, since upstream has no prompt
  enhancer for the animation mode. The canonical upstream animation input
  is the 4-word prompt "the girl is dancing" (see
  ``examples/input.txt``); we provide a two-stage version that derives
  the prompt from a driving video + a reference image, with an optional
  user hint.

The strings themselves live as ``.txt`` files under
``nodes/llm/prompts/scail2/`` and are loaded via ``prompts/loader.py``
to keep the Python surface small.
"""
from __future__ import annotations

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text


# --------------------------------------------------------------------------- #
# Task list (display strings for the ComfyUI dropdown)
# --------------------------------------------------------------------------- #
TASK_TYPES = (
    "character_replacement - 角色替换",
    "motion_transfer - 动作迁移",
)

TASK_CODES = (
    "character_replacement",
    "motion_transfer",
)

# Mirrors upstream ``--max_example_chars`` default in prompt_enhancer.py.
MAX_EXAMPLE_CHARS = 4000


# --------------------------------------------------------------------------- #
# Replacement-mode templates (upstream verbatim)
# --------------------------------------------------------------------------- #
_CAPTION_REPLACEMENT_PATH = "scail2/caption_replacement"
_ENHANCE_REPLACEMENT_PATH = "scail2/enhance_replacement"
_EXAMPLES_REPLACEMENT_PATH = "scail2/examples_replacement"


def caption_replacement_prompt() -> str:
    """Stage 1 system prompt: caption the source video for replacement.

    Verbatim copy of ``VIDEO_CAPTION_PROMPT`` from upstream
    ``prompt_enhancer.py``.
    """
    return load_prompt_text(_CAPTION_REPLACEMENT_PATH)


def enhance_replacement_prompt(instruction: str, caption: str, examples: str) -> str:
    """Stage 2 user text: rewrite the caption as a post-replacement prompt.

    Substitutes ``{instruction}``, ``{caption}``, ``{examples}`` into
    the upstream ``REPLACEMENT_PROMPT_TEMPLATE``.
    """
    template = load_prompt_text(_ENHANCE_REPLACEMENT_PATH)
    return template.format(
        instruction=(instruction or "").strip(),
        caption=(caption or "").strip(),
        examples=(examples or "(No examples provided.)").strip(),
    )


def bundled_examples_replacement(max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Return the bundled few-shot examples for replacement mode, truncated."""
    text = load_prompt_text(_EXAMPLES_REPLACEMENT_PATH).strip()
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# Motion-transfer templates (MieNodes original)
# --------------------------------------------------------------------------- #
_CAPTION_MOTION_TRANSFER_PATH = "scail2/caption_motion_transfer"
_ENHANCE_MOTION_TRANSFER_PATH = "scail2/enhance_motion_transfer"
_EXAMPLES_MOTION_TRANSFER_PATH = "scail2/examples_motion_transfer"


def caption_motion_transfer_prompt() -> str:
    """Stage 1 system prompt: caption the driving video for motion transfer."""
    return load_prompt_text(_CAPTION_MOTION_TRANSFER_PATH)


def enhance_motion_transfer_prompt(caption: str, user_hint: str, examples: str) -> str:
    """Stage 2 user text: rewrite the driving caption as an animation prompt.

    ``user_hint`` may be empty; the template explicitly handles that case
    and the caller should pass an empty string when the user did not
    supply one.
    """
    template = load_prompt_text(_ENHANCE_MOTION_TRANSFER_PATH)
    hint_clean = (user_hint or "").strip()
    return template.format(
        caption=(caption or "").strip(),
        user_hint=hint_clean or "No additional user hint; derive the motion purely from the driving-video caption above.",
        examples=(examples or "(No examples provided.)").strip(),
    )


def bundled_examples_motion_transfer(max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Return the bundled few-shot examples for motion-transfer mode, truncated."""
    text = load_prompt_text(_EXAMPLES_MOTION_TRANSFER_PATH).strip()
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_task_code(task_type: str) -> str:
    """Extract the short task code from a display string.

    Accepts:
      - new display strings like ``"motion_transfer - 动作迁移"``
      - the legacy bare code ``"motion_transfer"`` (saved workflows)
      - None / empty (passed through unchanged)
    """
    if not task_type:
        return task_type
    return task_type.split(" - ", 1)[0].strip()


def load_bundled_examples(task_code: str, max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Return the bundled few-shot examples for the given task code."""
    if task_code == "character_replacement":
        return bundled_examples_replacement(max_chars)
    if task_code == "motion_transfer":
        return bundled_examples_motion_transfer(max_chars)
    return ""