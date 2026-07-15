"""Krea-2 prompt-enhancement helpers for MieNodes.

Bundles the upstream ``expansion.txt`` (krea-ai/krea-2 docs) as the
system prompt and adds a small user template that injects the
target aspect ratio, mirroring the ``ideogram4_prompts`` pattern.

The Krea-2 expansion prompt is short and dependency-free: a single
T2I use case (text -> text) with no JSON contract, no composition
modes, no example bank. So this module stays intentionally small --
one system prompt, one user template, one resolver.
"""
from __future__ import annotations

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text

try:
    from _mienodes_internal.nodes.common.aspect_ratio import normalize_ratio_string
except ImportError:
    from ..common.aspect_ratio import normalize_ratio_string


# --------------------------------------------------------------------------- #
# System prompt (verbatim from krea-ai/krea-2 docs/expansion.txt)
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT_PATH = "krea2/expansion"


def load_krea2_system_prompt() -> str:
    """Return the verbatim Krea-2 ``expansion.txt`` system prompt.

    The text is loaded via the project's external-prompt loader so it
    can be patched at dev time without editing Python. Placeholders
    (none in this file) would be preserved by the loader.
    """
    return load_prompt_text(_SYSTEM_PROMPT_PATH)


# --------------------------------------------------------------------------- #
# User template
# --------------------------------------------------------------------------- #
# Per upstream ``prompting.md``: natural language prompts work best,
# long detailed prompts yield best results. So the user turn tells
# the LLM (a) the user's idea verbatim, and (b) the target aspect
# ratio for the planner. ``"auto"`` is preserved as-is -- the
# upstream Krea-2 model decides the framing.
_USER_TEMPLATE = (
    "Idea to expand into a Krea-2 image prompt:\n"
    "{user_prompt}\n\n"
    "Target aspect ratio: {aspect_ratio}\n"
    "(Use \"auto\" if the idea does not pin down an aspect ratio -- let the model choose.)"
)


def _format_user_content(user_prompt: str, aspect_ratio: str) -> str:
    """Substitute the user prompt and the resolved aspect ratio into
    ``_USER_TEMPLATE``. Strips neither end of the user's text; the
    upstream system prompt handles trimming in its own rule 7
    (respect existing detail).
    """
    return _USER_TEMPLATE.format(
        user_prompt=(user_prompt or "").strip(),
        aspect_ratio=aspect_ratio or "auto",
    )


# --------------------------------------------------------------------------- #
# Message builder
# --------------------------------------------------------------------------- #
def build_krea2_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Build the [system, user] chat pair for the Krea-2 expansion LLM.

    The system message is the upstream ``expansion.txt`` verbatim so
    the LLM receives the exact 9-rule contract. The user message
    carries the user's idea plus the resolved aspect ratio so the
    style/medium/composition planner has the framing hint.
    """
    ar = resolve_aspect_ratio(aspect_ratio)
    return [
        {"role": "system", "content": load_krea2_system_prompt()},
        {"role": "user", "content": _format_user_content(user_prompt, ar)},
    ]


# --------------------------------------------------------------------------- #
# Aspect-ratio helpers
# --------------------------------------------------------------------------- #
DEFAULT_ASPECT_RATIO = "1:1"


def resolve_aspect_ratio(value, fallback: str = DEFAULT_ASPECT_RATIO) -> str:
    """Normalize an aspect-ratio hint for the Krea-2 user template.

    Accepts ``W:H`` (kept as-is after gcd reduction), ``auto`` (kept
    verbatim so the upstream model can decide), and the empty string
    (falls back to ``DEFAULT_ASPECT_RATIO``). Garbage that does not
    parse as ``W:H`` also falls back rather than echoing through,
    because the upstream system prompt expects a valid ``W:H`` or
    ``auto`` and a malformed ratio would degrade planning quality.
    """
    normalized = normalize_ratio_string(str(value or "").strip())
    if not normalized:
        return fallback
    if normalized.lower() == "auto":
        return "auto"
    if ":" in normalized:
        return normalized
    return fallback
