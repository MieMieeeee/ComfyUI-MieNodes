"""Ideogram 4 prompt templates for MieNodes.

Each composition mode (``simple`` / ``complex`` / ``movable``) has its own complete
system prompt file — ``_system_simple.txt`` / ``_system_complex.txt`` /
``_system_movable.txt`` — built by appending the mode-specific directives to the
official ``magic_prompt_v1.txt`` [SYSTEM] block. ``Ideogram4PromptGenerator`` picks
one verbatim as the system message; no runtime concatenation. Mode directives live
in the system role so the model treats them as hard rules.

- ``simple`` — scene/collage/interior: positional desc, omit bbox.
- ``complex`` — typography-dense poster: bbox on every element (Flow-style).
- ``movable`` — editable layout: bbox is the sole position authority; desc and
  high_level_description carry no placement words, so elements move by editing bbox alone.

LLM output is validated and normalized by ``format_ideogram4_caption`` inside the generator.

All fragment files are ``_``-prefixed so the loader keeps them out of the generic
CustomSystemPromptGenerator dropdown — Ideogram 4 must be driven through
``Ideogram4PromptGenerator``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from _mienodes_internal.nodes.common.aspect_ratio import normalize_ratio_string
except ImportError:
    from ..common.aspect_ratio import normalize_ratio_string

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text

_PROMPT_DIR = Path(__file__).resolve().parent
_MAGIC_V1_FILE = _PROMPT_DIR / "prompts" / "ideogram4" / "magic_prompt_v1.txt"

USER_TEMPLATE_MAGIC_V1 = load_prompt_text("ideogram4/_user_template_magic_v1")

COMPOSITION_MODES: tuple[str, ...] = ("simple", "complex", "movable")

COMPOSITION_MODE_TOOLTIPS: dict[str, str] = {
    "simple": (
        "Scene/collage/interior/magazine hero — slim JSON, positional desc, omit bbox "
        "(matches most official Ideogram examples)."
    ),
    "complex": (
        "Typography-dense poster — bbox on every element for precise multi-zone layout "
        "(Flow / T-Rex style)."
    ),
    "movable": (
        "Editable layout — bbox is the sole position authority. desc and "
        "high_level_description carry NO placement words, so you can move any "
        "element by editing its bbox alone."
    ),
}

COMFYUI_PIPELINE_SUFFIX = (
    "\n\nCOMFYUI PIPELINE: Final pixel width/height come from an external Resolution Selector. "
    "Still emit top-level aspect_ratio exactly as required above — it drives bbox planning. "
    "The generator strips aspect_ratio before Ideogram sampling."
)

SYSTEM_SIMPLE = load_prompt_text("ideogram4/_system_simple")
SYSTEM_COMPLEX = load_prompt_text("ideogram4/_system_complex")
SYSTEM_MOVABLE = load_prompt_text("ideogram4/_system_movable")

_SYSTEM_PROMPTS = {
    "simple": SYSTEM_SIMPLE,
    "complex": SYSTEM_COMPLEX,
    "movable": SYSTEM_MOVABLE,
}


def resolve_aspect_ratio(value: str, *, fallback: str = "1:1") -> str:
    """Normalize ratio text for LLM hints; any valid ``W:H`` is kept as-is (no preset snapping)."""
    normalized = normalize_ratio_string(value)
    if not normalized:
        return fallback
    if normalized.lower() == "auto":
        return "auto"
    if ":" in normalized:
        return normalized
    return fallback


@lru_cache(maxsize=None)
def load_magic_v1_sections() -> dict[str, str]:
    """Parse ``ideogram4_magic_prompt_v1.txt`` into [META]/[SYSTEM]/[USER] blocks.

    Used at runtime to fetch the [USER] template (double-brace placeholders). The
    [SYSTEM] block is baked into each ``_system_*.txt`` at dev time, not read here.
    """
    raw = _MAGIC_V1_FILE.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[1:-1].strip().lower()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    if "system" not in sections:
        raise ValueError(f"{_MAGIC_V1_FILE.name} has no [SYSTEM] section")
    return sections


def _magic_v1_user_content(user_prompt: str, aspect_ratio: str) -> str:
    sections = load_magic_v1_sections()
    template = sections.get("user") or USER_TEMPLATE_MAGIC_V1
    user = (
        template.replace("{{aspect_ratio}}", aspect_ratio)
        .replace("{{original_prompt}}", user_prompt)
    )
    return user + COMFYUI_PIPELINE_SUFFIX


def build_official_v1_messages(
    user_prompt: str,
    aspect_ratio: str,
    *,
    composition_mode: str = "simple",
) -> list[dict]:
    """Official Ideogram magic prompt v1 system + ComfyUI pipeline user content.

    Each composition mode resolves to one complete system prompt file
    (``_system_{mode}.txt``): the [SYSTEM] block plus mode-specific directives, all
    in the system role so the model treats them as hard rules.
    """
    mode = (composition_mode or "simple").strip().lower()
    if mode not in _SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown composition_mode {composition_mode!r}; "
            f"expected one of {', '.join(COMPOSITION_MODES)}"
        )
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[mode]},
        {"role": "user", "content": _magic_v1_user_content(user_prompt, aspect_ratio)},
    ]


def build_ideogram4_messages(
    user_prompt: str,
    aspect_ratio: str,
    *,
    composition_mode: str = "simple",
) -> list[dict]:
    """Build LLM messages for Ideogram 4 caption generation."""
    return build_official_v1_messages(
        user_prompt, aspect_ratio, composition_mode=composition_mode
    )
