"""Ideogram 4 prompt templates for MieNodes.

Uses official ``ideogram4_magic_prompt_v1.txt`` as the single system prompt base.
``composition_mode`` on ``Ideogram4PromptGenerator`` selects bbox strategy:

- ``simple`` — scene/collage/interior: positional desc, omit bbox (official slim path).
- ``complex`` — typography-dense poster: bbox on every element (Flow-style).

LLM output is validated and normalized by ``format_ideogram4_caption`` inside the generator.
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

USER_TEMPLATE_MAGIC_V1 = load_prompt_text("ideogram4/user_template_magic_v1")

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

COMPOSITION_SIMPLE_SUFFIX = load_prompt_text("ideogram4/composition_simple")
COMPOSITION_COMPLEX_SUFFIX = load_prompt_text("ideogram4/composition_complex")
COMPOSITION_MOVABLE_SUFFIX = load_prompt_text("ideogram4/composition_movable")
MOVABLE_SYSTEM_OVERRIDE = load_prompt_text("ideogram4/movable_system_override")

_COMPOSITION_SUFFIX = {
    "simple": COMPOSITION_SIMPLE_SUFFIX,
    "complex": COMPOSITION_COMPLEX_SUFFIX,
    "movable": COMPOSITION_MOVABLE_SUFFIX,
}

# Deprecated aliases kept for local test scripts / backward compatibility.
PROMPT_PROFILES: tuple[str, ...] = ("official_v1", "full_palette", "compact")
FULL_PALETTE_APPENDIX = load_prompt_text("ideogram4/full_palette_appendix")
COMPACT_SYSTEM_PROMPT = load_prompt_text("ideogram4/compact_system")
COMFYUI_V1_USER_SUFFIX = COMFYUI_PIPELINE_SUFFIX
COMFYUI_FULL_PALETTE_USER_SUFFIX = (
    "\n\nCOMFYUI PIPELINE: Final pixel width/height come from an external Resolution Selector. "
    "Use the aspect-ratio hint for bbox planning only — do NOT emit aspect_ratio in JSON."
)


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
    """Parse ``ideogram4_magic_prompt_v1.txt`` into [META]/[SYSTEM]/[USER] blocks."""
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


def _magic_v1_user_content(
    user_prompt: str,
    aspect_ratio: str,
    *,
    composition_mode: str = "simple",
) -> str:
    sections = load_magic_v1_sections()
    template = sections.get("user") or USER_TEMPLATE_MAGIC_V1
    user = (
        template.replace("{{aspect_ratio}}", aspect_ratio)
        .replace("{{original_prompt}}", user_prompt)
    )
    mode = (composition_mode or "simple").strip().lower()
    if mode not in COMPOSITION_MODES:
        raise ValueError(
            f"Unknown composition_mode {composition_mode!r}; "
            f"expected one of {', '.join(COMPOSITION_MODES)}"
        )
    return user + COMFYUI_PIPELINE_SUFFIX + _COMPOSITION_SUFFIX[mode]


def build_official_v1_messages(
    user_prompt: str,
    aspect_ratio: str,
    *,
    composition_mode: str = "simple",
) -> list[dict]:
    """Official Ideogram magic prompt v1 + ComfyUI pipeline + composition mode.

    In ``movable`` mode, a system-level override is appended so the renderer
    reads position only from bbox (desc / high_level_description stay free of
    placement words, letting elements move by editing bbox alone).
    """
    sections = load_magic_v1_sections()
    system = sections["system"]
    if (composition_mode or "simple").strip().lower() == "movable":
        system = system + "\n\n" + MOVABLE_SYSTEM_OVERRIDE.strip()
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _magic_v1_user_content(
                user_prompt, aspect_ratio, composition_mode=composition_mode
            ),
        },
    ]


def build_ideogram4_messages(
    user_prompt: str,
    aspect_ratio: str,
    *,
    composition_mode: str = "simple",
    prompt_profile: str | None = None,
) -> list[dict]:
    """Build LLM messages for Ideogram 4 caption generation.

    ``prompt_profile`` is deprecated; only ``official_v1`` is supported and maps to
    ``composition_mode`` when provided alone.
    """
    if prompt_profile is not None:
        profile = prompt_profile.strip().lower()
        if profile not in ("official_v1", ""):
            raise ValueError(
                f"prompt_profile {prompt_profile!r} is deprecated; "
                f"use composition_mode={composition_mode!r} with official v1 only."
            )
    return build_official_v1_messages(
        user_prompt, aspect_ratio, composition_mode=composition_mode
    )


def build_magic_v1_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Legacy alias without ComfyUI suffix (tests / direct API use)."""
    sections = load_magic_v1_sections()
    template = sections.get("user") or USER_TEMPLATE_MAGIC_V1
    user = (
        template.replace("{{aspect_ratio}}", aspect_ratio)
        .replace("{{original_prompt}}", user_prompt)
    )
    return [
        {"role": "system", "content": sections["system"]},
        {"role": "user", "content": user},
    ]


def _full_palette_user_content(user_prompt: str, aspect_ratio: str) -> str:
    if aspect_ratio.lower() == "auto":
        ar_note = (
            "Target aspect ratio hint: auto — pick a composition-suited W:H for bbox planning only. "
            "Do NOT emit aspect_ratio in JSON."
        )
    else:
        ar_note = (
            f"Target aspect ratio hint: {aspect_ratio} (bbox planning only — "
            "do NOT emit aspect_ratio in JSON)."
        )
    return f"{ar_note}\n\nUser idea: {user_prompt}{COMFYUI_FULL_PALETTE_USER_SUFFIX}"


def _compact_user_content(user_prompt: str, aspect_ratio: str) -> str:
    if aspect_ratio.lower() == "auto":
        ar_note = "Target aspect ratio hint: auto (bbox planning only)."
    else:
        ar_note = f"Target aspect ratio hint: {aspect_ratio} (bbox planning only)."
    return f"{ar_note}\n\nUser idea: {user_prompt}"


def build_full_palette_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Deprecated — kept for local comparison scripts only."""
    sections = load_magic_v1_sections()
    system = sections["system"] + "\n\n" + FULL_PALETTE_APPENDIX.strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _full_palette_user_content(user_prompt, aspect_ratio)},
    ]


def build_compact_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Deprecated — kept for local comparison scripts only."""
    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": _compact_user_content(user_prompt, aspect_ratio)},
    ]


def build_full_schema_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Backward-compatible alias → deprecated ``full_palette``."""
    return build_full_palette_messages(user_prompt, aspect_ratio)
