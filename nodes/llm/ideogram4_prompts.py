"""Ideogram 4 prompt templates for MieNodes.

Profiles (``prompt_profile`` on ``Ideogram4PromptGenerator``):

- ``official_v1`` — official ``ideogram4_magic_prompt_v1.txt`` system prompt (strong LLMs).
- ``full_palette`` — v1 content strategy + full-schema ``style_description`` / ``color_palette``.
- ``compact`` — short prompt for light LLMs; still KJ/Formatter compatible.

All profiles emit JSON that ``Ideogram4PromptFormatter`` can normalize for sampling and
``Ideogram4PromptBuilderKJ`` ``import_json``.
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

PROMPT_PROFILES: tuple[str, ...] = ("official_v1", "full_palette", "compact")

PROMPT_PROFILE_TOOLTIPS: dict[str, str] = {
    "official_v1": (
        "Official Ideogram magic prompt v1 (~7k tokens). Best with Opus-class LLMs. "
        "Emits aspect_ratio + HLD + compositional_deconstruction; Formatter strips aspect_ratio."
    ),
    "full_palette": (
        "Official v1 content rules + structured style_description and color_palette fields. "
        "For medium/strong LLMs when you need palette steering."
    ),
    "compact": (
        "Short schema + core rules (~2k tokens). For light/cheap LLMs; optional style_description."
    ),
}

COMFYUI_V1_USER_SUFFIX = (
    "\n\nCOMFYUI PIPELINE: Final pixel width/height come from an external Resolution Selector. "
    "Still emit top-level aspect_ratio exactly as required above — it drives bbox planning. "
    "A downstream Formatter may remove aspect_ratio before Ideogram sampling."
)

COMFYUI_FULL_PALETTE_USER_SUFFIX = (
    "\n\nCOMFYUI PIPELINE: Final pixel width/height come from an external Resolution Selector. "
    "Use the aspect-ratio hint for bbox planning only — do NOT emit aspect_ratio in JSON."
)

FULL_PALETTE_APPENDIX = load_prompt_text("ideogram4/full_palette_appendix")

COMPACT_SYSTEM_PROMPT = load_prompt_text("ideogram4/compact_system")


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


def _magic_v1_user_content(user_prompt: str, aspect_ratio: str, *, suffix: str = "") -> str:
    sections = load_magic_v1_sections()
    template = sections.get("user") or USER_TEMPLATE_MAGIC_V1
    user = (
        template.replace("{{aspect_ratio}}", aspect_ratio)
        .replace("{{original_prompt}}", user_prompt)
    )
    return user + suffix


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


def build_official_v1_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Official Ideogram magic prompt v1 + ComfyUI pipeline user note."""
    sections = load_magic_v1_sections()
    return [
        {"role": "system", "content": sections["system"]},
        {
            "role": "user",
            "content": _magic_v1_user_content(
                user_prompt, aspect_ratio, suffix=COMFYUI_V1_USER_SUFFIX
            ),
        },
    ]


def build_full_palette_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Official v1 content strategy + structured style_description / color_palette."""
    sections = load_magic_v1_sections()
    system = sections["system"] + "\n\n" + FULL_PALETTE_APPENDIX.strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _full_palette_user_content(user_prompt, aspect_ratio)},
    ]


def build_compact_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Short prompt for light LLMs; Formatter/KJ compatible."""
    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": _compact_user_content(user_prompt, aspect_ratio)},
    ]


def build_ideogram4_messages(
    user_prompt: str,
    aspect_ratio: str,
    *,
    prompt_profile: str = "official_v1",
) -> list[dict]:
    """Build LLM messages for the selected Ideogram 4 caption profile."""
    profile = (prompt_profile or "official_v1").strip().lower()
    if profile not in PROMPT_PROFILES:
        raise ValueError(
            f"Unknown prompt_profile {prompt_profile!r}; "
            f"expected one of {', '.join(PROMPT_PROFILES)}"
        )
    builders = {
        "official_v1": build_official_v1_messages,
        "full_palette": build_full_palette_messages,
        "compact": build_compact_messages,
    }
    return builders[profile](user_prompt, aspect_ratio)


def build_magic_v1_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Legacy alias without ComfyUI suffix (tests / direct API use)."""
    sections = load_magic_v1_sections()
    return [
        {"role": "system", "content": sections["system"]},
        {"role": "user", "content": _magic_v1_user_content(user_prompt, aspect_ratio)},
    ]


def build_full_schema_messages(user_prompt: str, aspect_ratio: str) -> list[dict]:
    """Backward-compatible alias → ``full_palette`` profile."""
    return build_full_palette_messages(user_prompt, aspect_ratio)
