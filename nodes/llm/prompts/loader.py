"""External system-prompt loader for MieNodes LLM nodes.

Prompts live as plain ``.txt`` / ``.json`` files under this package directory
(``nodes/llm/prompts/``). Text files are returned **verbatim** — placeholders
like ``{}``, ``{prompt}``, ``{image_num}`` survive untouched so callers can
``.format()`` / ``.replace()`` exactly as they did when prompts were inline
string constants. The loader never applies ``.format()``.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_ROOT = Path(__file__).resolve().parent

# The H3, H3 loop and storyboard prompt families moved to
# ComfyUI-Mie-Core, because the plan builder that reads them is shared with
# the H3 engine. This loader still serves every other family, and it falls
# back to the core so that a prompt that lives there is still found - and
# still listed in the dropdown.
_CORE_PROMPTS_ROOT = None


def _core_prompts_root():
    global _CORE_PROMPTS_ROOT
    if _CORE_PROMPTS_ROOT is None:
        try:
            from mie_prompt.prompts import prompt_root
            _CORE_PROMPTS_ROOT = prompt_root()
        except ImportError:
            _CORE_PROMPTS_ROOT = Path()
    return _CORE_PROMPTS_ROOT

# Matches a str.format-style placeholder only: "{}" or "{identifier}" such as
# "{prompt}", "{image_num}", "{language_name}". Deliberately does NOT match JSON
# embedded in a prompt (e.g. {"camera": {...}}), so a prompt that merely
# illustrates JSON output is still offered in the generic dropdown.
# Used to keep placeholder-bearing prompts out of CustomSystemPromptGenerator,
# since those need caller-specific kwargs and only make sense in their nodes.
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_]*\}")


def _resolve(logical_name: str, suffix: str) -> Path:
    name = f"{logical_name.replace('.', '/')}{suffix}"
    local = _PROMPTS_ROOT / name
    if local.is_file():
        return local
    shared = _core_prompts_root() / name
    if shared.is_file():
        return shared
    return local


@lru_cache(maxsize=None)
def load_prompt_text(logical_name: str) -> str:
    """Load a ``.txt`` prompt by logical name, e.g. ``load_prompt_text("hunyuan/t2v")``.

    Returns the raw file text (utf-8, universal newlines). Placeholders are
    preserved verbatim. Raises ``FileNotFoundError`` if the file is missing.
    """
    path = _resolve(logical_name, ".txt")
    if not path.is_file():
        raise FileNotFoundError(f"prompt text not found: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_prompt_dict(logical_name: str) -> dict:
    """Load a ``.json`` prompt dict by logical name, e.g. ``load_prompt_dict("kontext/presets")``."""
    path = _resolve(logical_name, ".json")
    if not path.is_file():
        raise FileNotFoundError(f"prompt dict not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def _txt_logical_names() -> list[str]:
    names: set[str] = set()
    for root in (_PROMPTS_ROOT, _core_prompts_root()):
        if not root or not root.is_dir():
            continue
        for p in sorted(root.glob("**/*.txt")):
            if p.name.startswith("_"):  # _t2i_note.txt etc. are build blocks
                continue
            rel = p.relative_to(root).with_suffix("")
            names.add(str(rel).replace("\\", "/"))
    return sorted(names)


def list_builtin_prompts() -> list[str]:
    """All builtin prompt logical names (``.txt`` only; ``.json`` dicts are not
    single system prompts and are excluded). Sorted, build-blocks (``_``-prefixed) skipped."""
    return _txt_logical_names()


def list_usable_builtin_prompts() -> list[str]:
    """Builtin ``.txt`` prompts safe to use verbatim as a generic system prompt.

    Excludes prompts containing ``{placeholder}`` tokens (bernini task templates,
    ``zimage/t2i`` ``{prompt}``, ``hunyuan/i2v`` ``{}``) — those need
    caller-specific kwargs and only make sense in their dedicated nodes.
    """
    usable: list[str] = []
    for name in _txt_logical_names():
        if _PLACEHOLDER_RE.search(load_prompt_text(name)):
            continue
        usable.append(name)
    return usable


def reload_prompt(logical_name: str | None = None) -> None:
    """Clear the lru_cache. Tests / dev only — builtin files are read-only at runtime."""
    load_prompt_text.cache_clear()
    load_prompt_dict.cache_clear()
