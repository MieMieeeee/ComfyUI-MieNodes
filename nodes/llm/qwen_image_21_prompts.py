"""Qwen-Image-2.1 prompt-enhancer helpers (loader map + block extractor).

Companion module to ``qwen_image_21_prompt_generator.py``. Holds:

* ``MODES``: ordered tuple of dropdown entries. ``remove_bg`` is included
  even though the node short-circuits it (no LLM call) — see
  ``nodes/llm/prompts/qwen_image_21/_enhance_remove_bg.txt`` for the
  documentation block the node emits verbatim.
* ``_MODE_TO_PROMPT``: logical ``load_prompt_text`` name for each LLM-driven
  mode. ``remove_bg`` maps to ``None`` so the helper raises a loud error
  if the short-circuit is bypassed.
* ``_BLOCK_RE`` / ``_THINK_BLOCK_RE``: BEGIN/END extractor identical in
  spirit to ``minimax_h3_loop_user_input_enhancer._USER_INPUT_BLOCK_RE``
  but with the Qwen-specific block markers.
* ``extract_enhanced_prompt_block`` / ``split_enhancer_reply``: parser
  surface used by the node and the tests.
* ``CAPTION_MODES`` + ``caption_mode_code()``: cache-strategy dropdown
  mirroring ``minimax_h3_loop_prompt_generator.CAPTION_MODES``.
* ``caption_image_prompt()``: loader for the bundled caption system prompt
  (``qwen_image_21/_caption_image``). Used by the node to drive per-image
  captioning before the rewrite LLM call.
* ``caption_cache_key()`` / ``caption_cache_disk_root`` /
  ``read_caption_cache_disk`` / ``write_caption_cache_disk``: disk cache
  plumbing mirroring the H3 Loop captioner.
* ``scrub_caption_sheet_language``: strips turnaround-sheet layout,
  backdrop, and sheet lighting from a caption before it is locked as DATA.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text


# Ordered dropdown entries shown in the node menu. Bilingual
# ``code - 中文(标签)`` format mirrors ``MiniMaxH3LoopPromptGenerator``
# dropdown conventions (see _PACING_LABELS / _ENHANCE_USER_INPUT_LABELS /
# GENERATION_MODES in that file) so users can pick the right mode in
# either language. The code is what the node + tests branch on;
# the parenthetical is purely for the menu.
MODES: tuple[str, ...] = (
    "t2i - 文生图(默认)",
    "t2i_rgba - 透明文生图",
    "edit - 单图编辑",
    "multi_ref - 多参考编辑",
    "remove_bg - 抠图(无 LLM 调用)",
)

# Logical ``load_prompt_text`` name per mode. ``None`` marks the
# short-circuited mode so an accidental LLM call raises a clear error.
# Keys here are the bare code prefix (everything before the " - "),
# which is what ``parse_mode_label`` and the node branch on.
_MODE_TO_PROMPT: dict[str, str | None] = {
    "t2i": "qwen_image_21/_enhance_t2i",
    "t2i_rgba": "qwen_image_21/_enhance_t2i_rgba",
    "edit": "qwen_image_21/_enhance_edit",
    "multi_ref": "qwen_image_21/_enhance_multi_ref",
    "remove_bg": None,
}


def parse_mode_label(label: str) -> str:
    """Strip the `` - 中文`` suffix off a mode dropdown label and return
    the bare code (``t2i``, ``t2i_rgba``, etc.). Unknown / empty
    labels round-trip to ``""`` so callers can branch safely."""
    code = (label or "").split(" - ", 1)[0].strip()
    return code if code in _MODE_TO_PROMPT else ""

# Tolerant block extractor. Anchored on the literal BEGIN/END tokens;
# ``re.DOTALL`` so the captured block may span newlines. Mirrors the
# H3 Loop enhancer's regex.
_BLOCK_RE = re.compile(
    r"--- BEGIN enhanced_prompt ---\s*(.*?)\s*--- END enhanced_prompt ---",
    re.DOTALL,
)
# Strip reasoning-model ``<think>...</think>`` wrappers so they don't
# sneak into the captured block content.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Caption mode dropdown
# --------------------------------------------------------------------------- #
CAPTION_MODES: tuple[str, ...] = (
    "cache_memory_disk - 缓存:内存+磁盘(推荐)",
    "cache_memory_only - 缓存:仅内存",
    "no_cache - 禁用缓存",
    "force_recaption_once - 本次强制重打标",
)
CAPTION_MODE_CODES: tuple[str, ...] = (
    "cache_memory_disk",
    "cache_memory_only",
    "no_cache",
    "force_recaption_once",
)


def caption_mode_code(mode_label: str) -> str:
    """Return the canonical code from a dropdown label, or ``""`` if
    the value is unrecognised (mirrors ``minimax_h3_loop_prompt_generator
    .parse_caption_mode``)."""
    code = (mode_label or "").split(" - ", 1)[0].strip()
    return code if code in CAPTION_MODE_CODES else ""


def resolve_caption_controls(
    caption_mode: str,
) -> tuple[bool, str]:
    """Map the dropdown value to ``(force_recaption, cache_scope)``.

    Returns ``(force, scope)`` where ``scope`` is one of
    ``{"memory_disk", "memory_only", "disabled"}`` and ``force`` is a
    bool that bypasses the cache for a single call. Mirrors
    ``minimax_h3_loop_prompt_generator.resolve_caption_controls``.
    """
    code = caption_mode_code(caption_mode)
    if code == "cache_memory_disk":
        return False, "memory_disk"
    if code == "cache_memory_only":
        return False, "memory_only"
    if code == "no_cache":
        return False, "disabled"
    if code == "force_recaption_once":
        return True, "disabled"
    return False, "memory_disk"


def caption_image_prompt() -> str:
    """System prompt used to caption a single reference image.

    Same ``load_prompt_text`` shape as the rewrite system prompts; the
    caption model is fed the image plus a fixed user message and emits
    one tight English paragraph consumed verbatim by the rewrite stage.
    """
    return load_prompt_text("qwen_image_21/_caption_image")


# --------------------------------------------------------------------------- #
# Caption disk cache
# --------------------------------------------------------------------------- #
def caption_cache_key(image_url: str, prompt_text: str) -> str:
    """Hash an image data URL + the caption prompt into a stable cache key.

    The image URL embeds the full base64 of the pixel bytes, so hashing
    the URL is equivalent to hashing the pixel payload. The prompt-text
    hash invalidates the cache whenever ``caption_image_prompt()`` is
    upgraded.
    """
    h = hashlib.sha256()
    h.update((image_url or "").encode("utf-8"))
    h.update(b"|")
    h.update(
        hashlib.sha256((prompt_text or "").encode("utf-8"))
        .hexdigest()
        .encode("ascii")
    )
    return h.hexdigest()


def caption_cache_disk_root() -> str:
    """Resolve the on-disk caption cache directory.

    Resolution order:
      1) ComfyUI's runtime output dir via ``folder_paths.get_output_directory()``.
      2) Fallback ``<repo>/output/mienodes/caption_cache`` for tests /
         standalone import contexts where ``folder_paths`` is unavailable.

    Mirrors ``minimax_h3_loop_prompt_generator._caption_cache_disk_root``.
    """
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.join(
                os.path.abspath(str(output_dir)),
                "mien_nodes",
                "caption_cache",
            )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    # __file__ is nodes/llm/<thisfile>.py; walk up two to the repo root.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "output", "mien_nodes", "caption_cache")


def read_caption_cache_disk(root: str, key: str) -> Optional[str]:
    """Read a cached caption by sha256 key; ``None`` on cache miss / I/O
    error (caller treats both as a miss and re-runs the LLM)."""
    path = os.path.join(root, f"{key}.txt")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


# Turnaround-sheet phrases that must not reach the rewrite as locked
# DATA. Prompt rules are probabilistic; this pass is the guarantee and
# runs on the way into the cache and again when a cached caption is
# injected, so older disk entries are cleaned on use. Identity and
# wardrobe sentences are left in place. If stripping would erase the
# whole caption, the original is kept.
_CAPTION_SHEET_PHRASES: tuple = (
    (re.compile(
        r",?\s*shown\s+from\s+(?:side[,\s]*|front[,\s]*|back[,\s]*|and\s+|"
        r"three\s+|four\s+|multiple\s+|several\s+|various\s+|[a-z]+\s+)*"
        r"(?:angles|views|perspectives)",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*(?:in\s+)?(?:two|three|four|five|multiple|several|various|\d+)\s+"
        r"(?:different\s+|full-body\s+|full\s+body\s+)*"
        r"(?:views|angles|perspectives)",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*(?:turnaround|model|character)\s+sheet",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*(?:against\s+a\s+|on\s+a\s+)?(?:seamless\s+|pure\s+|light\s+"
        r"gray\s+|light\s+grey\s+)?white(?:[-/]\w+)?\s+"
        r"(?:studio\s+)?(?:background|backdrop)",
        re.IGNORECASE,
    ), ""),
    (re.compile(r",?\s*studio\s+backdrop", re.IGNORECASE), ""),
    (re.compile(
        r",?\s*(?:arranged\s+)?side[- ]by[- ]side",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*(?:and\s+)?(?:left|center|centre|right|middle|"
        r"far[- ]left|far[- ]right)\s+panel",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*under\s+(?:neutral|bright|soft|even|warm)\s+"
        r"(?:even\s+)?[a-z\s]{0,24}lighting",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*(?:bright\s+|soft\s+|even\s+)*even\s+frontal\s+lighting",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*(?:that\s+casts\s+a?\s*)?(?:a\s+)?(?:soft|gentle)\s+shadow\s+"
        r"beneath\s+the\s+(?:feet|paws|character)",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*standing\s+with\s+(?:her|his|their|both\s+)?arms\s+"
        r"(?:at|by)\s+(?:her|his|their)?\s*sides",
        re.IGNORECASE,
    ), ""),
    (re.compile(
        r",?\s*on\s+all\s+four\s+feet|,?\s*on\s+all\s+fours\b",
        re.IGNORECASE,
    ), ""),
)


def scrub_caption_sheet_language(about: str) -> str:
    """Strip turnaround-sheet layout, backdrop, and sheet lighting.

    Phrase-level: wardrobe and identity survive. Never returns empty
    when ``about`` itself was non-empty.
    """
    text = str(about or "").strip()
    if not text:
        return text
    out = text
    for pattern, repl in _CAPTION_SHEET_PHRASES:
        out = pattern.sub(repl, out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;])", r"\1", out)
    out = re.sub(r"(?<![.!?])\s*,\s*([.;])", r"\1", out)
    out = re.sub(r"[,;]\s*\.", ".", out)
    out = re.sub(r"\.\s*\.", ".", out)
    out = out.strip(" ,;")
    return out or text


def write_caption_cache_disk(root: str, key: str, about: str) -> Optional[str]:
    """Write a caption to disk; returns ``None`` on success or an error
    string on failure. Non-fatal: the in-memory tier still serves the
    current node lifetime if disk writes fail.
    """
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        return f"mkdir failed: {exc}"
    path = os.path.join(root, f"{key}.txt")
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(about)
        os.replace(tmp, path)
        return None
    except OSError as exc:
        err = f"write failed: {exc}"
        try:
            os.unlink(tmp)
        except OSError as cleanup_exc:
            err = f"{err}; tmp cleanup failed: {cleanup_exc}"
        return err


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def _split(raw_reply: str) -> tuple[str | None, str]:
    """Split a well-formed enhancer reply into ``(enhanced_prompt block,
    advice header)``.

    The header is everything before the BEGIN marker (one-line Notes,
    ``<think>`` blocks stripped). Returns ``(None, "")`` when the markers
    are missing — same tolerance rules as ``extract_enhanced_prompt_block``.
    """
    if not raw_reply:
        return None, ""
    text = _THINK_BLOCK_RE.sub("", raw_reply)
    idx = text.find("--- BEGIN enhanced_prompt ---")
    if idx < 0:
        return None, ""
    header = text[:idx].strip()
    match = _BLOCK_RE.search(text)
    block = match.group(1).strip() if match is not None else None
    return block, header


def extract_enhanced_prompt_block(raw_reply: str) -> str | None:
    """Pull the ``--- BEGIN enhanced_prompt ---`` ... ``--- END enhanced_prompt ---``
    block out of the LLM reply. Returns the trimmed inner text, or ``None``
    if the markers are missing.

    Tolerates a leading ``<think>...</think>`` wrapper (reasoning models)
    and any prose (one-line Notes / chatter) around the block.
    """
    return _split(raw_reply)[0]


def split_enhancer_reply(raw_reply: str) -> tuple[str | None, str]:
    """Split a well-formed enhancer reply into ``(enhanced_prompt block,
    advice header)`` — the Qwen-Image-2.1 equivalent of
    ``minimax_h3_loop_user_input_enhancer.split_enhancer_reply``. The node
    itself only reads the block; the helper is exposed for the test suite
    and for any future surface that wants the header (e.g. a preflight
    report).
    """
    return _split(raw_reply)


def resolve_prompt_name(mode: str) -> str | None:
    """Return the ``load_prompt_text`` logical name for ``mode``.

    Raises ``KeyError`` if ``mode`` is not in ``MODES``. Returns ``None``
    for ``remove_bg`` (the node short-circuits that mode and never calls
    the loader).
    """
    return _MODE_TO_PROMPT[mode]


def load_mode_prompt(mode: str) -> str:
    """Load the bundled system prompt for an LLM-driven ``mode``.

    Raises ``KeyError`` for unknown modes and ``FileNotFoundError`` (from
    ``load_prompt_text``) for missing files. Callers should treat the
    latter as a hard configuration error — the node raises ``RuntimeError``
    with a clear message rather than silently falling back.
    """
    name = resolve_prompt_name(mode)
    if name is None:
        raise FileNotFoundError(
            "qwen_image_21: mode 'remove_bg' has no bundled system prompt "
            "(the node short-circuits this mode)"
        )
    return load_prompt_text(name)


__all__ = [
    "CAPTION_MODES",
    "CAPTION_MODE_CODES",
    "MODES",
    "caption_cache_disk_root",
    "caption_cache_key",
    "caption_image_prompt",
    "caption_mode_code",
    "extract_enhanced_prompt_block",
    "load_mode_prompt",
    "parse_mode_label",
    "read_caption_cache_disk",
    "resolve_caption_controls",
    "resolve_prompt_name",
    "scrub_caption_sheet_language",
    "split_enhancer_reply",
    "write_caption_cache_disk",
]