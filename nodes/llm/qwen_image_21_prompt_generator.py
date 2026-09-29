"""Qwen-Image-2.1 prompt-enhancer ComfyUI node.

Takes a rough user draft (any shape — a sentence, an outline, a few
lines) and rewrites it into a prompt the Qwen-Image-2.1 model consumes
directly. The rewrite is driven by one of five bundled system prompts
under ``nodes/llm/prompts/qwen_image_21/``:

* ``t2i``       — observer English paragraph (PE-T2I 8-step + MieNodes wrapper)
* ``t2i_rgba``  — observer English paragraph with the RGBA bookend sandwich
                  forced on regardless of whether the user said "transparent"
* ``edit``      — single-image edit instruction (Edit Enhancer v2 + Space /
                   MieNodes hardening). The rewrite body is always English
                   regardless of the user's input language.
* ``multi_ref`` — multi-reference edit instruction (Edit Enhancer v2 +
                   multi-image slot bookkeeping; ``<image1>``…``<imageN>``
                   only, body always English).
* ``remove_bg`` — node short-circuits, returns a fixed BG-remove block; no
                   LLM call. The .txt file is documentation only.

The LLM reply is expected in the form:

    --- BEGIN enhanced_prompt ---
    <rewritten text>
    --- END enhanced_prompt ---

The node tolerates ``<think>...</think>`` wrappers (reasoning models) and
an optional one-line Notes header, then extracts the block with the
same regex shape as ``MiniMaxH3LoopUserInputEnhancer``. If the block is
missing the node raises ``RuntimeError`` rather than silently feeding
the user the raw LLM reply — silent fallback would let a misclassified
rewrite flow into Qwen-Image-2.1 and waste the whole pipeline.

When ``reference_images`` is connected and ``mode in {edit, multi_ref}``
the node first runs a per-image **captioning** stage (mirroring
``minimax_h3_loop_prompt_generator._caption_images``):

    - The caption system prompt lives at
      ``nodes/llm/prompts/qwen_image_21/_caption_image.txt``.
    - Each image is encoded to a JPEG data URL via
      ``core.utils.image_tensor_batch_to_data_urls`` and fed to the same
      LLM connector with the system prompt + image_url content parts.
    - Captions are cached in two tiers: in-memory for the lifetime of
      the node instance, on-disk under
      ``<ComfyUI output>/mien_nodes/caption_cache/<sha256>.txt`` for
      cross-session persistence. The cache key is
      ``sha256(image_url) | sha256(caption prompt)`` so any prompt
      upgrade invalidates the cache atomically.

The captions are then injected into the rewrite user message as
DATA (not as instructions): each ``<imageN>`` slot gets its locked
identity / wardrobe / accessories / palette description, so the rewrite
LLM doesn't have to guess. Captions are stripped of reasoning-model
``<think>`` wrappers before being injected.

Wire this node's ``enhanced_prompt`` output straight into the Qwen-Image-2.1
sampler node's prompt widget — the output is a plain STRING.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Optional

try:
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        CAPTION_MODES,
        MODES,
        caption_cache_disk_root,
        caption_cache_key,
        caption_image_prompt,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        extract_enhanced_prompt_block,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        load_mode_prompt,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        parse_mode_label,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        read_caption_cache_disk,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        resolve_caption_controls,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        scrub_caption_sheet_language,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        split_enhancer_reply,
    )
    from _mienodes_internal.nodes.llm.qwen_image_21_prompts import (
        write_caption_cache_disk,
    )
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    from .qwen_image_21_prompts import (
        CAPTION_MODES,
        MODES,
        caption_cache_disk_root,
        caption_cache_key,
        caption_image_prompt,
    )
    from .qwen_image_21_prompts import extract_enhanced_prompt_block
    from .qwen_image_21_prompts import load_mode_prompt
    from .qwen_image_21_prompts import parse_mode_label
    from .qwen_image_21_prompts import read_caption_cache_disk
    from .qwen_image_21_prompts import resolve_caption_controls
    from .qwen_image_21_prompts import scrub_caption_sheet_language
    from .qwen_image_21_prompts import split_enhancer_reply
    from .qwen_image_21_prompts import write_caption_cache_disk
    from ...core.utils import mie_log


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Default sampling knobs. ``temperature=0.7`` sits between the H3 loop's
# structured-output 0.4 and Ideogram4's 1.0 — image-prompt expansion is
# prose, not classification, so we want enough surface variation while
# still respecting the system-prompt rules.
_DEFAULT_TEMPERATURE = 0.7

# Caption-stage sampling knobs. The caption task is more constrained
# (one tight paragraph per image) and benefits from a lower temperature
# to keep the response shape stable — the rewrite LLM consumes the
# caption text verbatim, so "wild" caption variants across re-runs would
# surface as jitter in the final prompt. 0.4 matches the H3 Loop
# enhancer's own temperature floor.
_CAPTION_TEMPERATURE = 0.4

# Token budget. Reasoning models count their thinking against max_tokens;
# 4096 let the chain-of-thought consume the whole budget before the
# answer started (same failure class as the H3 loop dialogue extractor
# in 5e8a5c2 and the user-input enhancer in f344875). 16384 is the
# cap-not-target we settled on across the project.
_MAX_TOKENS_DEFAULT = 16384
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Caption reply budget. Reasoning models count their thinking against
# max_tokens, so the caption stage gets the same 16384 cap-not-target as
# the rewrite stage (f344875 / 5e8a5c2 failure class: the thinking burns
# the whole budget before the caption starts and the node reads a 200
# with empty content).
_CAPTION_MAX_TOKENS = 16384

# Per-call timeout override.
_DEFAULT_TIMEOUT = 300

# Model slot ceiling (matches the official edit pipeline upper bound;
# 11 nodes take ~6.7x longer than 3 in live timing). Beyond this we
# still surface the count to the LLM but emit a warning — the model
# may misbehave on >10 refs, but the user owns the upstream sampler
# configuration and we don't fail their run.
_MAX_REFERENCE_IMAGES = 10

# Fixed response for ``mode = remove_bg``. The node never calls an LLM
# in this mode — see ``nodes/llm/prompts/qwen_image_21/_enhance_remove_bg.txt``.
_REMOVE_BG_FIXED_BLOCK = (
    "--- BEGIN enhanced_prompt ---\n"
    "Remove the background, and output a PNG image with a fully transparent "
    "(alpha) background. Preserve fine details along the subject edges, "
    "including hair strands, fabric fringes, glass refraction and translucent "
    "materials. The subject remains unchanged in pose, expression, lighting "
    "direction, colour palette and proportions; only the background becomes "
    "transparent.\n"
    "--- END enhanced_prompt ---"
)

# Caption user-message template. Mirrors the H3 Loop caption user
# message — short, fixed, leaves the wording up to the caption system
# prompt.
_CAPTION_USER_TEXT = (
    "{n} reference image(s). Write ONE tight English paragraph "
    "(2–4 sentences, ~60–120 words). Different people stay different "
    "people. Repeated views of one character on a turnaround or "
    "character sheet are ONE person: identity, hair, wardrobe, and "
    "accessories only. Do not describe the sheet layout, the studio "
    "backdrop, the sheet lighting, or a pose for the downstream scene "
    "to redraw. No mood abstractions, no shot grammar, no timestamps. "
    "Reply with the paragraph ONLY."
)

# Strip reasoning-model ``<think>...</think>`` wrappers from a caption
# before injecting it into the rewrite user message. Same regex as
# the rewrite stage so the behaviour is consistent.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_caption(raw: str) -> str:
    """Drop reasoning wrappers and turnaround-sheet layout phrases.

    Applied before the caption is cached and again when an older cache
    entry is injected, so a stored "three views / white background"
    line cannot lock the rewrite to the sheet.
    """
    text = _THINK_BLOCK_RE.sub("", raw or "").strip()
    return scrub_caption_sheet_language(text)


def _build_caption_user_content(
    n_images: int, urls: list[str]
) -> list[dict]:
    """OpenAI-style multimodal content: text + image_url parts."""
    parts: list[dict] = [
        {"type": "text", "text": _CAPTION_USER_TEXT.format(n=n_images)},
    ]
    for url in urls:
        parts.append(
            {"type": "image_url", "image_url": {"url": url}}
        )
    return parts


# --------------------------------------------------------------------------- #
# Captioner (per-image LLM call + two-tier cache)
# --------------------------------------------------------------------------- #
class _ReferenceCaptioner:
    """Per-image caption generator with in-memory + on-disk caching.

    Mirrors the slice of ``minimax_h3_loop_prompt_generator._caption_images``
    that owns the cache lookup / write / miss path. The LLM call shape
    is identical (system prompt + multimodal user content); the cache
    strategy (memory → disk → LLM) and the key derivation (sha256 of the
    data URL + sha256 of the prompt) match H3 Loop so users can share
    cache files between the two nodes if they ever want to.

    The on-disk root is resolved lazily on the first cache miss so the
    node can be constructed in test contexts where ``folder_paths`` is
    not available (then falls back to ``<repo>/output/...``).
    """

    def __init__(
        self,
        llm,
        *,
        cache_scope: str = "memory_disk",
        force_recaption: bool = False,
    ):
        self.llm = llm
        self.cache_scope = str(cache_scope or "memory_disk")
        self.force_recaption = bool(force_recaption)
        self._mem: dict[str, str] = {}
        self._disk_root: Optional[str] = None

    def _disk_dir(self) -> str:
        if self._disk_root is None:
            self._disk_root = caption_cache_disk_root()
        return self._disk_root

    def caption_all(
        self,
        data_urls: list[str],
        *,
        seed=None,
    ) -> list[str]:
        """Caption every image in ``data_urls`` (one LLM call per cache
        miss). Returns exactly one entry per input, in the same order —
        a ``[caption, caption, ...]`` list the caller can index by image
        slot. An entry may be empty (scope="disabled", no force); the
        rewrite stage skips empty lines while keeping the slot numbering,
        so alignment between captions and ``<imageN>`` slots never
        shifts."""
        if not data_urls:
            return []
        sys_text = caption_image_prompt()
        return [
            self._caption_one(
                url,
                sys_text,
                slot_n=slot_n,
                seed=seed,
            )
            for slot_n, url in enumerate(data_urls, 1)
        ]

    def _caption_one(
        self,
        url: str,
        sys_text: str,
        *,
        slot_n: int,
        seed=None,
    ) -> str:
        # Honour ``cache_scope="disabled"`` (without ``force_recaption``)
        # by skipping both cache and LLM. The H3 Loop node guards this
        # at the call site too; doing it here makes the captioner safe
        # to invoke directly with any scope value.
        if self.cache_scope == "disabled" and not self.force_recaption:
            return ""
        key = caption_cache_key(url, sys_text)
        short_key = key[:12]
        # Tier 1: in-memory
        if not self.force_recaption and key in self._mem:
            mie_log(
                f"qwen_image_21 captioner[{slot_n}]: cache hit (memory, "
                f"key={short_key})"
            )
            return self._mem[key]
        # Tier 2: on-disk
        if (
            not self.force_recaption
            and self.cache_scope == "memory_disk"
        ):
            disk = read_caption_cache_disk(self._disk_dir(), key)
            if disk is not None:
                cleaned = _clean_caption(disk)
                if cleaned:
                    mie_log(
                        f"qwen_image_21 captioner[{slot_n}]: cache hit (disk, "
                        f"key={short_key}, {len(cleaned)} chars)"
                    )
                    self._mem[key] = cleaned
                    return cleaned
        if not self.force_recaption and self.cache_scope != "disabled":
            mie_log(
                f"qwen_image_21 captioner[{slot_n}]: cache miss "
                f"(key={short_key})"
            )
        # Miss — call the LLM.
        messages = [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": _build_caption_user_content(1, [url]),
            },
        ]
        # Per-call timeout override mirrors the rewrite path so a stuck
        # captioner can't drag the whole node past its parent timeout.
        prev_timeout = getattr(self.llm, "timeout", None)
        had_timeout = hasattr(self.llm, "timeout")
        try:
            self.llm.timeout = _DEFAULT_TIMEOUT
            t0 = time.perf_counter()
            raw = self.llm.invoke(
                messages,
                seed=seed,
                temperature=_CAPTION_TEMPERATURE,
                max_tokens=_CAPTION_MAX_TOKENS,
            )
            elapsed = time.perf_counter() - t0
            mie_log(
                f"qwen_image_21 captioner[{slot_n}]: invoke done "
                f"({getattr(self.llm, 'model', '?')}) in {elapsed:.2f}s"
            )
        finally:
            if had_timeout:
                self.llm.timeout = prev_timeout
            else:
                try:
                    del self.llm.timeout
                except AttributeError:
                    pass
        text = _clean_caption(raw or "")
        if not text:
            raise RuntimeError(
                f"qwen_image_21 captioner[{slot_n}]: empty reply from LLM "
                "(image has no observable content? try a different image "
                "or change caption_mode to no_cache)"
            )
        # Cache BEFORE returning — the next call (same image + same prompt)
        # hits in-memory without re-invoking.
        self._mem[key] = text
        if self.cache_scope == "memory_disk":
            err = write_caption_cache_disk(self._disk_dir(), key, text)
            if err:
                mie_log(
                    f"qwen_image_21 captioner[{slot_n}]: disk cache write "
                    f"failed (key={short_key}): {err}"
                )
        return text


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class _QwenImage21PromptEnhancer:
    """One-shot LLM call that rewrites a rough draft into the Qwen-Image-2.1
    node-ready prompt. Thin wrapper around the connector with a per-call
    timeout override, mirroring the ``MiniMaxH3LoopUserInputEnhancer``
    shape.

    ``captions`` (optional list of one caption per ``<imageN>`` slot) is
    injected into the user message as DATA, so the rewrite LLM has the
    reference identity / wardrobe locked before it picks a verb.
    """

    def __init__(
        self,
        llm_service_connector,
        *,
        mode: str,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _MAX_TOKENS_DEFAULT,
        timeout: int = _DEFAULT_TIMEOUT,
        usage_sink=None,
    ):
        self.llm = llm_service_connector
        self.mode = mode
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # ``None`` would mean "leave the connector's own timeout alone";
        # the node always wires a dropdown default, so this is always
        # set in practice.
        self._timeout_override = int(timeout) if timeout else None
        # Optional ``usage_sink(stage, messages, reply)`` callback. Not
        # wired today; reserved for future preflight-report surfaces.
        self._usage_sink = usage_sink

    def _invoke(self, messages: list[dict], seed=None) -> str:
        # The override must apply even when the connector class has no
        # ``timeout`` attribute of its own (we create it for the call
        # and remove it afterwards, so the connector's own default
        # handling is untouched).
        override = self._timeout_override
        had_timeout = hasattr(self.llm, "timeout")
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if override is not None:
                self.llm.timeout = override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.perf_counter() - t0
            if self._usage_sink is not None:
                try:
                    self._usage_sink("qwen_image_21_enhance", messages, out or "")
                except Exception:
                    pass
            try:
                mie_log(
                    f"qwen_image_21_prompt_generator: invoke done "
                    f"({getattr(self.llm, 'model', '?')}) in {elapsed:.2f}s"
                )
            except Exception:
                pass
            return out or ""
        finally:
            if override is not None:
                if had_timeout:
                    self.llm.timeout = prev_timeout
                else:
                    try:
                        del self.llm.timeout
                    except AttributeError:
                        pass

    def _build_user_message(
        self,
        draft: str,
        *,
        reference_count: int,
        captions: Optional[list[str]] = None,
    ) -> str:
        """Assemble the LLM user message: draft + mode context + reply
        template. ``reference_count`` and ``captions`` are only emitted
        for modes that actually use image slots (``edit``,
        ``multi_ref``); T2I / t2i_rgba always skip them.
        """
        mode_code = parse_mode_label(self.mode) or self.mode
        parts: list[str] = [
            "---BEGIN USER DRAFT---",
            (draft or "").strip(),
            "---END USER DRAFT---",
            "",
            f"Mode: {mode_code}",
        ]
        if mode_code in {"edit", "multi_ref"}:
            n = int(reference_count)
            # DATA only: the count and the ceiling. Slot usage
            # (<image1>…<imageN> vs natural reference) and the
            # character-sheet rule are owned by the mode's system
            # prompt, so the node carries no per-case rewrite logic.
            parts.append(
                f"reference_count: {n} "
                f"(model slot ceiling {_MAX_REFERENCE_IMAGES})"
            )
            if captions:
                # Inject as DATA, not instructions. The rewrite LLM must
                # use these to lock identity / wardrobe / palette but
                # must NOT echo them as text in the rewrite.
                clean_caps: list[str] = []
                for idx, cap in enumerate(captions[:n], 1):
                    text = _clean_caption(cap or "")
                    if text:
                        clean_caps.append(
                            f"- <image{idx}>: {text}"
                        )
                if clean_caps:
                    parts.append("")
                    parts.append(
                        "Reference captions (DATA — locked identity / "
                        "wardrobe / palette for downstream generation; "
                        "do not echo as text in the rewrite):"
                    )
                    parts.extend(clean_caps)
        parts.extend(
            [
                "",
                "Wrap your answer exactly as:",
                "--- BEGIN enhanced_prompt ---",
                "<one continuous paragraph; no markdown>",
                "--- END enhanced_prompt ---",
                "Nothing outside the block except an optional one-line "
                "Notes before it.",
            ]
        )
        return "\n".join(parts)

    def __call__(
        self,
        draft: str,
        *,
        reference_count: int,
        captions: Optional[list[str]] = None,
        seed=None,
    ) -> str:
        # Resolve the bilingual ``code - 中文(标签)`` widget value to
        # the bare code the prompt loader + membership checks branch on.
        mode_code = parse_mode_label(self.mode) or self.mode
        sys_text = load_mode_prompt(mode_code)
        user_msg = self._build_user_message(
            draft,
            reference_count=reference_count,
            captions=captions,
        )
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_msg},
        ]
        return self._invoke(messages, seed=seed)


# --------------------------------------------------------------------------- #
# Top-level helper (testable, retry policy)
# --------------------------------------------------------------------------- #
def _short_circuit_remove_bg() -> str:
    """Return the fixed BG-remove block. No LLM call."""
    return _REMOVE_BG_FIXED_BLOCK


def count_reference_images(images) -> int:
    """Return the batch size of an ``IMAGE`` socket (``shape[0]``), or
    ``0`` for unconnected / empty inputs.

    Mirrors the lenient shape handling in
    ``minimax_h3_loop_prompt_generator._caption_images``: any object
    exposing a ``shape`` attribute whose first axis is a positive int
    is accepted; anything else (None, plain objects, ``shape == ()``)
    returns 0. Real torch tensors expose ``shape`` but not always
    ``ndim`` in some test stubs, so the check goes through ``shape``.
    """
    if images is None:
        return 0
    shape = getattr(images, "shape", None)
    if not shape:
        return 0
    try:
        return int(shape[0])
    except (TypeError, IndexError, ValueError):
        return 0


def _normalize_images(images):
    """Coerce a real 4-D IMAGE batch into the (B, H, W, C) shape the
    encoder expects.

    Returns the batch unchanged if it's already 4-D, normalises a lone
    (H, W, C) tensor into (1, H, W, C), and returns ``None`` for
    anything else (including ``None``).
    """
    if images is None:
        return None
    if not hasattr(images, "ndim"):
        return None
    if images.ndim == 3:
        try:
            return images[None, ...]
        except Exception:
            return None
    if images.ndim == 4:
        return images
    return None


def encode_reference_images(images) -> list[str]:
    """Convert an IMAGE tensor batch to a list of JPEG data URLs.

    Defers to ``core.utils.image_tensor_batch_to_data_urls`` so the
    encoder is the same as the rest of the plugin (BGR conversion,
    JPEG quality, base64 framing all stay in one place). Returns an
    empty list if ``images`` is ``None`` / unknown shape / fails to
    encode.
    """
    if images is None:
        return []
    try:
        from ...core.utils import image_tensor_batch_to_data_urls
    except ImportError:
        from _mienodes_internal.core.utils import image_tensor_batch_to_data_urls
    norm = _normalize_images(images)
    if norm is None:
        return []
    try:
        urls = image_tensor_batch_to_data_urls(norm)
    except Exception as exc:
        mie_log(
            f"qwen_image_21_prompt_generator: image encode failed: {exc}"
        )
        return []
    return list(urls or [])


def caption_references(
    captioner: _ReferenceCaptioner,
    images,
    *,
    seed=None,
) -> list[str]:
    """Caption the IMAGE batch through ``captioner``.

    Returns ``[]`` if ``images`` is ``None`` / unknown shape. The
    captioner itself logs hit / miss / error per image; this helper
    only handles the encode step + iteration.
    """
    urls = encode_reference_images(images)
    if not urls:
        return []
    return captioner.caption_all(urls, seed=seed)


def run_enhancer(
    llm_service_connector,
    draft: str,
    *,
    mode: str,
    reference_count: int = 0,
    captions: Optional[list[str]] = None,
    seed=None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _MAX_TOKENS_DEFAULT,
    timeout: int = _DEFAULT_TIMEOUT,
    attempts: int = 3,
):
    """Run the rewrite with automatic retries on empty / block-less
    replies, then return the trimmed ``enhanced_prompt`` block.

    ``mode = remove_bg`` short-circuits: the fixed block is returned
    without an LLM call.

    Other modes retry a reply that is empty or has no BEGIN/END block with
    a fresh seed (same seed could deterministically reproduce the same
    empty answer); a genuine refusal that keeps its shape across all
    attempts still raises ``RuntimeError`` with the last reply head.
    """
    if mode == "remove_bg":
        return extract_enhanced_prompt_block(_short_circuit_remove_bg()) or ""

    enhancer = _QwenImage21PromptEnhancer(
        llm_service_connector,
        mode=mode,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    last_head = ""
    for attempt in range(1, max(1, attempts) + 1):
        attempt_seed = seed if attempt == 1 else (
            None if seed is None else int(seed) + attempt - 1
        )
        raw = enhancer(
            draft,
            reference_count=reference_count,
            captions=captions,
            seed=attempt_seed,
        )
        block, _header = split_enhancer_reply(raw)
        if block:
            if attempt > 1:
                mie_log(
                    "qwen_image_21_prompt_generator: succeeded on attempt "
                    f"{attempt}/{attempts}"
                )
            return block
        last_head = (raw or "")[:400]
        if attempt < attempts:
            mie_log(
                "qwen_image_21_prompt_generator: attempt "
                f"{attempt}/{attempts} reply had no enhanced_prompt block "
                f"({len(raw or '')} chars); retrying with a fresh seed"
            )
    raise RuntimeError(
        "Qwen-Image-2.1 prompt enhancer: the LLM reply did not contain a "
        "`--- BEGIN enhanced_prompt ---` ... `--- END enhanced_prompt ---` "
        f"block after {attempts} attempts. Raw reply head: {last_head!r}"
    )


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class QwenImage21PromptGenerator:
    """ComfyUI node: rough draft → Qwen-Image-2.1-ready STRING prompt.

    Sits next to ``MiniMaxH3LoopUserInputEnhancer`` in the same Prompt
    Generator category. The ``enhanced_prompt`` output plugs straight
    into the Qwen-Image-2.1 sampler / clip-text-encode nodes.

    Raises ``RuntimeError`` if the LLM reply does not contain a
    ``--- BEGIN enhanced_prompt ---`` ... ``--- END enhanced_prompt ---``
    block; the error includes the raw reply head so the user can
    diagnose. Silent fallback would let a misclassified reply flow into
    the sampler and waste the whole pipeline.

    ``mode = remove_bg`` is intentionally short-circuited: the node
    returns a fixed BG-remove block without an LLM call.

    The rewrite body is **always English**, regardless of the user's
    input language. The verbatim-language rule (text inside straight
    double quotes in the rewrite keeps the user's script) is encoded
    by the rewrite system prompt and the node does not second-guess it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "mode": (
                    list(MODES),
                    {
                        "default": MODES[0],
                        "tooltip": (
                            # English first, then Chinese — mirrors
                            # MiniMaxH3LoopPromptGenerator dropdown
                            # convention so users can read either.
                            "Mode / 模式:  "
                            "t2i 文生图(默认) — observer English paragraph.  "
                            "t2i_rgba 透明文生图 — same plus forced RGBA bookend sandwich.  "
                            "edit 单图编辑 — single-image edit instruction (rewrite body "
                            "is always English).  "
                            "multi_ref 多参考编辑 — multi-reference edit instruction "
                            "(<image1>…<imageN>; rewrite body always English).  "
                            "remove_bg 抠图 — fixed BG-remove block, no LLM call."
                        ),
                    },
                ),
                "user_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Rough idea / 草稿: a sentence, an outline, "
                            "a few lines, or a free paragraph. The rewriter "
                            "expands it into the prompt the Qwen-Image-2.1 "
                            "sampler consumes. Output is always English; "
                            "verbatim text inside straight double quotes in "
                            "your input is preserved as-is. / 扩写正文一律"
                            "英文；直双引号内的字符串保留原语种与逐字。"
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Seed / 种子 forwarded to the LLM call. 0 lets "
                            "the connector pick a fresh seed."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional IMAGE batch / 可选 IMAGE 批次.  "
                            "For mode=edit / multi_ref 单图编辑 / 多参考编辑, "
                            "the node captions each image (<image1>…<imageN>) "
                            "with the bundled _caption_image.txt system prompt "
                            "and injects the captions into the rewrite user "
                            "message as locked identity / wardrobe / palette "
                            "DATA. Pixels are visible to the captioner LLM "
                            "but NOT to the rewrite LLM. / 改写 LLM 看不到图。  "
                            "Recommended ceiling: 10 (model slot limit); "
                            "beyond that the node logs a warning but still "
                            "passes the full count.  "
                            "IMPORTANT / 注意: 多视图设定表是版式。四身横排喂进去，"
                            "成图会复制这排人，每个人一个姿势。先裁成单人"
                            "（正面或四分之三侧面）再接 reference_images，"
                            "输出用竖幅。编辑节点看见的仍是你接上的那张图；"
                            "这个节点只写提示词。 "
                            "A multi-view character sheet is a layout: four "
                            "standing views in, four poses out. Crop to one "
                            "person before wiring the image. "
                            "Wire any sampler / loader output that emits an "
                            "IMAGE batch — e.g. LoadImageBatch or a custom "
                            "collector."
                        ),
                    },
                ),
                "caption_mode": (
                    list(CAPTION_MODES),
                    {
                        "default": CAPTION_MODES[0],
                        "tooltip": (
                            "Caption cache strategy / 标注缓存策略 (only used "
                            "when reference_images is connected and mode in "
                            "{edit, multi_ref}).  "
                            "cache_memory_disk (recommended) 缓存:内存+磁盘 — "
                            "in-memory + persistent on-disk cache under "
                            "<ComfyUI output>/mien_nodes/caption_cache/.  "
                            "cache_memory_only 缓存:仅内存 — RAM-only cache, "
                            "lost on restart.  "
                            "no_cache 禁用缓存 — always bypass cache.  "
                            "force_recaption_once 本次强制重打标 — force fresh "
                            "captions now (e.g. after editing "
                            "_caption_image.txt)."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature / 采样温度. 0.0–2.0, step 0.05. "
                            "Lower = strict adherence to system prompt; "
                            "higher = more surface variation."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _MAX_TOKENS_DEFAULT,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                        "tooltip": (
                            "Token budget / Token 上限. 64–32768. Reasoning "
                            "models count their thinking against this cap; "
                            "16384 is the safe default for the whole plugin."
                        ),
                    },
                ),
                "timeout": (
                    [60, 120, 300, 600],
                    {
                        "default": _DEFAULT_TIMEOUT,
                        "tooltip": (
                            "Per-call timeout (seconds) / 单次调用超时(秒)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = MY_CATEGORY

    def enhance(
        self,
        llm_service_connector,
        mode,
        user_input,
        seed=None,
        reference_images=None,
        caption_mode=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        # Resolve bilingual widget value (``t2i - 文生图(默认)``) to
        # the bare code the rest of the node branches on.
        mode_code = parse_mode_label(mode) or mode
        n_images = count_reference_images(reference_images)
        if n_images > _MAX_REFERENCE_IMAGES:
            mie_log(
                f"qwen_image_21_prompt_generator: {n_images} reference "
                f"images connected (recommended ceiling "
                f"{_MAX_REFERENCE_IMAGES}); passing full count to LLM, "
                "model may misbehave on >10 refs"
            )

        # Captions: only run for modes that consume them. The caption
        # task is its own LLM call per image (cache-fronted), so we
        # gate on both `n_images > 0` AND mode in {edit, multi_ref} —
        # t2i / t2i_rgba / remove_bg never see reference data.
        captions: Optional[list[str]] = None
        if n_images > 0 and mode_code in {"edit", "multi_ref"}:
            force, scope = resolve_caption_controls(
                caption_mode or CAPTION_MODES[0]
            )
            if scope != "disabled" or force:
                captioner = _ReferenceCaptioner(
                    llm_service_connector,
                    cache_scope=scope,
                    force_recaption=force,
                )
                try:
                    captions = caption_references(
                        captioner, reference_images, seed=seed
                    )
                except RuntimeError as exc:
                    # Empty-reply or encode failures are surfaced as a
                    # clear error — silent fallback would let the rewrite
                    # LLM hallucinate identity, which is the exact
                    # failure mode the captioner is here to prevent.
                    raise RuntimeError(
                        f"qwen_image_21_prompt_generator: captioning failed "
                        f"({exc}); retry with caption_mode=no_cache to "
                        "bypass the cache, or check the image content"
                    ) from exc
            else:
                # scope == "disabled" AND not force_recaption → user
                # opted out of caching AND captions, so emit no captions.
                # The rewrite LLM will fall back to "describe what the
                # user said in user_input" — same as v1 behaviour.
                captions = None

        block = run_enhancer(
            llm_service_connector,
            user_input,
            mode=mode_code,
            reference_count=n_images,
            captions=captions,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return (block,)

    def is_changed(
        self,
        llm_service_connector,
        mode,
        user_input,
        seed=None,
        reference_images=None,
        caption_mode=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        # Bilingual widget value → bare code, so the same hash lands on
        # identical configurations regardless of the menu language.
        mode_code = parse_mode_label(mode) or mode
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_input or "",
            mode_code or "",
            str(seed),
            str(caption_mode or ""),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update(part.encode("utf-8"))
        # Hash the IMAGE batch enough to detect a same-shape content
        # swap (mirrors MiniMaxH3LoopPromptGenerator.is_changed).
        if reference_images is None:
            h.update(b"images:none")
        else:
            try:
                shape = tuple(reference_images.shape)
            except AttributeError:
                shape = ()
            h.update(repr(shape).encode("utf-8"))
            h.update(str(getattr(reference_images, "dtype", "")).encode("utf-8"))
            sample_bytes = b""
            try:
                flat = (
                    reference_images.detach().cpu().reshape(-1)
                    if hasattr(reference_images, "detach")
                    and hasattr(reference_images, "cpu")
                    else (
                        reference_images.reshape(-1)
                        if hasattr(reference_images, "reshape")
                        else None
                    )
                )
                if flat is not None:
                    n = (
                        int(flat.shape[0])
                        if hasattr(flat, "shape")
                        else len(flat)
                    )
                    chunks = []
                    if n > 0:
                        head = flat[:1024]
                        mid_start = max(0, (n // 2) - 512)
                        mid = flat[mid_start:mid_start + 1024]
                        tail = flat[-1024:] if n > 1024 else flat
                        for part in (head, mid, tail):
                            if hasattr(part, "numpy"):
                                chunks.append(bytes(part.numpy().tobytes()))
                            elif hasattr(part, "tobytes"):
                                chunks.append(bytes(part.tobytes()))
                    sample_bytes = b"".join(chunks)
            except Exception:
                sample_bytes = b""
            h.update(
                hashlib.md5(
                    sample_bytes, usedforsecurity=False
                ).hexdigest().encode("ascii")
            )
        # Hash the bundled system-prompt bodies (rewrite + caption) so
        # editing either .txt file in development invalidates ComfyUI's
        # node cache automatically.
        try:
            sys_text = load_mode_prompt(mode_code)
        except (KeyError, FileNotFoundError):
            sys_text = ""
        h.update(sys_text.encode("utf-8"))
        try:
            sys_caption = caption_image_prompt()
        except FileNotFoundError:
            sys_caption = ""
        h.update(sys_caption.encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()