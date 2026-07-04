"""SCAIL-2 prompt-enhancer ComfyUI node.

Two-stage LLM pipeline mirroring
https://github.com/zai-org/SCAIL-2/blob/wan-scail2/prompt_enhancer.py
for replacement mode, plus a MieNodes-original motion-transfer variant.

The node is registered under the same ``Prompt Generator`` category as
``BerniniPromptGenerator`` and ``Ideogram4PromptGenerator`` and follows
the same output shape (single ``STRING``).
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

try:
    from _mienodes_internal.core.utils import (
        image_tensor_batch_to_data_urls,
        mie_log,
    )
except ImportError:
    try:
        from ...core.utils import (
            image_tensor_batch_to_data_urls,
            mie_log,
        )
    except ImportError:
        from core.utils import (
            image_tensor_batch_to_data_urls,
            mie_log,
        )

try:
    from _mienodes_internal.nodes.llm.scail2_prompts import (
        MAX_EXAMPLE_CHARS,
        TASK_CODES,
        TASK_TYPES,
        bundled_examples_motion_transfer,
        bundled_examples_replacement,
        caption_motion_transfer_prompt,
        caption_replacement_prompt,
        enhance_motion_transfer_prompt,
        enhance_replacement_prompt,
        load_bundled_examples,
        parse_task_code,
    )
except ImportError:
    from .scail2_prompts import (
        MAX_EXAMPLE_CHARS,
        TASK_CODES,
        TASK_TYPES,
        bundled_examples_motion_transfer,
        bundled_examples_replacement,
        caption_motion_transfer_prompt,
        caption_replacement_prompt,
        enhance_motion_transfer_prompt,
        enhance_replacement_prompt,
        load_bundled_examples,
        parse_task_code,
    )


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

# Per-stage token budgets. Stage 1 writes a long English paragraph;
# stage 2 is constrained to 80-140 words by the prompt templates.
_DEFAULT_MAX_TOKENS_CAPTION = 2048
_DEFAULT_MAX_TOKENS_ENHANCE = 512

# Default number of frames sampled from the driving video for stage 1.
# Matches upstream ``prompt_enhancer.py`` default of 8.
DEFAULT_NUM_FRAMES = 8
MIN_NUM_FRAMES = 1
MAX_NUM_FRAMES = 16

# --------------------------------------------------------------------------- #
# Frame sampling (mirrors bernini_prompt_generator._sample_indices / _sample_urls)
# --------------------------------------------------------------------------- #
def _sample_indices(total: int, n: int) -> list[int]:
    """Return ``n`` unique indices in ``[0, total)`` sampled as evenly as possible.

    ``n`` is clamped to ``[1, total]`` before sampling. When ``n == 1`` the
    middle index is returned; when ``n == 2`` the endpoints are returned
    so a driving video gets both the first and last frame in its caption;
    for larger ``n`` the indices are spaced uniformly using
    ``round(i * (total - 1) / (n - 1))``.
    """
    if total <= 0:
        return []
    n = max(1, min(n, total))
    if n == 1:
        return [total // 2]
    if n == 2:
        return [0, total - 1]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _sample_urls(urls: list[str], n: int) -> list[str]:
    """Apply ``_sample_indices`` to a list of URLs, preserving order and
    dropping duplicate indices (can happen at the endpoints)."""
    if not urls:
        return []
    idx = _sample_indices(len(urls), n)
    seen: set[int] = set()
    out: list[str] = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(urls[i])
    return out


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class Scail2PromptEnhancer:
    """Two-stage SCAIL-2 prompt enhancer that talks to the project's LLMServiceConnector.

    Stage 1 captions the driving video; stage 2 writes the final positive
    description of the animated / replaced video. Each stage is one
    ``llm.invoke`` call against the same connector.

    Mirrors the structure of ``BerniniPromptEnhancer`` so the two LLM
    nodes can share the project's logging + timeout patterns.
    """

    def __init__(
        self,
        llm_service_connector: Any,
        *,
        num_frames: int = DEFAULT_NUM_FRAMES,
        image_detail: str = "auto",
        temperature: float = 0.4,
        max_tokens_caption: int = _DEFAULT_MAX_TOKENS_CAPTION,
        max_tokens_enhance: int = _DEFAULT_MAX_TOKENS_ENHANCE,
        timeout: Optional[int] = None,
    ):
        self.llm = llm_service_connector
        self.num_frames = max(MIN_NUM_FRAMES, min(int(num_frames), MAX_NUM_FRAMES))
        self.image_detail = image_detail
        self.temperature = float(temperature)
        self.max_tokens_caption = int(max_tokens_caption)
        self.max_tokens_enhance = int(max_tokens_enhance)
        # Per-call timeout override; None means leave the connector's
        # own timeout in place. Saved and restored around invoke() so
        # the connector object is safe to share with other nodes.
        self._timeout_override = int(timeout) if timeout else None

    # ------------------------------------------------------------------ #
    # Internal: one LLM call with timeout / log plumbing
    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        seed: Optional[int],
        stage: str,
    ) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out:
                mie_log(
                    f"Scail2 {stage}: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"Scail2 {stage}: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
            )
            return out.strip()
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_text: str,
        image_urls: list[str],
        image_detail: str,
    ) -> list[dict]:
        """Build a ``[system, user[...]]`` pair in OpenAI-chat format.

        Mirrors ``bernini_prompt_generator._build_messages``: each image
        gets a small ``[Image N]:`` caption preceding the image part so
        vision models that look at the textual layout of the conversation
        know which image is which.
        """
        parts: list[dict] = []
        for i, url in enumerate(image_urls or []):
            parts.append({"type": "text", "text": f"\n[Image {i}]:"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": image_detail},
                }
            )
        if user_text:
            parts.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parts},
        ]

    # ------------------------------------------------------------------ #
    # Per-task caption + enhance pipeline
    # ------------------------------------------------------------------ #
    def _caption(
        self,
        task_code: str,
        frame_urls: list[str],
        user_prompt: str,
        seed: Optional[int],
    ) -> str:
        """Stage 1: caption the driving video."""
        if task_code == "character_replacement":
            system = caption_replacement_prompt()
            user_text = (
                f"User replacement instruction: {user_prompt.strip()}\n"
                f"The following images are {len(frame_urls)} sampled frames in chronological order."
            )
        elif task_code == "motion_transfer":
            system = caption_motion_transfer_prompt()
            user_text = (
                f"User hint (may be empty): {user_prompt.strip() or '(no user hint)'}\n"
                f"The following images are {len(frame_urls)} sampled frames in chronological order."
            )
        else:
            raise ValueError(f"Scail2: unknown task code: {task_code!r}")
        messages = self._build_messages(
            system, user_text, frame_urls, self.image_detail
        )
        caption = self._invoke(
            messages,
            max_tokens=self.max_tokens_caption,
            seed=seed,
            stage=f"caption[{task_code}]",
        )
        mie_log(
            f"Scail2 caption[{task_code}]: {len(caption)} chars: {caption[:120]!r}"
        )
        return caption

    def _enhance(
        self,
        task_code: str,
        ref_urls: list[str],
        user_prompt: str,
        caption: str,
        seed: Optional[int],
    ) -> str:
        """Stage 2: rewrite the caption as a positive SCAIL-2 prompt."""
        if task_code == "character_replacement":
            examples = bundled_examples_replacement()
            user_text = enhance_replacement_prompt(
                instruction=user_prompt,
                caption=caption,
                examples=examples,
            )
            system = "You are a prompt enhancer for SCAIL-2 character replacement."
        elif task_code == "motion_transfer":
            examples = bundled_examples_motion_transfer()
            user_text = enhance_motion_transfer_prompt(
                caption=caption,
                user_hint=user_prompt,
                examples=examples,
            )
            system = "You are a prompt enhancer for SCAIL-2 character animation / motion transfer."
        else:
            raise ValueError(f"Scail2: unknown task code: {task_code!r}")
        messages = self._build_messages(
            system, user_text, ref_urls, self.image_detail
        )
        return self._invoke(
            messages,
            max_tokens=self.max_tokens_enhance,
            seed=seed,
            stage=f"enhance[{task_code}]",
        )

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        task_type: str,
        user_prompt: str,
        *,
        driving_video: Any = None,
        reference_images: Any = None,
        seed: Optional[int] = None,
    ) -> str:
        """Run the two-stage pipeline and return the final SCAIL-2 prompt.

        Returns ``user_prompt`` unchanged on missing media (matching
        Bernini's graceful-degradation behavior), on an LLM error, or
        on a ``character_replacement`` task with an empty user_prompt.

        ``motion_transfer`` accepts an empty user_prompt: in that case
        the stage-2 template falls back to deriving a motion description
        purely from the driving-video caption.
        """
        code = parse_task_code(task_type)
        if code not in TASK_CODES:
            mie_log(
                f"Scail2: unknown task {task_type!r} (parsed code={code!r}); returning original"
            )
            return user_prompt

        # Empty user_prompt rules:
        #   - character_replacement: required -> return original (empty) prompt.
        #   - motion_transfer: optional -> auto-derive from media.
        if not user_prompt or not user_prompt.strip():
            if code == "character_replacement":
                mie_log(
                    "Scail2: character_replacement requires a non-empty user_prompt; returning original"
                )
                return user_prompt

        # Both tasks need a driving video + at least one reference image.
        driving_urls = image_tensor_batch_to_data_urls(driving_video)
        ref_urls = image_tensor_batch_to_data_urls(reference_images)
        if not driving_urls:
            mie_log(f"Scail2: no driving_video frames provided (task={code}); returning original")
            return user_prompt
        if not ref_urls:
            mie_log(f"Scail2: no reference images provided (task={code}); returning original")
            return user_prompt

        frame_urls = _sample_urls(driving_urls, self.num_frames)
        mie_log(
            f"Scail2: task={code} driving_frames={len(driving_urls)}->{len(frame_urls)} "
            f"ref_imgs={len(ref_urls)} detail={self.image_detail} "
            f"temperature={self.temperature}"
        )

        caption = self._caption(code, frame_urls, user_prompt or "", seed=seed)
        if not caption:
            mie_log(
                f"Scail2: stage-1 caption returned empty (task={code}); returning original"
            )
            return user_prompt

        enhanced = self._enhance(code, ref_urls, user_prompt or "", caption, seed=seed)
        return enhanced or user_prompt


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class Scail2PromptGenerator:
    """ComfyUI node: rewrite a user prompt with SCAIL-2's two-stage pipeline.

    Selectable modes:
      character_replacement, motion_transfer

    Output: a single ``STRING`` containing the enhanced SCAIL-2 prompt.
    The intermediate stage-1 caption is logged via ``mie_log`` for
    debugging but not returned as a port.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "task_type": (list(TASK_TYPES), {"default": TASK_TYPES[0]}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            },
            "optional": {
                # Driving video (image batch). Required for both tasks; if
                # empty the enhancer returns the original prompt.
                "driving_video": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "For character_replacement: the source video to be edited (the subject being replaced appears in this video).\n\nFor motion_transfer: the driving video whose motion / pose / action is applied to the character in reference_images."
                        ),
                    },
                ),
                # Reference image(s). Required for both tasks; the first
                # one is the "replacement target" / "target character".
                "reference_images": ("IMAGE",),
                # How many frames to sample from ``driving_video`` for stage 1.
                # Matches upstream ``prompt_enhancer.py`` default of 8.
                "num_frames": (
                    "INT",
                    {
                        "default": DEFAULT_NUM_FRAMES,
                        "min": MIN_NUM_FRAMES,
                        "max": MAX_NUM_FRAMES,
                    },
                ),
                "image_detail": (
                    ["auto", "low", "high"],
                    {"default": "auto"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "max_tokens_caption": (
                    "INT",
                    {"default": _DEFAULT_MAX_TOKENS_CAPTION, "min": 64, "max": 32768},
                ),
                "max_tokens_enhance": (
                    "INT",
                    {"default": _DEFAULT_MAX_TOKENS_ENHANCE, "min": 64, "max": 32768},
                ),
                # Per-call timeout override. Long vision tasks on heavy
                # models sometimes need 60-300s; default 120 mirrors
                # Bernini 30 + the project-wide 30 default.
                "timeout": ([30, 60, 120, 300], {"default": 120}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scail2_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed=None,
        driving_video=None,
        reference_images=None,
        num_frames=DEFAULT_NUM_FRAMES,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=_DEFAULT_MAX_TOKENS_CAPTION,
        max_tokens_enhance=_DEFAULT_MAX_TOKENS_ENHANCE,
        timeout=120,
    ):
        enhancer = Scail2PromptEnhancer(
            llm_service_connector,
            num_frames=num_frames,
            image_detail=image_detail,
            temperature=temperature,
            max_tokens_caption=max_tokens_caption,
            max_tokens_enhance=max_tokens_enhance,
            timeout=timeout,
        )
        out = enhancer(
            task_type,
            user_prompt,
            driving_video=driving_video,
            reference_images=reference_images,
            seed=seed,
        )
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed=None,
        driving_video=None,
        reference_images=None,
        num_frames=DEFAULT_NUM_FRAMES,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=_DEFAULT_MAX_TOKENS_CAPTION,
        max_tokens_enhance=_DEFAULT_MAX_TOKENS_ENHANCE,
        timeout=120,
    ):
        h = hashlib.md5()
        for part in (
            task_type,
            user_prompt,
            str(seed),
            str(num_frames),
            image_detail,
            str(temperature),
            str(max_tokens_caption),
            str(max_tokens_enhance),
            str(timeout),
        ):
            h.update((part or "").encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        # Cheap media signature: just the tensor shape, not the full
        # pixel data. Matches Bernini's strategy so a tweak that
        # changes frame count or resolution triggers a re-run.
        for t in (driving_video, reference_images):
            if t is None:
                h.update(b"none")
            else:
                try:
                    shape = list(t.shape)
                except AttributeError:
                    shape = []
                h.update(repr(shape).encode("utf-8"))
        return h.hexdigest()
