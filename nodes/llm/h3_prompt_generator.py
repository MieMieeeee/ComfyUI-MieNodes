"""MiniMax H3 prompt-enhancer ComfyUI node.

Rewrites a user's rough idea into a structured MiniMax H3 video prompt
following the official VIDEO_PROMPT_WRITING_GUIDE (base + ref). Supports
six task paths (t2v / i2v_first / i2v_first_last / i2v_last / reference /
s2v) and a 19-class ``category`` dimension that injects per-use-case
default styling advice.

Two-stage pipeline (mirrors SCAIL-2's caption -> enhance shape):
  * Stage 1 (``reference`` path only): caption the reference images /
    sampled video frames so stage 2 knows which features to preserve.
  * Stage 2: write the final structured H3 prompt. For image-bearing
    paths the official alignment directive is prepended verbatim and the
    model is told NOT to re-describe appearances already visible in the
    reference.

The node is registered under the same ``Prompt Generator`` category as
``BerniniPromptGenerator`` and ``Scail2PromptGenerator`` and follows the
same output shape (single ``STRING``). The internal structure mirrors
``scail2_prompt_generator.py`` so the two LLM nodes share the project's
logging + timeout patterns.
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
    from _mienodes_internal.nodes.llm.h3_prompts import (
        CATEGORIES,
        CATEGORY_CODES,
        DEFAULT_HEIGHT,
        DEFAULT_OUTPUT_LANGUAGE,
        DEFAULT_WIDTH,
        IMAGE_TASK_CODES,
        MAX_EXAMPLE_CHARS,
        OUTPUT_LANGUAGES,
        TASK_CODES,
        TASK_TYPES,
        align_directive_for,
        aspect_ratio_string,
        bundled_examples_i2v,
        bundled_examples_t2v,
        caption_reference_prompt,
        category_advice,
        load_bundled_examples,
        parse_category,
        parse_task_code,
        system_reference_prompt,
        system_t2v_prompt,
        user_i2v_prompt,
        user_t2v_prompt,
    )
except ImportError:
    from .h3_prompts import (
        CATEGORIES,
        CATEGORY_CODES,
        DEFAULT_HEIGHT,
        DEFAULT_OUTPUT_LANGUAGE,
        DEFAULT_WIDTH,
        IMAGE_TASK_CODES,
        MAX_EXAMPLE_CHARS,
        OUTPUT_LANGUAGES,
        TASK_CODES,
        TASK_TYPES,
        align_directive_for,
        aspect_ratio_string,
        bundled_examples_i2v,
        bundled_examples_t2v,
        caption_reference_prompt,
        category_advice,
        load_bundled_examples,
        parse_category,
        parse_task_code,
        system_reference_prompt,
        system_t2v_prompt,
        user_i2v_prompt,
        user_t2v_prompt,
    )


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

# Per-stage token budgets.
# NOTE: reasoning models (MiniMax-M3, DeepSeek-R1, GLM-5.x) emit their
# chain-of-thought INSIDE the token budget before the final answer. If the
# <think> chain consumes the whole budget the model returns an empty
# content — this is the #1 cause of "H3 enhance[...]: returned empty
# after Xs" logs in the wild (e.g. M3 thinking for ~60s on a long i2v
# prompt and exhausting 4096 before the answer). So both budgets are sized
# to hold the think chain AND the ~350-800-word H3 answer with headroom.
# Giving more than needed costs nothing — the model stops as soon as the
# answer is done; only worst-case latency grows.
_DEFAULT_MAX_TOKENS_CAPTION = 4096
_DEFAULT_MAX_TOKENS_ENHANCE = 8192

# Default number of frames sampled from the reference video for stage 1
# (caption). Matches the SCAIL-2 default of 8. Renamed from ``num_frames``
# to ``caption_sample_frames`` at the node surface so users do not mistake
# it for the output video length.
DEFAULT_CAPTION_SAMPLE_FRAMES = 8
MIN_CAPTION_SAMPLE_FRAMES = 1
MAX_CAPTION_SAMPLE_FRAMES = 16


# --------------------------------------------------------------------------- #
# Default idea synthesis (used when the user leaves user_prompt empty)
# --------------------------------------------------------------------------- #
def _default_idea(task_code: str, category: str) -> str:
    """Synthesize a reasonable default idea when the user left ``user_prompt``
    blank. The node then runs the full pipeline against this fallback instead
    of returning an empty string.

    The fallback is intentionally generic and short: the LLM still has the
    reference media (for image-bearing tasks), the category advice, and the
    few-shot examples to do the heavy lifting. Each branch just names the
    kind of motion/change the task is about, so the model has a concrete
    anchor to describe.
    """
    cat = parse_category(category)
    cat_hint = ""
    # Only append the hint for a KNOWN category other than 'none', so an
    # unknown / typo'd category string does not leak into the prompt.
    if cat and cat != "none" and cat in CATEGORY_CODES:
        # e.g. " for a cinematic-story video" -- nudges the model toward the
        # category's vocabulary without forcing it.
        cat_hint = f" for a {cat} video"
    if task_code == "t2v":
        return (
            "Create a visually striking short scene with a clear subject, "
            "a distinct environment, and evolving motion over the duration."
            + cat_hint
        )
    if task_code in ("i2v_first",):
        return (
            "Bring this first frame to life with natural, evolving motion: "
            "continue the subject's action and develop the scene over time."
            + cat_hint
        )
    if task_code == "i2v_first_last":
        return (
            "Describe a smooth, plausible transition from the first frame "
            "to the last frame, focusing on how the subject, camera, and "
            "lighting move between the two states."
            + cat_hint
        )
    if task_code == "i2v_last":
        return (
            "Show how the scene could plausibly arrive at this final frame, "
            "building up to it with continuous motion."
            + cat_hint
        )
    if task_code == "reference":
        return (
            "Use the provided reference material to build a coherent scene: "
            "keep the referenced subjects/identity consistent and describe "
            "an engaging action or moment for them."
            + cat_hint
        )
    if task_code == "s2v":
        return (
            "Show this subject performing a natural, expressive action in a "
            "fitting environment; keep their identity consistent with the "
            "reference image."
            + cat_hint
        )
    # Defensive fallback for unknown codes (the enhancer short-circuits
    # unknown codes earlier, so this path is rarely hit).
    return "Create a compelling short video scene." + cat_hint


# --------------------------------------------------------------------------- #
# Frame sampling (mirrors scail2_prompt_generator._sample_indices / _sample_urls)
# --------------------------------------------------------------------------- #
def _sample_indices(total: int, n: int) -> list[int]:
    """Return ``n`` unique indices in ``[0, total)`` sampled as evenly as possible.

    ``n`` is clamped to ``[1, total]`` before sampling. When ``n == 1`` the
    middle index is returned; when ``n == 2`` the endpoints are returned
    so a reference video gets both the first and last frame in its caption;
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
class H3PromptEnhancer:
    """MiniMax H3 prompt enhancer that talks to the project's LLMServiceConnector.

    Two-stage only for the ``reference`` path (caption reference material,
    then write the final H3 prompt). All other paths are single-stage:
    T2V has nothing to caption; I2V/FL2V/L2V/S2V anchor identity via the
    alignment directive / <Subject N> tags, so re-describing the reference
    would violate the H3 "describe the change" rule.

    Mirrors the structure of ``Scail2PromptEnhancer`` so the two LLM nodes
    share the project's logging + timeout patterns.
    """

    def __init__(
        self,
        llm_service_connector: Any,
        *,
        caption_sample_frames: int = DEFAULT_CAPTION_SAMPLE_FRAMES,
        image_detail: str = "auto",
        temperature: float = 0.4,
        max_tokens_caption: int = _DEFAULT_MAX_TOKENS_CAPTION,
        max_tokens_enhance: int = _DEFAULT_MAX_TOKENS_ENHANCE,
        timeout: Optional[int] = None,
    ):
        self.llm = llm_service_connector
        self.caption_sample_frames = max(
            MIN_CAPTION_SAMPLE_FRAMES,
            min(int(caption_sample_frames), MAX_CAPTION_SAMPLE_FRAMES),
        )
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
                    f"H3 {stage}: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"H3 {stage}: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
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

        Mirrors ``scail2_prompt_generator._build_messages``: each image
        gets a small ``[Image N]:`` caption preceding the image part so
        vision models that read the textual layout of the conversation
        know which image is which (matters for H3 multi-image Ref2VA,
        where image order maps to <Picture N> / <Subject N> numbering).
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
    # Stage 1: caption reference material (reference path only)
    # ------------------------------------------------------------------ #
    def _caption(
        self,
        frame_urls: list[str],
        user_prompt: str,
        seed: Optional[int],
    ) -> str:
        """Stage 1: caption reference images / sampled video frames.

        Identifies identity, wardrobe, scene, and features to preserve so
        stage 2 can lock them item by item instead of re-describing them.
        """
        system = caption_reference_prompt()
        user_text = (
            f"User idea (the motion / change to depict): {user_prompt.strip()}\n"
            f"The following images are reference material (reference images first, "
            f"then sampled video frames) in connection order. Caption each in order "
            f"(Image 1, Image 2, ...)."
        )
        messages = self._build_messages(
            system, user_text, frame_urls, self.image_detail
        )
        caption = self._invoke(
            messages,
            max_tokens=self.max_tokens_caption,
            seed=seed,
            stage="caption[reference]",
        )
        mie_log(
            f"H3 caption[reference]: {len(caption)} chars: {caption[:120]!r}"
        )
        return caption

    # ------------------------------------------------------------------ #
    # Stage 2: write the final structured H3 prompt
    # ------------------------------------------------------------------ #
    def _enhance(
        self,
        task_code: str,
        ref_urls: list[str],
        user_prompt: str,
        caption: str,
        seed: Optional[int],
        *,
        duration,
        aspect_ratio: str,
        cat_advice: str,
        output_language: str = DEFAULT_OUTPUT_LANGUAGE,
    ) -> str:
        """Stage 2: write the final structured H3 prompt.

        Picks the system prompt (T2V vs reference), builds the right user
        template, and for image-bearing paths prepends the official
        alignment directive verbatim. The model is instructed NOT to
        re-describe appearances already visible in the reference.
        """
        # Normalize the language code; unknown values fall back to English.
        lang = output_language if output_language in OUTPUT_LANGUAGES else DEFAULT_OUTPUT_LANGUAGE
        examples = load_bundled_examples(task_code, MAX_EXAMPLE_CHARS)
        if task_code == "t2v":
            system = system_t2v_prompt()
            user_text = user_t2v_prompt(
                idea=user_prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                category_advice=cat_advice,
                examples=examples,
                output_language=lang,
            )
        else:
            system = system_reference_prompt()
            align = align_directive_for(task_code, duration)
            # Full-reference mode (Ref2VA) uses the six-section format from
            # ref-en.txt; all other image tasks use the base three-section
            # format from base-en.txt.
            ref_mode = task_code == "reference"
            user_text = user_i2v_prompt(
                align_directive=align,
                idea=user_prompt,
                caption=caption,
                duration=duration,
                aspect_ratio=aspect_ratio,
                category_advice=cat_advice,
                examples=examples,
                output_language=lang,
                reference_mode=ref_mode,
            )
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
        first_frame: Any = None,
        last_frame: Any = None,
        reference_images: Any = None,
        reference_video: Any = None,
        seed: Optional[int] = None,
        duration=6.0,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        category: str = "none",
        output_language: str = DEFAULT_OUTPUT_LANGUAGE,
    ) -> str:
        """Run the H3 pipeline and return the final structured prompt.

        On an empty ``user_prompt`` the enhancer synthesizes a reasonable
        default idea from the task + category and runs the full pipeline
        anyway (so users can leave the box blank when they have reference
        media or just want a category-driven result). Returns the original
        ``user_prompt`` unchanged only on missing required media for the
        chosen task, on an LLM error, or on an unknown task code -- so a
        bad input never breaks the ComfyUI workflow.

        ``duration`` is a float (seconds) written into the prompt header
        and the FL2VA/L2VA final-timestamp directive. ``width`` / ``height``
        are reduced to a simplified ``W:H`` ratio string for the prompt
        header (the downstream H3 video node sets the real render size).

        Pipeline shape by task:
          * ``t2v``            -- single stage, no media required.
          * ``i2v_first``      -- single stage, needs ``first_frame``.
          * ``i2v_first_last`` -- single stage, needs both frames.
          * ``i2v_last``       -- single stage, needs ``last_frame``.
          * ``reference``      -- two stages: caption reference images /
                                  sampled video frames, then enhance.
                                  Needs ``reference_images`` and/or
                                  ``reference_video``.
          * ``s2v``            -- single stage, needs ``reference_images``
                                  (defines the subject only).
        """
        code = parse_task_code(task_type)
        if code not in TASK_CODES:
            mie_log(
                f"H3: unknown task {task_type!r} (parsed code={code!r}); returning original"
            )
            return user_prompt

        # If the user left the idea blank, synthesize a reasonable default
        # from the task + category instead of returning an empty string.
        # The node still has the reference media (for image-bearing tasks),
        # the category advice, and the few-shot examples to drive generation.
        if not user_prompt or not user_prompt.strip():
            user_prompt = _default_idea(code, category)
            mie_log(
                f"H3: empty user_prompt; using default idea for task={code} "
                f"(category={parse_category(category) or 'none'}): {user_prompt[:80]!r}"
            )

        cat_advice = category_advice(category)
        # Reduce width x height to a "W:H" ratio string for the prompt
        # header. Pure prompt metadata; the downstream video node decides
        # the real render size.
        aspect_ratio = aspect_ratio_string(width, height)

        # ---- Per-task media routing ---------------------------------- #
        if code == "t2v":
            ref_urls: list[str] = []
            caption = ""
            mie_log(
                f"H3: task=t2v duration={duration}s ratio={aspect_ratio} "
                f"cat={parse_category(category) or 'none'} temperature={self.temperature} "
                f"(single stage, no media)"
            )
        elif code == "i2v_first":
            ref_urls = image_tensor_batch_to_data_urls(first_frame)
            if not ref_urls:
                mie_log("H3: i2v_first requires first_frame; returning original")
                return user_prompt
            caption = ""
            mie_log(
                f"H3: task=i2v_first imgs={len(ref_urls)} duration={duration}s "
                f"ratio={aspect_ratio} cat={parse_category(category) or 'none'} "
                f"temperature={self.temperature} (single stage)"
            )
        elif code == "i2v_first_last":
            first_urls = image_tensor_batch_to_data_urls(first_frame)
            last_urls = image_tensor_batch_to_data_urls(last_frame)
            if not first_urls or not last_urls:
                mie_log(
                    "H3: i2v_first_last requires both first_frame and last_frame; returning original"
                )
                return user_prompt
            ref_urls = first_urls + last_urls
            caption = ""
            mie_log(
                f"H3: task=i2v_first_last imgs={len(ref_urls)} duration={duration}s "
                f"ratio={aspect_ratio} cat={parse_category(category) or 'none'} "
                f"temperature={self.temperature} (single stage)"
            )
        elif code == "i2v_last":
            ref_urls = image_tensor_batch_to_data_urls(last_frame)
            if not ref_urls:
                mie_log("H3: i2v_last requires last_frame; returning original")
                return user_prompt
            caption = ""
            mie_log(
                f"H3: task=i2v_last imgs={len(ref_urls)} duration={duration}s "
                f"ratio={aspect_ratio} cat={parse_category(category) or 'none'} "
                f"temperature={self.temperature} (single stage)"
            )
        elif code == "reference":
            ref_img_urls = image_tensor_batch_to_data_urls(reference_images)
            ref_vid_urls = image_tensor_batch_to_data_urls(reference_video)
            if not ref_img_urls and not ref_vid_urls:
                mie_log(
                    "H3: reference requires reference_images and/or reference_video; returning original"
                )
                return user_prompt
            frame_urls = _sample_urls(ref_vid_urls, self.caption_sample_frames) if ref_vid_urls else []
            # Images anchor identity/wardrobe; sampled video frames anchor
            # motion/structure. Feed both to the captioner in connection
            # order (images first, then sampled frames).
            caption_input_urls = ref_img_urls + frame_urls
            ref_urls = ref_img_urls if ref_img_urls else frame_urls
            mie_log(
                f"H3: task=reference ref_imgs={len(ref_img_urls)} "
                f"ref_vid_frames={len(ref_vid_urls)}->{len(frame_urls)} "
                f"detail={self.image_detail} duration={duration}s ratio={aspect_ratio} "
                f"cat={parse_category(category) or 'none'} temperature={self.temperature} "
                f"(two stage)"
            )
            caption = self._caption(caption_input_urls, user_prompt, seed=seed)
            if not caption:
                mie_log(
                    "H3: stage-1 caption returned empty (task=reference); returning original"
                )
                return user_prompt
        elif code == "s2v":
            ref_urls = image_tensor_batch_to_data_urls(reference_images)
            if not ref_urls:
                mie_log("H3: s2v requires reference_images; returning original")
                return user_prompt
            caption = ""
            mie_log(
                f"H3: task=s2v imgs={len(ref_urls)} duration={duration}s "
                f"ratio={aspect_ratio} cat={parse_category(category) or 'none'} "
                f"temperature={self.temperature} (single stage)"
            )
        else:  # pragma: no cover -- guarded by TASK_CODES check above
            mie_log(f"H3: unhandled task code {code!r}; returning original")
            return user_prompt

        enhanced = self._enhance(
            code,
            ref_urls,
            user_prompt,
            caption,
            seed=seed,
            duration=duration,
            aspect_ratio=aspect_ratio,
            cat_advice=cat_advice,
            output_language=output_language,
        )
        return enhanced or user_prompt


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class MiniMaxH3PromptGenerator:
    """ComfyUI node: rewrite a user prompt into a structured MiniMax H3 prompt.

    Selectable task paths: t2v / i2v_first / i2v_first_last / i2v_last /
    reference / s2v. An optional ``category`` dropdown (19 H3 use-case
    classes + none) injects per-category default styling advice.

    Output: a single ``STRING`` containing the structured H3 prompt.
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
                # --- Media inputs (per H3 material roles) ---
                # I2VA first frame.
                "first_frame": ("IMAGE",),
                # FL2VA last frame (paired with first_frame).
                "last_frame": ("IMAGE",),
                # Ref2VA / S2V multi-image reference (IMAGE batch). For
                # Ref2VA, image order maps to <Picture N> / <Subject N>
                # numbering, so connect them in the intended order.
                "reference_images": ("IMAGE",),
                # Ref2VA video reference, supplied as an IMAGE batch (one
                # tensor per frame). ``caption_sample_frames`` controls how
                # many are sampled for the stage-1 caption.
                "reference_video": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Reference video for the 'reference' task, supplied as an IMAGE batch (one tensor per frame). "
                            "caption_sample_frames of them are sampled evenly for the stage-1 caption. Audio references are not wired "
                            "(ComfyUI IMAGE port carries no audio); describe audio inside user_prompt instead."
                        ),
                    },
                ),
                # --- Specs ---
                # Width / height in pixels. Reduced to a simplified "W:H"
                # ratio string written into the prompt header (pure prompt
                # metadata; the downstream H3 video node sets the real
                # render size, so any positive integers are accepted).
                "width": (
                    "INT",
                    {
                        "default": DEFAULT_WIDTH,
                        "min": 1,
                        "tooltip": (
                            "Output frame width in pixels. Together with height it defines the aspect ratio "
                            "written into the prompt header (e.g. 1280x720 -> '16:9'). The downstream H3 video "
                            "node sets the real render size."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": DEFAULT_HEIGHT,
                        "min": 1,
                        "tooltip": (
                            "Output frame height in pixels. Together with width it defines the aspect ratio "
                            "written into the prompt header (e.g. 720x1280 -> '9:16')."
                        ),
                    },
                ),
                # Duration in seconds (float). Unrestricted range -- the
                # value is written into the prompt header and the
                # FL2VA/L2VA final-timestamp directive. NOTE: the downstream
                # H3 video node only accepts 5/6/8/12/15s; other values
                # will make THAT node error, not this one.
                "duration": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "step": 0.1,
                        "round": 0.1,
                        "tooltip": (
                            "Target video duration in seconds (float). Written into the prompt header and the "
                            "FL2VA/L2VA final-timestamp directive. NOTE: the downstream H3 video node only "
                            "accepts 5/6/8/12/15s; other values will make the video node error, not this one."
                        ),
                    },
                ),
                # --- Use-case category (injects default styling advice) ---
                "category": (
                    list(CATEGORIES),
                    {
                        "default": CATEGORIES[0],
                        "tooltip": (
                            "H3 use-case category. Adds default styling advice to the prompt "
                            "(e.g. cinematic-story -> cinematic color grading + 35-50mm; action -> motion blur + shake). "
                            "Does not change the task path. 'none' = no category-specific advice."
                        ),
                    },
                ),
                # --- Output language of the final prompt ---
                "output_language": (
                    list(OUTPUT_LANGUAGES),
                    {
                        "default": DEFAULT_OUTPUT_LANGUAGE,
                        "tooltip": (
                            "Output language of the final H3 prompt. 'en' = English (default), 'zh' = Chinese. "
                            "The section headers (Core idea / Soundscape / Music / Do not include) switch to the "
                            "chosen language; the descriptive body is written in that language too. EXCEPTION: "
                            "dialogue, lyrics, and visible on-screen text always stay in their original language."
                        ),
                    },
                ),
                # --- Sampling / quality knobs ---
                # How many frames to sample from reference_video for stage 1.
                # Renamed from num_frames to caption_sample_frames so users
                # do not mistake it for the output video length.
                "caption_sample_frames": (
                    "INT",
                    {
                        "default": DEFAULT_CAPTION_SAMPLE_FRAMES,
                        "min": MIN_CAPTION_SAMPLE_FRAMES,
                        "max": MAX_CAPTION_SAMPLE_FRAMES,
                        "tooltip": (
                            "Number of frames sampled evenly from reference_video for the stage-1 caption "
                            "(reference task only). Higher = more accurate caption but slower and costlier; "
                            "lower = faster but may miss motion. Does NOT set the output video length."
                        ),
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
                # models sometimes need 60-300s; default 120 mirrors SCAIL-2.
                "timeout": ([30, 60, 120, 300], {"default": 120}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed=None,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video=None,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        duration=6.0,
        category=CATEGORIES[0],
        output_language=DEFAULT_OUTPUT_LANGUAGE,
        caption_sample_frames=DEFAULT_CAPTION_SAMPLE_FRAMES,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=_DEFAULT_MAX_TOKENS_CAPTION,
        max_tokens_enhance=_DEFAULT_MAX_TOKENS_ENHANCE,
        timeout=120,
    ):
        enhancer = H3PromptEnhancer(
            llm_service_connector,
            caption_sample_frames=caption_sample_frames,
            image_detail=image_detail,
            temperature=temperature,
            max_tokens_caption=max_tokens_caption,
            max_tokens_enhance=max_tokens_enhance,
            timeout=timeout,
        )
        out = enhancer(
            task_type,
            user_prompt,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            reference_video=reference_video,
            seed=seed,
            duration=duration,
            width=width,
            height=height,
            category=category,
            output_language=output_language,
        )
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed=None,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video=None,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        duration=6.0,
        category=CATEGORIES[0],
        output_language=DEFAULT_OUTPUT_LANGUAGE,
        caption_sample_frames=DEFAULT_CAPTION_SAMPLE_FRAMES,
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
            str(width),
            str(height),
            str(duration),
            category,
            output_language,
            str(caption_sample_frames),
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
        # pixel data. Matches SCAIL-2's strategy so a tweak that changes
        # frame count or resolution triggers a re-run.
        for t in (first_frame, last_frame, reference_images, reference_video):
            if t is None:
                h.update(b"none")
            else:
                try:
                    shape = list(t.shape)
                except AttributeError:
                    shape = []
                h.update(repr(shape).encode("utf-8"))
        return h.hexdigest()
