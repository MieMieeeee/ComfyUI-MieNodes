"""Bernini task-aware prompt generator.

Adapts ``bernini.prompt_enhancer.PromptEnhancer`` (bytedance/Bernini) for the
project's LLMServiceConnector pipeline. Routes the user's raw instruction to
one of 12 task-specific system prompts / user templates, optionally feeds in a
single reference image, a list of reference images, and/or a batch of source
video frames, and parses JSON-mode responses for the subject-driven tasks.

All template strings live in ``bernini_prompts`` (Apache 2.0, verbatim copy
from the upstream source). This file adds the I/O glue and the ComfyUI node
class.
"""

import hashlib
import json
import re
import time

import torch

try:
    from _mienodes_internal.core.utils import (
        image_tensor_to_data_url,
        image_tensor_batch_to_data_urls,
        build_multimodal_user_content,
        mie_log,
    )
except ImportError:
    from ...core.utils import (
        image_tensor_to_data_url,
        image_tensor_batch_to_data_urls,
        build_multimodal_user_content,
        mie_log,
    )

try:
    from _mienodes_internal.nodes.llm.bernini_prompts import (
        TASK_TYPES,
        TASK_CODES,
        JSON_MODE_TASKS,
        SYSTEM_PROMPTS,
        T2V_A14B_EN_SYS_PROMPT,
        parse_task_code,
        T2I_A14B_EN_SYS_PROMPT,
        R2V_TEMPLATE,
        R2I_TEMPLATE,
        RI2I_TEMPLATE,
        VR2V_TEMPLATE,
        V2V_TEMPLATE,
        I2I_TEMPLATE,
        I2V_TEMPLATE,
        VI2V_TEMPLATE,
        ADS2V_TEMPLATE,
    )
except ImportError:
    from .bernini_prompts import (
        TASK_TYPES,
        TASK_CODES,
        JSON_MODE_TASKS,
        SYSTEM_PROMPTS,
        T2V_A14B_EN_SYS_PROMPT,
        parse_task_code,
        T2I_A14B_EN_SYS_PROMPT,
        R2V_TEMPLATE,
        R2I_TEMPLATE,
        RI2I_TEMPLATE,
        VR2V_TEMPLATE,
        V2V_TEMPLATE,
        I2I_TEMPLATE,
        I2V_TEMPLATE,
        VI2V_TEMPLATE,
        ADS2V_TEMPLATE,
    )


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

# How many video frames to feed Bernini's v2v / vi2v / rv2v / ads2v tasks.
# Matches the upstream default.
DEFAULT_VIDEO_FRAMES = 3


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def _extract_json_text(text):
    """Pull ``rewritten_text`` out of a JSON-mode response, with fallbacks.

    Handles three common shapes the model may emit:
      1. A bare JSON object: ``{"rewritten_text": "..."}``
      2. A fenced JSON object: ````json\\n{...}\\n``` ````
      3. Prose around a JSON object: ``Sure! Here is ... {"rewritten_text": ...}``
    If none of those parse, returns the input stripped - same as the upstream
    fallback when ``json.loads`` succeeds but the key is missing.
    """
    if not text:
        return text
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("rewritten_text"), str):
            return obj["rewritten_text"].strip()
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("rewritten_text"), str):
                return obj["rewritten_text"].strip()
        except json.JSONDecodeError:
            pass
    return s


# --------------------------------------------------------------------------- #
# Media adapters (ComfyUI IMAGE tensor <-> OpenAI image_url content part)
# --------------------------------------------------------------------------- #
def _tensor_to_url(t):
    """Convert a single ComfyUI IMAGE tensor (H,W,C) into a data URL, or None."""
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return None
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        return None
    return image_tensor_to_data_url(t)


def _sample_indices(total, n):
    """Uniformly sample ``n`` indices in [0, total) preserving endpoints."""
    if total <= 0 or n <= 0:
        return []
    if total <= n:
        return list(range(total))
    if n == 1:
        return [total // 2]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _sample_urls(urls, n):
    if not urls or n <= 0:
        return []
    idx = _sample_indices(len(urls), n)
    idx = [max(0, min(i, len(urls) - 1)) for i in idx]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(urls[i])
    return out


def _build_messages(system_prompt, user_text, image_urls, image_detail="auto"):
    """Build chat messages with one system turn and one mixed-content user turn.

    Each image gets a small `[Image N]:` caption preceding the image part, so
    vision models that look at the textual layout of the conversation know which
    image is which (Bernini upstream behavior).
    """
    parts = []
    for i, url in enumerate(image_urls or []):
        parts.append({"type": "text", "text": f"\n[Image {i}]:"})
        parts.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
    if user_text:
        parts.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parts},
    ]


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #


def _image_tensor_summary(t):
    """Return a small dict describing ``t`` for diagnostic logging.

    Returns ``None`` for ``None`` / non-tensor inputs. For IMAGE tensors
    (3D HWC or 4D NHWC) it reports the frame count, height, width, and total
    pixel count - cheap to compute and very useful when figuring out why a
    request blew up in size.
    """
    if t is None or not isinstance(t, torch.Tensor):
        return None
    if t.ndim == 4:
        n, h, w = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
    elif t.ndim == 3:
        n, h, w = 1, int(t.shape[0]), int(t.shape[1])
    else:
        return {"shape": list(t.shape)}
    return {
        "count": n,
        "h": h,
        "w": w,
        "pixels": n * h * w,
        "dtype": str(t.dtype).replace("torch.", ""),
    }


def _url_bytes(urls):
    """Sum the byte length of every data URL prefix. Cheap and good enough
    for an at-a-glance size estimate in the log."""
    return sum(len(u) for u in urls or [])


class BerniniPromptEnhancer:
    """Task-aware prompt enhancer that talks to the project's LLMServiceConnector.

    Mirrors the routing in ``bernini.prompt_enhancer.PromptEnhancer`` but is
    decoupled from the OpenAI SDK - the chat call is delegated to whatever
    ``LLMServiceConnector`` the user wires into the ComfyUI node.
    """

    def __init__(
        self,
        llm_service_connector,
        video_frames=DEFAULT_VIDEO_FRAMES,
        temperature=0.7,
        top_p=0.9,
        max_tokens=8192,
        timeout=None,
    ):
        self.llm = llm_service_connector
        self.video_frames = max(1, int(video_frames))
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # Per-call timeout override (seconds). ``None`` means keep whatever
        # the connector already has. When set, ``_chat`` saves and restores
        # the connector's ``timeout`` around ``invoke`` so the connector
        # object is safe to share with other nodes.
        self._timeout_override = int(timeout) if timeout else None

    def _chat(self, system_prompt, user_text, image_urls, json_mode=False, image_detail="auto"):
        messages = _build_messages(system_prompt, user_text, image_urls, image_detail=image_detail)
        model_name = getattr(self.llm, "model", "?")
        img_bytes = _url_bytes(image_urls)
        mie_log(
            f"Bernini _chat: model={model_name} json_mode={json_mode} image_detail={image_detail} "
            f"images={len(image_urls)} img_bytes={img_bytes} (~{img_bytes / 1024:.1f} KB) "
            f"system_chars={len(system_prompt or '')} user_chars={len(user_text or '')}"
        )
        t0 = time.perf_counter()
        # Save and restore the connector's timeout so the override only
        # applies to this single call. Skip the dance entirely when the
        # override matches the connector's current value (or is unset).
        original_timeout = None
        timeout_restore_needed = (
            self._timeout_override is not None
            and getattr(self.llm, "timeout", None) != self._timeout_override
        )
        if timeout_restore_needed:
            original_timeout = self.llm.timeout
            self.llm.timeout = self._timeout_override
        try:
            out = self.llm.invoke(
                messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
        finally:
            if timeout_restore_needed:
                self.llm.timeout = original_timeout
        elapsed = time.perf_counter() - t0
        if out is None:
            mie_log(f"Bernini _chat: model={model_name} returned None after {elapsed:.2f}s")
            return None
        out = out.strip()
        if not out:
            mie_log(f"Bernini _chat: model={model_name} returned empty after {elapsed:.2f}s")
            return None
        mie_log(
            f"Bernini _chat: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
        )
        return _extract_json_text(out) if json_mode else out

    def __call__(
        self,
        task_type,
        user_prompt,
        source_images=None,
        reference_images=None,
        image_detail="auto",
    ):
        """Build a request for ``task_type`` and return the rewritten prompt.

        Media inputs:
          - `source_images`: the thing being operated on. 1 image for
            image-source tasks (i2i / ri2i / i2v), or a batch of video
            frames for video-source tasks (v2v / mv2v / vi2v / rv2v /
            vrc2v / ads2v). Unused for r2i / r2v (subject-driven new
            image / video generation).
          - `reference_images`: 0+ reference images. The subject for
            r2i / r2v, the guidance material for ri2i / vi2v / rv2v /
            vrc2v, extra visual context for i2i / i2v.

        Returns the original ``user_prompt`` unchanged if the LLM fails
        or the call is empty, matching the upstream's graceful-
        degradation behavior. Tasks that strictly require a source
        image (i2i / ri2i) also fall back to ``user_prompt`` when
        ``source_images`` is empty.
        """
        if not user_prompt or not user_prompt.strip():
            return user_prompt

        code = parse_task_code(task_type)

        # Convert both IMAGE tensors to data URLs up front.
        # `image_tensor_batch_to_data_urls` handles 3D (HWC) and 4D
        # (NHWC) tensors uniformly; an empty / None tensor yields [].
        source_urls = image_tensor_batch_to_data_urls(source_images)
        ref_urls = image_tensor_batch_to_data_urls(reference_images)

        summary = {
            "source": _image_tensor_summary(source_images),
            "ref": _image_tensor_summary(reference_images),
        }
        model_name = getattr(self.llm, "model", "?")
        video_frames_setting = getattr(self, "video_frames", None)
        max_tokens = getattr(self, "max_tokens", None)
        mie_log(
            f"Bernini: task={code} model={model_name} prompt_chars={len(user_prompt)} "
            f"video_frames_setting={video_frames_setting} max_tokens={max_tokens} "
            f"source={summary['source']} ref={summary['ref']}"
        )

        # For video-source tasks the routing branches below will use
        # `frame_urls` (sampled from source_urls). For image-source
        # tasks the routing branches use source_urls[:1] directly and
        # ignore `frame_urls`.
        frame_urls = _sample_urls(source_urls, self.video_frames)

        total_img_bytes = _url_bytes(source_urls) + _url_bytes(ref_urls)
        mie_log(
            f"Bernini: built request: source_urls={len(source_urls)} source_kb={_url_bytes(source_urls) / 1024:.1f} "
            f"ref_urls={len(ref_urls)} ref_kb={_url_bytes(ref_urls) / 1024:.1f} "
            f"frame_urls={len(frame_urls)} frame_kb={_url_bytes(frame_urls) / 1024:.1f} "
            f"total_img_kb={total_img_bytes / 1024:.1f}"
        )

        # Strip the " - 中文" display suffix to get the short code used
        # for routing and prompt lookups. Bare codes (legacy saved
        # workflows) pass through unchanged.
        base_sys = SYSTEM_PROMPTS.get(code, SYSTEM_PROMPTS["default"])
        json_mode = code in JSON_MODE_TASKS

        # ---- pure text tasks (no media) ----
        if code == "t2v":
            return (
                self._chat(T2V_A14B_EN_SYS_PROMPT, user_prompt, [], json_mode=False)
                or user_prompt
            )
        if code == "t2i":
            return (
                self._chat(T2I_A14B_EN_SYS_PROMPT, user_prompt, [], json_mode=False)
                or user_prompt
            )

        # ---- subject-driven generation (source is unused) ----
        if code == "r2v":
            text = R2V_TEMPLATE.format(
                image_num=max(len(ref_urls), 1), original_text=user_prompt
            )
            return self._chat(base_sys, text, ref_urls, json_mode=True, image_detail=image_detail) or user_prompt
        if code == "r2i":
            text = R2I_TEMPLATE.format(
                image_num=max(len(ref_urls), 1), original_text=user_prompt
            )
            return self._chat(base_sys, text, ref_urls, json_mode=True, image_detail=image_detail) or user_prompt

        # ---- image-source tasks (take first of source_urls) ----
        if code == "i2i":
            if not source_urls:
                return user_prompt
            text = I2I_TEMPLATE.format(user_prompt=user_prompt)
            return self._chat(base_sys, text, source_urls[:1], json_mode=False, image_detail=image_detail) or user_prompt
        if code == "i2v":
            # Prefer source_images[0] as anchor; fall back to ref_urls[0]
            # if only references are wired (matches the legacy i2v
            # behavior where the first video frame was used as anchor).
            if source_urls:
                imgs = source_urls[:1]
            elif ref_urls:
                imgs = ref_urls[:1]
            else:
                return user_prompt
            text = I2V_TEMPLATE.format(user_prompt=user_prompt, image_num=len(imgs))
            return self._chat(base_sys, text, imgs, json_mode=False, image_detail=image_detail) or user_prompt
        if code == "ri2i":
            # image0 = source (the image to edit), image1..N = references.
            # If no explicit source is wired, treat ref_urls[0] as the
            # source so the user can still feed a 1+ image batch.
            if source_urls:
                imgs = source_urls + ref_urls
                ref_num = len(ref_urls)
            elif ref_urls:
                imgs = ref_urls
                ref_num = max(len(ref_urls) - 1, 0)
            else:
                return user_prompt
            text = RI2I_TEMPLATE.format(
                ref_num=ref_num, original_text=user_prompt
            )
            return self._chat(base_sys, text, imgs, json_mode=True, image_detail=image_detail) or user_prompt

        # ---- video-source tasks (sampled frames + optional refs) ----
        if code in ("v2v", "mv2v"):
            text = V2V_TEMPLATE.format(user_prompt=user_prompt)
            return self._chat(base_sys, text, frame_urls, json_mode=False, image_detail=image_detail) or user_prompt
        if code == "ads2v":
            text = ADS2V_TEMPLATE.format(user_prompt=user_prompt)
            return self._chat(base_sys, text, frame_urls, json_mode=False, image_detail=image_detail) or user_prompt
        if code == "vi2v":
            text = VI2V_TEMPLATE.format(user_prompt=user_prompt, image_num=len(ref_urls))
            return (
                self._chat(base_sys, text, frame_urls + ref_urls, json_mode=False)
                or user_prompt
            )
        if code in ("rv2v", "vrc2v"):
            text = VR2V_TEMPLATE.format(
                image_num=max(len(ref_urls), 1), original_text=user_prompt
            )
            return (
                self._chat(base_sys, text, frame_urls + ref_urls, json_mode=True)
                or user_prompt
            )

        return user_prompt


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class BerniniPromptGenerator:
    """ComfyUI node that rewrites a user prompt with a Bernini task template.

    Selectable modes:
      t2i, t2v, i2i, r2i, ri2i, i2v, v2v, mv2v, r2v, vi2v, rv2v, vrc2v, ads2v

    Note: `ri2i` is a MieNodes extension task (not part of the upstream
    bytedance/Bernini 12-task set) that fills the symmetric gap between
    `i2i` (single source image) and `r2i` (reference-driven new image).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "task_type": (list(TASK_TYPES), {"default": "t2i - 文生图"}),
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
                # The thing being operated on. For image-source tasks
                # (i2i / ri2i / i2v) the first image is used. For
                # video-source tasks (v2v / mv2v / vi2v / rv2v / vrc2v
                # / ads2v) the batch is sampled to `video_frames` by
                # index. For r2i / r2v (subject-driven generation)
                # this is unused - the subject lives in `reference_images`.
                "source_images": ("IMAGE",),
                "reference_images": ("IMAGE",),
                # How many of `source_images` to forward when the task
                # treats it as video frames. Ignored for image-source
                # tasks. Default 3 matches the upstream Bernini v2v/vi2v
                # templates. Range 1-16.
                "video_frames": (
                    "INT",
                    {"default": DEFAULT_VIDEO_FRAMES, "min": 1, "max": 16},
                ),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 8192, "min": 64, "max": 32768},
                ),
                # Per-call timeout override (seconds). Replaces the
                # connector's own `timeout` for this call only and is
                # restored on return. Defaults to 30s to match the
                # connector default; bump to 60/120/300 for heavy
                # vision tasks (e.g. rv2v with many frames on M3).
                "timeout": ([30, 60, 120, 300], {"default": 30}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bernini_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed=None,
        source_images=None,
        reference_images=None,
        video_frames=DEFAULT_VIDEO_FRAMES,
        image_detail="auto",
        temperature=0.7,
        top_p=0.9,
        max_tokens=8192,
        timeout=30,
    ):
        enhancer = BerniniPromptEnhancer(
            llm_service_connector,
            video_frames=video_frames,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            task_type,
            user_prompt,
            source_images=source_images,
            reference_images=reference_images,
            image_detail=image_detail,
        )
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed,
        source_images=None,
        reference_images=None,
        video_frames=DEFAULT_VIDEO_FRAMES,
        image_detail="auto",
        temperature=0.7,
        top_p=0.9,
        max_tokens=8192,
        timeout=30,
    ):
        h = hashlib.md5()
        h.update((task_type or "").encode("utf-8"))
        h.update((user_prompt or "").encode("utf-8"))
        h.update(str(seed).encode("utf-8"))
        h.update(str(video_frames).encode("utf-8"))
        h.update((image_detail or "auto").encode("utf-8"))
        h.update(str(temperature).encode("utf-8"))
        h.update(str(top_p).encode("utf-8"))
        h.update(str(max_tokens).encode("utf-8"))
        h.update(str(timeout).encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(llm_service_connector.api_url).encode("utf-8"))
            h.update(str(llm_service_connector.api_token).encode("utf-8"))
            h.update(str(llm_service_connector.model).encode("utf-8"))
        # A short signature of the media inputs is enough to detect change;
        # the full data URL is large and changes the hash too aggressively.
        for src in (source_images, reference_images):
            url = _tensor_to_url(src) or ""
            h.update(url[:64].encode("utf-8"))
        return h.hexdigest()
