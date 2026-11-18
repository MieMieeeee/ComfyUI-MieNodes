"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import bernini_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

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
                "source": ("IMAGE",),
                "reference_images": ("IMAGE",),
                # Video-reference frames. Used by ads2v (ad video to
                # insert into the source video) and vrc2v (reference
                # video content alongside subject images).
                "reference_video": ("IMAGE",),
                # How many of `source` to forward when the task treats
                # it as video frames. Ignored for image-source tasks.
                # Default 3 matches the upstream Bernini v2v/vi2v
                # templates. Range 1-16.
                "video_frames": (
                    "INT",
                    {"default": DEFAULT_VIDEO_FRAMES, "min": 1, "max": 16},
                ),
                # How many of `reference_video` to forward to the LLM.
                # 0 (default) preserves the legacy "all frames" behavior;
                # 1-16 uniformly samples that many frames. Only used by
                # ads2v and vrc2v tasks, which actually consume the input.
                "reference_video_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16},
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
        source=None,
        reference_images=None,
        reference_video=None,
        video_frames=DEFAULT_VIDEO_FRAMES,
        reference_video_frames=0,
        image_detail="auto",
        temperature=0.7,
        top_p=0.9,
        max_tokens=8192,
        timeout=30,
    ):
        enhancer = BerniniPromptEnhancer(
            llm_service_connector,
            video_frames=video_frames,
            reference_video_frames=reference_video_frames,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            task_type,
            user_prompt,
            source=source,
            reference_images=reference_images,
            reference_video=reference_video,
            image_detail=image_detail,
        )
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        task_type,
        user_prompt,
        seed,
        source=None,
        reference_images=None,
        reference_video=None,
        video_frames=DEFAULT_VIDEO_FRAMES,
        reference_video_frames=0,
        image_detail="auto",
        temperature=0.7,
        top_p=0.9,
        max_tokens=8192,
        timeout=30,
    ):
        h = hashlib.md5(usedforsecurity=False)
        h.update((task_type or "").encode("utf-8"))
        h.update((user_prompt or "").encode("utf-8"))
        h.update(str(seed).encode("utf-8"))
        h.update(str(video_frames).encode("utf-8"))
        h.update(str(reference_video_frames).encode("utf-8"))
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
        for src in (source, reference_images, reference_video):
            url = _tensor_to_url(src) or ""
            h.update(url[:64].encode("utf-8"))
        return h.hexdigest()
