"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import scail2_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

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
        h = hashlib.md5(usedforsecurity=False)
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
