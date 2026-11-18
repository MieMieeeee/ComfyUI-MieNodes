"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import h3_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

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
        h = hashlib.md5(usedforsecurity=False)
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
