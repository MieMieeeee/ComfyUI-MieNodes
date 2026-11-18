"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import ltx25_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class LTX25PromptGenerator:
    """ComfyUI node: plain text (+ optional first frame) -> LTX-2.5
    audio-visual caption paragraph.

    Sends the user's idea to the configured LLM with the official
    LTX-2.5 (gemma4) system prompt and returns the single-paragraph
    caption. In i2v mode the reference IMAGE is attached directly to
    the single multimodal call.
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
                            "t2v: text-to-video caption (text-only LLM call).\n"
                            "i2v: image-to-video caption -- connect the reference "
                            "first frame to `image`; it is attached directly to "
                            "the multimodal LLM call, which then writes the "
                            "caption that begins on that exact frame."
                        ),
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Short idea / raw request to expand into a LTX-2.5 "
                            "caption. Every element you state is preserved; left "
                            "blank, the node synthesizes a default idea."
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
                    },
                ),
            },
            "optional": {
                # Reference first frame for i2v mode. Ignored in t2v.
                # A connected batch uses only the first image.
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "i2v only: the exact first frame of the video. "
                            "Required in i2v mode; without it the node returns "
                            "the original prompt."
                        ),
                    },
                ),
                "image_detail": (
                    ["auto", "low", "high"],
                    {"default": "auto"},
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "0.8 matches the LTX2 sibling node. Official gemma4 "
                            "enhancement runs greedy for deterministic captions "
                            "-- set 0.0 to reproduce that."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _DEFAULT_MAX_TOKENS,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                    },
                ),
                # Per-call timeout override. Long vision-model responses
                # on heavy prompts sometimes need 60-300s; 120 mirrors
                # the rest of the family.
                "timeout": ([30, 60, 120, 300], {"default": _DEFAULT_TIMEOUT}),
                # Multi-shot caption toggle (LTX-2.5 native multi-cut).
                # When True, an explicit directive is appended to the user
                # turn covering the C1-C4 cut checklist (see spec sec 4.2 /
                # template E); i2v also gets the sec 4.5 OPENING-shot note.
                # Default off -- the upstream single-shot caption style is
                # the safe default.
                "multishot": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ltx25_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        mode,
        user_prompt,
        seed=None,
        image=None,
        image_detail="auto",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        multishot=False,
    ):
        enhancer = LTX25PromptEnhancer(
            llm_service_connector,
            image_detail=image_detail,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            mode, user_prompt, image=image, seed=seed, multishot=bool(multishot)
        )
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        mode,
        user_prompt,
        seed=None,
        image=None,
        image_detail="auto",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        multishot=False,
    ):
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_prompt,
            mode,
            image_detail,
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
            str(bool(multishot)),
        ):
            h.update((part or "").encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        # Cheap media signature: just the tensor shape, not the full
        # pixel data. Matches SCAIL-2 / H3's strategy so swapping the
        # i2v reference first frame triggers a re-run.
        if image is None:
            h.update(b"none")
        else:
            try:
                shape = list(image.shape)
            except AttributeError:
                shape = []
            h.update(repr(shape).encode("utf-8"))
        return h.hexdigest()
