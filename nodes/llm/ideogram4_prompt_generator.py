"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import ideogram4_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class Ideogram4PromptGenerator:
    """ComfyUI node: plain text → validated Ideogram 4 JSON caption via LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        tooltips = " | ".join(
            f"{mode}: {COMPOSITION_MODE_TOOLTIPS[mode]}" for mode in COMPOSITION_MODES
        )
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "composition_mode": (
                    COMPOSITION_MODES,
                    {
                        "default": "simple",
                        "tooltip": tooltips,
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
                "aspect_ratio": (
                    "STRING",
                    {
                        "default": "1:1",
                        "forceInput": True,
                        "tooltip": (
                            "Composition hint (W:H). Connect AspectRatioFromSize; "
                            "defaults to 1:1 when empty."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "timeout": ([30, 60, 120, 300], {"default": 120}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ideogram4_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        user_prompt,
        composition_mode="simple",
        seed=None,
        aspect_ratio="1:1",
        temperature=1.0,
        timeout=120,
    ):
        enhancer = Ideogram4PromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            timeout=timeout,
            composition_mode=composition_mode,
        )
        effective_ar = resolve_aspect_ratio((aspect_ratio or "").strip() or "1:1")
        out = enhancer(user_prompt, aspect_ratio=effective_ar, seed=seed)
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        user_prompt,
        composition_mode="simple",
        seed=None,
        aspect_ratio="1:1",
        temperature=1.0,
        timeout=120,
    ):
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_prompt,
            composition_mode,
            str(seed),
            aspect_ratio,
            str(temperature),
            str(timeout),
        ):
            h.update((part or "").encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
