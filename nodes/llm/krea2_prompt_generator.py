"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import krea2_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class Krea2PromptGenerator:
    """ComfyUI node: plain text -> expanded Krea-2 image prompt paragraph.

    Sends the user's idea to the configured LLM with Krea-2's official
    ``expansion.txt`` system prompt (bundled verbatim) and returns the
    single-paragraph answer. Optional ``aspect_ratio`` hint is forwarded
    in the user message; pass ``"auto"`` to let the model decide.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Plain-text idea to expand into a Krea-2 prompt. "
                            "Per Krea docs, long detailed prompts yield the best "
                            "results; the upstream LLM will lightly polish rather "
                            "than over-expand if the idea is already detailed."
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
                # Aspect-ratio hint. Krea-2 itself picks the framing for
                # ``"auto"``; passing ``1:1`` (default) or any ``W:H``
                # nudges the style/medium/composition planner inside the
                # upstream expansion prompt. Connect AspectRatioFromSize
                # to feed an external pipeline decision.
                "aspect_ratio": (
                    "STRING",
                    {
                        "default": DEFAULT_ASPECT_RATIO,
                        "forceInput": True,
                        "tooltip": (
                            "Composition hint (W:H or 'auto'). Connect AspectRatioFromSize; "
                            "defaults to 1:1 when empty."
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
                    },
                ),
                # Per-call timeout override. Long reasoning-model
                # responses on heavy prompts sometimes need 60-300s;
                # 120 s matches the rest of the family.
                "timeout": ([30, 60, 120, 300], {"default": _DEFAULT_TIMEOUT}),
                # Token budget. Krea docs recommend long detailed prompts;
                # reasoning models spend part of this on their internal
                # chain-of-thought before the visible paragraph.
                "max_tokens": (
                    "INT",
                    {
                        "default": _MAX_TOKENS_DEFAULT,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("krea2_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        user_prompt,
        seed=None,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        temperature=_DEFAULT_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT,
        max_tokens=_MAX_TOKENS_DEFAULT,
    ):
        enhancer = Krea2PromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(user_prompt, aspect_ratio=aspect_ratio, seed=seed)
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        user_prompt,
        seed=None,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        temperature=_DEFAULT_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT,
        max_tokens=_MAX_TOKENS_DEFAULT,
    ):
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_prompt,
            aspect_ratio,
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update((part or "").encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
