"""Ideogram 4 structured-caption prompt generator.

Expands a plain user idea into validated, compact Ideogram 4 JSON via LLM.
Formatting and schema validation are built in — invalid LLM output raises ``ValueError``,
consistent with other MieNodes prompt generator nodes.
"""

from __future__ import annotations

import hashlib

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:
        def mie_log(msg):  # noqa: ARG001
            pass

try:
    from _mienodes_internal.nodes.llm.ideogram4_prompts import (
        COMPOSITION_MODES,
        COMPOSITION_MODE_TOOLTIPS,
        build_ideogram4_messages,
        resolve_aspect_ratio,
    )
except ImportError:
    from .ideogram4_prompts import (
        COMPOSITION_MODES,
        COMPOSITION_MODE_TOOLTIPS,
        build_ideogram4_messages,
        resolve_aspect_ratio,
    )


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

_MAX_TOKENS = 16384


try:
    from _mienodes_internal.nodes.llm.ideogram4_prompt_formatter import format_ideogram4_caption
except ImportError:
    from .ideogram4_prompt_formatter import format_ideogram4_caption


def postprocess_caption(raw_text: str) -> str:
    """Parse and validate LLM output into compact JSON. Raises on parse/validation failure."""
    prompt, log = format_ideogram4_caption(raw_text)
    if log and log != "OK: no fixes needed":
        mie_log(f"[Ideogram4PromptGenerator] {log}")
    return prompt


class Ideogram4PromptEnhancer:
    """Expand a user idea into Ideogram 4 JSON via LLM."""

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = 1.0,
        timeout: int = 120,
        composition_mode: str = "simple",
    ):
        self.llm = llm_service_connector
        self.temperature = temperature
        self.timeout = timeout
        self.composition_mode = composition_mode

    def _invoke(self, messages: list[dict], seed=None) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if prev_timeout is not None:
                self.llm.timeout = self.timeout
            return self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=_MAX_TOKENS,
            )
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    def __call__(
        self,
        user_prompt: str,
        aspect_ratio: str = "1:1",
        seed=None,
    ) -> str:
        prompt = (user_prompt or "").strip()
        if not prompt:
            raise ValueError("user_prompt is empty")

        ar = resolve_aspect_ratio(aspect_ratio)
        messages = build_ideogram4_messages(
            prompt, ar, composition_mode=self.composition_mode
        )
        raw = self._invoke(messages, seed=seed)
        return postprocess_caption(raw)


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
        h = hashlib.md5()
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
