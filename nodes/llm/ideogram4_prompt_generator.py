"""Ideogram 4 structured-caption prompt generator.

Expands a plain user idea into Ideogram 4 JSON via LLM. Output passes through
``Ideogram4PromptFormatter`` for ComfyUI sampling and ``Ideogram4PromptBuilderKJ`` import.
Aspect ratio is supplied externally (``AspectRatioFromSize``); ``aspect_ratio`` in JSON is
stripped by Formatter when present.
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
        PROMPT_PROFILES,
        build_ideogram4_messages,
        resolve_aspect_ratio,
    )
except ImportError:
    from .ideogram4_prompts import (
        PROMPT_PROFILES,
        build_ideogram4_messages,
        resolve_aspect_ratio,
    )


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

_STRIP_ASPECT_RATIO = True
_STRIP_BBOXES = False
_MAX_TOKENS = 16384


try:
    from _mienodes_internal.nodes.llm.ideogram4_prompt_formatter import (
        compact_caption,
        format_ideogram4_caption,
        parse_caption_dict,
        strip_code_fences,
    )
except ImportError:
    from .ideogram4_prompt_formatter import (
        compact_caption,
        format_ideogram4_caption,
        parse_caption_dict,
        strip_code_fences,
    )


def postprocess_caption(raw_text: str) -> str:
    """Parse LLM output into compact JSON for sampling / KJ import."""
    try:
        prompt, _log = format_ideogram4_caption(raw_text)
        return prompt
    except ValueError:
        mie_log("[Ideogram4PromptGenerator] validation failed; best-effort compact output")
        text = strip_code_fences(raw_text)
        data, _ = parse_caption_dict(text)
        if data is None:
            return text.strip()
        if _STRIP_ASPECT_RATIO:
            data.pop("aspect_ratio", None)
        if _STRIP_BBOXES:
            elements = data.get("compositional_deconstruction", {}).get("elements", [])
            if isinstance(elements, list):
                for element in elements:
                    if isinstance(element, dict):
                        element.pop("bbox", None)
        return compact_caption(data)


class Ideogram4PromptEnhancer:
    """Expand a user idea into Ideogram 4 JSON via LLM."""

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = 1.0,
        timeout: int = 120,
        prompt_profile: str = "official_v1",
    ):
        self.llm = llm_service_connector
        self.temperature = temperature
        self.timeout = timeout
        self.prompt_profile = prompt_profile

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
        messages = build_ideogram4_messages(prompt, ar, prompt_profile=self.prompt_profile)
        raw = self._invoke(messages, seed=seed)
        return postprocess_caption(raw)


class Ideogram4PromptGenerator:
    """ComfyUI node: plain text → Ideogram 4 JSON caption via LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt_profile": (
                    PROMPT_PROFILES,
                    {
                        "default": "official_v1",
                        "tooltip": (
                            "official_v1: Ideogram magic prompt (strong LLM). "
                            "full_palette: v1 rules + structured color palettes (medium+). "
                            "compact: short prompt (light LLM)."
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
        prompt_profile="official_v1",
        seed=None,
        aspect_ratio="1:1",
        temperature=1.0,
        timeout=120,
    ):
        enhancer = Ideogram4PromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            timeout=timeout,
            prompt_profile=prompt_profile,
        )
        effective_ar = resolve_aspect_ratio((aspect_ratio or "").strip() or "1:1")
        out = enhancer(user_prompt, aspect_ratio=effective_ar, seed=seed)
        return (out,)

    def is_changed(
        self,
        llm_service_connector,
        user_prompt,
        prompt_profile="official_v1",
        seed=None,
        aspect_ratio="1:1",
        temperature=1.0,
        timeout=120,
    ):
        h = hashlib.md5()
        for part in (
            user_prompt,
            prompt_profile,
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
