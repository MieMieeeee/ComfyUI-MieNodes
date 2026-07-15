"""Krea-2 prompt-enhancer ComfyUI node.

Expands a plain user idea into a single-paragraph, Krea-2-flavoured
text-to-image prompt by sending the upstream ``expansion.txt`` system
prompt to the configured ``LLMServiceConnector``. No images, no JSON
contract -- the LLM emits one cohesive paragraph per upstream rule 6.

Sits next to ``Ideogram4PromptGenerator`` and ``BerniniPromptGenerator``
in the same Prompt Generator category and follows the same I/O shape:
single STRING output, ``is_changed`` hash, per-call timeout override.
"""
from __future__ import annotations

import hashlib
import re
import time

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:

        def mie_log(msg):  # noqa: ARG001
            pass


try:
    from _mienodes_internal.nodes.llm.krea2_prompts import (
        DEFAULT_ASPECT_RATIO,
        build_krea2_messages,
        resolve_aspect_ratio,
    )
except ImportError:
    from .krea2_prompts import (
        DEFAULT_ASPECT_RATIO,
        build_krea2_messages,
        resolve_aspect_ratio,
    )


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Default token budget. Krea docs explicitly say long detailed prompts
# yield best results, so we leave generous headroom. Reasoning models
# (M3, DeepSeek-R1, GLM-5.x) also emit a chain-of-thought block inside
# this budget before the final paragraph; 4096 is the lowest number
# that comfortably covers both. Mirrors the project's pattern of
# 2048-8192 across the LLM node family.
_MAX_TOKENS_DEFAULT = 4096
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Default sampling knobs. ``temperature=0.7`` matches the Bernini / Flux2
# family; Krea-2's own examples range from photoreal to stylised, so a
# mild default is a reasonable middle ground.
_DEFAULT_TEMPERATURE = 0.7

# Default timeout. Krea-2 expansion runs as one LLM call (no vision,
# no JSON parsing), so 120 s is comfortably above the typical 10-30 s
# response window for most backends.
_DEFAULT_TIMEOUT = 120

# Strip a leading ``<think>...</think>`` block that reasoning models
# may emit before the visible answer. The Krea-2 system prompt tells
# the LLM to keep planning internal (rule 3) and to write one cohesive
# paragraph (rule 6), so any leaked thinking wrapper must be removed
# before the node returns the string.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


# --------------------------------------------------------------------------- #
# Response postprocess
# --------------------------------------------------------------------------- #
def postprocess_paragraph(raw_text: str) -> str:
    """Strip a leading ``<think>...</think>`` block (if any) and return
    the visible paragraph. Whitespace-only inputs collapse to ``""``.

    Per upstream rule 6 the visible answer is one cohesive paragraph
    with no bullets / JSON / markdown; reasoning models may still emit
    a leading think wrapper, so we sanitize it here.
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = _THINK_BLOCK_RE.sub("", text, count=1).strip()
    return text


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class Krea2PromptEnhancer:
    """Expand a user idea into a Krea-2 paragraph via the configured LLM.

    Mirrors the I/O shape of ``Ideogram4PromptEnhancer`` so the node
    can be swapped into existing workflows with minimal wiring.
    """

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _MAX_TOKENS_DEFAULT,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Per-call timeout override. ``None`` would mean "leave the
        # connector's own timeout alone"; the node always wires a
        # dropdown default, so this is always set in practice.
        self._timeout_override = int(timeout) if timeout else None

    def _invoke(self, messages: list[dict], seed=None) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None and prev_timeout is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out:
                mie_log(
                    f"Krea2: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"Krea2: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
            )
            return out
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    def __call__(
        self,
        user_prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        seed=None,
    ) -> str:
        """Run one LLM call and return the postprocessed paragraph.

        Raises ``ValueError`` when ``user_prompt`` is empty -- the
        upstream expansion prompt is meaningless without an idea to
        expand, and silent fall-through to the LLM would burn tokens.
        """
        prompt = (user_prompt or "").strip()
        if not prompt:
            raise ValueError("Krea2PromptEnhancer: user_prompt is empty")

        ar = resolve_aspect_ratio(aspect_ratio)
        messages = build_krea2_messages(prompt, ar)
        raw = self._invoke(messages, seed=seed)
        return postprocess_paragraph(raw)


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
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
        h = hashlib.md5()
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
