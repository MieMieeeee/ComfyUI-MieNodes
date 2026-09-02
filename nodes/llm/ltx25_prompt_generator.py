"""LTX-2.5 prompt-enhancer ComfyUI node.

Converts a user's short idea (plus, for i2v, the reference first frame)
into a single-paragraph audio-visual caption in the style of the LTX-2.5
training captions, using the official gemma4 system prompts bundled
under ``prompts/ltx25/`` and an ``LLMServiceConnector``-backed LLM.
LTX-2.5 is gemma4-only (see ``ltx25_prompts.py`` for the evidence
chain); the only mode switch is t2v vs i2v.

Pipeline shape (single-stage for both modes, mirroring upstream
``base_encoder.py`` and the Bernini / H3 i2v paths):
  * ``t2v`` -- one LLM call, text only.
  * ``i2v`` -- one multimodal LLM call: the reference first frame is
               attached to the user turn directly, next to the official
               ``"User Raw Input Prompt: ..."`` line.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional

try:
    from _mienodes_internal.core.utils import (
        image_tensor_batch_to_data_urls,
        mie_log,
    )
except ImportError:
    try:
        from ...core.utils import (
            image_tensor_batch_to_data_urls,
            mie_log,
        )
    except ImportError:
        from core.utils import (
            image_tensor_batch_to_data_urls,
            mie_log,
        )

try:
    from _mienodes_internal.nodes.llm.ltx25_prompts import (
        MODE_CODES,
        MODES,
        append_multishot_directive,
        build_i2v_user_text,
        build_t2v_user_text,
        load_system_prompt,
        parse_mode,
    )
except ImportError:
    from .ltx25_prompts import (
        MODE_CODES,
        MODES,
        append_multishot_directive,
        build_i2v_user_text,
        build_t2v_user_text,
        load_system_prompt,
        parse_mode,
    )


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Token budget.
# NOTE: reasoning models (MiniMax-M3, DeepSeek-R1, GLM-5.x) emit their
# chain-of-thought INSIDE the token budget before the final answer; if
# the think chain exhausts the budget the model returns empty content
# (the #1 cause of "returned empty after Xs" in this family). Upstream
# gemma4 enhances with max_new_tokens 600 on a local encoder -- no
# think chain -- so this budget only raises the ceiling for external
# reasoning LLMs and costs nothing when unused.
_DEFAULT_MAX_TOKENS = 8192
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Default sampling knobs. ``temperature=0.8`` matches the LTX2 sibling
# node (creative expansion). Upstream gemma4 enhancement runs greedy
# (do_sample=False) for deterministic captions -- set 0.0 to reproduce
# that behavior on a capable connector model.
_DEFAULT_TEMPERATURE = 0.8

# Default per-call timeout. A single LLM call at 120 s covers the
# typical 10-60 s vision-model response window with headroom.
_DEFAULT_TIMEOUT = 120

# Strip a leading ``<think>...</think>`` block that reasoning models
# may emit before the visible answer. Both official system prompts
# demand "Output ONLY the caption text -- no JSON, no preamble", so any
# leaked thinking wrapper must be removed before the node returns.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def postprocess_caption(raw_text: str) -> str:
    """Strip a leading ``<think>...</think>`` block (if any) and return
    the visible paragraph. Whitespace-only inputs collapse to ``""``."""
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = _THINK_BLOCK_RE.sub("", text, count=1).strip()
    return text


# --------------------------------------------------------------------------- #
# Default idea synthesis (used when the user leaves user_prompt empty)
# --------------------------------------------------------------------------- #
def _default_idea(mode_code: str) -> str:
    """Synthesize a reasonable default idea when ``user_prompt`` is
    blank, so the node still runs the full pipeline instead of
    returning an empty string (mirrors the h3 ``_default_idea`` shape).
    """
    if mode_code == "i2v":
        return (
            "Bring this first frame to life with natural, evolving "
            "motion: continue the subject's action and develop the "
            "scene over time."
        )
    return (
        "Create a visually striking short scene with a clear subject, "
        "a distinct environment, and evolving motion over the duration."
    )


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class LTX25PromptEnhancer:
    """LTX-2.5 caption enhancer that talks to the project's
    LLMServiceConnector.

    Single-stage for both modes; mirrors the structure of
    ``Scail2PromptEnhancer`` / ``H3PromptEnhancer`` so the LLM nodes
    share the project's logging + timeout patterns.
    """

    def __init__(
        self,
        llm_service_connector: Any,
        *,
        image_detail: str = "auto",
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: Optional[int] = None,
    ):
        self.llm = llm_service_connector
        self.image_detail = image_detail
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Per-call timeout override; None means leave the connector's
        # own timeout in place. Saved and restored around invoke() so
        # the connector object is safe to share with other nodes.
        self._timeout_override = int(timeout) if timeout else None

    # ------------------------------------------------------------------ #
    # Internal: one LLM call with timeout / log plumbing
    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        messages: list[dict],
        *,
        temperature: float,
        seed: Optional[int],
        stage: str,
    ) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out:
                mie_log(
                    f"LTX25 {stage}: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"LTX25 {stage}: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
            )
            return out.strip()
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_text: str,
        image_urls: list[str],
        image_detail: str,
    ) -> list[dict]:
        """Build a ``[system, user[...]]`` pair in OpenAI-chat format.

        Mirrors ``scail2_prompt_generator._build_messages``: each image
        gets a small ``[Image N]:`` caption preceding the image part so
        vision models that read the textual layout of the conversation
        know which image is which.
        """
        parts: list[dict] = []
        for i, url in enumerate(image_urls or []):
            parts.append({"type": "text", "text": f"\n[Image {i}]:"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": image_detail},
                }
            )
        if user_text:
            parts.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parts},
        ]

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        mode: str,
        user_prompt: str,
        *,
        image: Any = None,
        seed: Optional[int] = None,
        multishot: bool = False,
    ) -> str:
        """Run the LTX-2.5 pipeline and return the final caption.

        Returns the original ``user_prompt`` unchanged on missing
        reference media for i2v, so a bad input never breaks the
        ComfyUI workflow (scail2/h3 pattern).

        Pipeline shape by mode:
          * ``t2v`` -- single stage, no media required.
          * ``i2v`` -- needs ``image``; takes the FIRST tensor of the
                       batch (the official i2v reference is exactly one
                       first frame) and attaches it to the single
                       multimodal enhance call, as upstream
                       ``enhance_i2v`` does.
        """
        mode_code = parse_mode(mode)
        if mode_code not in MODE_CODES:
            mie_log(
                f"LTX25: unknown mode {mode!r} (parsed code={mode_code!r}); "
                f"falling back to t2v"
            )
            mode_code = "t2v"

        prompt = (user_prompt or "").strip()
        if not prompt:
            prompt = _default_idea(mode_code)
            mie_log(
                f"LTX25: empty user_prompt; using default idea for mode={mode_code}: "
                f"{prompt[:80]!r}"
            )

        system = load_system_prompt(mode_code)

        if mode_code == "t2v":
            mie_log(
                f"LTX25: mode=t2v temperature={self.temperature} "
                f"(single stage, no media)"
            )
            user_text = build_t2v_user_text(prompt)
            if multishot:
                user_text = append_multishot_directive(user_text, mode_code)
            messages = self._build_messages(
                system, user_text, [], self.image_detail
            )
            return postprocess_caption(
                self._invoke(
                    messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage="enhance[t2v]",
                )
            )

        # ---- i2v: single multimodal call ---------------------------- #
        image_urls = image_tensor_batch_to_data_urls(image)
        if not image_urls:
            mie_log("LTX25: i2v requires image (first frame); returning original")
            return user_prompt or ""
        if len(image_urls) > 1:
            mie_log(
                f"LTX25: i2v got a batch of {len(image_urls)} images; "
                "using the first as the reference first frame"
            )
        first_frame_url = image_urls[0]

        mie_log(
            f"LTX25: mode=i2v temperature={self.temperature} "
            f"(single stage, first frame attached)"
        )
        user_text = build_i2v_user_text(prompt)
        if multishot:
            user_text = append_multishot_directive(user_text, mode_code)
        messages = self._build_messages(
            system, user_text, [first_frame_url], self.image_detail
        )
        return postprocess_caption(
            self._invoke(
                messages,
                temperature=self.temperature,
                seed=seed,
                stage="enhance[i2v]",
            )
        )


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
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
        h = hashlib.md5()
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
