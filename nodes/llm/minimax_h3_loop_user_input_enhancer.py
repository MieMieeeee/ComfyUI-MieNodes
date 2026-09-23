"""MiniMax H3 Loop user-input preprocessor ComfyUI node.

Takes a rough user draft (any shape — a sentence, an outline, a few
lines) and rewrites it into the exact `user_input` format that the
``MiniMaxH3LoopPromptGenerator`` node expects. The rewrite is driven
by the bundled system prompt at
``nodes/llm/prompts/h3_loop/user_input_enhancer.txt`` which encodes
the node's hard contracts (speaker IDs, `<d>` verbatim span, reference
modes, per-shot schema) into four task-type branches.

Wire this node's ``user_input`` output straight into the H3 Loop
node's ``user_input`` widget — the connector carries through.

The preprocessor LLM is expected to reply with a structured block:

    Classification: <Dialogue | Action | Narration | Reference-driven>
    Notes for the user: <1-3 lines>
    --- BEGIN user_input ---
    <rewritten text>
    --- END user_input ---

We tolerate ``<think>...</think>`` wrappers (reasoning models) and any
prose around the block, but if the block is missing the node raises
``RuntimeError`` rather than silently feeding the user the raw LLM
reply — silent fallback would let a mis-classified rewrite flow into
H3 loop and waste the whole pipeline.
"""
from __future__ import annotations

import hashlib
import re
import time

try:
    from _mienodes_internal.nodes.llm.prompts.loader import (
        load_prompt_text,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        REFERENCE_MODES,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator import (
        LOOP_CATEGORIES,
        _PACING_LABELS,
    )
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    from .prompts.loader import load_prompt_text
    from .minimax_h3_loop_prompts import REFERENCE_MODES
    from .minimax_h3_loop_prompt_generator import LOOP_CATEGORIES, _PACING_LABELS
    from ...core.utils import mie_log


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Default sampling knobs. ``temperature=0.4`` matches the H3 loop node's
# own structured-output temperature (see
# ``minimax_h3_loop_prompt_generator.py:_DEFAULT_TEMPERATURE``): low
# enough to keep the four-task-type classification stable, high enough
# to allow surface variation in the rewrite.
_DEFAULT_TEMPERATURE = 0.4

# Token budget. The preprocessor reply itself is small (Classification
# + 1-3 line Notes + BEGIN/END block), but reasoning models count their
# thinking against max_tokens — 4096 let the chain-of-thought consume
# the whole budget before the answer started, producing HTTP-200
# replies with EMPTY content (live failure 2026-09-21: three
# consecutive response_chars=0 attempts, ~20-30s of thinking each).
# Same failure class as the dialogue extractor fixed in 5e8a5c2; the
# budget is a cap, not a target, so non-reasoning models pay nothing.
_MAX_TOKENS_DEFAULT = 16384
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Per-call timeout override (matches H3 loop node's dropdown).
_DEFAULT_TIMEOUT = 300

# Logical name for ``load_prompt_text`` — resolves to
# ``nodes/llm/prompts/h3_loop/user_input_enhancer.txt``.
_PROMPT_NAME = "h3_loop/user_input_enhancer"

# Tolerant block extractor. Anchored on the literal BEGIN/END tokens;
# uses ``re.DOTALL`` so the captured block may span newlines (a typical
# rewrite is 5-30 lines of dialogue / beats).
_USER_INPUT_BLOCK_RE = re.compile(
    r"--- BEGIN user_input ---\s*(.*?)\s*--- END user_input ---",
    re.DOTALL,
)
# Strip reasoning-model ``<think>...</think>`` wrappers so they don't
# sneak into the captured block content.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def extract_user_input_block(raw_reply: str) -> str | None:
    """Pull the ``--- BEGIN user_input ---`` ... ``--- END user_input ---``
    block out of the LLM reply. Returns the trimmed inner text, or
    ``None`` if the markers are missing.

    Tolerates a leading ``<think>...</think>`` wrapper (reasoning
    models) and any prose (Classification / Notes / chatter) around
    the block.
    """
    return split_enhancer_reply(raw_reply)[0]


def split_enhancer_reply(raw_reply: str) -> tuple[str | None, str]:
    """Split a well-formed enhancer reply into ``(user_input block,
    advice header)``.

    The header is everything before the BEGIN marker (Classification
    line + "Notes for the user", ``<think>`` blocks stripped). The
    Loop Plan Generator surfaces it in its preflight report when it
    runs the enhancer inline, so the rewrite stays inspectable after
    the fact even without the two-node checkpoint. Returns
    ``(None, "")`` when the markers are missing — same tolerance rules
    as ``extract_user_input_block``.
    """
    if not raw_reply:
        return None, ""
    text = _THINK_BLOCK_RE.sub("", raw_reply)
    idx = text.find("--- BEGIN user_input ---")
    if idx < 0:
        return None, ""
    header = text[:idx].strip()
    match = _USER_INPUT_BLOCK_RE.search(text)
    block = match.group(1).strip() if match is not None else None
    return block, header


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class _UserInputEnhancer:
    """One-shot LLM call that rewrites a rough draft into the H3 Loop
    node's ``user_input`` format.

    Thin wrapper around the connector with a per-call timeout override,
    mirroring the ``Krea2PromptEnhancer`` shape.
    """
    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _MAX_TOKENS_DEFAULT,
        timeout: int = _DEFAULT_TIMEOUT,
        usage_sink=None,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # ``None`` would mean "leave the connector's own timeout alone";
        # the node always wires a dropdown default, so this is always
        # set in practice.
        self._timeout_override = int(timeout) if timeout else None
        # Optional ``usage_sink(stage, messages, reply)`` callback the
        # Loop Plan Generator wires when it runs this enhancer inline,
        # so the rewrite shows up in the node's usage summary.
        self._usage_sink = usage_sink

    def _invoke(self, messages: list[dict], seed=None) -> str:
        # The override must apply even when the connector class has no
        # ``timeout`` attribute of its own (we create it for the call
        # and remove it afterwards, so the connector's own default
        # handling is untouched).
        override = self._timeout_override
        had_timeout = hasattr(self.llm, "timeout")
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if override is not None:
                self.llm.timeout = override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.perf_counter() - t0
            if self._usage_sink is not None:
                try:
                    self._usage_sink("user_input_enhance", messages, out or "")
                except Exception:
                    pass
            try:
                mie_log(
                    f"h3_loop_user_input_enhancer: invoke done "
                    f"({getattr(self.llm, 'model', '?')}) in {elapsed:.2f}s"
                )
            except Exception:
                pass
            return out or ""
        finally:
            if override is not None:
                if had_timeout:
                    self.llm.timeout = prev_timeout
                else:
                    try:
                        del self.llm.timeout
                    except AttributeError:
                        pass

    def __call__(
        self,
        draft: str,
        *,
        category: str,
        reference_mode: str,
        pacing: str = "",
        seed=None,
    ) -> str:
        sys_text = load_prompt_text(_PROMPT_NAME)
        context_lines = [
            f"  category: {category}",
            f"  reference_mode: {reference_mode}",
        ]
        if (pacing or "").strip():
            context_lines.append(f"  pacing: {pacing}")
        user_msg = (
            "---BEGIN DRAFT---\n"
            f"{draft or ''}"
            "\n---END DRAFT---\n\n"
            "Context for this rewrite (mirrors the H3 Loop Plan Generator "
            "node widgets so the preprocessor picks the right branch):\n"
            + "\n".join(context_lines)
            + "\n\n"
            "Reply with the standard shape:\n"
            "  Classification: <Dialogue|Action|Narration|Reference-driven>\n"
            "  Notes for the user: <1-3 short lines>\n"
            "  --- BEGIN user_input ---\n"
            "  <rewritten text>\n"
            "  --- END user_input ---"
        )
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_msg},
        ]
        return self._invoke(messages, seed=seed)


def run_enhancer(
    llm_service_connector,
    draft: str,
    *,
    category: str,
    reference_mode: str,
    pacing: str = "",
    seed=None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _MAX_TOKENS_DEFAULT,
    timeout: int = _DEFAULT_TIMEOUT,
    usage_sink=None,
    attempts: int = 3,
):
    """Run the rewrite with automatic retries on empty / block-less
    replies, then return ``(user_input_block, advice_header)``.

    Some providers occasionally answer HTTP 200 with an EMPTY content
    string (observed with MiniMax-M3: ``response_chars=0`` after ~16s).
    The connector only retries transport/HTTP errors, so without this
    wrapper a single empty reply kills the whole run. Retry policy: a
    reply that is empty or has no BEGIN/END block is retried with a
    fresh seed (same seed could deterministically reproduce the same
    empty answer); a genuine refusal that keeps its shape across all
    attempts still raises ``RuntimeError`` with the last reply head.
    """
    enhancer = _UserInputEnhancer(
        llm_service_connector,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        usage_sink=usage_sink,
    )
    last_head = ""
    for attempt in range(1, max(1, attempts) + 1):
        attempt_seed = seed if attempt == 1 else (
            None if seed is None else int(seed) + attempt - 1
        )
        raw = enhancer(
            draft,
            category=category,
            reference_mode=reference_mode,
            pacing=pacing,
            seed=attempt_seed,
        )
        block, header = split_enhancer_reply(raw)
        if block:
            if attempt > 1:
                mie_log(
                    "h3_loop_user_input_enhancer: succeeded on attempt "
                    f"{attempt}/{attempts}"
                )
            return block, header
        last_head = (raw or "")[:400]
        if attempt < attempts:
            mie_log(
                "h3_loop_user_input_enhancer: attempt "
                f"{attempt}/{attempts} reply had no user_input block "
                f"({len(raw or '')} chars); retrying with a fresh seed"
            )
    raise RuntimeError(
        "MiniMax H3 Loop user-input enhancer: the LLM reply did not "
        "contain a `--- BEGIN user_input ---` ... "
        f"`--- END user_input ---` block after {attempts} attempts. "
        f"Raw reply head: {last_head!r}"
    )


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class MiniMaxH3LoopUserInputEnhancer:
    """ComfyUI node: rough draft -> node-ready ``user_input`` STRING.

    Sits next to ``MiniMaxH3LoopPromptGenerator`` in the same Prompt
    Generator category. Wire its ``user_input`` output socket straight
    into the loop node's ``user_input`` widget — the names match so
    no rewiring is needed.

    Raises ``RuntimeError`` if the LLM reply does not contain a
    ``--- BEGIN user_input ---`` ... ``--- END user_input ---`` block;
    the error includes the raw reply head so the user can diagnose.
    Silent fallback would let a mis-classified reply flow into H3 loop
    and waste the whole pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "draft": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Your rough idea in any shape — a sentence, "
                            "an outline, a few lines, a free paragraph. "
                            "The preprocessor rewrites it into the format "
                            "the H3 Loop Plan Generator node consumes."
                        ),
                    },
                ),
                "category": (
                    list(LOOP_CATEGORIES),
                    {
                        "default": LOOP_CATEGORIES[0],
                        "tooltip": (
                            "Mirrors the H3 Loop node's category widget. "
                            "Sets the cinematography advice the preprocessor "
                            "will respect (none / dialogue / action)."
                        ),
                    },
                ),
                "reference_mode": (
                    list(REFERENCE_MODES),
                    {
                        "default": REFERENCE_MODES[0],
                        "tooltip": (
                            "Mirrors the H3 Loop node's reference_mode widget. "
                            "CRITICAL: if you have reference images, set this "
                            "to fl2va or ref2va here AND on the loop node — "
                            "t2va ignores the images socket."
                        ),
                    },
                ),
                "pacing": (
                    list(_PACING_LABELS),
                    {
                        "default": _PACING_LABELS[1],
                        "tooltip": (
                            "Mirrors the H3 Loop node's pacing widget. The "
                            "rewrite matches this tempo: fast -> more, "
                            "shorter beats/dialogue lines with quick "
                            "back-and-forth and dense chained action; "
                            "normal -> default shaping; slow -> fewer, "
                            "longer beats, calm unhurried action."
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
                        "tooltip": (
                            "Seed forwarded to the LLM call. 0 lets the "
                            "connector pick a fresh seed."
                        ),
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _MAX_TOKENS_DEFAULT,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                    },
                ),
                "timeout": (
                    [60, 120, 300, 600],
                    {"default": _DEFAULT_TIMEOUT},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("user_input",)
    FUNCTION = "enhance"
    CATEGORY = MY_CATEGORY

    def enhance(
        self,
        llm_service_connector,
        draft,
        category,
        reference_mode,
        pacing=_PACING_LABELS[1],
        seed=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        block, _header = run_enhancer(
            llm_service_connector,
            draft,
            category=category,
            reference_mode=reference_mode,
            pacing=pacing,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return (block,)

    def is_changed(
        self,
        llm_service_connector,
        draft,
        category,
        reference_mode,
        pacing=_PACING_LABELS[1],
        seed=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            draft or "",
            category or "",
            reference_mode or "",
            pacing or "",
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update(part.encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
