"""MiniMax H3 Loop user-input preprocessor ComfyUI node.

Takes a rough user draft (any shape — a sentence, an outline, a few
lines) and rewrites it into the exact `user_input` format that the
``MiniMaxH3LoopPromptGenerator`` node expects. A draft the downstream
parser accepts as a board, or a draft that is a single spoken line,
loads ``user_input_enhancer_lock.txt`` (those lines stay byte for
byte). A free-form draft, including prose that only contains a colon
label such as ``要求：…``, loads ``user_input_enhancer_compose.txt``.
Either way the selected ``reference_mode`` appends its own contract
from ``user_input_enhancer_modes.txt`` — t2va, i2va, fl2va, and ref2va
are different downstream implementations.

Wire this node's ``user_input`` output straight into the H3 Loop
node's ``user_input`` widget — the connector carries through. The
second ``advice`` output carries the Classification, the LLM's own
notes, this module's mechanical findings, and a locally computed
cut-budget suggestion; it is optional, so existing single-socket
workflows keep working unchanged. ComfyUI does not draw this node's
``ui.text`` payload (``js/showAnything.js`` only renders
``ShowAnything|Mie``). Read the advice from the second socket.

The preprocessor LLM is expected to reply with a structured block:

    Classification: <Dialogue | Action | Narration | Reference-driven>
    Notes for the user: <1-3 lines>
    --- BEGIN user_input ---
    <rewritten text>
    --- END user_input ---

We tolerate ``<think>...</think>`` wrappers (reasoning models) and any
prose around the block, but if the block is missing on every attempt
the node raises ``RuntimeError`` rather than silently feeding the user
the raw LLM reply — silent fallback would let a mis-classified rewrite
flow into H3 loop and waste the whole pipeline.

Content is verified AFTER the block is extracted: spoken lines are
contract data and must survive verbatim (see ``verify_block_content``).
A locked block that paraphrases or drops a line is pinned back on that
first reply, so no word is lost and the defect does not cost another
model call. An empty reply or a missing block is retried once.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

try:
    from _mienodes_internal.nodes.llm.prompts.loader import (
        load_prompt_text,
    )
    from _mienodes_internal.nodes.llm.dialogue_segmenter import (
        PACING_PRESETS as _DLG_PACING_PRESETS,
        distribute_lines_to_scenes as _dlg_distribute_lines_to_scenes,
        estimate_shot_budget as _dlg_estimate_shot_budget,
        min_total_seconds_for_scene_count as _dlg_min_total_seconds_for_scene_count,
        parse_structured_dialogue_turns as _dlg_parse_structured_dialogue_turns,
    )
    from _mienodes_internal.nodes.llm.dialogue_segmenter import (
        _NON_SPEAKER_PREFIXES as _DLG_NON_SPEAKER_PREFIXES,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        REFERENCE_MODES,
        parse_reference_mode,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator import (
        LOOP_CATEGORIES,
        _PACING_LABELS,
    )
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    from .prompts.loader import load_prompt_text
    from .dialogue_segmenter import (
        PACING_PRESETS as _DLG_PACING_PRESETS,
        distribute_lines_to_scenes as _dlg_distribute_lines_to_scenes,
        estimate_shot_budget as _dlg_estimate_shot_budget,
        min_total_seconds_for_scene_count as _dlg_min_total_seconds_for_scene_count,
        parse_structured_dialogue_turns as _dlg_parse_structured_dialogue_turns,
    )
    from .dialogue_segmenter import (
        _NON_SPEAKER_PREFIXES as _DLG_NON_SPEAKER_PREFIXES,
    )
    from .minimax_h3_loop_prompts import REFERENCE_MODES, parse_reference_mode
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

# Lock: drafts that already contain speaker：line utterances.
# Compose: free-form drafts. Modes: one contract per reference_mode.
# ``user_input_enhancer.txt`` is only a pointer for humans.
_PROMPT_LOCK = "h3_loop/user_input_enhancer_lock"
_PROMPT_COMPOSE = "h3_loop/user_input_enhancer_compose"
_PROMPT_MODES = "h3_loop/user_input_enhancer_modes"
_PROMPT_NAME = _PROMPT_LOCK

# Structured drafts are a contract, not a creative pass. Temperature 0
# keeps the model from paraphrasing lines it has been told to copy.
_LOCK_TEMPERATURE = 0.0

# Empty / markerless replies get one fresh-seed retry. A bad block on
# a locked draft is pinned immediately and does not spend this budget.
_TRANSPORT_ATTEMPTS = 2

_MODE_HEADING_RE = re.compile(r"^## mode:([A-Za-z0-9]+)\s*$", re.MULTILINE)

# Process cache. Tests point ``_CACHE_ROOT_OVERRIDE`` at a temp dir so
# a previous run's disk entries cannot satisfy a later test.
_MEM_CACHE: dict[str, tuple[str, str]] = {}
_CACHE_ROOT_OVERRIDE: str | None = None

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
# Content contract — the LLM proposes, this module disposes
# --------------------------------------------------------------------------- #
# The enhancer is a REWRITER, so by default it is free to smooth the
# user's prose. Spoken lines are the one exception: the downstream
# ``<d>[Language]...</d>`` blocks are word-perfect contract data, and a
# paraphrase silently changes what the actor says (live 2026-10-05:
# 「老板，你这里剩了那么多，怎么办？」came back as 「老板，你今天剩了那么多，
# 要怎么处理？」and 「明天怎么办？」as 「那明天还做这么多吗？」).
#
# The loop node's own ``concept[start:end] == text`` span check cannot
# catch this — it validates the ALREADY-REWRITTEN text against itself,
# so a clean paraphrase passes. The only place the original words still
# exist is here, before the rewrite.
#
# Policy: PIN the original lines on the first bad block. Never raise
# OVER A WORD, and do not spend another model call on a paraphrase the
# same prompt just produced. Failing the node would throw away the
# setting paragraph and leave the user with nothing to run. A reply
# that never produced a block at all still raises — there is nothing
# to salvage. Empty replies get one fresh-seed retry.
#
# Free-form prose drafts have no locatable source lines, so the check
# is skipped for them — narration is allowed to be rewritten, because
# there is no verbatim contract on it. A colon line sitting inside
# that prose (``要求：…``, ``Goal: …``) is not a source line either:
# see ``draft_is_locked``.


def collect_draft_speech(draft: str) -> list[tuple[str, str]]:
    """Every ``(speaker, line)`` utterance in the draft, in order.

    Deliberately NOT ``parse_structured_dialogue_turns``. That function
    applies two confidence rules that are correct for deciding whether to
    take the downstream fast path but wrong for deciding what is
    protected contract data:

    * it needs **two or more** matching lines, so a one-line draft
      returned nothing and its single utterance shipped unprotected;
    * it rejects the WHOLE draft once a non-dialogue line appears after
      the dialogue starts, so every real line in such a draft became
      unprotected too.

    Both are routing decisions, and this scanner is not the route.
    ``draft_is_locked`` decides lock versus compose. This scanner only
    lists candidate utterances, with no minimum and no early rejection,
    so a camera line between two real turns does not hide the second
    turn once the draft has been accepted as locked.
    """
    if not draft:
        return []
    pairs: list[tuple[str, str]] = []
    for raw in draft.split("\n"):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = _structured_line_match(stripped)
        if parsed:
            pairs.append(parsed)
    return pairs


def draft_dialogue_turns(draft: str) -> list[tuple[str, str]]:
    """Alias of :func:`collect_draft_speech`, kept for the pin path."""
    return collect_draft_speech(draft)


def draft_dialogue_lines(draft: str) -> list[str]:
    """The draft's own spoken lines (see :func:`collect_draft_speech`)."""
    return [line for _speaker, line in collect_draft_speech(draft)]


def _block_speech(block: str) -> set[tuple[str, str]]:
    """``(speaker, line)`` pairs the block actually contains as speech."""
    pairs: set[tuple[str, str]] = set()
    for raw in (block or "").split("\n"):
        parsed = _structured_line_match(raw.strip())
        if parsed:
            pairs.add(parsed)
    return pairs


def _speaker_aliases(speaker: str) -> set[str]:
    """Names the same character may legitimately go by.

    Normalising the roster is one of this node's jobs: a draft that says
    ``图3的小猫`` may legitimately be rewritten to ``小猫`` so the
    speaker map has one stable key. Requiring the exact draft spelling
    would reject that correct work, so both the full form and the
    picture-prefix-stripped form count as the same speaker.
    """
    name = (speaker or "").strip()
    if not name:
        return set()
    aliases = {name}
    stripped = re.sub(
        r"^(参考图|图片|图|reference\s*picture|picture|image|photo)\s*\d*"
        r"\s*(的)?",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    if stripped:
        aliases.add(stripped)
    return aliases


def missing_dialogue_lines(
    block: str,
    source_turns: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Source utterances the block does not carry, speaker and all.

    Matching is on the ``(speaker, line)`` pair, not a substring test.
    A substring test passes when a short utterance is swallowed by a
    longer neighbour (``那明天怎么办？`` contains ``明天怎么办？``) and
    passes when the right words are handed to the wrong character — both
    of which change what the finished video says.

    The speaker side compares under :func:`_speaker_aliases`, because
    normalising ``图3的小猫`` to ``小猫`` is this node's job, not a
    contract violation. The WORDS, however, must match exactly: that is
    the part the actor says out loud.
    """
    if not source_turns:
        return []
    present = _block_speech(block)
    missing: list[tuple[str, str]] = []
    for speaker, line in source_turns:
        aliases = _speaker_aliases(speaker)
        if any((alias, line) in present for alias in aliases):
            continue
        missing.append((speaker, line))
    return missing


def extra_dialogue_lines(
    block: str,
    source_turns: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Speaker lines in ``block`` that the draft did not contain.

    The lock prompt forbids inventing a spoken line. This is the
    opposite direction of :func:`missing_dialogue_lines` and is a
    notice, not a pin: the source lines are already intact.
    """
    if not source_turns:
        return []
    extras: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in (block or "").split("\n"):
        parsed = _structured_line_match(raw.strip())
        if not parsed or parsed in seen:
            continue
        seen.add(parsed)
        speaker, line = parsed
        aliases = _speaker_aliases(speaker)
        if any(
            src_line == line and aliases & _speaker_aliases(src_speaker)
            for src_speaker, src_line in source_turns
        ):
            continue
        extras.append(parsed)
    return extras


def _is_prologue_line(stripped: str) -> bool:
    """A non-empty line that is production direction, not speech.

    Defined as the complement of :func:`_structured_line_match` so the
    two can never disagree about where the dialogue region starts. That
    automatically covers free prose (a setting paragraph), the
    production-directive blacklist (``镜头``, ``camera``, ``场景设定``,
    …) and the picture-binding labels (``参考图1：黑猫``) the blacklist
    does not list — losing one of those silently drops the
    character-to-picture assignment.
    """
    return bool(stripped) and _structured_line_match(stripped) is None


def pin_dialogue_lines(block: str, source_turns: list[tuple[str, str]]) -> str:
    """Rebuild the block's dialogue region from the draft's own turns.

    Deliberately positional rather than fuzzy: once the retries are
    exhausted we no longer trust the rewrite's line pairing (it may have
    merged, split, re-ordered or reworded lines), and a wrong fuzzy match
    would corrupt the board — worse than one imperfect line.

    The rewrite's non-dialogue material is the part worth keeping, so we
    keep EVERY prose line it produced — not just the ones above the first
    spoken line. A ``镜头：`` line or a picture binding that the model
    placed inside the dialogue region is exactly the material this change
    exists to protect, and dropping it would hand back a board whose
    words are right but whose staging and image bindings are gone.

    The result is what the downstream parser expects: all prose first,
    then ``speaker：line`` and nothing else.
    """
    if not source_turns:
        return block

    # Region split uses the DOWNSTREAM parser's own notion of a speaker
    # line, not this module's stricter one. The pin's output has to be
    # parseable by the very parser that will consume it, so hoisting a
    # line the downstream call would still read as speech would just
    # relocate a phantom speaker rather than remove it.
    prologue: list[str] = []
    seen_prologue: set[str] = set()
    for raw in (block or "").split("\n"):
        stripped = raw.strip()
        if not stripped or _downstream_speaker_line(stripped):
            continue
        if _line_restates_speech(stripped, source_turns):
            # A paraphrase written as prose would sit next to the
            # restored line and the board would say the sentence twice.
            continue
        if stripped in seen_prologue:
            continue
        seen_prologue.add(stripped)
        prologue.append(stripped)

    rebuilt = list(prologue)
    if prologue:
        rebuilt.append("")
    rebuilt.extend(f"{speaker}：{line}" for speaker, line in source_turns)
    return "\n".join(rebuilt)


def _downstream_speaker_line(stripped: str) -> bool:
    """Whether the DOWNSTREAM parser would read this line as speech.

    Exactly the blacklist it applies, nothing more.
    """
    if not stripped:
        return False
    match = re.match(
        r"^(?P<speaker>[^\s：:]{1,16})\s*[：:]\s*(?P<text>\S.*)$", stripped
    )
    if not match:
        return False
    speaker = match.group("speaker").strip()
    if speaker.lower() in _DLG_NON_SPEAKER_PREFIXES:
        return False
    return any("一" <= c <= "鿿" or c.isalpha() for c in speaker)


def _structured_line_match(stripped: str):
    """``(speaker, text)`` when the line is a canonical dialogue line.

    Mirrors the downstream parser's own speaker test, including its
    production-directive blacklist: ``镜头：正反打`` has the same
    ``X：Y`` shape as ``黑猫：你好`` but is a camera instruction, and
    treating it as the first utterance would drop the entire prologue
    (setting paragraph, picture bindings, camera direction) when the
    block is rebuilt.

    Also rejects picture-binding labels (``参考图1：黑猫``,
    ``Picture 2：…``) — but ONLY for the pin's own region split, and
    only because the downstream parser has the same hole: it will read
    ``参考图1：黑猫`` as a character named 参考图1 uttering 「黑猫」. The
    pin must not hoist such a line into the prologue, because hoisting
    does not fix that — it just relocates the phantom speaker to the
    first line, where it is even more damaging (it becomes the S1 that
    every later speaker ID counts from).
    """
    if not stripped:
        return None
    match = re.match(
        r"^(?P<speaker>[^\s：:]{1,16})\s*[：:]\s*(?P<text>\S.*)$", stripped
    )
    if not match:
        return None
    speaker = match.group("speaker").strip()
    if speaker.lower() in _DLG_NON_SPEAKER_PREFIXES:
        return None
    if re.fullmatch(
        r"(参考图|图片|图|reference\s*picture|picture|image|photo)\s*\d*",
        speaker,
        re.IGNORECASE,
    ):
        return None
    if not any("一" <= c <= "鿿" or c.isalpha() for c in speaker):
        return None
    return speaker, match.group("text").rstrip()


def _is_production_directive_line(stripped: str) -> bool:
    """A ``镜头：…`` / ``camera: …`` line, not a character utterance."""
    if not stripped:
        return False
    match = re.match(
        r"^(?P<speaker>[^\s：:]{1,16})\s*[：:]\s*\S", stripped
    )
    if not match:
        return False
    return match.group("speaker").strip().lower() in _DLG_NON_SPEAKER_PREFIXES


def draft_is_locked(draft: str) -> bool:
    """Whether this draft takes the lock prompt, temperature 0, and pin.

    Lock when the downstream parser accepts the draft as a board (two
    or more speaker lines, prologue only, nothing after the dialogue),
    when two or more speaker lines are present even if a note or a
    ``镜头：`` line sits beside them, or when the draft's only
    non-directive line is a single spoken utterance.

    A single colon line surrounded by ordinary sentences does not
    lock. ``要求：视频里不要出现人类`` inside a prose draft is a label.
    Pinning it would guarantee a speaker named 要求. The same shape
    with no surrounding prose (``黑猫：今天天气真好。``) stays locked:
    one spoken line is contract data, and a word list is not a reliable
    way to tell the two apart.
    """
    if _dlg_parse_structured_dialogue_turns(draft or "")[0]:
        return True
    nonempty = [ln.strip() for ln in (draft or "").splitlines() if ln.strip()]
    if not nonempty:
        return False
    speech = 0
    free_prose = False
    for line in nonempty:
        if _structured_line_match(line):
            speech += 1
            continue
        if _is_production_directive_line(line):
            continue
        free_prose = True
    if speech >= 2:
        return True
    return speech == 1 and not free_prose


def lock_source_turns(draft: str) -> list[tuple[str, str]]:
    """Utterances the lock path must keep, or ``[]`` on the compose path."""
    if not draft_is_locked(draft):
        return []
    return collect_draft_speech(draft)


def _common_subsequence_len(left: str, right: str) -> int:
    """Length of the longest common subsequence. Lines are short."""
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for char in left:
        cur = [0] * (len(right) + 1)
        for index, other in enumerate(right, start=1):
            if char == other:
                cur[index] = prev[index - 1] + 1
            elif cur[index - 1] >= prev[index]:
                cur[index] = cur[index - 1]
            else:
                cur[index] = prev[index]
        prev = cur
    return prev[-1]


def _line_restates_speech(line: str, source_turns: list[tuple[str, str]]) -> bool:
    """True when a prose line is a rewritten copy of a locked utterance.

    Exact containment counts. So does a subsequence that covers at least
    half of the spoken line and at least six characters — enough to catch
    「老板，你今天剩了那么多，要怎么处理？」standing in for
    「老板，你这里剩了那么多，怎么办？」, and not enough to drop a setting
    sentence that merely shares a name.
    """
    text = (line or "").strip()
    if not text:
        return False
    for _speaker, spoken in source_turns:
        spoken = (spoken or "").strip()
        if len(spoken) < 2:
            continue
        if spoken in text:
            return True
        common = _common_subsequence_len(text, spoken)
        if common >= 6 and common / len(spoken) >= 0.5:
            return True
    return False


def verify_block_content(
    block: str,
    draft: str,
    classification: str,
) -> tuple[list[str], list[str]]:
    """Post-rewrite content audit.

    Returns ``(errors, notices)``. ``errors`` are contract violations
    that should trigger a corrective retry; ``notices`` are advisory
    lines for the user's advice output and never fail the run.

    Two independent checks:

    1. **Verbatim lines** — every utterance on the lock path must survive
       with its exact speaker AND words. Compose drafts, including a
       colon label inside prose, state no protected utterances. A
       one-line spoken draft is still protected: the parser's two-line
       floor is a routing rule, not a reason to drop a single line.
    2. **Board shape** — when the draft is a real multi-line dialogue
       board, the block must ALSO parse via the same deterministic parser
       the loop node uses. This catches the failure mode where a stray
       ``镜头：`` line lands inside the dialogue block and sends the whole
       board back to the expensive LLM extraction path. Narration and
       action boards are exempt: they are not supposed to parse as
       dialogue.
    """
    errors: list[str] = []
    notices: list[str] = []

    is_dialogue = (classification or "").strip().lower().startswith("dialogue")
    source_turns = lock_source_turns(draft)
    draft_is_board = bool(_dlg_parse_structured_dialogue_turns(draft or "")[0])

    missing = missing_dialogue_lines(block, source_turns)
    if missing:
        preview = "; ".join(f"{spk}：{ln[:24]}" for spk, ln in missing[:3])
        errors.append(
            f"{len(missing)} spoken line(s) were not preserved verbatim "
            f"(speaker and words must both match): {preview}"
        )
    elif source_turns:
        extras = extra_dialogue_lines(block, source_turns)
        if extras:
            preview = "; ".join(f"{spk}：{ln[:24]}" for spk, ln in extras[:3])
            notices.append(
                f"Notice: {len(extras)} spoken line(s) are not in the draft "
                f"(a locked rewrite must not invent lines): {preview}"
            )

    if draft_is_board:
        parsed, _prologue = _dlg_parse_structured_dialogue_turns(block or "")
        if not parsed:
            errors.append(
                "the dialogue block does not parse as `speaker：line` "
                "lines — something other than dialogue was placed inside "
                "the dialogue block (a 镜头/camera line there breaks the "
                "downstream parser)"
            )
    elif is_dialogue and not source_turns:
        notices.append(
            "classified as Dialogue but no `speaker：line` utterance was "
            "found in the draft to preserve — check the classification"
        )

    return errors, notices


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
        prompt_text: str = "",
    ) -> str:
        sys_text = prompt_text or _assemble_system_prompt(_PROMPT_LOCK, reference_mode)
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


def _parse_classification(header: str) -> str:
    """Pull the ``Classification:`` code out of the advice header."""
    match = re.search(
        r"^\s*Classification\s*[:：]\s*(.+?)\s*$",
        header or "",
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def suggest_scene_count(
    block: str,
    pacing: str = "",
) -> list[str]:
    """Local, deterministic cut-budget advice for a dialogue block.

    Runs the SAME estimator the loop node uses (``estimate_shot_budget``
    + ``distribute_lines_to_scenes``), so the numbers here are the
    numbers the node would compute — no LLM guesswork. Emitted as
    advice text only: the loop node keeps ownership of the
    ``scene_count`` / ``total_duration_seconds`` widgets, and adding
    duplicates here would fight them (a value set on this standalone
    node has no effect on the loop node's cut decision).

    Returns ``[]`` for boards that are not parseable dialogue.
    """
    turns, _prologue = _dlg_parse_structured_dialogue_turns(block or "")
    if not turns:
        return []
    key = (pacing or "").split(" - ", 1)[0].strip().lower()
    preset = _DLG_PACING_PRESETS.get(key) or _DLG_PACING_PRESETS["normal"]
    try:
        budgets, report = _dlg_estimate_shot_budget(turns, preset)
        fine, _fine_report = _dlg_estimate_shot_budget(
            turns, preset, max_shot_seconds=0.0
        )
    except Exception as exc:  # never let advice break the rewrite
        mie_log(
            f"h3_loop_user_input_enhancer: scene-count advice skipped "
            f"({exc!r})"
        )
        return []

    line_total = len(fine)
    # The longest single LINE's own duration is the floor any scene count
    # must respect: a line can never be split across scenes, so a
    # per-scene share below it truncates the line.
    #
    # Scene weight sums the FINE (per-line) budgets directly. Looking the
    # duration up from the coarse per-turn budgets instead would count a
    # same-speaker turn once PER LINE — a two-line turn's 7.3s would be
    # added twice inside one scene and roughly double the suggested
    # total.
    longest = max((b.duration_sec for b in fine), default=0.0)
    notes = [
        f"Board: {len(turns)} turn(s), {line_total} spoken line(s), "
        f"≈{report.total_sec:.0f}s of speech at {key or 'normal'} pacing."
    ]
    for candidate in (5, 4, 3):
        if candidate > line_total:
            continue
        try:
            min_total = _dlg_min_total_seconds_for_scene_count(fine, candidate)
        except Exception:
            continue
        if min_total <= 0:
            continue
        notes.append(
            f"Suggest scene_count={candidate} with "
            f"total_duration_seconds≈{min_total} "
            f"(≈{min_total / candidate:.1f}s per scene; the longest single "
            f"line needs {longest:.1f}s and cannot be split, so do not cut "
            f"finer than this)."
        )
        break
    if len(notes) == 1:
        notes.append(
            "Too few lines to suggest a cut count — this board will be "
            "planned automatically."
        )
    return notes


def clear_enhancer_cache() -> None:
    """Drop the in-process rewrite cache. Disk files are left in place."""
    _MEM_CACHE.clear()


def _enhancer_cache_root(override: str | None = None) -> str:
    """Same resolution order as the caption cache, different subdirectory."""
    if override:
        return override
    if _CACHE_ROOT_OVERRIDE:
        return _CACHE_ROOT_OVERRIDE
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.join(
                os.path.abspath(str(output_dir)),
                "mien_nodes",
                "h3_loop_enhancer_cache",
            )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "output", "mien_nodes", "h3_loop_enhancer_cache")


def _mode_contract(reference_mode: str) -> str:
    """The picture contract for this reference_mode, or ``""``."""
    code = parse_reference_mode(reference_mode)
    text = load_prompt_text(_PROMPT_MODES)
    parts = _MODE_HEADING_RE.split(text)
    # split keeps the heading text: [preamble, code, body, code, body, ...]
    mapping: dict[str, str] = {}
    body_iter = iter(parts[1:])
    for key, body in zip(body_iter, body_iter):
        mapping[key.strip()] = body.strip()
    return mapping.get(code, "")


def _assemble_system_prompt(prompt_name: str, reference_mode: str) -> str:
    base = load_prompt_text(prompt_name).rstrip()
    extra = _mode_contract(reference_mode)
    if not extra:
        return base
    return base + "\n\n" + extra


def _connector_cache_identity(connector) -> str:
    """Provider identity for the rewrite cache.

    The model name is not enough: a local OpenAI-compatible server and
    the official API can share a model id and must not share a cached
    rewrite. ``get_state`` already carries url, token, and model — the
    same string ``is_changed`` hashes.
    """
    get_state = getattr(connector, "get_state", None)
    if callable(get_state):
        try:
            state = get_state()
        except Exception:
            state = None
        if state:
            return str(state)
    return "\0".join((
        str(getattr(connector, "api_url", "") or ""),
        str(getattr(connector, "api_token", "") or ""),
        str(getattr(connector, "model", "") or ""),
    ))


def _cache_key(
    *,
    prompt_text: str,
    draft: str,
    category: str,
    reference_mode: str,
    pacing: str,
    model: str,
    provider: str,
    locked: bool,
    seed,
    temperature: float,
) -> str:
    hasher = hashlib.sha256()
    parts = [
        prompt_text,
        draft,
        category,
        reference_mode,
        pacing,
        model,
        provider,
        "lock" if locked else "compose",
    ]
    if not locked:
        # Free-form rewrites are allowed to vary with seed and temperature.
        # Locked rewrites are a contract: the same draft and mode reuse
        # the pinned or faithful result.
        parts.extend((str(seed), str(temperature)))
    for part in parts:
        hasher.update((part or "").encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _cache_get(key: str, cache_root: str | None) -> tuple[str, str] | None:
    hit = _MEM_CACHE.get(key)
    if hit is not None:
        return hit
    path = os.path.join(_enhancer_cache_root(cache_root), f"{key}.json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    block = data.get("block") if isinstance(data, dict) else None
    advice = data.get("advice") if isinstance(data, dict) else None
    if not isinstance(block, str) or not block or not isinstance(advice, str):
        return None
    _MEM_CACHE[key] = (block, advice)
    return _MEM_CACHE[key]


def _cache_put(key: str, result: tuple[str, str], cache_root: str | None) -> None:
    _MEM_CACHE[key] = result
    root = _enhancer_cache_root(cache_root)
    path = os.path.join(root, f"{key}.json")
    tmp = path + ".tmp"
    try:
        os.makedirs(root, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {"block": result[0], "advice": result[1]},
                handle,
                ensure_ascii=False,
            )
        os.replace(tmp, path)
    except OSError as exc:
        try:
            mie_log(f"h3_loop_user_input_enhancer: cache write failed ({exc!r})")
        except Exception:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass


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
    attempts: int = _TRANSPORT_ATTEMPTS,
    cache: bool = True,
    cache_root: str | None = None,
):
    """Rewrite ``draft`` and return ``(user_input_block, advice)``.

    Locked drafts use the lock prompt at temperature 0. A block that
    changes a line or parks a camera line inside the dialogue is pinned
    on that first reply. Empty replies and missing markers get one
    fresh-seed retry (``attempts``, default 2).

    Free-form drafts, including a colon label inside prose, use the
    compose prompt at the caller's temperature. They have no lines to
    pin, so a well-formed block is returned as-is.

    Successful results, including a pin, are cached. The key includes
    ``reference_mode`` and the connector's ``get_state()``. Locked
    results ignore seed and temperature, so changing the cut widgets
    on the planner does not pay for another rewrite of the same draft.
    Failures are not cached.
    """
    source_turns = lock_source_turns(draft)
    locked = bool(source_turns)
    prompt_name = _PROMPT_LOCK if locked else _PROMPT_COMPOSE
    used_temperature = _LOCK_TEMPERATURE if locked else float(temperature)
    prompt_text = _assemble_system_prompt(prompt_name, reference_mode)
    transport_budget = max(1, int(attempts))

    cache_key = None
    if cache:
        cache_key = _cache_key(
            prompt_text=prompt_text,
            draft=draft or "",
            category=category or "",
            reference_mode=reference_mode or "",
            pacing=pacing or "",
            model=str(getattr(llm_service_connector, "model", "") or ""),
            provider=_connector_cache_identity(llm_service_connector),
            locked=locked,
            seed=seed,
            temperature=used_temperature,
        )
        hit = _cache_get(cache_key, cache_root)
        if hit is not None:
            return hit

    enhancer = _UserInputEnhancer(
        llm_service_connector,
        temperature=used_temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        usage_sink=usage_sink,
    )
    last_head = ""
    best: tuple[str, str, list[str], list[str]] | None = None
    for attempt in range(1, transport_budget + 1):
        if attempt == 1:
            attempt_seed = seed
        else:
            attempt_seed = None if seed is None else int(seed) + attempt - 1
        raw = enhancer(
            draft,
            category=category,
            reference_mode=reference_mode,
            pacing=pacing,
            seed=attempt_seed,
            prompt_text=prompt_text,
        )
        block, header = split_enhancer_reply(raw)
        if not block:
            last_head = (raw or "")[:400]
            if attempt < transport_budget:
                mie_log(
                    f"h3_loop_user_input_enhancer: attempt {attempt}/{transport_budget} "
                    f"reply had no user_input block ({len(raw or '')} chars); "
                    f"retrying with a fresh seed"
                )
            continue

        classification = _parse_classification(header)
        errors, notices = verify_block_content(block, draft, classification)
        if not errors:
            if attempt > 1:
                mie_log(
                    f"h3_loop_user_input_enhancer: succeeded on attempt "
                    f"{attempt}/{transport_budget}"
                )
            result = (
                block,
                _build_advice(
                    header,
                    notices,
                    suggest_scene_count(block, pacing),
                    reference_mode,
                ),
            )
            if cache and cache_key:
                _cache_put(cache_key, result, cache_root)
            return result

        if not locked:
            result = (
                block,
                _build_advice(
                    header,
                    notices + errors,
                    suggest_scene_count(block, pacing),
                    reference_mode,
                ),
            )
            if cache and cache_key:
                _cache_put(cache_key, result, cache_root)
            return result

        best = (block, header, errors, notices)
        mie_log(
            f"h3_loop_user_input_enhancer: content verification failed "
            f"({errors[0][:160]!r}); pinning the draft's own lines"
        )
        break

    if best is None:
        raise RuntimeError(
            "MiniMax H3 Loop user-input enhancer: the LLM reply did not "
            "contain a `--- BEGIN user_input ---` ... "
            f"`--- END user_input ---` block after {transport_budget} attempts. "
            f"Raw reply head: {last_head!r}"
        )
    block, header, errors, notices = best
    pinned = pin_dialogue_lines(block, source_turns)
    still_missing = missing_dialogue_lines(pinned, source_turns)
    pinned_note = ""
    if still_missing:
        pinned_note = (
            f"WARNING: {len(still_missing)} spoken line(s) could not be "
            f"restored automatically — check the block against your draft "
            f"before running. Affected: "
            + "; ".join(f"{spk}：{ln[:24]}" for spk, ln in still_missing[:3])
        )
    elif pinned != block:
        pinned_note = (
            f"Pinned {len(source_turns)} spoken line(s) back to the user's "
            f"exact wording (speaker and words) after the rewrite altered "
            f"them."
        )
    mie_log(
        f"h3_loop_user_input_enhancer: pinned dialogue lines back "
        f"({len(still_missing)} still missing)"
    )
    result = (
        pinned,
        _build_advice(
            header,
            notices + ([pinned_note] if pinned_note else []),
            suggest_scene_count(pinned, pacing),
            reference_mode,
        ),
    )
    if cache and cache_key:
        _cache_put(cache_key, result, cache_root)
    return result


def _build_advice(
    header: str,
    notices: list[str],
    scene_notes: list[str],
    reference_mode: str = "",
) -> str:
    """Assemble the advice string surfaced by the node's second output.

    Carries the mode this rewrite used, the LLM's Classification / Notes,
    and this module's mechanical findings.
    """
    parts: list[str] = []
    code = parse_reference_mode(reference_mode) if reference_mode else ""
    if code:
        parts.append(f"reference_mode: {code}")
    if (header or "").strip():
        parts.append(header.strip())
    for note in notices:
        if note:
            parts.append(note)
    for note in scene_notes:
        if note:
            parts.append(note)
    return "\n".join(parts)


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
                            "Mirrors the H3 Loop node's pacing widget. "
                            "Existing speaker：line utterances are never "
                            "rewritten to hit a tempo; pacing only changes "
                            "how long the setting paragraph is. On a "
                            "free-form draft, fast packs more, shorter "
                            "beats and slow uses fewer, longer ones."
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_input", "advice")
    FUNCTION = "enhance"
    OUTPUT_NODE = True
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
        block, advice = run_enhancer(
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
        shown = advice or ""
        if block:
            shown = f"{shown}\n\n--- user_input ---\n{block}"
        return {"ui": {"text": shown}, "result": (block, advice)}

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
