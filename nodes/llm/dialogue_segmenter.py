# -*- coding: utf-8 -*-
"""Dialogue turn data model, span validation, and pacing-aware length
estimator for ``MiniMaxH3LoopPromptGenerator``.

The pipeline's semantic entry point is one LLM span-anchored extraction
call (see ``H3LoopPromptEnhancer.extract_dialogue``): the LLM returns
each utterance with its character span in the source concept, and this
module mechanically verifies ``concept[start:end]`` equals the reported
text — so three invariants hold by construction:

1. No dropped dialogue lines: every extracted line lands in exactly one
   ``<d>[Language]...</d>`` block in the output.
2. No splitting or paraphrasing of a line: each extracted line becomes
   exactly one verbatim ``<d>`` block (span anchoring makes anything
   else impossible to accept).
3. Same-speaker consecutive lines share one scene (turn grouping): the
   scene boundary only moves when the speaker changes.

This module is the single source of truth for those invariants; the rest
of the pipeline consumes its output rather than re-deriving turn /
budget structure from raw concept text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# H3 timing constants (self-contained: we don't lazy-pull from the
# prompts module because the segmenter must load in test environments
# where the prompts package isn't importable through its full name).
_math = math

_FPS = 24
_GRID_STEP = 17
_MIN_LENGTH_FRAMES = 5
_MAX_LENGTH_FRAMES = 3592  # 17*211+5; ~149.667 s


def _seconds_to_length(duration_seconds):
    """Round a duration request up onto the H3 17k+5 frame grid.

    Mirrors ``minimax_h3_loop_prompts.seconds_to_length``. The two
    implementations are kept in sync deliberately; if upstream changes
    the grid step, the estimator tests will fail loudly on the mismatch.
    """
    raw = _math.ceil(float(duration_seconds) * _FPS - 1e-9)
    k = max(0, _math.ceil((raw - _MIN_LENGTH_FRAMES) / _GRID_STEP - 1e-9))
    length = _MIN_LENGTH_FRAMES + k * _GRID_STEP
    if length > _MAX_LENGTH_FRAMES:
        length = _MAX_LENGTH_FRAMES
    return length


def _length_to_seconds(length):
    return round(int(length) / _FPS, 2)


# Public constants + helpers (re-exported for callers).
FPS = _FPS
GRID_STEP = _GRID_STEP
MIN_LENGTH_FRAMES = _MIN_LENGTH_FRAMES
MAX_LENGTH_FRAMES = _MAX_LENGTH_FRAMES
seconds_to_length = _seconds_to_length
length_to_seconds = _length_to_seconds


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DialogueTurn:
    """A contiguous block of dialogue lines by one speaker.

    Lines are kept verbatim (no whitespace normalization beyond strip) so
    the round-trip check against ``<d>[Language]...</d>`` output is
    exact.

    ``start`` and ``end`` are **character offsets into the original
    concept string**. They anchor the turn to the source: span
    ``concept[start:end]`` must equal ``" ".join(lines)`` (after
    stripping the surrounding quote / lead verb), so the LLM
    extractor cannot silently paraphrase, merge, or invent dialogue.
    """

    speaker: str
    lines: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0

    @property
    def line_count(self) -> int:
        return len(self.lines)


@dataclass
class Pacing:
    """A pacing preset: TTS rates for duration estimates plus the
    creative TEMPO the preset implies (target average scene length).

    ``target_scene_sec`` is the cut-density rule: for a given time
    budget, scene_count auto = total / target_scene_sec — fast pacing
    cuts more often (shorter scenes), slow pacing holds longer (fewer,
    longer scenes). It is a creative guideline, not a hard clamp.
    """

    name: str
    rate_cn: float           # Chinese characters per second
    rate_en: float           # English words per second
    pause_sec: float         # pause between consecutive lines in a turn
    head_tail_pad_sec: float  # pad at the start and end of the turn
    target_scene_sec: float = 7.0  # average scene length the tempo implies

    def per_line_seconds(self, line: str) -> float:
        """Estimate the spoken duration of one line."""
        chars = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
        if chars:
            return chars / self.rate_cn
        words = len(line.split())
        return max(words, 1) / self.rate_en


@dataclass
class ShotBudget:
    """Per-scene budget derived from one turn.

    A turn may split into multiple budgets when its raw runtime
    exceeds ``max_shot_seconds`` (per-line packing). In that case each
    sub-budget keeps the ``turn_index`` pointing back to the originating
    turn so downstream validators can re-attach the right dialogue
    line list.
    """

    shot_index: int
    turn_index: int
    speaker: str
    lines: list[str]
    line_count: int
    char_count: int
    estimated_speech_sec: float
    pause_total_sec: float
    head_tail_pad_sec: float
    raw_seconds: float
    rounded_length_frames: int
    duration_sec: float


@dataclass
class PacingReport:
    """Top-level summary the node exposes via preflight."""

    pacing: str
    total_sec: float
    total_frames: int
    shot_count: int
    overbudget_turns: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pacing presets
# --------------------------------------------------------------------------- #
PACING_PRESETS: dict[str, Pacing] = {
    "fast": Pacing(
        name="fast",
        rate_cn=4.5,
        rate_en=3.5,
        pause_sec=1.0,
        head_tail_pad_sec=0.3,
        target_scene_sec=4.5,
    ),
    "normal": Pacing(
        name="normal",
        rate_cn=3.5,
        rate_en=2.5,
        pause_sec=1.5,
        head_tail_pad_sec=0.3,
        target_scene_sec=7.0,
    ),
    "slow": Pacing(
        name="slow",
        rate_cn=2.8,
        rate_en=2.0,
        pause_sec=2.0,
        head_tail_pad_sec=0.4,
        target_scene_sec=12.0,
    ),
}


def scenes_for_duration(total_sec: float, pacing: Pacing) -> int:
    """The scene count a pacing preset implies for a time budget:
    ``round(total / target average scene length)``, at least one.

    This is the AUTO cut-density rule — fast pacing cuts more often for
    the same duration, slow pacing holds shots longer. An explicit
    scene_count always overrides it; time stays the only hard
    constraint."""
    if total_sec is None or total_sec <= 0:
        return 1
    target = float(getattr(pacing, "target_scene_sec", 0) or 0)
    if target <= 0:
        return 1
    return max(1, int(round(float(total_sec) / target)))


# --------------------------------------------------------------------------- #
# Span-anchored turn extraction (LLM-driven with mechanical verification)
# --------------------------------------------------------------------------- #
# The dialogue invariant chain is now:
#   concept (user text)
#     -> LLM returns {turns:[{speaker, lines:[{text, start, end}]}]}
#     -> node validates concept[start:end] == text  (impossible to lie)
#     -> node merges adjacent same-speaker turns + drops overlaps
#     -> estimator + per-shot LLM (unchanged downstream)
#
# The LLM does semantic judgement (which characters speak, where each
# line lives). The node does mechanical judgement (does each span
# literally match the source). That split keeps "1 line == 1 <d>
# block, verbatim" a guarantee rather than a probability.


@dataclass
class ExtractedLine:
    """One line of dialogue as the LLM extractor returned it.

    ``text`` is the spoken words (no leading verb like "said", no
    surrounding quotes); ``start``/``end`` are character offsets into
    the original ``concept`` string such that
    ``concept[start:end] == text``.
    """

    text: str
    start: int
    end: int


# Speakers that look like ``speaker：line`` but are production
# directives, not characters — a concept line like ``镜头：缓慢推进``
# must never become a spoken turn. ``场景设定`` is here because the
# upstream UserInputEnhancer's canonical rewrite OPENS with a
# ``场景设定：`` paragraph; without this entry the fast-path parser
# made the setting paragraph the first speaker's utterance (live
# failure 2026-09-22: the setting became an S1 off-screen voiceover).
_NON_SPEAKER_PREFIXES = {
    "镜头", "画面", "场景", "场景设定", "场景描述", "设定", "音乐", "字幕",
    "旁白", "画外音", "配音", "叙述", "备注", "注", "风格", "节奏", "时长",
    "相机", "摄像机", "视角", "背景", "环境",
    "camera", "scene", "setting", "shot", "music", "subtitle", "note",
    "style", "pacing", "narration", "narrator", "voiceover",
    "voice-over", "voice over", "pov", "description",
}

_STRUCTURED_LINE_RE = None  # compiled lazily below (module keeps no re dep)


def _structured_line_re():
    global _STRUCTURED_LINE_RE
    if _STRUCTURED_LINE_RE is None:
        import re as _re

        _STRUCTURED_LINE_RE = _re.compile(
            r"^(?P<speaker>[^\s：:]{1,16})\s*[：:]\s*(?P<text>\S.*)$"
        )
    return _STRUCTURED_LINE_RE


def parse_structured_dialogue_turns(
    concept: str,
) -> tuple[Optional[list[DialogueTurn]], str]:
    """Deterministically parse a canonical ``speaker：line`` concept.

    The enhancer (and the skill's canonical format) emits one
    ``莎莉猫：你好。`` line per utterance. When EVERY non-empty line
    after an optional narration prologue matches that shape, the LLM
    extractor adds nothing — this parser computes exact spans for free
    and the extraction call is skipped.

    Confidence guards (any miss returns ``(None, "")`` so the caller
    falls back to the LLM extractor):
      - at least 2 dialogue lines (a single colon line is too weak a
        signal — prose like ``注意：以下…`` must not become speech);
      - the speaker slot is 1-16 chars, no inner whitespace-only, and
        not a production directive (镜头/画面/音乐/…);
      - once the first dialogue line appears, no later non-empty line
        may be non-dialogue (mixed tails go to the LLM).

    Same-speaker consecutive lines merge into one turn (invariant 3).
    Returns ``(turns, prologue_text)``; ``prologue_text`` is the joined
    narration lines before the first dialogue line ("" when absent).
    """
    if not concept or not concept.strip():
        return None, ""
    pattern = _structured_line_re()
    matches: list[tuple[int, int, str, str]] = []  # (line_start, line_end, speaker, text)
    prologue_lines: list[str] = []
    in_dialogue = False
    offset = 0
    for raw_line in concept.split("\n"):
        stripped = raw_line.strip()
        next_offset = offset + len(raw_line) + 1
        if stripped:
            m = pattern.match(stripped)
            speaker = m.group("speaker").strip() if m else ""
            plausible = (
                bool(m)
                and speaker.lower() not in _NON_SPEAKER_PREFIXES
                and any(
                    "\u4e00" <= c <= "\u9fff" or c.isalpha()
                    for c in speaker
                )
                and m.group("text").strip()
            )
            if plausible:
                if not in_dialogue:
                    in_dialogue = True
                # Locate the spoken text's exact span inside the raw
                # line so concept[start:end] == text holds exactly.
                colon_pos = raw_line.find("：") if "：" in raw_line else raw_line.find(":")
                # Prefer the colon the regex matched on the stripped
                # line: same character in the raw line (strip only
                # removes surrounding whitespace).
                text_start_in_line = colon_pos + 1
                while (
                    text_start_in_line < len(raw_line)
                    and raw_line[text_start_in_line].isspace()
                ):
                    text_start_in_line += 1
                text = m.group("text").rstrip()
                text_end_in_line = text_start_in_line + len(text)
                matches.append(
                    (
                        offset + text_start_in_line,
                        offset + min(text_end_in_line, len(raw_line)),
                        speaker,
                        text,
                    )
                )
            elif in_dialogue:
                # Non-dialogue content after dialogue started -> not
                # confidently structured.
                return None, ""
            else:
                prologue_lines.append(stripped)
        offset = next_offset
    if len(matches) < 2:
        return None, ""
    turns: list[DialogueTurn] = []
    for start, end, speaker, text in matches:
        if turns and turns[-1].speaker == speaker and start >= turns[-1].end:
            turns[-1].lines.append(text)
            turns[-1].end = end
            continue
        turns.append(
            DialogueTurn(speaker=speaker, lines=[text], start=start, end=end)
        )
    if not turns:
        return None, ""
    prologue = "\n".join(prologue_lines).strip()
    return turns, prologue


def turns_from_extraction(
    raw_turns: list[dict],
) -> list[DialogueTurn]:
    """Convert the LLM's raw turn dicts into ``DialogueTurn`` objects.

    Each raw turn is ``{"speaker": str, "lines":
    [{"text": str, "start": int, "end": int}, ...]}``. We DO NOT
    validate the spans here — that's the caller's job
    (``validate_extraction``). We only coerce types.
    """
    out: list[DialogueTurn] = []
    for t in raw_turns:
        if not isinstance(t, dict):
            continue
        speaker = str(t.get("speaker") or "").strip()
        if not speaker:
            continue
        lines_raw = t.get("lines") or []
        line_objs: list[ExtractedLine] = []
        text_only: list[str] = []
        span_start = 0
        span_end = 0
        for li, entry in enumerate(lines_raw):
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            try:
                s = int(entry.get("start"))
                e = int(entry.get("end"))
            except (TypeError, ValueError):
                s, e = 0, 0
            if not text:
                continue
            line_objs.append(ExtractedLine(text=text, start=s, end=e))
            text_only.append(text)
            if li == 0:
                span_start = s
            span_end = e
        if not text_only:
            continue
        out.append(
            DialogueTurn(
                speaker=speaker,
                lines=text_only,
                start=span_start,
                end=span_end,
            )
        )
    return out


def validate_extraction(
    concept: str,
    turns: list[DialogueTurn],
) -> list[str]:
    """Turn-level substring check kept for older tests.

    Production extraction does **not** call this. ``DialogueTurn``
    only stores the spoken strings, not per-line spans, so this
    helper searches with ``str.find`` and can match the wrong
    occurrence of a repeated line. The live path is
    ``relocate_extracted_lines`` + ``validate_extracted_lines``
    (exact ``concept[start:end] == text`` per line).

    Returns a list of error strings (empty = pass). The checks:
      1. Every line appears as a substring of ``concept`` inside
         the turn's outer span (or anywhere, as a fallback).
      2. Spans are non-negative and in-bounds of ``len(concept)``.
      3. Spans are non-overlapping and ordered (start monotonically
         non-decreasing across turns).
    """
    errs: list[str] = []
    if concept is None:
        return ["concept is None"]
    L = len(concept)
    prev_end = 0
    for ti, turn in enumerate(turns):
        if not turn.lines:
            errs.append(f"turn {ti} ({turn.speaker!r}): empty lines")
            continue
        # Walk every line of this turn; spans should be a contiguous
        # non-overlapping run inside (turn.start, turn.end).
        prev_line_end = turn.start
        for li, line in enumerate(turn.lines):
            # Find this line's span by line index. We stored per-line
            # spans only on ExtractedLine; here we only have lines.
            # Re-derive spans from the turn's outer span by greedy
            # substring search — this is the "LLM pointed us to the
            # neighbourhood, we confirm the text matches" check.
            # We can't recover per-line offsets from DialogueTurn;
            # callers that want strict per-line validation should pass
            # the ExtractedLine list to validate_extracted_lines() below.
            # For backward compatibility (tests, estimator, etc.) we
            # still run the simple "turn span covers every line" check:
            pos = concept.find(line, prev_line_end, turn.end or L)
            if pos < 0:
                pos = concept.find(line, 0, L)
                if pos < 0:
                    errs.append(
                        f"turn {ti} line {li}: line not found anywhere "
                        f"in concept: {line!r}"
                    )
                    continue
            # span must fit inside the turn's outer window
            if turn.end and pos + len(line) > turn.end:
                errs.append(
                    f"turn {ti} line {li}: line extends past turn.end "
                    f"({turn.end} vs pos+len={pos + len(line)})"
                )
                continue
            prev_line_end = pos + len(line)
        # turn-level: outer span in-bounds + ordered
        if turn.end and turn.end > L:
            errs.append(
                f"turn {ti}: end {turn.end} exceeds concept length {L}"
            )
        if turn.start < 0:
            errs.append(f"turn {ti}: negative start {turn.start}")
        if turn.start > turn.end and turn.end > 0:
            errs.append(
                f"turn {ti}: start {turn.start} > end {turn.end}"
            )
        # cross-turn: starts are non-decreasing and non-overlapping
        if turn.start < prev_end:
            errs.append(
                f"turn {ti}: start {turn.start} overlaps previous "
                f"turn's end {prev_end}"
            )
        prev_end = max(prev_end, turn.end or 0)
    return errs


def relocate_extracted_lines(
    concept: str,
    raw_lines: list[ExtractedLine],
) -> tuple[list[ExtractedLine], list[str]]:
    """Rewrite spans that miss the source text when the spoken words
    occur exactly once in the unused remainder of ``concept``.

    The LLM extractor is asked for character offsets, but models
    commonly return byte offsets, off-by-one indices, or copy a
    wrong few-shot span. If ``concept[start:end]`` does not equal
    ``text`` yet ``text`` has exactly one unused occurrence, we
    accept that occurrence. Ambiguous (0 or 2+ unused hits) lines
    are left unchanged so ``validate_extracted_lines`` can fail them.

    Returns ``(relocated_lines, notes)``. Notes are human-readable
    relocation records for the preflight/summary, not errors.
    """
    if concept is None:
        return list(raw_lines or []), ["concept is None"]
    notes: list[str] = []
    occupied: list[tuple[int, int]] = []
    out: list[ExtractedLine] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < pe and end > ps for ps, pe in occupied)

    for i, line in enumerate(raw_lines or []):
        text = line.text
        s, e = line.start, line.end
        L = len(concept)
        in_bounds = 0 <= s <= e <= L
        slice_ok = in_bounds and concept[s:e].strip() == text.strip()
        if slice_ok and not _overlaps(s, e):
            occupied.append((s, e))
            out.append(line)
            continue
        hits: list[tuple[int, int]] = []
        cursor = 0
        while text:
            pos = concept.find(text, cursor)
            if pos < 0:
                break
            endp = pos + len(text)
            if not _overlaps(pos, endp):
                hits.append((pos, endp))
            cursor = pos + 1
        if len(hits) == 1:
            ns, ne = hits[0]
            notes.append(
                f"line {i}: relocated span [{s}, {e}) -> [{ns}, {ne}) "
                f"for {text!r}"
            )
            occupied.append((ns, ne))
            out.append(ExtractedLine(text=text, start=ns, end=ne))
        else:
            out.append(line)
            if slice_ok:
                occupied.append((s, e))
    return out, notes


def validate_extracted_lines(
    concept: str,
    raw_lines: list[ExtractedLine],
) -> list[str]:
    """Stricter variant: validates each ``ExtractedLine`` against
    ``concept[start:end] == text`` exactly. Use this when the LLM
    returned per-line spans and we want the strictest possible
    guarantee. Call ``relocate_extracted_lines`` first if the
    extractor is allowed to miss the offset but not the text.
    """
    errs: list[str] = []
    if concept is None:
        return ["concept is None"]
    L = len(concept)
    prev_end = 0
    for i, line in enumerate(raw_lines):
        s, e, text = line.start, line.end, line.text
        if s < 0 or e > L or s > e:
            errs.append(
                f"line {i}: span [{s}, {e}) invalid for concept "
                f"len {L}"
            )
            continue
        slice_text = concept[s:e]
        # Allow the LLM to anchor on whitespace; strip both ends.
        if slice_text.strip() != text.strip():
            errs.append(
                f"line {i}: concept[{s}:{e}] = {slice_text!r}, "
                f"expected {text!r}"
            )
            continue
        if s < prev_end:
            errs.append(
                f"line {i}: span [{s}, {e}) overlaps previous line "
                f"ending at {prev_end}"
            )
        prev_end = e
    return errs


def is_narrator_only(turns: list[DialogueTurn]) -> bool:
    """True if no turn has any text (extraction said 'pure narration')."""
    return not any(t.lines for t in turns)


def narrator_fallback_turn(concept: str) -> DialogueTurn:
    """Construct the (narrator) fallback turn when extraction found
    no dialogue. The whole concept becomes one beat."""
    return DialogueTurn(
        speaker="(narrator)",
        lines=[(concept or "").strip() or "(empty concept)"],
        start=0,
        end=len(concept or ""),
    )


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
def estimate_shot_budget(
    turns: list[DialogueTurn],
    pacing: Pacing,
    *,
    fps: int = FPS,
    max_shot_seconds: float = 14.0,
) -> tuple[list[ShotBudget], PacingReport]:
    """Produce a per-turn ``ShotBudget`` plus an overall ``PacingReport``.

    For each turn:
        speech_sec  = sum(pacing.per_line_seconds(line) for line in turn.lines)
        pause_sec   = (line_count - 1) * pacing.pause_sec     (none if 1 line)
        pad_sec     = 2 * pacing.head_tail_pad_sec
        raw_sec     = speech_sec + pause_sec + pad_sec

    If ``raw_sec > max_shot_seconds`` AND the turn carries more than one
    line, the turn is split per-line into consecutive sub-budgets (no
    line is ever cut mid-utterance). Single-line overbudget turns stay
    as a single budget but are flagged in ``overbudget_turns`` so the
    caller can surface a preflight warning.
    """
    secs_to_len, len_to_secs = seconds_to_length, length_to_seconds
    budgets: list[ShotBudget] = []
    overbudget: list[int] = []
    total_frames = 0
    shot_index = 0

    for turn_idx, turn in enumerate(turns):
        per_line_secs = [pacing.per_line_seconds(ln) for ln in turn.lines]
        turn_raw = (
            sum(per_line_secs)
            + max(0, turn.line_count - 1) * pacing.pause_sec
            + 2 * pacing.head_tail_pad_sec
        )

        if turn_raw <= max_shot_seconds or turn.line_count == 1:
            # One budget for the whole turn.
            length = _safe_secs_to_len(secs_to_len, turn_raw)
            char_count = _char_count_for(turn.lines)
            budgets.append(
                ShotBudget(
                    shot_index=shot_index,
                    turn_index=turn_idx,
                    speaker=turn.speaker,
                    lines=list(turn.lines),
                    line_count=turn.line_count,
                    char_count=char_count,
                    estimated_speech_sec=round(sum(per_line_secs), 3),
                    pause_total_sec=round(
                        max(0, turn.line_count - 1) * pacing.pause_sec, 3
                    ),
                    head_tail_pad_sec=2 * pacing.head_tail_pad_sec,
                    raw_seconds=round(turn_raw, 3),
                    rounded_length_frames=length,
                    duration_sec=len_to_secs(length),
                )
            )
            total_frames += length
            if turn_raw > max_shot_seconds:
                overbudget.append(shot_index)
            shot_index += 1
            continue

        # Multi-line overbudget turn: split per-line. Each sub-budget
        # holds >= 1 line; we never cut a single line across shots.
        # Pack consecutive lines greedily until the running raw_sec would
        # exceed max_shot_seconds, then start a new sub-budget.
        running_secs = 0.0
        bucket: list[str] = []
        bucket_secs: list[float] = []
        bucket_chars = 0

        def _flush() -> None:
            nonlocal running_secs, bucket, bucket_secs, bucket_chars
            nonlocal shot_index, total_frames
            if not bucket:
                return
            pause = max(0, len(bucket) - 1) * pacing.pause_sec
            raw = sum(bucket_secs) + pause + 2 * pacing.head_tail_pad_sec
            length = _safe_secs_to_len(secs_to_len, raw)
            budgets.append(
                ShotBudget(
                    shot_index=shot_index,
                    turn_index=turn_idx,
                    speaker=turn.speaker,
                    lines=list(bucket),
                    line_count=len(bucket),
                    char_count=bucket_chars,
                    estimated_speech_sec=round(sum(bucket_secs), 3),
                    pause_total_sec=round(pause, 3),
                    head_tail_pad_sec=2 * pacing.head_tail_pad_sec,
                    raw_seconds=round(raw, 3),
                    rounded_length_frames=length,
                    duration_sec=len_to_secs(length),
                )
            )
            total_frames += length
            # A sub-budget whose own raw runtime is over the cap means
            # even a single line inside it is too long for one shot
            # (we never cut a single line). Surface it in overbudget
            # so the caller can warn the user.
            if raw > max_shot_seconds:
                overbudget.append(shot_index)
            shot_index += 1
            running_secs = 0.0
            bucket = []
            bucket_secs = []
            bucket_chars = 0

        for line, sec in zip(turn.lines, per_line_secs):
            line_chars = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
            projected = (
                running_secs + sec
                + (max(0, len(bucket)) * pacing.pause_sec)
                + 2 * pacing.head_tail_pad_sec
            )
            if projected > max_shot_seconds and bucket:
                _flush()
            bucket.append(line)
            bucket_secs.append(sec)
            bucket_chars += line_chars
            running_secs += sec
        _flush()

    report = PacingReport(
        pacing=pacing.name,
        total_sec=round(total_frames / fps, 2),
        total_frames=total_frames,
        shot_count=len(budgets),
        overbudget_turns=overbudget,
    )
    return budgets, report


def _safe_secs_to_len(secs_to_len, raw_sec: float) -> int:
    try:
        return secs_to_len(raw_sec)
    except ValueError:
        return MIN_LENGTH_FRAMES


def _char_count_for(lines: list[str]) -> int:
    return sum(sum(1 for c in ln if "\u4e00" <= c <= "\u9fff") for ln in lines)


# --------------------------------------------------------------------------- #
# Override helpers
# --------------------------------------------------------------------------- #
# NOTE: a previous ``rebalance_shots_to_count`` helper used to live here
# and was supposed to merge adjacent same-speaker budgets when the user
# asked for fewer shots than the natural turn count. The implementation
# never actually merged anything (it only ever appended), and after the
# per-line split path landed the budget list is the source of truth — a
# caller that wants merging must explicitly concatenate budgets across
# same-speaker turns. The function was removed; the dialog-driven
# generator raises RuntimeError when ``shot_count < len(turns)`` so the
# user is never silently forced into a partial board.


def scene_raw_seconds(budgets: list, pacing) -> float:
    """Raw duration of a scene packing consecutive budgets into ONE shot.

    Each per-turn budget carries its own head/tail pad and intra-turn
    pauses; a packed scene keeps ONE pad for the whole scene and needs
    one ``pause_sec`` at every budget boundary::

        raw = sum(speech) + sum(intra pauses)
              + (n - 1) * pause_sec + 2 * head_tail_pad_sec
    """
    if not budgets:
        return 0.0
    speech = sum(b.estimated_speech_sec for b in budgets)
    pauses = sum(b.pause_total_sec for b in budgets)
    boundaries = max(0, len(budgets) - 1) * pacing.pause_sec
    pad = 2 * pacing.head_tail_pad_sec
    return round(speech + pauses + boundaries + pad, 3)


def group_budgets_into_scenes(
    budgets: list,
    pacing,
    *,
    bias: str = "balanced",
    max_shot_seconds: float = 14.0,
) -> list:
    """Group consecutive per-turn budgets into scenes honouring split_bias.

    Scene cuts are EXPENSIVE in the chain pipeline: every cut regenerates
    a carried overlap and risks visual discontinuity, so the default is
    to keep the scene count MINIMAL. A speaker change alone never forces
    a cut — consecutive turns pack into one scene until the H3
    single-generation window fills.

    ``aggressive``: one budget per scene — quick-cut rhythm, one shot
    per dialogue turn.

    ``balanced`` / ``conservative`` (and any unknown value): greedily
    pack CONSECUTIVE budgets into multi-turn scenes while
    ``scene_raw_seconds`` stays within ``max_shot_seconds``. A turn is
    never cut mid-utterance and same-speaker consecutive lines (already
    merged into one turn) are never separated — packing only ever
    MERGES across speaker changes, so all three dialogue invariants
    survive.

    Returns a list of scenes; each scene is a non-empty list of
    consecutive ``ShotBudget`` entries.
    """
    if bias == "aggressive":
        return [[b] for b in budgets]
    scenes: list = []
    current: list = []
    for budget in budgets:
        if current and (
            scene_raw_seconds(current + [budget], pacing) <= max_shot_seconds
        ):
            current.append(budget)
            continue
        if current:
            scenes.append(current)
        current = [budget]
    if current:
        scenes.append(current)
    return scenes


def distribute_lines_to_scenes(
    budgets: list,
    *,
    scene_count: int,
) -> list:
    """Distribute FINE-GRAINED (per-line) budgets across exactly
    ``scene_count`` contiguous scenes, EVENLY BY ESTIMATED SPEECH TIME.

    TIME is the only hard constraint in this pipeline: the caller sets
    every scene's duration afterwards (equal share of the budget) and
    merely WARNS when a scene's speech does not fit — whether the lines
    can actually be spoken inside their clip is not a packing
    constraint. So this helper balances the SPEAKING load, nothing
    else: each of the ``scene_count - 1`` boundaries is placed at the
    line edge whose accumulated speech is closest to its even share
    (``k * total / scene_count``). Every scene keeps at least one line,
    order is preserved, and no line is ever split.

    Requires ``1 <= scene_count <= len(budgets)``; callers top up
    counts above the line count with reaction cuts instead.
    """
    n = max(1, int(scene_count))
    if n >= len(budgets):
        return [[b] for b in budgets]
    speeches = [float(b.estimated_speech_sec or 0.0) for b in budgets]
    prefix = [0.0]
    for s in speeches:
        prefix.append(prefix[-1] + s)
    total = prefix[-1]
    target_share = total / n
    scenes: list = []
    start = 0
    for k in range(1, n):
        # Boundary for scene k: the line edge closest to k even shares,
        # keeping at least one line per remaining scene.
        best_end, best_diff = start + 1, None
        for end in range(start + 1, len(budgets) - (n - k) + 1):
            diff = abs((prefix[end] - prefix[start]) - k * target_share)
            if best_diff is None or diff < best_diff:
                best_diff, best_end = diff, end
        scenes.append(list(budgets[start:best_end]))
        start = best_end
    scenes.append(list(budgets[start:]))
    return scenes


def scale_shots_to_total(
    budgets: list[ShotBudget],
    target_total_sec: float,
    *,
    fps: int = FPS,
    min_shot_sec: float = 4.0,
    max_shot_sec: float = 14.0,
) -> list[ShotBudget]:
    """Scale each budget's length proportionally toward ``target_total_sec``.

    Each shot is clamped to the H3 single-generation band
    ``[min_shot_sec, max_shot_sec]``. After the proportional pass,
    leftover frames are walked onto the 17-frame grid and given to
    (or taken from) shots that still have headroom, so the sum gets
    as close to the target as the band allows. When every shot is
    already at a clamp, the sum cannot hit the target — that is
    inherent, not a silent floor.
    """
    if target_total_sec <= 0 or not budgets:
        return budgets
    secs_to_len, len_to_secs = seconds_to_length, length_to_seconds
    current_total = sum(b.duration_sec for b in budgets)
    if current_total <= 0:
        return budgets
    factor = target_total_sec / current_total
    try:
        min_len = secs_to_len(min_shot_sec)
        max_len = min(secs_to_len(max_shot_sec), MAX_LENGTH_FRAMES)
    except ValueError:
        min_len = MIN_LENGTH_FRAMES
        max_len = MAX_LENGTH_FRAMES
    lengths: list[int] = []
    for b in budgets:
        target_sec = max(min_shot_sec, min(max_shot_sec, b.duration_sec * factor))
        try:
            new_length = secs_to_len(target_sec)
        except ValueError:
            new_length = b.rounded_length_frames
        lengths.append(max(min_len, min(max_len, new_length)))
    desired_frames = max(min_len, int(round(float(target_total_sec) * fps)))
    current_frames = sum(lengths)
    for _ in range(len(lengths) * 32):
        diff = desired_frames - current_frames
        if abs(diff) < GRID_STEP:
            break
        if diff > 0:
            idx = max(
                range(len(lengths)),
                key=lambda i: (max_len - lengths[i]) if lengths[i] < max_len else -1,
            )
            if lengths[idx] + GRID_STEP > max_len:
                break
            lengths[idx] += GRID_STEP
            current_frames += GRID_STEP
        else:
            idx = max(
                range(len(lengths)),
                key=lambda i: (lengths[i] - min_len) if lengths[i] > min_len else -1,
            )
            if lengths[idx] - GRID_STEP < min_len:
                break
            lengths[idx] -= GRID_STEP
            current_frames -= GRID_STEP
    out: list[ShotBudget] = []
    for b, new_length in zip(budgets, lengths):
        out.append(
            ShotBudget(
                shot_index=b.shot_index,
                turn_index=b.turn_index,
                speaker=b.speaker,
                lines=list(b.lines),
                line_count=b.line_count,
                char_count=b.char_count,
                estimated_speech_sec=b.estimated_speech_sec,
                pause_total_sec=b.pause_total_sec,
                head_tail_pad_sec=b.head_tail_pad_sec,
                raw_seconds=b.raw_seconds,
                rounded_length_frames=new_length,
                duration_sec=len_to_secs(new_length),
            )
        )
    return out


def _cons_reindex(budgets: list[ShotBudget]) -> list[ShotBudget]:
    out: list[ShotBudget] = []
    for i, b in enumerate(budgets):
        out.append(
            ShotBudget(
                shot_index=i,
                turn_index=b.turn_index,
                speaker=b.speaker,
                lines=list(b.lines),
                line_count=b.line_count,
                char_count=b.char_count,
                estimated_speech_sec=b.estimated_speech_sec,
                pause_total_sec=b.pause_total_sec,
                head_tail_pad_sec=b.head_tail_pad_sec,
                raw_seconds=b.raw_seconds,
                rounded_length_frames=b.rounded_length_frames,
                duration_sec=b.duration_sec,
            )
        )
    return out


def pacing_report_text(report: PacingReport) -> str:
    """Plain-text rendering for preflight display."""
    over = (
        f"; overbudget turns: {report.overbudget_turns}"
        if report.overbudget_turns
        else ""
    )
    return (
        f"Pacing: {report.pacing}\n"
        f"Estimated total: {report.total_sec}s ({report.total_frames} frames)\n"
        f"Scenes: {report.shot_count}{over}"
    )