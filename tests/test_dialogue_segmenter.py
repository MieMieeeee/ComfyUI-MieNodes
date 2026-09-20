# -*- coding: utf-8 -*-
"""Unit tests for the dialogue-segmenter module.

The module no longer does format parsing — that responsibility
moved to ``H3LoopPromptEnhancer.extract_dialogue``, which uses an
LLM with span-anchored verification. What remains here is:

  * ``ExtractedLine`` dataclass + the mechanical validators
    (``turns_from_extraction``, ``validate_extraction``,
    ``validate_extracted_lines``).
  * ``estimate_shots` budget math + per-pacing totals.
  * ``PACING_PRESETS`` stable shape.
  * ``scale_shots_to_total`` proportional scaling.
  * ``pacing_report_text`` plain-text rendering.
  * ``is_narrator_only`` / ``narrator_fallback_turn`` helpers.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_DIR / "nodes" / "llm"


def _ensure_pkg(fqn: str, path: Path | None = None):
    if fqn in sys.modules:
        return sys.modules[fqn]
    mod = types.ModuleType(fqn)
    if path is not None:
        mod.__path__ = [str(path)]
    mod.__package__ = fqn
    sys.modules[fqn] = mod
    return mod


def _load_file(fqn: str, path: Path):
    if fqn in sys.modules:
        del sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def seg():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.dialogue_segmenter",
        LLM_DIR / "dialogue_segmenter.py",
    )


# --------------------------------------------------------------------------- #
# turns_from_extraction
# --------------------------------------------------------------------------- #
def test_turns_from_extraction_basic(seg):
    """LLM returns per-line spans; we coerce to DialogueTurn with
    merged outer start/end and verbatim line texts."""
    raw = [
        {
            "speaker": "Sahli",
            "lines": [{"text": "hello.", "start": 12, "end": 18}],
        },
        {
            "speaker": "Harry",
            "lines": [
                {"text": "hi.", "start": 30, "end": 33},
                {"text": "bye.", "start": 40, "end": 44},
            ],
        },
    ]
    turns = seg.turns_from_extraction(raw)
    assert len(turns) == 2
    assert turns[0].speaker == "Sahli"
    assert turns[0].lines == ["hello."]
    assert turns[0].start == 12 and turns[0].end == 18
    assert turns[1].speaker == "Harry"
    assert turns[1].lines == ["hi.", "bye."]
    assert turns[1].start == 30 and turns[1].end == 44


def test_turns_from_extraction_skips_empty_speaker(seg):
    raw = [
        {"speaker": "", "lines": [{"text": "x", "start": 0, "end": 1}]},
        {"speaker": "Sahli", "lines": []},
        {"speaker": "Harry", "lines": [{"text": "hi", "start": 5, "end": 7}]},
    ]
    turns = seg.turns_from_extraction(raw)
    assert len(turns) == 1
    assert turns[0].speaker == "Harry"


# --------------------------------------------------------------------------- #
# validate_extracted_lines
# --------------------------------------------------------------------------- #
def test_validate_extracted_lines_passes_on_exact_match(seg):
    """The mechanical validator accepts a span whose slice exactly
    equals the line text (modulo whitespace padding)."""
    concept = "前文 Sahli: 你好。 后文"
    # "你好" sits at offset 10..12 in the concept.
    lines = [seg.ExtractedLine(text="你好", start=10, end=12)]
    assert seg.validate_extracted_lines(concept, lines) == []


def test_validate_extracted_lines_flags_paraphrase(seg):
    """LLM rewrote the line -> slice won't match -> validator flags it."""
    concept = "Sahli said: 你好。Harry said: 不好。"
    lines = [seg.ExtractedLine(text="你好世界", start=12, end=15)]
    errors = seg.validate_extracted_lines(concept, lines)
    assert any("expected" in e and "你好世界" in e for e in errors)


def test_validate_extracted_lines_flags_overlap(seg):
    concept = "Sahli: 你好呀。Harry: 不。"
    # First line owns [7:10) ("你好呀"); second line claims [7:11)
    # which starts before the first line ends -> overlap, AND the
    # slice "你好呀。" contains inner punctuation the second line's
    # text "好呀。" can't reproduce after strip alone.
    lines = [
        seg.ExtractedLine(text="你好呀", start=7, end=10),
        seg.ExtractedLine(text="好呀。", start=7, end=10),
    ]
    errors = seg.validate_extracted_lines(concept, lines)
    # Either overlap or content mismatch counts — both are
    # violations of the contract. We only require SOME error.
    assert errors, "expected at least one validation error"
    assert any(
        "overlap" in e or "concept[7:10]" in e for e in errors
    )


def test_validate_extracted_lines_flags_out_of_bounds(seg):
    concept = "hi"
    lines = [seg.ExtractedLine(text="hi", start=0, end=10)]
    errors = seg.validate_extracted_lines(concept, lines)
    assert any("invalid for concept len" in e for e in errors)


def test_validate_extracted_lines_allows_trimming(seg):
    """LLM is allowed to skip leading whitespace in the slice; we
    trim both sides before comparing."""
    concept = "Sahli:    你好。  "
    # '你好' sits at [12:14); bracket [8:14) — slice is
    # '  你好。 ' which after strip == '你好。'.
    lines = [seg.ExtractedLine(text="你好。", start=8, end=14)]
    assert seg.validate_extracted_lines(concept, lines) == []


def test_relocate_extracted_lines_fixes_unique_miss(seg):
    """Off-by-one / byte-offset spans that still uniquely locate the
    spoken words are rewritten so the strict validator can pass."""
    concept = '公猫问："给够钱就行？" 母猫答："给够钱。"'
    # The few-shot bug: [4:10] slices '"给够钱就行' instead of the
    # utterance at [5:11].
    lines = [seg.ExtractedLine(text="给够钱就行？", start=4, end=10)]
    relocated, notes = seg.relocate_extracted_lines(concept, lines)
    assert notes
    assert relocated[0].start == 5
    assert relocated[0].end == 11
    assert seg.validate_extracted_lines(concept, relocated) == []


def test_relocate_extracted_lines_leaves_ambiguous_line(seg):
    concept = "甲：你好。乙：你好。"
    # Two unused hits for 你好。; original span is wrong.
    lines = [seg.ExtractedLine(text="你好。", start=0, end=2)]
    relocated, notes = seg.relocate_extracted_lines(concept, lines)
    assert notes == []
    assert relocated[0].start == 0
    assert seg.validate_extracted_lines(concept, relocated)


# --------------------------------------------------------------------------- #
# estimate_shot_budget
# --------------------------------------------------------------------------- #
def test_estimator_basic_pacing_normal(seg):
    pacing = seg.PACING_PRESETS["normal"]
    turns = [seg.DialogueTurn(speaker="Sahli", lines=["你好。"])]
    budgets, report = seg.estimate_shot_budget(turns, pacing)
    assert len(budgets) == 1
    assert budgets[0].line_count == 1
    assert 1.0 < budgets[0].raw_seconds < 2.0
    assert budgets[0].rounded_length_frames >= seg.MIN_LENGTH_FRAMES
    assert report.total_frames >= budgets[0].rounded_length_frames - 1


def test_estimator_multi_line_turns_add_pause(seg):
    pacing = seg.PACING_PRESETS["normal"]
    turns = [seg.DialogueTurn(speaker="Sahli", lines=["hi.", "how are you."])]
    budgets, _ = seg.estimate_shot_budget(turns, pacing)
    assert budgets[0].line_count == 2
    assert budgets[0].pause_total_sec == 1.5


def test_estimator_marks_overbudget_turns(seg):
    pacing = seg.PACING_PRESETS["normal"]
    turns = [seg.DialogueTurn(
        speaker="S",
        lines=[f"你{'啊' * 60}。"],
    )]
    _, report = seg.estimate_shot_budget(turns, pacing)
    assert 0 in report.overbudget_turns


def test_estimator_total_seconds_positive(seg):
    turns = [seg.DialogueTurn(speaker="A", lines=["hi."])]
    _, report = seg.estimate_shot_budget(turns, seg.PACING_PRESETS["normal"])
    assert report.total_sec > 0
    assert report.shot_count >= 1


def test_pacing_presets_have_expected_keys(seg):
    for key in ("fast", "normal", "slow"):
        p = seg.PACING_PRESETS[key]
        assert p.name == key
        assert p.rate_cn > 0
        assert p.rate_en > 0
        assert p.pause_sec >= 0
        assert p.head_tail_pad_sec >= 0


# --------------------------------------------------------------------------- #
# scale_shots_to_total
# --------------------------------------------------------------------------- #
def test_scale_to_total_zero_or_negative_is_noop(seg):
    turns = [seg.DialogueTurn(speaker="A", lines=["x."])]
    budgets, _ = seg.estimate_shot_budget(turns, seg.PACING_PRESETS["normal"])
    assert seg.scale_shots_to_total(budgets, 0) == budgets
    assert seg.scale_shots_to_total(budgets, -5) == budgets


def test_scale_to_total_increases_budget_when_target_larger(seg):
    turns = [seg.DialogueTurn(speaker="A", lines=["x."])]
    budgets, _ = seg.estimate_shot_budget(turns, seg.PACING_PRESETS["normal"])
    before = sum(b.rounded_length_frames for b in budgets)
    scaled = seg.scale_shots_to_total(budgets, 60.0)
    after = sum(b.rounded_length_frames for b in scaled)
    assert after > before


def test_scale_to_total_stays_inside_band_when_target_is_tight(seg):
    """Four short lines scaled toward 15s cannot go below 4s each, so
    the sum may overshoot; every shot stays in [4, 14]."""
    turns = [
        seg.DialogueTurn(speaker=f"S{i}", lines=["你好。"])
        for i in range(4)
    ]
    budgets, _ = seg.estimate_shot_budget(turns, seg.PACING_PRESETS["normal"])
    scaled = seg.scale_shots_to_total(budgets, 15.0)
    assert len(scaled) == 4
    for b in scaled:
        assert 4.0 <= b.duration_sec <= 14.0
    # Residual is walked onto the 17-frame grid among shots with room;
    # with every shot already at the 4s floor the sum cannot hit 15s.
    assert sum(b.duration_sec for b in scaled) >= 15.0 - 1.0


# --------------------------------------------------------------------------- #
# pacing_report_text
# --------------------------------------------------------------------------- #
def test_pacing_report_text_renders(seg):
    turns = [seg.DialogueTurn(speaker="A", lines=["x."])]
    _, report = seg.estimate_shot_budget(turns, seg.PACING_PRESETS["normal"])
    text = seg.pacing_report_text(report)
    assert "Pacing" in text
    assert "Scenes:" in text


# --------------------------------------------------------------------------- #
# narrator helpers
# --------------------------------------------------------------------------- #
def test_is_narrator_only_on_empty(seg):
    assert seg.is_narrator_only([]) is True
    assert seg.is_narrator_only(
        [seg.DialogueTurn(speaker="A", lines=["x."])]
    ) is False


def test_narrator_fallback_turn_carries_concept(seg):
    concept = "wind through the alley"
    turn = seg.narrator_fallback_turn(concept)
    assert turn.speaker == "(narrator)"
    assert turn.lines == [concept]
    assert turn.start == 0
    assert turn.end == len(concept)


def test_narrator_fallback_turn_empty_concept(seg):
    turn = seg.narrator_fallback_turn("")
    assert turn.speaker == "(narrator)"
    assert "(empty concept)" in turn.lines[0]


# --------------------------------------------------------------------------- #
# scene_raw_seconds — the packing math used by split_bias!=aggressive
# --------------------------------------------------------------------------- #
def _make_budget(
    seg,
    *,
    speech: float,
    pauses: float = 0.0,
    line_count: int = 1,
    raw: float | None = None,
):
    """Build a ShotBudget whose scene-math fields are exactly what the
    packing formula consumes (estimated_speech_sec / pause_total_sec)
    plus a derived ``raw_seconds``. A budget with N>=2 lines carries
    (N-1)*pause_sec of intra-turn pause on top of speech.
    """
    if raw is None:
        raw = speech + pauses + 2 * 0.3  # 0.3 == normal head_tail_pad
    return seg.ShotBudget(
        shot_index=0,
        turn_index=0,
        speaker="X",
        lines=["x."],
        line_count=line_count,
        char_count=0,
        estimated_speech_sec=speech,
        pause_total_sec=pauses,
        head_tail_pad_sec=2 * 0.3,
        raw_seconds=raw,
        rounded_length_frames=seg.MIN_LENGTH_FRAMES,
        duration_sec=raw,
    )


def test_scene_raw_seconds_single_budget_equals_budget(seg):
    """One-budget scene: scene_raw_seconds must equal the budget's own
    raw_seconds (no inter-budget pause, one pad)."""
    pacing = seg.PACING_PRESETS["normal"]
    b = _make_budget(seg, speech=2.0)
    assert seg.scene_raw_seconds([b], pacing) == b.raw_seconds


def test_scene_raw_seconds_multi_budget_adds_inter_pauses(seg):
    """N budgets in one scene = Σspeech + Σintra-pauses
    + (n-1)*pause_sec + 2*head_tail_pad_sec.
    With two single-line budgets (no intra pauses), normal pacing
    (pause_sec=1.5) and pad=0.6, the inter-budget pause is the only
    delta vs. naive speech-sum."""
    pacing = seg.PACING_PRESETS["normal"]
    a = _make_budget(seg, speech=1.0, pauses=0.0)
    b = _make_budget(seg, speech=2.0, pauses=0.0)
    raw = seg.scene_raw_seconds([a, b], pacing)
    # 1.0 + 2.0 + 0 + (2-1)*1.5 + 2*0.3 = 5.2
    assert raw == round(1.0 + 2.0 + 1.5 + 2 * 0.3, 3)


def test_scene_raw_seconds_includes_intra_turn_pauses(seg):
    """If a budget spans multiple lines, its own pause_total_sec must
    be summed into the scene raw (scene keeps one pad, but every
    per-turn pause stays)."""
    pacing = seg.PACING_PRESETS["normal"]
    a = _make_budget(seg, speech=1.0, pauses=1.5)  # 2-line turn @ normal
    b = _make_budget(seg, speech=2.0, pauses=0.0)
    raw = seg.scene_raw_seconds([a, b], pacing)
    assert raw == round(1.0 + 2.0 + 1.5 + 1.5 + 2 * 0.3, 3)


def test_scene_raw_seconds_empty_scene_is_zero(seg):
    assert seg.scene_raw_seconds([], seg.PACING_PRESETS["normal"]) == 0.0


# --------------------------------------------------------------------------- #
# group_budgets_into_scenes — split_bias semantics
# --------------------------------------------------------------------------- #
def test_group_budgets_into_scenes_aggressive_one_budget_per_scene(seg):
    """aggressive: every budget is its own scene, regardless of fit.

    The generator strips the widget's ``" - description"`` tail via
    ``parse_split_bias`` before this function sees the bias, so we
    pass the bare code here."""
    pacing = seg.PACING_PRESETS["normal"]
    budgets = [_make_budget(seg, speech=1.0) for _ in range(5)]
    scenes = seg.group_budgets_into_scenes(
        budgets, pacing, bias="aggressive"
    )
    assert len(scenes) == 5
    assert all(len(s) == 1 for s in scenes)
    assert [s[0] for s in scenes] == budgets


def test_group_budgets_into_scenes_balanced_packs_consecutive_budgets(seg):
    """balanced (default) greedily packs consecutive budgets into one
    scene while the packing math stays <= 14s. A speaker switch
    alone NEVER opens a new scene."""
    pacing = seg.PACING_PRESETS["normal"]
    # 5 tiny single-line budgets — all pack into one scene.
    budgets = [_make_budget(seg, speech=1.0) for _ in range(5)]
    scenes = seg.group_budgets_into_scenes(budgets, pacing, bias="balanced")
    assert len(scenes) == 1
    assert len(scenes[0]) == 5


def test_group_budgets_into_scenes_conservative_packs_consecutive_budgets(seg):
    """conservative behaves the same as balanced — only aggressive
    deviates. Both must pack."""
    pacing = seg.PACING_PRESETS["normal"]
    budgets = [_make_budget(seg, speech=1.0) for _ in range(4)]
    for label in ("conservative - 更少分场/更长镜头", "balanced", "未知偏置"):
        scenes = seg.group_budgets_into_scenes(budgets, pacing, bias=label)
        assert len(scenes) == 1, (
            f"{label!r} must pack into one scene, got {len(scenes)}"
        )


def test_group_budgets_into_scenes_splits_when_window_fills(seg):
    """When adding the next budget would push scene_raw_seconds past
    14s, the packer opens a new scene — speaker change is irrelevant."""
    pacing = seg.PACING_PRESETS["normal"]
    # 5s speech each, normal pacing: 1 budget = 5 + 0 + 0.6 = 5.6s.
    # Two pack to 5+5 + 1.5 + 0.6 = 12.1s (fits, ≤14); three = 18.6s
    # (over) so b3 opens a new scene. b3+b4 = 12.1s (fits); b5 alone.
    budgets = [_make_budget(seg, speech=5.0) for _ in range(5)]
    scenes = seg.group_budgets_into_scenes(budgets, pacing, bias="balanced")
    # Three scenes: [b1,b2], [b3,b4], [b5].
    assert len(scenes) == 3
    assert [len(s) for s in scenes] == [2, 2, 1]
    # Every scene's raw must be within the 14s single-generation window.
    for scene in scenes:
        assert seg.scene_raw_seconds(scene, pacing) <= 14.0


def test_group_budgets_into_scenes_speaker_change_does_not_cut(seg):
    """Packing is purely a time-budget decision: alternating speakers
    pack together whenever the time window allows."""
    pacing = seg.PACING_PRESETS["normal"]
    # Two tiny single-line budgets from different speakers; 1.5s of
    # speech each fits trivially under the 14s window.
    a = _make_budget(seg, speech=1.5)
    a_speaker = "alt"
    # Rebuild a second budget from the other speaker.
    b = _make_budget(seg, speech=1.5)
    b_speaker = "bob"
    # Patch the speakers (the helper hard-codes "X").
    a.speaker = a_speaker
    b.speaker = b_speaker
    scenes = seg.group_budgets_into_scenes(
        [a, b], pacing, bias="balanced"
    )
    assert len(scenes) == 1
    assert scenes[0][0].speaker == a_speaker
    assert scenes[0][1].speaker == b_speaker


def test_group_budgets_into_scenes_overbudget_budget_kept_alone(seg):
    """A single over-cap budget must NOT be merged with the next one
    — packing checks the running total, and the over-cap budget alone
    already saturates the 14s window."""
    pacing = seg.PACING_PRESETS["normal"]
    over = _make_budget(seg, speech=20.0)  # way over 14s
    tiny = _make_budget(seg, speech=1.0)
    scenes = seg.group_budgets_into_scenes(
        [over, tiny], pacing, bias="balanced"
    )
    # over alone → scene 1; tiny alone → scene 2.
    assert len(scenes) == 2
    assert scenes[0] == [over]
    assert scenes[1] == [tiny]


def test_group_budgets_into_scenes_empty_budgets(seg):
    """No budgets → no scenes; the function must not raise."""
    scenes = seg.group_budgets_into_scenes(
        [], seg.PACING_PRESETS["normal"], bias="balanced"
    )
    assert scenes == []


def test_group_budgets_into_scenes_custom_window(seg):
    """``max_shot_seconds`` is honoured — a 4s budget alone fits a
    3s window would not happen (single-budget scene always allowed),
    but a 4s budget + 1.5s boundary at a 5s window DOES split."""
    pacing = seg.PACING_PRESETS["normal"]
    a = _make_budget(seg, speech=2.0)  # 2 + 0 + 0.6 = 2.6s
    b = _make_budget(seg, speech=2.0)
    # At window=3s: 2.6 + 2.0 + 1.5 + 0.6 = 6.7 → split.
    scenes = seg.group_budgets_into_scenes(
        [a, b], pacing, bias="balanced", max_shot_seconds=3.0
    )
    assert len(scenes) == 2

# --------------------------------------------------------------------------- #
# distribute_lines_to_scenes — TIME-only model: speech is spread evenly,
# fitting inside the scene duration is a warning, not a constraint
# --------------------------------------------------------------------------- #
def test_distribute_even_speech_across_target_scenes(seg):
    """4 lines with unequal speech, target 2 scenes: boundaries fall at
    total/2 — both scenes carry ~half the speaking load, order kept."""
    turns = [
        seg.DialogueTurn(speaker="A猫", lines=["短。"]),                # ~0.3s
        seg.DialogueTurn(speaker="B猫", lines=["这一句要长得多得多。"]),   # ~2s
        seg.DialogueTurn(speaker="A猫", lines=["又一句长一些的台词。"]),   # ~1.7s
        seg.DialogueTurn(speaker="B猫", lines=["再短。"]),               # ~0.3s
    ]
    budgets, _ = seg.estimate_shot_budget(
        turns, seg.PACING_PRESETS["normal"], max_shot_seconds=0.0
    )
    assert len(budgets) == 4
    scenes = seg.distribute_lines_to_scenes(budgets, scene_count=2)
    assert len(scenes) == 2
    assert all(scenes), "no empty scene"
    # Order preserved, every line exactly once.
    packed = [ln for s in scenes for b in s for ln in b.lines]
    assert packed == [t for t in ("短。", "这一句要长得多得多。", "又一句长一些的台词。", "再短。")]
    # Speech roughly balanced (within one line's tolerance).
    loads = [sum(b.estimated_speech_sec for b in s) for s in scenes]
    total = sum(loads)
    assert max(loads) <= total * 0.75, loads


def test_distribute_target_equals_lines_one_per_scene(seg):
    budgets, _ = seg.estimate_shot_budget(
        [seg.DialogueTurn(speaker="A猫", lines=["你好。", "再见。"]),
         seg.DialogueTurn(speaker="B猫", lines=["好的。"])],
        seg.PACING_PRESETS["normal"], max_shot_seconds=0.0,
    )
    scenes = seg.distribute_lines_to_scenes(budgets, scene_count=3)
    assert [len(s) for s in scenes] == [1, 1, 1]


def test_distribute_one_huge_line_falls_back_to_count_split(seg):
    """A single line whose speech exceeds the per-scene target must not
    leave an empty scene — contiguous count split kicks in."""
    budgets, _ = seg.estimate_shot_budget(
        [seg.DialogueTurn(speaker="A猫", lines=["极短。"]),
         seg.DialogueTurn(speaker="B猫", lines=["超" * 60 + "长台词。"]),
         seg.DialogueTurn(speaker="A猫", lines=["极短。"]),
         seg.DialogueTurn(speaker="B猫", lines=["也短。"])],
        seg.PACING_PRESETS["normal"], max_shot_seconds=0.0,
    )
    scenes = seg.distribute_lines_to_scenes(budgets, scene_count=2)
    assert len(scenes) == 2
    assert all(scenes)
    packed = [b for s in scenes for b in s]
    assert packed == budgets


# --------------------------------------------------------------------------- #
# scenes_for_duration — pacing owns the cut density
# --------------------------------------------------------------------------- #
def test_scenes_for_duration_per_pacing(seg):
    """Same budget, different tempo -> different cut counts: fast cuts
    ~4.5s scenes, normal ~7s, slow ~12s."""
    fast = seg.PACING_PRESETS["fast"]
    normal = seg.PACING_PRESETS["normal"]
    slow = seg.PACING_PRESETS["slow"]
    assert seg.scenes_for_duration(20, fast) == 4      # 20/4.5 = 4.4
    assert seg.scenes_for_duration(20, normal) == 3    # 20/7 = 2.9
    assert seg.scenes_for_duration(20, slow) == 2      # 20/12 = 1.7
    assert seg.scenes_for_duration(10, fast) == 2
    assert seg.scenes_for_duration(0, fast) == 1
    assert seg.scenes_for_duration(-5, normal) == 1


def test_pacing_presets_carry_target_scene_sec(seg):
    for key, expect in (("fast", 4.5), ("normal", 7.0), ("slow", 12.0)):
        assert seg.PACING_PRESETS[key].target_scene_sec == expect


# --------------------------------------------------------------------------- #
# parse_structured_dialogue_turns — deterministic canonical-format path
# --------------------------------------------------------------------------- #
def test_parse_structured_dialogue_turns_canonical(seg):
    concept = "莎莉猫：你好。\n哈利猫：为什么。\n莎莉猫：再见。"
    turns, prologue = seg.parse_structured_dialogue_turns(concept)
    assert [t.speaker for t in turns] == ["莎莉猫", "哈利猫", "莎莉猫"]
    assert prologue == ""
    for t in turns:
        assert concept[t.start:t.end] == t.lines[0]


def test_parse_structured_dialogue_turns_prologue_and_exact_spans(seg):
    concept = "夏日午后的庭院，阳光斜照。\n莎莉猫：你好。\n哈利猫：好的。"
    turns, prologue = seg.parse_structured_dialogue_turns(concept)
    assert prologue == "夏日午后的庭院，阳光斜照。"
    for t in turns:
        assert concept[t.start:t.end] == t.lines[0]


def test_parse_structured_dialogue_turns_merges_adjacent_same_speaker(seg):
    turns, _ = seg.parse_structured_dialogue_turns(
        "甲：第一句。\n甲：第二句。\n乙：回话。"
    )
    assert [(t.speaker, t.lines) for t in turns] == [
        ("甲", ["第一句。", "第二句。"]), ("乙", ["回话。"]),
    ]


def test_parse_structured_dialogue_turns_rejects_directives(seg):
    assert seg.parse_structured_dialogue_turns(
        "镜头：缓慢推进\n音乐：轻快"
    ) == (None, "")


def test_parse_structured_dialogue_turns_rejects_mixed_tail(seg):
    assert seg.parse_structured_dialogue_turns(
        "甲：你好。\n乙：好的。\n突然，画外传来巨响。"
    ) == (None, "")


def test_parse_structured_dialogue_turns_single_line_not_confident(seg):
    # One colon line is too weak a signal — falls back to the LLM.
    assert seg.parse_structured_dialogue_turns("注意：这是一段说明。") == (None, "")
    assert seg.parse_structured_dialogue_turns("") == (None, "")


def test_parse_structured_dialogue_turns_english_speakers(seg):
    turns, _ = seg.parse_structured_dialogue_turns(
        "Sahli: hello there.\nMolly: hi back."
    )
    assert [t.speaker for t in turns] == ["Sahli", "Molly"]
