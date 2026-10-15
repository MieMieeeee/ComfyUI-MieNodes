# -*- coding: utf-8 -*-
"""Cut-budget guidance in the loop node's run summary.

The rule this file pins down: the summary only speaks about the cut when
the cut is still a CHOICE the user can act on.

* widgets at 0/0 (auto) -> name a concrete alternative to minimal-cut
  packing, so the user can dial it in next run;
* explicit values that cannot hold the speech -> say by how much, because
  that truncates a spoken line;
* explicit values that DO fit -> say nothing. A "try scene_count=4"
  beside a finished 5-scene plan_json is noise: the cut already happened
  and the user cannot act on it without a rerun anyway.

Live motivation (2026-10-05): a 7-line board run with scene_count=5 /
total_duration_seconds=45 truncated its longest line (10.1s of speech
into a 9.42s clip), and nothing in the report said so.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
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
def gen():
    root = "_mie_cut_budget"
    _ensure_pkg(root, PROJECT_DIR)
    _ensure_pkg(f"{root}.core", PROJECT_DIR / "core")
    _load_file(f"{root}.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg(f"{root}.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg(f"{root}.nodes.llm", LLM_DIR)
    _ensure_pkg(f"{root}.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(f"{root}.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py")
    _load_file(
        f"{root}.nodes.llm.dialogue_segmenter", LLM_DIR / "dialogue_segmenter.py"
    )
    _load_file(
        f"{root}.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    return _load_file(
        f"{root}.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )


def _seven_lines():
    """The board from the live run: 7 lines, the 6th needing ~10.1s."""
    concept = (
        "场景设定：深夜便利店，货架灯白光。\n"
        "小猫：老板，今天剩了那么多，要怎么处理？\n"
        "虎斑：全部倒掉。\n"
        "小猫：那明天我们还做这么多吗？\n"
        "虎斑：先做。\n"
        "小猫：不怕亏钱吗？\n"
        "虎斑：怕，但更怕账不清。宁可少挣点，也不能坏了规矩。"
        "规矩立住了，路才会越走越宽。\n"
        "小猫：还是我们这儿好，吃了不烧心。"
    )
    return concept


def _advice(gen, *, turns=None, pacing=None, scene_count=0, total=0, scenes=0):
    dlg = _dlg()
    if turns is None:
        turns, _pro = dlg.parse_structured_dialogue_turns(_seven_lines())
    if pacing is None:
        pacing = dlg.PACING_PRESETS["normal"]
    return gen._cut_budget_advice(
        turns=turns,
        pacing_obj=pacing,
        user_scene_count=scene_count,
        explicit_total=total,
        delivered_scenes=scenes,
    )


def _dlg():
    from importlib import import_module
    return import_module("_mie_cut_budget.nodes.llm.dialogue_segmenter")


# --------------------------------------------------------------------------- #
# Explicit values that truncate speech -> must say so
# --------------------------------------------------------------------------- #
def test_explicit_budget_too_small_warns_with_the_shortfall(gen):
    """The live failure: 45s across 5 scenes = 9s each, but the heaviest
    scene needs ~10.1s, so the longest line gets cut."""
    notes = _advice(gen, scene_count=5, total=45, scenes=5)

    assert notes, "a truncating budget must produce a warning"
    note = notes[0]
    assert "Cut budget" in note
    assert "truncated" in note
    # The shortfall must be quantified, not just flagged.
    assert "45s" in note and "5 scene" in note
    # 10.12s x 5 = 50.6 -> must round UP to 51, never down to 50
    # (a value that still truncates is worse than no advice).
    assert "51" in note, f"suggested total must be a working value: {note}"


def test_subsecond_shortfall_is_not_warned(gen):
    """10.0s per scene against 10.1s of speech is inside the estimator's
    own head/tail pad. Warning about it would train the user to ignore
    the line that matters (9.0 vs 10.1)."""
    notes = _advice(gen, scene_count=4, total=40, scenes=4)

    assert notes == [], f"a sub-second shortfall must stay quiet, got {notes}"


def test_explicit_budget_that_fits_says_nothing(gen):
    """55s across 5 scenes gives 11s per scene — the heaviest scene needs
    ~10.1s, so it fits and the summary must stay quiet about cuts."""
    notes = _advice(gen, scene_count=5, total=55, scenes=5)

    assert notes == [], f"no advice expected for a workable budget, got {notes}"


def test_four_scenes_at_40s_is_workable(gen):
    notes = _advice(gen, scene_count=4, total=40, scenes=4)

    assert notes == [], f"40s / 4 scenes fits, got {notes}"


# --------------------------------------------------------------------------- #
# Auto cut -> suggest a concrete alternative
# --------------------------------------------------------------------------- #
def test_auto_cut_suggests_a_concrete_alternative(gen):
    notes = _advice(gen, scene_count=0, total=0, scenes=2)

    assert notes, "auto cut should offer a concrete alternative"
    note = notes[0]
    assert "Cut budget" in note
    assert "scene_count=" in note
    assert "total_duration_seconds" in note
    # Must not just restate what already happened.
    assert f"used {2} scene" in note


def test_auto_cut_names_the_unsplittable_line(gen):
    """The suggestion has to say WHY it cannot cut finer, otherwise the
    number looks arbitrary."""
    notes = _advice(gen, scene_count=0, total=0, scenes=2)

    assert "cannot be split" in notes[0]
    assert "longest single line" in notes[0]


def test_two_line_packing_appends_reroll_hint(gen):
    """2026-10-15 live ASR forensics: H3 swallowed the shorter FIRST
    line in 3 of 4 two-line clips (raw generated audio, prompts
    verbatim-correct). More lines than scenes = at least one two-line
    scene, so the re-roll recipe rides along after the cut note."""
    notes = _advice(gen, scene_count=0, total=0, scenes=4)
    assert len(notes) >= 2
    assert "re-roll" in notes[1]
    assert "scene_range" in notes[1]


def test_one_line_per_scene_has_no_reroll_hint(gen):
    """A board whose scenes each hold one line is not the risk class;
    the hint stays quiet."""
    notes = _advice(gen, scene_count=0, total=0, scenes=7)
    assert all("re-roll" not in n for n in notes)


def test_auto_cut_skipped_when_alternative_equals_delivered(gen):
    """Suggesting scene_count=3 beside a delivered 3-scene board is the
    exact noise this rule exists to prevent."""
    dlg = _dlg()
    # A 4-line board: the only alternative on offer (3) is reachable,
    # so the delivered count is what decides whether we stay quiet.
    short = "图1的黑猫：好。\n图2的白猫：嗯。\n图1的黑猫：行。\n图2的白猫：走。"
    t2, _p2 = dlg.parse_structured_dialogue_turns(short)

    same = gen._cut_budget_advice(
        turns=t2,
        pacing_obj=dlg.PACING_PRESETS["normal"],
        user_scene_count=0,
        explicit_total=0,
        delivered_scenes=3,
    )
    assert same == [], f"delivered 3, offered 3 — must be quiet, got {same}"

    different = gen._cut_budget_advice(
        turns=t2,
        pacing_obj=dlg.PACING_PRESETS["normal"],
        user_scene_count=0,
        explicit_total=0,
        delivered_scenes=2,
    )
    assert "scene_count=3" in different[0]


# --------------------------------------------------------------------------- #
# Nothing to say
# --------------------------------------------------------------------------- #
def test_single_line_board_has_no_choice(gen):
    """One line, one scene — there is no cut decision to advise on."""
    from importlib import import_module
    dlg = import_module("_mie_cut_budget.nodes.llm.dialogue_segmenter")
    # Two lines is the parser floor; one line never reaches this path.
    turns, _p = dlg.parse_structured_dialogue_turns("图1的黑猫：好。\n图2的白猫：嗯。")

    notes = gen._cut_budget_advice(
        turns=turns,
        pacing_obj=dlg.PACING_PRESETS["normal"],
        user_scene_count=0,
        explicit_total=0,
        delivered_scenes=2,
    )
    assert notes == []


def test_never_raises_on_a_well_formed_board(gen):
    """Advice must not be able to break a run."""
    from importlib import import_module
    dlg = import_module("_mie_cut_budget.nodes.llm.dialogue_segmenter")
    turns, _p = dlg.parse_structured_dialogue_turns(_seven_lines())

    for kwargs in (
        {"user_scene_count": 0, "explicit_total": 0, "delivered_scenes": 3},
        {"user_scene_count": 5, "explicit_total": 45, "delivered_scenes": 5},
        {"user_scene_count": 5, "explicit_total": 0, "delivered_scenes": 5},
        {"user_scene_count": 0, "explicit_total": 45, "delivered_scenes": 3},
    ):
        out = gen._cut_budget_advice(
            turns=turns, pacing_obj=dlg.PACING_PRESETS["normal"], **kwargs
        )
        assert isinstance(out, list)
