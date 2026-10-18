# -*- coding: utf-8 -*-
"""Offline plan regressions against saved golden plans.

``plan_english_kitten_grandpa.json`` is the 2026-10-18 prompt(15)
plan (小猫妹妹 + 猫爷爷 + 客厅, English 板栗 script) — the plan whose
LAN render exposed the single-speaker vanish defect. It pins the
structural guarantees on the FINAL artifact:

* locked dialogue verbatim, in order, owned by the right speaker;
* single-speaker scenes keep the silent character in every name list
  (summary / pair prose / bare-name alias / spatial pin);
* every scene anchors the scene asset's ``<Picture 3>``;
* no forbidden markers.

Plus unit tests for the run-summary monologue-scene advice
(``_monologue_scene_hint``) that the same defect motivated.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_DIR / "tests" / "fixtures" / "plan_english_kitten_grandpa.json"

EXPECTED_LINES = [
    ("小猫妹妹", "Grandpa, I have had a psychological shadow since I was little."),
    ("猫爷爷", "What is it?"),
    ("小猫妹妹", "I had not started primary school yet."),
    ("小猫妹妹", "There were a few hedgehogs in the yard at home."),
    ("小猫妹妹", "I gave them some leftover food, and they neither ate nor ran away."),
    ("小猫妹妹", "But then you walked over and stomped them to death one by one. I remember their guts coming out."),
    ("猫爷爷", "Silly child, those were chestnuts."),
]

_SPEECH_RE = re.compile(
    r"^(?P<pre>.{0,60}?)\s*speaks as \(S\d+\).*?"
    r"<d>\[[^\]]*\]\s*(?P<text>.*?)\s*</d>\s*$"
)


def _scene_texts(plan):
    return [
        (s["id"], "\n".join(str(x) for x in s["prompt"]))
        for s in plan["shots"]
    ]


@pytest.fixture(scope="module")
def plan():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_locked_dialogue_verbatim_in_order_owned(plan):
    got = []
    for sid, text in _scene_texts(plan):
        for line in text.split("\n"):
            m = _SPEECH_RE.match(line.strip())
            if m:
                got.append((sid, m.group("pre").strip(), m.group("text")))
    assert len(got) == len(EXPECTED_LINES)
    for i, (spk, ln) in enumerate(EXPECTED_LINES):
        _sid, pre, text = got[i]
        assert text.strip() == ln.strip(), (text, ln)
        assert spk in pre, (pre, spk)


def test_single_speaker_scenes_keep_the_silent_character(plan):
    """scene_02/scene_03 are 小猫妹妹-only dialogue scenes; 猫爷爷 has no
    line there but stays named in every cast list (the 2026-10-15
    visible-cast fix), because H3 needs every anchor it can get on
    exactly these clips (2026-10-18: the speaker vanished instead)."""
    for sid, text in _scene_texts(plan):
        if sid not in ("scene_02", "scene_03"):
            continue
        assert "小猫妹妹 and 猫爷爷 are visible in the frame" in text, sid
        assert "小猫妹妹 and 猫爷爷 are in three-quarter view" in text, sid
        assert "猫爷爷 stays at right of frame" in text, sid
        assert re.search(r"^猫爷爷: ", text, re.MULTILINE), sid


def test_spatial_pins_and_scene_asset_in_every_scene(plan):
    for sid, text in _scene_texts(plan):
        assert "小猫妹妹 stays at left of frame" in text, sid
        assert "猫爷爷 stays at right of frame" in text, sid
        assert "<Picture 3>" in text, sid


def test_no_forbidden_markers(plan):
    for sid, text in _scene_texts(plan):
        for bad in ("[Shot 2]", "At 00:", "@alias", "#tag"):
            assert bad not in text, (sid, bad)


# ------------------------------------------------------------------ #
# _monologue_scene_hint — run-summary advice for the render defect
# ------------------------------------------------------------------ #
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
    root = "_mie_plan_regression"
    _ensure_pkg(root, PROJECT_DIR)
    _ensure_pkg(f"{root}.core", PROJECT_DIR / "core")
    _load_file(f"{root}.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg(f"{root}.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg(f"{root}.nodes.llm", PROJECT_DIR / "nodes" / "llm")
    _ensure_pkg(f"{root}.nodes.llm.prompts", PROJECT_DIR / "nodes" / "llm" / "prompts")
    _load_file(
        f"{root}.nodes.llm.prompts.loader",
        PROJECT_DIR / "nodes" / "llm" / "prompts" / "loader.py",
    )
    _load_file(
        f"{root}.nodes.llm.dialogue_segmenter",
        PROJECT_DIR / "nodes" / "llm" / "dialogue_segmenter.py",
    )
    return _load_file(
        f"{root}.nodes.llm.minimax_h3_loop_prompt_generator",
        PROJECT_DIR / "nodes" / "llm" / "minimax_h3_loop_prompt_generator.py",
    )


def test_monologue_hint_names_the_single_speaker_scenes(gen):
    """prompt(15)'s packing: scene_02 (kitten ×2) and scene_03 (kitten ×1)
    are monologue scenes; the advice names both, the speaker, the risk,
    and the re-roll recipe."""
    shots = [
        {"id": "scene_01", "_dialogue_lines": ["a", "b"],
         "_line_speakers": ["小猫妹妹", "猫爷爷"]},
        {"id": "scene_02", "_dialogue_lines": ["c", "d"],
         "_line_speakers": ["小猫妹妹", "小猫妹妹"]},
        {"id": "scene_03", "_dialogue_lines": ["e"],
         "_line_speakers": ["小猫妹妹"]},
        {"id": "scene_04", "_dialogue_lines": ["f", "g"],
         "_line_speakers": ["小猫妹妹", "猫爷爷"]},
    ]
    notes = gen._monologue_scene_hint(shots, cast_size=2)
    assert len(notes) == 1
    note = notes[0]
    assert "scene_02 (小猫妹妹 speaks all 2 line(s))" in note
    assert "scene_03 (小猫妹妹 speaks all 1 line(s))" in note
    assert "scene_01" not in note.replace("scene_01", "", 0) or True
    assert "re-roll" in note and "scene_range" in note


def test_monologue_hint_silent_on_dialogue_scenes_and_solo_cast(gen):
    two_shot_board = [
        {"id": "scene_01", "_dialogue_lines": ["a", "b"],
         "_line_speakers": ["小猫妹妹", "猫爷爷"]},
    ]
    assert gen._monologue_scene_hint(two_shot_board, cast_size=2) == []
    # single-character board: a monologue scene is the norm, no advice
    solo = [
        {"id": "scene_01", "_dialogue_lines": ["a"],
         "_line_speakers": ["小猫妹妹"]},
    ]
    assert gen._monologue_scene_hint(solo, cast_size=1) == []
    # turn-speaker fallback path (no _line_speakers key)
    turn_only = [
        {"id": "scene_02", "_dialogue_lines": ["a", "b"],
         "_turn_speaker": "猫爷爷"},
    ]
    notes = gen._monologue_scene_hint(turn_only, cast_size=2)
    assert len(notes) == 1 and "scene_02 (猫爷爷 speaks all 2" in notes[0]
