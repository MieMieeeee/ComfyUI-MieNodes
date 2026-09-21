# -*- coding: utf-8 -*-
"""Regression tests for the 2026-09-22 live failure (English-dialogue
board, phantom ``场景设定`` speaker, unused reference pictures).

Live symptoms these tests lock down:
  1. The enhancer's canonical ``场景设定：`` setting paragraph became the
     FIRST speaker (S1) — on the deterministic fast path (stoplist gap)
     and potentially on the LLM extractor path (no meta-speaker filter).
     The setting paragraph was then rendered as an off-screen voiceover
     <d>[Chinese] block.
  2. Every per-shot user template carried "dialogue is Chinese by
     default" even for an all-English board, pulling against the
     verbatim English <d> blocks.
  3. Five reference pictures were wired while the concept named only
     图1-3: Pictures 4/5 silently bound Subjects 4/5; and the concept's
     黑猫 label contradicted Picture 1's caption (a brown tabby),
     baking two conflicting identities into the plan — silently.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
LLM_DIR = PROJECT_DIR / "nodes" / "llm"


def _ensure_pkg(fqn, path=None):
    if fqn in sys.modules:
        return sys.modules[fqn]
    mod = types.ModuleType(fqn)
    if path is not None:
        mod.__path__ = [str(path)]
    mod.__package__ = fqn
    sys.modules[fqn] = mod
    return mod


def _load_file(fqn, path):
    if fqn in sys.modules:
        del sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader",
        PROMPTS_DIR / "loader.py",
    )
    _load_file("_mienodes_internal.nodes.llm.h3_prompts", LLM_DIR / "h3_prompts.py")
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
        LLM_DIR / "minimax_h3_storyboard_prompts.py",
    )
    lp = _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    lg = _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )
    return lg, lp


# The user's scenario, in the enhancer's canonical rewrite shape:
# setting paragraph (场景设定：...) + five English speaker lines.
CONCEPT = (
    "场景设定：暖色灯光的家庭客厅室内。参考图 1 → 黑猫（猫爸爸，画面右边）；"
    "参考图 2 → 白猫（猫妈妈，画面左边）；参考图 3 → 小猫（站画面中间）。"
    "对称中景，机位固定不动。\n"
    "小猫：Mommy, daddy is so ugly, why did you marry him.\n"
    "白猫：Iguess I was blind.\n"
    "小猫：Daddy, why did you marry a blind lady?\n"
    "白猫：Oh my god.\n"
    "黑猫：Well, nobody's perfect."
)


# --------------------------------------------------------------------------- #
# Fix 1a — deterministic fast path: 场景设定 is in the stoplist
# --------------------------------------------------------------------------- #
def test_fast_path_setting_paragraph_is_prologue_not_speaker(mods):
    lg, _lp = mods
    turns, prologue = lg._dlg_parse_structured_dialogue_turns(CONCEPT)
    assert turns is not None, "canonical concept must parse deterministically"
    assert [t.speaker for t in turns] == ["小猫", "白猫", "小猫", "白猫", "黑猫"]
    assert "场景设定" not in [t.speaker for t in turns]
    assert "暖色灯光" in prologue, "the setting paragraph is narration prose"


# --------------------------------------------------------------------------- #
# Fix 1b — full pipeline: no phantom S1, all <d> blocks English
# --------------------------------------------------------------------------- #
class _Conn:
    model = "stub"
    timeout = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)

    def get_state(self):
        return "stub"


PREFIX_REPLY = (
    "3D CG rendered animation in a warm-toned family living room.\n\n"
    "CAST:\n"
    "黑猫: brown tabby father cat in a tweed blazer.\n"
    "白猫: cream-furred mother cat in a plaid jacket.\n"
    "小猫: ginger kitten in yellow overalls.\n"
)


def test_english_dialogue_pipeline_no_phantom_speaker(mods):
    lg, _lp = mods
    # Normal pacing, scene_count auto -> minimal-cut packing.
    # Speech math: [t1..t3] packs one scene (~12s), [t4..t5] the second
    # (~4.5s). 5 turns >= 5 -> the stage-1 prefix LLM call runs.
    conn = _Conn([
        PREFIX_REPLY,
        "integrated_multimodal_description:\n[Shot 1] The kitten and "
        "mother perform the first exchange.\n\n"
        "overall_soundscape:\nRoom tone.\n\n"
        "non_diegetic_music:\nNo non-diegetic music.\n",
        "integrated_multimodal_description:\n[Shot 1] The mother reacts "
        "and the father closes the exchange.\n\n"
        "overall_soundscape:\nRoom tone.\n\n"
        "non_diegetic_music:\nNo non-diegetic music.\n",
    ])
    out = lg.H3LoopPromptEnhancer(conn, temperature=0.4, timeout=60)(
        user_input=CONCEPT,
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 2
    all_text = "\n".join(
        "\n".join(s["prompt"]) for s in plan["shots"]
    )
    # The phantom speaker never appears; every <d> block is English.
    assert "场景设定" not in all_text
    assert "<d>[Chinese]" not in all_text
    d_lines = [ln for ln in all_text.split("\n") if "<d>" in ln]
    assert len(d_lines) == 5
    assert all("<d>[English]" in ln for ln in d_lines)
    # Speaker order: first spoken line owns S1 (小猫), then 白猫=S2,
    # 黑猫=S3 — the setting paragraph eats no ID. The appended block
    # format is `name[, cast identity]. (S<n>): <d>[Lang] text</d>`.
    assert any(ln.startswith("小猫") and "(S1):" in ln for ln in d_lines)
    assert any(ln.startswith("白猫") and "(S2):" in ln for ln in d_lines)
    assert any(ln.startswith("黑猫") and "(S3):" in ln for ln in d_lines)
    # 1 prefix call + 2 shot calls; no extractor call (fast path).
    assert len(conn.calls) == 3


# --------------------------------------------------------------------------- #
# Fix 1c — LLM extractor path: meta-speaker turns are dropped mechanically
# --------------------------------------------------------------------------- #
def _llm_path_concept():
    # A prose tail after the dialogue defeats the fast-path confidence
    # guard, forcing the LLM extractor.
    return (
        "场景设定：客厅里灯光温暖。\n"
        "甲猫：你好。\n"
        "乙猫：再见。\n"
        "后来他们一起走出门去。"
    )


def test_llm_extractor_drops_meta_speaker(mods):
    lg, _lp = mods
    concept = _llm_path_concept()

    def span(text):
        s = concept.index(text)
        return s, s + len(text)

    reply = json.dumps({"turns": [
        {"speaker": "场景设定",
         "lines": [{"text": "客厅里灯光温暖。", "start": span("客厅里灯光温暖。")[0],
                    "end": span("客厅里灯光温暖。")[1]}]},
        {"speaker": "甲猫",
         "lines": [{"text": "你好。", "start": span("你好。")[0],
                    "end": span("你好。")[1]}]},
        {"speaker": "乙猫",
         "lines": [{"text": "再见。", "start": span("再见。")[0],
                    "end": span("再见。")[1]}]},
    ]}, ensure_ascii=False)
    conn = _Conn([reply])
    turns = lg.H3LoopPromptEnhancer(conn).extract_dialogue(concept)
    assert [t.speaker for t in turns] == ["甲猫", "乙猫"]


def test_llm_extractor_meta_only_returns_empty_without_retries(mods):
    lg, _lp = mods
    concept = _llm_path_concept()
    setting = "客厅里灯光温暖。"
    s = concept.index(setting)
    reply = json.dumps({"turns": [
        {"speaker": "场景设定",
         "lines": [{"text": setting, "start": s, "end": s + len(setting)}]},
    ]}, ensure_ascii=False)
    conn = _Conn([reply])
    turns = lg.H3LoopPromptEnhancer(conn).extract_dialogue(concept)
    assert turns == [], "meta-only extraction is a narration board"
    # Immediate narrator fallback: exactly ONE extractor attempt.
    assert len(conn.calls) == 1


# --------------------------------------------------------------------------- #
# Fix 2 — dynamic dialogue language policy
# --------------------------------------------------------------------------- #
def test_language_policy_follows_the_lines(mods):
    _lg, lp = mods
    base = dict(
        concept="c", prefix_text="p", category="none",
        continuation_block="", shot={"id": "s", "description": "d"},
        clip_index=1, clip_count=1, duration_seconds=5.0,
        language_name="en",
    )
    en = lp.build_shot_user_text(
        dialogue_lines=["Hello there.", "How are you?"], **base
    )
    assert "All spoken dialogue is English" in en
    assert "Chinese by default" not in en

    zh = lp.build_shot_user_text(
        dialogue_lines=["你好。", "最近怎么样？"], **base
    )
    assert "All spoken dialogue is Chinese" in zh

    mixed = lp.build_shot_user_text(
        dialogue_lines=["Hello.", "你好。"], **base
    )
    assert "mixed-language" in mixed

    none = lp.build_shot_user_text(**base)
    assert "No spoken dialogue" in none


def test_single_call_policy_follows_the_lines(mods):
    _lg, lp = mods
    shots = [
        {"id": "s1", "description": "d", "_dialogue_lines": ["Hello."]},
        {"id": "s2", "description": "d", "_dialogue_lines": ["World."]},
    ]
    text = lp.build_single_call_user_text(
        concept="c", prefix_text="p", category="none", shots=shots,
        duration_seconds=10, language_name="en", cast_sheet="A: cat.",
    )
    assert "All spoken dialogue is English" in text
    assert "Chinese by default" not in text


def test_shot_template_has_no_hardcoded_chinese_default():
    txt = (PROMPTS_DIR / "h3_loop" / "shot_user_template.txt").read_text(
        encoding="utf-8"
    )
    assert "Chinese by default" not in txt
    assert "{dialogue_language_policy}" in txt


# --------------------------------------------------------------------------- #
# Fix 3 — manifest consistency warnings
# --------------------------------------------------------------------------- #
MANIFEST_5 = [
    {"slot": "Picture 1", "role": "identity",
     "about": "a chubby brown tabby cat with dark brown/grayish-brown "
              "striped fur in a tweed blazer"},
    {"slot": "Picture 2", "role": "identity",
     "about": "cream-furred cat with a blonde hairdo, white collared "
              "shirt, plaid jacket"},
    {"slot": "Picture 3", "role": "identity",
     "about": "fluffy ginger-orange kitten in yellow corduroy overalls"},
    {"slot": "Picture 4", "role": "identity",
     "about": "glossy black ceramic cat figurine on a parquet floor"},
    {"slot": "Picture 5", "role": "identity",
     "about": "glossy black ceramic cat figurine, seated, waving"},
]


def test_unreferenced_pictures_and_colour_contradictions_warn(mods):
    lg, _lp = mods
    warns = lg._manifest_consistency_warnings(CONCEPT, MANIFEST_5, "ref2va")
    joined = "\n".join(warns)
    # Pictures 4/5 wired but the concept names 图1-3 only.
    assert any("4" in w and "5" in w and "never referenced" in w for w in warns)
    # 黑猫 (black family) vs Picture 1's brown-tabby caption.
    assert any("黑猫" in w and "Picture 1" in w for w in warns)
    # 白猫 (white family) vs Picture 2's cream/white caption: SAME
    # family — no contradiction fires (precision over recall).
    assert not any("白猫" in w and "Picture 2" in w for w in warns)
    # 小猫 carries no colour token: never flagged.
    assert not any("小猫" in w for w in warns)
    _ = joined


def test_manifest_warnings_off_for_t2va(mods):
    lg, _lp = mods
    assert lg._manifest_consistency_warnings(CONCEPT, MANIFEST_5, "t2va") == []


def test_no_unused_warning_when_all_pictures_referenced(mods):
    lg, _lp = mods
    manifest_3 = MANIFEST_5[:3]
    warns = lg._manifest_consistency_warnings(CONCEPT, manifest_3, "ref2va")
    assert not any("never referenced" in w for w in warns)
