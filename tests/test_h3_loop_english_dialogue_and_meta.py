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
    # (~4.5s). 5 turns >= 5 -> the stage-1 prefix LLM call runs. The
    # concept mentions 图1-3 -> the semantic-facts gate fires -> one
    # facts call (layout/bindings) between parse and prefix.
    FACTS = json.dumps({
        "turns": [],
        "layout": [
            {"name": "黑猫", "position": "right of frame"},
            {"name": "白猫", "position": "left of frame"},
            {"name": "小猫", "position": "center of frame"},
        ],
        "bindings": [
            {"picture": 1, "name": "黑猫"},
            {"picture": 2, "name": "白猫"},
            {"picture": 3, "name": "小猫"},
        ],
    }, ensure_ascii=False)
    conn = _Conn([
        FACTS,
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
    # 黑猫=S3 — the setting paragraph eats no ID. The speech sentence
    # is `name ... as (Sn) <voice> <d>`, with no colon before the tag.
    assert any(ln.startswith("小猫") and "as (S1) " in ln for ln in d_lines)
    assert any(ln.startswith("白猫") and "as (S2) " in ln for ln in d_lines)
    assert any(ln.startswith("黑猫") and "as (S3) " in ln for ln in d_lines)
    mommy = next(ln for ln in d_lines if "Mommy, daddy is so ugly" in ln)
    daddy = next(ln for ln in d_lines if "Daddy, why did you marry" in ln)
    # Facts layout: 白猫 left, 黑猫 right. Line-initial vocative only —
    # the "daddy" inside the Mommy line does not flip the facing.
    assert "turns to face 白猫 on the LEFT" in mommy
    assert "turns to face 黑猫" not in mommy
    assert "turns to face 黑猫 on the RIGHT" in daddy
    assert all(":" not in ln.split("<d>")[0] for ln in d_lines)
    mom_lines = [ln for ln in d_lines if ln.startswith("白猫")]
    assert mom_lines and all("turns to face" not in ln for ln in mom_lines)
    # 1 facts call + 1 prefix call + 2 shot calls; no extractor call
    # for turns (fast path).
    assert len(conn.calls) == 4
    # The LLM-harvested layout rides every shot user template.
    shot_user = [
        c[1]["content"] for c in conn.calls if "Clip duration" in c[1]["content"]
    ]
    assert len(shot_user) == 2
    for t in shot_user:
        assert "黑猫 stays at right of frame" in t
        assert "白猫 stays at left of frame" in t
        assert "小猫 stays at center of frame" in t


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


# --------------------------------------------------------------------------- #
# Label-policy tolerance — the 2026-09-21 20:33 live crash
# --------------------------------------------------------------------------- #
def test_label_policy_tolerates_unreferenced_identity_pictures(mods):
    _lg, lp = mods
    # A plan that references only Subjects/Pictures 1-3 while the
    # manifest carries five identity slots (4/5 = the duplicated
    # cat2.jpg figurine picture the concept never names).
    plan = {
        "prompt_prefix": ["style line"],
        "shots": [
            {"prompt": [
                "subject_definitions:",
                "<Subject 1> binds to <Picture 1>: brown tabby father cat.",
                "<Subject 2> binds to <Picture 2>: cream mother cat.",
                "<Subject 3> binds to <Picture 3>: ginger kitten.",
                "summary:", "[reference generation] ...",
                "retention_analysis:",
                "<Subject 1> -> <Picture 1>: fully_preserved - locked.",
                "<Subject 2> -> <Picture 2>: fully_preserved - locked.",
                "<Subject 3> -> <Picture 3>: fully_preserved - locked.",
                "detailed_description:", "the three cats talk",
                "overall_soundscape:", "room tone",
                "non_diegetic_music:", "No non-diegetic music.",
            ]},
        ],
    }
    # Concept names 图1-3 only -> Subjects 4/5 are optional.
    assert lp.validate_label_policy(
        plan, "ref2va", MANIFEST_5, referenced_pictures={1, 2, 3}
    ) == []
    # Strict callers (no referenced set) keep the old contract.
    strict = lp.validate_label_policy(plan, "ref2va", MANIFEST_5)
    assert any("Subject [4, 5] never appears" in e for e in strict)
    # A REQUIRED subject (named by the concept) that never appears
    # still fails.
    partial = lp.validate_label_policy(
        plan, "ref2va", MANIFEST_5, referenced_pictures={1, 2, 3, 4}
    )
    assert any("Subject [4] never appears" in e for e in partial)


def test_label_policy_waives_pairing_for_unreferenced_subjects(mods):
    """2026-09-21 20:58 live crash: the model voluntarily mentioned
    <Subject 4>/<Subject 5> in scene_02 WITHOUT their paired <Picture>
    labels; the pairing rule killed the finished 226s run. Mentions of
    Subjects bound to concept-unreferenced pictures are tolerated noise."""
    _lg, lp = mods
    plan = {
        "prompt_prefix": ["style line"],
        "shots": [
            {"prompt": [
                "subject_definitions:",
                "<Subject 1> binds to <Picture 1>: brown tabby father cat.",
                "<Subject 2> binds to <Picture 2>: cream mother cat.",
                "<Subject 3> binds to <Picture 3>: ginger kitten.",
                "<Subject 4>: figurine, not used by the story.",
                "summary:", "[reference generation] ...",
                "retention_analysis:",
                "<Subject 1> -> <Picture 1>: fully_preserved - locked.",
                "<Subject 2> -> <Picture 2>: fully_preserved - locked.",
                "<Subject 3> -> <Picture 3>: fully_preserved - locked.",
                "<Subject 4>: reference - not on screen.",
                "detailed_description:", "the three cats talk",
                "overall_soundscape:", "room tone",
                "non_diegetic_music:", "No non-diegetic music.",
            ]},
        ],
    }
    # <Subject 4> appears WITHOUT <Picture 4>: tolerated for an unused
    # picture; a REQUIRED subject missing its pair still fails.
    assert lp.validate_label_policy(
        plan, "ref2va", MANIFEST_5, referenced_pictures={1, 2, 3}
    ) == []
    referenced_broken = lp.validate_label_policy(
        plan, "ref2va", MANIFEST_5, referenced_pictures={1, 2, 3, 4}
    )
    assert any(
        "<Subject 4> needs its paired <Picture 4>" in e
        for e in referenced_broken
    )
    # Unpaired mention of a REQUIRED subject (2) still fails outright.
    broken = {
        "prompt_prefix": ["style line"],
        "shots": [
            {"prompt": [
                "subject_definitions:",
                "<Subject 1> binds to <Picture 1>: tabby.",
                "<Subject 2>: cream mother cat, no picture pair.",
                "summary:", "[reference generation] ...",
                "retention_analysis:",
                "<Subject 1> -> <Picture 1>: fully_preserved - locked.",
                "<Subject 2>: fully_preserved - locked.",
                "detailed_description:", "talk",
                "overall_soundscape:", "tone",
                "non_diegetic_music:", "No non-diegetic music.",
            ]},
        ],
    }
    errs = lp.validate_label_policy(
        broken, "ref2va", MANIFEST_5, referenced_pictures={1, 2, 3}
    )
    assert any("<Subject 2> needs its paired <Picture 2>" in e for e in errs)


def test_reference_directive_teaches_only_used_pictures(mods):
    _lg, lp = mods
    directive = lp.build_reference_directive(
        "ref2va", MANIFEST_5, 1, 5.0, referenced_pictures={1, 2, 3}
    )
    assert "do NOT reference" in directive
    assert "<Subject 1> -> <Picture 1>" in directive
    assert "<Subject 4>" not in directive.split("do NOT reference")[0]
    assert "<Subject 5>" not in directive.split("do NOT reference")[0]
    # Without the filter the directive keeps teaching all five slots.
    full = lp.build_reference_directive("ref2va", MANIFEST_5, 1, 5.0)
    assert "<Subject 4> -> <Picture 4>" in full
    assert "<Subject 5> -> <Picture 5>" in full


def test_role_declaration_binding_form_parses(mods):
    """The natural role-declaration phrasing
    ``图1的黑猫是爸爸，图2的白猫是妈妈`` must bind the NAME (黑猫), not
    swallow the role into the name (黑猫是爸爸) — the CAST override keys
    on the parsed name."""
    lg, _lp = mods
    text = (
        "图1的黑猫是爸爸，图2的白猫是妈妈，图3的小猫是女儿\n\n"
        "镜头是图2白猫的视角，对着图3的小猫和图1的黑猫\n\n"
        "图3的小猫：Mommy, daddy is so ugly, why did you marry him.\n"
        "图2的白猫：Iguess I was blind.\n"
        "图3的小猫：Daddy, why did you marry a blind lady?\n"
        "图2的白猫：Oh my god.\n"
        "图1的黑猫：Well, nobody's perfect."
    )
    # Dialogue: the two prose lines are the prologue, 5 speaker turns.
    turns, prologue = lg._dlg_parse_structured_dialogue_turns(text)
    assert turns is not None and len(turns) == 5
    assert "是爸爸" in prologue and "视角" in prologue
    # Bindings: exact names, first mention wins (declaration line).
    assert lg._concept_picture_bindings(text) == [
        ("黑猫", 1), ("白猫", 2), ("小猫", 3),
    ]
    assert sorted(lg._referenced_picture_numbers(text)) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# 2026-09-21 21:33 live output checks: turnaround-sheet captions polluted
# the identity lines ("seamless white studio background" inside living-
# room scenes) and the prefix pinned the OFF-SCREEN POV holder to center
# via a garbage spatial key ("的小猫=center of frame").
# --------------------------------------------------------------------------- #
def test_spatial_layout_strips_picture_prefix_fragments(mods):
    _lg, lp = mods
    # The form that produced the live garbage key: 图N的<name> with a
    # prefixed paren position. Also locks the CJK class widening —
    # 黑 (U+9ED1) sits above the old [一-龥] (U+9FA5) ceiling, so
    # 黑猫 never extracted at all before.
    out = lp.extract_spatial_layout(
        "图1的黑猫 (at left of frame), 图3的小猫（站center of frame）"
        "面对着镜头。"
    )
    assert out == {"黑猫": "left of frame", "小猫": "center of frame"}


def test_caption_prompt_describes_character_not_sheet():
    txt = (PROMPTS_DIR / "h3" / "caption_reference.txt").read_text(
        encoding="utf-8"
    )
    assert "NOT OF THE SHEET" in txt
    assert "seamless white background" in txt  # the forbidden vocabulary


def test_prefix_spatial_pin_uses_roster_names():
    txt = (PROMPTS_DIR / "h3_loop" / "prefix_synth_system.txt").read_text(
        encoding="utf-8"
    )
    assert "ROSTER NAMES" in txt
    assert "NO position pin" in txt          # off-screen POV holder
    assert '"Subject 1 stays' not in txt     # no Subject-number example


# --------------------------------------------------------------------------- #
# FULL-WORKFLOW E2E — mirrors the user's attached workflow (node 61) with
# its exact widget values: ref2va + 5 wired pictures (Pictures 4/5 the
# SAME image, like the duplicated cat2.jpg), auto-enhance ON, pacing
# normal, scene_count 0, per_shot, category none, output_language en,
# seed_mode per_scene_increment, temperature 0.4 / max_tokens 16384 /
# timeout 300, caption cache memory+disk. Locks down the whole chain:
# enhancer rewrite -> captions (incl. duplicate-image cache hit) ->
# structured parse (5 turns) -> minimal-cut packing (2 scenes) -> LLM
# prefix -> six-section shots (Subjects 1-3 only) -> label policy OK.
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402  (test-local, after module constants)

WF_RAW_INPUT = (
    "镜头从图2的白猫视角对着图3的小猫，图1的黑猫在图3的小猫旁边。\n"
    "图3的小猫：Mommy,daddy is so ugly,why did you marry him.\n"
    "图2的白猫：Iguess I was blind\n"
    "图3的小猫：daddy,why did you marry a blind lady\n"
    "图2的白猫：oh my god\n"
    "图1的黑猫：well nobody's perfect\n"
)

# What M3's auto-enhance actually returns for this draft (shape from the
# live log: rewritten 196 -> 302 chars; branch-D binding line + the 5
# English speaker lines kept verbatim).
WF_ENHANCED = (
    "Classification: Reference-driven\n"
    "Notes for the user: speaker names stable; binding line added.\n"
    "--- BEGIN user_input ---\n"
    "场景设定：温暖家庭客厅。参考图 1 → 黑猫（画面右边）；"
    "参考图 2 → 白猫（画面左边）；参考图 3 → 小猫（画面中间）。\n"
    "小猫：Mommy, daddy is so ugly, why did you marry him.\n"
    "白猫：Iguess I was blind.\n"
    "小猫：Daddy, why did you marry a blind lady?\n"
    "白猫：Oh my god.\n"
    "黑猫：Well, nobody's perfect.\n"
    "--- END user_input ---"
)

WF_CAPTIONS = [
    "brown_tabby_cat in a dark brown tweed blazer over a white shirt, "
    "chubby bipedal build, amber eyes",
    "cream_curly_kitten with a blonde hairdo, beige plaid jacket, "
    "white shirt, navy bow tie",
    "fluffy ginger kitten in yellow corduroy overalls, round build",
    "black cat figurine on a beige parquet floor, glossy ceramic finish",
    # slot 5 never consumed: identical pixels to slot 4 -> memory hit
]

WF_PREFIX = (
    "3D CG animation in a warm family living room.\n\n"
    "CAST:\n"
    "黑猫: brown tabby father cat in a tweed blazer.\n"
    "白猫: cream-furred mother cat in a plaid jacket.\n"
    "小猫: ginger kitten in yellow overalls.\n"
)


def _wf_shot_reply(subjects, body):
    lines = [
        "subject_definitions:",
        "<Subject 1> binds to <Picture 1>: brown tabby father cat.",
        "<Subject 2> binds to <Picture 2>: cream-furred mother cat.",
        "<Subject 3> binds to <Picture 3>: ginger kitten.",
        "",
        "summary:",
        "[reference generation] family exchange in the living room.",
        "",
        "retention_analysis:",
        "<Subject 1> -> <Picture 1>: fully_preserved - identity locked.",
        "<Subject 2> -> <Picture 2>: fully_preserved - identity locked.",
        "<Subject 3> -> <Picture 3>: fully_preserved - identity locked.",
        "",
        "detailed_description:",
        body,
        "",
        "overall_soundscape:",
        "Warm room tone carries across the boundary; close paw rustle.",
        "",
        "non_diegetic_music:",
        "No non-diegetic music.",
    ]
    return "\n".join(lines) + "\n"


def test_full_workflow_e2e_ref2va_english_dialogue(mods, monkeypatch, tmp_path):
    lg, _lp = mods
    monkeypatch.setattr(
        lg, "_caption_cache_disk_root", lambda: str(tmp_path)
    )

    # 5-image batch (N,H,W,C); slots 4 and 5 carry IDENTICAL pixels
    # (the workflow wires cat2.jpg twice via LoadImage nodes 59/60).
    # Gradient content, NOT flat constants — JPEG quantisation collapses
    # near-black constant frames to identical bytes, which would make
    # every slot's cache key collide (real photos never do).
    def _img(seed_v):
        rng = np.random.default_rng(seed_v)
        return rng.random((8, 8, 3), dtype=np.float32)

    images = np.stack([_img(1), _img(2), _img(3), _img(4), _img(4)])

    # Exact LLM call sequence this widget set produces:
    #   1 auto-enhance, 4 caption misses (5th is a memory-cache hit),
    #   1 semantic-facts call (canonical turns from the fast path; the
    #   concept's 图1-3 mentions fire the facts gate), 1 prefix (5
    #   turns >= LLM-prefix threshold), 2 shot calls.
    WF_FACTS = json.dumps({
        "turns": [],
        "layout": [
            {"name": "黑猫", "position": "right of frame"},
            {"name": "白猫", "position": "left of frame"},
            {"name": "小猫", "position": "center of frame"},
        ],
        "bindings": [
            {"picture": 1, "name": "黑猫"},
            {"picture": 2, "name": "白猫"},
            {"picture": 3, "name": "小猫"},
        ],
    }, ensure_ascii=False)
    conn = _Conn([
        WF_ENHANCED,
        *WF_CAPTIONS[:4],
        WF_FACTS,
        WF_PREFIX,
        _wf_shot_reply(
            [1, 2, 3],
            "[Shot 1] The kitten asks the mother; the mother answers; "
            "the kitten presses the question.",
        ),
        # scene_02 replays the 2026-09-21 20:58 live failure: the model
        # VOLUNTARILY mentions the unused Subjects 4/5 without pairing
        # them with <Picture 4>/<Picture 5> — the run must still pass.
        _wf_shot_reply(
            [1, 2, 3],
            "[Shot 1] The mother gasps and the father delivers the "
            "closing line. <Subject 4>: reference - not on screen. "
            "<Subject 5>: reference - not on screen.",
        ),
    ])

    out = lg.H3LoopPromptEnhancer(conn, temperature=0.4, max_tokens=16384,
                                  timeout=300)(
        user_input=WF_RAW_INPUT,
        enhance_user_input=True,          # on - 自动润色后再规划
        seed=199573759057408,
        scene_count=0,
        total_duration_seconds=0,
        pacing="normal - 正常（语速·推荐）",
        generation_mode="per_shot - 逐场生成(推荐)",
        category="none - 不指定",
        output_language="en",
        seed_mode="per_scene_increment - 每场seed递增(推荐)",
        reference_mode="ref2va - 参考图(N张/全场景)",
        caption_mode="cache_memory_disk - 缓存:内存+磁盘(推荐)",
        images=images,
    )

    # ── LLM spend is exactly the scripted sequence: the duplicated
    #    picture was served from the memory cache, not re-captioned.
    assert len(conn.calls) == 9, (
        f"expected 9 LLM calls (enhance+4 captions+facts+prefix+2 "
        f"shots); got {len(conn.calls)}"
    )

    plan = json.loads(out["plan_json"])
    assert set(plan.keys()) == {"shots", "prompt_prefix"}
    assert out["board_kind"] == "dialogue"
    assert len(plan["shots"]) == 2
    for s in plan["shots"]:
        assert set(s.keys()) == {"id", "prompt", "length", "seed"}
        assert 96 <= s["length"] <= 345  # 4s floor .. 14s cap on the grid
    # per_scene_increment seeds from the workflow's base. Note the base
    # wraps modulo 10**12 by design (derive_seed_base keeps per-scene
    # increments inside the uint64 digit-string headroom).
    wf_base = 199573759057408 % 10**12
    assert [s["seed"] for s in plan["shots"]] == [
        str(wf_base + 1), str(wf_base + 2),
    ]

    all_text = "\n".join("\n".join(s["prompt"]) for s in plan["shots"])
    # Dialogue: 5 verbatim English lines, S1=小猫 S2=白猫 S3=黑猫,
    # no phantom setting speaker, no Chinese speech block.
    d_lines = [ln for ln in all_text.split("\n") if "<d>" in ln]
    assert len(d_lines) == 5
    assert all("<d>[English]" in ln for ln in d_lines)
    assert "<d>[Chinese]" not in all_text
    assert "场景设定" not in all_text
    assert any(ln.startswith("小猫") and "as (S1) " in ln for ln in d_lines)
    assert any(ln.startswith("白猫") and "as (S2) " in ln for ln in d_lines)
    assert any(ln.startswith("黑猫") and "as (S3) " in ln for ln in d_lines)
    assert all(":" not in ln.split("<d>")[0] for ln in d_lines)
    # Scene split: [t1-3] then [t4-5].
    assert sum(1 for ln in plan["shots"][0]["prompt"] if "<d>" in ln) == 3
    assert sum(1 for ln in plan["shots"][1]["prompt"] if "<d>" in ln) == 2
    # Six-section schema everywhere (ref2va).
    for s in plan["shots"]:
        for header in ("subject_definitions:", "summary:",
                       "retention_analysis:", "detailed_description:",
                       "overall_soundscape:", "non_diegetic_music:"):
            assert header in s["prompt"]
    # Subjects 4/5 (the duplicated figurine picture) are NOT TAUGHT by
    # the directive; scene_02's voluntary unpaired mentions are
    # tolerated; the plan still validates (both post-spend hard fails).
    shot_user_texts = [c[1]["content"] for c in conn.calls]
    assert all("do NOT reference" in t for t in shot_user_texts[-2:])
    assert all("<Subject 4> -> <Picture 4>" not in t for t in shot_user_texts[-2:])
    # The description body is node-owned speech, so a Subject mention
    # that lived only in the model's beat prose is not kept. The run
    # still returns a plan (validation did not die on that mention).
    assert "<Subject 4>" not in "\n".join(plan["shots"][1]["prompt"])

    # ── Caption-grounded CAST: the <d> speaker identity lines follow
    #    the BOUND PICTURE's caption, never the name's literal meaning
    #    (the 2026-09-21 split-identity failure: 黑猫's line said
    #    "black short-haired" while Picture 1 is a brown tabby).
    s1_text = "\n".join(plan["shots"][0]["prompt"])
    s2_text = "\n".join(plan["shots"][1]["prompt"])
    # 黑猫 (bound to Picture 1 = brown tabby caption) speaks in scene_02.
    # Caption pin lives under subject_definitions, not on the <d> line.
    hei_pin = next(ln for ln in s2_text.split("\n") if ln.startswith("黑猫:"))
    assert "brown_tabby_cat" in hei_pin
    assert "black" not in hei_pin.lower()
    assert "<d>" not in hei_pin
    hei_speech = next(
        ln for ln in s2_text.split("\n") if "<d>" in ln and "黑猫" in ln
    )
    assert "brown_tabby_cat" not in hei_speech
    bai_pin = [ln for ln in s1_text.split("\n") if ln.startswith("白猫:")]
    assert bai_pin and all("cream" in ln.lower() for ln in bai_pin)
    bai_speech = [ln for ln in s1_text.split("\n") if "<d>" in ln and "白猫" in ln]
    assert bai_speech and all("cream_curly" not in ln for ln in bai_speech)
    # The override is surfaced in the summary.
    assert "identity pin" in out["summary"]

    # ── Summary: the guardrails that must be loud, not fatal.
    pre = out["summary"]
    assert "category: none -> dialogue" in pre          # auto-upgrade note
    assert "never referenced" in pre                    # Pictures 4/5
    assert "黑猫" in pre and "Picture 1" in pre         # colour contradiction


# --------------------------------------------------------------------------- #
# Semantic-facts refactor (2026-09-21 review): layout / bindings are
# LLM-judged + mechanically verified; regex extractors demoted to
# fallback.
# --------------------------------------------------------------------------- #
def test_normalize_fact_position(mods):
    _lg, lp = mods
    n = lp.normalize_fact_position
    assert n("left of frame") == "left of frame"
    assert n("画面右侧") == "right of frame"
    assert n("中间") == "center of frame"
    assert n("Center of the Frame.") == "center of frame"
    # Relative anchors survive with their reference target.
    assert n("beside 小猫") == "beside 小猫"
    assert n("next to the kitten") == "beside the kitten"
    assert n("left of 小猫") == "left of 小猫"
    assert n("somewhere nice") is None
    assert n("") is None


def test_harvest_semantic_facts_drops_unverified(mods):
    _lg, lp = mods
    concept = "图1的黑猫在图3的小猫旁边。黑猫坐在左边。"
    facts = lp.harvest_semantic_facts({
        "layout": [
            {"name": "黑猫", "position": "beside 小猫"},
            {"name": "小猫", "position": "center of frame"},
            {"name": "幽灵猫", "position": "left of frame"},  # not in text
            {"name": "黑猫", "position": "right of frame"},   # first wins
        ],
        "bindings": [
            {"picture": 1, "name": "黑猫"},
            {"picture": 3, "name": "小猫"},
            {"picture": 12, "name": "黑猫"},                  # slot out of range
            {"picture": 2, "name": "幽灵猫"},                 # name not in text
            {"picture": 2, "name": "黑猫"},                   # dup name ignored
        ],
    }, concept)
    assert facts["layout"] == {
        "黑猫": "beside 小猫", "小猫": "center of frame",
    }
    assert facts["bindings"] == [("黑猫", 1), ("小猫", 3)]


def test_extract_board_facts_only_mode_ignores_turn_quality(mods):
    """want_turns=False (fast-path boards): garbage turns never trigger
    retries — the call exists purely for the facts."""
    lg, _lp = mods
    concept = "场景设定：客厅，黑猫在左边。\n甲猫：你好。\n乙猫：再见。"
    reply = json.dumps({
        "turns": [{"speaker": "nonsense",
                   "lines": [{"text": "span junk", "start": 99, "end": 1}]}],
        "layout": [{"name": "黑猫", "position": "left of frame"}],
        "bindings": [],
    }, ensure_ascii=False)
    conn = _Conn([reply])
    turns, facts = lg.H3LoopPromptEnhancer(conn).extract_board_facts(
        concept, want_turns=False
    )
    assert turns == []
    assert facts["layout"] == {"黑猫": "left of frame"}
    assert len(conn.calls) == 1  # no span-validation retry


def test_facts_gate_skips_call_without_candidates(mods):
    """Canonical boards with no 图N / positional language skip the
    facts call entirely (cost + stub-ordering contract)."""
    lg, _lp = mods
    plain = "甲猫：你好。\n乙猫：好的。"
    # Direct gate check.
    assert not lg._FACT_CANDIDATE_RE.search(plain)
    assert lg._FACT_CANDIDATE_RE.search(CONCEPT)  # 参考图 N mentions
    # Pipeline-level: a canonical no-candidate board makes NO facts
    # call (stub = prefix + shots only, as before the refactor).
    prefix = "Style.\n\nCAST:\n甲猫: x.\n乙猫: y.\n"
    shot = ("integrated_multimodal_description:\n[Shot 1] ok.\n\n"
            "overall_soundscape:\ntone\n\n"
            "non_diegetic_music:\nNo non-diegetic music.\n")
    conn = _Conn([prefix, shot])
    lg.H3LoopPromptEnhancer(conn, temperature=0.4, timeout=60)(
        user_input=plain, seed=1,
    )
    assert len(conn.calls) == 2


# --------------------------------------------------------------------------- #
# 2026-09-21 23:02 output review: prefix asserted "dialogue spoken in
# Chinese" on an all-English board, and the captioner dodged the sheet
# vocabulary ban with "shown from side, front, and back angles" +
# composition framing + sheet lighting. Mechanical backstops.
# --------------------------------------------------------------------------- #
def test_scrub_prefix_dialogue_language(mods):
    lg, _lp = mods
    lines = lg.scrub_prefix_dialogue_language([
        "Style line. No on-screen text, no logos; dialogue spoken in "
        "Chinese.",
        "Speech is delivered in English throughout.",
        "spoken lines are spoken in Mandarin",
        "Untouched line about palette and tempo.",
    ])
    assert "Chinese" not in lines[0]
    assert "Style line" in lines[0] and "logos" in lines[0]
    assert "English" not in lines[1]
    # A line that was ONLY the assertion is dropped outright.
    assert not any("Mandarin" in ln for ln in lines)
    assert len(lines) == 3
    assert lines[-1].startswith("Untouched")


def test_scrub_caption_sheet_language_live_phrases(mods):
    lg, _lp = mods
    s = lg.scrub_caption_sheet_language
    # Live leak 1: multi-view clause inside an identity sentence.
    out = s(
        "A fluffy anthropomorphic cream-and-white cat with curly wavy "
        "fur, pink inner ears, large expressive brown eyes, and a bushy "
        "tail stands upright on two white paws, shown from side, front, "
        "and back angles. The cat wears a beige glen plaid blazer."
    )
    assert "shown from" not in out and "angles" not in out
    assert "curly wavy fur" in out and "glen plaid blazer" in out
    # Live leak 2: composition framing + sheet lighting tail.
    out = s(
        "Palette stays muted earth tones (gray-brown fur, taupe-brown "
        "jacket, ivory shirt, tan tie) under neutral even lighting, "
        "framing the cat centered and occupying most of the frame so "
        "the face, tie knot, paw placement, and tail silhouette are "
        "the lockable identity cues for every shot."
    )
    assert "framing" not in out and "occupying" not in out
    assert "under neutral" not in out
    assert "gray-brown fur" in out          # the palette facts survive
    # Live leak 3: sheet lighting + ground shadow + stance.
    out = s(
        "A fluffy kitten in yellow corduroy overalls lit by bright even "
        "frontal lighting that casts a soft shadow beneath the paws, "
        "standing on all four feet, with amber eyes."
    )
    assert "frontal lighting" not in out
    assert "shadow beneath" not in out
    assert "all four" not in out
    assert "yellow corduroy overalls" in out and "amber eyes" in out
    # Older leak forms (turnaround/three views/white background).
    out = s(
        "A tabby cat in a tweed blazer, three views on a seamless "
        "white background, turnaround sheet."
    )
    assert "three views" not in out and "white background" not in out
    assert "turnaround" not in out and "tweed blazer" in out
    # Never returns empty.
    assert s("shown from three angles") == "shown from three angles"


def test_shot_addendum_has_no_chinese_default():
    txt = (PROMPTS_DIR / "h3_loop" / "shot_system_addendum.txt").read_text(
        encoding="utf-8"
    )
    assert "default spoken lines to Chinese" not in txt


def test_prefix_system_never_states_dialogue_language():
    txt = (PROMPTS_DIR / "h3_loop" / "prefix_synth_system.txt").read_text(
        encoding="utf-8"
    )
    assert "NEVER states a spoken-dialogue language" in txt


# --------------------------------------------------------------------------- #
# 2026-09-22 C2 live failure: 妈妈's two lines (one per scene) had two
# different TTS voices — scenes generate independently and no voice
# descriptor rode the tags. Voice sheet + per-scene re-attachment.
# --------------------------------------------------------------------------- #
def test_default_voice_for_is_stable_and_gendered(mods):
    _lg, lp = mods
    v = lp.default_voice_for
    assert v("妈妈") == v("妈妈") and "female" in v("妈妈")
    assert "male" in v("爸爸") and "female" not in v("爸爸")
    assert "female" in v("mother cat")
    assert "child" not in v("妈妈")  # heuristic is adult-tier; fine
    # Unknown names get a STABLE neutral descriptor (stability is the
    # feature — cross-scene consistency).
    assert v("阿猫") == v("阿猫") and "mid-range" in v("阿猫")


def test_voice_sheet_parsed_from_prefix_reply(mods):
    _lg, lp = mods
    raw = (
        "Style paragraph.\n\nCAST:\n妈妈: cream cat.\n爸爸: tabby cat.\n\n"
        "VOICE:\n妈妈: adult female, warm mid-range pitch, soft timbre\n"
        "爸爸: adult male, low pitch, gravel timbre\n"
    )
    lines, cast, voices = lp.split_prefix_and_cast(raw), None, None
    # via split_prefix_sections:
    _, _, cast, voices = lp.split_prefix_sections(raw)
    assert cast["妈妈"] == "cream cat."
    assert voices["妈妈"].startswith("adult female")
    assert voices["爸爸"].startswith("adult male")
    # legacy wrapper still works
    l2, c2 = lp.split_prefix_and_cast(raw)
    assert l2 and c2["妈妈"] == "cream cat."


def test_blocks_carry_voice_per_scene(mods):
    """Each spoken line is one sentence: ``as (Sn) <descriptor> <d>``.

    The descriptor sits immediately before the tag, with no colon
    between ``(Sn)`` and ``<d>`` (C4 put it there and the line was
    lip-synced to the wrong character). CAST identity is a separate
    line on the speaker's first clip and carries no ``<d>``.
    """
    _lg, lp = mods
    blocks = lp.assemble_dialogue_line_blocks(
        ["Hello.", "World."],
        line_speakers=["妈妈", "妈妈"],
        speaker_id_map={"妈妈": "S1"},
        speaker_identities={"妈妈": "cream cat caption"},
        first_appearance_speakers={"妈妈"},
        speaker_voices={"妈妈": "adult female, warm mid-range pitch"},
    )
    assert blocks == [
        "妈妈 speaks as (S1) adult female, warm mid-range pitch "
        "<d>[English] Hello.</d>",
        "妈妈 speaks as (S1) adult female, warm mid-range pitch "
        "<d>[English] World.</d>",
    ]
    assert "cream cat caption" not in "\n".join(blocks)
    blocks2 = lp.assemble_dialogue_line_blocks(
        ["Oh my god."],
        line_speakers=["妈妈"],
        speaker_id_map={"妈妈": "S1"},
        speaker_identities={"妈妈": "cream cat caption"},
        first_appearance_speakers=set(),
        speaker_voices={"妈妈": "adult female, warm mid-range pitch"},
    )
    assert blocks2[0] == (
        "妈妈 speaks as (S1) adult female, warm mid-range pitch "
        "<d>[English] Oh my god.</d>"
    )
    assert blocks2[0].split("<d>")[0].endswith(
        "as (S1) adult female, warm mid-range pitch "
    )


def test_voice_performance_directive_in_template(mods):
    """The prose-side binding: every shot user template carries the
    per-speaker performance phrases with exact descriptors; later
    scenes restate them (scenes generate independently)."""
    _lg, lp = mods
    base = dict(
        concept="c", prefix_text="p", category="none",
        continuation_block="", shot={"id": "s", "description": "d"},
        clip_index=2, clip_count=2, duration_seconds=5.0,
        language_name="en",
        speaker_id_map={"妈妈": "S1"},
        speaker_voices={"妈妈": "adult female, warm mid-range pitch"},
    )
    t = lp.build_shot_user_text(
        dialogue_lines=["Oh my god."], line_speakers=["妈妈"],
        turn_speaker="妈妈", **base,
    )
    assert "Voice performance binding" in t
    assert "- 妈妈 as (S1) adult female, warm mid-range pitch" in t
    # Heuristic fallback when no sheet entry (unknown name).
    t2 = lp.build_shot_user_text(
        dialogue_lines=["Hi."], line_speakers=["阿猫"],
        turn_speaker="阿猫", **{**base, "speaker_id_map": {"阿猫": "S2"},
                                 "speaker_voices": {}},
    )
    assert "- 阿猫 as (S2) adult, mid-range pitch, natural timbre" in t2
    # No dialogue -> no directive.
    t3 = lp.build_shot_user_text(
        dialogue_lines=None, turn_speaker="", **base,
    )
    assert "Voice performance binding" not in t3
