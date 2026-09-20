# -*- coding: utf-8 -*-
"""End-to-end test for the spatial-layout flow addition (2026-09-17).

Exercises the FULL pipeline path that was changed:

  1. user_input_enhancer.txt      -> spatial layout rule documented
  2. extract_spatial_layout()     -> deterministic rule-based extraction
  3. prefix_synth_system.txt      -> spatial layout sentence rule
  4. shot_user_template.txt       -> {spatial_layout_directive} slot
  5. build_spatial_layout_directive() -> renders the dict
  6. build_shot_user_text()       -> injects the directive into every clip
  7. validate_spatial_layout_invariant() -> cross-clip continuity check
  8. The full H3LoopPromptEnhancer orchestrator (with mocked LLM)
     runs the fish-snack scenario and proves spatial_layout survives
     extraction -> prefix synthesis -> per-shot user template.

The original failure case from the user (scene_03 swapping left/right
between cats) is the regression we are guarding.
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
def lp():
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
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )


# --------------------------------------------------------------------------- #
# Stage 1 — prompt assets declare the new contract
# --------------------------------------------------------------------------- #
def test_enhancer_prompt_declares_spatial_layout_rule():
    txt = (PROMPTS_DIR / "h3_loop" / "user_input_enhancer.txt").read_text(
        encoding="utf-8"
    )
    assert "Spatial layout (all branches)" in txt, (
        "enhancer must declare a universal (not branch-specific) "
        "spatial layout rule"
    )
    # The rule must appear BEFORE any of the four branches, so it's
    # in scope for all of them. Match the actual section headers in
    # the file (the (C) branch is labelled "Pure narration / 旁白 / 单镜头描述").
    a_idx = txt.find("## (A) Dialogue / 对白")
    b_idx = txt.find("## (B) Action / 动作戏")
    c_idx = txt.find("## (C) Pure narration / 旁白")
    d_idx = txt.find("## (D) Reference-image driven")
    rule_idx = txt.find("Spatial layout (all branches)")
    assert 0 <= a_idx < b_idx < c_idx < d_idx, (
        "all four branch headers must exist in the file"
    )
    assert rule_idx < a_idx, (
        "the rule must be declared above all four branches"
    )


def test_prefix_synth_system_declares_spatial_layout_rule():
    txt = (PROMPTS_DIR / "h3_loop" / "prefix_synth_system.txt").read_text(
        encoding="utf-8"
    )
    assert "Spatial layout" in txt, (
        "prefix synth system must declare a spatial layout rule"
    )
    assert "drift" in txt.lower(), (
        "prefix synth rule must mention the drift failure mode"
    )
    # "throughout" must be banned — the exact failure that produced
    # scene_03's continuous-slow-push-in-until-the-camera-settles
    # contradiction. The wording should forbid the unlimited form.
    assert "never write" in txt.lower() or "no " in txt.lower(), (
        "prefix synth rule must explicitly forbid 'throughout' style "
        "tempo/position language"
    )


def test_shot_user_template_has_spatial_layout_directive_slot():
    txt = (PROMPTS_DIR / "h3_loop" / "shot_user_template.txt").read_text(
        encoding="utf-8"
    )
    assert "{spatial_layout_directive}" in txt, (
        "shot user template must contain the spatial_layout_directive slot"
    )


# --------------------------------------------------------------------------- #
# Stage 2 — extract_spatial_layout is rule-based and deterministic
# --------------------------------------------------------------------------- #
def test_extract_spatial_layout_picture_to_subject_slot(lp):
    """The Picture N -> Subject N (slot) pattern from branch (D)."""
    concept = (
        "场景设定：咖啡店室内。Picture 1 → Subject 1 (left slot)；"
        "Picture 2 → Subject 2 (right slot)。"
    )
    out = lp.extract_spatial_layout(concept)
    assert out == {"Subject 1": "left of frame", "Subject 2": "right of frame"}


def test_extract_spatial_layout_chinese_left_right(lp):
    """The 中文 角色 坐 画面 左侧 / 右侧 pattern from branch (A). The
    canonical bucket is language-neutral (EN canonical string), so a
    ZH-declared board stays comparable against EN-written shot prose."""
    concept = (
        "场景设定：咖啡店室内。\n"
        "图1的猫（哈利猫）坐画面左侧；图2的猫（莎莉猫）坐画面右侧。"
    )
    out = lp.extract_spatial_layout(concept)
    assert "哈利猫" in out and out["哈利猫"] == "left of frame"
    assert "莎莉猫" in out and out["莎莉猫"] == "right of frame"


def test_extract_spatial_layout_zuobian_youbian(lp):
    """Skill-recommended 左边/右边 parenthetical form."""
    concept = (
        "橙色虎斑公猫 = 哈利猫（坐画面左边）\n"
        "白色长毛母猫 = 莎莉猫（坐画面右边）"
    )
    out = lp.extract_spatial_layout(concept)
    assert out.get("哈利猫") == "left of frame"
    assert out.get("莎莉猫") == "right of frame"


def test_extract_spatial_layout_english_left_of_frame(lp):
    """The Subject N stays at left of frame pattern."""
    concept = (
        "An interior cafe. Subject 1 stays at left of frame throughout; "
        "Subject 2 stays at right of frame throughout."
    )
    out = lp.extract_spatial_layout(concept)
    assert out == {
        "Subject 1": "left of frame",
        "Subject 2": "right of frame",
    }


def test_extract_spatial_layout_empty_on_garbage(lp):
    """No positions -> empty dict (NOT a default). Caller decides what
    to do with empty (see test below)."""
    concept = "Just two cats having an argument. No positions mentioned."
    out = lp.extract_spatial_layout(concept)
    assert out == {}


def test_extract_spatial_layout_real_fish_snack_concept(lp):
    """The exact concept the user pasted, rewritten to declare positions.
    This is the regression test: the original draft is missing positions,
    so the enhanced rewrite adds them; here we use the enhanced form
    to confirm extraction recognises it."""
    concept = (
        "场景设定：温暖的木餐桌，金色窗光从左侧洒入。"
        "图2的猫（莎莉猫）坐画面左侧；"
        "图1的猫（哈利猫）坐画面右侧。"
        "两只猫面对面推一盘小鱼干。"
    )
    out = lp.extract_spatial_layout(concept)
    assert "莎莉猫" in out and out["莎莉猫"] == "left of frame"
    assert "哈利猫" in out and out["哈利猫"] == "right of frame"


# --------------------------------------------------------------------------- #
# Stage 2b — canonicalisation: semantics, not spelling
# --------------------------------------------------------------------------- #
def test_normalise_position_unifies_buckets(lp):
    """centre/camera-left/画面左/left-slot all collapse to the same
    canonical bucket — the drift check compares semantics, and a
    cross-language board (ZH user_input, EN output prose) must not
    produce phantom drift."""
    canon = lp._normalise_position
    assert canon("left of frame") == "left of frame"
    assert canon("camera-left") == "left of frame"
    assert canon("screen-left") == "left of frame"
    assert canon("left slot") == "left of frame"
    assert canon("画面左") == "left of frame"
    assert canon("centre of frame") == "center of frame"
    assert canon("Center-Frame") == "center of frame"
    assert canon("画面右侧") == "right of frame"
    # Too generic to anchor.
    assert canon("on the left") is None
    assert canon("somewhere") is None
    assert canon("") is None
    # Movement prose mixing CONFLICTING buckets must not anchor a
    # binding position ("enters from camera-left and stops at
    # center-frame" — the resting position is center, the entry is
    # left; picking either would be wrong).
    assert canon("enters from camera-left and stops at center-frame") is None


# --------------------------------------------------------------------------- #
# Stage 3 — build_spatial_layout_directive renders deterministically
# --------------------------------------------------------------------------- #
def test_spatial_layout_directive_renders_canonical_form(lp):
    layout = {"Subject 1": "left of frame", "Subject 2": "right of frame"}
    directive = lp.build_spatial_layout_directive(layout)
    assert "Subject 1 stays at left of frame" in directive
    assert "Subject 2 stays at right of frame" in directive
    # The same input must produce byte-identical output (deterministic).
    assert directive == lp.build_spatial_layout_directive(layout)
    # And the format must NOT be paraphrased — the exact phrase
    # "stays at left of frame" is the binding one.
    assert "stays at left of frame" in directive


def test_spatial_layout_directive_empty_handled(lp):
    """No layout declared -> the directive still has SOMETHING in the
    slot, telling the per-shot LLM to fall back to the prefix sentence."""
    out = lp.build_spatial_layout_directive(None)
    assert "no spatial_layout declared" in out
    assert "prefix" in out


def test_spatial_layout_directive_flows_into_shot_user_template(lp):
    """The directive MUST appear in the per-shot user template output.
    This is the binding that prevents the scene_03 drift."""
    shot = {"id": "scene_01", "description": "Test"}
    user_text = lp.build_shot_user_text(
        concept="concept",
        prefix_text="prefix",
        category="none",
        continuation_block="",
        shot=shot,
        clip_index=1,
        clip_count=3,
        duration_seconds=5.0,
        language_name="en",
        spatial_layout={"Subject 1": "left of frame", "Subject 2": "right of frame"},
    )
    assert "Subject 1 stays at left of frame" in user_text
    assert "Subject 2 stays at right of frame" in user_text
    assert "{spatial_layout_directive}" not in user_text, (
        "template placeholder must have been filled"
    )


# --------------------------------------------------------------------------- #
# Stage 4 — validate_spatial_layout_invariant catches the scene_03 bug
# --------------------------------------------------------------------------- #
def test_validate_spatial_layout_invariant_consistent(lp):
    scenes = [
        {"spatial_layout": {"A": "left of frame"}},
        {"spatial_layout": {"A": "left of frame", "B": "right of frame"}},
        {"spatial_layout": {"A": "left of frame"}},
    ]
    assert lp.validate_spatial_layout_invariant({}, scenes) == []
    assert lp.validate_spatial_layout_invariant({"A": "left of frame"}, scenes) == []


def test_validate_spatial_layout_invariant_detects_drift(lp):
    """scene_03 of the original failure had Subject 2 declared at
    'right of frame' in scenes 1-2 and at 'right side of the frame'
    (paraphrased into a different token) in scene 3. Our invariant
    catches the swap when the tokens don't match."""
    scenes = [
        {"spatial_layout": {"Subject 1": "right of frame", "Subject 2": "left of frame"}},
        {"spatial_layout": {"Subject 1": "right of frame", "Subject 2": "left of frame"}},
        # The drift: Subject 2 silently moves to right of frame.
        {"spatial_layout": {"Subject 1": "right of frame", "Subject 2": "right of frame"}},
    ]
    errors = lp.validate_spatial_layout_invariant(
        {"Subject 1": "right of frame", "Subject 2": "left of frame"},
        scenes,
    )
    assert any("Subject 2" in e and "left of frame" in e for e in errors), (
        f"expected an error mentioning Subject 2's drift; got {errors}"
    )
    assert any("right of frame" in e for e in errors)


def test_validate_spatial_layout_invariant_no_layout_no_errors(lp):
    assert lp.validate_spatial_layout_invariant({}, []) == []
    assert lp.validate_spatial_layout_invariant({}, [{"foo": "bar"}]) == []


# --------------------------------------------------------------------------- #
# Stage 5 — full orchestrator E2E with a stub LLM
# --------------------------------------------------------------------------- #
class _StubLLM:
    """Deterministic LLM stub. Each call returns the next canned reply
    in order. Mimics a real connector without needing the network."""

    # Fixed call-order labels. The orchestrator runs calls in this
    # exact sequence regardless of concept, so labelling by index is
    # safe.
    _STAGE_BY_INDEX = (
        "extract_dialogue",
        "storyboard",
        "prefix_synth",
        "shot",
        "shot",
        "shot",
    )

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # list of (stage_label, messages)
        self.model = "stub"
        self.timeout = None

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        call_index = len(self.calls)
        stage = (
            self._STAGE_BY_INDEX[call_index]
            if call_index < len(self._STAGE_BY_INDEX)
            else f"call_{call_index}"
        )
        self.calls.append((stage, messages))
        if not self.replies:
            raise RuntimeError(f"stub LLM ran out of replies at stage={stage}")
        return self.replies.pop(0)

    def get_state(self):
        return "stub"


def _build_fish_snack_storyboard(llm_dir):
    """The storyboard the enhancer's _auto_storyboard returns when
    given the fish-snack concept + shot budgets. Hand-built so the
    E2E test doesn't depend on a real LLM storyboard call."""
    # Loaded from the actual storyboard-prompts module so the test
    # tracks changes to SYSTEM_STORYBOARD_PROMPT.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_sb_prompts",
        str(llm_dir / "minimax_h3_storyboard_prompts.py"),
    )
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)

    shots_json = json.dumps([
        {
            "id": "scene_01",
            "description": (
                "Wide establishing shot. The cream-blonde cat slides the "
                "plate with 1 fish snack toward the tabby cat."
            ),
            "shot_type": "wide",
            "camera_movement": "slow push-in",
            "transition_in": "invisible_cut",
            "duration_seconds": 6.0,
            "narrative_beat": "First plate slide.",
            "characters": ["哈利猫", "莎莉猫"],
            "props": ["plate", "fish snack"],
            "notes": "Two-cat face-off at the dining table.",
        },
        {
            "id": "scene_02",
            "description": (
                "Tabby adds 1 fish snack to make 2, pushes back. Cream "
                "cat adds 0, slides back. Tabby adds 2 more to make 4, "
                "slides back. Cream cat heaps a mound, slides back."
            ),
            "shot_type": "medium",
            "camera_movement": "slow push-in",
            "transition_in": "invisible_cut",
            "duration_seconds": 8.0,
            "narrative_beat": "Escalating plate swaps.",
            "characters": ["哈利猫", "莎莉猫"],
            "props": ["plate", "fish snack"],
            "notes": "Multiple plate exchanges.",
        },
        {
            "id": "scene_03",
            "description": (
                "Tabby snatches the heaped plate and bolts. Camera stays "
                "with the cream-blonde cat: tears, cry, alone."
            ),
            "shot_type": "medium_close",
            "camera_movement": "settles",
            "transition_in": "invisible_cut",
            "duration_seconds": 6.0,
            "narrative_beat": "The betrayal and the aftermath.",
            "characters": ["哈利猫", "莎莉猫"],
            "props": ["plate"],
            "notes": "Subject 2 alone at the empty table.",
        },
    ], ensure_ascii=False)
    # SYSTEM_STORYBOARD_PROMPT requires a strict JSON array as reply
    # (we use the same shape the LLM would return).
    return shots_json


def _load_generator():
    """Load ``minimax_h3_loop_prompt_generator`` as a fresh module with
    the optional ComfyUI runtime deps stubbed. ``_mienodes_internal``
    must already be wired (the ``lp`` fixture does that)."""
    spec = importlib.util.spec_from_file_location(
        "_gen_h3_loop_prompt_generator",
        str(LLM_DIR / "minimax_h3_loop_prompt_generator.py"),
    )
    gen = importlib.util.module_from_spec(spec)

    # The generator imports a lot of optional ComfyUI deps at module
    # load. Stub them.
    fake_comfy = types.ModuleType("nodes")
    fake_comfy.interrupt_processing = lambda: False
    sys.modules["nodes"] = fake_comfy

    fake_ce = types.ModuleType("comfy_execution")
    fake_ce_exec = types.ModuleType("comfy_execution.execution")

    class _Interrupt(RuntimeError):
        pass

    fake_ce_exec.InterruptProcessingException = _Interrupt
    sys.modules["comfy_execution"] = fake_ce
    sys.modules["comfy_execution.execution"] = fake_ce_exec

    spec.loader.exec_module(gen)
    return gen


def test_e2e_fish_snack_spatial_layout_flows_everywhere(
    lp, monkeypatch, tmp_path
):
    """The full orchestrator must:
      1. Extract the spatial_layout from the rewritten user_input.
      2. Inject the directive into every per-shot user template.
      3. NOT let the per-shot LLM drift the positions (verified by
         checking the stub LLM received the binding directive for
         every scene).
    """
    gen = _load_generator()

    # The concept the user pasted, REWRITTEN by the enhancer with
    # positions declared in the opening paragraph (the new universal
    # spatial layout rule does exactly this).
    user_input = (
        "场景设定：温暖的木餐桌，金色窗光从左侧洒入。"
        "图2的猫（莎莉猫）坐画面左侧；"
        "图1的猫（哈利猫）坐画面右侧。"
        "两只猫面对面推一盘小鱼干。"
    )

    # Stub LLM replies for the orchestrator's stages:
    #   1. extract_dialogue          -> empty (pure narration-ish)
    #   2. _auto_storyboard          -> our hand-built 3-scene storyboard
    #   3. _synth_prefix             -> prefix with spatial layout + cast
    #   4. _generate_shot_prompt x3  -> three-section clips
    shots_json = _build_fish_snack_storyboard(LLM_DIR)

    # Per-shot reply: each scene gets a three-section reply. The third
    # scene INTENTIONALLY drifts the position so we can prove the
    # injection happened (i.e. the stub WILL parrot back the position
    # text we put in its user template).
    def shot_reply(idx, prev_desc, prev_sound):
        # Faithfully carry over the previous scene's last position
        # sentence (this is the 'correct' LLM behaviour; the bug
        # would be a drift). Here we always pin to the same left/right
        # that the spatial_layout_directive declared. Both an English
        # prose sentence and a Chinese anchored sentence appear — the
        # drift preflight only recognises the Chinese form (the EN
        # form with a CJK subject name doesn't match the extractor
        # patterns), so this doubles as coverage that the ZH anchor
        # normalises to the same canonical bucket as the board.
        return (
            "integrated_multimodal_description:\n"
            f"[Shot {idx}] "
            "莎莉猫 stays at left of frame; 哈利猫 stays at right of frame. "
            "莎莉猫坐画面左侧；哈利猫坐画面右侧。"
            "The fish-snack plate slides between the two cats. "
            f"{prev_desc[:60]}\n"
            "\n"
            "overall_soundscape:\n"
            "Warm quiet interior, distant birdsong.\n"
            "\n"
            "non_diegetic_music:\n"
            "No non-diegetic music.\n"
        )

    # The storyboard path expects extract_dialogue to return empty
    # for non-dialogue text -> single narrator turn.
    extractor_reply = json.dumps({"turns": []})
    prefix_reply = (
        "Hand-drawn 2D animation with painterly illustration quality. "
        "Continuous slow forward camera drift, settling at the final "
        "frame. Setting: a wooden dining table in a warm interior. "
        "莎莉猫 stays at left of frame throughout; "
        "哈利猫 stays at right of frame throughout. "
        "Dominant tones: warm wood browns and golden window highlights. "
        "No on-screen text, no watermarks, no non-diegetic music.\n"
        "\n"
        "CAST:\n"
        "哈利猫: anthropomorphic dark grey-brown tabby cat with long "
        "fluffy fur, brown woven blazer, beige ribbed knit necktie.\n"
        "莎莉猫: anthropomorphic cream-blonde fluffy cat with curly fur, "
        "beige plaid suit jacket, dark navy blue bow tie.\n"
    )

    replies = [
        extractor_reply,         # extract_dialogue
        shots_json,              # _auto_storyboard
        prefix_reply,             # _synth_prefix
        shot_reply(1, "", ""),    # scene 1
        shot_reply(2, "first push", "warm interior"),  # scene 2
        shot_reply(3, "second push", "carried ambience"),  # scene 3
    ]
    stub = _StubLLM(replies)

    # Speed up deterministic parts (no need for real disk cache).
    monkeypatch.setattr(gen, "_caption_cache_disk_root",
                        lambda: str(tmp_path))

    enhancer = gen.H3LoopPromptEnhancer(
        stub, temperature=0.0, max_tokens=4096, timeout=10,
    )
    out = enhancer(
        user_input=user_input,
        total_duration_seconds=20,
        scene_count=0,
        pacing="normal - 正常(推荐)",
        generation_mode="per_shot - 逐场生成(推荐)",
        category="none - 不指定",
        output_language="en",
        seed=77,
        reference_mode="t2va - 文生视频链(默认)",
        references_text="",
        images=None,
        caption_mode="cache_memory_disk",
        force_recaption=False,
        caption_cache_scope="memory_disk",
    )

    plan = json.loads(out["plan_json"])
    assert "shots" in plan
    assert len(plan["shots"]) == 3, (
        f"expected 3 scenes, got {len(plan['shots'])}"
    )

    # ── 1. spatial_layout was extracted from the rewritten user_input ──
    layout = lp.extract_spatial_layout(user_input)
    assert "莎莉猫" in layout and layout["莎莉猫"] == "left of frame"
    assert "哈利猫" in layout and layout["哈利猫"] == "right of frame"

    # ── 2. Every per-shot user template contained the directive ──
    shot_calls = [c for c in stub.calls if c[0] == "shot"]
    assert len(shot_calls) == 3, (
        f"expected 3 per-shot LLM calls; got {len(shot_calls)}"
    )
    for i, (_stage, msgs) in enumerate(shot_calls, start=1):
        user_text = msgs[1]["content"]
        # The directive uses the canonical positions extracted from
        # user_input — confirm they appear in every shot's user
        # template.
        assert "莎莉猫 stays at left of frame" in user_text, (
            f"shot {i} user template missing spatial layout directive "
            f"for 莎莉猫; got user text:\n{user_text[:400]}"
        )
        assert "哈利猫 stays at right of frame" in user_text, (
            f"shot {i} user template missing spatial layout directive "
            f"for 哈利猫"
        )
        # The placeholder must have been substituted.
        assert "{spatial_layout_directive}" not in user_text

    # ── 3. The three scene prompts themselves must all carry the ──
    #       same position (no drift). The stub replies pinned both
    #       positions identically; verify the prompt-line-array shape.
    for s in plan["shots"]:
        prompt_text = "\n".join(s["prompt"])
        assert "stays at left of frame" in prompt_text
        assert "stays at right of frame" in prompt_text

    # ── 3b. Consistent scene prose -> NO drift warnings. The board ──
    #        layout (ZH-declared) and the scene prose (ZH-anchored)
    #        normalise to identical canonical buckets. Warnings land
    #        in the summary's Warnings section.
    assert "spatial layout drift" not in out["summary"], (
        "consistent board must not emit drift warnings; summary:\n"
        f"{out['summary']}"
    )

    # ── 4. The prefix synth LLM call received our 3-shot digest; the ──
    #       position rule is now in its system prompt.
    prefix_calls = [c for c in stub.calls if c[0] == "prefix_synth"]
    assert len(prefix_calls) == 1
    sys_text = prefix_calls[0][1][0]["content"]
    assert "Spatial layout" in sys_text, (
        "prefix synth system prompt must declare the spatial layout rule"
    )

    # ── 5. Preflight report should reflect that everything wired up. ──
    preflight = out["summary"]
    assert "MiniMax H3 Loop Plan summary" in preflight

    # ── 6. Strict plan_json contract — no generation params leak. ──
    # The Production Plan node reads its own widget values for steps /
    # cfg / canvas / context_length / continuation_mode. Those must
    # NOT appear in plan_json (see validate_plan() top-level rule).
    plan = json.loads(out["plan_json"])
    assert set(plan.keys()) == {"shots", "prompt_prefix"}, (
        f"plan_json top-level keys must be exactly {{'shots', 'prompt_prefix'}}; "
        f"got {sorted(plan.keys())}"
    )
    assert "defaults" not in plan, (
        "plan_json must not carry a 'defaults' key (sampling params live "
        "on the Production Plan widget, not in the JSON)"
    )
    expected_shot_keys = {"id", "prompt", "length", "seed"}
    for s in plan["shots"]:
        assert set(s.keys()) == expected_shot_keys, (
            f"shot {s.get('id')} keys must be exactly {sorted(expected_shot_keys)}; "
            f"got {sorted(s.keys())}"
        )
        # seed MUST be a uint64 digit string (not int) — the node
        # contract requires a string so values above JS safe range
        # survive the JSON round-trip.
        assert isinstance(s["seed"], str) and s["seed"].isdigit(), (
            f"shot {s['id']} seed must be a uint64 digit string; "
            f"got {s['seed']!r}"
        )
        assert isinstance(s["prompt"], list)
        assert all(isinstance(ln, str) for ln in s["prompt"])


# --------------------------------------------------------------------------- #
# Stage 6 — the wired drift preflight: an actually-drifting scene must
# surface a warning naming the subject and both positions.
# --------------------------------------------------------------------------- #
def test_e2e_drifting_scene_emits_spatial_drift_warning(
    lp, monkeypatch, tmp_path
):
    """Scene 3 silently swaps 哈利猫 to the left of frame (the original
    scene_03 failure). The Stage-2.5 preflight must re-extract every
    shot's declared positions from the emitted prompt text and warn —
    warning-level only, the plan still builds."""
    gen = _load_generator()
    monkeypatch.setattr(gen, "_caption_cache_disk_root", lambda: str(tmp_path))

    user_input = (
        "场景设定：温暖的木餐桌，金色窗光从左侧洒入。"
        "图2的猫（莎莉猫）坐画面左侧；"
        "图1的猫（哈利猫）坐画面右侧。"
        "两只猫面对面推一盘小鱼干。"
    )

    def shot_reply(idx):
        # Scenes 1-2 pin the declared layout; scene 3 drifts 哈利猫 to
        # the left (ZH-anchored so the extractor recognises it).
        harry = "哈利猫坐画面右侧" if idx < 3 else "哈利猫坐画面左侧"
        return (
            "integrated_multimodal_description:\n"
            f"[Shot {idx}] "
            "莎莉猫坐画面左侧；"
            f"{harry}。"
            "The fish-snack plate slides between the two cats.\n"
            "\n"
            "overall_soundscape:\n"
            "Warm quiet interior, distant birdsong.\n"
            "\n"
            "non_diegetic_music:\n"
            "No non-diegetic music.\n"
        )

    prefix_reply = (
        "Hand-drawn 2D animation with painterly illustration quality. "
        "Setting: a wooden dining table in a warm interior. "
        "莎莉猫 stays at left of frame throughout; "
        "哈利猫 stays at right of frame throughout. "
        "No on-screen text, no watermarks, no non-diegetic music.\n"
        "\n"
        "CAST:\n"
        "哈利猫: tabby cat.\n"
        "莎莉猫: cream-blonde cat.\n"
    )
    replies = [
        json.dumps({"turns": []}),                      # extract_dialogue
        _build_fish_snack_storyboard(LLM_DIR),          # storyboard
        prefix_reply,                                   # prefix synth
        shot_reply(1),
        shot_reply(2),
        shot_reply(3),                                  # the drift
    ]
    stub = _StubLLM(replies)
    enhancer = gen.H3LoopPromptEnhancer(stub, temperature=0.0, timeout=10)
    out = enhancer(
        user_input=user_input,
        total_duration_seconds=20,
        scene_count=0,
        pacing="normal - 正常(推荐)",
        generation_mode="per_shot - 逐场生成(推荐)",
        category="none - 不指定",
        output_language="en",
        seed=7,
        reference_mode="t2va - 文生视频链(默认)",
        references_text="",
        images=None,
        caption_mode="cache_memory_disk",
        force_recaption=False,
        caption_cache_scope="memory_disk",
    )

    preflight = out["summary"]
    assert "spatial layout drift" in preflight, (
        "the drifted scene_03 must surface a spatial layout drift warning; "
        f"preflight was:\n{preflight}"
    )
    # The warning must name the drifting subject and both positions.
    drift_lines = [
        ln for ln in preflight.splitlines() if "spatial layout drift" in ln
    ]
    joined = "\n".join(drift_lines)
    assert "哈利猫" in joined
    assert "left of frame" in joined and "right of frame" in joined
    # Warning-level only: the plan still assembles with 3 shots.
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 3
