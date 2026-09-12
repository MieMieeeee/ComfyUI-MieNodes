# -*- coding: utf-8 -*-
"""End-to-end tests for the ``MiniMaxH3LoopPromptGenerator`` node with a
stubbed LLM connector: full pipeline (Stage 0-3) in per_shot and
single_call modes, plan shape vs. the real Production Plan JSON, and
failure handling."""
import importlib.util
import json
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
def lg():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"
    )
    _load_file("_mienodes_internal.nodes.llm.h3_prompts", LLM_DIR / "h3_prompts.py")
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
        LLM_DIR / "minimax_h3_storyboard_prompts.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )


class FakeConnector:
    model = "fake-model"

    def get_state(self):
        return "fake-state"


class ScriptedConnector(FakeConnector):
    """Serves queued replies; understands which stage is calling by the
    user content so tests can mix prefix + shot replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("scripted connector exhausted")
        return self.replies.pop(0)


def _storyboard_json(n=3):
    shots = []
    for i in range(1, n + 1):
        shots.append(
            {
                "id": f"scene_{i:02d}",
                "description": f"Shot {i} description of the courtyard.",
                "shot_type": "medium_shot",
                "camera_movement": "slow_push_in",
                "transition_in": "fade_from_black" if i == 1 else "hard_cut",
                "duration_seconds": 10,
                "narrative_beat": "establish",
                "characters": ["young_woman"],
                "props": ["porcelain_bowl"],
                "notes": "Carry the bowl.",
            }
        )
    return json.dumps(shots, ensure_ascii=False)


def _clip_reply(i):
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] Clip {i}: the bowl settles, ice slips.\n"
        "\n"
        "overall_soundscape:\n"
        "Cicadas hold; one bright clink.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _single_call_reply(n):
    items = []
    for i in range(1, n + 1):
        items.append(
            {
                "id": f"scene_{i:02d}",
                "integrated_multimodal_description": f"[Shot 1] Clip {i} body text.",
                "overall_soundscape": "Rain on glass, distant cicadas.",
                "non_diegetic_music": "No non-diegetic music.",
            }
        )
    return json.dumps(items, ensure_ascii=False)


PREFIX_REPLY = (
    "Always the same young woman in a Jiangnan courtyard at high summer: "
    "loosely pinned black hair, white cotton blouse, pale jade bracelet."
)


# --------------------------------------------------------------------------- #
# per_shot happy path
# --------------------------------------------------------------------------- #
def test_e2e_per_shot_full_pipeline(lg):
    conn = ScriptedConnector([PREFIX_REPLY, _clip_reply(1), _clip_reply(2), _clip_reply(3)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "A Jiangnan courtyard summer",
        _storyboard_json(3),
        duration_seconds=10,
        generation_mode="per_shot",
        category="cinematic-story - 电影短片/MV/戏剧",
        width=544,
        height=960,
        output_language="en",
        seed=1000,
    )
    # 1 prefix call + 3 shot calls.
    assert len(conn.calls) == 4
    # Prefix call used the prefix templates; shot calls used the shot system.
    assert "prompt prefix" in conn.calls[0][0]["content"].lower()
    for call in conn.calls[1:]:
        assert "Loop-plan addendum" in call[0]["content"]
    # Shots 2+ got the continuation block with the previous clip's FULL
    # description and its exact soundscape bed.
    assert "Motion Context" in conn.calls[2][1]["content"]
    assert "Clip 1" in conn.calls[2][1]["content"]
    assert "Cicadas hold" in conn.calls[2][1]["content"]
    assert "Clip 2" in conn.calls[3][1]["content"]
    assert "Cicadas hold" in conn.calls[3][1]["content"]
    # Pacing: every shot call carries the beat budget scaled to the ACTUAL
    # grid duration (10s request -> 243 frames -> 10.13s), and clips 2+
    # note the carried overlap is excluded from the budget.
    for call in conn.calls[1:]:
        assert "2-3 distinct action beats" in call[1]["content"]
        assert "10.1 seconds" in call[1]["content"]
    assert "overlap" in conn.calls[2][1]["content"]
    assert "overlap" not in conn.calls[1][1]["content"]
    # Prefix synthesis is grounded in the storyboard digest.
    prefix_user = conn.calls[0][1]["content"]
    assert "scene_01" in prefix_user and "Shot 1 description" in prefix_user

    plan = json.loads(out["plan_json"])
    assert list(plan.keys()) == ["defaults", "shots", "prompt_prefix"]
    assert plan["defaults"] == {"steps": 20}
    assert len(plan["shots"]) == 3
    for i, shot in enumerate(plan["shots"], start=1):
        assert shot["id"] == f"scene_{i:02d}"
        assert shot["length"] == 243
        assert shot["steps"] == 20
        assert shot["seed"] == "1000"  # unified seed (default)
        assert shot["prompt"][0] == "integrated_multimodal_description:"
        assert shot["prompt"][3] == "overall_soundscape:"
        assert shot["prompt"][6] == "non_diegetic_music:"
    assert plan["prompt_prefix"] == [PREFIX_REPLY]
    assert json.loads(out["prompt_prefix_out"]) == [PREFIX_REPLY]
    assert len(json.loads(out["shot_prompts"])) == 3
    assert "Shots: 3" in out["preflight_report"]
    assert "729" in out["preflight_report"]  # 3 * 243 frames total
    assert "| 3 | `scene_03` | 243 |" in out["plan_preview"]


def test_e2e_user_prefix_skips_stage1(lg):
    conn = ScriptedConnector([_clip_reply(1)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "concept",
        '[{"id": "s1", "description": "d"}]',
        prompt_prefix_input="My prefix paragraph one.\n\nMy prefix paragraph two.",
        seed=5,
    )
    assert len(conn.calls) == 1  # no prefix call
    plan = json.loads(out["plan_json"])
    assert plan["prompt_prefix"] == [
        "My prefix paragraph one.",
        "My prefix paragraph two.",
    ]


def test_e2e_duration_from_storyboard_and_node_default(lg):
    board = json.dumps(
        [
            {"id": "a", "description": "d", "duration_seconds": 5},  # -> 124
            {"id": "b", "description": "d"},                          # -> 243 (10s default)
        ]
    )
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", board, duration_seconds=10, prompt_prefix_input="p", seed=1
    )
    plan = json.loads(out["plan_json"])
    assert plan["shots"][0]["length"] == 124
    assert plan["shots"][1]["length"] == 243


def test_e2e_overrides_applied(lg):
    board = _storyboard_json(2)
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c",
        board,
        prompt_prefix_input="p",
        per_shot_overrides="scene_02:length=481\nscene_02:seed=4480\nscene_01:steps=30",
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert plan["shots"][1]["length"] == 481
    assert plan["shots"][1]["seed"] == "4480"
    assert plan["shots"][0]["steps"] == 30
    assert plan["shots"][0]["seed"] == "1"  # unified seed (default), base=1


# --------------------------------------------------------------------------- #
# single_call mode
# --------------------------------------------------------------------------- #
def test_e2e_single_call(lg):
    conn = ScriptedConnector([PREFIX_REPLY, _single_call_reply(3)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c",
        _storyboard_json(3),
        generation_mode="single_call - 单次调用(快/省)",
        prompt_prefix_input=None or "",
        seed=9,
    )
    assert len(conn.calls) == 2  # prefix + one big call
    assert "SINGLE-CALL FORMAT OVERRIDE" in conn.calls[1][1]["content"]
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 3
    assert plan["shots"][2]["prompt"][1] == "[Shot 1] Clip 3 body text."
    assert all(s["seed"] == "9" for s in plan["shots"])  # unified seed (default)


def test_parse_generation_mode(lg):
    assert lg.parse_generation_mode("per_shot - 逐场生成(推荐)") == "per_shot"
    assert lg.parse_generation_mode("single_call") == "single_call"
    assert lg.parse_generation_mode("weird") == "per_shot"
    assert lg.parse_generation_mode("") == "per_shot"


# --------------------------------------------------------------------------- #
# Auto-storyboard mode (shot_count) + trimming
# --------------------------------------------------------------------------- #
def test_auto_storyboard_standalone(lg):
    """shots_text empty + shot_count>0 -> one-node pipeline: storyboard
    call, prefix call, then per-shot calls. The inline storyboard always
    uses the chain-compatible single_continuous style."""
    conn = ScriptedConnector(
        [_storyboard_json(2), PREFIX_REPLY, _clip_reply(1), _clip_reply(2)]
    )
    out = lg.H3LoopPromptEnhancer(conn)(
        "一只水墨风格的猫与狗在屋顶大战",
        "",
        shot_count=2,
        seed=100,
    )
    assert len(conn.calls) == 4  # storyboard + prefix + 2 shots
    # First call is the storyboard pipeline, pinned to single_continuous.
    assert "storyboard" in conn.calls[0][0]["content"].lower()
    assert "exactly 2 shots" in conn.calls[0][1]["content"]
    assert "single_continuous" in conn.calls[0][1]["content"]
    assert "水墨" in conn.calls[0][1]["content"]
    # Prefix call is grounded in the auto-storyboard digest.
    assert "scene_01" in conn.calls[1][1]["content"]
    plan = json.loads(out["plan_json"])
    assert [s["id"] for s in plan["shots"]] == ["scene_01", "scene_02"]
    assert plan["shots"][0]["seed"] == "100"  # unified seed (default)
    assert "Shots: 2" in out["preflight_report"]


def test_shot_count_trims_incoming(lg):
    conn = ScriptedConnector([PREFIX_REPLY, _clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", _storyboard_json(3), shot_count=2, prompt_prefix_input=None or "", seed=1
    )
    plan = json.loads(out["plan_json"])
    assert [s["id"] for s in plan["shots"]] == ["scene_01", "scene_02"]
    assert "trimmed incoming storyboard" in out["preflight_report"]


def test_empty_shots_and_zero_count_still_raises(lg):
    with pytest.raises(RuntimeError):
        lg.H3LoopPromptEnhancer(FakeConnector())("c", "", shot_count=0, seed=1)


def test_auto_storyboard_parse_retry(lg):
    conn = ScriptedConnector(
        ["garbage", _storyboard_json(1), PREFIX_REPLY, _clip_reply(1)]
    )
    out = lg.H3LoopPromptEnhancer(conn)("c", "", shot_count=1, seed=1)
    assert len(json.loads(out["plan_json"])["shots"]) == 1
    assert len(conn.calls) == 4  # 2 storyboard attempts + prefix + shot
    # Retry carries the corrective user turn.
    assert len(conn.calls[1]) == 3
    assert "ONLY the JSON array" in conn.calls[1][2]["content"]


def test_auto_storyboard_accepts_shots_wrapped_object(lg):
    """A model that replies {"shots": [...]} instead of a bare array is
    unwrapped instead of failing the node."""
    wrapped = json.dumps({"shots": json.loads(_storyboard_json(2))})
    conn = ScriptedConnector([wrapped, PREFIX_REPLY, _clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)("c", "", shot_count=2, seed=1)
    assert len(json.loads(out["plan_json"])["shots"]) == 2
    assert len(conn.calls) == 4  # no wasted retry


# --------------------------------------------------------------------------- #
# Natural-language input + validation errors
# --------------------------------------------------------------------------- #
def test_e2e_natural_language_shots(lg):
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c",
        "第一场：她端起碗\n第二场：冰块轻响",
        prompt_prefix_input="p",
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert plan["shots"][0]["id"] == "clip_0001"
    assert plan["shots"][1]["id"] == "clip_0002"


def test_empty_shots_text_raises(lg):
    with pytest.raises(RuntimeError):
        lg.H3LoopPromptEnhancer(FakeConnector())("c", "   ", seed=1)


def test_bad_canvas_raises(lg):
    with pytest.raises(RuntimeError, match="multiples of 32"):
        lg.H3LoopPromptEnhancer(FakeConnector())(
            "c", _storyboard_json(1), width=550, height=960, seed=1
        )


def test_bad_overrides_raise(lg):
    with pytest.raises(RuntimeError, match="unknown shot id"):
        lg.H3LoopPromptEnhancer(FakeConnector())(
            "c", _storyboard_json(1), per_shot_overrides="nope:steps=5", seed=1
        )
    with pytest.raises(RuntimeError, match="off the H3 grid"):
        lg.H3LoopPromptEnhancer(FakeConnector())(
            "c", _storyboard_json(1), per_shot_overrides="scene_01:length=240", seed=1
        )


# --------------------------------------------------------------------------- #
# Retry behavior
# --------------------------------------------------------------------------- #
def test_shot_parse_retry_then_success(lg):
    conn = ScriptedConnector(
        [
            PREFIX_REPLY,
            "garbage reply",
            _clip_reply(1),
            _clip_reply(2),
        ]
    )
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", _storyboard_json(2), prompt_prefix_input=None or "", seed=1
    )
    assert len(json.loads(out["plan_json"])["shots"]) == 2
    assert len(conn.calls) == 4  # prefix + retry + retry + shot2
    # The parse retry carries a corrective user turn naming the expected
    # headers (live E2E: identical resends taught the model nothing).
    retry_call = conn.calls[2]
    assert len(retry_call) == 3
    assert "could not be parsed" in retry_call[2]["content"]
    assert "integrated_multimodal_description:" in retry_call[2]["content"]


def test_shot_parse_failure_raises_no_partial_plan(lg):
    conn = ScriptedConnector([PREFIX_REPLY, "bad", "still bad", _clip_reply(2)])
    with pytest.raises(RuntimeError, match="scene_01"):
        lg.H3LoopPromptEnhancer(conn)(
            "c", _storyboard_json(2), prompt_prefix_input=None or "", seed=1
        )


def test_prefix_retry(lg):
    conn = ScriptedConnector(["", PREFIX_REPLY, _clip_reply(1)])
    out = lg.H3LoopPromptEnhancer(conn)("c", _storyboard_json(1), seed=1)
    assert len(conn.calls) == 3  # 2 prefix attempts + 1 shot
    assert json.loads(out["plan_json"])["prompt_prefix"] == [PREFIX_REPLY]


def test_think_block_stripped(lg):
    reply = "<think>chain of thought</think>\n" + _clip_reply(1)
    conn = ScriptedConnector([PREFIX_REPLY, reply])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", _storyboard_json(1), prompt_prefix_input=None or "", seed=1
    )
    plan = json.loads(out["plan_json"])
    assert "think" not in json.dumps(plan["shots"][0]["prompt"])


# --------------------------------------------------------------------------- #
# unified_seed option
# --------------------------------------------------------------------------- #
def test_unified_seed_false_diversity(lg):
    """unified_seed=False restores per-clip sequential seeds (base+N)."""
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", _storyboard_json(2), unified_seed=False, prompt_prefix_input="p", seed=1000
    )
    plan = json.loads(out["plan_json"])
    assert [s["seed"] for s in plan["shots"]] == ["1001", "1002"]


def test_unified_seed_all_clips_share(lg):
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", _storyboard_json(2), unified_seed=True, prompt_prefix_input="p", seed=1000
    )
    plan = json.loads(out["plan_json"])
    assert [s["seed"] for s in plan["shots"]] == ["1000", "1000"]


def test_unified_seed_zero_seed_derives_once(lg):
    """seed=0 + unified: wall-clock base is derived ONCE and shared."""
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c", _storyboard_json(2), prompt_prefix_input="p", seed=0
    )
    seeds = [s["seed"] for s in json.loads(out["plan_json"])["shots"]]
    assert seeds[0] == seeds[1] and int(seeds[0]) > 0


def test_unified_seed_override_wins(lg):
    conn = ScriptedConnector([_clip_reply(1), _clip_reply(2)])
    out = lg.H3LoopPromptEnhancer(conn)(
        "c",
        _storyboard_json(2),
        prompt_prefix_input="p",
        seed=1000,
        per_shot_overrides="scene_02:seed=4480",
    )
    seeds = [s["seed"] for s in json.loads(out["plan_json"])["shots"]]
    assert seeds == ["1000", "4480"]


# --------------------------------------------------------------------------- #
# Node wrapper + is_changed
# --------------------------------------------------------------------------- #
def test_node_generate_returns_five_outputs(lg):
    conn = ScriptedConnector([PREFIX_REPLY, _clip_reply(1), _clip_reply(2)])
    node = lg.MiniMaxH3LoopPromptGenerator()
    result = node.generate(
        conn,
        "concept",
        _storyboard_json(2),
        seed=42,
        timeout=60,
    )
    assert len(result) == 5
    plan_json, shot_prompts, prefix_out, report, preview = result
    plan = json.loads(plan_json)
    assert [s["seed"] for s in plan["shots"]] == ["42", "42"]  # unified
    assert len(json.loads(shot_prompts)) == 2
    assert json.loads(prefix_out)
    assert "Shots: 2" in report
    assert "`scene_01`" in preview


def test_is_changed_stable_and_sensitive(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = dict(
        concept="c",
        shots_text=_storyboard_json(2),
        seed=0,
        duration_seconds=10,
        generation_mode="per_shot - 逐场生成(推荐)",
        category="none - 不指定",
        width=544,
        height=960,
        output_language="en",
        prompt_prefix_input="",
        per_shot_overrides="",
        temperature=0.4,
        max_tokens=8192,
        timeout=120,
    )
    a = node.is_changed(FakeConnector(), **base)
    b = node.is_changed(FakeConnector(), **base)
    assert a == b
    for key, value in (
        ("seed", 1),
        ("shots_text", _storyboard_json(3)),
        ("per_shot_overrides", "scene_01:steps=25"),
    ):
        assert node.is_changed(FakeConnector(), **dict(base, **{key: value})) != a
