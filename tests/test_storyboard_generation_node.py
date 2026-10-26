"""Generation must lock the caller source and use the portable asset instruction."""
import copy
import json
from pathlib import Path

import pytest
from .test_h3_storyboard_generator import sb, ScriptedConnector
from .test_pov_offscreen import mods
from .test_storyboard_generation_contract import apple


def options(raw):
    settings = "\n".join("角色%d：%s | 画面内 | %s | %s" %
                         (i, r["name"], "左侧" if i == 1 else "右侧", r["voice"])
                         for i, r in enumerate(raw["roles"], 1))
    return dict(name=raw["name"], character_count=2, character_settings=settings,
                target_duration_seconds=20, camera_framing="two_shot", camera_movement="locked",
                camera_perspective="observer", look=raw["look"], setting=raw["setting"],
                constraints=raw["constraints"], continuation_overlap_frames=22, language="zh")


def enhancer(sb, conn):
    cls = getattr(sb, "AssetStoryboardEnhancer", None)
    assert cls is not None, "the existing generator needs an asset-native planner"
    return cls(conn)


def test_generator_emits_valid_asset_and_uses_one_portable_instruction(sb, mods):
    raw = apple()
    conn = ScriptedConnector([json.dumps(raw, ensure_ascii=False)])
    asset = enhancer(sb, conn)(raw["source_script"], 2, seed=4, **options(raw))
    assert asset == mods["storyboard"].validate_storyboard_payload(asset)
    resource = Path(sb.__file__).parent / "prompts/h3_storyboard/system_asset.txt"
    assert conn.calls[0][0]["content"] == resource.read_text(encoding="utf-8")
    assert len(conn.calls) == 1


@pytest.mark.parametrize("edit", ["words", "actor", "camera", "source", "count"])
def test_generator_retries_semantic_violations_with_specific_feedback(sb, mods, edit):
    raw, bad = apple(), apple()
    if edit == "words": bad["shots"][0]["events"][1]["text"] = "吃一个吧。"
    if edit == "actor": bad["shots"][0]["events"][0]["character"] = "character_1"
    if edit == "camera": bad["camera"]["movement"] = "gentle_follow"
    if edit == "source": bad["source_script"] += "\n角色1：我吃苹果。"
    if edit == "count": bad["shots"] = [dict(bad["shots"][0], events=[e for s in bad["shots"] for e in s["events"]])]
    conn = ScriptedConnector([json.dumps(bad), json.dumps(raw)])
    asset = enhancer(sb, conn)(raw["source_script"], 2, **options(raw))
    assert len(conn.calls) == 2
    assert conn.calls[1][-1]["role"] == "user"
    assert "validation" in conn.calls[1][-1]["content"].lower()
    assert asset["source_script"] == raw["source_script"]


def test_generator_fails_closed_after_bounded_retries(sb, mods):
    raw = apple()
    conn = ScriptedConnector(["[]"] * 3)
    with pytest.raises(RuntimeError, match="validation|asset"):
        enhancer(sb, conn)(raw["source_script"], 2, **options(raw))
    assert len(conn.calls) == 3


def test_existing_node_exposes_asset_and_explicit_role_controls(sb):
    inputs = sb.MiniMaxH3StoryboardGenerator.INPUT_TYPES()
    keys = set(inputs.get("required", {})) | set(inputs.get("optional", {}))
    assert {"character_count", "character_settings", "target_duration_seconds", "camera_framing"} <= keys
    assert "MIE_STORYBOARD" in sb.MiniMaxH3StoryboardGenerator.RETURN_TYPES


def test_preflight_does_not_reject_a_feasible_uneven_grouping(sb, mods):
    raw = apple()
    actions = ["角色1动作：轻轻挥手。"] * 9
    raw["source_script"] = "\n".join(actions + ["角色2：" + "好" * 225])
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    events = contract.parse_script(raw["source_script"], raw["roles"])
    raw["target_duration_seconds"] = 35
    raw["shots"] = [
        {"id": "scene_01", "duration_seconds": 5, "events": events[:9]},
        {"id": "scene_02", "duration_seconds": 30, "events": events[9:]},
    ]
    raw["execution"]["directions"] = {e["id"]: "Character 1 waves gently." for e in events[:9]}
    opts = dict(options(raw), target_duration_seconds=35)
    conn = ScriptedConnector([json.dumps(raw)])
    result = enhancer(sb, conn)(raw["source_script"], 2, **opts)
    assert len(conn.calls) == 1
    assert result["source_script"] == raw["source_script"]


def test_preview_text_carries_render_risk_warnings(sb):
    hedgehog = json.loads(
        (Path(__file__).parent / "fixtures" / "storyboard_v2_hedgehog.json").read_text(encoding="utf-8"))
    assert "⚠" in sb.preview_asset(hedgehog)
    assert "⚠" not in sb.preview_asset(apple())
