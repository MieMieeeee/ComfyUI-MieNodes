# -*- coding: utf-8 -*-
"""Free-creation (concept) mode + human-lenient parsing (2026-10-29).

Two surfaces:

* ``AssetStoryboardEnhancer(mode="concept")`` — the user pastes anything
  (an idea, a story paragraph, loose dialogue lines); the LLM authors the
  complete v2 asset. Wiring facts (name / audio_mode /
  continuation_overlap_frames) stay caller-owned, everything else is a
  non-binding preference, and the same validate_v2 + retry loop guards
  the result.
* ``parse_script`` / ``parse_roles`` leniency for the locked mode:
  speech verbs (说/问/喊/回答…) count as dialogue labels, settings
  accept commas as separators, and every parse error echoes the offending
  line and teaches the format instead of just refusing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]


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
def mods():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.assets", PROJECT_DIR / "nodes" / "assets")
    _load_file("_mienodes_internal.nodes.assets.library",
               PROJECT_DIR / "nodes" / "assets" / "library.py")
    contract = _load_file("_mienodes_internal.nodes.assets.storyboard_contract",
                          PROJECT_DIR / "nodes" / "assets" / "storyboard_contract.py")
    _ensure_pkg("_mienodes_internal.nodes.llm", PROJECT_DIR / "nodes" / "llm")
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts",
                PROJECT_DIR / "nodes" / "llm" / "prompts")
    generator = _load_file("_mienodes_internal.nodes.llm.storyboard_asset_generator",
                           PROJECT_DIR / "nodes" / "llm" / "storyboard_asset_generator.py")
    return {"contract": contract, "generator": generator}


class FakeConnector:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, seed=0, temperature=0.2, max_tokens=8192):
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def verified_asset():
    """The docs prompt's own verified example (起身邪笑), as a dict."""
    text = (PROJECT_DIR / "docs" / "script_asset_system_prompt.txt").read_text("utf-8")
    line = next(l for l in text.splitlines() if l.startswith('{"version"'))
    return json.loads(line)


def _planner(mods, replies):
    """The concept-mode planner with the continuity critic switched off.

    These tests exercise the CONTRACT path (caller-owned fields, event
    fidelity, the repair turn). The critic is a second reader with its
    own provider call and its own test file
    (test_script_asset_critic.py); leaving it on here would only blur
    the call counts.
    """
    class _ConceptPlanner:
        def __init__(self, enhancer):
            self._enhancer = enhancer
            self.llm = enhancer.llm

        def __call__(self, *args, **kwargs):
            kwargs.setdefault("critic", "off")
            return self._enhancer(*args, **kwargs)

    return _ConceptPlanner(mods["generator"].AssetStoryboardEnhancer(FakeConnector(replies)))


def test_concept_mode_authors_a_full_asset_from_free_text(mods, verified_asset):
    reply = dict(verified_asset)
    reply["name"] = "我的新剧本"
    planner = _planner(mods, [json.dumps(reply, ensure_ascii=False)])
    asset = planner(
        "深夜办公室，甲方连珠炮骂乙方，乙方最后默默扭过头去。", 0,
        mode="concept", name="我的新剧本")
    assert asset["version"] == 2 and asset["name"] == "我的新剧本"
    # the request payload is free-form: mode=concept + preferences, no locked_events
    payload = json.loads(planner.llm.calls[0][1]["content"])
    assert payload["mode"] == "concept"
    assert "locked_events" not in payload
    assert payload["caller_owned"]["name"] == "我的新剧本"
    assert payload["preferences"]["camera"]["framing"] == "medium_shot"


def test_concept_mode_enforces_caller_owned_fields_with_retry(mods, verified_asset):
    bad = dict(verified_asset)
    bad["name"] = "擅自改名"
    good = dict(verified_asset)
    good["name"] = "我的新剧本"
    planner = _planner(mods, [json.dumps(bad, ensure_ascii=False),
                              json.dumps(good, ensure_ascii=False)])
    asset = planner("一个点子", 0, mode="concept", name="我的新剧本")
    assert asset["name"] == "我的新剧本"
    assert len(planner.llm.calls) == 2
    assert "caller-owned name changed" in planner.llm.calls[1][-1]["content"]


def test_concept_mode_enforces_overlap_and_rejects_empty_concept(mods, verified_asset):
    reply = dict(verified_asset)
    reply["name"] = "我的新剧本"
    reply["continuation_overlap_frames"] = 0
    planner = _planner(mods, [json.dumps(reply, ensure_ascii=False)] * 3)
    with pytest.raises(RuntimeError, match="continuation_overlap_frames"):
        planner("又一个点子", 0, mode="concept", name="我的新剧本")
    with pytest.raises(ValueError, match="concept 是空的"):
        empty = _planner(mods, [])
        empty("", 0, mode="concept")


def test_concept_mode_validation_errors_feed_the_retry_loop(mods, verified_asset):
    broken = dict(verified_asset)
    broken["name"] = "我的新剧本"
    broken["target_duration_seconds"] = 0.1  # validator: duration too short
    good = dict(verified_asset, name="我的新剧本")
    planner = _planner(mods, [json.dumps(broken, ensure_ascii=False),
                              json.dumps(good, ensure_ascii=False)])
    asset = planner("第三个点子", 0, mode="concept", name="我的新剧本")
    assert asset["target_duration_seconds"] == good["target_duration_seconds"]
    assert "Asset validation failed" in planner.llm.calls[1][-1]["content"]


def _small_locked_asset():
    """A complete v2 asset whose header EXACTLY matches the locked-mode
    call parameters used below (every header field is caller-owned)."""
    return {
        "version": 2, "name": "我的新剧本", "language": "zh",
        "roles": [
            {"id": "character_1", "name": "甲", "presence": "on_screen",
             "initially_visible": True, "position": "left", "voice": ""},
            {"id": "character_2", "name": "乙", "presence": "on_screen",
             "initially_visible": True, "position": "right", "voice": ""}],
        "camera": {"framing": "medium_shot", "movement": "locked",
                   "perspective": "observer"},
        "look": "", "setting": "", "constraints": "",
        "target_duration_seconds": 8, "continuation_overlap_frames": 22,
        "source_script": "角色2：你好。\n角色1动作：点头。",
        "shots": [{"id": "scene_01", "duration_seconds": 7, "events": [
            {"id": "event_001", "type": "dialogue", "character": "character_2",
             "text": "你好。", "language": "zh"},
            {"id": "event_002", "type": "action", "character": "character_1",
             "text": "点头。", "language": "zh"}]}],
        "execution": {
            "look": "", "setting": "",
            "voices": {"character_1": "", "character_2": ""},
            "visible_state": "Character 1 and Character 2 stand facing each "
                             "other in the present, both visible.",
            "directions": {"event_002": "Character 1 nods once."}}}


def test_locked_mode_shot_count_zero_means_auto(mods):
    planner = _planner(mods, [json.dumps(_small_locked_asset(), ensure_ascii=False)])
    asset = planner("角色2：你好。\n角色1动作：点头。", 0,
                    name="我的新剧本", target_duration_seconds=8,
                    character_settings="角色1：甲 | 画面内 | 左侧\n角色2：乙 | 画面内 | 右侧")
    assert len(asset["shots"]) == 1
    payload = json.loads(planner.llm.calls[0][1]["content"])
    assert "required_shot_count" not in payload


# --------------------------------------------------------------------------- #
# Lenient parsing for the locked mode
# --------------------------------------------------------------------------- #
def test_speech_verbs_count_as_dialogue_labels(mods):
    parse = mods["contract"].parse_script
    roles = mods["contract"].parse_roles(1, "角色1：甲 | 画面内 | 中间")
    events = parse("角色1说：你好。\n角色1问：在吗\n角色1喊：站住！\n角色1回答：不行", roles)
    assert [e["type"] for e in events] == ["dialogue"] * 4
    assert events[0]["text"] == "你好。"


def test_parse_errors_echo_the_offending_line_and_teach(mods):
    parse = mods["contract"].parse_script
    roles = mods["contract"].parse_roles(1, "角色1：甲 | 画面内 | 中间")
    with pytest.raises(ValueError, match="深夜，两人坐在沙发上"):
        parse("深夜，两人坐在沙发上", roles)
    with pytest.raises(ValueError, match="concept 概念创作"):
        parse("旁白：从前有座山", roles)


def test_settings_accept_comma_separators_and_teach_on_mismatch(mods):
    parse_roles = mods["contract"].parse_roles
    roles = parse_roles(2, "角色1：小明，画外，无\n角色2：小红，画面内，右侧，清亮女声")
    assert roles[0]["name"] == "小明" and roles[0]["presence"] == "off_screen"
    assert roles[0]["position"] == "none"
    assert roles[1]["name"] == "小红" and roles[1]["voice"] == "清亮女声"
    with pytest.raises(ValueError, match="character_count=2"):
        parse_roles(2, "角色1：小明，画外\n角色2：小红，画面内\n角色3：小刚，画面内")
    with pytest.raises(ValueError, match="超出 character_count"):
        parse_roles(1, "角色1：小明，画外\n角色5：小刚，画面内")


# --------------------------------------------------------------------------- #
# Revision mode: iterate on the SAVED asset
# --------------------------------------------------------------------------- #
def test_revise_mode_produces_next_version_from_base(mods, verified_asset):
    base = dict(verified_asset)  # 起身邪笑 (audio_mode generated, overlap 22)
    revised = json.loads(json.dumps(base, ensure_ascii=False))
    revised["name"] = "疑惑邪笑"
    revised["shots"][0]["events"][4]["text"] = "你的账号已被冻结。"
    revised["source_script"] = base["source_script"].replace("封禁", "冻结")
    planner = _planner(mods, [json.dumps(revised, ensure_ascii=False)])
    asset = planner("把最后一句台词的「封禁」改成「冻结」，其余一律不动", 0,
                    mode="revise", base_asset=base, name="疑惑邪笑")
    assert asset["name"] == "疑惑邪笑"
    assert "冻结" in asset["shots"][0]["events"][4]["text"]
    payload = json.loads(planner.llm.calls[0][1]["content"])
    assert payload["mode"] == "revise" and payload["base_asset"]["name"] == base["name"]


def test_revise_mode_inherits_wiring_facts_loudly(mods, verified_asset):
    base = dict(verified_asset)
    bad = json.loads(json.dumps(base, ensure_ascii=False))
    bad["name"] = "疑惑邪笑"
    bad["audio_mode"] = "reference"  # 模型不得翻转接线事实
    good = json.loads(json.dumps(base, ensure_ascii=False))
    good["name"] = "疑惑邪笑"
    planner = _planner(mods, [json.dumps(bad, ensure_ascii=False),
                              json.dumps(good, ensure_ascii=False)])
    asset = planner("微调一版", 0, mode="revise", base_asset=base, name="疑惑邪笑")
    assert asset.get("audio_mode", "generated") == base.get("audio_mode", "generated")
    assert "caller-owned audio_mode" in planner.llm.calls[1][-1]["content"]


def test_revise_mode_requires_base_and_instruction(mods, verified_asset):
    planner = _planner(mods, [])
    with pytest.raises(ValueError, match="storyboard"):
        planner("改一版", 0, mode="revise", name="x")
    with pytest.raises(ValueError, match="修订要求"):
        planner("", 0, mode="revise", base_asset=verified_asset, name="x")


def test_asset_diff_reports_reworded_and_untouched_events(mods, verified_asset):
    gen = mods["generator"]
    old = dict(verified_asset)
    new = json.loads(json.dumps(verified_asset, ensure_ascii=False))
    new["shots"][0]["events"][4]["text"] = "你的账号已被冻结。"
    new["source_script"] = old["source_script"].replace("封禁", "冻结")
    new["light"] = "night_screen"
    diff = gen.asset_diff(old, new)
    text = "\n".join(diff)
    assert "5 条逐字未动" in text or "逐字未动" in text
    assert "改写" in text and "冻结" in text
    assert "night_screen" in text  # light change reported

