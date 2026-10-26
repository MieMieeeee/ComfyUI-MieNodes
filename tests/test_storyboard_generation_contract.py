"""Contract tests for user-authored role events and portable storyboard assets."""
import copy
import json
from pathlib import Path

import pytest
from .test_pov_offscreen import mods, _char

FIXTURES = Path(__file__).parent / "fixtures"


def apple():
    return json.loads((FIXTURES / "storyboard_v2_apple.json").read_text(encoding="utf-8"))


def flat(asset):
    return [event for shot in asset["shots"] for event in shot["events"]]


def test_internal_camera_cut_reaches_v3_without_creating_a_loop_segment(mods):
    raw = apple()
    raw["shots"][0]["events"] += raw["shots"][1]["events"]
    raw["shots"] = raw["shots"][:1]
    raw["shots"][0]["camera_cuts"] = [{"before_event": "event_008", "framing": "close_up"}]
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        raw, character_1=_char("one", "one actor", "male"),
        character_2=_char("two", "second actor", "female"))
    plan = json.loads(out[0])
    assert len(plan["shots"]) == 1
    assert "two shot" in plan["shots"][0]["prompt"][0]
    prompt = "\n".join(plan["shots"][0]["prompt"])
    assert "[Shot 2] close up" in prompt
    assert prompt.index("[Shot 2]") < prompt.index(raw["execution"]["directions"]["event_008"])
    assert flat(raw) == flat(mods["storyboard"].validate_storyboard_payload(raw))


@pytest.mark.parametrize("cut", [
    {"before_event": "event_999", "framing": "close_up"},
    {"before_event": "event_002", "framing": "invented"},
])
def test_invalid_internal_camera_cuts_are_rejected(mods, cut):
    raw = apple()
    raw["shots"][0]["camera_cuts"] = [cut]
    with pytest.raises(ValueError, match="camera_cut"):
        mods["storyboard"].validate_storyboard_payload(raw)


def test_v2_accepts_ordered_actions_and_seven_locked_dialogue_lines(mods):
    raw = apple()
    cleaned = mods["storyboard"].validate_storyboard_payload(raw)
    assert cleaned["version"] == 2
    assert flat(cleaned) == flat(raw)
    assert len([e for e in flat(cleaned) if e["type"] == "dialogue"]) == 7


@pytest.mark.parametrize("edit,match", [
    ("speaker", "source|event"), ("words", "source|event"),
    ("order", "source|event"), ("extra", "role|character"),
    ("camera", "camera|movement"), ("duration", "duration"),
])
def test_v2_rejects_changed_source_events_and_invalid_options(mods, edit, match):
    raw = apple()
    if edit == "speaker":
        raw["shots"][0]["events"][1]["character"] = "character_1"
    elif edit == "words":
        raw["shots"][0]["events"][1]["text"] = "要不要吃苹果呀？"
    elif edit == "order":
        raw["shots"][0]["events"].reverse()
    elif edit == "extra":
        raw["roles"].append(dict(raw["roles"][0], id="character_4"))
    elif edit == "camera":
        raw["camera"]["movement"] = "invented_orbit"
    else:
        raw["target_duration_seconds"] = 0
    with pytest.raises(ValueError, match=match):
        mods["storyboard"].validate_storyboard_payload(raw)


def test_v2_compiles_through_real_v3_without_changing_dialogue(mods):
    raw = apple()
    first = _char("tabby", "A tabby cat.", "male")
    second = _char("cream", "A cream cat.", "female")
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        raw, seed=77, character_1=first, character_2=second)
    plan = json.loads(out[0])
    text = "\n".join("\n".join(s["prompt"]) for s in plan["shots"])
    for e in flat(raw):
        expected = e["text"] if e["type"] == "dialogue" else raw["execution"]["directions"][e["id"]]
        assert expected in text
    assert "<Picture 1>" in text and "<Picture 2>" in text
    assert "(S2)" in text and "(S1)" in text
    assert all(a.get("audio") is None for a in out[4])
    delivered = sum(s["length"] for s in plan["shots"]) - 22
    assert abs(delivered / 24 - 20) <= 0.36


def test_v2_hedgehogs_remain_speech_not_extra_cast(mods):
    raw = json.loads((FIXTURES / "storyboard_v2_hedgehog.json").read_text(encoding="utf-8"))
    cleaned = mods["storyboard"].validate_storyboard_payload(raw)
    assert len(cleaned["roles"]) == 2
    assert all(e["type"] == "dialogue" for e in flat(cleaned))
    assert flat(cleaned) == flat(raw)


def test_single_short_dialogue_tail_is_not_given_audio_filler_time(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    raw = json.loads((FIXTURES / "storyboard_v2_hedgehog.json").read_text(encoding="utf-8"))
    events = [event for shot in raw["shots"] for event in shot["events"]]
    raw["shots"] = [
        {"id": "scene_01", "duration_seconds": 5, "events": events[:2]},
        {"id": "scene_02", "duration_seconds": 7, "events": events[2:4]},
        {"id": "scene_03", "duration_seconds": 12, "events": events[4:6]},
        {"id": "scene_04", "duration_seconds": 6, "events": events[6:]},
    ]
    frames = contract.allocate_frames(raw)
    assert frames[-1] <= 90
    delivered = sum(frames) - raw["continuation_overlap_frames"] * (len(frames) - 1)
    assert abs(delivered / 24 - raw["target_duration_seconds"]) <= 0.36


def test_v2_save_load_is_canonical_and_idempotent(mods, tmp_path):
    sb = mods["storyboard"]
    cleaned = sb.validate_storyboard_payload(apple())
    assert sb.validate_storyboard_payload(cleaned) == cleaned
    saved, path, _ = sb.save_storyboard(str(tmp_path), cleaned)
    loaded, _ = sb.load_storyboard(str(tmp_path), cleaned["name"])
    assert loaded == saved == cleaned
    assert json.loads(Path(path).read_text(encoding="utf-8"))["version"] == 2
    assert sb.storyboard_requirements(loaded)[-1]["required"] is False


def test_v2_action_only_shot_needs_no_fake_dialogue(mods):
    raw = apple()
    raw["roles"] = raw["roles"][:1]
    raw["source_script"] = "角色1动作：抬起一只手，轻轻挥手。"
    raw["target_duration_seconds"] = 4
    raw["shots"] = [{"id": "scene_01", "duration_seconds": 4, "events": [
        {"id": "event_001", "type": "action", "character": "character_1",
         "text": "抬起一只手，轻轻挥手。", "language": "zh"}]}]
    raw["execution"]["voices"] = {"character_1": "male"}
    raw["execution"]["directions"] = {"event_001": "Character 1 raises a hand and waves gently."}
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        raw, character_1=_char("one", "one actor", "voice"))
    text = out[0]
    assert "waves gently" in text and "<d>" not in text


def test_v2_explicit_entrance_is_not_overridden_by_first_frame_pins(mods):
    raw = apple()
    raw["roles"][1]["initially_visible"] = False
    raw["source_script"] = "角色1：你好。\n角色2入场：推门进来。\n角色2：你好。"
    raw["target_duration_seconds"] = 8
    raw["shots"] = [{"id": "scene_01", "duration_seconds": 8, "events": [
        {"id": "event_001", "type": "dialogue", "character": "character_1", "text": "你好。", "language": "zh"},
        {"id": "event_002", "type": "enter", "character": "character_2", "text": "推门进来。", "language": "zh"},
        {"id": "event_003", "type": "dialogue", "character": "character_2", "text": "你好。", "language": "zh"}]}]
    raw["execution"]["directions"] = {"event_002": "Character 2 opens the door and enters."}
    place = dict(_char("room", "A wide empty room.", ""), kind="scene", audio=None)
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        raw, character_1=_char("one", "actor one", "voice"),
        character_2=_char("two", "actor two", "voice"), scene_1=place)
    assert "Not visible at the opening: Character 2" in out[0]
    assert "Character 2 enters the frame" in out[0]
    assert "Keep them continuously visible from the first frame" not in out[0]


def test_portable_system_instruction_contains_valid_asset_example(mods):
    resource = Path(__file__).parents[1] / "nodes/llm/prompts/h3_storyboard/system_asset.txt"
    example = next(line for line in resource.read_text(encoding="utf-8").splitlines() if line.startswith('{"version":2'))
    asset = mods["storyboard"].validate_storyboard_payload(json.loads(example))
    assert asset["roles"][0]["presence"] == "off_screen"


def test_offscreen_position_matches_portable_schema(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    roles = contract.parse_roles(2, "角色1：旁白 | 画外\n角色2：听众 | 画面内")
    assert roles[0]["position"] == "none"
    raw = apple()
    raw["roles"][0].update(presence="off_screen", initially_visible=False, position="left")
    with pytest.raises(ValueError, match="off_screen.*position"):
        contract.validate_v2(raw)


def test_later_shots_do_not_reset_initial_blocking_after_source_actions(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    runtime = contract.to_runtime(apple())
    assert "Character 1 begins at left" in runtime["shots"][0]["prompt"]
    assert "begins at left" not in runtime["shots"][1]["prompt"]
    assert "exact written order" in runtime["shots"][1]["prompt"]


def test_real_save_and_load_node_bundles_feed_v3(mods, tmp_path):
    from nodes.assets.nodes import SaveStoryboardAsset, LoadStoryboardAsset, NEW_NAME_SENTINEL
    raw = apple()
    saved = SaveStoryboardAsset().save(
        library_dir=str(tmp_path), name=NEW_NAME_SENTINEL,
        new_name=raw["name"], json_text=json.dumps(raw))[1]
    loaded = LoadStoryboardAsset().load(str(tmp_path), raw["name"])[0]
    for bundle in (saved, loaded):
        assert "_path" in bundle
        out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
            bundle, character_1=_char("one", "actor one", "male"),
            character_2=_char("two", "actor two", "female"))
        assert len(json.loads(out[0])["shots"]) == 2
        assert "_path" not in out[0]


def test_image_only_character_loads_for_generated_voice_storyboard(monkeypatch):
    from nodes.assets import nodes
    from .test_asset_library import make_png
    monkeypatch.setattr(nodes.library, "load_character", lambda *args: {
        "name": "ImageOnly", "description": "One actor.", "voice": "male",
        "png_bytes_list": [make_png()], "audio_bytes_list": [], "audio_exts": []})
    result = nodes.LoadAssetV2().load(name="ImageOnly")
    assert result[3] is not None
    assert result[4] is None and result[5]["audio"] is None


def test_h3_execution_contains_chinese_only_inside_dialogue_tags(mods):
    import re
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    runtime = contract.to_runtime(apple())
    prompt = "\n".join(s["prompt"] for s in runtime["shots"])
    outside = re.sub(r"<d>.*?</d>", "", prompt, flags=re.S)
    assert re.search(r"[\u3400-\u9fff]", outside) is None
    assert "single red apple" in prompt
    assert "stated quantity" in prompt
    assert "incidental reference props with their initial holder" in prompt


def test_english_execution_cannot_omit_or_add_action_ids(mods):
    raw = apple()
    raw["execution"]["directions"].pop("event_001")
    with pytest.raises(ValueError, match="execution.*event|direction"):
        mods["storyboard"].validate_storyboard_payload(raw)


def test_english_direction_owner_must_match_source_actor(mods):
    raw = apple()
    raw["execution"]["directions"]["event_001"] = "Character 1 holds the red apple."
    with pytest.raises(ValueError, match="direction.*Character 2"):
        mods["storyboard"].validate_storyboard_payload(raw)


@pytest.mark.parametrize("field", ["visible_state", "direction"])
def test_negative_visual_instructions_must_be_rephrased_as_positive_state(mods, field):
    raw = apple()
    if field == "visible_state":
        raw["execution"][field] = "Character 1 does not eat the apple."
    else:
        raw["execution"]["directions"]["event_008"] = "Character 1 does not accept the apple."
    with pytest.raises(ValueError, match="positive.*state|positive.*physical"):
        mods["storyboard"].validate_storyboard_payload(raw)


def test_storyboard_warnings_flag_floor_budget_and_single_speaker_continuations(mods):
    from importlib import import_module
    import math
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    clean = apple()
    assert contract.storyboard_warnings(clean) == []

    lone = json.loads((FIXTURES / "storyboard_v2_hedgehog.json").read_text(encoding="utf-8"))
    events = [event for shot in lone["shots"] for event in shot["events"]]
    lone["shots"] = [
        {"id": "scene_01", "duration_seconds": 5, "events": events[:2]},
        {"id": "scene_02", "duration_seconds": 7, "events": events[2:4]},
        {"id": "scene_03", "duration_seconds": 12, "events": events[4:6]},
        {"id": "scene_04", "duration_seconds": 6, "events": events[6:]},
    ]
    warnings = contract.storyboard_warnings(lone)
    assert not any(w.startswith("scene_01") for w in warnings)
    for shot_id in ("scene_02", "scene_03", "scene_04"):
        assert any(w.startswith(shot_id) and "单一角色独占" in w for w in warnings)
    assert not any("零缓冲" in w for w in warnings)

    overlap, count = lone["continuation_overlap_frames"], len(lone["shots"])
    minimum = sum(max(0, math.ceil((max(1.0, contract.estimate_shot_seconds(s)) * 24
                                    + (overlap if i else 0) - 5) / 17))
                 for i, s in enumerate(lone["shots"]))
    lone["target_duration_seconds"] = (17 * minimum - overlap * (count - 1) + 5 * count) / 24
    assert any("零缓冲" in w for w in contract.storyboard_warnings(lone))


def test_storyboard_warnings_flag_a_shot_pinned_below_one_pause_unit(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    def event(eid, kind, who, text):
        return {"id": eid, "type": kind, "character": who, "text": text, "language": "zh"}
    raw = {"shots": [
        {"id": "scene_01", "duration_seconds": 8, "events": [
            event("event_001", "dialogue", "character_1", "你一个月玩手游充多少钱？"),
            event("event_001b", "action", "character_1", "他笑了。"),
            event("event_001c", "action", "character_1", "他点头。")]},
        {"id": "scene_02", "duration_seconds": 0.5, "events": [
            event("event_002", "action", "character_2", "他愣住。"),
            event("event_003", "dialogue", "character_2", "你好呀朋友们。"),
            event("event_004", "action", "character_2", "他挠头。")]},
    ], "target_duration_seconds": 12, "continuation_overlap_frames": 22}
    warnings = contract.storyboard_warnings(raw)
    assert any(w.startswith("scene_02") and "预算贴地" in w for w in warnings)
    assert not any(w.startswith("scene_01") for w in warnings)


def test_voice_trailing_punctuation_is_normalized_before_splicing(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    raw = apple()
    raw["roles"][0]["voice"] = "adult male, calm professional tenor."
    raw["execution"]["voices"]["character_1"] = "adult male, calm professional tenor."
    clean = mods["storyboard"].validate_storyboard_payload(raw)
    assert clean["roles"][0]["voice"] == "adult male, calm professional tenor"
    runtime = contract.to_runtime(raw)
    spoken = " ".join(line for line in runtime["shots"][0]["prompt"].splitlines() if "speaks as" in line)
    assert "tenor: <d>" in spoken
    assert "tenor.: <d>" not in spoken


def test_both_storyboard_system_prompts_carry_the_render_review_rules():
    root = Path(__file__).parents[1]
    docs_prompt = (root / "docs" / "storyboard_asset_system_prompt.txt").read_text(encoding="utf-8")
    node_prompt = (root / "nodes/llm/prompts/h3_storyboard/system_asset.txt").read_text(encoding="utf-8")
    shared = [
        "FEWEST",
        "first_person",
        "vertical relation",
        "only phone in the room",
        "bare phrase with no trailing punctuation",
        "setting TEXT overrides the wired scene image",
    ]
    for rule in shared:
        assert rule in docs_prompt, "docs prompt missing: " + rule
        assert rule in node_prompt, "node prompt missing: " + rule
