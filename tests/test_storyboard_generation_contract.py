"""Contract tests for user-authored role events and portable storyboard assets."""
import copy
import json
import sys
from pathlib import Path

import pytest
from .test_pov_offscreen import mods, _char

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT_DIR = Path(__file__).resolve().parents[1]


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
    loaded = LoadStoryboardAsset().load(str(tmp_path), raw["name"])["result"][0]
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


# (removed 2026-10-29: superseded by tests/test_script_asset_prompt_sync.py,
#  which enforces the full CRAFT RULES section equality plus a shared
#  marker list — a stronger guard than this per-phrase check)


def test_speech_budget_warning_flags_a_clip_packed_for_a_slow_delivery(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    def event(eid, kind, who, text):
        return {"id": eid, "type": kind, "character": who, "text": text, "language": "zh"}
    raw = {"shots": [{"id": "scene_01", "duration_seconds": 6.0, "events": [
        event("event_001", "dialogue", "character_1", "你看见楼下停的那辆奔驰了吗"),
        event("event_002", "dialogue", "character_1", "现在要结婚了给我送来了请帖要我随礼"),
    ]}], "target_duration_seconds": 6.0, "continuation_overlap_frames": 22}
    warnings = contract.storyboard_warnings(raw)
    assert any("慢速语速" in w for w in warnings)
    # the same content with room to breathe is clean
    raw["shots"][0]["duration_seconds"] = 14.0
    raw["target_duration_seconds"] = 14.0
    assert not any("慢速语速" in w for w in contract.storyboard_warnings(raw))


def test_bundled_assets_answer_names_the_user_library_does_not_have(mods, tmp_path, monkeypatch):
    """Assets shipped inside the plugin are a fallback, never an override."""
    import importlib
    import shutil
    from importlib import import_module
    library = importlib.import_module(mods["storyboard"].__package__ + ".library")
    storyboard = importlib.import_module(mods["storyboard"].__package__ + ".storyboard")
    plugin_root = tmp_path / "plugin"
    (plugin_root / "asset_library" / "characters").mkdir(parents=True)
    (plugin_root / "bundled_assets" / "characters" / "内置猫").mkdir(parents=True)
    (plugin_root / "bundled_assets" / "characters" / "内置猫" / "images").mkdir()
    # a character entry needs at least one real PNG to pass the library's own check
    png = next(Path("C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/custom_nodes/"
                    "comfyui_mienodes/asset_library/characters").rglob("*.png"), None)
    if png is not None:
        shutil.copy(png, plugin_root / "bundled_assets" / "characters" / "内置猫" / "images" / "01.png")
    (plugin_root / "bundled_assets" / "characters" / "内置猫" / "character.json").write_text(
        json.dumps({"name": "内置猫", "description": "bundled probe", "voice": "young female"},
                   ensure_ascii=False), encoding="utf-8")
    (plugin_root / "bundled_assets" / "storyboards" / "内置分镜").mkdir(parents=True)
    (plugin_root / "bundled_assets" / "storyboards" / "内置分镜" / "storyboard.json").write_text(
        json.dumps(apple(), ensure_ascii=False), encoding="utf-8")
    # the storyboard's name must match its directory
    payload = json.loads((plugin_root / "bundled_assets" / "storyboards" / "内置分镜"
                          / "storyboard.json").read_text(encoding="utf-8"))
    payload["name"] = "内置分镜"
    (plugin_root / "bundled_assets" / "storyboards" / "内置分镜" / "storyboard.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(library, "PLUGIN_ROOT", plugin_root)
    monkeypatch.setattr(storyboard, "_library", library)
    names = library.list_characters("", plugin_root=plugin_root).get("names", "").splitlines()
    assert "内置猫" in names
    sb_names = storyboard.list_storyboards("").get("names", "").splitlines()
    assert "内置分镜" in sb_names


# --------------------------------------------------------------------------- #
# 2026-10-28 review fixes: scene ambience on the v2 track, script-asset
# delete, skip-status surfacing, bundled cache key, constraints preview.
# --------------------------------------------------------------------------- #
def _scene(name, description, ambience=""):
    import torch
    return {"kind": "scene", "name": name, "description": description,
            "ambience": ambience, "image": torch.zeros(1, 8, 8, 3)}


def test_v2_scene_ambience_replaces_the_fallback_soundscape(mods):
    raw = apple()
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    actors = dict(character_1=_char("角色1", "First actor.", "male voice"),
                  character_2=_char("角色2", "Second actor.", "female voice"))
    out = node.generate_v3(
        raw, scene_1=_scene("客厅", "A cozy living room.",
                            "Quiet room tone with soft indoor air, no music"),
        **actors)
    joined = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    assert ("overall_soundscape: Quiet room tone with soft indoor air, "
            "no music") in joined
    assert "Natural room tone, breathing" not in joined
    assert "Quiet room tone" in out[1]  # the run summary names the resolved line

    out2 = node.generate_v3(raw, scene_1=_scene("客厅", "A cozy living room."),
                            **actors)
    joined2 = "\n".join(json.loads(out2[0])["shots"][0]["prompt"])
    assert "Natural room tone, breathing" in joined2  # no ambience -> fallback


def test_preview_shows_constraints_marked_not_compiled(mods):
    cleaned = mods["storyboard"].validate_storyboard_payload(apple())
    text = mods["storyboard"].storyboard_preview(cleaned)
    assert "全程持有苹果" in text
    assert "不进编译" in text


def test_delete_storyboard_removes_the_asset_directory(mods, tmp_path):
    lib = mods["storyboard"]
    cleaned, _path, status = lib.save_storyboard(str(tmp_path), apple())
    assert status == "created"
    result = lib.delete_storyboard(str(tmp_path), cleaned["name"])
    assert result["ok"] is True and result["kind"] == "storyboard"
    assert result["deleted_files"] >= 1
    with pytest.raises(FileNotFoundError):
        lib.delete_storyboard(str(tmp_path), cleaned["name"])


def test_augment_payload_surfaces_storyboard_skip_reasons(mods, tmp_path):
    lib = mods["storyboard"]
    bad = tmp_path / "storyboards" / "坏剧本"
    bad.mkdir(parents=True)
    (bad / "storyboard.json").write_text("{ not json", encoding="utf-8")
    payload = lib.augment_assets_payload({"ok": True, "error": None}, str(tmp_path))
    assert payload["storyboards"] == []
    assert "坏剧本" in payload["error"]
    payload2 = lib.augment_assets_payload({"ok": True, "error": "prior note"},
                                          str(tmp_path))
    assert "prior note" in payload2["error"] and "坏剧本" in payload2["error"]


def test_cache_key_covers_the_bundled_fallback(mods, tmp_path, monkeypatch):
    lib = mods["storyboard"]
    bundled = tmp_path / "bundled"
    target = bundled / "storyboards" / apple()["name"]
    target.mkdir(parents=True)
    (target / "storyboard.json").write_text(
        json.dumps(apple(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(lib._library, "bundled_library_root",
                        lambda plugin_root=None: bundled)
    key = lib.storyboard_cache_key(str(tmp_path / "user_lib"), apple()["name"])
    assert key != lib._library.MISSING_CACHE_MARKER


# --------------------------------------------------------------------------- #
# 2026-10-29 ruling: scene is its own field (dropped when a scene asset is
# wired), light is a preset selector, setting narrows to staging.
# --------------------------------------------------------------------------- #
def _v2_with_scene_light(apple_base, light="night_screen", **extra):
    raw = apple_base()
    raw["setting"] = "第一镜两人站在桌前；第二镜甲后退半步。"
    raw["scene"] = "一间深夜的办公室，靠窗工位上有亮着的显示器。"
    raw["light"] = light
    if light == "custom":
        raw["light_text"] = "凌晨深夜，窗外漆黑，只有暖台灯。"
    raw["execution"]["setting"] = ("First shot both stand at the desk; second shot "
                                   "Character 1 steps half a pace back.")
    raw["execution"]["scene"] = "A late-night office with a lit monitor by the window."
    if light == "custom":
        raw["execution"]["light"] = "Pitch-dark night sky; lit by a warm desk lamp only."
    raw.update(extra)
    return raw


def test_v2_scene_line_present_without_a_scene_asset(mods):
    raw = _v2_with_scene_light(apple)
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    out = node.generate_v3(raw, character_1=_char("甲", "First.", "male"),
                           character_2=_char("乙", "Second.", "female"))
    joined = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    assert "Scene context: A late-night office" in joined
    assert "Opening staging: First shot both stand" in joined
    assert "Light: Night: the window shows a pitch-dark sky" in joined


def test_v2_wired_scene_asset_overrides_the_script_scene(mods):
    raw = _v2_with_scene_light(apple)
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    out = node.generate_v3(raw, character_1=_char("甲", "First.", "male"),
                           character_2=_char("乙", "Second.", "female"),
                           scene_1=_scene("办公室", "An office.",
                                          "Quiet room tone with soft indoor air"))
    joined = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    # the script's scene prose is mechanically DROPPED...
    assert "Scene context:" not in joined
    # ...and so is the light preset (2026-10-31 principle: the wired scene
    # asset's own lighting wins)...
    assert not any(line.startswith("Light:") for line in joined.split("\n"))
    # ...while staging and the ambience swap survive
    assert "Opening staging: First shot both stand" in joined
    assert "overall_soundscape: Quiet room tone with soft indoor air" in joined
    # the run summary says the override happened
    assert "overridden" in out[1]
    assert "DROPPED" in out[1]


def test_v2_custom_light_compiles_from_execution(mods):
    raw = _v2_with_scene_light(apple, light="custom")
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    out = node.generate_v3(raw, character_1=_char("甲", "First.", "male"),
                           character_2=_char("乙", "Second.", "female"))
    joined = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    assert "Light: Pitch-dark night sky; lit by a warm desk lamp only." in joined
    assert "night_screen" not in joined  # no preset sentence leaked in


def test_v2_light_contract_errors(mods):
    with pytest.raises(ValueError, match="light_text"):
        validate = mods["storyboard"].validate_storyboard_payload
        validate(_v2_with_scene_light(apple, light="custom", light_text=""))
    with pytest.raises(ValueError, match="light_text is only valid"):
        validate(_v2_with_scene_light(apple, light_text="凌晨深夜，窗外漆黑。"))


# --------------------------------------------------------------------------- #
# 2026-10-30 live flow-backs: shouting-rate warning, same-speaker burst
# warning, first-mismatch detail in the verbatim error, dialogue-rewrite ⚠.
# --------------------------------------------------------------------------- #
def _minimal_asset(events, duration=12.0):
    return {"shots": [{"id": "scene_01", "duration_seconds": duration,
                       "events": events}],
            "target_duration_seconds": duration, "continuation_overlap_frames": 22}


def test_shouting_exclamation_ratio_warning(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    def event(eid, kind, who, text):
        return {"id": eid, "type": kind, "character": who, "text": text, "language": "zh"}
    angry = _minimal_asset([
        event("event_001", "dialogue", "character_1", "我逼逼咄人！"),
        event("event_002", "dialogue", "character_1", "搞笑吧你！"),
        event("event_003", "dialogue", "character_1", "你每次都站在高德地图制高点！"),
        event("event_004", "dialogue", "character_1", "你了不起！"),
        event("event_005", "dialogue", "character_1", "你高清！"),
        event("event_006", "action", "character_2", "翻白眼，摊手"),
    ])
    warnings = contract.storyboard_warnings(angry)
    assert any("怒斥语速" in w and "2.5" in w for w in warnings)
    calm = _minimal_asset([
        event("event_001", "dialogue", "character_1", "我逼逼咄人"),
        event("event_002", "dialogue", "character_1", "搞笑吧你"),
        event("event_003", "action", "character_2", "翻白眼，摊手"),
    ])
    assert not any("怒斥语速" in w for w in contract.storyboard_warnings(calm))


def test_same_speaker_burst_lip_drift_warning(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    def event(eid, kind, who, text):
        return {"id": eid, "type": kind, "character": who, "text": text, "language": "zh"}
    burst = _minimal_asset([
        event("event_00%d" % i, "dialogue", "character_1", "台词%d" % i) for i in range(1, 6)
    ] + [event("event_006", "action", "character_2", "翻白眼")])
    warnings = contract.storyboard_warnings(burst)
    assert any("连珠炮" in w and "口型" in w for w in warnings)
    four = _minimal_asset([
        event("event_00%d" % i, "dialogue", "character_1", "台词%d" % i) for i in range(1, 5)
    ] + [event("event_005", "action", "character_2", "翻白眼")])
    assert not any("连珠炮" in w for w in contract.storyboard_warnings(four))


def test_verbatim_error_names_the_first_mismatch(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    raw = apple()
    raw["shots"][0]["events"][3]["text"] = "你了不起了！"
    with pytest.raises(ValueError, match="source events differ at #4.*你了不起"):
        contract.validate_v2(raw)


def test_asset_diff_marks_dialogue_rewrites_with_a_warning(mods):
    import importlib.util
    import types as _types
    if "_mienodes_internal.nodes.llm.storyboard_asset_generator" not in sys.modules:
        pkg = _types.ModuleType("_mienodes_internal.nodes.llm")
        pkg.__path__ = [str(PROJECT_DIR / "nodes" / "llm")]
        pkg.__package__ = "_mienodes_internal.nodes.llm"
        sys.modules["_mienodes_internal.nodes.llm"] = pkg
        spec = importlib.util.spec_from_file_location(
            "_mienodes_internal.nodes.llm.storyboard_asset_generator",
            str(PROJECT_DIR / "nodes" / "llm" / "storyboard_asset_generator.py"))
        generator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = generator
        spec.loader.exec_module(generator)
    else:
        generator = sys.modules["_mienodes_internal.nodes.llm.storyboard_asset_generator"]
    old = apple()
    new = json.loads(json.dumps(old))
    flat_old = [e for s in old["shots"] for e in s["events"]]
    target = next(e for e in flat_old if e["type"] == "dialogue")
    for s in new["shots"]:
        for e in s["events"]:
            if e["id"] == target["id"]:
                e["text"] = "了不起了！"
    new["source_script"] = old["source_script"].replace(target["text"], "了不起了！")
    diff = "\n".join(generator.asset_diff(old, new))
    assert "⚠ 台词改写" in diff and "了不起了" in diff


def test_wired_scene_plus_light_preset_warns_text_beats_image(mods):
    """2026-10-31 live (LAN cafe): an explicit light preset overrides the
    wired scene image's own lighting — the run summary must say so."""
    raw = dict(apple())
    raw["light"] = "late_night"
    out = mods["builder"].assemble_plan(
        storyboard=raw, seed_base=7,
        slot_bundles={
            1: _char("甲", "First.", "male voice"),
            2: _char("乙", "Second.", "female voice"),
            9: _scene("咖啡馆-真实", "A bright cafe.")})
    joined = "\n".join(out["summary_lines"])
    prompt = "\n".join(out["plan"]["shots"][0]["prompt"])
    assert "light preset 'late_night' is DROPPED" in joined
    assert not any(line.startswith("Light:") for line in prompt.split("\n"))
    raw2 = dict(apple())
    raw2.pop("light", None)
    out2 = mods["builder"].assemble_plan(
        storyboard=raw2, seed_base=7,
        slot_bundles={
            1: _char("甲", "First.", "male voice"),
            2: _char("乙", "Second.", "female voice"),
            9: _scene("咖啡馆-真实", "A bright cafe.")})
    assert "light preset" not in "\n".join(out2["summary_lines"])


def test_wired_character_asset_overrides_script_voice_copy(mods):
    """2026-10-31 principle: the wired character asset owns the cast — its
    voice copy replaces the script's voice requirement in the prompt."""
    raw = dict(apple())
    out = mods["builder"].assemble_plan(
        storyboard=raw, seed_base=7,
        slot_bundles={
            1: _char("甲", "First.", "warm gravelly elder voice from the ASSET"),
            2: _char("乙", "Second.", "asset voice for second")})
    prompt = "\n".join(out["plan"]["shots"][0]["prompt"])
    assert "warm gravelly elder voice from the ASSET" in prompt
    assert "asset voice for second" in prompt
    joined = "\n".join(out["summary_lines"])
    assert "overrides the script's voice copy" in joined
    # 剧本原文音色不得残留
    script_voice = raw["execution"]["voices"]["character_1"]
    if script_voice:
        assert script_voice not in prompt


def test_read_meta_returns_sectioned_preview(mods, tmp_path):
    """2026-11-01: the load node shows the FULL content the moment a name
    is picked — /meta returns one section per display column."""
    lib = mods["storyboard"]
    cleaned, _path, _status = lib.save_storyboard(str(tmp_path), apple())
    meta = lib.read_storyboard_meta(str(tmp_path), cleaned["name"])
    titles = [s["title"] for s in meta["sections"]]
    assert any("概要" in t for t in titles)
    overview = next(s for s in meta["sections"] if "概要" in s["title"])
    assert overview["lines"][0].startswith("剧情（对白连读）")
    assert "「" in overview["lines"][0]
    assert any("角色" in t for t in titles)
    assert any("事件" in t for t in titles)
    assert any(t.startswith("画风 · look（默认画风") for t in titles)
    assert any(t.startswith("时长 · Duration") for t in titles)
    assert any(t.startswith("相机 · Camera") for t in titles)
    assert not any("续接重叠" in line for s in meta["sections"] for line in s["lines"])
    assert any(t.startswith("场景 · scene（默认环境") for t in titles)
    assert any(t.startswith("光 · light（默认时段") for t in titles)
    cast = next(s for s in meta["sections"] if "角色 · Cast" in s["title"])
    assert any(line.startswith("角色1（character_1）") for line in cast["lines"])
    assert any("约束" in t and "不进编译" in t for t in titles)
    events = next(s for s in meta["sections"] if "事件" in s["title"])
    assert any("我逼逼" not in line and "苹果" in line or "递" in line for line in events["lines"])


def test_multiline_defaults_fully_yield_to_a_wired_scene(mods):
    """2026-11-01 third-party review P1: multi-line look / scene / custom
    light must vanish COMPLETELY when a scene asset is wired — the old
    per-line strip leaked every line after the first."""
    def build(kind):
        raw = _v2_with_scene_light(apple)
        raw["look"] = "画风第一行\n画风第二行"
        raw["execution"]["look"] = "Style line one.\nStyle line two."
        raw["scene"] = "喷泉广场第一行\n喷泉广场第二行"
        raw["execution"]["scene"] = "A fountain plaza line one.\nLine two."
        raw["light"] = "custom"
        raw["light_text"] = "红光第一行\n红光第二行"
        raw["execution"]["light"] = "Red glow line one.\nRed glow line two."
        node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
        bundles = {"character_1": _char("甲", "First.", "male"),
                   "character_2": _char("乙", "Second.", "female")}
        if kind == "scene":
            bundles["scene_1"] = _scene("广场", "A plaza.", "soft air")
        out = node.generate_v3(raw, **bundles)
        return "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    wired = build("scene")
    for residue in ("Style line one", "Style line two", "fountain plaza",
                    "Line two", "Red glow line one", "Red glow line two",
                    "Scene context:", "Light:"):
        assert residue not in wired, residue
    # 不接场景：三段默认完整保留（含多行）
    plain = build("none")
    for kept in ("Style line one", "Style line two", "fountain plaza",
                 "Red glow line one"):
        assert kept in plain, kept
    assert "Scene context:" in plain and "Light:" in plain
