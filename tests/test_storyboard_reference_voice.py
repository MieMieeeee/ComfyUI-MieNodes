"""Reference voices must survive V3 without leaking off-screen portraits."""
import json
import pytest
from .test_pov_offscreen import mods, _char
from .test_storyboard_generation_contract import FIXTURES

def reference_board():
    raw = json.loads((FIXTURES / "storyboard_v2_hedgehog.json").read_text(encoding="utf-8"))
    raw["audio_mode"] = "reference"
    raw["roles"][0].update(presence="off_screen", initially_visible=False, position="none")
    return raw

def test_reference_mode_preserves_audio_in_role_order_and_excludes_caller_image(mods):
    caller, girl = _char("caller", "caller", "male"), _char("girl", "girl", "female")
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        reference_board(), character_1=caller, character_2=girl)
    roster = out[4]
    assert roster[0]["image"] is None
    assert roster[0]["audio"] is caller["audio"]
    assert roster[1]["image"] is girl["image"]
    assert roster[1]["audio"] is girl["audio"]
    prompt = out[0]
    assert "(S1)" in prompt and "<Audio 1>" in prompt
    assert "(S2)" in prompt and "<Audio 2>" in prompt
    assert caller["image"] is not None

@pytest.mark.parametrize("missing", ["asset", "audio"])
def test_reference_mode_rejects_a_missing_caller_voice(mods, missing):
    caller = _char("caller", "caller", "male")
    if missing == "audio":
        caller["audio"] = None
    with pytest.raises(ValueError, match="character_1|slot 1"):
        mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
            reference_board(), character_1=None if missing == "asset" else caller,
            character_2=_char("girl", "girl", "female"))

def test_reference_mode_is_visible_in_asset_requirements(mods):
    requirements = mods["storyboard"].storyboard_requirements(reference_board())
    caller = next(r for r in requirements if r["socket"] == "character_1")
    assert caller["required"]
    assert "reference" in caller["note"].lower()

def test_generator_reference_mode_is_a_locked_caller_option(mods):
    import importlib
    module = importlib.import_module(mods["builder"].__package__ + ".storyboard_asset_generator")
    class Connector:
        def invoke(self, messages, **kwargs):
            request = json.loads(messages[1]["content"])
            assert request["asset_header"]["audio_mode"] == "reference"
            raise RuntimeError("header inspected before provider request")
    with pytest.raises(RuntimeError, match="header inspected"):
        module.AssetStoryboardEnhancer(Connector())(
            "角色1：你好。\n角色2：你好。", 1, target_duration_seconds=5, audio_mode="reference")

def test_reference_roster_reaches_actual_image_and_audio_sockets_in_order(mods):
    from nodes.assets.nodes import AssetToSlots
    caller, girl = _char("caller", "caller", "male"), _char("girl", "girl", "female")
    room = dict(_char("room", "room", ""), kind="scene", audio=None)
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        reference_board(), character_1=caller, character_2=girl, scene_1=room)
    slots = AssetToSlots().split(out[4])
    assert slots[0] == 2
    assert slots[1] is girl["image"] and slots[2] is room["image"]
    assert slots[10] == 2
    assert slots[11] is caller["audio"] and slots[12] is girl["audio"]

@pytest.mark.parametrize("mode", ["invalid", None, True])
def test_unknown_audio_mode_is_rejected(mods, mode):
    raw = reference_board()
    raw["audio_mode"] = mode
    with pytest.raises(ValueError, match="audio_mode"):
        mods["storyboard"].validate_storyboard_payload(raw)

def test_visible_cast_is_bound_in_the_first_shot_sentence_before_room_context(mods):
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        reference_board(), character_1=_char("caller", "caller", "male"),
        character_2=_char("girl", "girl", "female"))
    opening = json.loads(out[0])["shots"][0]["prompt"][0]
    assert "Character 2" in opening and "<Picture 1>" in opening
    assert "first frame" in opening
    assert "Character 1" not in opening

def test_generator_cannot_invent_reference_mode_when_generated_was_selected(mods):
    import importlib
    module = importlib.import_module(mods["builder"].__package__ + ".storyboard_asset_generator")
    class Connector:
        def invoke(self, messages, **kwargs):
            req = json.loads(messages[1]["content"])
            candidate = dict(req["asset_header"], audio_mode="reference", shots=[{
                "id":"scene_01", "duration_seconds":5, "events":req["locked_events"]}])
            return json.dumps(candidate)
    with pytest.raises(RuntimeError, match="audio_mode"):
        module.AssetStoryboardEnhancer(Connector())(
            "角色1：你好。\n角色2：你好。", 1, target_duration_seconds=5, audio_mode="generated")

def test_offscreen_dialogue_keeps_the_visible_listener_in_the_same_sentence(mods):
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        reference_board(), character_1=_char("caller", "caller", "male"),
        character_2=_char("girl", "girl", "female"))
    lines = json.loads(out[0])["shots"][0]["prompt"]
    offscreen_line = next(line for line in lines if "(S1)" in line and "<d>" in line)
    assert offscreen_line.startswith("Character 2")
    assert "on-screen" in offscreen_line and "closed" in offscreen_line
