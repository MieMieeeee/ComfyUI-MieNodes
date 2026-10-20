"""The user's successful single-image prompt must survive V3 unchanged."""
import copy
import json
from pathlib import Path

import pytest
from .test_pov_offscreen import mods, _char

FIXTURES = Path(__file__).parent / "fixtures"


def board():
    return json.loads((FIXTURES / "storyboard_pov_authored_15s.json").read_text(encoding="utf-8"))


def character():
    return _char("Claude娘", "A blonde anime woman.", "a voice that must not be injected")


def test_authored_asset_keeps_six_lines_and_exact_prompt(mods):
    raw = board()
    cleaned = mods["storyboard"].validate_storyboard_payload(raw)
    assert cleaned["assembly_mode"] == "authored"
    assert cleaned["audio_mode"] == "generated"
    assert len(cleaned["shots"][0]["lines"]) == 6
    assert cleaned["shots"][0]["prompt"] == (FIXTURES / "pov_known_good_prompt.txt").read_text(encoding="utf-8")


def test_v3_reproduces_single_image_without_reference_audio_or_scene(mods):
    char = character()
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(board(), seed=123, character_2=char)
    plan = json.loads(out[0])
    assert plan["prompt_prefix"] == []
    assert len(plan["shots"]) == 1
    assert "\n".join(plan["shots"][0]["prompt"]) == board()["shots"][0]["prompt"]
    assert plan["shots"][0]["length"] == 362
    assert plan["shots"][0]["seed"] == "123"
    assert out[3].shape[0] == 1
    assert len(out[4]) == 1
    assert out[4][0]["image"] is char["image"]
    assert out[4][0].get("audio") is None
    assert char["audio"] is not None  # no mutation of the shared asset


def test_generated_voice_ignores_asset_voice_and_audio(mods):
    char = character()
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    first = node.generate_v3(board(), seed=123, character_2=char)
    char["audio"] = None
    char["voice"] = ""
    second = node.generate_v3(board(), seed=123, character_2=char)
    assert first[0] == second[0]


def test_authored_recipe_makes_scene_optional_and_voice_source_clear(mods):
    cleaned = mods["storyboard"].validate_storyboard_payload(board())
    recipe = mods["storyboard"].storyboard_requirements(cleaned)
    assert next(x for x in recipe if x["socket"] == "scene_1")["required"] is False
    assert "generated" in next(x for x in recipe if x["socket"] == "character_1")["note"].lower()


@pytest.mark.parametrize("edit", ["words", "lang", "missing_prompt", "missing_duration"])
def test_authored_prompt_must_match_its_declared_script(mods, edit):
    raw = board()
    if edit == "words":
        raw["shots"][0]["lines"][0]["text"] = "different words"
    elif edit == "lang":
        raw["shots"][0]["lines"][0]["lang"] = "zh"
    elif edit == "missing_prompt":
        del raw["shots"][0]["prompt"]
    else:
        del raw["shots"][0]["duration_seconds"]
    with pytest.raises(ValueError):
        mods["storyboard"].validate_storyboard_payload(raw)


def test_authored_generated_voice_keeps_both_speaker_tags_without_caller_asset(mods):
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(board(), seed=123, character_2=character())
    text = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    assert text.count("(S1)") == 3
    assert text.count("(S2)") == 3
    assert "subject_definitions:" not in text
    assert "The previous line finishes" not in text


def test_authored_generated_voice_also_drops_wired_offscreen_image_and_audio(mods):
    caller = character()
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        board(), character_1=caller, character_2=character())
    assert len(out[4]) == 2
    assert out[4][0]["image"] is None
    assert all(asset.get("audio") is None for asset in out[4])
    assert caller["image"] is not None and caller["audio"] is not None
    assert out[3].shape[0] == 1


def test_authored_requires_visible_character_image(mods):
    char = character()
    char["image"] = None
    with pytest.raises(ValueError, match="requires an image"):
        mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(board(), character_2=char)


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf")])
def test_authored_rejects_invalid_duration(mods, duration):
    raw = board()
    raw["shots"][0]["duration_seconds"] = duration
    with pytest.raises(ValueError, match="duration_seconds"):
        mods["storyboard"].validate_storyboard_payload(raw)


def scene():
    asset = character()
    asset.update(kind="scene", name="Cafe", description="A cafe with green walls and wooden tables.", audio=None)
    return asset


def test_authored_optional_scene_binds_reference_roles_without_rewriting_script(mods):
    raw, place, char = board(), scene(), character()
    original = copy.deepcopy(raw)
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        raw, character_2=char, scene_1=place)
    plan = json.loads(out[0])
    context = "\n".join(plan["prompt_prefix"])
    assert "<Picture 2>" in context and "only the environment, materials and lighting" in context
    assert "<Picture 1>" in context and "appearance only" in context
    assert "not use their backgrounds" in context
    assert place["description"] not in context
    assert "Cloud is the on-screen character shown in <Picture 1>" in context
    assert "does not define camera position, shot size or character blocking" in context
    assert "continuously visible from the first frame" in context
    assert "\n".join(plan["shots"][0]["prompt"]) == raw["shots"][0]["prompt"]
    assert raw == original
    assert len(out[4]) == 2
    assert out[4][0]["image"] is char["image"]
    assert out[4][1]["image"] is place["image"]
    assert all(asset.get("audio") is None for asset in out[4])


def test_authored_scene_picture_number_ignores_offscreen_roster_entry(mods):
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        board(), character_1=character(), character_2=character(), scene_1=scene())
    prefix = "\n".join(json.loads(out[0])["prompt_prefix"])
    assert "<Picture 2>" in prefix
    assert "<Picture 3>" not in prefix
    assert out[4][0]["image"] is None
    assert len(out[4]) == 3


def test_authored_scene_reference_follows_all_visible_character_pictures(mods):
    raw = board()
    raw["cast"]["Listener"] = {"character_slot": 3}
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(
        raw, character_2=character(), character_3=character(), scene_1=scene())
    prefix = "\n".join(json.loads(out[0])["prompt_prefix"])
    assert "Scene reference: <Picture 3>" in prefix
    assert "<Picture 1>, <Picture 2>" in prefix
    assert "Cloud is the on-screen character shown in <Picture 1>" in prefix
    assert "Listener is the on-screen character shown in <Picture 2>" in prefix


def test_authored_scene_source_framing_cannot_become_global_direction(mods):
    place = scene()
    place["description"] = "A cafe interior fills the frame. In the foreground, three tables fill a wide shot."
    original = copy.deepcopy(place["description"])
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    first = node.generate_v3(board(), character_2=character(), scene_1=place)
    place["description"] = "A garden seen from above, with a fountain in the lower-right corner."
    second = node.generate_v3(board(), character_2=character(), scene_1=place)
    assert json.loads(first[0]) == json.loads(second[0])
    assert first[4][-1]["description"] == original
    assert "fills the frame" not in first[0]
    assert "spatial layout" not in first[0]


def test_authored_optional_scene_can_be_removed_without_leaving_context(mods):
    raw = board()
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    with_scene = node.generate_v3(raw, character_2=character(), scene_1=scene())
    without_scene = node.generate_v3(raw, character_2=character())
    assert json.loads(with_scene[0])["prompt_prefix"]
    assert json.loads(without_scene[0])["prompt_prefix"] == []
    assert "\n".join(json.loads(without_scene[0])["shots"][0]["prompt"]) == raw["shots"][0]["prompt"]
