"""Portable script cast/environment defaults resolve per connected socket."""
import copy
import json
import re
from pathlib import Path
import pytest
from .test_pov_offscreen import mods, _char
from .test_h3_storyboard_generator import sb
from .test_storyboard_generation_contract import apple, _scene

def designed():
    raw = apple()
    raw["character_designs"] = {"character_1": "红色外套的猫", "character_2": "蓝色帽子的猫"}
    raw["execution"]["character_designs"] = {"character_1": "A cat wearing a red coat.", "character_2": "A cat wearing a blue hat."}
    raw["scene"] = "月光下的花园"
    raw["execution"]["scene"] = "A moonlit garden."
    raw["light"] = "late_night"
    return raw

def run(mods, raw, **kwargs):
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(raw, **kwargs)
    plan = json.loads(out[0])
    text = "\n".join(plan.get("prompt_prefix", []) + ["\n".join(s["prompt"]) for s in plan["shots"]])
    return out, text

@pytest.mark.parametrize("slots", [(), (1,), (2,), (1, 2)])
@pytest.mark.parametrize("scene", [False, True])
def test_each_visual_socket_is_independently_optional(mods, slots, scene):
    kwargs = {"character_%d" % i: _char(str(i), "An actor.", "distinct voice") for i in slots}
    if scene:
        kwargs["scene_1"] = _scene("office", "A bright office.")
    out, text = run(mods, apple(), **kwargs)
    refs = {int(n) for n in re.findall(r"<Picture (\d+)>", text)}
    assert refs == set(range(1, len(slots) + int(scene) + 1))
    assert len(out[4]) == len(slots) + int(scene)
    if not slots and not scene:
        assert out[3] is None
    for i in (1, 2):
        assert ("Character %d stays exclusively off-screen" % i) not in text
    for i, slot in enumerate(slots, 1):
        assert "Character %d is the figure in <Picture %d>" % (slot, i) in text
    for shot in apple()["shots"]:
        for event in shot["events"]:
            assert (event["text"] if event["type"] == "dialogue" else apple()["execution"]["directions"][event["id"]]) in text

@pytest.mark.parametrize("slots", [(), (1,), (2,), (1, 2)])
@pytest.mark.parametrize("scene", [False, True])
def test_defaults_only_compile_for_unwired_visual_slots(mods, slots, scene):
    raw = designed()
    original = copy.deepcopy(raw)
    kwargs = {"character_%d" % i: _char(str(i), "An actor.", "distinct voice") for i in slots}
    if scene:
        kwargs["scene_1"] = _scene("office", "A bright office.")
    out, text = run(mods, raw, **kwargs)
    assert ("red coat" in text) == (1 not in slots)
    assert ("blue hat" in text) == (2 not in slots)
    assert ("moonlit garden" in text) == (not scene)
    assert ("Light: " in text) == (not scene)
    assert raw == original

def test_default_design_save_load_roundtrip_and_preview(mods, tmp_path):
    raw = designed()
    cleaned, path, status = mods["storyboard"].save_storyboard(str(tmp_path), raw)
    loaded, _ = mods["storyboard"].load_storyboard(str(tmp_path), raw["name"])
    assert loaded["character_designs"] == raw["character_designs"]
    assert loaded["execution"]["character_designs"] == raw["execution"]["character_designs"]
    assert "红色外套的猫" in mods["storyboard"].storyboard_preview(cleaned)

@pytest.mark.parametrize("case", ["unknown_role", "missing_translation", "extra_translation", "chinese_execution", "offscreen_design"])
def test_invalid_default_designs_rejected(mods, case):
    raw = designed()
    if case == "unknown_role":
        raw["character_designs"]["character_8"] = "猫"
    elif case == "missing_translation":
        del raw["execution"]["character_designs"]["character_1"]
    elif case == "extra_translation":
        raw["execution"]["character_designs"]["character_8"] = "A cat."
    elif case == "chinese_execution":
        raw["execution"]["character_designs"]["character_1"] = "猫"
    else:
        raw["roles"][0].update(presence="off_screen", initially_visible=False, position="none")
    with pytest.raises(ValueError, match="character_designs"):
        mods["storyboard"].validate_storyboard_payload(raw)

def test_reference_audio_still_requires_connected_assets(mods):
    raw = apple()
    raw["audio_mode"] = "reference"
    with pytest.raises(ValueError, match="requires a character asset"):
        run(mods, raw)

def test_v2_socket_requirements_report_optional_visuals(mods):
    requirements = mods["storyboard"].storyboard_requirements(apple())
    assert all(not r["required"] for r in requirements)

def test_schema_accepts_designs_and_scene_light(mods):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1] / "nodes/llm/prompts/h3_storyboard/asset.schema.json").read_text("utf-8"))
    jsonschema.Draft202012Validator(schema).validate(designed())


def test_generator_locks_default_designs_and_preview(sb, mods):
    from .test_h3_storyboard_generator import ScriptedConnector
    from .test_storyboard_generation_node import options
    raw = designed()
    connector = ScriptedConnector([json.dumps(raw, ensure_ascii=False)])
    asset = sb.AssetStoryboardEnhancer(connector)(
        raw["source_script"], 2, character_designs=json.dumps(raw["character_designs"]),
        scene=raw["scene"], light=raw["light"], **options(raw))
    assert asset["character_designs"] == raw["character_designs"]
    payload = json.loads(connector.calls[0][1]["content"])
    assert payload["asset_header"]["character_designs"] == raw["character_designs"]
    assert "红色外套的猫" in sb.preview_asset(asset)


def test_both_system_instructions_define_optional_visuals():
    root = Path(__file__).resolve().parents[1]
    for name in ["docs/script_asset_system_prompt.txt", "nodes/llm/prompts/h3_storyboard/system_asset.txt"]:
        text = (root / name).read_text("utf-8")
        assert "on-screen characters are REQUIRED" not in text
        assert "character_designs" in text
        assert "Each visual socket is independently OPTIONAL" in text


def test_sparse_wiring_matches_both_assemblers_and_actual_picture_sockets(mods):
    from nodes.assets.nodes import AssetToSlots
    actor = _char("second", "An actor.", "distinct voice")
    room = _scene("office", "A bright office.")
    kwargs = dict(character_2=actor, scene_1=room)
    out, text = run(mods, designed(), **kwargs)
    other = mods["builder"].BuildPlanFromStoryboard().build(designed(), **kwargs)
    assert out[0] == other[0]
    slots = AssetToSlots().split(out[4])
    assert slots[0] == 2
    assert slots[1] is actor["image"] and slots[2] is room["image"]
    assert "Character 2 is the figure in <Picture 1>" in text
    assert "Scene reference: <Picture 2>" in text


def test_standalone_compiler_does_not_invent_portraits(mods):
    from importlib import import_module
    contract = import_module(mods["storyboard"].__package__ + ".storyboard_contract")
    runtime = contract.to_runtime(designed())
    assert all("<Picture " not in shot["prompt"] for shot in runtime["shots"])


@pytest.mark.parametrize("value", ["[1]", "invalid-json", {"character_1": 7}, {1: "a cat"}])
def test_generator_rejects_invalid_default_designs_before_provider(sb, value):
    from .test_h3_storyboard_generator import ScriptedConnector
    connector = ScriptedConnector([])
    with pytest.raises(ValueError, match="character_designs"):
        sb.AssetStoryboardEnhancer(connector)("an idea", 0, mode="concept", character_designs=value)
    assert not connector.calls


def test_v3_is_changed_is_pure_content(mods):
    """2026-11-01 final ruling: the assembly node carries NO seed/overlap
    widgets — its digest is storyboard + wired bundles, nothing else."""
    builder = mods["builder"]
    a = builder.MiniMaxH3LoopPromptGeneratorV3.IS_CHANGED(storyboard=None)
    b = builder.MiniMaxH3LoopPromptGeneratorV3.IS_CHANGED(storyboard=None)
    assert a == b
    req = builder.MiniMaxH3LoopPromptGeneratorV3.INPUT_TYPES()["required"]
    assert sorted(req) == ["storyboard"]


def test_v2_plans_never_carry_seeds(mods):
    """2026-11-01 ruling: the assembly node is PURE assembly — v2 plans
    carry no per-shot seeds at all (the chain Plan node's base_seed owns
    seeding; Review Gate rerolls single scenes)."""
    builder = mods["builder"]
    node = builder.MiniMaxH3LoopPromptGeneratorV3()
    out = node.generate_v3(apple(),
                           character_1=_char("甲", "First.", "male"),
                           character_2=_char("乙", "Second.", "female"))
    shots = json.loads(out[0])["shots"]
    assert shots and all("seed" not in s for s in shots)

