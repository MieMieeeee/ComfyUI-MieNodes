"""Generation rejects default scenery smuggled into unconditional performance copy."""
import copy
import importlib
import json
from pathlib import Path
import pytest
from .test_pov_offscreen import mods
from .test_h3_storyboard_generator import sb, ScriptedConnector
from .test_storyboard_generation_node import options
from .test_storyboard_generation_contract import apple

def reviewer(mods):
    return importlib.import_module(mods["storyboard"].__package__ + ".storyboard_review")

@pytest.mark.parametrize("field,text", [
    ("visible_state", " Bright afternoon daylight comes in from one side."),
    ("directions.event_001", " Character 1 keeps typing in the lamplight."),
    ("visible_state", " Both actors stand against a deep-night urban backdrop."),
    ("setting", " The room is illuminated by warm lamplight."),
    ("directions.event_001", " Neon lighting fills the scene."),
    ("visible_state", " Character 1 has blue eyes and blonde hair."),
])
def test_default_visuals_in_performance_fields_are_reported(mods, field, text):
    raw = apple()
    target = raw["execution"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] += text
    issues = reviewer(mods).review_asset_independence(raw)
    assert any(i["path"] == "execution." + field for i in issues)

def test_props_spatial_relations_and_dialogue_are_not_defaults(mods):
    raw = apple()
    raw["execution"]["visible_state"] += " A plot-required dark sedan is directly below the viewing position, seen from above. Character 1 looks down at the phone."
    raw["shots"][0]["events"][1]["text"] = "I wore a red coat at midnight in a neon-lit city."
    assert reviewer(mods).review_asset_independence(raw) == []

def test_copying_a_declared_default_back_into_visible_state_is_reported(mods):
    raw = apple()
    raw["execution"]["scene"] = "A garden with a marble fountain."
    raw["execution"]["visible_state"] += " A garden with a marble fountain."
    issues = reviewer(mods).review_asset_independence(raw)
    assert any(i["path"] == "execution.visible_state" and i["target"] == "scene" for i in issues)

def test_generator_repairs_visual_pollution_before_returning(sb, mods):
    good, bad = apple(), apple()
    bad["execution"]["visible_state"] += " Both actors stand against a deep-night urban backdrop."
    conn = ScriptedConnector([json.dumps(bad), json.dumps(good)])
    result = sb.AssetStoryboardEnhancer(conn)(good["source_script"], 2, **options(good))
    assert len(conn.calls) == 2
    feedback = conn.calls[1][-1]["content"]
    assert "execution.visible_state" in feedback and "scene" in feedback
    assert "deep-night" not in result["execution"]["visible_state"]
    assert result["source_script"] == good["source_script"]

def test_generator_fails_after_repeated_visual_pollution(sb, mods):
    bad = apple()
    bad["execution"]["visible_state"] += " Both actors stand against a deep-night urban backdrop."
    conn = ScriptedConnector([json.dumps(bad)] * 3)
    with pytest.raises(RuntimeError, match="execution.visible_state"):
        sb.AssetStoryboardEnhancer(conn)(bad["source_script"], 2, **options(bad))

def test_both_prompts_require_counterfactual_review():
    root = Path(__file__).resolve().parents[1]
    for rel in ["docs/script_asset_system_prompt.txt", "nodes/llm/prompts/h3_storyboard/system_asset.txt"]:
        text = (root / rel).read_text("utf-8")
        assert "COUNTERFACTUAL REPLACEMENT REVIEW" in text
        assert "deep-night urban backdrop" in text
