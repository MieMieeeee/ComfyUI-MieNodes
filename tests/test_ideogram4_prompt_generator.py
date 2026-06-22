"""Unit tests for Ideogram 4 prompt generator (no live LLM calls)."""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME


@pytest.fixture(scope="module")
def ideogram4_modules():
    load_plugin_module()
    prompts = importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ideogram4_prompts")
    gen = importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ideogram4_prompt_generator")
    return prompts, gen


def test_build_official_v1_messages_simple(ideogram4_modules):
    prompts, _ = ideogram4_modules
    messages = prompts.build_official_v1_messages("a red apple", "16:9", composition_mode="simple")
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "16:9" in user
    assert "a red apple" in user
    assert "OUTPUT CONTRACT" in system
    assert "aspect_ratio" in system
    assert "COMFYUI PIPELINE" in user
    assert "Resolution Selector" in user
    assert "COMPOSITION MODE (simple)" in system
    assert "Omit bbox" in system
    assert "Single source of truth for characters" in system
    assert "No readable copy in HLD" in system


def test_build_official_v1_messages_complex(ideogram4_modules):
    prompts, _ = ideogram4_modules
    messages = prompts.build_official_v1_messages("jazz poster", "2:3", composition_mode="complex")
    system = messages[0]["content"]
    assert "COMPOSITION MODE (complex)" in system
    assert "Every element must have a bbox" in system


def test_build_official_v1_messages_movable(ideogram4_modules):
    prompts, _ = ideogram4_modules
    simple = prompts.build_official_v1_messages("a red apple", "16:9", composition_mode="simple")
    movable = prompts.build_official_v1_messages("knit lamb", "1:1", composition_mode="movable")
    system = movable[0]["content"]
    user = movable[1]["content"]
    # the movable directive is fully in the system message now
    assert "MOVABLE" in system
    assert "BBOX-ONLY" in system
    assert "OVERRIDES" in system
    assert "IGNORE the earlier" in system
    assert "SOLE position authority" in system
    # user content carries no mode directive
    assert "COMPOSITION MODE" not in user
    assert "SOLE position authority" not in user
    # the movable system (long override) is strictly longer than simple's
    assert len(system) > len(simple[0]["content"])
    # simple mode is unchanged — no override leaked into it
    assert "MOVABLE" not in simple[0]["content"]


def test_build_ideogram4_messages_mode_dispatch(ideogram4_modules):
    prompts, _ = ideogram4_modules
    for mode in prompts.COMPOSITION_MODES:
        msgs = prompts.build_ideogram4_messages("test", "1:1", composition_mode=mode)
        assert len(msgs) == 2
        assert f"COMPOSITION MODE ({mode})" in msgs[0]["content"]
    with pytest.raises(ValueError, match="Unknown composition_mode"):
        prompts.build_ideogram4_messages("test", "1:1", composition_mode="nope")


def test_postprocess_strips_aspect_ratio_keeps_bbox(ideogram4_modules):
    _, gen = ideogram4_modules
    raw = """```json
{"aspect_ratio":"1:1","high_level_description":"test","compositional_deconstruction":{"background":"bg","elements":[{"type":"obj","bbox":[1,2,3,4],"desc":"x"}]}}
```"""
    out = gen.postprocess_caption(raw)
    data = json.loads(out)
    assert "aspect_ratio" not in data
    assert data["compositional_deconstruction"]["elements"][0]["bbox"] == [1, 2, 3, 4]
    assert "\n" not in out


def test_postprocess_repair_trailing_comma(ideogram4_modules):
    _, gen = ideogram4_modules
    raw = '{"high_level_description":"x","compositional_deconstruction":{"background":"b","elements":[{"type":"obj","desc":"x"},],},}'
    out = gen.postprocess_caption(raw)
    data = json.loads(out)
    assert data["high_level_description"] == "x"


def test_postprocess_raises_on_invalid_json(ideogram4_modules):
    _, gen = ideogram4_modules
    with pytest.raises(ValueError, match="cannot parse JSON"):
        gen.postprocess_caption("not json at all")


def test_postprocess_raises_on_schema_error(ideogram4_modules):
    _, gen = ideogram4_modules
    with pytest.raises(ValueError, match="compositional_deconstruction"):
        gen.postprocess_caption('{"high_level_description":"only this"}')


def test_node_registered_in_plugin():
    plugin = load_plugin_module()
    assert "Ideogram4PromptGenerator|Mie" in plugin.NODE_CLASS_MAPPINGS


def test_input_types(ideogram4_modules):
    _, gen = ideogram4_modules
    inputs = gen.Ideogram4PromptGenerator.INPUT_TYPES()
    assert "user_prompt" in inputs["required"]
    assert "llm_service_connector" in inputs["required"]
    assert "composition_mode" in inputs["required"]
    assert "prompt_profile" not in inputs["required"]
    assert "aspect_ratio" in inputs["optional"]
