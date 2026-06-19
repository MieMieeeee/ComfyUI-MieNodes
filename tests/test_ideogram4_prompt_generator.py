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


def test_build_official_v1_messages(ideogram4_modules):
    prompts, _ = ideogram4_modules
    messages = prompts.build_official_v1_messages("a red apple", "16:9")
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


def test_build_full_palette_messages(ideogram4_modules):
    prompts, _ = ideogram4_modules
    messages = prompts.build_full_palette_messages("a red apple", "16:9")
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "FULL SCHEMA EXTENSION" in system
    assert "style_description" in system
    assert "color_palette" in system
    assert "OUTPUT CONTRACT" in system  # v1 base retained
    assert "16:9" in user
    assert "do not emit aspect_ratio" in user.lower()


def test_build_compact_messages(ideogram4_modules):
    prompts, _ = ideogram4_modules
    messages = prompts.build_compact_messages("a red apple", "1:1")
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "compositional_deconstruction" in system
    assert "aspect_ratio" not in system or "no top-level" in system
    assert "1:1" in user


def test_build_ideogram4_messages_profile_dispatch(ideogram4_modules):
    prompts, _ = ideogram4_modules
    for profile in prompts.PROMPT_PROFILES:
        msgs = prompts.build_ideogram4_messages("test", "1:1", prompt_profile=profile)
        assert len(msgs) == 2
    with pytest.raises(ValueError, match="Unknown prompt_profile"):
        prompts.build_ideogram4_messages("test", "1:1", prompt_profile="nope")


def test_build_full_schema_messages_alias(ideogram4_modules):
    prompts, _ = ideogram4_modules
    legacy = prompts.build_full_schema_messages("x", "4:5")
    modern = prompts.build_full_palette_messages("x", "4:5")
    assert legacy[0]["content"] == modern[0]["content"]
    assert legacy[1]["content"] == modern[1]["content"]


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


def test_node_registered_in_plugin():
    plugin = load_plugin_module()
    assert "Ideogram4PromptGenerator|Mie" in plugin.NODE_CLASS_MAPPINGS


def test_input_types(ideogram4_modules):
    _, gen = ideogram4_modules
    inputs = gen.Ideogram4PromptGenerator.INPUT_TYPES()
    assert "user_prompt" in inputs["required"]
    assert "llm_service_connector" in inputs["required"]
    assert "prompt_profile" in inputs["required"]
    assert "aspect_ratio" in inputs["optional"]
    assert "strip_aspect_ratio" not in inputs.get("optional", {})
    assert "strip_bboxes" not in inputs.get("optional", {})
