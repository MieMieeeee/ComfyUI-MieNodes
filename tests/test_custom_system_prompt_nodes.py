"""Tests for the CustomSystemPrompt three-node set (Generator + Add + Remove).

Mirrors the existing Kontext three-node pattern but for generic user-defined
system prompts. User data is isolated to user_system_prompts.json (gitignored),
kept separate from user_kontext_presets.json.
"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME  # noqa: E402


class _FakeConnector:
    def __init__(self):
        self.captured = None
        self.captured_kwargs = None

    def invoke(self, messages, **kwargs):
        self.captured = messages
        self.captured_kwargs = kwargs
        return "ok"

    def get_state(self):
        return "fake"


@pytest.fixture(scope="module")
def csm():
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.custom_system_prompt")


def test_three_nodes_registered():
    plugin = load_plugin_module()
    for name in (
        "CustomSystemPromptGenerator",
        "AddCustomSystemPrompt",
        "RemoveCustomSystemPrompt",
    ):
        assert f"{name}|Mie" in plugin.NODE_CLASS_MAPPINGS, f"{name} not registered"


def test_generator_input_types(csm):
    req = csm.CustomSystemPromptGenerator.INPUT_TYPES()["required"]
    assert "llm_service_connector" in req
    assert "input_text" in req
    assert "system_prompt_name" in req
    assert "seed" in req
    assert isinstance(req["system_prompt_name"], tuple) and len(req["system_prompt_name"][0]) > 0


def test_dropdown_lists_usable_builtins_excludes_placeholders(csm):
    choices = csm.CustomSystemPromptGenerator.INPUT_TYPES()["required"]["system_prompt_name"][0]
    # pure-text builtins are offered
    assert "hunyuan/t2v" in choices
    assert "flux2/t2i" in choices
    # placeholder-bearing prompts are NOT (they need caller kwargs)
    assert "hunyuan/i2v" not in choices  # contains {}
    assert "zimage/t2i" not in choices  # contains {prompt}
    assert "bernini/r2v" not in choices  # contains {image_num}


def test_generator_builds_messages_with_builtin(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn,
        input_text="a red apple",
        system_prompt_name="hunyuan/t2v",
        seed=1,
    )
    assert conn.captured[0]["role"] == "system"
    assert conn.captured[0]["content"]  # non-empty system prompt
    assert "a red apple" in conn.captured[1]["content"]


def test_generator_unknown_prompt_raises(csm):
    conn = _FakeConnector()
    with pytest.raises(ValueError, match="Unknown system prompt"):
        csm.CustomSystemPromptGenerator().generate(
            llm_service_connector=conn,
            input_text="x",
            system_prompt_name="does/not/exist",
            seed=1,
        )


def test_add_remove_roundtrip_updates_dropdown(csm, monkeypatch, tmp_path):
    fake_file = tmp_path / "user_system_prompts.json"
    monkeypatch.setattr(csm, "USER_CUSTOM_FILE", str(fake_file))

    ok, _ = csm.AddCustomSystemPrompt().add_prompt(prompt_name="MyTestPrompt", system_prompt="be brief")
    assert ok
    choices = csm.CustomSystemPromptGenerator.INPUT_TYPES()["required"]["system_prompt_name"][0]
    assert "MyTestPrompt" in choices

    # the added prompt is actually usable end-to-end
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="MyTestPrompt", seed=1
    )
    assert conn.captured[0]["content"] == "be brief"

    ok2, _ = csm.RemoveCustomSystemPrompt().remove_preset(preset_name="MyTestPrompt")
    assert ok2
    choices2 = csm.CustomSystemPromptGenerator.INPUT_TYPES()["required"]["system_prompt_name"][0]
    assert "MyTestPrompt" not in choices2


def test_add_rejects_empty_and_duplicate(csm, monkeypatch, tmp_path):
    fake_file = tmp_path / "user_system_prompts.json"
    monkeypatch.setattr(csm, "USER_CUSTOM_FILE", str(fake_file))

    ok, _ = csm.AddCustomSystemPrompt().add_prompt(prompt_name="", system_prompt="x")
    assert not ok
    ok, _ = csm.AddCustomSystemPrompt().add_prompt(prompt_name="x", system_prompt="")
    assert not ok

    ok, _ = csm.AddCustomSystemPrompt().add_prompt(prompt_name="Dup", system_prompt="one")
    assert ok
    ok2, msg = csm.AddCustomSystemPrompt().add_prompt(prompt_name="Dup", system_prompt="two")
    assert not ok2
    assert "already exists" in msg


def _img(n=1):
    """Tiny ComfyUI IMAGE batch (n,8,8,3) for multimodal tests."""
    return torch.rand(n, 8, 8, 3)


def test_generator_has_optional_media_inputs(csm):
    opt = csm.CustomSystemPromptGenerator.INPUT_TYPES().get("optional", {})
    for key in ("source", "reference_images", "reference_video", "video_frames", "reference_video_frames", "image_detail"):
        assert key in opt, f"missing optional media input: {key}"


def test_generator_pure_text_without_media(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="hello", system_prompt_name="hunyuan/t2v", seed=1
    )
    # no media -> user content stays a plain string (unchanged behavior)
    assert isinstance(conn.captured[1]["content"], str)
    assert "hello" in conn.captured[1]["content"]


def test_generator_includes_image_parts_with_source(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn,
        input_text="describe this",
        system_prompt_name="hunyuan/t2v",
        seed=1,
        source=_img(1),
    )
    content = conn.captured[1]["content"]
    assert isinstance(content, list)
    assert any(p.get("type") == "image_url" for p in content)
    assert any(p.get("type") == "text" and "describe this" in p.get("text", "") for p in content)


def test_generator_reference_images_forwarded(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="hunyuan/t2v", seed=1,
        reference_images=_img(3),
    )
    img_parts = [p for p in conn.captured[1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 3


def test_generator_reference_video_sampled(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="hunyuan/t2v", seed=1,
        reference_video=_img(10), reference_video_frames=3,
    )
    img_parts = [p for p in conn.captured[1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 3  # sampled down to reference_video_frames


def test_generator_reference_video_zero_forwards_all(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="hunyuan/t2v", seed=1,
        reference_video=_img(10), reference_video_frames=0,
    )
    img_parts = [p for p in conn.captured[1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 10  # 0 = forward all (legacy behavior)


def test_generator_reference_video_default_samples_three(csm):
    conn = _FakeConnector()
    # default reference_video_frames is 3 (not 0/all) so big batches stay small
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="hunyuan/t2v", seed=1,
        reference_video=_img(10),
    )
    img_parts = [p for p in conn.captured[1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 3


def test_generator_source_uses_video_frames(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="hunyuan/t2v", seed=1,
        source=_img(8), video_frames=2,
    )
    img_parts = [p for p in conn.captured[1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 2  # source sampled by video_frames, not reference_video_frames


def test_generator_passes_max_tokens(csm):
    conn = _FakeConnector()
    csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=conn, input_text="x", system_prompt_name="hunyuan/t2v", seed=1,
        max_tokens=2048,
    )
    assert conn.captured_kwargs.get("max_tokens") == 2048


def test_generator_returns_empty_when_invoke_none(csm):
    class _NoneConn(_FakeConnector):
        def invoke(self, messages, **kwargs):
            return None

    out = csm.CustomSystemPromptGenerator().generate(
        llm_service_connector=_NoneConn(), input_text="x", system_prompt_name="hunyuan/t2v", seed=1
    )
    assert out == ("",)  # graceful empty instead of crashing on None
