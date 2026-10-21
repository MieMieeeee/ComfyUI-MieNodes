# -*- coding: utf-8 -*-
"""Tests for the Save-based LLM auto-describe nodes."""
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tests.test_asset_library import (  # noqa: E402
    make_png,
    make_wav,
    png_bytes_list_to_image_tensor,
    wav_bytes_to_audio,
)


def _load_node_module():
    for fqn, sub in [
        ("_mienodes_internal", None),
        ("_mienodes_internal.core", PROJECT_DIR / "core"),
        ("_mienodes_internal.nodes", PROJECT_DIR / "nodes"),
        ("_mienodes_internal.nodes.assets", PROJECT_DIR / "nodes/assets"),
        ("_mienodes_internal.nodes.llm", PROJECT_DIR / "nodes/llm"),
        ("_mienodes_internal.nodes.llm.prompts", PROJECT_DIR / "nodes/llm/prompts"),
    ]:
        if fqn in sys.modules:
            continue
        m = types.ModuleType(fqn)
        if sub:
            m.__path__ = [str(sub)]
        m.__package__ = fqn
        sys.modules[fqn] = m
    for fqn, path in [
        ("_mienodes_internal.core.utils", PROJECT_DIR / "core/utils.py"),
        ("_mienodes_internal.nodes.llm.prompts.loader", PROJECT_DIR / "nodes/llm/prompts/loader.py"),
        ("_mienodes_internal.nodes.llm.h3_prompts", PROJECT_DIR / "nodes/llm/h3_prompts.py"),
        ("_mienodes_internal.nodes.assets.library", PROJECT_DIR / "nodes/assets/library.py"),
        ("_mienodes_internal.nodes.assets.nodes", PROJECT_DIR / "nodes/assets/nodes.py"),
    ]:
        if fqn in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(fqn, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location(
        "_mienodes_internal.nodes.llm.asset_auto_describe",
        str(PROJECT_DIR / "nodes/llm/asset_auto_describe.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Conn:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture()
def tmp_library(tmp_path, monkeypatch):
    monkeypatch.setenv("MIE_ASSET_LIBRARY", str(tmp_path))
    return tmp_path


def test_character_auto_drafts_and_saves(tmp_library):
    mod = _load_node_module()
    conn = _Conn([
        "A cute dazed 19-year-old Chinese girl with a yellow star hairpin.",
        "young adult female, soft mid-high airy voice",
    ])
    node = mod.SaveCharacterAssetAutoV2()
    name, description, voice, image, status, bundle = node.save_auto(
        conn, name="孙女",
        image=png_bytes_list_to_image_tensor([make_png()]),
    )
    assert name == "孙女"
    assert "star hairpin" in description and description == bundle["description"]
    assert voice.startswith("young adult female") and voice == bundle["voice"]
    assert "LLM drafted description and voice" in status
    assert "never listened" in status
    # two calls: multimodal caption then text voice
    assert "image_url" in json.dumps(conn.calls[0])
    assert isinstance(conn.calls[1][1]["content"], str)
    # really saved to the fixed library
    char_dir = tmp_library / "characters" / "孙女"
    assert (char_dir / "character.json").exists()
    assert (char_dir / "images" / "01.png").exists()


def test_filled_fields_save_verbatim_zero_llm_calls(tmp_library):
    mod = _load_node_module()
    conn = _Conn([])
    node = mod.SaveCharacterAssetAutoV2()
    _, description, voice, _, status, _ = node.save_auto(
        conn, name="爷爷", description="handwritten", voice="handwritten voice",
        image=png_bytes_list_to_image_tensor([make_png((1, 2, 3))]),
        image_prompt="不该被用到", voice_prompt="也不该被用到",
    )
    assert description == "handwritten" and voice == "handwritten voice"
    assert conn.calls == []  # pure save, no LLM round-trips
    assert "LLM drafted" not in status


def test_character_author_hints_reach_calls(tmp_library):
    mod = _load_node_module()
    conn = _Conn([
        "A stern war-veteran captain with a burn scar on the left cheek.",
        "elderly male, gravelly and hoarse, slow deliberate pace",
    ])
    node = mod.SaveCharacterAssetAutoV2()
    _, description, voice, _, status, _ = node.save_auto(
        conn, name="（✏️ 新名字…）", new_name="船长",
        image=png_bytes_list_to_image_tensor([make_png((4, 5, 6))]),
        image_prompt="左脸颊有一道烧伤疤，务必写进描述",
        voice_prompt="沙哑的老烟嗓，语速慢",
    )
    caption_blob = json.dumps(conn.calls[0], ensure_ascii=False)
    assert "Author notes" in caption_blob and "烧伤疤" in caption_blob
    voice_user = conn.calls[1][1]["content"]
    assert "ground truth" in voice_user and "沙哑的老烟嗓" in voice_user
    # new_name flows into the voice call (resolved save name)
    assert "船长" in voice_user
    assert (tmp_library / "characters" / "船长" / "character.json").exists()


def test_scene_auto_drafts_description_only(tmp_library):
    mod = _load_node_module()
    conn = _Conn(["A rainy neon alley at night."])
    node = mod.SaveSceneAssetAutoV2()
    name, description, image, status, bundle = node.save_auto(
        conn, name="小巷",
        image=png_bytes_list_to_image_tensor([make_png((7, 8, 9))]),
        image_prompt="雨夜霓虹",
    )
    assert name == "小巷" and "neon alley" in description
    assert len(conn.calls) == 1  # no voice call for scenes
    blob = json.dumps(conn.calls[0], ensure_ascii=False)
    assert "雨夜霓虹" in blob
    assert "LLM drafted description" in status
    assert (tmp_library / "scenes" / "小巷" / "scene.json").exists()


def test_auto_save_input_contract_matches_function_signatures():
    mod = _load_node_module()
    for cls in (mod.SaveAssetAutoV2, mod.SaveCharacterAssetAutoV2,
                mod.SaveSceneAssetAutoV2):
        declared = set(cls.INPUT_TYPES().get("required", {}))
        declared.update(cls.INPUT_TYPES().get("optional", {}))
        accepted = set(inspect.signature(getattr(cls, cls.FUNCTION)).parameters)
        assert declared <= accepted, "%s misses %s" % (cls.__name__, declared - accepted)


def test_scene_auto_preserves_declared_ambience(tmp_library):
    mod = _load_node_module()
    node = mod.SaveSceneAssetAutoV2()
    out = node.save_auto(
        _Conn([]), name="安静室内", description="A quiet indoor room.",
        ambience="quiet room tone with soft indoor air",
        image=png_bytes_list_to_image_tensor([make_png((7, 8, 9))]),
    )
    assert out[4]["ambience"] == "quiet room tone with soft indoor air"


def test_no_picture_and_empty_description_is_loud(tmp_library):
    mod = _load_node_module()
    node = mod.SaveCharacterAssetAutoV2()
    with pytest.raises(RuntimeError, match="needs a picture"):
        node.save_auto(_Conn([]), name="无图", description="   ")
    scene = mod.SaveSceneAssetAutoV2()
    with pytest.raises(RuntimeError, match="needs a picture"):
        scene.save_auto(_Conn([]), name="无图", description="")
    with pytest.raises(RuntimeError, match="invoke"):
        mod.SaveCharacterAssetAutoV2().save_auto(
            object(), name="x", description="filled",
            image=png_bytes_list_to_image_tensor([make_png()]),
        )


def test_upload_picker_feeds_draft_and_save(tmp_library, monkeypatch):
    mod = _load_node_module()
    conn = _Conn([
        "An orange tabby cat sleeping on a bookshelf.",
        "adult neutral, soft mid-register purring warmth",
    ])
    png_bytes = make_png()

    # the auto module holds its own import reference — patch it there
    monkeypatch.setattr(
        mod, "upload_image_to_tensor",
        lambda name: png_bytes_list_to_image_tensor([png_bytes]),
    )

    node = mod.SaveCharacterAssetAutoV2()
    name, description, _, _, status, _ = node.save_auto(
        conn, name="橘猫", image_upload="uploaded.png",
    )
    assert "bookshelf" in description
    assert (tmp_library / "characters" / "橘猫" / "images" / "01.png").exists()
    assert "LLM drafted description" in status


def test_media_priority_socket_beats_upload(tmp_library, monkeypatch):
    """Q1 regression: wired sockets must win over the upload pickers."""
    mod = _load_node_module()
    import _mienodes_internal.nodes.assets.nodes as assets_nodes

    called = []
    monkeypatch.setattr(
        assets_nodes, "upload_image_to_tensor",
        lambda name: called.append("image") or png_bytes_list_to_image_tensor([make_png()]),
    )
    monkeypatch.setattr(
        assets_nodes, "upload_audio_to_audio",
        lambda name: called.append("audio") or {"waveform": None, "sample_rate": 44100},
    )

    node = mod.SaveCharacterAssetAutoV2()
    node.save_auto(
        _Conn([]), name="优先级", description="filled", voice="filled",
        image=png_bytes_list_to_image_tensor([make_png((2, 2, 2))]),
        audio=wav_bytes_to_audio(make_wav(freq=220.0)),
        image_upload="picked.png", audio_upload="picked.wav",
    )
    assert called == []  # both sockets wired -> pickers never resolved
    char_dir = tmp_library / "characters" / "优先级"
    assert (char_dir / "character.json").exists()


def test_unified_auto_kind_branches(tmp_library):
    mod = _load_node_module()
    conn = _Conn([
        "A rainy neon alley at night.",
        "young adult female voice",  # must NOT be consumed in scene mode
    ])
    node = mod.SaveAssetAutoV2()
    name, description, voice, image, status, bundle = node.save_auto(
        conn, kind="scene", name="夜巷",
        image=png_bytes_list_to_image_tensor([make_png()]),
        voice_prompt="场景模式不该用到的音色提示",
    )
    assert name == "夜巷" and "neon alley" in description
    assert len(conn.calls) == 1  # scenes never call the voice drafter
    assert bundle["kind"] == "scene"
    assert (tmp_library / "scenes" / "夜巷" / "scene.json").exists()

    conn2 = _Conn([
        "A cute dazed 19-year-old Chinese girl with a yellow star hairpin.",
        "young adult female, soft mid-high airy voice",
    ])
    out = mod.SaveAssetAutoV2().save_auto(
        conn2, kind="character", name="女孩",
        image=png_bytes_list_to_image_tensor([make_png((9, 9, 9))]),
    )
    assert out[5]["kind"] == "character"
    assert "drafted description and voice" in out[4]
    assert (tmp_library / "characters" / "女孩" / "character.json").exists()


def test_clean_trims_at_sentence_boundary():
    """2026-10-14 live: 黑白/银白's saved descriptions were truncated
    mid-word at exactly 800 chars ('multi-pane windo'). The captioner's
    _clean now trims to the last sentence end inside the cap."""
    from nodes.llm.asset_auto_describe import _clean
    long = ("A young woman with twin tails and a maid dress. "
            "She sits centered on a pale classroom desk in a sunlit "
            "room with a bright multi-pane window. " * 12)
    out = _clean(long, 300)
    assert len(out) <= 301
    assert out.endswith(".") and not out.endswith("windo")
    # under the cap: untouched
    assert _clean("short text", 300) == "short text"
