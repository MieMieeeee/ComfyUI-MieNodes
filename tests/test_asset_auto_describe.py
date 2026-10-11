# -*- coding: utf-8 -*-
"""Tests for the LLM-assisted asset description drafter."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tests.test_asset_library import make_png, png_bytes_list_to_image_tensor  # noqa: E402


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


def test_character_bundle_full_draft():
    mod = _load_node_module()
    conn = _Conn([
        "A kind elderly Chinese grandfather, 68 years old, short silver-white "
        "hair with white temples, black thick-rimmed reading glasses.",
        "elderly male, low thick steady voice, warm cheerful undertone",
    ])
    bundle = {
        "kind": "character", "name": "爷爷",
        "description": "", "voice": "",
        "image": png_bytes_list_to_image_tensor([make_png((8, 8, 8))]),
        "audio": None,
    }
    node = mod.AssetAutoDescribe()
    name, description, voice, status = node.describe(conn, asset=bundle)
    assert name == "爷爷"
    assert "silver-white hair" in description
    assert voice.startswith("elderly male")
    # call 1 = multimodal caption (image part present), call 2 = text voice
    blob1 = json.dumps(conn.calls[0])
    assert "image_url" in blob1
    assert isinstance(conn.calls[1][1]["content"], str)  # pure text
    assert "爷爷" in conn.calls[1][1]["content"]
    assert "never listened" in status


def test_scene_bundle_no_voice_call():
    mod = _load_node_module()
    conn = _Conn(["A cozy family living room at night with a beige sofa."])
    bundle = {
        "kind": "scene", "name": "客厅",
        "description": "",
        "image": png_bytes_list_to_image_tensor([make_png((1, 2, 3))]),
    }
    name, description, voice, status = mod.AssetAutoDescribe().describe(conn, scene_asset=bundle)
    assert name == "客厅" and "sofa" in description
    assert voice == "" and len(conn.calls) == 1  # no voice call for scenes


def test_raw_image_and_errors():
    mod = _load_node_module()
    conn = _Conn([
        "A cute dazed 19-year-old Chinese girl with a yellow star hairpin.",
        "young adult female, soft mid-high airy voice",
    ])
    node = mod.AssetAutoDescribe()
    name, description, voice, _ = node.describe(
        conn, image=png_bytes_list_to_image_tensor([make_png()]), name="孙女"
    )
    assert name == "孙女" and "star hairpin" in description
    with pytest.raises(RuntimeError, match="needs a picture"):
        node.describe(_Conn([]))
    with pytest.raises(RuntimeError, match="invoke"):
        node.describe(object())


def test_author_hints_flow_into_user_turns():
    mod = _load_node_module()
    conn = _Conn([
        "A stern war-veteran captain with a burn scar on the left cheek.",
        "elderly male, gravelly and hoarse, slow deliberate pace",
    ])
    bundle = {
        "kind": "character", "name": "船长",
        "description": "", "voice": "",
        "image": png_bytes_list_to_image_tensor([make_png((4, 5, 6))]),
        "audio": None,
    }
    node = mod.AssetAutoDescribe()
    _, _, _, status = node.describe(
        conn, asset=bundle,
        image_prompt="左脸颊有一道烧伤疤，务必写进描述",
        voice_prompt="沙哑的老烟嗓，语速慢",
    )
    # hint 1 rides in the caption call's USER text (system keeps its contract)
    caption_blob = json.dumps(conn.calls[0], ensure_ascii=False)
    assert "Author notes" in caption_blob and "烧伤疤" in caption_blob
    # hint 2 rides in the voice call's user text as ground truth
    voice_user = conn.calls[1][1]["content"]
    assert "ground truth" in voice_user and "沙哑的老烟嗓" in voice_user
    assert "author hints applied (image+voice)" in status


def test_empty_hints_keep_messages_byte_identical():
    mod = _load_node_module()
    replies = ["desc.", "voice."]
    plain, hinted = _Conn(replies), _Conn(replies)
    kwargs = dict(
        image=png_bytes_list_to_image_tensor([make_png()]), name="x",
    )
    mod.AssetAutoDescribe().describe(plain, **kwargs)
    mod.AssetAutoDescribe().describe(hinted, image_prompt="", voice_prompt="   ", **kwargs)
    assert json.dumps(plain.calls) == json.dumps(hinted.calls)


def test_scene_hint_only_affects_caption_call():
    mod = _load_node_module()
    conn = _Conn(["A rainy neon alley at night."])
    bundle = {
        "kind": "scene", "name": "小巷",
        "image": png_bytes_list_to_image_tensor([make_png((7, 8, 9))]),
    }
    _, _, _, status = mod.AssetAutoDescribe().describe(
        conn, scene_asset=bundle, image_prompt="雨夜霓虹", voice_prompt="不该出现"
    )
    blob = json.dumps(conn.calls[0], ensure_ascii=False)
    assert "雨夜霓虹" in blob and len(conn.calls) == 1
    assert "voice" not in status.split("—")[0]  # no voice draft for scenes
    assert "author hints applied (image)" in status
