# -*- coding: utf-8 -*-
"""Tests for Bernini's per-call reference_video_frames sampling.

Mirrors the structure of ``test_bernini_timeout_override.py`` so the
fixtures and module loading stay in sync. Covers:

* ``reference_video_frames`` actually samples the reference video
  (the routing previously forwarded the entire batch verbatim).
* ``reference_video_frames=0`` keeps the legacy "all frames" behavior.
* The ComfyUI node exposes a ``reference_video_frames`` widget.
* ``is_changed`` factors in ``reference_video_frames`` so a tweak
  actually re-runs the node.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
BERNINI_PATH = PROJECT_DIR / "nodes" / "llm" / "bernini_prompt_generator.py"
UTILS_PATH = PROJECT_DIR / "core" / "utils.py"
PROMPTS_PATH = PROJECT_DIR / "nodes" / "llm" / "bernini_prompts.py"


def _load_bernini():
    if "_mienodes_internal" not in sys.modules:
        ip = types.ModuleType("_mienodes_internal")
        ip.__path__ = [str(PROJECT_DIR)]
        ip.__package__ = "_mienodes_internal"
        sys.modules["_mienodes_internal"] = ip
    if "_mienodes_internal.core" not in sys.modules:
        core = types.ModuleType("_mienodes_internal.core")
        core.__path__ = [str(PROJECT_DIR / "core")]
        core.__package__ = "_mienodes_internal.core"
        sys.modules["_mienodes_internal.core"] = core
    if "_mienodes_internal.core.utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "_mienodes_internal.core.utils", str(UTILS_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_mienodes_internal.core.utils"] = mod
        spec.loader.exec_module(mod)
    if "_mienodes_internal.nodes" not in sys.modules:
        n = types.ModuleType("_mienodes_internal.nodes")
        n.__path__ = [str(PROJECT_DIR / "nodes")]
        n.__package__ = "_mienodes_internal.nodes"
        sys.modules["_mienodes_internal.nodes"] = n
    if "_mienodes_internal.nodes.llm" not in sys.modules:
        nllm = types.ModuleType("_mienodes_internal.nodes.llm")
        nllm.__path__ = [str(PROJECT_DIR / "nodes" / "llm")]
        nllm.__package__ = "_mienodes_internal.nodes.llm"
        sys.modules["_mienodes_internal.nodes.llm"] = nllm
    if "_mienodes_internal.nodes.llm.bernini_prompts" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "_mienodes_internal.nodes.llm.bernini_prompts", str(PROMPTS_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_mienodes_internal.nodes.llm.bernini_prompts"] = mod
        spec.loader.exec_module(mod)
    if "_mienodes_internal.nodes.llm.bernini_prompt_generator" in sys.modules:
        del sys.modules["_mienodes_internal.nodes.llm.bernini_prompt_generator"]
    spec = importlib.util.spec_from_file_location(
        "_mienodes_internal.nodes.llm.bernini_prompt_generator", str(BERNINI_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mienodes_internal.nodes.llm.bernini_prompt_generator"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bernini():
    return _load_bernini()


def _make_connector():
    c = MagicMock()
    c.model = "M3"
    c.timeout = 30
    c.invoke.return_value = "rewritten prompt"
    c.get_state.return_value = "state-fingerprint"
    return c


def _make_video_batch(n=5):
    # (N, H, W, C) ComfyUI IMAGE layout. Tiny 4x4 keeps data URLs short.
    return torch.zeros(n, 4, 4, 3, dtype=torch.float32)


def _capture_chat(bernini):
    """Patch ``BerniniPromptEnhancer._chat`` and return the captured url batches."""
    captured = []

    def fake_chat(self, system, user, urls, **kwargs):
        captured.append(list(urls))
        return "rewritten"

    patcher = patch.object(bernini.BerniniPromptEnhancer, "_chat", fake_chat)
    return captured, patcher


def test_reference_video_frames_samples_uniformly(bernini):
    """5-frame source + 5-frame reference_video, sample 3/2 -> 5 urls total."""
    captured, patcher = _capture_chat(bernini)
    with patch.object(bernini, "mie_log"):
        with patcher:
            enhancer = bernini.BerniniPromptEnhancer(
                _make_connector(),
                video_frames=3,
                reference_video_frames=2,
            )
            # vrc2v routes through frame_urls (sampled from source) +
            # ref_total = ref_img_urls + ref_vid_urls.
            enhancer(
                "vrc2v - 参考内容视频编辑",
                "edit me",
                source=_make_video_batch(5),
                reference_video=_make_video_batch(5),
            )
    assert len(captured) == 1
    urls = captured[0]
    # source contributes 3 sampled frames, reference_video contributes 2.
    assert len(urls) == 5, f"expected 3 source + 2 ref_vid, got {len(urls)}"


def test_reference_video_frames_zero_keeps_all_frames(bernini):
    """``reference_video_frames=0`` must not drop any reference_video frames (legacy)."""
    captured, patcher = _capture_chat(bernini)
    with patch.object(bernini, "mie_log"):
        with patcher:
            enhancer = bernini.BerniniPromptEnhancer(
                _make_connector(),
                video_frames=3,
                reference_video_frames=0,
            )
            enhancer(
                "vrc2v - 参考内容视频编辑",
                "edit me",
                source=_make_video_batch(5),
                reference_video=_make_video_batch(5),
            )
    assert len(captured) == 1
    urls = captured[0]
    # 3 sampled source + all 5 reference_video frames.
    assert len(urls) == 8, f"expected 3 source + 5 ref_vid (no sampling), got {len(urls)}"


def test_reference_video_frames_sampled_does_not_drop_source(bernini):
    """Sampling reference_video must not affect source sampling (independent)."""
    captured, patcher = _capture_chat(bernini)
    with patch.object(bernini, "mie_log"):
        with patcher:
            enhancer = bernini.BerniniPromptEnhancer(
                _make_connector(),
                video_frames=1,
                reference_video_frames=2,
            )
            enhancer(
                "ads2v - 视频植入视频",
                "place ad",
                source=_make_video_batch(10),
                reference_video=_make_video_batch(10),
            )
    urls = captured[0]
    # ads2v: frame_urls (sampled) + ref_vid_urls (sampled)
    # source 10 -> 1 middle frame, reference_video 10 -> 2 endpoints.
    assert len(urls) == 3, f"expected 1 source + 2 ref_vid, got {len(urls)}"


def test_reference_video_only_path(bernini):
    """``vrc2v`` with no source and only reference_video still works."""
    captured, patcher = _capture_chat(bernini)
    with patch.object(bernini, "mie_log"):
        with patcher:
            enhancer = bernini.BerniniPromptEnhancer(
                _make_connector(),
                video_frames=3,
                reference_video_frames=2,
            )
            enhancer(
                "vrc2v - 参考内容视频编辑",
                "use only ref video",
                reference_video=_make_video_batch(5),
            )
    # No source -> frame_urls empty; ref_video contributes 2 sampled frames.
    urls = captured[0]
    assert len(urls) == 2, f"expected 2 ref_vid frames, got {len(urls)}"


def test_input_types_has_reference_video_frames(bernini):
    inputs = bernini.BerniniPromptGenerator.INPUT_TYPES()
    opt = inputs["optional"]
    assert "reference_video_frames" in opt, (
        "Bernini node INPUT_TYPES must expose 'reference_video_frames'"
    )
    spec = opt["reference_video_frames"]
    # spec shape: ("INT", {default, min, max, ...})
    assert isinstance(spec, tuple) and spec[0] == "INT"
    meta = spec[1]
    assert "default" in meta
    assert "min" in meta
    assert "max" in meta
    # 0 must be allowed (= legacy "all frames" behavior).
    assert meta["min"] <= 0 <= meta["max"]


def test_is_changed_includes_reference_video_frames(bernini):
    node = bernini.BerniniPromptGenerator()
    connector = _make_connector()
    h2 = node.is_changed(
        connector, "vrc2v - 参考内容视频编辑", "hi", 0, reference_video_frames=2
    )
    h4 = node.is_changed(
        connector, "vrc2v - 参考内容视频编辑", "hi", 0, reference_video_frames=4
    )
    h0 = node.is_changed(
        connector, "vrc2v - 参考内容视频编辑", "hi", 0, reference_video_frames=0
    )
    assert h2 != h4, "is_changed should differ when reference_video_frames changes"
    assert h2 != h0, "is_changed should differ when toggling 0 vs n"
