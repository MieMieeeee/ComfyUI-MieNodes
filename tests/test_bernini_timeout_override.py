# -*- coding: utf-8 -*-
"""Tests for Bernini's per-call timeout override."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


def _make_connector(initial_timeout=30):
    c = MagicMock()
    c.model = "M3"
    c.timeout = initial_timeout
    c.invoke.return_value = "rewritten prompt"
    c.get_state.return_value = "state-fingerprint"
    return c


def test_timeout_override_sets_and_restores(bernini):
    connector = _make_connector(initial_timeout=30)
    with patch.object(bernini, "mie_log"):
        enhancer = bernini.BerniniPromptEnhancer(connector, timeout=60)
        enhancer("t2i - 文生图", "hello world")
    # During the call, the timeout should have been set to 60
    # After the call, it should be restored
    assert connector.timeout == 30, f"timeout not restored: {connector.timeout}"
    # And the invoke should have been called with the connector whose timeout was 60
    # We can check the side-effect happened on the same connector object
    assert connector.invoke.call_count == 1


def test_timeout_override_no_op_when_matches(bernini):
    """If override equals connector's current value, no swap is needed."""
    connector = _make_connector(initial_timeout=30)
    with patch.object(bernini, "mie_log"):
        enhancer = bernini.BerniniPromptEnhancer(connector, timeout=30)
        enhancer("t2i - 文生图", "hello world")
    assert connector.timeout == 30
    assert connector.invoke.call_count == 1


def test_timeout_none_keeps_connector_default(bernini):
    """``timeout=None`` (the default) must not change the connector."""
    connector = _make_connector(initial_timeout=15)  # someone set a custom default
    with patch.object(bernini, "mie_log"):
        enhancer = bernini.BerniniPromptEnhancer(connector)  # no timeout kwarg
        enhancer("t2i - 文生图", "hello world")
    assert connector.timeout == 15
    assert connector.invoke.call_count == 1


def test_input_types_has_timeout_dropdown(bernini):
    inputs = bernini.BerniniPromptGenerator.INPUT_TYPES()
    opt = inputs["optional"]
    assert "timeout" in opt, "Bernini node INPUT_TYPES must expose 'timeout'"
    spec = opt["timeout"]
    # spec is ([30, 60, 120, 300], {"default": 30})
    choices, meta = spec
    assert 30 in choices and 60 in choices and 120 in choices
    assert meta["default"] == 30


def test_is_changed_includes_timeout(bernini):
    node = bernini.BerniniPromptGenerator()
    connector = _make_connector(initial_timeout=30)
    h30 = node.is_changed(connector, "t2i - 文生图", "hi", 0, timeout=30)
    h60 = node.is_changed(connector, "t2i - 文生图", "hi", 0, timeout=60)
    assert h30 != h60, "is_changed should differ when timeout changes"
