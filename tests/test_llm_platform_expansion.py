# -*- coding: utf-8 -*-
"""Tests for the 2026-09-29 platform expansion connectors.

Pins the seven connectors added after the audit (see
``_plan_llm_platform_expansion.md`` for the doc research trail):
Doubao (Volcano Ark), Qianfan ERNIE, iFLYTEK Spark, OpenAI, xAI Grok,
OpenRouter, and Anthropic Claude (official OpenAI-compat layer).

Every URL / model id below was taken from official docs (or the live
OpenRouter catalog) — changing one is a billing/protocol contract
change and must be deliberate.

The fixture / module-loading pattern mirrors ``test_llm_bailian_plans.py``.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
LLM_PATH = PROJECT_DIR / "services" / "llm.py"
UTILS_PATH = PROJECT_DIR / "core" / "utils.py"


def _load_llm_module():
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

    for name in ("services", "services.llm"):
        if name in sys.modules:
            del sys.modules[name]
    if "_mienodes_internal.services" not in sys.modules:
        svcs = types.ModuleType("_mienodes_internal.services")
        svcs.__path__ = [str(PROJECT_DIR / "services")]
        svcs.__package__ = "_mienodes_internal.services"
        sys.modules["_mienodes_internal.services"] = svcs
    if "_mienodes_internal.services.__init__" not in sys.modules:
        init = types.ModuleType("_mienodes_internal.services.__init__")
        sys.modules["_mienodes_internal.services.__init__"] = init

    spec = importlib.util.spec_from_file_location(
        "_mienodes_internal.services.llm", str(LLM_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mienodes_internal.services.llm"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def llm_module():
    return _load_llm_module()


CONNECTOR_URLS = {
    "DoubaoConnectorGeneral": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "QianfanConnectorGeneral": "https://qianfan.baidubce.com/v2/chat/completions",
    "SparkConnectorGeneral": "https://spark-api-open.xf-yun.com/x2/chat/completions",
    "OpenAIConnectorGeneral": "https://api.openai.com/v1/chat/completions",
    "GrokConnectorGeneral": "https://api.x.ai/v1/chat/completions",
    "OpenRouterConnectorGeneral": "https://openrouter.ai/api/v1/chat/completions",
    "ClaudeConnectorGeneral": "https://api.anthropic.com/v1/chat/completions",
}

SET_NODES = {
    "SetDoubaoLLMServiceConnector": ("doubao", "doubao-seed-2-0-pro-260215"),
    "SetQianfanLLMServiceConnector": ("qianfan", "ernie-5.1"),
    "SetSparkLLMServiceConnector": ("spark", "spark-x"),
    "SetOpenAILLMServiceConnector": ("openai", "gpt-6-astra"),
    "SetGrokLLMServiceConnector": ("grok", "grok-4.7"),
    "SetOpenRouterLLMServiceConnector": ("openrouter", "openrouter/auto"),
    "SetClaudeLLMServiceConnector": ("anthropic", "claude-sonnet-5-5"),
}


def test_all_connector_classes_exist(llm_module):
    for cls in CONNECTOR_URLS:
        assert hasattr(llm_module, cls), cls


def test_all_set_nodes_exist(llm_module):
    for node in SET_NODES:
        assert hasattr(llm_module, node), node


@pytest.mark.parametrize("cls", sorted(CONNECTOR_URLS))
def test_urls_pinned(llm_module, cls):
    assert llm_module.__dict__[cls].api_url == CONNECTOR_URLS[cls]


@pytest.mark.parametrize("node", sorted(SET_NODES))
def test_set_node_config_key_and_default_model(llm_module, node):
    config_key, default_model = SET_NODES[node]
    spec = llm_module.__dict__[node].INPUT_TYPES()
    assert spec["optional"]["config_key"][1]["default"] == config_key
    models, meta = spec["required"]["model_select"]
    assert meta["default"] == default_model
    assert "Custom" in models


@pytest.mark.parametrize("node", sorted(SET_NODES))
def test_set_node_execute_builds_right_connector(llm_module, node):
    from unittest.mock import patch
    config_key, default_model = SET_NODES[node]
    connector_cls = {
        "SetDoubaoLLMServiceConnector": "DoubaoConnectorGeneral",
        "SetQianfanLLMServiceConnector": "QianfanConnectorGeneral",
        "SetSparkLLMServiceConnector": "SparkConnectorGeneral",
        "SetOpenAILLMServiceConnector": "OpenAIConnectorGeneral",
        "SetGrokLLMServiceConnector": "GrokConnectorGeneral",
        "SetOpenRouterLLMServiceConnector": "OpenRouterConnectorGeneral",
        "SetClaudeLLMServiceConnector": "ClaudeConnectorGeneral",
    }[node]
    with patch("services.llm.resolve_token", return_value="x"), \
         patch("services.llm.mie_log"):
        connector = llm_module.__dict__[node]().execute(
            api_token="x", model_select=default_model, custom_model=""
        )[0]
    assert isinstance(connector, llm_module.__dict__[connector_cls])
    assert connector.model == default_model


# ---------------------------------------------------------------------------
# Payload shapes (slim vs standard)
# ---------------------------------------------------------------------------

def test_openai_payload_uses_max_completion_tokens(llm_module):
    # gpt-5+/o-series reject the legacy max_tokens field server-side.
    c = llm_module.OpenAIConnectorGeneral("sk-x", "gpt-6-astra")
    payload = c.generate_payload([{"role": "user", "content": "hi"}], max_tokens=256)
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 256
    assert payload["model"] == "gpt-6-astra"
    assert payload["stream"] is False


def test_claude_payload_is_slim(llm_module):
    # The official OpenAI-compat layer silently ignores response_format /
    # penalties and requires n=1 — don't send the noise.
    c = llm_module.ClaudeConnectorGeneral("sk-ant-x", "claude-sonnet-5-5")
    payload = c.generate_payload(
        [{"role": "user", "content": "hi"}], max_tokens=128, temperature=0.3
    )
    assert payload == {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "max_tokens": 128,
    }


def test_spark_payload_is_slim(llm_module):
    c = llm_module.SparkConnectorGeneral("pw", "spark-x")
    payload = c.generate_payload([{"role": "user", "content": "hi"}], max_tokens=64)
    assert set(payload) == {"model", "messages", "stream", "max_tokens"}
    assert payload["model"] == "spark-x"


@pytest.mark.parametrize("cls, model", [
    ("DoubaoConnectorGeneral", "doubao-seed-2-0-pro-260215"),
    ("QianfanConnectorGeneral", "ernie-5.1"),
    ("GrokConnectorGeneral", "grok-4.7"),
    ("OpenRouterConnectorGeneral", "openrouter/auto"),
])
def test_standard_payload_platforms_keep_full_params(llm_module, cls, model):
    # These four accept the standard OpenAI parameter set per their docs.
    c = llm_module.__dict__[cls]("k", model)
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    assert payload["max_tokens"] == 512
    assert "temperature" in payload
    assert "response_format" in payload


# ---------------------------------------------------------------------------
# Dropdown sanity vs doc research
# ---------------------------------------------------------------------------

def test_doubao_dropdown_has_seed2_and_legacy(llm_module):
    models, _ = llm_module.SetDoubaoLLMServiceConnector.INPUT_TYPES()[
        "required"]["model_select"]
    assert "doubao-seed-2-0-pro-260215" in models
    assert "doubao-seed-1-6" in models


def test_claude_dropdown_uses_dashed_ids(llm_module):
    models, _ = llm_module.SetClaudeLLMServiceConnector.INPUT_TYPES()[
        "required"]["model_select"]
    assert "claude-sonnet-5-5" in models
    assert "claude-opus-5-5" in models
    # Anthropic ids use dashes, not dots.
    assert not any(m.startswith("claude-") and "." in m for m in models)


def test_openai_dropdown_gpt6_families(llm_module):
    models, _ = llm_module.SetOpenAILLMServiceConnector.INPUT_TYPES()[
        "required"]["model_select"]
    assert "gpt-6-astra" in models
    assert "gpt-6-luna-pro" in models


# ---------------------------------------------------------------------------
# Registration + keys example
# ---------------------------------------------------------------------------

def test_keys_example_includes_new_platforms():
    data = json.loads(
        (PROJECT_DIR / "mie_llm_keys.json.example").read_text(encoding="utf-8")
    )
    for key in ("doubao", "qianfan", "spark", "openai", "grok",
                "openrouter", "anthropic"):
        assert data.get(key) == "", key
