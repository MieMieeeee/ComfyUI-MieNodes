# -*- coding: utf-8 -*-
"""Tests for the Alibaba Bailian Token Plan / Coding Plan connectors.

These guard against accidental regression of the most user-visible facts:

- Bailian has THREE independent endpoints with non-overlapping API keys.
  The PAYG / 按量计费 tier uses ``dashscope.aliyuncs.com`` with an ``sk-``
  key. The Token Plan / 套餐 tier uses ``token-plan.cn-beijing.maas.aliyuncs.com``
  with an ``sk-sp-`` key. The Coding Plan / 编程订阅 tier uses
  ``coding.dashscope.aliyuncs.com`` with an ``sk-cp-`` key. Cross-use always
  fails server-side.
- The Token Plan and Coding Plan connectors inherit
  ``StandardOpenAICompatibleConnector`` so they pick up ``max_tokens``,
  ``temperature``, ``top_p``, etc. The legacy PAYG ``BailianLLMServiceConnector``
  uses a slim payload (``{model, messages, stream}``) because some
  Qwen3-Max checkpoints on the PAYG tier historically 400'd on
  ``response_format``. Tests pin both shapes.

The fixture / module-loading pattern mirrors ``test_llm_mimo.py``.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Class existence + URL pinning
# ---------------------------------------------------------------------------

def test_token_plan_class_exists(llm_module):
    assert hasattr(llm_module, "BailianTokenPlanConnectorGeneral")


def test_coding_plan_class_exists(llm_module):
    assert hasattr(llm_module, "BailianCodingPlanConnectorGeneral")


def test_set_token_plan_node_class_exists(llm_module):
    assert hasattr(llm_module, "SetBailianTokenPlanLLMServiceConnector")


def test_set_coding_plan_node_class_exists(llm_module):
    assert hasattr(llm_module, "SetBailianCodingPlanLLMServiceConnector")


def test_token_plan_url_pinned(llm_module):
    # The MaaS workspace endpoint for cn-beijing. Changing this is a billing
    # contract change and must be deliberate.
    assert (
        llm_module.BailianTokenPlanConnectorGeneral.api_url
        == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_coding_plan_url_pinned(llm_module):
    assert (
        llm_module.BailianCodingPlanConnectorGeneral.api_url
        == "https://coding.dashscope.aliyuncs.com/v1/chat/completions"
    )


def test_three_bailian_connectors_have_distinct_urls(llm_module):
    # Cross-tier key reuse is a common user mistake. Catching accidental
    # URL aliasing here prevents silent regression.
    urls = {
        llm_module.BailianLLMServiceConnector.api_url,
        llm_module.BailianTokenPlanConnectorGeneral.api_url,
        llm_module.BailianCodingPlanConnectorGeneral.api_url,
    }
    assert len(urls) == 3, (
        f"Bailian tiers must use 3 distinct endpoints; got {urls}"
    )


def test_token_plan_url_is_not_dashscope_compatible_mode(llm_module):
    # Defense-in-depth: the PAYG endpoint must never leak into the Token
    # Plan connector. Bailian's docs state that the API keys are scoped to
    # a single endpoint and cross-use returns 401 / 403.
    assert "dashscope.aliyuncs.com" not in llm_module.BailianTokenPlanConnectorGeneral.api_url


# ---------------------------------------------------------------------------
# Set-node INPUT_TYPES: config_key + dropdown model coverage
# ---------------------------------------------------------------------------

def test_set_token_plan_uses_bailian_token_plan_config_key(llm_module):
    spec = llm_module.SetBailianTokenPlanLLMServiceConnector.INPUT_TYPES()
    assert spec["optional"]["config_key"][1]["default"] == "bailian_token_plan"


def test_set_coding_plan_uses_bailian_coding_config_key(llm_module):
    spec = llm_module.SetBailianCodingPlanLLMServiceConnector.INPUT_TYPES()
    assert spec["optional"]["config_key"][1]["default"] == "bailian_coding"


def test_set_token_plan_dropdown_includes_coder_and_vl(llm_module):
    # The Token Plan is multimodal: must include at least one VL model and
    # at least one Coder model in the dropdown (Qwen-Image is fine to skip).
    models = llm_module.SetBailianTokenPlanLLMServiceConnector.INPUT_TYPES()[
        "required"
    ]["model_select"][0]
    assert any("vl" in m.lower() for m in models), models
    assert any("coder" in m.lower() for m in models), models
    assert "Custom" in models


def test_set_coding_plan_dropdown_is_coder_only(llm_module):
    # Coding Plan is Qwen-Coder-only. We must not list non-Coder models in
    # the dropdown because selecting one would fail server-side with 400.
    models = llm_module.SetBailianCodingPlanLLMServiceConnector.INPUT_TYPES()[
        "required"
    ]["model_select"][0]
    assert all(("coder" in m.lower()) or (m == "Custom") for m in models), (
        f"Coding Plan dropdown leaked non-coder model: {models}"
    )
    assert "Custom" in models


def test_set_token_plan_execute_routes_to_token_plan_url(llm_module):
    fake_token = "x"
    fake_model = "qwen3-max"
    with patch("services.llm.resolve_token", return_value=fake_token), \
         patch("services.llm.mie_log"):
        node = llm_module.SetBailianTokenPlanLLMServiceConnector()
        connector = node.execute(
            api_token=fake_token,
            model_select=fake_model,
            custom_model="",
        )[0]
    assert isinstance(connector, llm_module.BailianTokenPlanConnectorGeneral)
    assert connector.api_url == llm_module.BailianTokenPlanConnectorGeneral.api_url
    assert connector.model == fake_model


def test_set_coding_plan_execute_routes_to_coding_plan_url(llm_module):
    fake_token = "x"
    fake_model = "qwen3-coder-plus"
    with patch("services.llm.resolve_token", return_value=fake_token), \
         patch("services.llm.mie_log"):
        node = llm_module.SetBailianCodingPlanLLMServiceConnector()
        connector = node.execute(
            api_token=fake_token,
            model_select=fake_model,
            custom_model="",
        )[0]
    assert isinstance(connector, llm_module.BailianCodingPlanConnectorGeneral)
    assert connector.api_url == llm_module.BailianCodingPlanConnectorGeneral.api_url
    assert connector.model == fake_model


# ---------------------------------------------------------------------------
# Payload shape (StandardOpenAICompatibleConnector default)
# ---------------------------------------------------------------------------

def test_token_plan_payload_uses_max_tokens_and_temperature(llm_module):
    # Unlike the legacy PAYG BailianLLMServiceConnector (slim payload),
    # the Token Plan connector inherits StandardOpenAICompatibleConnector
    # which sends the full OpenAI param set.
    c = llm_module.BailianTokenPlanConnectorGeneral("sk-sp-x", "qwen3-max")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "qwen3-max"
    assert payload["stream"] is False
    assert "max_tokens" in payload
    assert "temperature" in payload
    assert "top_p" in payload
    # `response_format` must be present so we don't get default mode
    # mismatches between callers that set JSON mode and callers that don't.
    assert payload.get("response_format") == {"type": "text"}


def test_coding_plan_payload_uses_max_tokens_and_temperature(llm_module):
    c = llm_module.BailianCodingPlanConnectorGeneral("sk-cp-x", "qwen3-coder-plus")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "qwen3-coder-plus"
    assert payload["stream"] is False
    assert "max_tokens" in payload
    assert "temperature" in payload


def test_token_plan_payload_honors_caller_overrides(llm_module):
    c = llm_module.BailianTokenPlanConnectorGeneral("sk-sp-x", "qwen3-max")
    payload = c.generate_payload(
        [{"role": "user", "content": "hi"}],
        max_tokens=256,
        temperature=0.3,
        top_p=0.5,
    )
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.5


# ---------------------------------------------------------------------------
# End-to-end invoke (mocked HTTP)
# ---------------------------------------------------------------------------

def test_token_plan_invoke_posts_to_token_plan_url(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r) as fake_post, \
         patch("services.llm.resolve_token", return_value="x"):
        c = llm_module.BailianTokenPlanConnectorGeneral("x", "qwen3-max")
        out = c.invoke([{"role": "user", "content": "hi"}], max_tokens=128)
    assert out == "hello"
    called_url = fake_post.call_args.args[0]
    assert called_url == llm_module.BailianTokenPlanConnectorGeneral.api_url
    sent = fake_post.call_args.kwargs["json"]
    assert sent["max_tokens"] == 128
    assert sent["model"] == "qwen3-max"


def test_coding_plan_invoke_posts_to_coding_plan_url(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {"content": "code ok"}}]}
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r) as fake_post, \
         patch("services.llm.resolve_token", return_value="x"):
        c = llm_module.BailianCodingPlanConnectorGeneral("x", "qwen3-coder-plus")
        out = c.invoke([{"role": "user", "content": "write hello world"}])
    assert out == "code ok"
    called_url = fake_post.call_args.args[0]
    assert called_url == llm_module.BailianCodingPlanConnectorGeneral.api_url
    assert fake_post.call_args.kwargs["headers"]["Authorization"] == "Bearer x"


def test_token_plan_5xx_is_retried_with_body_snippet(llm_module):
    # Retry contract must match every other connector in this module.
    r5 = MagicMock()
    r5.status_code = 503
    r5.text = "upstream busy"
    r5.json.side_effect = ValueError("not json for 5xx")
    r2 = MagicMock()
    r2.status_code = 200
    r2.json.return_value = {"choices": [{"message": {"content": "retry ok"}}]}

    captured = []
    with patch.object(llm_module, "mie_log", side_effect=lambda m: captured.append(m)), \
         patch("services.llm.time.sleep"), \
         patch("services.llm.requests.post", side_effect=[r5, r2]), \
         patch("services.llm.resolve_token", return_value="x"):
        c = llm_module.BailianTokenPlanConnectorGeneral("x", "qwen3-max")
        out = c.invoke([{"role": "user", "content": "hi"}])
    assert out == "retry ok"
    joined = "\n".join(captured)
    assert "attempt 1/3" in joined
    assert "attempt 2/3" in joined
    assert "HTTP 503" in joined
    assert "upstream busy" in joined


# ---------------------------------------------------------------------------
# mie_llm_keys.json.example must list both new keys
# ---------------------------------------------------------------------------

def test_keys_example_includes_bailian_token_plan_and_coding():
    import json
    p = PROJECT_DIR / "mie_llm_keys.json.example"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "bailian_token_plan" in data, "bailian_token_plan key must be present"
    assert "bailian_coding" in data, "bailian_coding key must be present"
    assert data["bailian_token_plan"] == ""
    assert data["bailian_coding"] == ""
