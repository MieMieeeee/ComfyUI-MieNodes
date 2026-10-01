# -*- coding: utf-8 -*-
"""Regression tests for the 2026-09-29 LLM platform audit.

Pins the dropdown / payload changes made while verifying every connector
against the live APIs and official docs (see
``_plan_llm_platform_audit.md`` at the repo root for the full audit log):

- GitHub Models was fully retired by GitHub on 2026-07-30 -> connector
  removed from code, ``__init__.py`` and ``mie_llm_keys.json.example``.
- Kimi locks sampling params server-side (temperature=1.0, top_p=0.95,
  n=1, penalties=0; official docs recommend NOT sending them) -> the
  Kimi payload omits them.
- Model dropdowns refreshed against live ``GET /models`` responses
  (SiliconFlow / ZhiPu / Kimi / DeepSeek / Bailian) and official docs
  (Gemini, MiMo, Bailian Token / Coding plans).

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


def _dropdown(llm_module, node_cls_name):
    return llm_module.__dict__[node_cls_name].INPUT_TYPES()["required"][
        "model_select"
    ]


# ---------------------------------------------------------------------------
# GitHub Models retirement (2026-07-30)
# ---------------------------------------------------------------------------

def test_github_models_connector_removed(llm_module):
    assert not hasattr(llm_module, "GithubModelsConnectorGeneral")
    assert not hasattr(llm_module, "SetGithubModelsLLMServiceConnector")


def test_github_models_gone_from_keys_example():
    data = json.loads(
        (PROJECT_DIR / "mie_llm_keys.json.example").read_text(encoding="utf-8")
    )
    assert "github_models" not in data
    assert "github_models" not in data["_comment"]


# ---------------------------------------------------------------------------
# Kimi: locked sampling params + refreshed dropdown
# ---------------------------------------------------------------------------

def test_kimi_payload_omits_locked_sampling_params(llm_module):
    c = llm_module.KimiConnectorGeneral("sk-x", "kimi-k3")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    # Official docs lock these server-side; sending other values 400s live
    # (`invalid temperature: only 1 is allowed for this model`).
    for banned in ("temperature", "top_p", "top_k", "frequency_penalty",
                   "presence_penalty", "n"):
        assert banned not in payload, f"Kimi payload must not send {banned}"
    assert payload["model"] == "kimi-k3"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 512


def test_kimi_payload_forwards_max_tokens_override(llm_module):
    c = llm_module.KimiConnectorGeneral("sk-x", "kimi-k3")
    payload = c.generate_payload(
        [{"role": "user", "content": "hi"}], max_tokens=4096, temperature=0.2
    )
    assert payload["max_tokens"] == 4096
    # Caller temperature is deliberately dropped, not forwarded.
    assert "temperature" not in payload


def test_kimi_dropdown_2026_09(llm_module):
    models, meta = _dropdown(llm_module, "SetKimiLLMServiceConnector")
    assert models == [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
        "Custom",
    ]
    assert meta["default"] == "kimi-k3"


def test_kimi_offline_models_not_listed(llm_module):
    models, _ = _dropdown(llm_module, "SetKimiLLMServiceConnector")
    # kimi-k2.5 and the whole moonshot-v1 family went offline 2026-08-31.
    assert "kimi-k2.5" not in models
    assert not any(m.startswith("moonshot-v1") for m in models)


# ---------------------------------------------------------------------------
# Dropdown refresh per live /models + official docs
# ---------------------------------------------------------------------------

def test_siliconflow_dropdown_2026_09(llm_module):
    models, meta = _dropdown(llm_module, "SetSiliconFlowLLMServiceConnector")
    assert "zai-org/GLM-5.3" in models
    assert "moonshotai/Kimi-K2.7-Code" in models
    assert "Qwen/Qwen3.8-27B" in models
    # Qwen3.5-397B-A17B no longer appears in GET /v1/models.
    assert "Qwen/Qwen3.5-397B-A17B" not in models
    assert meta["default"] == "deepseek-ai/DeepSeek-V4-Flash"


@pytest.mark.parametrize("node", ["SetZhiPuLLMServiceConnector",
                                  "SetZhiPuCodeLLMServiceConnector"])
def test_zhipu_dropdowns_2026_09(llm_module, node):
    models, meta = _dropdown(llm_module, node)
    assert "glm-5.3" in models
    assert "glm-5.3-flash" in models
    assert "glm-5.3-flashx" in models
    assert meta["default"] == "glm-5.3"
    # Both endpoints expose the same 11-model list (verified live).
    other = _dropdown(
        llm_module,
        "SetZhiPuCodeLLMServiceConnector" if node == "SetZhiPuLLMServiceConnector"
        else "SetZhiPuLLMServiceConnector",
    )
    assert models == other[0]


def test_deepseek_dropdown_2026_09(llm_module):
    models, meta = _dropdown(llm_module, "SetDeepSeekLLMServiceConnector")
    # deepseek-v4-flash was renamed deepseek-flash server-side.
    assert models == ["deepseek-v4-pro", "deepseek-flash", "Custom"]
    assert meta["default"] == "deepseek-flash"


def test_bailian_payg_dropdown_2026_09(llm_module):
    models, meta = _dropdown(llm_module, "SetBailianLLMServiceConnector")
    assert "qwen3.8-max" in models
    assert "kimi-k3" in models
    assert "deepseek-v4.1-flash" in models
    assert meta["default"] == "qwen3.8-max"


@pytest.mark.parametrize("node", ["SetMiMoLLMServiceConnector",
                                  "SetMiMoTokenPlanLLMServiceConnector"])
def test_mimo_dropdowns_2026_09(llm_module, node):
    models, meta = _dropdown(llm_module, node)
    assert "mimo-v2.6-pro" in models
    assert "mimo-v2.6-flash" in models
    assert meta["default"] == "mimo-v2.6-pro"
    # v2-* generation no longer listed on the product page.
    assert not any(m.startswith("mimo-v2-") for m in models)


def test_gemini_dropdown_2026_09(llm_module):
    models, meta = _dropdown(llm_module, "SetGeminiLLMServiceConnector")
    assert meta["default"] == "gemini-3.8-flash"
    assert "gemini-3.1-pro-preview" in models
    # gemini-3.1-pro (stable) never existed; 2.5 family is restricted to
    # pre-existing users, so neither belongs in the dropdown.
    assert "gemini-3.1-pro" not in models
    assert not any(m.startswith("gemini-2.5") for m in models)


# ---------------------------------------------------------------------------
# Response hygiene: orphan closing think tags
# ---------------------------------------------------------------------------

def test_sanitize_response_strips_orphan_close_think(llm_module):
    # Seen live from SiliconFlow DeepSeek-V4-Flash: 'OK</think>OK'.
    c = llm_module.GeneralLLMServiceConnector(
        "https://example.invalid", "x", "m"
    )
    assert c._sanitize_response("OK</think>OK") == "OKOK"
    assert c._sanitize_response("a<think>hidden</think>b") == "ab"
    assert c._sanitize_response("</thinking>tail") == "tail"
    assert c._sanitize_response("no tags") == "no tags"
