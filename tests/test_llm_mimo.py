# -*- coding: utf-8 -*-
"""Tests for the Xiaomi MiMo (and MiMo Token Plan) connectors in services/llm.py."""
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
    """Load services/llm.py into a synthetic package context (mirrors
    `test_llm_retry_logging.py`)."""
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
# Connector class structure
# ---------------------------------------------------------------------------

def test_standard_connector_uses_xiaomimimo_api_url(llm_module):
    assert (
        llm_module.MiMoConnectorGeneral.api_url
        == "https://api.xiaomimimo.com/v1/chat/completions"
    )


def test_token_plan_connector_uses_token_plan_api_url(llm_module):
    assert (
        llm_module.MiMoTokenPlanConnectorGeneral.api_url
        == "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
    )


def test_two_connectors_have_distinct_base_urls(llm_module):
    # Catches accidental copy-paste of the standard URL into the token-plan class.
    assert (
        llm_module.MiMoConnectorGeneral.api_url
        != llm_module.MiMoTokenPlanConnectorGeneral.api_url
    )


# ---------------------------------------------------------------------------
# Payload shape: max_completion_tokens, defaults, no top_k/n/response_format
# ---------------------------------------------------------------------------

def test_standard_generate_payload_uses_max_completion_tokens(llm_module):
    c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5-pro")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "mimo-v2.5-pro"
    assert "max_completion_tokens" in payload
    # The legacy OpenAI key must not be present (MiMo only documents the new name).
    assert "max_tokens" not in payload
    # Stream must be off by default (we don't support streaming yet).
    assert payload["stream"] is False


def test_token_plan_generate_payload_uses_max_completion_tokens(llm_module):
    c = llm_module.MiMoTokenPlanConnectorGeneral("tok", "mimo-v2.5-pro")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "mimo-v2.5-pro"
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert payload["stream"] is False


@pytest.mark.parametrize("cls_name", ["MiMoConnectorGeneral", "MiMoTokenPlanConnectorGeneral"])
def test_generate_payload_omits_top_k_n_and_response_format(llm_module, cls_name):
    # The generic OpenAI-compat connector in this file sends `top_k`, `n`, and
    # `response_format`; MiMo's docs never show these, so the MiMo override
    # must strip them. Sending an unknown field to MiMo is likely a 400.
    cls = getattr(llm_module, cls_name)
    c = cls("tok", "mimo-v2.5-pro")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    assert "top_k" not in payload
    assert "n" not in payload
    assert "response_format" not in payload


@pytest.mark.parametrize("cls_name", ["MiMoConnectorGeneral", "MiMoTokenPlanConnectorGeneral"])
def test_generate_payload_uses_mimo_default_temperature_and_top_p(llm_module, cls_name):
    cls = getattr(llm_module, cls_name)
    c = cls("tok", "mimo-v2.5-pro")
    payload = c.generate_payload([{"role": "user", "content": "hi"}])
    # MiMo docs default to temperature=1.0, top_p=0.95 (not the generic 0.7/0.9).
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95


def test_generate_payload_honors_caller_overrides(llm_module):
    # `max_tokens` from the CallLLMService node should be remapped to
    # `max_completion_tokens`. Temperature/top_p overrides must also be honored.
    c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5")
    payload = c.generate_payload(
        [{"role": "user", "content": "hi"}],
        max_tokens=256,
        temperature=0.3,
        top_p=0.5,
    )
    assert payload["max_completion_tokens"] == 256
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.5


# ---------------------------------------------------------------------------
# image_url detail sanitization
# ---------------------------------------------------------------------------

def _img_msg(detail="auto", url="data:image/png;base64,QUJD"):
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ],
    }


@pytest.mark.parametrize("cls_name", ["MiMoConnectorGeneral", "MiMoTokenPlanConnectorGeneral"])
def test_sanitize_drops_detail_auto_but_keeps_low_and_high(llm_module, cls_name):
    # MiMo docs example never sets a `detail` field. Stripping `"auto"` is the
    # safe default; explicit `low` / `high` from the caller must survive.
    cls = getattr(llm_module, cls_name)
    c = cls("tok", "mimo-v2.5")
    msgs = [_img_msg("auto"), _img_msg("low"), _img_msg("high")]

    sanitized = c._sanitize_image_detail(msgs)

    auto_part = sanitized[0]["content"][0]["image_url"]
    low_part = sanitized[1]["content"][0]["image_url"]
    high_part = sanitized[2]["content"][0]["image_url"]

    assert "detail" not in auto_part, "auto detail must be stripped"
    assert low_part.get("detail") == "low"
    assert high_part.get("detail") == "high"


def test_sanitize_does_not_mutate_input_messages(llm_module):
    # Defensive copy: the input list must not be modified in place, so callers
    # that reuse the same messages across retries / providers see no surprises.
    c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5")
    msgs = [_img_msg("auto")]
    original = msgs[0]["content"][0]["image_url"].get("detail")
    c._sanitize_image_detail(msgs)
    assert msgs[0]["content"][0]["image_url"].get("detail") == original


def test_sanitize_passes_through_non_image_parts_untouched(llm_module):
    c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5")
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ],
    }]
    out = c._sanitize_image_detail(msgs)
    assert out[0]["content"][0] == {"type": "text", "text": "describe this"}


# ---------------------------------------------------------------------------
# End-to-end invoke (uses mocked HTTP, mirrors test_llm_retry_logging.py)
# ---------------------------------------------------------------------------

def test_invoke_posts_to_standard_url_and_returns_content(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "choices": [{"message": {"content": "hello from MiMo"}}]
    }
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r) as fake_post, \
         patch("services.llm.resolve_token", return_value="sk-test"):
        c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5-pro")
        out = c.invoke([{"role": "user", "content": "hi"}], max_tokens=128)
    assert out == "hello from MiMo"
    # The HTTP target must be the standard MiMo endpoint.
    called_url = fake_post.call_args.args[0]
    assert called_url == "https://api.xiaomimimo.com/v1/chat/completions"
    # And the payload must use max_completion_tokens (not max_tokens).
    sent = fake_post.call_args.kwargs["json"]
    assert sent["max_completion_tokens"] == 128
    assert "max_tokens" not in sent


def test_invoke_posts_to_token_plan_url(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r) as fake_post, \
         patch("services.llm.resolve_token", return_value="tp-test"):
        c = llm_module.MiMoTokenPlanConnectorGeneral("tp-test", "mimo-v2.5-pro")
        out = c.invoke([{"role": "user", "content": "hi"}])
    assert out == "ok"
    called_url = fake_post.call_args.args[0]
    assert called_url == "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"


def test_5xx_response_is_retried_with_body_snippet(llm_module):
    # Mirrors the 5xx retry contract used by every other connector in this
    # module. Guards against accidentally breaking the retry contract when
    # refactoring the MiMo override.
    r5 = MagicMock()
    r5.status_code = 503
    r5.text = "upstream busy"
    r5.json.side_effect = ValueError("not json for 5xx")
    r2 = MagicMock()
    r2.status_code = 200
    r2.json.return_value = {
        "choices": [{"message": {"content": "second try ok"}}]
    }

    captured = []
    with patch.object(llm_module, "mie_log", side_effect=lambda m: captured.append(m)), \
         patch("services.llm.time.sleep"), \
         patch("services.llm.requests.post", side_effect=[r5, r2]), \
         patch("services.llm.resolve_token", return_value="tok"):
        c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5-pro")
        out = c.invoke([{"role": "user", "content": "hi"}])
    assert out == "second try ok"
    joined = "\n".join(captured)
    assert "attempt 1/3" in joined
    assert "attempt 2/3" in joined
    assert "HTTP 503" in joined
    assert "upstream busy" in joined


def test_think_block_is_stripped_from_mimo_response(llm_module):
    # The base connector strips `<think>...</think>` from the assistant content.
    # MiMo reasoning models (Pro/Omni) emit a `reasoning_content` field, but
    # if any reasoning bleeds into `content`, it must be scrubbed before the
    # downstream prompt-rewriter nodes see it.
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "choices": [{
            "message": {
                "content": "<think>chain of thought</think>final answer"
            }
        }]
    }
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r), \
         patch("services.llm.resolve_token", return_value="tok"):
        c = llm_module.MiMoConnectorGeneral("tok", "mimo-v2.5-pro")
        out = c.invoke([{"role": "user", "content": "hi"}])
    assert "<think>" not in out
    assert out == "final answer"
