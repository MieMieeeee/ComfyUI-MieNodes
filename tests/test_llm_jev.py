# -*- coding: utf-8 -*-
"""Tests for the Jev / System One decision-model connector (2026-09-29).

Jev models (TypeSafe AI's Jev + open replicas Kev-4B / SemIf /
diffusiongemma) are NOT chat models: the standard /v1/chat/completions
endpoint rejects them. They live on the bare /v1/systemone endpoint with
a `state + questions` payload and answer with structured decisions
(noul / choice / score). SiliconFlow's endpoint + protocol shape were
verified live on 2026-09-29 with a real key (see
`_plan_llm_platform_expansion.md`); these tests pin the contract with
mocked HTTP. The connector is SiliconFlow-scoped (TypeSafe's own hosted
endpoint is out of scope).

The fixture / module-loading pattern mirrors ``test_llm_bailian_plans.py``.
"""
import importlib.util
import json
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
# Class + URL pinning
# ---------------------------------------------------------------------------

def test_jev_connector_and_nodes_exist(llm_module):
    assert hasattr(llm_module, "SiliconFlowJevConnectorGeneral")
    assert hasattr(llm_module, "SetSiliconFlowJevLLMServiceConnector")
    assert hasattr(llm_module, "CallJevDecision")


def test_jev_url_pinned_to_siliconflow(llm_module):
    c = llm_module.SiliconFlowJevConnectorGeneral("k", "Kev-4B")
    assert c.api_url == "https://api.siliconflow.cn/v1/systemone"
    # Bare /v1/systemone path — NOT the chat-completions endpoint.
    assert not c.api_url.endswith("/chat/completions")


def test_set_jev_execute_builds_siliconflow_connector(llm_module):
    with patch("services.llm.resolve_token", return_value="x"), \
         patch("services.llm.mie_log"):
        node = llm_module.SetSiliconFlowJevLLMServiceConnector()
        c = node.execute(api_token="x", model_select="Kev-4B", custom_model="")[0]
        cc = node.execute(api_token="x", model_select="Custom",
                          custom_model="SemIf")[0]
    assert c.api_url == "https://api.siliconflow.cn/v1/systemone"
    assert c.model == "Kev-4B"
    assert cc.model == "SemIf"


# ---------------------------------------------------------------------------
# ask() protocol shape (mocked HTTP)
# ---------------------------------------------------------------------------

_LIVE_SHAPE = {
    "model": "Kev-4B",
    "answers": {
        "urgency": {"type": "noul", "noul": 0.9586},
        "topic": {"type": "choice", "choice": "billing", "confidence": 0.4434,
                  "probabilities": {"billing": 0.6289, "technical": 0.3603}},
        "severity": {"type": "score", "score": 1.6709,
                     "legend": {"0": "Low", "1": "Medium", "2": "High"},
                     "confidence": 0.8355},
    },
    "usage": {"input_tokens": 77, "output_tokens": 175},
}


def test_ask_posts_state_and_questions(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = _LIVE_SHAPE
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r) as fake_post:
        c = llm_module.SiliconFlowJevConnectorGeneral("x", "Kev-4B", max_retries=1)
        questions = {"urgency": {"type": "noul", "instructions": "urgent?"}}
        out = c.ask("customer email", questions)
    assert out == _LIVE_SHAPE
    sent = fake_post.call_args.kwargs["json"]
    assert sent == {"model": "Kev-4B", "state": "customer email",
                    "questions": questions}
    assert fake_post.call_args.args[0] == "https://api.siliconflow.cn/v1/systemone"


def test_ask_retries_on_5xx(llm_module):
    r5, r2 = MagicMock(), MagicMock()
    r5.status_code, r5.text = 503, "busy"
    r2.status_code = 200
    r2.json.return_value = _LIVE_SHAPE
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.time.sleep"), \
         patch("services.llm.requests.post", side_effect=[r5, r2]):
        c = llm_module.SiliconFlowJevConnectorGeneral("x", "Kev-4B")
        out = c.ask("s", {"q": {"type": "noul", "instructions": "?"}})
    assert out == _LIVE_SHAPE


def test_invoke_requires_questions_and_serializes_answers(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = _LIVE_SHAPE
    c = llm_module.SiliconFlowJevConnectorGeneral("x", "Kev-4B", max_retries=1)
    # Without questions: loud pointer to the right node, not a silent 400.
    with pytest.raises(Exception, match="CallJevDecision"):
        c.invoke([{"role": "user", "content": "hi"}])
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r):
        out = c.invoke([{"role": "user", "content": "hi"}],
                       questions={"q": {"type": "noul", "instructions": "?"}})
    assert json.loads(out) == _LIVE_SHAPE["answers"]


# ---------------------------------------------------------------------------
# CallJevDecision node
# ---------------------------------------------------------------------------

def _jev_conn(llm_module):
    with patch("services.llm.resolve_token", return_value="x"), \
         patch("services.llm.mie_log"):
        return llm_module.SetSiliconFlowJevLLMServiceConnector().execute(
            api_token="x", model_select="Kev-4B", custom_model="")[0]


def test_call_jev_decision_returns_full_json(llm_module):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = _LIVE_SHAPE
    node = llm_module.CallJevDecision()
    questions = json.dumps({"urgency": {"type": "noul",
                                        "instructions": "urgent?"}})
    with patch.object(llm_module, "mie_log"), \
         patch("services.llm.requests.post", return_value=r):
        out = node.execute(_jev_conn(llm_module), "customer email", questions)[0]
    assert json.loads(out) == _LIVE_SHAPE


def test_call_jev_decision_rejects_bad_json(llm_module):
    node = llm_module.CallJevDecision()
    with pytest.raises(Exception, match="not valid JSON"):
        node.execute(_jev_conn(llm_module), "s", "{not json")


def test_call_jev_decision_rejects_empty_and_non_object(llm_module):
    node = llm_module.CallJevDecision()
    with pytest.raises(Exception, match="non-empty JSON object"):
        node.execute(_jev_conn(llm_module), "s", "")
    with pytest.raises(Exception, match="non-empty JSON object"):
        node.execute(_jev_conn(llm_module), "s", "[1, 2]")


def test_call_jev_decision_rejects_chat_connector(llm_module):
    chat_conn = llm_module.SiliconFlowConnectorGeneral("x", "m", max_retries=1)
    node = llm_module.CallJevDecision()
    with pytest.raises(Exception, match="SetSiliconFlowJevLLMServiceConnector"):
        node.execute(chat_conn, "s", '{"q": {}}')


def test_keys_example_has_no_typesafe_slot():
    # The connector is SiliconFlow-scoped and reuses the `siliconflow` slot.
    data = json.loads(
        (PROJECT_DIR / "mie_llm_keys.json.example").read_text(encoding="utf-8")
    )
    assert "typesafe" not in data
