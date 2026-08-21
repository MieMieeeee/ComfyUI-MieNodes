# -*- coding: utf-8 -*-
"""Tests for the Ollama connector and SetOllamaLLMServiceConnector node in services/llm.py."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
LLM_PATH = PROJECT_DIR / "services" / "llm.py"
UTILS_PATH = PROJECT_DIR / "core" / "utils.py"


def _load_llm_module():
    """Load services/llm.py into a synthetic package context (mirrors
    `test_llm_retry_logging.py` and `test_llm_mimo.py`)."""
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
# Connector construction: URL composition + token handling
# ---------------------------------------------------------------------------

def test_ollama_connector_appends_v1_chat_completions_path(llm_module):
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "qwen2.5")
    assert c.api_url == "http://127.0.0.1:11434/v1/chat/completions"


def test_ollama_connector_strips_trailing_slash_on_host(llm_module):
    c = llm_module.OllamaConnectorGeneral("http://localhost:11434/", "llama3.2")
    assert c.api_url == "http://localhost:11434/v1/chat/completions"


def test_ollama_connector_handles_lan_host(llm_module):
    c = llm_module.OllamaConnectorGeneral("http://192.168.1.10:11434", "gemma3")
    assert c.api_url == "http://192.168.1.10:11434/v1/chat/completions"


def test_ollama_connector_handles_docker_host(llm_module):
    c = llm_module.OllamaConnectorGeneral(
"http://host.docker.internal:11434", "deepseek-r1")
    assert c.api_url == "http://host.docker.internal:11434/v1/chat/completions"


def test_ollama_connector_uses_placeholder_token_when_empty(llm_module):
    """Empty api_token must not produce a bare 'Bearer ' header -- the
    connector falls back to 'ollama' as a non-empty placeholder."""
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "qwen2.5", api_token="")
    assert c.api_token == "ollama"


def test_ollama_connector_passes_explicit_token_through(llm_module):
    """If the caller supplied a real token (e.g. for a reverse-proxy with auth),
    do NOT overwrite it with the placeholder."""
    c = llm_module.OllamaConnectorGeneral(
"http://127.0.0.1:11434", "qwen2.5", api_token="my-secret")
    assert c.api_token == "my-secret"


def test_ollama_connector_preserves_model_field(llm_module):
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "qwen3:32b")
    assert c.model == "qwen3:32b"


# ---------------------------------------------------------------------------
# Image-detail sanitization: Ollama is NOT like MiniMax / MiMo
# ---------------------------------------------------------------------------

def test_ollama_does_not_strip_image_detail_auto(llm_module):
    """Ollama's OpenAI-compat endpoint accepts `detail: \"auto\"` on image_url
    parts (unlike MiniMax / MiMo which 400 on it). The base-class identity
    behavior must be preserved -- Ollama must NOT use _drop_image_detail_auto."""
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "llava")
    msgs = [{
"role": "user",
"content": [{
"type": "image_url",
"image_url": {"url": "data:image/jpeg;base64,xxx", "detail": "auto"},
}],
}]
    sanitized = c._sanitize_image_detail(msgs)
    assert sanitized[0]["content"][0]["image_url"]["detail"] == "auto"


def test_ollama_does_not_use_drop_image_detail_auto(llm_module):
    """Defensive: make sure we did not accidentally copy the MiniMax/MiMo
    override onto the Ollama connector."""
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "qwen2.5")
    assert c._sanitize_image_detail.__func__ is llm_module.GeneralLLMServiceConnector._sanitize_image_detail


# ---------------------------------------------------------------------------
# Thinking-block sanitization: inherited from base class
# ---------------------------------------------------------------------------

def test_ollama_strips_think_blocks_by_default(llm_module):
    """Reasoning models (deepseek-r1, qwen3) emit <think>...</think>.
    The base-class _sanitize_response should strip them automatically."""
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "deepseek-r1")
    raw = "<think>internal chain of thought</think>\n\nFinal answer here."
    cleaned = c._sanitize_response(raw)
    assert "<think>" not in cleaned
    assert cleaned == "Final answer here."


def test_ollama_preserve_thinking_keeps_think_block(llm_module):
    """When preserve_thinking=True, the chain of thought is preserved
    (used by CheckLLMServiceConnectivity diagnostics)."""
    c = llm_module.OllamaConnectorGeneral("http://127.0.0.1:11434", "deepseek-r1")
    raw = "<think>chain</think>final"
    assert c._sanitize_response(raw, preserve_thinking=True) == raw


# ---------------------------------------------------------------------------
# SetOllamaLLMServiceConnector node: input shape + execute behavior
# ---------------------------------------------------------------------------

def test_node_input_types_default_host_is_localhost(llm_module):
    types_ = llm_module.SetOllamaLLMServiceConnector.INPUT_TYPES()
    assert types_["required"]["host"][1]["default"] == "http://127.0.0.1:11434"


def test_node_input_types_model_is_freeform_string_with_hint(llm_module):
    """model is a plain STRING with a placeholder hint -- we let the user
    type whatever they have locally (could be a tag, a partial hash, etc.)."""
    types_ = llm_module.SetOllamaLLMServiceConnector.INPUT_TYPES()
    field = types_["required"]["model"]
    assert field[0] == "STRING"
    assert field[1]["default"] == ""
    assert "placeholder" in field[1] and field[1]["placeholder"]


def test_node_input_types_api_token_is_optional(llm_module):
    """Unlike every other Set*LLMServiceConnector, api_token is in `optional`
    because Ollama does not require auth."""
    types_ = llm_module.SetOllamaLLMServiceConnector.INPUT_TYPES()
    assert "api_token" in types_["optional"]


def test_node_input_types_config_key_default_is_ollama(llm_module):
    """Default config_key must match the ollama entry in mie_llm_keys.json.example."""
    types_ = llm_module.SetOllamaLLMServiceConnector.INPUT_TYPES()
    assert types_["optional"]["config_key"][1]["default"] == "ollama"


def test_node_blank_model_falls_back_to_qwen2_5(llm_module):
    """If the user left the model field blank, fall back to a sensible default
    rather than failing with 'model is required'."""
    node = llm_module.SetOllamaLLMServiceConnector()
    connector = node.execute(host="http://127.0.0.1:11434", model="")[0]
    assert connector.model == "qwen2.5"


def test_node_passes_user_model_through(llm_module):
    node = llm_module.SetOllamaLLMServiceConnector()
    connector = node.execute(host="http://127.0.0.1:11434", model="llama3.2-vision")[0]
    assert connector.model == "llama3.2-vision"
    assert connector.api_url == "http://127.0.0.1:11434/v1/chat/completions"


def test_node_returns_ollama_connector_type(llm_module):
    """Sanity check that we did not accidentally route through the General connector."""
    node = llm_module.SetOllamaLLMServiceConnector()
    connector = node.execute(host="http://127.0.0.1:11434", model="qwen2.5")[0]
    assert type(connector).__name__ == "OllamaConnectorGeneral"


def test_node_returns_tuple_of_one_connector(llm_module):
    """The node must return a 1-tuple to satisfy ComfyUI's RETURN_TYPES contract."""
    node = llm_module.SetOllamaLLMServiceConnector()
    out = node.execute(host="http://127.0.0.1:11434", model="qwen2.5")
    assert isinstance(out, tuple)
    assert len(out) == 1

# ---------------------------------------------------------------------------
# timeout field: cold-start friendly default (Ollama can take 30-90s to
# load a 7B+ model from disk on first call; the base class default of 30s
# would always timeout).
# ---------------------------------------------------------------------------

def test_node_input_types_timeout_default_is_60(llm_module):
    types_ = llm_module.SetOllamaLLMServiceConnector.INPUT_TYPES()
    field = types_["optional"]["timeout"]
    assert field[0] == "INT"
    assert field[1]["default"] == 60
    # Reasonable bounds so the user can't accidentally set timeout=0 or
    # a multi-hour value that would hang the workflow.
    assert field[1]["min"] == 1
    assert field[1]["max"] >= 60


def test_node_default_timeout_propagates_to_connector(llm_module):
    """When the user leaves the timeout input blank, the connector gets 60s."""
    node = llm_module.SetOllamaLLMServiceConnector()
    connector = node.execute(host="http://127.0.0.1:11434", model="qwen2.5")[0]
    assert connector.timeout == 60


def test_node_explicit_timeout_propagates_to_connector(llm_module):
    """An explicit timeout value must reach the GeneralLLMServiceConnector base."""
    node = llm_module.SetOllamaLLMServiceConnector()
    connector = node.execute(
        host="http://127.0.0.1:11434", model="qwen2.5", timeout=120,
    )[0]
    assert connector.timeout == 120


def test_node_small_timeout_also_propagates(llm_module):
    """User can dial timeout DOWN to e.g. 10s for fast fail on quick models."""
    node = llm_module.SetOllamaLLMServiceConnector()
    connector = node.execute(
        host="http://127.0.0.1:11434", model="qwen2.5", timeout=10,
    )[0]
    assert connector.timeout == 10
