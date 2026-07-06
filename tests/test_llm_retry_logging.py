# -*- coding: utf-8 -*-
"""Tests for the per-attempt retry logging in services/llm.py."""
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
    """Load services/llm.py into a synthetic package context."""
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

    if "services" in sys.modules:
        del sys.modules["services"]
    if "services.llm" in sys.modules:
        del sys.modules["services.llm"]
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


def _make_5xx_then_200():
    r5 = MagicMock()
    r5.status_code = 503
    r5.text = "upstream busy\nplease retry later"
    r5.json.side_effect = ValueError("not json for 5xx")
    r2 = MagicMock()
    r2.status_code = 200
    r2.json.return_value = {"choices": [{"message": {"content": "ok answer"}}]}
    return [r5, r2]


def test_5xx_retry_logs_body_snippet_and_per_attempt(llm_module):
    captured = []
    with patch.object(llm_module, "mie_log", side_effect=lambda m: captured.append(m)), \
         patch("services.llm.time.sleep"), \
         patch("services.llm.requests.post", side_effect=_make_5xx_then_200()), \
         patch("services.llm.resolve_token", return_value="tok"):
        c = llm_module.GeneralLLMServiceConnector(
            api_url="https://x/v1/chat/completions", manual_token="", model="M3"
        )
        out = c.invoke([{"role": "user", "content": "hi"}])
    assert out == "ok answer"
    joined = "\n".join(captured)
    # Per-attempt start logs (both attempts)
    assert "attempt 1/3" in joined
    assert "attempt 2/3" in joined
    # 5xx body snippet is included
    assert "HTTP 503" in joined
    assert "upstream busy" in joined
    # Per-attempt success log with elapsed + response_chars
    assert "ok in" in joined
    assert "response_chars=9" in joined


def test_timeout_path_logs_elapsed_and_raises_after_max_retries(llm_module):
    captured = []
    import requests

    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout("read timed out after 30s")

    with patch.object(llm_module, "mie_log", side_effect=lambda m: captured.append(m)), \
         patch("services.llm.time.sleep") as fake_sleep, \
         patch("services.llm.requests.post", side_effect=raise_timeout), \
         patch("services.llm.resolve_token", return_value="tok"):
        c = llm_module.GeneralLLMServiceConnector(
            api_url="https://x/v1/chat/completions", manual_token="", model="M3"
        )
        with pytest.raises(Exception) as exc_info:
            c.invoke([{"role": "user", "content": "hi"}])
    # All 3 attempts started
    joined = "\n".join(captured)
    assert "attempt 1/3" in joined
    assert "attempt 2/3" in joined
    assert "attempt 3/3" in joined
    # Elapsed is reported for the failing attempt
    assert "Timeout after" in joined
    # Only 2 sleeps (between attempts 1->2 and 2->3)
    assert fake_sleep.call_count == 2
    # Final exception mentions max retries
    assert "Max retries (3) exceeded" in str(exc_info.value)


def test_default_timeout_is_30(llm_module):
    import inspect
    sig = inspect.signature(llm_module.GeneralLLMServiceConnector.__init__)
    assert sig.parameters["timeout"].default == 30


# --------------------------------------------------------------------------- #
# Reasoning-model response parsing
#
# MiniMax-M3 / DeepSeek-R1 API / GLM-5.x emit chain-of-thought that may
# consume the whole token budget before the answer. Two shapes are
# handled: (a) reasoning inlined in ``content`` as ``<think>...</think>``
# and stripped by ``_sanitize_response`` (leaving content empty), and
# (b) reasoning split into a separate ``reasoning_content`` field. When
# the sanitized ``content`` is empty, ``invoke`` falls back to
# ``reasoning_content`` so callers don't get a bare empty string.
# --------------------------------------------------------------------------- #
def _conn(llm_module):
    return llm_module.GeneralLLMServiceConnector(
        api_url="https://x/v1/chat/completions", manual_token="", model="M3"
    )


def _run_with_response(llm_module, response_json, captured):
    """Drive ``invoke`` with a single canned 200 response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = response_json
    with patch.object(llm_module, "mie_log", side_effect=lambda m: captured.append(m)), \
         patch("services.llm.time.sleep"), \
         patch("services.llm.requests.post", return_value=resp), \
         patch("services.llm.resolve_token", return_value="tok"):
        return _conn(llm_module).invoke([{"role": "user", "content": "hi"}])


def test_response_falls_back_to_reasoning_content_when_content_empty(llm_module):
    """content = <think>...</think> (sanitize -> empty) + reasoning_content
    set -> invoke returns the reasoning_content text, and a fallback log is
    emitted. Regression for the SCAIL-2 enhance-stage "returned empty" bug."""
    captured = []
    out = _run_with_response(
        llm_module,
        {
            "choices": [{
                "message": {
                    "content": "<think>long chain of thought</think>",
                    "reasoning_content": "The real enhanced prompt.",
                }
            }]
        },
        captured,
    )
    assert out == "The real enhanced prompt."
    joined = "\n".join(captured)
    assert "falling back to reasoning_content" in joined
    # response_chars reflects post-sanitize (post-fallback) length, not the
    # raw content length (36 chars for the <think>...</think> string).
    assert "response_chars=25" in joined  # len("The real enhanced prompt.")


def test_response_returns_content_when_non_empty_after_sanitize(llm_module):
    """Normal case: content has a think block + real answer -> return the
    sanitized answer; reasoning_content is NOT consulted."""
    captured = []
    out = _run_with_response(
        llm_module,
        {
            "choices": [{
                "message": {
                    "content": "<think>chain</think>The answer paragraph.",
                    "reasoning_content": "SHOULD NOT BE USED",
                }
            }]
        },
        captured,
    )
    assert out == "The answer paragraph."
    joined = "\n".join(captured)
    assert "falling back" not in joined


def test_response_chars_log_uses_post_sanitize_length(llm_module):
    """The success log's response_chars reflects the sanitized length, not
    the raw content length. Regression: the old log reported the pre-sanitize
    length and masked the 'returned empty' condition."""
    captured = []
    _run_with_response(
        llm_module,
        {
            "choices": [{
                "message": {
                    "content": "<think>chain</think>answer",  # sanitize -> "answer" (6)
                    "reasoning_content": "",
                }
            }]
        },
        captured,
    )
    joined = "\n".join(captured)
    assert "response_chars=6" in joined
    # The raw content was 27 chars; ensure we are NOT reporting that.
    assert "response_chars=27" not in joined


def test_response_returns_empty_when_both_content_and_reasoning_empty(llm_module):
    """Both content (after sanitize) and reasoning_content empty -> return
    empty string, no crash. Preserves current behavior; the SCAIL-2 enhancer
    handles the empty return upstream by falling back to the original prompt."""
    captured = []
    out = _run_with_response(
        llm_module,
        {
            "choices": [{
                "message": {
                    "content": "<think>chain</think>",
                    "reasoning_content": "",
                }
            }]
        },
        captured,
    )
    assert out == ""
