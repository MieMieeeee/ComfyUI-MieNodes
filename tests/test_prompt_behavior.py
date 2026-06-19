"""End-to-end behavior check for the method-local prompt replacements.

``translator`` (f-string -> ``.format``) and ``PromptGenerator.generate_prompt``
(4 in-method branches) had no importable constant, so their snapshot is
substring-only. This drives the actual node methods with a fake connector and
asserts the system content they send — closing the gap for the f-string→format
conversion and the four branch loads.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME  # noqa: E402


class _FakeConnector:
    def __init__(self):
        self.captured = None

    def invoke(self, messages, **kwargs):
        self.captured = messages
        return "ok"

    def get_state(self):
        return "fake"


@pytest.fixture(scope="module")
def llm():
    load_plugin_module()
    return {
        "translator": importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.translator"),
        "pg": importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.prompt_generator"),
    }


def test_translator_system_content_uses_loaded_template(llm):
    conn = _FakeConnector()
    llm["translator"].TextTranslator().translate_text(conn, text="hello", target_language="ja")
    sys_content = conn.captured[0]["content"]
    assert "translation engineer" in sys_content
    assert "Translate any user input into Japanese." in sys_content
    assert "{language_name}" not in sys_content  # placeholder was filled, not leaked


def test_translator_user_content_passed_through(llm):
    conn = _FakeConnector()
    llm["translator"].TextTranslator().translate_text(conn, text="bonjour", target_language="en")
    assert conn.captured[1]["content"] == "bonjour"


def test_prompt_generator_four_branches_load_distinct_prompts(llm):
    pg = llm["pg"].PromptGenerator()
    conn = _FakeConnector()

    pg.generate_prompt(conn, input_text="", mode="advanced", seed=1)
    adv_empty = conn.captured[0]["content"]
    assert "Generate exactly 1 random" in adv_empty

    pg.generate_prompt(conn, input_text="", mode="simple", seed=1)
    simple_empty = conn.captured[0]["content"]
    assert "expert prompt creator" in simple_empty

    pg.generate_prompt(conn, input_text="a cat", mode="simple", seed=1)
    translate = conn.captured[0]["content"]
    assert "expert prompt translator" in translate
    assert conn.captured[1]["content"] == "a cat"

    pg.generate_prompt(conn, input_text="a cat", mode="advanced", seed=1)
    expand = conn.captured[0]["content"]
    assert "mission is to analyze" in expand

    # all four branches resolved to distinct prompts
    assert len({adv_empty, simple_empty, translate, expand}) == 4
