# -*- coding: utf-8 -*-
"""Tests for the Krea-2 prompt generator (text -> text T2I prompt expansion).

Mirrors the loader-stubbing pattern from ``test_scail2_prompts.py`` so the
project layout (root has hyphens, cannot be imported as a normal package)
keeps working under pytest.

Coverage:
* Bundled prompt files (``expansion.txt`` + ``UPSTREAM.md``) are present.
* ``expansion.txt`` is a verbatim copy of the upstream system prompt
  (key phrases + size).
* ``krea2_prompts.build_krea2_messages`` returns 2 messages with the
  expansion prompt as system content and aspect_ratio + user prompt in
  the user content.
* ``krea2_prompts.resolve_aspect_ratio`` handles empty / ``auto`` /
  valid ``W:H`` / garbage inputs.
* ``Krea2PromptGenerator`` is importable, has the right INPUT_TYPES
  shape, returns a single STRING, and lives in the Prompt Generator
  category.
* ``Krea2PromptEnhancer`` happy path runs one LLM call, returns the
  stripped paragraph, and raises ``ValueError`` on empty user_prompt.
* ``is_changed`` is stable for identical inputs and varies when the
  meaningful inputs change.
* The plugin root exposes ``Krea2PromptGenerator|Mie`` after a full
  module load.
"""
import importlib
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "tests"))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME  # noqa: E402

KREA2_PROMPTS_PATH = PROJECT_DIR / "nodes" / "llm" / "krea2_prompts.py"
KREA2_GEN_PATH = PROJECT_DIR / "nodes" / "llm" / "krea2_prompt_generator.py"
KREA2_PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts" / "krea2"


@pytest.fixture(scope="module")
def krea2():
    """Import the helper + generator modules under the ``PACKAGE_NAME``
    namespace, matching the runtime layout the project uses inside ComfyUI.
    """
    load_plugin_module()
    prompts = importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.krea2_prompts")
    gen = importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.krea2_prompt_generator")
    return prompts, gen


# --------------------------------------------------------------------------- #
# Bundled prompt-file tests
# --------------------------------------------------------------------------- #
def test_prompt_files_exist():
    """``expansion.txt`` and ``UPSTREAM.md`` are on disk under
    ``nodes/llm/prompts/krea2/``.
    """
    assert (KREA2_PROMPTS_DIR / "expansion.txt").is_file()
    assert (KREA2_PROMPTS_DIR / "UPSTREAM.md").is_file()


def test_expansion_txt_matches_upstream():
    """``expansion.txt`` is a verbatim copy of krea-ai/krea-2's
    ``docs/expansion.txt`` (2111 bytes, 9 rules).
    """
    text = (KREA2_PROMPTS_DIR / "expansion.txt").read_text(encoding="utf-8")
    # Upstream file is exactly 2111 bytes.
    assert len(text) == 2111, f"unexpected size {len(text)} (expected 2111)"
    # Key phrases from upstream expansion.txt must all be present.
    must_have = [
        "You are an expert prompt engineer for text-to-image models.",
        "Think step by step about the request before writing the answer",
        "Then output a single expanded prompt paragraph.",
        "Faithfulness First",
        "Practical T2I Structure",
        "Style Planning Stays Internal",
        "Text Rendering",
        "Avoid Over-Specification",
        "Respect Existing Detail",
        "Respect the Human Form",
        "Preserve User Medium",
    ]
    for phrase in must_have:
        assert phrase in text, f"missing phrase: {phrase!r}"
    # All 9 numbered rules must be present (1. .. 9.).
    for i in range(1, 10):
        assert f"\n{i}. **" in text, f"missing rule {i}"


def test_upstream_md_links_to_krea2_docs():
    """``UPSTREAM.md`` references both upstream URLs so the source is
    easy to audit.
    """
    text = (KREA2_PROMPTS_DIR / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "krea-ai/krea-2" in text
    assert "expansion.txt" in text
    assert "prompting.md" in text


# --------------------------------------------------------------------------- #
# Helper module tests
# --------------------------------------------------------------------------- #
def test_load_krea2_system_prompt_returns_bundled_text(krea2):
    """``load_krea2_system_prompt()`` returns the verbatim expansion.txt
    contents (no transformation).
    """
    prompts, _ = krea2
    text = prompts.load_krea2_system_prompt()
    assert text == (KREA2_PROMPTS_DIR / "expansion.txt").read_text(encoding="utf-8")
    assert "Faithfulness First" in text


def test_build_krea2_messages_structure(krea2):
    """``build_krea2_messages`` returns a 2-element list with system and
    user roles, carrying the system prompt verbatim and the user prompt
    plus the resolved aspect_ratio in the user content.
    """
    prompts, _ = krea2
    messages = prompts.build_krea2_messages("a red apple", "16:9")
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # System carries the upstream expansion.txt verbatim.
    assert "Faithfulness First" in messages[0]["content"]
    assert "Preserve User Medium" in messages[0]["content"]
    # User carries both the user prompt and the aspect ratio.
    assert "a red apple" in messages[1]["content"]
    assert "16:9" in messages[1]["content"]


def test_build_krea2_messages_auto_aspect_ratio(krea2):
    """``aspect_ratio='auto'`` is preserved in the user content and does
    not get coerced to a fallback ratio.
    """
    prompts, _ = krea2
    messages = prompts.build_krea2_messages("a cat", "auto")
    assert "auto" in messages[1]["content"]
    assert "AUTO" in messages[1]["content"].upper() or "auto" in messages[1]["content"]


def test_resolve_aspect_ratio(krea2):
    """``resolve_aspect_ratio`` accepts ``W:H``, ``auto``, and an empty
    string; garbage falls back to the default.
    """
    prompts, _ = krea2
    assert prompts.resolve_aspect_ratio("16:9") == "16:9"
    assert prompts.resolve_aspect_ratio("3:2") == "3:2"
    assert prompts.resolve_aspect_ratio("auto") == "auto"
    assert prompts.resolve_aspect_ratio("AUTO") == "auto"
    # Empty -> fallback.
    assert prompts.resolve_aspect_ratio("") == "1:1"
    # Garbage -> fallback (does not crash, does not echo garbage).
    assert prompts.resolve_aspect_ratio("not-a-ratio") == "1:1"
    # Custom fallback is honored.
    assert prompts.resolve_aspect_ratio("", fallback="4:3") == "4:3"


# --------------------------------------------------------------------------- #
# ComfyUI-node contract tests
# --------------------------------------------------------------------------- #
def test_comfyui_node_input_types(krea2):
    """``Krea2PromptGenerator.INPUT_TYPES`` exposes the right ports.
    Required: llm_service_connector, user_prompt, seed.
    Optional: aspect_ratio (forceInput STRING), temperature, timeout, max_tokens.
    """
    _, gen = krea2
    spec = gen.Krea2PromptGenerator.INPUT_TYPES()
    required = spec["required"]
    assert "llm_service_connector" in required
    assert "user_prompt" in required
    assert "seed" in required
    # user_prompt should be a multiline STRING widget.
    up = required["user_prompt"]
    assert up[0] == "STRING"
    assert up[1].get("multiline") is True
    optional = spec["optional"]
    assert "aspect_ratio" in optional
    assert optional["aspect_ratio"][0] == "STRING"
    assert optional["aspect_ratio"][1].get("forceInput") is True
    assert "temperature" in optional
    assert "timeout" in optional
    assert "max_tokens" in optional


def test_comfyui_node_return_types(krea2):
    """Single STRING port named ``krea2_prompt``."""
    _, gen = krea2
    assert gen.Krea2PromptGenerator.RETURN_TYPES == ("STRING",)
    assert gen.Krea2PromptGenerator.RETURN_NAMES == ("krea2_prompt",)


def test_comfyui_node_category(krea2):
    """Lives in the same Prompt Generator category as Bernini / Ideogram4 /
    Scail2.
    """
    _, gen = krea2
    cat = gen.Krea2PromptGenerator.CATEGORY
    assert "Prompt Generator" in cat


def test_node_registered_in_plugin():
    """The plugin root exposes ``Krea2PromptGenerator|Mie`` after a full
    module load.
    """
    plugin = load_plugin_module()
    assert "Krea2PromptGenerator|Mie" in plugin.NODE_CLASS_MAPPINGS
    display = plugin.NODE_DISPLAY_NAME_MAPPINGS
    assert "Krea2PromptGenerator|Mie" in display
    assert "Krea2" in display["Krea2PromptGenerator|Mie"]


# --------------------------------------------------------------------------- #
# Enhancer behavior
# --------------------------------------------------------------------------- #
def test_enhancer_happy_path_invokes_llm_once(krea2, monkeypatch):
    """A non-empty prompt triggers exactly one LLM call; the returned text
    is stripped of leading/trailing whitespace.
    """
    _, gen = krea2

    captured = {}

    class FakeConnector:
        model = "fake-model"

        def get_state(self):
            return "fake-state"

        def invoke(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "  An expanded prompt paragraph.  "

    enhancer = gen.Krea2PromptEnhancer(FakeConnector(), timeout=120)
    out = enhancer("a red apple", "16:9", seed=42)
    assert out == "An expanded prompt paragraph."
    # Two messages: system + user, in that order.
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"
    # Temperature and max_tokens are forwarded.
    assert "temperature" in captured["kwargs"]
    assert "max_tokens" in captured["kwargs"]


def test_enhancer_empty_user_prompt_raises(krea2):
    """Empty user_prompt is a programmer error -- the upstream
    expansion.txt expects a real prompt. The enhancer must raise
    ``ValueError`` instead of silently calling the LLM with no input.
    """
    _, gen = krea2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    enhancer = gen.Krea2PromptEnhancer(FakeConnector())
    with pytest.raises(ValueError, match="user_prompt"):
        enhancer("", "1:1")


def test_enhancer_strips_think_block(krea2):
    """Reasoning models (M3, DeepSeek-R1, GLM-5.x) may emit a leading
    ``<think>...</think>`` block before the final paragraph. Per upstream
    rule 6 the visible answer is the paragraph only, so the enhancer
    strips any leading think block.
    """
    _, gen = krea2

    class FakeConnector:
        model = "fake-model"

        def get_state(self):
            return "fake-state"

        def invoke(self, messages, **kwargs):
            return (
                "<think>Let me think about this user prompt...</think>\n"
                "An expanded prompt paragraph."
            )

    enhancer = gen.Krea2PromptEnhancer(FakeConnector())
    out = enhancer("a red apple", "1:1")
    assert out == "An expanded prompt paragraph."
    assert "<think>" not in out


def test_enhancer_normalizes_aspect_ratio(krea2, monkeypatch):
    """An empty aspect_ratio falls back to the default ``1:1`` rather
    than being passed verbatim.
    """
    _, gen = krea2

    captured = {}

    class FakeConnector:
        model = "fake-model"

        def get_state(self):
            return "fake-state"

        def invoke(self, messages, **kwargs):
            captured["user_content"] = messages[1]["content"]
            return "ok"

    enhancer = gen.Krea2PromptEnhancer(FakeConnector())
    enhancer("a cat", "")
    # The user content must NOT contain an empty aspect_ratio line.
    assert "Aspect ratio: \n" not in captured["user_content"]
    assert "1:1" in captured["user_content"]


def test_enhancer_auto_aspect_ratio_preserved(krea2):
    """``auto`` is a valid aspect-ratio hint for Krea-2 (the upstream
    model decides). It must not be coerced to ``1:1``.
    """
    _, gen = krea2

    captured = {}

    class FakeConnector:
        model = "fake-model"

        def get_state(self):
            return "fake-state"

        def invoke(self, messages, **kwargs):
            captured["user_content"] = messages[1]["content"]
            return "ok"

    enhancer = gen.Krea2PromptEnhancer(FakeConnector())
    enhancer("a cat", "auto")
    assert "auto" in captured["user_content"]


def test_enhancer_timeout_override_restores_connector(krea2):
    """The per-call timeout override is saved and restored around the
    invoke, so the connector object is safe to share with other nodes.
    """
    _, gen = krea2

    class FakeConnector:
        model = "fake-model"
        timeout = 30

        def get_state(self):
            return "fake-state"

        def invoke(self, messages, **kwargs):
            # During invoke the override should have been applied.
            assert self.timeout == 120
            return "ok"

    connector = FakeConnector()
    enhancer = gen.Krea2PromptEnhancer(connector, timeout=120)
    enhancer("hello", "1:1")
    assert connector.timeout == 30, "timeout was not restored after invoke"


# --------------------------------------------------------------------------- #
# is_changed
# --------------------------------------------------------------------------- #
def test_is_changed_stable_for_identical_inputs(krea2):
    """Same inputs -> same hash."""
    _, gen = krea2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    node = gen.Krea2PromptGenerator()
    a = node.is_changed(
        FakeConnector(),
        user_prompt="a cat",
        seed=0,
        aspect_ratio="1:1",
        temperature=0.7,
        timeout=120,
        max_tokens=4096,
    )
    b = node.is_changed(
        FakeConnector(),
        user_prompt="a cat",
        seed=0,
        aspect_ratio="1:1",
        temperature=0.7,
        timeout=120,
        max_tokens=4096,
    )
    assert a == b


def test_is_changed_varies_with_user_prompt(krea2):
    _, gen = krea2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    node = gen.Krea2PromptGenerator()
    a = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, aspect_ratio="1:1"
    )
    b = node.is_changed(
        FakeConnector(), user_prompt="a dog", seed=0, aspect_ratio="1:1"
    )
    assert a != b


def test_is_changed_varies_with_aspect_ratio(krea2):
    _, gen = krea2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    node = gen.Krea2PromptGenerator()
    a = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, aspect_ratio="1:1"
    )
    b = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, aspect_ratio="16:9"
    )
    assert a != b


def test_is_changed_varies_with_temperature(krea2):
    _, gen = krea2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    node = gen.Krea2PromptGenerator()
    a = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, temperature=0.7
    )
    b = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, temperature=0.9
    )
    assert a != b


def test_is_changed_varies_with_timeout(krea2):
    _, gen = krea2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    node = gen.Krea2PromptGenerator()
    a = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, timeout=30
    )
    b = node.is_changed(
        FakeConnector(), user_prompt="a cat", seed=0, timeout=120
    )
    assert a != b
