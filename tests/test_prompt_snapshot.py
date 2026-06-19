"""Byte-level snapshot safety net for the prompt-externalization refactor.

Asserts ``load_prompt_text`` / ``load_prompt_dict`` reproduce the EXACT current
values of the in-code prompt constants, so moving prompts out to files cannot
change behavior.

Coverage:
- Module-level constants (the bulk): full ``==`` snapshots.
- dict constants: ``==`` plus key-order (dropdown default = first key).
- Placeholder-bearing prompts: extra asserts that ``{}`` / ``{prompt}`` /
  ``{image_num}`` etc. survive verbatim (loader must never ``.format()``).
- Bernini templates: each must ``.format(**kwargs)`` without KeyError.
- Method-local prompts (PromptGenerator.generate_prompt branches, TextTranslator)
  have no importable constant, so they are pinned by signature substrings plus
  (translator) the ``.format()`` result.

These are RED until the loader + prompt files exist (phase 1), then go GREEN.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME  # noqa: E402


@pytest.fixture(scope="module")
def loader():
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.prompts.loader")


def _mod(name):
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.{name}")


# (logical_name, module, attr) — full == snapshots against current constants.
TEXT_PROMPTS = [
    ("hunyuan/t2v", "prompt_generator", "HYVIDEO_T2V_SYSTEM_PROMPT"),
    ("hunyuan/i2v", "prompt_generator", "HYVIDEO_I2V_SYSTEM_PROMPT"),
    ("zimage/t2i", "prompt_generator", "ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE"),
    ("flux2/t2i", "prompt_generator", "FLUX2_T2I_SYSTEM_PROMPT"),
    ("ltx2/system", "prompt_generator", "LTX2_SYSTEM_PROMPT"),
    ("flux_klein/t2v", "prompt_generator", "FLUX_KLEIN_T2V_SYSTEM_PROMPT"),
    ("bernini/t2v_a14b_en", "bernini_prompts", "T2V_A14B_EN_SYS_PROMPT"),
    ("bernini/t2i_a14b_en", "bernini_prompts", "T2I_A14B_EN_SYS_PROMPT"),
    ("bernini/r2v", "bernini_prompts", "R2V_TEMPLATE"),
    ("bernini/r2i", "bernini_prompts", "R2I_TEMPLATE"),
    ("bernini/vr2v", "bernini_prompts", "VR2V_TEMPLATE"),
    ("bernini/v2v", "bernini_prompts", "V2V_TEMPLATE"),
    ("bernini/i2i", "bernini_prompts", "I2I_TEMPLATE"),
    ("bernini/i2v", "bernini_prompts", "I2V_TEMPLATE"),
    ("bernini/vi2v", "bernini_prompts", "VI2V_TEMPLATE"),
    ("bernini/ads2v", "bernini_prompts", "ADS2V_TEMPLATE"),
    ("bernini/ri2i", "bernini_prompts", "RI2I_TEMPLATE"),
    ("ideogram4/compact_system", "ideogram4_prompts", "COMPACT_SYSTEM_PROMPT"),
    ("ideogram4/full_palette_appendix", "ideogram4_prompts", "FULL_PALETTE_APPENDIX"),
    ("ideogram4/user_template_magic_v1", "ideogram4_prompts", "USER_TEMPLATE_MAGIC_V1"),
]

DICT_PROMPTS = [
    ("kontext/presets", "prompt_generator", "KONTEXT_PRESETS"),
    ("frame_transition/system_prompts", "prompt_generator", "FRAME_TRANSITION_SYSTEM_PROMPTS"),
    ("bernini/system_prompts", "bernini_prompts", "SYSTEM_PROMPTS"),
]


@pytest.mark.parametrize("logical_name,module,attr", TEXT_PROMPTS)
def test_text_prompt_snapshot(loader, logical_name, module, attr):
    expected = getattr(_mod(module), attr)
    assert loader.load_prompt_text(logical_name) == expected


@pytest.mark.parametrize("logical_name,module,attr", DICT_PROMPTS)
def test_dict_prompt_snapshot(loader, logical_name, module, attr):
    expected = getattr(_mod(module), attr)
    loaded = loader.load_prompt_dict(logical_name)
    assert loaded == expected
    assert list(loaded.keys()) == list(expected.keys())  # order = dropdown default


def test_placeholder_braces_preserved(loader):
    # loader must return raw text; placeholders survive verbatim.
    assert "{}" in loader.load_prompt_text("hunyuan/i2v")
    assert "{prompt}" in loader.load_prompt_text("zimage/t2i")


def test_kontext_has_cjk_keys(loader):
    keys = "".join(loader.load_prompt_dict("kontext/presets").keys())
    assert any(ord(c) > 0x7F for c in keys)  # Chinese preset names preserved


def test_bernini_templates_format_without_keyerror(loader):
    cases = {
        "bernini/r2v": {"image_num": 1, "original_text": "x"},
        "bernini/r2i": {"image_num": 2, "original_text": "y"},
        "bernini/vr2v": {"image_num": 1, "original_text": "z"},
        "bernini/v2v": {"user_prompt": "p"},
        "bernini/i2i": {"user_prompt": "p"},
        "bernini/i2v": {"user_prompt": "p", "image_num": 1},
        "bernini/vi2v": {"user_prompt": "p", "image_num": 2},
        "bernini/ads2v": {"user_prompt": "p"},
        "bernini/ri2i": {"ref_num": 1, "original_text": "t"},
    }
    for name, kwargs in cases.items():
        text = loader.load_prompt_text(name)
        formatted = text.format(**kwargs)
        assert formatted != text  # substitution actually happened


# Method-local prompts (no importable constant) — pinned by signature content.
def test_prompt_generator_branch_substrings(loader):
    assert "Generate exactly 1 random" in loader.load_prompt_text("prompt_generator/random_advanced")
    assert "expert prompt creator" in loader.load_prompt_text("prompt_generator/random_simple")
    assert "expert prompt translator" in loader.load_prompt_text("prompt_generator/translate_simple")
    assert "mission is to analyze" in loader.load_prompt_text("prompt_generator/expand_advanced")


def test_translator_template_formats(loader):
    tmpl = loader.load_prompt_text("translator/system_template")
    assert "{language_name}" in tmpl
    out = tmpl.format(language_name="Chinese")
    assert "Translate any user input into Chinese." in out
    assert "translation engineer" in out
