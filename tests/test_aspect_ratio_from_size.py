"""Tests for general aspect-ratio helpers and node."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME


@pytest.fixture(scope="module")
def aspect_module():
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.common.aspect_ratio")


@pytest.fixture(scope="module")
def prompts_module():
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ideogram4_prompts")


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1024, 1024, "1:1"),
        (1280, 720, "16:9"),
        (720, 1280, "9:16"),
        (1152, 864, "4:3"),
        (1008, 672, "3:2"),
        (1440, 1012, "360:253"),
    ],
)
def test_ratio_from_size(aspect_module, width, height, expected):
    assert aspect_module.ratio_from_size(width, height) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("16:9 (Widescreen)", "16:9"),
        ("1280:720", "1280:720"),
        ("auto", "auto"),
        ("", ""),
    ],
)
def test_normalize_ratio_string(aspect_module, text, expected):
    assert aspect_module.normalize_ratio_string(text) == expected


def test_resolve_aspect_ratio_keeps_non_preset(prompts_module):
    assert prompts_module.resolve_aspect_ratio("360:253") == "360:253"
    assert prompts_module.resolve_aspect_ratio("21:9") == "21:9"
    assert prompts_module.resolve_aspect_ratio("16:9 (Widescreen)") == "16:9"
    assert prompts_module.resolve_aspect_ratio("auto") == "auto"


def test_aspect_ratio_node_registered():
    plugin = load_plugin_module()
    assert "AspectRatioFromSize|Mie" in plugin.NODE_CLASS_MAPPINGS
    assert "Ideogram4AspectRatioFromSize|Mie" not in plugin.NODE_CLASS_MAPPINGS


def test_aspect_ratio_node_convert(aspect_module):
    node = aspect_module.AspectRatioFromSize()
    ratio, = node.convert(1440, 1012)
    assert ratio == "360:253"
