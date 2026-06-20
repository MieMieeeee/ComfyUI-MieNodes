"""Tests for Ideogram 4 caption formatting helpers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME


@pytest.fixture(scope="module")
def fmt_module():
    load_plugin_module()
    import importlib
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.llm.ideogram4_prompt_formatter")


VALID_MINIMAL = {
    "compositional_deconstruction": {
        "background": "A plain white tabletop",
        "elements": [
            {"type": "obj", "desc": "A red apple centered on the table"}
        ],
    }
}


def test_strip_fences_and_compact(fmt_module):
    raw = '```json\n' + json.dumps(VALID_MINIMAL) + '\n```'
    out, log = fmt_module.format_ideogram4_caption(raw)
    data = json.loads(out)
    assert "compositional_deconstruction" in data
    assert "removed markdown" in log
    assert "\n" not in out


def test_fail_missing_compositional(fmt_module):
    with pytest.raises(ValueError, match="compositional_deconstruction"):
        fmt_module.format_ideogram4_caption('{"high_level_description":"only this"}')


def test_fail_empty_elements(fmt_module):
    bad = {
        "compositional_deconstruction": {
            "background": "bg",
            "elements": [],
        }
    }
    with pytest.raises(ValueError, match="at least one element"):
        fmt_module.format_ideogram4_caption(json.dumps(bad))


def test_normalize_hex_and_reorder(fmt_module):
    raw = {
        "style_description": {
            "medium": "photograph",
            "photo": "50mm",
            "lighting": "soft",
            "aesthetics": "clean",
            "color_palette": ["#ff0000"],
        },
        "compositional_deconstruction": {
            "elements": [{"desc": "apple", "type": "obj"}],
            "background": "white",
        },
    }
    out, log = fmt_module.format_ideogram4_caption(json.dumps(raw))
    data = json.loads(out)
    assert data["style_description"]["color_palette"] == ["#FF0000"]
    assert list(data["style_description"].keys())[:2] == ["aesthetics", "lighting"]
    assert "normalized" in log or "reordered" in log


def test_formatter_node_not_registered():
    plugin = load_plugin_module()
    assert "Ideogram4PromptFormatter|Mie" not in plugin.NODE_CLASS_MAPPINGS


def test_removes_stray_aspect_ratio_keeps_bbox(fmt_module):
    raw = {
        "aspect_ratio": "16:9",
        "compositional_deconstruction": {
            "background": "bg",
            "elements": [{"type": "obj", "bbox": [100, 200, 300, 400], "desc": "x"}],
        },
    }
    out, log = fmt_module.format_ideogram4_caption(json.dumps(raw))
    data = json.loads(out)
    assert "aspect_ratio" not in data
    assert data["compositional_deconstruction"]["elements"][0]["bbox"] == [100, 200, 300, 400]
    assert "removed top-level aspect_ratio" in log


def test_trailing_comma_repair(fmt_module):
    raw = '{"compositional_deconstruction":{"background":"bg","elements":[{"type":"obj","desc":"x"},],},}'
    out, log = fmt_module.format_ideogram4_caption(raw)
    assert json.loads(out)["compositional_deconstruction"]["elements"][0]["desc"] == "x"


def test_repair_text_from_quoted_desc(fmt_module):
    raw = {
        "compositional_deconstruction": {
            "background": "studio",
            "elements": [
                {
                    "type": "text",
                    "bbox": [850, 50, 950, 150],
                    "desc": "magazine masthead text 'VELVET' in elegant serif typography",
                    "color_palette": ["#1A1A1A"],
                }
            ],
        }
    }
    out, log = fmt_module.format_ideogram4_caption(json.dumps(raw))
    data = json.loads(out)
    assert data["compositional_deconstruction"]["elements"][0]["text"] == "VELVET"
    assert "inferred elements[0].text" in log


def test_fail_text_without_literal_or_quotes(fmt_module):
    raw = {
        "compositional_deconstruction": {
            "background": "studio",
            "elements": [
                {
                    "type": "text",
                    "bbox": [800, 880, 980, 940],
                    "desc": "cover date and issue number in clean sans-serif",
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="'text' must exist"):
        fmt_module.format_ideogram4_caption(json.dumps(raw))
