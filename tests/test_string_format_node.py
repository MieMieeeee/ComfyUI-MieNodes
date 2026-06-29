"""Tests for StringFormat|Mie (str.format-style template, autogrow inputs).

The class itself is exercised end-to-end through `format()`; the JS-side
autogrow UX is verified manually because the project has no Playwright
hook for LiteGraph node trees (the contract test below just guarantees
the schema stays stable for saved workflows).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module  # noqa: E402


@pytest.fixture(scope="module")
def plugin():
    return load_plugin_module()


@pytest.fixture(scope="module")
def cls(plugin):
    return plugin.NODE_CLASS_MAPPINGS["StringFormat|Mie"]


@pytest.fixture()
def node(cls):
    return cls()


# ------------------------------ schema ----------------------------------- #

def test_node_registered(plugin):
    assert "StringFormat|Mie" in plugin.NODE_CLASS_MAPPINGS
    assert "StringFormat|Mie" in plugin.NODE_DISPLAY_NAME_MAPPINGS


def test_input_types_shape(cls):
    spec = cls.INPUT_TYPES()
    required = spec.get("required") or {}
    optional = spec.get("optional") or {}
    # Required: only the template widget.
    assert list(required.keys()) == ["template"]
    # Optional: contiguous value_0..value_15, all STRING forceInput.
    names = [f"value_{i}" for i in range(16)]
    assert list(optional.keys()) == names
    for name, spec_value in optional.items():
        assert spec_value[0] == "STRING"
        # forceInput is what lets the JS extension addInput() the next slot
        # when the previous one is wired; without it the slot would be a
        # widget and cannot accept an input link.
        assert spec_value[1].get("forceInput") is True


def test_validate_inputs_always_true(cls):
    # Mirrors SaveAny/LoadAny: structurally always valid. The JS
    # extension owns the "how many slots are visible" UX.
    assert cls.VALIDATE_INPUTS(template="x") is True
    assert cls.VALIDATE_INPUTS(template="x", value_0="a") is True
    assert cls.VALIDATE_INPUTS(template="x", value_0="a", value_99="b") is True


def test_function_and_return_types(cls):
    assert cls.FUNCTION == "format"
    assert cls.RETURN_TYPES == ("STRING",)
    assert cls.RETURN_NAMES == ("result",)


# ----------------------------- format() ---------------------------------- #

def test_basic_positional(node):
    assert node.format("{0} + {1} = {2}",
                       value_0="1", value_1="2", value_2="3") == ("1 + 2 = 3",)


def test_unconnected_slots_become_empty(node):
    # Only value_0 and value_1 wired; {2} should render as "" rather
    # than raising IndexError.
    assert node.format("{0} + {1} = {2}",
                       value_0="1", value_1="2") == ("1 + 2 = ",)


def test_no_values_just_template(node):
    assert node.format("hello world") == ("hello world",)


def test_empty_template(node):
    assert node.format("") == ("",)
    assert node.format("", value_0="x") == ("",)


def test_none_template(node):
    # Defensive: a None template should not crash, just render empty.
    assert node.format(None, value_0="x") == ("",)


def test_format_spec_width(node):
    assert node.format("{0:>5}|{1:<3}", value_0="a", value_1="b") == ("    a|b  ",)


def test_literal_braces_escape(node):
    # `{{` and `}}` are the str.format escape for literal `{` and `}`.
    assert node.format("literal {{0}}={0}", value_0="val") == ("literal {0}=val",)


def test_multiline_template(node):
    assert node.format("line1:{0}\nline2:{1}",
                       value_0="a", value_1="b") == ("line1:a\nline2:b",)


def test_all_sixteen_slots(node):
    all_16 = {f"value_{i}": str(i) for i in range(16)}
    tpl = "-".join(f"{{{i}}}" for i in range(16))
    assert node.format(tpl, **all_16) == ("0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15",)


def test_index_out_of_range_falls_back(node):
    # Template references {16} but MAX_FORMAT_VALUES is 16 (slots 0-15).
    # The lenient behavior is to log and return the raw template so the
    # user can spot their mistake on the wire.
    assert node.format("{0}-{16}", value_0="a") == ("{0}-{16}",)


def test_invalid_format_spec_falls_back(node, capsys):
    # `{0:` is an unclosed format spec; str.format raises ValueError.
    r = node.format("{0:", value_0="x")
    assert r == ("{0:",)
    # The failure is logged for the operator; the user-facing value is
    # the raw template (so the workflow can still run with a visible
    # template string instead of crashing).
    captured = capsys.readouterr()
    assert "StringFormat|Mie" in captured.out
    assert "format failed" in captured.out


def test_numeric_strings_kept_as_strings(node):
    # All values are STRING; numeric formatting stays string-side.
    assert node.format("{0}+{1}", value_0="10", value_1="5") == ("10+5",)


# ------------------------ JS extension presence -------------------------- #

def test_js_extension_file_exists():
    js_path = Path(__file__).resolve().parents[1] / "js" / "stringFormatAutogrow.js"
    assert js_path.is_file(), "js/stringFormatAutogrow.js must exist for the autogrow UX"


def test_js_extension_wires_target_node():
    js_text = (Path(__file__).resolve().parents[1] / "js" / "stringFormatAutogrow.js").read_text(
        encoding="utf-8"
    )
    # The extension must target the same node id the backend uses.
    assert '"StringFormat|Mie"' in js_text
    # The extension must hook the LiteGraph node lifecycle: onNodeCreated
    # for the initial trim, onConnectionsChange for the autogrow grow.
    assert "onNodeCreated" in js_text
    assert "onConnectionsChange" in js_text
    # The slot name + type must match the Python INPUT_TYPES contract.
    assert "value_" in js_text
    assert '"STRING"' in js_text
