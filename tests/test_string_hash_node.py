"""Tests for StringHash|Mie (sha256 of a STRING, truncated to a hex digest).

Mirrors the test style of test_string_format_node.py: import the plugin
module via the shared helper, then exercise the class through its
public execute() method. No JS extension to test (StringHash has a
single STRING input, so the autogrow UX is irrelevant).
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
    return plugin.NODE_CLASS_MAPPINGS["StringHash|Mie"]


@pytest.fixture()
def node(cls):
    return cls()


# ------------------------------ schema ----------------------------------- #

def test_node_registered(plugin):
    assert "StringHash|Mie" in plugin.NODE_CLASS_MAPPINGS
    assert "StringHash|Mie" in plugin.NODE_DISPLAY_NAME_MAPPINGS


def test_input_types_shape(cls):
    spec = cls.INPUT_TYPES()
    required = spec.get("required") or {}
    optional = spec.get("optional") or {}
    # Required: just the text widget.
    assert list(required.keys()) == ["text"]
    # text is a single-line STRING widget, not multiline -- multiline would
    # add newlines that change the hash for a key built on a one-line prompt.
    text_spec = required["text"]
    assert text_spec[0] == "STRING"
    assert text_spec[1].get("multiline") is False
    # Optional: length INT in [4, 64], default 12 (matches ImageHash|Mie).
    assert list(optional.keys()) == ["length"]
    length_spec = optional["length"]
    assert length_spec[0] == "INT"
    assert length_spec[1].get("default") == 12
    assert length_spec[1].get("min") == 4
    assert length_spec[1].get("max") == 64


def test_function_and_return_types(cls):
    assert cls.FUNCTION == "execute"
    assert cls.RETURN_TYPES == ("STRING",)
    assert cls.RETURN_NAMES == ("hash",)


# ----------------------------- execute() --------------------------------- #

def test_known_vector_human(node):
    # Sanity check against a hand-computed sha256.
    import hashlib
    expected = hashlib.sha256("human".encode("utf-8")).hexdigest()[:12]
    assert node.execute("human") == (expected,)


def test_deterministic(node):
    # Same input -> same output, every call.
    a = node.execute("a person in a red shirt")
    b = node.execute("a person in a red shirt")
    assert a == b
    assert len(a[0]) == 12


def test_different_inputs_different_output(node):
    # A single-char change should fully change the hash (avalanche).
    a = node.execute("person")
    b = node.execute("persons")
    assert a != b


def test_empty_string_returns_zero_sentinel(node):
    # Empty input is a legitimate unconnected-input state; should not
    # crash, should return the all-zero sentinel so the cache-key
    # template still produces a valid filename.
    assert node.execute("") == ("0" * 12,)


def test_none_input_returns_zero_sentinel(node):
    # Defensive: same behavior as empty when the slot is genuinely None.
    assert node.execute(None) == ("0" * 12,)


def test_length_truncation(node):
    assert node.execute("human", 8) == ("79a54787",)
    assert node.execute("human", 16) == ("79a5478768d24474",)
    # Full sha256.
    assert node.execute("human", 64) == (
        "79a5478768d2447431a90f7f4549df735f50ad541371464c248abc7522dc3a01",
    )


def test_length_clamps_out_of_range(node):
    # Below min -> 4 chars (clamped up).
    assert node.execute("human", 2) == ("79a5",)
    # Above max -> 64 chars (clamped down).
    h = node.execute("human", 999)[0]
    assert len(h) == 64


def test_length_invalid_value_falls_back_to_default(node):
    # Non-int (e.g. a STRING that snuck in via *) should not crash;
    # fall back to the default 12.
    assert node.execute("human", "twelve") == ("79a5478768d2",)


def test_filename_hostile_inputs_stay_safe(node):
    # The whole point of hashing the prompt: spaces and slashes in
    # the original text must not appear in the cache-key component.
    h1 = node.execute("a person in a red shirt")[0]
    h2 = node.execute("person/face")[0]
    h3 = node.execute("a\nmultiline\nprompt")[0]
    for h in (h1, h2, h3):
        assert " " not in h
        assert "/" not in h
        assert "\\" not in h
        assert "\n" not in h
        assert len(h) == 12
    # And distinct: each prompt is a different cache key.
    assert len({h1, h2, h3}) == 3
