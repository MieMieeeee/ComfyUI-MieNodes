"""Tests for the ``RoundToMultiple`` node and its helpers."""

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module, PACKAGE_NAME


@pytest.fixture(scope="module")
def number_ops():
    load_plugin_module()
    return importlib.import_module(f"{PACKAGE_NAME}.nodes.common.number_ops")


@pytest.fixture(scope="module")
def plugin():
    return load_plugin_module()


# ---------- pure helper coverage ------------------------------------------

@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0),
        (0.5, 1),
        (1.5, 2),
        (2.5, 3),       # half-away-from-zero, not banker's rounding
        (3.5, 4),
        (-0.5, -1),
        (-1.5, -2),
        (-2.5, -3),     # half-away-from-zero on the negative side
        (1234.49, 1234),
        (1234.51, 1235),
    ],
)
def test_round_half_away_from_zero(number_ops, x, expected):
    assert number_ops._round_half_away_from_zero(x) == expected


@pytest.mark.parametrize(
    ("value", "multiple", "rounding", "expected"),
    [
        # ceil always goes up
        (17, 8, "ceil", 24),
        (16, 8, "ceil", 16),    # already on the grid
        (0, 8, "ceil", 0),
        (-17, 8, "ceil", -16),  # ceil of -17/-2.125 is -2
        # floor always goes down
        (17, 8, "floor", 16),
        (24, 8, "floor", 24),
        (-17, 8, "floor", -24),
        # round is nearest, half away from zero
        (17, 8, "round", 16),   # 17/8 = 2.125 -> 2
        (20, 8, "round", 24),   # 20/8 = 2.5 -> 3 (away from zero)
        (12, 8, "round", 16),   # 12/8 = 1.5 -> 2 (half away from zero)
        (-20, 8, "round", -24), # -20/8 = -2.5 -> -3 (away from zero)
        # non-integer multiple
        (5.3, 0.5, "round", 5.5),
        (5.3, 0.5, "floor", 5.0),
        (5.3, 0.5, "ceil", 5.5),
        # already a multiple
        (32, 8, "round", 32),
    ],
)
def test_snap_to_multiple(number_ops, value, multiple, rounding, expected):
    result = number_ops._snap_to_multiple(value, multiple, rounding)
    assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-9)


@pytest.mark.parametrize("bad", ["banker", "", "CEIL", None])
def test_snap_to_multiple_rejects_unknown_mode(number_ops, bad):
    with pytest.raises(ValueError):
        number_ops._snap_to_multiple(10.0, 8.0, bad)


def test_snap_to_multiple_zero_multiple_falls_back(number_ops):
    # Zero step would normally raise ZeroDivisionError; the helper degrades
    # to a step of 1 so the workflow keeps running.
    assert number_ops._snap_to_multiple(7.4, 0.0, "round") == 7.0
    assert number_ops._snap_to_multiple(7.4, 0.0, "ceil") == 8.0


def test_snap_to_multiple_nan_multiple_falls_back(number_ops):
    nan = float("nan")
    assert math.isclose(
        number_ops._snap_to_multiple(7.4, nan, "round"),
        7.0,
        rel_tol=0,
        abs_tol=1e-9,
    )


# ---------- node behavior --------------------------------------------------


def test_round_to_multiple_node_outputs_int_and_float(number_ops):
    node = number_ops.RoundToMultiple()
    f_out, i_out = node.round(17.0, 8.0, "ceil")
    assert isinstance(f_out, float)
    assert isinstance(i_out, int)
    assert f_out == 24.0
    assert i_out == 24


def test_round_to_multiple_node_accepts_int_input(number_ops):
    # ComfyUI will hand ints to a FLOAT input; make sure the node still
    # returns matching int/float ports.
    node = number_ops.RoundToMultiple()
    f_out, i_out = node.round(20, 8, "round")
    assert f_out == 24.0
    assert i_out == 24


def test_input_types_declares_required_fields(number_ops):
    spec = number_ops.RoundToMultiple.INPUT_TYPES()
    required = spec["required"]
    assert set(required) == {"value", "multiple", "rounding"}
    assert required["value"][0] == "FLOAT"
    assert required["multiple"][0] == "FLOAT"
    assert list(required["rounding"][0]) == ["round", "floor", "ceil"]
    assert number_ops.RoundToMultiple.RETURN_TYPES == ("FLOAT", "INT")
    assert number_ops.RoundToMultiple.RETURN_NAMES == ("float", "int")


# ---------- registration ---------------------------------------------------


def test_round_to_multiple_node_registered(plugin):
    assert "RoundToMultiple|Mie" in plugin.NODE_CLASS_MAPPINGS
    cls = plugin.NODE_CLASS_MAPPINGS["RoundToMultiple|Mie"]
    assert isinstance(cls, type)
    # Display name should mention the operation
    assert "Round To Multiple" in plugin.NODE_DISPLAY_NAME_MAPPINGS["RoundToMultiple|Mie"]