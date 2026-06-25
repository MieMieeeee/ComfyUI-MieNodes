"""Tests for decrement-mode parsing and the MieLoopIfCurrentIdx branch node."""

import pytest

import loop as loop_module


# ---------- decrement parser helpers -----------------------------------

@pytest.mark.parametrize(
    ("total", "step", "expected"),
    [
        # User's example
        (5, 2, [2, 2, 1]),
        # Even total divides cleanly
        (6, 2, [2, 2, 2]),
        # Step larger than total -> single round
        (3, 5, [3]),
        # Step equals total -> single round
        (4, 4, [4]),
        # Total of 0 -> empty
        (0, 2, []),
        # Larger total, larger step
        (10, 3, [3, 3, 3, 1]),
        # Odd total, even step
        (7, 2, [2, 2, 2, 1]),
        # Step of 1
        (4, 1, [1, 1, 1, 1]),
    ],
)
def test_parse_int_decrement( total, step, expected):
    result = loop_module._parse_int_decrement(total, step)
    assert [p["value"] for p in result] == expected


@pytest.mark.parametrize("bad_total", [-1, -10])
def test_parse_int_decrement_rejects_negative( bad_total):
    with pytest.raises(ValueError):
        loop_module._parse_int_decrement(bad_total, 2)


@pytest.mark.parametrize("bad_step", [0, -1, -10])
def test_parse_int_decrement_rejects_non_positive_step( bad_step):
    with pytest.raises(ValueError):
        loop_module._parse_int_decrement(5, bad_step)


def test_parse_int_decrement_total_zero_is_empty_not_error():
    # An empty loop is valid; the start node treats count=0 as a no-op
    # that still returns a fully-formed loop_ctx.
    assert loop_module._parse_int_decrement(0, 2) == []


@pytest.mark.parametrize(
    ("total", "step", "expected"),
    [
        (5.0, 2.0, [2.0, 2.0, 1.0]),
        (1.0, 0.4, [0.4, 0.4, 0.2]),
        (0.0, 1.0, []),
        (3.0, 5.0, [3.0]),
    ],
)
def test_parse_float_decrement( total, step, expected):
    result = loop_module._parse_float_decrement(total, step)
    actual = [p["value"] for p in result]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("bad_total", [-1.0, float("nan"), float("inf")])
def test_parse_float_decrement_rejects_bad_total( bad_total):
    with pytest.raises(ValueError):
        loop_module._parse_float_decrement(bad_total, 1.0)


@pytest.mark.parametrize("bad_step", [0.0, -1.0, float("nan"), float("inf")])
def test_parse_float_decrement_rejects_bad_step( bad_step):
    with pytest.raises(ValueError):
        loop_module._parse_float_decrement(5.0, bad_step)


# ---------- dispatch wiring -------------------------------------------

def test_param_mode_set_includes_decrement():
    # The combo box shown in the UI must contain the new option.
    modes = loop_module.MieLoopStart.INPUT_TYPES()["required"]["param_mode"][0]
    assert "decrement" in modes


def test_parse_params_list_dispatches_int_decrement():
    result = loop_module._parse_params_list(
        "int", "decrement",
        int_decrement_total=5, int_decrement_step=2,
    )
    assert [p["value"] for p in result] == [2, 2, 1]


def test_parse_params_list_dispatches_float_decrement():
    result = loop_module._parse_params_list(
        "float", "decrement",
        float_decrement_total=1.0, float_decrement_step=0.4,
    )
    actual = [p["value"] for p in result]
    assert actual == pytest.approx([0.4, 0.4, 0.2], abs=1e-9)


# ---------- MieLoopStart integration ----------------------------------

def _ctx_skeleton(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "L",
        "run_id": "R",
        "mode": "for_each",
        "index": 0,
        "count": 3,
        "is_last": False,
        "params_list": [{"value": 1}, {"value": 2}, {"value": 3}],
        "current_params": {"value": 1},
        "state": {},
        "collectors": {
            "image": {"ref": None, "count": 0},
            "text": {"ref": None, "count": 0},
            "json": {"ref": None, "count": 0},
        },
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    ctx.update(overrides)
    return ctx


def test_loop_start_with_decrement_mode_initializes_context():
    node = loop_module.MieLoopStart()
    ctx_out, index, count, is_last = node.execute(
        "dec_loop",
        param_type="int",
        param_mode="decrement",
        int_decrement_total=5,
        int_decrement_step=2,
    )
    assert index == 0
    assert count == 3
    assert is_last is False
    assert ctx_out["loop_id"] == "dec_loop"
    assert ctx_out["count"] == 3
    assert [p["value"] for p in ctx_out["params_list"]] == [2, 2, 1]
    assert ctx_out["current_params"]["value"] == 2


def test_loop_start_with_decrement_step_larger_than_total():
    node = loop_module.MieLoopStart()
    ctx_out, _, count, is_last = node.execute(
        "dec_loop",
        param_type="int",
        param_mode="decrement",
        int_decrement_total=3,
        int_decrement_step=5,
    )
    assert count == 1
    assert is_last is True
    assert [p["value"] for p in ctx_out["params_list"]] == [3]


def test_loop_start_with_decrement_total_zero_creates_empty_loop():
    node = loop_module.MieLoopStart()
    ctx_out, index, count, is_last = node.execute(
        "dec_loop",
        param_type="int",
        param_mode="decrement",
        int_decrement_total=0,
        int_decrement_step=2,
    )
    assert count == 0
    assert is_last is True
    assert ctx_out["params_list"] == []
    assert index == 0


def test_loop_start_with_float_decrement():
    node = loop_module.MieLoopStart()
    ctx_out, _, count, _ = node.execute(
        "dec_loop",
        param_type="float",
        param_mode="decrement",
        float_decrement_total=1.0,
        float_decrement_step=0.4,
    )
    assert count == 3
    actual = [p["value"] for p in ctx_out["params_list"]]
    assert actual == pytest.approx([0.4, 0.4, 0.2], abs=1e-9)


# ---------- MieLoopIfCurrentIdx ----------------------------------------

def _ctx_with_index(index, count=4, **overrides):
    # _validate_loop_ctx requires count == len(params_list), so build
    # a fresh params_list of the requested length.
    params = [{"value": i + 1} for i in range(count)]
    return _ctx_skeleton(
        index=index,
        count=count,
        params_list=params,
        is_last=index == count - 1,
        current_params=params[index] if 0 <= index < count else {},
        **overrides,
    )


def test_if_current_idx_node_equality_match():
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(2, count=4)
    out_ctx, value = node.execute(ctx, "==", 2, "T", "F")
    assert out_ctx["index"] == 2
    # Equality with compare_value=2 on round 2 -> THEN branch wins
    assert value == "T"


def test_if_current_idx_node_equality_mismatch():
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(1, count=4)
    _, value = node.execute(ctx, "==", 2, "T", "F")
    # compare_value=2 but round=1 -> condition False, ELSE branch wins
    assert value == "F"


@pytest.mark.parametrize(
    ("operator", "index", "compare", "expected"),
    [
        ("==", 0, 0, True),
        ("==", 1, 0, False),
        ("!=", 0, 0, False),
        ("!=", 1, 0, True),
        ("<",  1, 2, True),
        ("<",  2, 2, False),
        ("<=", 2, 2, True),
        ("<=", 3, 2, False),
        (">",  3, 2, True),
        (">",  2, 2, False),
        (">=", 2, 2, True),
        (">=", 1, 2, False),
    ],
)
def test_if_current_idx_operators( operator, index, compare, expected):
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(index, count=4)
    _, value = node.execute(ctx, operator, compare, "T", "F")
    # On True -> "T"; on False -> "F"
    assert (value == "T") is expected


def test_if_current_idx_passes_loop_ctx_through():
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(1, count=4)
    out_ctx, value = node.execute(ctx, "==", 1, None, None)
    # Both branches unconnected -> selected value is None
    assert value is None
    # Same dict content, may or may not be the same object (no requirement)
    assert out_ctx["index"] == 1
    assert out_ctx["count"] == 4


def test_if_current_idx_unconnected_branches_are_none():
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(0, count=2)
    _, value = node.execute(ctx, "==", 0)
    # Condition True, both branches unconnected -> selected value is None
    assert value is None


def test_if_current_idx_unknown_operator_raises():
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(0, count=2)
    with pytest.raises(ValueError):
        node.execute(ctx, "===", 0)


def test_if_current_idx_rejects_non_integer_compare():
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(0, count=2)
    with pytest.raises(ValueError):
        node.execute(ctx, "==", "not-a-number")


def test_if_current_idx_accepts_then_else_for_arbitrary_types():
    # then/else are any_typ: callers can route images, strings, dicts.
    node = loop_module.MieLoopIfCurrentIdx()
    ctx = _ctx_with_index(2, count=4)
    then_payload = {"name": "first", "frame": 0}
    else_payload = {"name": "rest", "frame": None}
    _, value = node.execute(
        ctx, "==", 0, then_payload, else_payload
    )
    # compare_value=0 but round=2 -> condition False -> ELSE branch wins
    assert value is else_payload


# ---------- registration -----------------------------------------------

def test_if_current_idx_registered_in_plugin_mappings():
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_plugin_imports import load_plugin_module

    plugin = load_plugin_module()
    assert "MieLoopIfCurrentIdx|Mie" in plugin.NODE_CLASS_MAPPINGS
    cls = plugin.NODE_CLASS_MAPPINGS["MieLoopIfCurrentIdx|Mie"]
    assert isinstance(cls, type)
    assert cls.__name__ == "MieLoopIfCurrentIdx"
    # Display name should match what we registered
    assert "If Current Idx" in plugin.NODE_DISPLAY_NAME_MAPPINGS["MieLoopIfCurrentIdx|Mie"]
