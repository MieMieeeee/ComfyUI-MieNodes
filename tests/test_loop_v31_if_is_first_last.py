"""Tests for the MieLoopIfIsFirst / MieLoopIfIsLast branch nodes."""

import pytest

import loop as loop_module


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


def _ctx_with(index, is_last, count=4):
    # _validate_loop_ctx requires count == len(params_list).
    params = [{"value": i + 1} for i in range(count)]
    return _ctx_skeleton(
        index=index,
        is_last=is_last,
        count=count,
        params_list=params,
        current_params=params[index] if 0 <= index < count else {},
    )


# ---------- MieLoopIfIsFirst -------------------------------------------

@pytest.mark.parametrize(
    ("index", "is_last", "expected"),
    [
        (0, False, True),    # round 0 of 3 -> first, not last
        (1, False, False),   # middle
        (2, True,  False),   # last
        (0, True,  True),    # single-round loop: first AND last
    ],
)
def test_if_is_first_detects_index_zero(index, is_last, expected):
    node = loop_module.MieLoopIfIsFirst()
    ctx = _ctx_with(index, is_last, count=3)
    out_ctx, value = node.execute(ctx, "T", "F")
    # value == "T" iff the boolean decision was True
    assert (value == "T") is expected
    # loop_ctx must still be passed through untouched
    assert out_ctx["index"] == index


def test_if_is_first_passes_loop_ctx_through():
    node = loop_module.MieLoopIfIsFirst()
    ctx = _ctx_with(0, False, count=3)
    out_ctx, value = node.execute(ctx, "T", "F")
    assert out_ctx["index"] == 0
    assert out_ctx["count"] == 3
    assert value == "T"


def test_if_is_first_routes_then_else_branches():
    node = loop_module.MieLoopIfIsFirst()
    ctx_first = _ctx_with(0, False, count=3)
    ctx_mid = _ctx_with(1, False, count=3)
    _, value_first = node.execute(ctx_first, "first-round", "later")
    _, value_mid = node.execute(ctx_mid, "first-round", "later")
    # On the first round the THEN branch wins
    assert value_first == "first-round"
    # On a middle round the ELSE branch wins
    assert value_mid == "later"


def test_if_is_first_unconnected_branches_are_none():
    node = loop_module.MieLoopIfIsFirst()
    ctx = _ctx_with(0, False, count=3)
    out_ctx, value = node.execute(ctx)
    assert out_ctx["index"] == 0
    # The condition is True, so ``then_value`` (unconnected) is returned
    assert value is None


def test_if_is_first_accepts_arbitrary_branch_types():
    node = loop_module.MieLoopIfIsFirst()
    ctx = _ctx_with(0, False, count=3)
    then_payload = {"role": "init", "warmup": True}
    else_payload = {"role": "loop", "warmup": False}
    out_ctx, value = node.execute(ctx, then_payload, else_payload)
    assert out_ctx["index"] == 0
    assert value is then_payload


@pytest.mark.parametrize(
    ("index", "is_last", "expected"),
    [
        (0, False, False),  # round 0 of 3 -> not last
        (1, False, False),  # middle
        (2, True,  True),   # last
        (0, True,  True),   # single-round loop: first AND last
    ],
)
def test_if_is_last_mirrors_ctx_flag(index, is_last, expected):
    node = loop_module.MieLoopIfIsLast()
    ctx = _ctx_with(index, is_last, count=3)
    out_ctx, value = node.execute(ctx, "T", "F")
    assert (value == "T") is expected
    assert out_ctx["index"] == index


def test_if_is_last_passes_loop_ctx_through():
    node = loop_module.MieLoopIfIsLast()
    ctx = _ctx_with(2, True, count=3)
    out_ctx, value = node.execute(ctx, "T", "F")
    assert out_ctx["index"] == 2
    assert out_ctx["is_last"] is True
    assert value == "T"


def test_if_is_last_routes_then_else_branches():
    node = loop_module.MieLoopIfIsLast()
    ctx_last = _ctx_with(2, True, count=3)
    ctx_mid = _ctx_with(1, False, count=3)
    _, value_last = node.execute(ctx_last, "flush", "loop")
    _, value_mid = node.execute(ctx_mid, "flush", "loop")
    assert value_last == "flush"
    assert value_mid == "loop"


def test_if_is_last_unconnected_branches_are_none():
    node = loop_module.MieLoopIfIsLast()
    ctx = _ctx_with(2, True, count=3)
    out_ctx, value = node.execute(ctx)
    assert out_ctx["index"] == 2
    assert value is None


def test_if_is_last_accepts_arbitrary_branch_types():
    node = loop_module.MieLoopIfIsLast()
    ctx = _ctx_with(2, True, count=3)
    then_payload = {"role": "flush", "save": True}
    else_payload = {"role": "loop", "save": False}
    out_ctx, value = node.execute(ctx, then_payload, else_payload)
    assert out_ctx["is_last"] is True
    assert value is then_payload


# ---------- node contract & registration ------------------------------

def test_if_is_first_contract():
    inputs = loop_module.MieLoopIfIsFirst.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]
    assert list(required) == ["loop_ctx"]
    assert "then_value" in optional
    assert "else_value" in optional
    # Selected branch is exposed as a single any-typ value port.
    assert loop_module.MieLoopIfIsFirst.RETURN_TYPES == (
        "MIE_LOOP_CTX", loop_module.any_typ,
    )
    assert loop_module.MieLoopIfIsFirst.RETURN_NAMES == ("loop_ctx", "value")


def test_if_is_last_contract():
    inputs = loop_module.MieLoopIfIsLast.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]
    assert list(required) == ["loop_ctx"]
    assert "then_value" in optional
    assert "else_value" in optional
    # Selected branch is exposed as a single any-typ value port.
    assert loop_module.MieLoopIfIsLast.RETURN_TYPES == (
        "MIE_LOOP_CTX", loop_module.any_typ,
    )
    assert loop_module.MieLoopIfIsLast.RETURN_NAMES == ("loop_ctx", "value")


def test_first_and_last_nodes_registered():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_plugin_imports import load_plugin_module

    plugin = load_plugin_module()
    for key in ("MieLoopIfIsFirst|Mie", "MieLoopIfIsLast|Mie"):
        assert key in plugin.NODE_CLASS_MAPPINGS, key
        cls = plugin.NODE_CLASS_MAPPINGS[key]
        assert isinstance(cls, type)
        assert cls.__name__ in ("MieLoopIfIsFirst", "MieLoopIfIsLast")

    assert "If Is First" in plugin.NODE_DISPLAY_NAME_MAPPINGS["MieLoopIfIsFirst|Mie"]
    assert "If Is Last" in plugin.NODE_DISPLAY_NAME_MAPPINGS["MieLoopIfIsLast|Mie"]
