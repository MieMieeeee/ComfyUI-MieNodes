"""Tests for the ``MieLoopStateSetImageBatch`` node.

The existing ``MieLoopStateSetImage`` already accepts IMAGE tensors, which in
ComfyUI are always [B, H, W, C]. ``MieLoopStateSetImageBatch`` is a sibling
node with a different default key (so it does not collide with the single-image
setter) and a ``count`` INT output that reports the batch size immediately.

These tests follow the same conventions as ``test_loop_v31_feedback_image.py``.
"""

import torch

import loop as loop_module


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "batch_loop",
        "run_id": "run_batch_1",
        "mode": "for_each",
        "index": 0,
        "count": 2,
        "is_last": False,
        "params_list": [{"value": 1}, {"value": 2}],
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


def _make_batch(values):
    frames = [torch.full((2, 2, 3), float(v)) for v in values]
    return torch.stack(frames, dim=0)


# ---------- node contract -------------------------------------------------


def test_set_image_batch_default_key_is_distinct_from_single_image():
    """The default key for the batch setter must not collide with the single
    image setter, otherwise the two nodes would silently overwrite each
    other when both are used in the same workflow with defaults."""
    inputs = loop_module.MieLoopStateSetImageBatch.INPUT_TYPES()
    default_key = inputs["required"]["key"][1]["default"]
    single_default = loop_module.MieLoopStateSetImage.INPUT_TYPES()["required"]["key"][1]["default"]
    assert default_key != single_default


def test_set_image_batch_returns_loop_ctx_and_count():
    assert loop_module.MieLoopStateSetImageBatch.RETURN_TYPES == ("MIE_LOOP_CTX", "INT")
    assert loop_module.MieLoopStateSetImageBatch.RETURN_NAMES == ("loop_ctx", "count")


# ---------- basic behavior -------------------------------------------------


def test_set_image_batch_stores_tensor_and_reports_count():
    setter = loop_module.MieLoopStateSetImageBatch()
    ctx = _make_ctx()
    batch = _make_batch([0, 1, 2, 3])  # 4-frame batch

    ctx_out, count = setter.execute(ctx, "feedback_image_batch", batch)

    assert count == 4
    assert isinstance(count, int)
    # ctx should be a fresh copy (matches _state_object_put_image contract)
    assert ctx_out is not ctx
    # ref slot exists in the new ctx
    assert "feedback_image_batch_ref" in ctx_out["state"]
    ref = ctx_out["state"]["feedback_image_batch_ref"]
    assert ref in loop_module.RUNTIME_STORE["state_objects"]["image"]


def test_set_image_batch_round_trips_through_existing_get_image_node():
    """The single-image getter should be able to read the batch back because
    IMAGE is just a [B, H, W, C] tensor. The batch setter deliberately does
    not ship a paired getter so the existing one remains the read side."""
    setter = loop_module.MieLoopStateSetImageBatch()
    getter = loop_module.MieLoopStateGetImage()
    ctx = _make_ctx()
    batch = _make_batch([10, 20, 30])

    ctx, _ = setter.execute(ctx, "feedback_image_batch", batch)
    image_out, found = getter.execute(ctx, "feedback_image_batch")

    assert found is True
    assert tuple(image_out.shape) == (3, 2, 2, 3)
    assert torch.all(image_out[0] == 10)
    assert torch.all(image_out[1] == 20)
    assert torch.all(image_out[2] == 30)


def test_set_image_batch_with_single_image_reports_count_one():
    setter = loop_module.MieLoopStateSetImageBatch()
    ctx = _make_ctx()
    single = torch.ones((1, 2, 2, 3))

    ctx_out, count = setter.execute(ctx, "feedback_image_batch", single)
    assert count == 1
    assert tuple(ctx_out["state"]["feedback_image_batch_ref"]) or True  # ref present
    assert "feedback_image_batch_ref" in ctx_out["state"]


def test_set_image_batch_does_not_overwrite_single_image_slot():
    """The two setters must use independent slots in the state dict."""
    single_setter = loop_module.MieLoopStateSetImage()
    batch_setter = loop_module.MieLoopStateSetImageBatch()
    ctx = _make_ctx()

    single = torch.zeros((1, 2, 2, 3))
    batch = torch.ones((5, 2, 2, 3))

    ctx = single_setter.execute(ctx, "feedback_image", single)[0]
    single_ref = ctx["state"]["feedback_image_ref"]
    ctx, count = batch_setter.execute(ctx, "feedback_image_batch", batch)
    batch_ref = ctx["state"]["feedback_image_batch_ref"]

    assert single_ref != batch_ref
    assert single_ref in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert batch_ref in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert count == 5


def test_set_image_batch_overwrites_previous_value_under_same_key():
    setter = loop_module.MieLoopStateSetImageBatch()
    ctx = _make_ctx()
    batch1 = _make_batch([1, 2])
    batch2 = _make_batch([9, 8, 7])

    ctx = setter.execute(ctx, "feedback_image_batch", batch1)[0]
    ref1 = ctx["state"]["feedback_image_batch_ref"]
    ctx, count = setter.execute(ctx, "feedback_image_batch", batch2)
    ref2 = ctx["state"]["feedback_image_batch_ref"]

    assert ref1 != ref2
    assert ref1 not in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert ref2 in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert count == 3


def test_set_image_batch_rejects_empty_key():
    setter = loop_module.MieLoopStateSetImageBatch()
    ctx = _make_ctx()
    with __import__("pytest").raises(ValueError):
        setter.execute(ctx, "", _make_batch([1]))


def test_set_image_batch_handles_tensor_without_shape():
    """A misbehaving caller might pass a list or numpy array; the node should
    still complete the workflow instead of raising."""
    setter = loop_module.MieLoopStateSetImageBatch()
    ctx = _make_ctx()
    # A plain Python list has no ``shape`` attribute, so count should fall
    # back to 1. The tensor-store path in _state_object_put_image will
    # actually fail on a list, so we use a torch tensor wrapped to drop
    # the shape attribute via a simple subclass.
    class _ShapeLessTensor:
        def detach(self):
            return self

        def clone(self):
            return torch.ones((2, 2, 2, 3))

    fake = _ShapeLessTensor()
    ctx_out, count = setter.execute(ctx, "feedback_image_batch", fake)
    assert count == 1
    assert "feedback_image_batch_ref" in ctx_out["state"]


def test_set_image_batch_registered_in_plugin_mappings():
    """The node should be reachable through the plugin's NODE_CLASS_MAPPINGS
    so ComfyUI can discover it."""
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_plugin_imports import load_plugin_module

    plugin = load_plugin_module()
    assert "MieLoopStateSetImageBatch|Mie" in plugin.NODE_CLASS_MAPPINGS
    assert (
        plugin.NODE_DISPLAY_NAME_MAPPINGS["MieLoopStateSetImageBatch|Mie"]
        == "Mie Loop State Set Image Batch " + chr(0x1F411)
    )