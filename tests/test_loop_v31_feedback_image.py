import torch

import loop as loop_module


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "feedback_loop",
        "run_id": "run_feedback_1",
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


def test_ensure_runtime_meta_initializes_state_object_refs():
    meta = loop_module._ensure_runtime_meta("run_feedback_meta")
    assert "state_object_refs" in meta
    assert meta["state_object_refs"]["image"] == []


def test_image_select_frame_supports_positive_and_negative_index():
    node = loop_module.MieImageSelectFrame()
    images = torch.stack(
        [
            torch.zeros((2, 2, 3)),
            torch.ones((2, 2, 3)),
            torch.full((2, 2, 3), 2.0),
        ],
        dim=0,
    )

    first = node.execute(images, 0)[0]
    last = node.execute(images, -1)[0]

    assert tuple(first.shape) == (1, 2, 2, 3)
    assert tuple(last.shape) == (1, 2, 2, 3)
    assert torch.all(first == 0)
    assert torch.all(last == 2)


def test_state_set_get_and_cleanup_image():
    setter = loop_module.MieLoopStateSetImage()
    getter = loop_module.MieLoopStateGetImage()
    cleaner = loop_module.MieLoopStateCleanupImage()
    ctx = _make_ctx()
    image = torch.ones((1, 2, 2, 3))

    ctx = setter.execute(ctx, "feedback_image", image)[0]
    image_out, found = getter.execute(ctx, "feedback_image")
    assert found is True
    assert tuple(image_out.shape) == (1, 2, 2, 3)
    assert torch.all(image_out == 1)

    ctx, cleaned = cleaner.execute(ctx, "feedback_image")
    assert cleaned is True
    assert "feedback_image_ref" not in ctx["state"]

    fallback = torch.zeros((1, 2, 2, 3))
    image_out, found = getter.execute(ctx, "feedback_image", fallback)
    assert found is False
    assert torch.all(image_out == 0)


def test_state_set_image_overwrites_old_ref():
    setter = loop_module.MieLoopStateSetImage()
    ctx = _make_ctx()
    image1 = torch.ones((1, 2, 2, 3))
    image2 = torch.full((1, 2, 2, 3), 2.0)

    ctx = setter.execute(ctx, "feedback_image", image1)[0]
    ref1 = ctx["state"]["feedback_image_ref"]
    ctx = setter.execute(ctx, "feedback_image", image2)[0]
    ref2 = ctx["state"]["feedback_image_ref"]

    assert ref1 != ref2
    assert ref1 not in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert ref2 in loop_module.RUNTIME_STORE["state_objects"]["image"]


def test_cleanup_runtime_for_run_removes_state_objects():
    setter = loop_module.MieLoopStateSetImage()
    ctx = _make_ctx(run_id="run_cleanup_feedback")
    ctx = setter.execute(ctx, "feedback_image", torch.ones((1, 2, 2, 3)))[0]
    ref = ctx["state"]["feedback_image_ref"]

    loop_module._cleanup_runtime_for_run(ctx["run_id"])

    assert ref not in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert ctx["run_id"] not in loop_module.RUNTIME_STORE["meta"]


def test_prune_runtime_store_removes_orphan_state_objects():
    store = loop_module.RUNTIME_STORE.setdefault("state_objects", {}).setdefault("image", {})
    store["stateimg_orphan"] = torch.ones((1, 1, 1, 3))
    loop_module._prune_runtime_store()
    assert "stateimg_orphan" not in loop_module.RUNTIME_STORE["state_objects"]["image"]
