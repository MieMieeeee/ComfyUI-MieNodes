import torch

import loop as loop_module


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "feedback_e2e_loop",
        "run_id": "run_feedback_e2e_1",
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


def _make_batch(values):
    frames = [torch.full((2, 2, 3), float(v)) for v in values]
    return torch.stack(frames, dim=0)


def test_feedback_image_round_trip_across_rounds():
    select_frame = loop_module.MieImageSelectFrame()
    set_image = loop_module.MieLoopStateSetImage()
    get_image = loop_module.MieLoopStateGetImage()

    round0_ctx = _make_ctx(index=0, current_params={"value": 1})
    batch0 = _make_batch([0, 1, 2])
    selected0 = select_frame.execute(batch0, 1)[0]
    round0_ctx = set_image.execute(round0_ctx, "feedback_image", selected0)[0]

    round1_ctx = round0_ctx.copy()
    round1_ctx["index"] = 1
    round1_ctx["current_params"] = {"value": 2}

    image1, found1 = get_image.execute(round1_ctx, "feedback_image")
    assert found1 is True
    assert tuple(image1.shape) == (1, 2, 2, 3)
    assert torch.all(image1 == 1)


def test_feedback_image_overwrites_previous_round_value():
    select_frame = loop_module.MieImageSelectFrame()
    set_image = loop_module.MieLoopStateSetImage()
    get_image = loop_module.MieLoopStateGetImage()

    ctx = _make_ctx(index=0)
    ctx = set_image.execute(ctx, "feedback_image", select_frame.execute(_make_batch([1, 2]), 0)[0])[0]
    ref1 = ctx["state"]["feedback_image_ref"]

    ctx["index"] = 1
    ctx["current_params"] = {"value": 2}
    ctx = set_image.execute(ctx, "feedback_image", select_frame.execute(_make_batch([8, 9]), -1)[0])[0]
    ref2 = ctx["state"]["feedback_image_ref"]

    image2, found2 = get_image.execute(ctx, "feedback_image")
    assert found2 is True
    assert torch.all(image2 == 9)
    assert ref1 != ref2
    assert ref1 not in loop_module.RUNTIME_STORE["state_objects"]["image"]
    assert ref2 in loop_module.RUNTIME_STORE["state_objects"]["image"]


def test_feedback_image_survives_loop_end_done_until_explicit_cleanup():
    select_frame = loop_module.MieImageSelectFrame()
    set_image = loop_module.MieLoopStateSetImage()
    get_image = loop_module.MieLoopStateGetImage()
    cleanup_image = loop_module.MieLoopStateCleanupImage()
    end = loop_module.MieLoopEnd()

    ctx = _make_ctx(
        index=2,
        count=3,
        is_last=True,
        params_list=[{"value": 1}, {"value": 2}, {"value": 3}],
        current_params={"value": 3},
    )
    ctx = set_image.execute(ctx, "feedback_image", select_frame.execute(_make_batch([4, 5, 6]), 0)[0])[0]
    ref = ctx["state"]["feedback_image_ref"]

    end_ctx, done = end.execute(ctx, "{}")
    assert done is True

    image_after_end, found_after_end = get_image.execute(end_ctx, "feedback_image")
    assert found_after_end is True
    assert torch.all(image_after_end == 4)
    assert ref in loop_module.RUNTIME_STORE["state_objects"]["image"]

    end_ctx, cleaned = cleanup_image.execute(end_ctx, "feedback_image")
    assert cleaned is True
    assert ref not in loop_module.RUNTIME_STORE["state_objects"]["image"]


def test_feedback_image_get_uses_fallback_before_first_write():
    get_image = loop_module.MieLoopStateGetImage()
    fallback = torch.full((1, 2, 2, 3), 7.0)
    ctx = _make_ctx(index=0)

    image, found = get_image.execute(ctx, "feedback_image", fallback)
    assert found is False
    assert torch.all(image == 7)
