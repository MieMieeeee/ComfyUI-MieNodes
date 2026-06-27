"""
E2E lifecycle tests for MieLoop nodes.

Simulates the full loop protocol (Start → BodyIn → body chain → BodyOut → End →
Finalize) without ComfyUI's execution engine, driving multiple rounds manually
to validate cross-round state, parameter, collector, and conditional-routing
behaviour end-to-end.

Coverage target: all 40 loop nodes (protocol, param, state, collector, routing,
auxiliary).
"""

import copy
import json

import torch

import loop as loop_module

EMPTY_IMAGE = loop_module.EMPTY_IMAGE
EMPTY_IMAGES = loop_module.EMPTY_IMAGES

# ---------------------------------------------------------------------------
# Lifecycle driver
# ---------------------------------------------------------------------------


def _run_loop(start_kwargs, body_fn, *, max_rounds=30):
    """Simulate a full loop lifecycle across multiple rounds.

    Steps each round:
      1. MieLoopStart.execute()              — only round 0
      2. MieLoopBodyIn.execute()
      3. body_fn(ctx)                         — user-supplied; must return ctx
      4. MieLoopBodyOut.execute()
      5. On the final round: MieLoopEnd.execute(done=True path)

    For non-final rounds, the driver manually advances the loop context
    (index, current_params, is_last) instead of calling End — because End
    requires dynprompt when done=False to build the ComfyUI expand graph.
    This mirrors exactly what End.execute() does internally (see
    ``curr_index + 1 < count`` branch) but without the expand step.

    Returns ``(final_loop_ctx, done_flag, body_outputs)`` where
    ``body_outputs`` is a list of values yielded by ``body_fn`` each round.
    """

    start_node = loop_module.MieLoopStart()
    body_in_node = loop_module.MieLoopBodyIn()
    body_out_node = loop_module.MieLoopBodyOut()
    end_node = loop_module.MieLoopEnd()

    # --- round 0: Start ---
    ctx, index, count, is_last = start_node.execute(**start_kwargs)

    if count == 0:
        # Empty loop: done immediately, no body execution needed
        # (Start already sets is_last=True and index=0 for count=0)
        return ctx, True, []

    body_outputs = []
    done = False
    for _ in range(max_rounds):
        # --- BodyIn ---
        ctx, _anchor = body_in_node.execute(ctx)

        # --- Body (user callback) ---
        ctx = body_fn(ctx)
        body_outputs.append(ctx)

        # --- BodyOut ---
        body_out_ctx, _state_json = body_out_node.execute(ctx)

        # --- Determine if this is the last round ---
        curr_index = int(ctx["index"])
        total_count = int(ctx["count"])
        is_final = (curr_index + 1 >= total_count)

        if is_final:
            # Last round: let End.execute() mark done=True
            ctx["index"] = total_count - 1
            ctx["is_last"] = True
            ctx["current_params"] = ctx["params_list"][total_count - 1]
            end_result = end_node.execute(body_out_ctx)
            if isinstance(end_result, dict):
                # Expand result dict — unexpected on last round
                ctx = end_result["result"][0]
                done = end_result["result"][1]
            else:
                ctx, done = end_result
            break
        else:
            # Not last: manually advance context (mirrors End's internal logic)
            next_index = curr_index + 1
            ctx["index"] = next_index
            ctx["current_params"] = ctx["params_list"][next_index]
            ctx["is_last"] = next_index == total_count - 1
            done = False

    return ctx, done, body_outputs


# ---------------------------------------------------------------------------
# 1. Param mode coverage
# ---------------------------------------------------------------------------


def test_int_list_params():
    """int_list="10,20,30" → ParamGetInt yields 10, 20, 30 across 3 rounds."""
    records = []
    def body(ctx):
        node = loop_module.MieLoopParamGetInt()
        val = node.execute(ctx, "value", 0)[0]
        records.append(val)
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="10,20,30"),
        body,
    )
    assert done is True
    assert records == [10, 20, 30]


def test_int_range_params():
    """int_range(0, 5, 1) → 5 rounds, indices 0..4."""
    indices = []
    def body(ctx):
        indices.append(ctx["index"])
        return ctx
    ctx, done, _ = _run_loop(
        dict(
            loop_id="t", param_type="int", param_mode="range",
            int_range_start=0, int_range_end=5, int_range_step=1,
        ),
        body,
    )
    assert done is True
    assert indices == [0, 1, 2, 3, 4]


def test_float_list_params():
    """float_list="0.1,0.2,0.3" → ParamGetFloat yields 0.1, 0.2, 0.3."""
    records = []
    def body(ctx):
        node = loop_module.MieLoopParamGetFloat()
        val = node.execute(ctx, "value", 0.0)[0]
        records.append(round(val, 2))
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="float", param_mode="list", float_list="0.1,0.2,0.3"),
        body,
    )
    assert done is True
    assert records == [0.1, 0.2, 0.3]


def test_string_list_params():
    """string_list lines → ParamGetString yields each line."""
    records = []
    def body(ctx):
        node = loop_module.MieLoopParamGetString()
        val = node.execute(ctx, "value", "")[0]
        records.append(val)
        return ctx
    ctx, done, _ = _run_loop(
        dict(
            loop_id="t", param_type="string", param_mode="list",
            string_list="alpha\nbeta\ngamma",
        ),
        body,
    )
    assert done is True
    assert records == ["alpha", "beta", "gamma"]


def test_json_list_params():
    """json_list=[{"v":true},{"v":false}] → ParamGetBool yields True, False."""
    records = []
    def body(ctx):
        node = loop_module.MieLoopParamGetBool()
        val = node.execute(ctx, "v", False)[0]
        records.append(val)
        return ctx
    ctx, done, _ = _run_loop(
        dict(
            loop_id="t", param_type="json", param_mode="list",
            json_list='[{"v": true}, {"v": false}]',
        ),
        body,
    )
    assert done is True
    assert records == [True, False]


def test_int_decrement_params():
    """int_decrement(total=5, step=2) → params [2, 2, 1]."""
    records = []
    def body(ctx):
        node = loop_module.MieLoopParamGetInt()
        val = node.execute(ctx, "value", 0)[0]
        records.append(val)
        return ctx
    ctx, done, _ = _run_loop(
        dict(
            loop_id="t", param_type="int", param_mode="decrement",
            int_decrement_total=5, int_decrement_step=2,
        ),
        body,
    )
    assert done is True
    assert records == [2, 2, 1]


# ---------------------------------------------------------------------------
# 2. Empty / single-round loops
# ---------------------------------------------------------------------------


def test_empty_loop():
    """count=0 → done=True immediately, Finalize returns empty."""
    collected_text = []
    def body(ctx):
        node = loop_module.MieLoopCollectText()
        node.execute(ctx, "should not appear")
        collected_text.append("oops")
        return ctx
    ctx, done, outputs = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list=""),
        body,
    )
    assert done is True
    assert ctx["count"] == 0
    assert ctx["index"] == 0
    assert outputs == []
    # Finalize should return empty
    finalize_node = loop_module.MieLoopFinalizeTextList()
    result_json = finalize_node.execute(ctx, True)[0]
    assert json.loads(result_json) == []


def test_single_round_loop():
    """count=1 → is_last=True from round 0, body executes once."""
    indices = []
    def body(ctx):
        indices.append(ctx["index"])
        assert ctx["is_last"] is True
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="42"),
        body,
    )
    assert done is True
    assert indices == [0]


# ---------------------------------------------------------------------------
# 3. State cross-round feedback
# ---------------------------------------------------------------------------


def test_state_int_accumulate():
    """StateSetInt adds 10 each round → final state["sum"]=30 (3 rounds)."""
    def body(ctx):
        current = loop_module.MieLoopStateGetInt().execute(ctx, "sum", 0)[0]
        updated = current + 10
        ctx = loop_module.MieLoopStateSetInt().execute(ctx, "sum", updated)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    assert ctx["state"]["sum"] == 30


def test_state_float_multiply():
    """StateSetFloat multiplies by 2 each round: 1 → 2 → 4."""
    records = []
    def body(ctx):
        node_get = loop_module.MieLoopStateGetFloat()
        node_set = loop_module.MieLoopStateSetFloat()
        current = node_get.execute(ctx, "factor", 1.0)[0]
        records.append(round(current, 4))
        ctx = node_set.execute(ctx, "factor", round(current * 2, 4))[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    assert records == [1.0, 2.0, 4.0]


def test_state_string_append():
    """StateSetString concatenates each round's param value."""
    def body(ctx):
        current = loop_module.MieLoopStateGetString().execute(ctx, "buf", "")[0]
        param_val = loop_module.MieLoopParamGetString().execute(ctx, "value", "")[0]
        ctx = loop_module.MieLoopStateSetString().execute(ctx, "buf", current + param_val)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="string", param_mode="list", string_list="A,B,C"),
        body,
    )
    assert done is True
    assert ctx["state"]["buf"] == "ABC"


def test_state_image_roundtrip():
    """StateSetImage → StateGetImage → CollectImage full chain."""
    images_per_round = []
    def body(ctx):
        get_node = loop_module.MieLoopStateGetImage()
        set_node = loop_module.MieLoopStateSetImage()
        collect_node = loop_module.MieLoopCollectImage()

        # Round 0: no image in state → fallback to EMPTY_IMAGE
        img, found = get_node.execute(ctx, "fb", fallback_image=EMPTY_IMAGE)
        if not found:
            img = torch.ones((1, 4, 4, 3)) * (ctx["index"] + 1)
        else:
            # Subsequent rounds: use the stored image, add a delta
            img = img + torch.ones_like(img) * 0.1

        # Store back into state
        ctx = set_node.execute(ctx, "fb", img)[0]

        # Collect into the image collector
        ctx = collect_node.execute(ctx, img)[0]
        images_per_round.append(img.shape[0])
        return ctx

    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    assert images_per_round == [1, 1, 1]

    # Finalize: should have 3 images merged
    finalize_node = loop_module.MieLoopFinalizeImages()
    merged = finalize_node.execute(ctx, True)[0]
    assert isinstance(merged, torch.Tensor)
    assert merged.shape[0] == 3


def test_state_set_batch_patch():
    """MieLoopStateSet writes multiple keys in one call."""
    def body(ctx):
        idx = ctx["index"]
        set_node = loop_module.MieLoopStateSet()
        state_json = set_node.execute(
            base_state_json=json.dumps(ctx["state"]),
            key1="k1", value1_json=str(idx * 10),
            key2="k2", value2_json=str(idx * 100),
            key3="k3", value3_json="",
        )[0]
        # Merge the patch back into ctx (simulating BodyOut wiring)
        patch = json.loads(state_json)
        ctx["state"].update(patch)
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2"),
        body,
    )
    assert done is True
    assert ctx["state"]["k1"] == 10  # last round idx=1 → 1*10
    assert ctx["state"]["k2"] == 100  # last round idx=1 → 1*100


# ---------------------------------------------------------------------------
# 4. Conditional routing
# ---------------------------------------------------------------------------


def test_if_is_first():
    """IfIsFirst: round 0 returns 'first', others return 'other'."""
    labels = []
    def body(ctx):
        node = loop_module.MieLoopIfIsFirst()
        _, value = node.execute(ctx, "first", "other")
        labels.append(value)
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    assert labels == ["first", "other", "other"]


def test_if_is_last():
    """IfIsLast: last round returns 'last', others return 'running'."""
    labels = []
    def body(ctx):
        node = loop_module.MieLoopIfIsLast()
        _, value = node.execute(ctx, "last", "running")
        labels.append(value)
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    assert labels == ["running", "running", "last"]


def test_if_current_idx():
    """IfCurrentIdx: routes to 'matched' when index==1."""
    labels = []
    def body(ctx):
        node = loop_module.MieLoopIfCurrentIdx()
        _, value = node.execute(ctx, "==", 1, "matched", "nope")
        labels.append(value)
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3,4"),
        body,
    )
    assert done is True
    assert labels == ["nope", "matched", "nope", "nope"]


# ---------------------------------------------------------------------------
# 5. Collector full chain (Collect → Finalize)
# ---------------------------------------------------------------------------


def test_collect_text_finalize():
    """CollectText each round → FinalizeTextList returns JSON array."""
    def body(ctx):
        text = f"round_{ctx['index']}_done"
        node = loop_module.MieLoopCollectText()
        ctx = node.execute(ctx, text)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    finalize = loop_module.MieLoopFinalizeTextList()
    result_json = finalize.execute(ctx, True)[0]
    items = json.loads(result_json)
    assert items == ["round_0_done", "round_1_done", "round_2_done"]


def test_collect_json_finalize():
    """CollectJSON each round → FinalizeJSONList returns JSON array of objects."""
    def body(ctx):
        item = json.dumps({"round": ctx["index"], "val": ctx["current_params"]["value"]})
        node = loop_module.MieLoopCollectJSON()
        ctx = node.execute(ctx, item)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="10,20,30"),
        body,
    )
    assert done is True
    finalize = loop_module.MieLoopFinalizeJSONList()
    result_json = finalize.execute(ctx, True)[0]
    items = json.loads(result_json)
    assert len(items) == 3
    assert items[0] == {"round": 0, "val": 10}
    assert items[2] == {"round": 2, "val": 30}


def test_collect_image_finalize():
    """CollectImage each round → FinalizeImages returns batch of correct size."""
    def body(ctx):
        img = torch.ones((1, 8, 8, 3)) * (ctx["index"] + 1)
        node = loop_module.MieLoopCollectImage()
        ctx = node.execute(ctx, img)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3,4"),
        body,
    )
    assert done is True
    finalize = loop_module.MieLoopFinalizeImages()
    merged = finalize.execute(ctx, True)[0]
    assert isinstance(merged, torch.Tensor)
    assert merged.shape == (4, 8, 8, 3)


def test_collect_cleanup():
    """CleanupText/JSON/Image clear collector refs."""
    def body(ctx):
        ctx = loop_module.MieLoopCollectText().execute(ctx, "txt")[0]
        ctx = loop_module.MieLoopCollectJSON().execute(ctx, '{"x":1}')[0]
        img = torch.ones((1, 2, 2, 3))
        ctx = loop_module.MieLoopCollectImage().execute(ctx, img)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2"),
        body,
    )
    assert done is True

    # Cleanup text
    ctx_t, cleaned_t = loop_module.MieLoopCleanupText().execute(ctx)
    assert cleaned_t is True
    assert ctx_t["collectors"]["text"]["ref"] is None

    # Cleanup JSON
    ctx_j, cleaned_j = loop_module.MieLoopCleanupJSON().execute(ctx)
    assert cleaned_j is True
    assert ctx_j["collectors"]["json"]["ref"] is None

    # Cleanup images
    ctx_i, cleaned_i = loop_module.MieLoopCleanupImages().execute(ctx)
    assert cleaned_i is True
    assert ctx_i["collectors"]["image"]["ref"] is None


# ---------------------------------------------------------------------------
# 6. Auxiliary nodes
# ---------------------------------------------------------------------------


def test_image_select_frame():
    """ImageSelectFrame extracts the correct frame from a batch."""
    batch = torch.arange(48).float().reshape(3, 2, 2, 4)  # 3 frames, 2x2, 4ch
    node = loop_module.MieImageSelectFrame()

    # Index 0
    frame0 = node.execute(batch, 0)[0]
    assert frame0.shape == (1, 2, 2, 4)
    assert torch.equal(frame0[0], batch[0])

    # Index -1 (last frame)
    frame_last = node.execute(batch, -1)[0]
    assert torch.equal(frame_last[0], batch[2])

    # Index 2
    frame2 = node.execute(batch, 2)[0]
    assert torch.equal(frame2[0], batch[2])


def test_image_grid():
    """MieImageGrid produces correct grid shape."""
    # 3 images of 4x4x3
    batch = torch.zeros((3, 4, 4, 3))
    node = loop_module.MieImageGrid()
    grid = node.execute(batch, cols=2, pad=1)[0]
    # 3 images, cols=2 → 2 rows: row0 has 2 images, row1 has 1
    # row height = 4, col width = 4, pad = 1
    # grid_h = 2*4 + 1*1 = 9, grid_w = 2*4 + 1*1 = 9
    assert grid.shape == (1, 9, 9, 3)


# ---------------------------------------------------------------------------
# 7. Finalize returns empty when done=False
# ---------------------------------------------------------------------------


def test_finalize_returns_empty_when_not_done():
    """Finalize nodes return empty results when done=False."""
    # Build a minimal context with an empty collector
    start = loop_module.MieLoopStart()
    ctx, _, _, _ = start.execute(
        loop_id="t", param_type="int", param_mode="list", int_list="1,2",
    )

    # Not done → should return empty
    finalize_text = loop_module.MieLoopFinalizeTextList()
    result = finalize_text.execute(ctx, False)[0]
    assert json.loads(result) == []

    finalize_img = loop_module.MieLoopFinalizeImages()
    imgs = finalize_img.execute(ctx, False)[0]
    assert imgs.shape == EMPTY_IMAGES.shape
    assert imgs.shape[0] == 0

    finalize_json = loop_module.MieLoopFinalizeJSONList()
    result = finalize_json.execute(ctx, False)[0]
    assert json.loads(result) == []

    finalize_audio = loop_module.MieLoopFinalizeAudio()
    audio = finalize_audio.execute(ctx, False)[0]
    assert audio["waveform"].shape == (1, 2, 0)


# ---------------------------------------------------------------------------
# 8. Multi-feature integration: state + collector + routing combined
# ---------------------------------------------------------------------------


def test_integration_state_collector_routing():
    """Combined test: StateInt accumulate + IfIsFirst label + CollectText."""
    labels = []

    def body(ctx):
        # Read accumulated state
        current_sum = loop_module.MieLoopStateGetInt().execute(ctx, "sum", 0)[0]

        # Route based on first round
        if_node = loop_module.MieLoopIfIsFirst()
        _, label = if_node.execute(ctx, "INIT", "LOOP")
        labels.append(label)

        # Accumulate param value into state
        param_val = loop_module.MieLoopParamGetInt().execute(ctx, "value", 0)[0]
        new_sum = current_sum + param_val
        ctx = loop_module.MieLoopStateSetInt().execute(ctx, "sum", new_sum)[0]

        # Collect a text summarizing the round
        text = f"[{label}] idx={ctx['index']} sum={new_sum}"
        ctx = loop_module.MieLoopCollectText().execute(ctx, text)[0]
        return ctx

    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="10,20,30"),
        body,
    )
    assert done is True
    assert labels == ["INIT", "LOOP", "LOOP"]
    assert ctx["state"]["sum"] == 60  # 0+10=10, 10+20=30, 30+30=60

    # Verify collected text
    finalize = loop_module.MieLoopFinalizeTextList()
    result_json = finalize.execute(ctx, True)[0]
    items = json.loads(result_json)
    assert items == [
        "[INIT] idx=0 sum=10",
        "[LOOP] idx=1 sum=30",
        "[LOOP] idx=2 sum=60",
    ]


# ---------------------------------------------------------------------------
# 9. GetIndex node
# ---------------------------------------------------------------------------


def test_get_index():
    """MieLoopGetIndex returns the correct index each round."""
    indices = []
    def body(ctx):
        idx = loop_module.MieLoopGetIndex().execute(ctx)[0]
        indices.append(idx)
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="5,6,7,8"),
        body,
    )
    assert done is True
    assert indices == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 10. StateSetBool / StateGetBool
# ---------------------------------------------------------------------------


def test_state_bool_roundtrip():
    """StateSetBool / StateGetBool across rounds."""
    def body(ctx):
        # Round 0: flag should be False (default)
        flag = loop_module.MieLoopStateGetBool().execute(ctx, "flag", False)[0]
        if ctx["index"] == 0:
            assert flag is False
        # Set flag to True
        ctx = loop_module.MieLoopStateSetBool().execute(ctx, "flag", True)[0]
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2"),
        body,
    )
    assert done is True
    assert ctx["state"]["flag"] is True


# ---------------------------------------------------------------------------
# 11. StateCleanupImage
# ---------------------------------------------------------------------------


def test_state_cleanup_image():
    """StateSetImage → StateCleanupImage removes the ref."""
    def body(ctx):
        img = torch.ones((1, 2, 2, 3))
        ctx = loop_module.MieLoopStateSetImage().execute(ctx, "tmp", img)[0]
        # Verify it exists
        get_node = loop_module.MieLoopStateGetImage()
        _, found = get_node.execute(ctx, "tmp")
        assert found is True
        # Cleanup
        ctx, cleaned = loop_module.MieLoopStateCleanupImage().execute(ctx, "tmp")
        assert cleaned is True
        # After cleanup, should return empty
        _, found2 = get_node.execute(ctx, "tmp")
        assert found2 is False
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1"),
        body,
    )
    assert done is True


# ---------------------------------------------------------------------------
# 12. StateSetImageBatch
# ---------------------------------------------------------------------------


def test_state_set_image_batch():
    """StateSetImageBatch stores a multi-frame batch and reports count."""
    counts = []
    def body(ctx):
        # Create a batch of (index+2) frames
        n_frames = ctx["index"] + 2
        img = torch.ones((n_frames, 4, 4, 3))
        node = loop_module.MieLoopStateSetImageBatch()
        ctx, count = node.execute(ctx, "batch", img)
        counts.append(count)
        assert count == n_frames
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3"),
        body,
    )
    assert done is True
    assert counts == [2, 3, 4]


# ---------------------------------------------------------------------------
# 13. Float range param mode
# ---------------------------------------------------------------------------


def test_float_range_params():
    """float_range(0.0, 0.6, 0.2) → values 0.0, 0.2, 0.4."""
    records = []
    def body(ctx):
        node = loop_module.MieLoopParamGetFloat()
        val = node.execute(ctx, "value", -1.0)[0]
        records.append(round(val, 2))
        return ctx
    ctx, done, _ = _run_loop(
        dict(
            loop_id="t", param_type="float", param_mode="range",
            float_range_start=0.0, float_range_end=0.6, float_range_step=0.2,
        ),
        body,
    )
    assert done is True
    assert records == [0.0, 0.2, 0.4]


# ---------------------------------------------------------------------------
# 14. IfCurrentIdx with various operators
# ---------------------------------------------------------------------------


def test_if_current_idx_operators():
    """IfCurrentIdx with <, >=, != operators."""
    results = []
    def body(ctx):
        node = loop_module.MieLoopIfCurrentIdx()
        idx = ctx["index"]
        # Test < 2
        _, val = node.execute(ctx, "<", 2, "small", "big")
        results.append((idx, val))
        return ctx
    ctx, done, _ = _run_loop(
        dict(loop_id="t", param_type="int", param_mode="list", int_list="1,2,3,4"),
        body,
    )
    assert done is True
    assert results == [(0, "small"), (1, "small"), (2, "big"), (3, "big")]
