"""
Tests for node execute() methods in loop.py.
Covers: MieLoopStart, MieLoopResume, MieLoopBodyIn, MieLoopBodyOut,
        MieLoopEnd, param/state getters, MieLoopStateSet,
        MieLoopCollectImage, MieLoopFinalizeImages, MieLoopCleanupImages,
        MieImageGrid.
"""

import json
import pytest
import torch
from unittest.mock import patch, MagicMock

from loop import (
    MieLoopStart,
    MieLoopResume,
    MieLoopBodyIn,
    MieLoopBodyOut,
    MieLoopEnd,
    MieLoopParamGetInt,
    MieLoopParamGetFloat,
    MieLoopParamGetString,
    MieLoopParamGetBool,
    MieLoopStateGetInt,
    MieLoopStateGetFloat,
    MieLoopStateGetString,
    MieLoopStateGetBool,
    MieLoopStateSet,
    MieLoopCollectImage,
    MieLoopFinalizeImages,
    MieLoopCleanupImages,
    MieImageGrid,
    RUNTIME_STORE,
    EMPTY_IMAGES,
)


# ---- MieLoopStart ----


class TestMieLoopStart:
    def test_basic_int_list(self):
        node = MieLoopStart()
        ctx, index, count, is_last = node.execute(
            loop_id="scan_steps",
            params_mode="int_list",
            int_list="8,9,10",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        assert ctx["version"] == 3
        assert index == 0
        assert count == 3
        assert is_last is False
        assert ctx["run_id"] != ""
        assert ctx["count"] == 3

    def test_string_list(self):
        node = MieLoopStart()
        ctx, index, count, is_last = node.execute(
            loop_id="test",
            params_mode="string_list",
            int_list="",
            string_list="cat\ndog\ncar",
            json_list="[]",
            initial_state_json="{}",
        )
        assert count == 3
        assert index == 0
        assert is_last is False

    def test_json_list(self):
        node = MieLoopStart()
        ctx, index, count, is_last = node.execute(
            loop_id="test",
            params_mode="json_list",
            int_list="",
            string_list="",
            json_list='[{"steps":8},{"steps":9}]',
            initial_state_json="{}",
        )
        assert count == 2
        assert index == 0
        assert is_last is False

    def test_empty_params(self):
        node = MieLoopStart()
        ctx, index, count, is_last = node.execute(
            loop_id="test",
            params_mode="int_list",
            int_list="",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        assert count == 0
        assert index == 0
        assert is_last is True

    def test_count_1_immediate_last(self):
        node = MieLoopStart()
        ctx, index, count, is_last = node.execute(
            loop_id="test",
            params_mode="int_list",
            int_list="42",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        assert count == 1
        assert index == 0
        assert is_last is True

    def test_initial_state(self):
        node = MieLoopStart()
        ctx, *_ = node.execute(
            loop_id="test",
            params_mode="int_list",
            int_list="1",
            string_list="",
            json_list="[]",
            initial_state_json='{"key":"val"}',
        )
        assert ctx["state"]["key"] == "val"

    def test_meta_json(self):
        node = MieLoopStart()
        ctx, *_ = node.execute(
            loop_id="test",
            params_mode="int_list",
            int_list="1",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
            meta_json='{"body_in_id":"10","body_out_id":"20","end_id":"30"}',
        )
        assert ctx["meta"]["body_in_id"] == "10"
        assert ctx["meta"]["body_out_id"] == "20"
        assert ctx["meta"]["end_id"] == "30"

    def test_resume_valid(self):
        node = MieLoopStart()
        # First create a ctx via normal start
        ctx0, _, _, _ = node.execute(
            loop_id="scan_steps",
            params_mode="int_list",
            int_list="8,9,10",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        ctx0["index"] = 1
        resume_json = json.dumps(ctx0)
        ctx, index, count, is_last = node.execute(
            loop_id="scan_steps",
            params_mode="int_list",
            int_list="",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
            resume_loop_ctx=resume_json,
        )
        assert index == 1
        assert count == 3

    def test_resume_wrong_loop_id(self):
        node = MieLoopStart()
        ctx0, *_ = node.execute(
            loop_id="scan_steps",
            params_mode="int_list",
            int_list="1,2",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        resume_json = json.dumps(ctx0)
        with pytest.raises(ValueError, match="loop_id"):
            node.execute(
                loop_id="wrong_id",
                params_mode="int_list",
                int_list="",
                string_list="",
                json_list="[]",
                initial_state_json="{}",
                resume_loop_ctx=resume_json,
            )

    def test_resume_out_of_range(self):
        node = MieLoopStart()
        ctx0, *_ = node.execute(
            loop_id="scan_steps",
            params_mode="int_list",
            int_list="1,2",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        ctx0["index"] = 2  # >= count
        resume_json = json.dumps(ctx0)
        with pytest.raises(ValueError, match="out of range"):
            node.execute(
                loop_id="scan_steps",
                params_mode="int_list",
                int_list="",
                string_list="",
                json_list="[]",
                initial_state_json="{}",
                resume_loop_ctx=resume_json,
            )


# ---- MieLoopResume ----


class TestMieLoopResume:
    def test_valid(self, sample_loop_ctx):
        node = MieLoopResume()
        ctx_json = json.dumps(sample_loop_ctx)
        result = node.execute(loop_ctx_json=ctx_json)
        assert len(result) == 1
        assert result[0]["run_id"] == "test_run_123"

    def test_invalid_json(self):
        node = MieLoopResume()
        with pytest.raises(ValueError, match="invalid JSON"):
            node.execute(loop_ctx_json="not json")


# ---- MieLoopBodyIn ----


class TestMieLoopBodyIn:
    def test_valid(self, sample_loop_ctx):
        node = MieLoopBodyIn()
        ctx, anchor = node.execute(loop_ctx=sample_loop_ctx)
        assert ctx["run_id"] == "test_run_123"
        assert anchor is None

    def test_with_anchor(self, sample_loop_ctx):
        node = MieLoopBodyIn()
        ctx, anchor = node.execute(loop_ctx=sample_loop_ctx, anchor="my_anchor")
        assert anchor == "my_anchor"

    def test_records_body_in_id(self, sample_loop_ctx):
        node = MieLoopBodyIn()
        ctx, _ = node.execute(loop_ctx=sample_loop_ctx, unique_id=42)
        assert ctx["meta"]["body_in_id"] == "42"


# ---- MieLoopBodyOut ----


class TestMieLoopBodyOut:
    def test_basic(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        result = node.execute(loop_ctx=sample_loop_ctx)
        assert result[0]["run_id"] == "test_run_123"
        # result = (ctx, value_image, value_string, state_json, any1..any5)

    def test_stores_value_any(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        ctx = node.execute(loop_ctx=sample_loop_ctx, value_any_1="hello")[0]
        assert ctx["value_any"][0] == "hello"

    def test_empty_image(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        result = node.execute(loop_ctx=sample_loop_ctx)
        # value_image (index 1) should be EMPTY_IMAGE when None
        assert result[1].shape == EMPTY_IMAGES.shape or result[1].shape[0] == 1

    def test_state_json(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        result = node.execute(loop_ctx=sample_loop_ctx, state_json='{"k":"v"}')
        parsed = json.loads(result[3])
        assert parsed["k"] == "v"


# ---- MieLoopEnd ----


class TestMieLoopEnd:
    def _make_last_ctx(self, sample_loop_ctx):
        """Modify ctx so index is at the last position (done=True)."""
        ctx = sample_loop_ctx.copy()
        ctx["index"] = 2
        ctx["count"] = 3
        ctx["is_last"] = True
        return ctx

    def test_not_done_no_body_ids_raises(self, sample_loop_ctx):
        """index=0, count=3, dynprompt provided but body_in_id missing → raises."""
        node = MieLoopEnd()
        sample_loop_ctx["meta"]["body_in_id"] = None
        with pytest.raises(ValueError, match="body_in_id/body_out_id"):
            node.execute(
                loop_ctx=sample_loop_ctx,
                state_json="{}",
                dynprompt={"10": {"class_type": "X"}},
            )

    def test_last_iteration_done(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        _, done, _, _ = node.execute(loop_ctx=ctx, state_json="{}")
        assert done is True

    def test_count_1_immediate_done(self):
        node = MieLoopStart()
        start_ctx, *_ = node.execute(
            loop_id="test",
            params_mode="int_list",
            int_list="5",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
        )
        end_node = MieLoopEnd()
        _, done, _, _ = end_node.execute(loop_ctx=start_ctx, state_json="{}")
        assert done is True

    def test_state_update(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        result_ctx, *_ = node.execute(loop_ctx=ctx, state_json='{"accumulated":5}')
        assert result_ctx["state"]["accumulated"] == 5

    def test_value_any_stored(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        result_ctx, *_ = node.execute(
            loop_ctx=ctx, state_json="{}", value_any_1="hello"
        )
        assert result_ctx["value_any"][0] == "hello"

    def test_no_dynprompt_not_done_raises(self, sample_loop_ctx):
        # index=0, count=3 → not done, dynprompt=None → raises
        node = MieLoopEnd()
        with pytest.raises(ValueError, match="dynprompt"):
            node.execute(loop_ctx=sample_loop_ctx, state_json="{}")

    def test_done_cleanup(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        # Ensure runtime meta exists
        from loop import _ensure_runtime_meta

        _ensure_runtime_meta(ctx["run_id"])
        node = MieLoopEnd()
        node.execute(loop_ctx=ctx, state_json="{}")
        # After done, run_id should be cleaned from RUNTIME_STORE["meta"]
        assert ctx["run_id"] not in RUNTIME_STORE["meta"]

    def test_done_returns_grid(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        _, done, final_images, final_grid = node.execute(loop_ctx=ctx, state_json="{}")
        assert done is True
        # No collected images → EMPTY_IMAGES for both
        assert final_images.shape[0] == 0
        assert final_grid.shape[0] == 0

    def test_records_end_id(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        result_ctx, *_ = node.execute(loop_ctx=ctx, state_json="{}", unique_id="99")
        assert result_ctx["meta"]["end_id"] == "99"


# ---- Param getters ----


class TestParamGetters:
    def test_get_int_existing_key(self, sample_loop_ctx):
        node = MieLoopParamGetInt()
        result = node.execute(loop_ctx=sample_loop_ctx, key="value", default_value=0)
        assert result == (1,)

    def test_get_int_missing_key_default(self, sample_loop_ctx):
        node = MieLoopParamGetInt()
        result = node.execute(
            loop_ctx=sample_loop_ctx, key="missing_key", default_value=10
        )
        assert result == (10,)

    def test_get_float_basic(self, sample_loop_ctx):
        sample_loop_ctx["current_params"]["ratio"] = 0.5
        node = MieLoopParamGetFloat()
        result = node.execute(loop_ctx=sample_loop_ctx, key="ratio", default_value=0.0)
        assert result == (0.5,)

    def test_get_string_basic(self, sample_loop_ctx):
        sample_loop_ctx["current_params"]["name"] = "hello"
        node = MieLoopParamGetString()
        result = node.execute(loop_ctx=sample_loop_ctx, key="name", default_value="")
        assert result == ("hello",)

    def test_get_bool_coerce(self, sample_loop_ctx):
        sample_loop_ctx["current_params"]["flag"] = "true"
        node = MieLoopParamGetBool()
        result = node.execute(loop_ctx=sample_loop_ctx, key="flag", default_value=False)
        assert result == (True,)


# ---- State getters ----


class TestStateGetters:
    def test_get_int_basic(self, sample_loop_ctx):
        sample_loop_ctx["state"]["count"] = 5
        node = MieLoopStateGetInt()
        result = node.execute(loop_ctx=sample_loop_ctx, key="count", default_value=0)
        assert result == (5,)


# ---- MieLoopStateSet ----


class TestMieLoopStateSet:
    def test_single_key(self):
        node = MieLoopStateSet()
        result = node.execute(
            base_state_json="{}",
            key1="name",
            value1_json='"alice"',
        )
        parsed = json.loads(result[0])
        assert parsed["name"] == "alice"

    def test_multiple_keys(self):
        node = MieLoopStateSet()
        result = node.execute(
            base_state_json="{}",
            key1="a",
            value1_json="1",
            key2="b",
            value2_json='"x"',
            key3="c",
            value3_json="true",
        )
        parsed = json.loads(result[0])
        assert parsed["a"] == 1
        assert parsed["b"] == "x"
        assert parsed["c"] is True

    def test_empty_key_skipped(self):
        node = MieLoopStateSet()
        result = node.execute(
            base_state_json='{"existing":1}',
            key1="",
            value1_json='"ignored"',
        )
        parsed = json.loads(result[0])
        assert "existing" in parsed
        assert "name" not in parsed

    def test_json_value(self):
        node = MieLoopStateSet()
        result = node.execute(
            base_state_json="{}",
            key1="data",
            value1_json='{"nested":true}',
        )
        parsed = json.loads(result[0])
        assert parsed["data"] == {"nested": True}


# ---- MieLoopCollectImage ----


class TestMieLoopCollectImage:
    def test_creates_ref(self, sample_loop_ctx):
        node = MieLoopCollectImage()
        img = torch.rand(1, 64, 64, 3)
        (ctx,) = node.execute(loop_ctx=sample_loop_ctx, image=img)
        assert ctx["collectors"]["images"]["ref"] is not None
        assert ctx["collectors"]["images"]["count"] == 1

    def test_appends(self, sample_loop_ctx):
        node = MieLoopCollectImage()
        img = torch.rand(1, 64, 64, 3)
        (ctx1,) = node.execute(loop_ctx=sample_loop_ctx, image=img)
        (ctx2,) = node.execute(loop_ctx=ctx1, image=img)
        assert ctx2["collectors"]["images"]["count"] == 2


# ---- MieLoopFinalizeImages ----


class TestMieLoopFinalizeImages:
    def test_done_false_returns_execution_blocker(self, sample_loop_ctx):
        """When done=False, FinalizeImages returns ExecutionBlocker to prevent propagation."""
        node = MieLoopFinalizeImages()
        result = node.execute(loop_ctx=sample_loop_ctx, done=False)
        # Should return ExecutionBlocker (from comfy_execution.graph_utils),
        # which is mocked as MagicMock in test conftest — verify it's not a tensor
        assert not isinstance(result[0], torch.Tensor)
        # Verify it has the ExecutionBlocker .message attribute
        assert hasattr(result[0], "message")


# ---- MieLoopCleanupImages ----


class TestMieLoopCleanupImages:
    def test_removes_images(self, sample_loop_ctx):
        from loop import _ensure_runtime_meta

        _ensure_runtime_meta(sample_loop_ctx["run_id"])
        # Simulate collected images
        sample_loop_ctx["collectors"]["images"]["ref"] = "test_ref_abc"
        RUNTIME_STORE["images"]["test_ref_abc"] = [torch.rand(1, 8, 8, 3)]
        node = MieLoopCleanupImages()
        ctx, cleaned = node.execute(loop_ctx=sample_loop_ctx)
        assert cleaned is True
        assert "test_ref_abc" not in RUNTIME_STORE["images"]


# ---- MieImageGrid ----


class TestMieImageGrid:
    def test_execute(self):
        node = MieImageGrid()
        imgs = torch.rand(4, 32, 32, 3)
        result = node.execute(images=imgs, cols=2, pad=0)
        assert result[0].shape[0] == 1  # single grid image
        assert result[0].shape[1] == 64  # 2 cols × 32
        assert result[0].shape[2] == 64  # 2 rows × 32
