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
    MieLoopGetIndex,
    MieLoopParamGetInt,
    MieLoopParamGetFloat,
    MieLoopParamGetString,
    MieLoopParamGetBool,
    MieLoopStateGetInt,
    MieLoopStateGetFloat,
    MieLoopStateGetString,
    MieLoopStateGetBool,
    MieLoopStateSet,
    MieLoopStateSetInt,
    MieLoopCollectImage,
    MieLoopFinalizeImages,
    MieLoopCleanupImages,
    MieLoopCollectAudio,
    MieLoopFinalizeAudio,
    MieLoopCleanupAudio,
    MieImageGrid,
    RUNTIME_STORE,
    EMPTY_IMAGES,
)


# ---- MieLoopStart ----


class TestMieLoopStart:
    def test_input_types_put_initial_state_in_optional(self):
        input_types = MieLoopStart.INPUT_TYPES()
        assert "initial_state_json" not in input_types["required"]
        assert "initial_state_json" in input_types["optional"]

    def _execute_start(self, **overrides):
        node = MieLoopStart()
        kwargs = {
            "loop_id": "scan_steps",
            "param_type": "int",
            "param_mode": "list",
            "int_list": "8,9,10",
            "float_list": "",
            "string_list": "",
            "json_list": "[]",
            "int_range_start": 0,
            "int_range_end": 0,
            "int_range_step": 1,
            "float_range_start": 0.0,
            "float_range_end": 0.0,
            "float_range_step": 1.0,
            "initial_state_json": "{}",
        }
        kwargs.update(overrides)
        return node.execute(**kwargs)

    def test_basic_int_list(self):
        ctx, index, count, is_last = self._execute_start()
        assert ctx["version"] == 3
        assert index == 0
        assert count == 3
        assert is_last is False
        assert ctx["run_id"] != ""
        assert ctx["count"] == 3

    def test_int_range(self):
        ctx, index, count, is_last = self._execute_start(
            param_type="int",
            param_mode="range",
            int_list="",
            int_range_start=2,
            int_range_end=6,
            int_range_step=2,
        )
        assert ctx["params_list"] == [{"value": 2}, {"value": 4}]
        assert count == 2
        assert index == 0
        assert is_last is False

    def test_float_list(self):
        ctx, index, count, is_last = self._execute_start(
            param_type="float",
            param_mode="list",
            int_list="",
            float_list="0.5,1.25,2.0",
        )
        assert ctx["params_list"] == [{"value": 0.5}, {"value": 1.25}, {"value": 2.0}]
        assert count == 3
        assert index == 0
        assert is_last is False

    def test_float_range(self):
        ctx, index, count, is_last = self._execute_start(
            param_type="float",
            param_mode="range",
            int_list="",
            float_range_start=0.1,
            float_range_end=0.3,
            float_range_step=0.1,
        )
        assert ctx["params_list"] == [{"value": 0.1}, {"value": 0.2}]
        assert count == 2
        assert index == 0
        assert is_last is False

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
        assert ctx["collectors"] == {
            "image": {"ref": None, "count": 0},
            "text": {"ref": None, "count": 0},
            "json": {"ref": None, "count": 0},
            "audio": {"ref": None, "count": 0},
        }

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
        assert ctx["collectors"] == {
            "image": {"ref": None, "count": 0},
            "text": {"ref": None, "count": 0},
            "json": {"ref": None, "count": 0},
            "audio": {"ref": None, "count": 0},
        }

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
    def test_input_types_put_state_json_in_optional(self):
        input_types = MieLoopBodyOut.INPUT_TYPES()
        assert "state_json" not in input_types["required"]
        assert "state_json" in input_types["optional"]

    def test_basic(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        result = node.execute(loop_ctx=sample_loop_ctx)
        assert result[0]["run_id"] == "test_run_123"
        assert len(result) == 2

    def test_returns_state_json(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        result = node.execute(loop_ctx=sample_loop_ctx, state_json='{"k":"v"}')
        assert result[1] == '{"k": "v"}'

    def test_state_json(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        result = node.execute(loop_ctx=sample_loop_ctx, state_json='{"k":"v"}')
        parsed = json.loads(result[1])
        assert parsed["k"] == "v"

    def test_cloned_round_keeps_original_body_out_id(self, sample_loop_ctx):
        node = MieLoopBodyOut()
        sample_loop_ctx["meta"]["body_out_id"] = "20"
        ctx = node.execute(loop_ctx=sample_loop_ctx, unique_id="48.0.0.20")[0]
        assert ctx["meta"]["body_out_id"] == "20"


# ---- MieLoopEnd ----


class TestMieLoopEnd:
    def test_input_types_put_state_json_in_optional(self):
        input_types = MieLoopEnd.INPUT_TYPES()
        assert "state_json" not in input_types["required"]
        assert "state_json" in input_types["optional"]

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
        _, done = node.execute(loop_ctx=ctx, state_json="{}")
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
        _, done = end_node.execute(loop_ctx=start_ctx, state_json="{}")
        assert done is True

    def test_state_update(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        result_ctx, *_ = node.execute(loop_ctx=ctx, state_json='{"accumulated":5}')
        assert result_ctx["state"]["accumulated"] == 5

    def test_no_dynprompt_not_done_raises(self, sample_loop_ctx):
        # index=0, count=3 → not done, dynprompt=None → raises
        node = MieLoopEnd()
        with pytest.raises(ValueError, match="dynprompt"):
            node.execute(loop_ctx=sample_loop_ctx, state_json="{}")

    def test_done_keeps_runtime_meta_for_finalize(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        # Ensure runtime meta exists
        from loop import _ensure_runtime_meta

        _ensure_runtime_meta(ctx["run_id"])
        node = MieLoopEnd()
        node.execute(loop_ctx=ctx, state_json="{}")
        # After done, runtime meta remains for finalize / cleanup nodes to consume.
        assert ctx["run_id"] in RUNTIME_STORE["meta"]

    def test_done_returns_only_ctx_and_done(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        result = node.execute(loop_ctx=ctx, state_json="{}")
        assert len(result) == 2
        _, done = result
        assert done is True

    def test_records_end_id(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        node = MieLoopEnd()
        result_ctx, *_ = node.execute(loop_ctx=ctx, state_json="{}", unique_id="99")
        assert result_ctx["meta"]["end_id"] == "99"

    def test_cloned_round_keeps_original_end_id(self, sample_loop_ctx):
        ctx = self._make_last_ctx(sample_loop_ctx)
        ctx["meta"]["end_id"] = "30"
        node = MieLoopEnd()
        result_ctx, *_ = node.execute(
            loop_ctx=ctx, state_json="{}", unique_id="48.0.0.30"
        )
        assert result_ctx["meta"]["end_id"] == "30"


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


class TestLoopGetIndex:
    def test_get_index_zero_based(self, sample_loop_ctx):
        node = MieLoopGetIndex()
        result = node.execute(loop_ctx=sample_loop_ctx)
        assert result == (0,)

    def test_get_index_current_round(self, sample_loop_ctx):
        sample_loop_ctx["index"] = 2
        node = MieLoopGetIndex()
        result = node.execute(loop_ctx=sample_loop_ctx)
        assert result == (2,)


# ---- State getters ----


class TestStateGetters:
    def test_get_int_basic(self, sample_loop_ctx):
        sample_loop_ctx["state"]["count"] = 5
        node = MieLoopStateGetInt()
        result = node.execute(loop_ctx=sample_loop_ctx, key="count", default_value=0)
        assert result == (5,)


# ---- MieLoopStateSet ----


class TestMieLoopStateSet:
    def test_input_types_put_base_state_json_in_optional(self):
        input_types = MieLoopStateSet.INPUT_TYPES()
        assert "base_state_json" not in input_types["required"]
        assert "base_state_json" in input_types["optional"]


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


class TestMieLoopStateSetInt:
    def test_sets_int_in_loop_ctx_state(self, sample_loop_ctx):
        node = MieLoopStateSetInt()
        (ctx,) = node.execute(loop_ctx=sample_loop_ctx, key="k", value=42)
        assert ctx["state"]["k"] == 42


# ---- MieLoopCollectImage ----


class TestMieLoopCollectImage:
    def test_creates_ref(self, sample_loop_ctx):
        node = MieLoopCollectImage()
        img = torch.rand(1, 64, 64, 3)
        (ctx,) = node.execute(loop_ctx=sample_loop_ctx, image=img)
        assert ctx["collectors"]["image"]["ref"] is not None
        assert ctx["collectors"]["image"]["count"] == 1

    def test_appends(self, sample_loop_ctx):
        node = MieLoopCollectImage()
        img = torch.rand(1, 64, 64, 3)
        (ctx1,) = node.execute(loop_ctx=sample_loop_ctx, image=img)
        (ctx2,) = node.execute(loop_ctx=ctx1, image=img)
        assert ctx2["collectors"]["image"]["count"] == 2


# ---- MieLoopFinalizeImages ----


class TestMieLoopFinalizeImages:
    def test_done_false_returns_empty_images(self, sample_loop_ctx):
        node = MieLoopFinalizeImages()
        result = node.execute(loop_ctx=sample_loop_ctx, done=False)
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape[0] == 0


# ---- MieLoopCleanupImages ----


class TestMieLoopCleanupImages:
    def test_removes_images(self, sample_loop_ctx):
        from loop import _ensure_runtime_meta

        _ensure_runtime_meta(sample_loop_ctx["run_id"])
        # Simulate collected images
        sample_loop_ctx["collectors"]["image"]["ref"] = "test_ref_abc"
        RUNTIME_STORE["collectors"]["image"]["test_ref_abc"] = [torch.rand(1, 8, 8, 3)]
        node = MieLoopCleanupImages()
        ctx, cleaned = node.execute(loop_ctx=sample_loop_ctx)
        assert cleaned is True
        assert "test_ref_abc" not in RUNTIME_STORE["collectors"]["image"]
        assert ctx["collectors"]["image"] == {"ref": None, "count": 0}


# ---- MieImageGrid ----


class TestMieImageGrid:
    def test_execute(self):
        node = MieImageGrid()
        imgs = torch.rand(4, 32, 32, 3)
        result = node.execute(images=imgs, cols=2, pad=0)
        assert result[0].shape[0] == 1  # single grid image
        assert result[0].shape[1] == 64  # 2 cols × 32
        assert result[0].shape[2] == 64  # 2 rows × 32
