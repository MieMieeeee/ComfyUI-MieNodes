"""
Unit tests for pure helper functions in loop.py.

Covers:
  - Parsing / validation (lines 27-122)
  - Type coercion / classification (lines 125-259)
"""

import pytest

# conftest.py registers the module as both mie_pkg.loop and plain loop
from loop import (
    _require_dict,
    _parse_json_object,
    _parse_json_array,
    _parse_params_list,
    _validate_loop_ctx,
    _coerce_bool,
    _value_from_json_string,
    _base_class_type,
    _is_protocol_class,
    _is_collector_class,
    _is_excluded_output_class,
    is_link,
    _get_inputs,
    _get_class_type,
)


# ── _require_dict ──────────────────────────────────────────────────────────


def test_require_dict_valid():
    assert _require_dict({"a": 1}, "x") == {"a": 1}


def test_require_dict_invalid_string():
    with pytest.raises(ValueError, match="must be an object"):
        _require_dict("hello", "x")


def test_require_dict_invalid_list():
    with pytest.raises(ValueError, match="must be an object"):
        _require_dict([1, 2], "x")


def test_require_dict_invalid_none():
    with pytest.raises(ValueError, match="must be an object"):
        _require_dict(None, "x")


# ── _parse_json_object ─────────────────────────────────────────────────────


def test_parse_json_object_valid():
    assert _parse_json_object('{"a":1}', "test") == {"a": 1}


def test_parse_json_object_invalid():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_json_object("not json", "test")


def test_parse_json_object_not_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        _parse_json_object("[1,2]", "test")


# ── _parse_json_array ──────────────────────────────────────────────────────


def test_parse_json_array_valid():
    assert _parse_json_array("[1,2]", "test") == [1, 2]


def test_parse_json_array_invalid():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_json_array("not json", "test")


def test_parse_json_array_not_array():
    with pytest.raises(ValueError, match="must be a JSON array"):
        _parse_json_array('{"a":1}', "test")


# ── _parse_params_list ─────────────────────────────────────────────────────


def test_parse_params_list_int_list_basic():
    result = _parse_params_list("int_list", "8,9,10", "", "[]")
    assert result == [{"value": 8}, {"value": 9}, {"value": 10}]


def test_parse_params_list_int_list_chinese_comma():
    result = _parse_params_list("int_list", "8\uff0c9\uff0c10", "", "[]")
    assert result == [{"value": 8}, {"value": 9}, {"value": 10}]


def test_parse_params_list_int_list_newlines_tabs():
    result = _parse_params_list("int_list", "8\n9\t10", "", "[]")
    assert result == [{"value": 8}, {"value": 9}, {"value": 10}]


def test_parse_params_list_int_list_empty():
    result = _parse_params_list("int_list", "", "", "[]")
    assert result == []


def test_parse_params_list_int_list_spaces():
    result = _parse_params_list("int_list", " 8 , 9 ", "", "[]")
    assert result == [{"value": 8}, {"value": 9}]


def test_parse_params_list_int_list_invalid_token():
    with pytest.raises(ValueError, match="invalid integer token"):
        _parse_params_list("int_list", "8,abc,10", "", "[]")


def test_parse_params_list_string_list_multiline():
    result = _parse_params_list("string_list", "", "cat\ndog\ncar", "[]")
    assert len(result) == 3
    assert result[0] == {"value": "cat"}
    assert result[1] == {"value": "dog"}
    assert result[2] == {"value": "car"}


def test_parse_params_list_string_list_single_csv():
    result = _parse_params_list("string_list", "", "cat,dog,car", "[]")
    assert len(result) == 3
    assert result[0] == {"value": "cat"}
    assert result[1] == {"value": "dog"}
    assert result[2] == {"value": "car"}


def test_parse_params_list_string_list_empty():
    result = _parse_params_list("string_list", "", "", "[]")
    assert result == []


def test_parse_params_list_json_list_valid():
    result = _parse_params_list("json_list", "", "", '[{"steps":8},{"steps":9}]')
    assert len(result) == 2
    assert result[0] == {"steps": 8}
    assert result[1] == {"steps": 9}


def test_parse_params_list_json_list_empty():
    result = _parse_params_list("json_list", "", "", "[]")
    assert result == []


def test_parse_params_list_json_list_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_params_list("json_list", "", "", "not json")


def test_parse_params_list_json_list_non_dict_items():
    with pytest.raises(ValueError, match="array of objects"):
        _parse_params_list("json_list", "", "", "[1,2,3]")


def test_parse_params_list_unknown_mode():
    with pytest.raises(ValueError, match="Unknown params_mode"):
        _parse_params_list("unknown", "", "", "[]")


# ── _validate_loop_ctx ─────────────────────────────────────────────────────


def test_validate_loop_ctx_valid(sample_loop_ctx):
    result = _validate_loop_ctx(sample_loop_ctx)
    assert result is sample_loop_ctx


def test_validate_loop_ctx_not_dict():
    with pytest.raises(ValueError, match="must be an object"):
        _validate_loop_ctx("not a dict")


def test_validate_loop_ctx_wrong_version(sample_loop_ctx):
    sample_loop_ctx["version"] = 2
    with pytest.raises(ValueError, match="version must be 3"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_wrong_mode(sample_loop_ctx):
    sample_loop_ctx["mode"] = "while"
    with pytest.raises(ValueError, match="mode must be for_each"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_missing_run_id(sample_loop_ctx):
    sample_loop_ctx["run_id"] = ""
    with pytest.raises(ValueError, match="run_id is required"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_count_mismatch(sample_loop_ctx):
    sample_loop_ctx["count"] = 5
    with pytest.raises(ValueError, match="count does not match"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_index_out_of_range(sample_loop_ctx):
    sample_loop_ctx["index"] = 5
    with pytest.raises(ValueError, match="index out of range"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_count_0_index_0(sample_loop_ctx):
    sample_loop_ctx["count"] = 0
    sample_loop_ctx["index"] = 0
    sample_loop_ctx["params_list"] = []
    sample_loop_ctx["current_params"] = {}
    sample_loop_ctx["is_last"] = True
    result = _validate_loop_ctx(sample_loop_ctx)
    assert result["count"] == 0


def test_validate_loop_ctx_count_0_index_1(sample_loop_ctx):
    sample_loop_ctx["count"] = 0
    sample_loop_ctx["index"] = 1
    sample_loop_ctx["params_list"] = []
    with pytest.raises(ValueError, match="index must be 0"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_missing_current_params(sample_loop_ctx):
    sample_loop_ctx["current_params"] = "bad"
    with pytest.raises(ValueError, match="current_params must be an object"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_missing_state(sample_loop_ctx):
    sample_loop_ctx["state"] = "bad"
    with pytest.raises(ValueError, match="state must be an object"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_missing_collectors(sample_loop_ctx):
    sample_loop_ctx["collectors"] = "bad"
    with pytest.raises(ValueError, match="collectors must be an object"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_missing_meta_body_in_id(sample_loop_ctx):
    del sample_loop_ctx["meta"]["body_in_id"]
    with pytest.raises(ValueError, match="body_in_id is required"):
        _validate_loop_ctx(sample_loop_ctx)


def test_validate_loop_ctx_returns_ctx(sample_loop_ctx):
    result = _validate_loop_ctx(sample_loop_ctx)
    assert result is sample_loop_ctx


# ── _value_from_json_string ────────────────────────────────────────────────


def test_value_from_json_string_valid_json():
    result, ok = _value_from_json_string('{"key":1}')
    assert result == {"key": 1}
    assert ok is True


def test_value_from_json_string_empty():
    result, ok = _value_from_json_string("")
    assert result is None
    assert ok is False


def test_value_from_json_string_whitespace():
    result, ok = _value_from_json_string("   ")
    assert result is None
    assert ok is False


def test_value_from_json_string_plain_string():
    result, ok = _value_from_json_string("hello")
    assert result == "hello"
    assert ok is True


def test_value_from_json_string_number():
    result, ok = _value_from_json_string("42")
    assert result == 42
    assert ok is True


# ── _coerce_bool ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["true", "1", "yes", "y", "on", True, 1])
def test_coerce_bool_true_values(value):
    assert _coerce_bool(value) is True


@pytest.mark.parametrize("value", ["false", "0", "no", "n", "off", "", False, 0])
def test_coerce_bool_false_values(value):
    assert _coerce_bool(value) is False


def test_coerce_bool_float():
    assert _coerce_bool(1.0) is True
    assert _coerce_bool(0.0) is False


@pytest.mark.parametrize("value", ["TRUE", "True", "TrUe"])
def test_coerce_bool_case_insensitive(value):
    assert _coerce_bool(value) is True


def test_coerce_bool_invalid():
    with pytest.raises(ValueError, match="cannot cast"):
        _coerce_bool("maybe")


def test_coerce_bool_none():
    with pytest.raises(ValueError, match="cannot cast"):
        _coerce_bool(None)


# ── _base_class_type ───────────────────────────────────────────────────────


def test_base_class_type_with_suffix():
    assert _base_class_type("SomeNode|Mie") == "SomeNode"


def test_base_class_type_without_suffix():
    assert _base_class_type("SomeNode") == "SomeNode"


def test_base_class_type_empty():
    assert _base_class_type("") == ""


# ── _is_protocol_class ─────────────────────────────────────────────────────


def test_is_protocol_class_body_in():
    assert _is_protocol_class("MieLoopBodyIn") is True


def test_is_protocol_class_body_out():
    assert _is_protocol_class("MieLoopBodyOut") is True


def test_is_protocol_class_end():
    assert _is_protocol_class("MieLoopEnd") is True


def test_is_protocol_class_start():
    assert _is_protocol_class("MieLoopStart") is False


def test_is_protocol_class_other():
    assert _is_protocol_class("KSampler") is False


# ── _is_collector_class ────────────────────────────────────────────────────


def test_is_collector_class_image():
    assert _is_collector_class("MieLoopCollectImage") is True


def test_is_collector_class_other():
    assert _is_collector_class("MieLoopBodyIn") is False


# ── _is_excluded_output_class ──────────────────────────────────────────────


def test_is_excluded_save_image():
    assert _is_excluded_output_class("SaveImage") is True


def test_is_excluded_preview():
    assert _is_excluded_output_class("PreviewImage") is True


def test_is_excluded_loop_start():
    assert _is_excluded_output_class("MieLoopStart") is True


def test_is_excluded_loop_resume():
    assert _is_excluded_output_class("MieLoopResume") is True


def test_is_excluded_viewer():
    assert _is_excluded_output_class("ImageViewer") is True


def test_is_excluded_display():
    assert _is_excluded_output_class("DisplayAnything") is True


def test_is_excluded_ksampler():
    assert _is_excluded_output_class("KSampler") is False


def test_is_excluded_lora_loader():
    assert _is_excluded_output_class("LoraLoader") is False


# ── is_link ────────────────────────────────────────────────────────────────


def test_is_link_valid():
    assert is_link(["node1", 0]) is True


def test_is_link_valid_int_id():
    assert is_link([123, 1]) is True


def test_is_link_empty_list():
    assert is_link([]) is False


def test_is_link_short_list():
    assert is_link(["node1"]) is False


def test_is_link_wrong_types():
    assert is_link([123, "0"]) is False


def test_is_link_not_list():
    assert is_link("string") is False


def test_is_link_dict():
    assert is_link({"a": 1}) is False


# ── _get_inputs ────────────────────────────────────────────────────────────


def test_get_inputs_dict_with_inputs():
    assert _get_inputs({"inputs": {"a": 1}}) == {"a": 1}


def test_get_inputs_dict_no_inputs():
    assert _get_inputs({"class_type": "X"}) == {}


def test_get_inputs_dict_inputs_not_dict():
    assert _get_inputs({"inputs": "bad"}) == {}


def test_get_inputs_object_with_inputs():
    class FakeNode:
        inputs = {"x": 2}

    assert _get_inputs(FakeNode()) == {"x": 2}


def test_get_inputs_none():
    assert _get_inputs(None) == {}


# ── _get_class_type ────────────────────────────────────────────────────────


def test_get_class_type_dict():
    assert _get_class_type({"class_type": "KSampler"}) == "KSampler"


def test_get_class_type_object():
    class FakeNode:
        class_type = "MyNode"

    assert _get_class_type(FakeNode()) == "MyNode"


def test_get_class_type_none():
    assert _get_class_type(None) == ""
