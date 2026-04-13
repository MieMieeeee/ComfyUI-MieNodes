import sys

import loop as loop_module
from loop import MieLoopBodyOut, MieLoopEnd, _build_expand_graph_for_next_round

_conftest_mod = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
FakeGraphBuilder = _conftest_mod.FakeGraphBuilder if _conftest_mod else None


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "collector_loop",
        "run_id": "run_collector_1",
        "mode": "for_each",
        "index": 0,
        "count": 2,
        "is_last": False,
        "params_list": [{"value": 1}, {"value": 2}],
        "current_params": {"value": 1},
        "state": {},
        "collectors": {"image": {"ref": None, "count": 0}},
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    ctx.update(overrides)
    return ctx


def test_expand_clones_intermediate_non_business_nodes(monkeypatch):
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    dynprompt = {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0]}},
        "11": {"class_type": "PrimitiveNode", "inputs": {"value": 7}},
        "12": {"class_type": "MathExpression", "inputs": {"a": ["11", 0]}},
        "15": {"class_type": "KSampler|Mie", "inputs": {"loop_ctx": ["10", 0], "steps": ["12", 0]}},
        "20": {"class_type": "MieLoopBodyOut|Mie", "inputs": {"loop_ctx": ["15", 0], "value_any_1": ["12", 0]}},
        "30": {"class_type": "MieLoopEnd|Mie", "inputs": {"loop_ctx": ["20", 0], "state_json": "{}"}},
    }
    detect_result = {
        "body_nodes_business": ["15"],
        "body_nodes_filtered": ["10", "11", "12", "15", "20"],
        "backward_set": ["10", "11", "12", "15", "20"],
        "forward_set": ["10", "11", "12", "15", "20", "30"],
    }
    graph, _ = _build_expand_graph_for_next_round(_make_ctx(), dynprompt, "10", "20", "30", detect_result)
    class_types = {node["class_type"] for node in graph.values()}
    assert "PrimitiveNode" in class_types
    assert "MathExpression" in class_types


def test_bodyout_and_end_store_value_any():
    ctx = _make_ctx(count=1, is_last=True, params_list=[{"value": 1}], current_params={"value": 1})
    body_out = MieLoopBodyOut()
    body_out_result = body_out.execute(ctx, state_json="{}", value_any_1="hello", value_any_3=123)
    body_ctx = body_out_result[0]
    assert body_ctx["value_any"] == ["hello", None, 123, None, None]

    end = MieLoopEnd()
    end_result = end.execute(body_ctx, "{}", value_any_2="world")
    end_ctx = end_result[0]
    assert end_ctx["value_any"] == ["hello", "world", 123, None, None]


def test_expand_injects_previous_value_any_into_cloned_bodyout(monkeypatch):
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    dynprompt = {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0]}},
        "15": {"class_type": "KSampler|Mie", "inputs": {"loop_ctx": ["10", 0]}},
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "value_any_1": None, "value_any_2": None},
        },
        "30": {"class_type": "MieLoopEnd|Mie", "inputs": {"loop_ctx": ["20", 0], "state_json": "{}"}},
    }
    detect_result = {
        "body_nodes_business": ["15"],
        "body_nodes_filtered": ["10", "15", "20"],
        "backward_set": ["10", "15", "20"],
        "forward_set": ["10", "15", "20", "30"],
    }
    next_ctx = _make_ctx(value_any=["foo", {"bar": 1}, None, None, None])
    graph, _ = _build_expand_graph_for_next_round(next_ctx, dynprompt, "10", "20", "30", detect_result)
    bodyout = next(node for node in graph.values() if node["class_type"] == "MieLoopBodyOut|Mie")
    assert bodyout["inputs"]["value_any_1"] == "foo"
    assert bodyout["inputs"]["value_any_2"] == {"bar": 1}


def test_expand_preserves_bodyout_loop_ctx_chain_for_feedback_nodes(monkeypatch):
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    dynprompt = {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0], "anchor": ["45", 0]}},
        "45": {"class_type": "LoadImage", "inputs": {"image": "foo.png"}},
        "46": {
            "class_type": "MieLoopStateGetImage|Mie",
            "inputs": {"loop_ctx": ["10", 0], "fallback_image": ["10", 1], "key": "feedback_image"},
        },
        "52": {"class_type": "AddNumberWatermarkForImage|Mie", "inputs": {"images": ["46", 0]}},
        "49": {
            "class_type": "MieLoopStateSetImage|Mie",
            "inputs": {"loop_ctx": ["10", 0], "image": ["52", 0], "key": "feedback_image"},
        },
        "59": {
            "class_type": "MieLoopCollectImage|Mie",
            "inputs": {"loop_ctx": ["49", 0], "image": ["52", 0]},
        },
        "20": {"class_type": "MieLoopBodyOut|Mie", "inputs": {"loop_ctx": ["59", 0], "value_image": ["52", 0]}},
        "30": {"class_type": "MieLoopEnd|Mie", "inputs": {"loop_ctx": ["20", 0], "state_json": "{}"}},
    }
    detect_result = {
        "body_nodes_business": ["46", "49", "52"],
        "body_nodes_filtered": ["10", "46", "49", "52", "59", "20"],
        "backward_set": ["10", "46", "49", "52", "59", "20"],
        "forward_set": ["10", "46", "49", "52", "59", "20", "30"],
    }
    graph, _ = _build_expand_graph_for_next_round(_make_ctx(), dynprompt, "10", "20", "30", detect_result)

    bodyout = next(node for node in graph.values() if node["class_type"] == "MieLoopBodyOut|Mie")
    collect = next(node for node in graph.values() if node["class_type"] == "MieLoopCollectImage|Mie")
    end = next(node for node in graph.values() if node["class_type"] == "MieLoopEnd|Mie")

    assert bodyout["inputs"]["loop_ctx"] == [collect.get("override_display_id", "fake.59"), 0] or bodyout["inputs"]["loop_ctx"] == ["fake.59", 0]
    assert end["inputs"]["loop_ctx"] == ["fake.20", 0]
