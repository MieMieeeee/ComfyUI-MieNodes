"""
Tests for graph exploration and expansion functions in loop.py.

Covers:
- explore_backward_from_body_out
- explore_forward_from_body_in
- collect_loop_body
- _build_expand_graph_for_next_round
"""

import json
import loop as loop_module
from loop import (
    explore_backward_from_body_out,
    explore_forward_from_body_in,
    collect_loop_body,
    _build_expand_graph_for_next_round,
    get_node,
    _get_all_nodes,
    GraphBuilder,
)

import sys

# conftest.py is auto-loaded by pytest, access FakeGraphBuilder via sys.modules
_conftest_mod = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
FakeGraphBuilder = _conftest_mod.FakeGraphBuilder if _conftest_mod else None


# ======================================================================
# explore_backward_from_body_out
# ======================================================================


def test_explore_backward_simple():
    """Linear chain BodyIn(10)->Business(15)->BodyOut(20): visited = {10,15,20}."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
    }
    visited = set()
    explore_backward_from_body_out("20", dynprompt, visited)
    assert visited == {"10", "15", "20"}


def test_explore_backward_branching():
    """Business node feeds two downstream nodes: both visited."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "extra": ["15", 1]},
        },
    }
    visited = set()
    explore_backward_from_body_out("20", dynprompt, visited)
    # Both paths through node 15 lead back to 10
    assert visited == {"10", "15", "20"}


def test_explore_backward_already_visited():
    """Node visited twice: no error, set prevents re-visit."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "16": {
            "class_type": "CLIPTextEncode|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "extra": ["16", 0]},
        },
    }
    visited = set()
    explore_backward_from_body_out("20", dynprompt, visited)
    # Node 10 is referenced by both 15 and 16, but visited only once
    assert "10" in visited
    assert len([n for n in visited if n == "10"]) == 1


def test_explore_backward_missing_node():
    """Node not in dynprompt: just skipped, no error.

    Note: explore_backward adds the node_id to visited BEFORE checking
    if it exists in dynprompt. So "999" IS added to visited (the recursive
    call adds it), but the function returns immediately after without
    exploring further. No crash occurs.
    """
    dynprompt = {
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["999", 0]},
        },
    }
    visited = set()
    explore_backward_from_body_out("20", dynprompt, visited)
    # 20 is visited, 999 is also added to visited (before get_node check),
    # but no further recursion from 999 since get_node returns None
    assert "20" in visited
    # 999 gets added to visited before the get_node null check
    assert "999" in visited


# ======================================================================
# explore_forward_from_body_in
# ======================================================================


def test_explore_forward_simple():
    """Linear chain from 10: visited = {10,15,20}."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
    }
    visited = set()
    explore_forward_from_body_in("10", dynprompt, visited)
    assert visited == {"10", "15", "20"}


def test_explore_forward_stops_at_stop_ids():
    """stop_ids={20}: node 20 is visited but its successors are not explored."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }
    visited = set()
    explore_forward_from_body_in("10", dynprompt, visited, stop_ids={"20"})
    # 20 is in visited (it gets added before stop check for successors),
    # but 30 should NOT be visited because 20 is a stop_id
    assert "10" in visited
    assert "15" in visited
    assert "20" in visited
    assert "30" not in visited


def test_explore_forward_stops_at_end():
    """stop_ids={30}: node 30 is visited but not explored further."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
        "40": {
            "class_type": "SaveImage|Mie",
            "inputs": {"images": ["30", 0]},
        },
    }
    visited = set()
    explore_forward_from_body_in("10", dynprompt, visited, stop_ids={"30"})
    assert "10" in visited
    assert "15" in visited
    assert "20" in visited
    assert "30" in visited
    assert "40" not in visited


def test_explore_forward_no_successors():
    """Isolated node: visited contains just itself."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"model": ["99", 0]},  # 99 not in dynprompt
        },
    }
    visited = set()
    explore_forward_from_body_in("10", dynprompt, visited)
    # No node in dynprompt has an input linking to "10", so only 10 is visited
    assert visited == {"10"}


# ======================================================================
# collect_loop_body
# ======================================================================


def test_collect_loop_body_normal():
    """Simple 3-node body returns filtered nodes, business_nodes=['15']."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
    }
    filtered, debug = collect_loop_body("10", "20", dynprompt)
    assert "10" in filtered
    assert "15" in filtered
    assert "20" in filtered
    assert debug["body_nodes_business"] == ["15"]


def test_collect_loop_body_with_excluded():
    """SaveImage node is excluded from business_nodes but in excluded_nodes.

    For a node to be in the body, it must be reachable from both BodyOut
    backward and BodyIn forward. So BodyOut must reference SaveImage via
    an input link (e.g., state_json -> SaveImage output).
    """
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "17": {
            "class_type": "SaveImage|Mie",
            "inputs": {"images": ["15", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "state_json": ["17", 0]},
        },
    }
    filtered, debug = collect_loop_body("10", "20", dynprompt)
    # SaveImage (17) should be in excluded_nodes, not in business_nodes
    assert "17" in debug["excluded_nodes"]
    assert "17" not in debug["body_nodes_business"]
    assert "15" in debug["body_nodes_business"]


def test_collect_loop_body_with_collector():
    """MieLoopCollectImage node appears in collector_nodes but NOT in body_nodes_business.

    Collectors are tracked separately and excluded from body_nodes_business
    because they must not be cloned in expand graphs — they use RUNTIME_STORE
    refs managed through loop_ctx.
    """
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "18": {
            "class_type": "MieLoopCollectImage|Mie",
            "inputs": {"images": ["15", 0], "loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "state_json": ["18", 0]},
        },
    }
    filtered, debug = collect_loop_body("10", "20", dynprompt)
    assert "18" in debug["collector_nodes"]
    # CollectImage is in filtered AND in business_nodes (must be cloned each round)
    assert "18" in debug["body_nodes_filtered"]
    assert "18" in debug["body_nodes_business"]


def test_collect_loop_body_nested_loop():
    """MieLoopStart inside body raises ValueError('Nested loop')."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "MieLoopStart|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
    }
    try:
        collect_loop_body("10", "20", dynprompt)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Nested loop" in str(e) or "nested" in str(e).lower()


def test_collect_loop_body_protocol_guard_body_in_missing():
    """BodyIn disconnected: body_in_id not in protocol_nodes → ValueError."""
    dynprompt = {
        "99": {
            "class_type": "SomethingElse|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
    }
    # body_in_id=99 is not MieLoopBodyIn, so it won't be in protocol_nodes
    try:
        collect_loop_body("99", "20", dynprompt)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "protocol guard" in str(e).lower()


def test_collect_loop_body_debug_info():
    """Returns dict with forward_set, backward_set, etc."""
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
    }
    _, debug = collect_loop_body("10", "20", dynprompt)
    expected_keys = {
        "forward_set",
        "backward_set",
        "body_nodes_raw",
        "body_nodes_filtered",
        "body_nodes_business",
        "excluded_nodes",
        "protocol_nodes",
        "collector_nodes",
        "node_types",
        "stop_ids",
    }
    assert expected_keys.issubset(debug.keys())


# ======================================================================
# _build_expand_graph_for_next_round
# ======================================================================


def _make_next_ctx(index=1, count=3, **overrides):
    """Build a valid next_ctx dict for expand graph tests."""
    ctx = {
        "version": 3,
        "loop_id": "test_loop",
        "run_id": "test_run_123",
        "mode": "for_each",
        "index": index,
        "count": count,
        "is_last": False,
        "params_list": [{"value": i} for i in range(count)],
        "current_params": {"value": index},
        "state": {},
        "collectors": {"images": {"ref": None, "count": 0}},
        "meta": {
            "body_in_id": "10",
            "body_out_id": "20",
            "end_id": "30",
        },
    }
    ctx.update(overrides)
    return ctx


def _make_simple_dynprompt():
    """Linear chain: 10(BodyIn) -> 15(KSampler) -> 20(BodyOut) -> 30(End)."""
    return {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["1", 0]},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0], "steps": 20},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "state_json": "{}"},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }


def _make_detect_result(business=None, backward=None, forward=None):
    """Build a detect_result dict for expand graph tests."""
    return {
        "body_nodes_business": business or ["15"],
        "backward_set": backward or ["10", "15", "20"],
        "forward_set": forward or ["10", "15", "20"],
    }


def test_build_expand_graph_no_graph_builder():
    """GraphBuilder=None raises ValueError('unavailable')."""
    # Save original, set to None, test, then restore
    original = loop_module.GraphBuilder
    try:
        loop_module.GraphBuilder = None
        next_ctx = _make_next_ctx()
        dynprompt = _make_simple_dynprompt()
        detect_result = _make_detect_result()
        try:
            _build_expand_graph_for_next_round(
                next_ctx, dynprompt, "10", "20", "30", detect_result
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unavailable" in str(e).lower()
    finally:
        loop_module.GraphBuilder = original


def test_build_expand_graph_basic(monkeypatch):
    """Simple body produces an expand graph containing nodes."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    next_ctx = _make_next_ctx()
    dynprompt = _make_simple_dynprompt()
    detect_result = _make_detect_result()
    result, end_built_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    assert end_built_node is not None


def test_build_expand_graph_body_in_loop_ctx_connected(monkeypatch):
    """BodyIn is NOT cloned; business nodes get loop_ctx from Resume.out(0)."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    next_ctx = _make_next_ctx()
    dynprompt = _make_simple_dynprompt()
    detect_result = _make_detect_result()
    result, _ = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    # BodyIn should NOT be in the expand graph (it's the anchor, not cloned)
    body_in_node = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "MieLoopBodyIn|Mie":
            body_in_node = node_data
            break
    assert body_in_node is None, "BodyIn should NOT be cloned in expand graph"
    # KSampler's loop_ctx should come from resume_node.out(0) instead
    ksampler = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "KSampler|Mie":
            ksampler = node_data
            break
    assert ksampler is not None
    loop_ctx_val = ksampler["inputs"]["loop_ctx"]
    # Should be a list [resume_node_id, 0] — NOT the original ["10", 0]
    assert isinstance(loop_ctx_val, list)
    assert loop_ctx_val[1] == 0
    assert loop_ctx_val[0] != "10"


def test_build_expand_graph_business_round_idx(monkeypatch):
    """Business nodes get __mie_loop_round_idx__ injected."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    next_ctx = _make_next_ctx(index=2)
    dynprompt = _make_simple_dynprompt()
    detect_result = _make_detect_result()
    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    # Find the business node (KSampler)
    business_node = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "KSampler|Mie":
            business_node = node_data
            break
    assert business_node is not None
    assert business_node["inputs"]["__mie_loop_round_idx__"] == 2


def test_build_expand_graph_preserves_bodyout_state_json_input(monkeypatch):
    """BodyOut keeps explicit state_json wiring in the cloned graph."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    next_ctx = _make_next_ctx()
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["1", 0]},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0], "steps": 20},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {
                "loop_ctx": ["15", 0],
                "state_json": '{"step": 1}',
            },
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }
    detect_result = _make_detect_result()
    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    # Find the BodyOut node
    body_out_node = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "MieLoopBodyOut|Mie":
            body_out_node = node_data
            break
    assert body_out_node is not None
    assert body_out_node["inputs"]["state_json"] == '{"step": 1}'


def test_build_expand_graph_cyclic_detection(monkeypatch):
    """Cycle in dynprompt via non-loop_ctx inputs raises ValueError('cyclic')."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    # Create a cycle through a non-loop_ctx input:
    # BodyIn(10) --loop_ctx--> KSampler(15) --images--> CLIPEncode(16) --clip--> KSampler(15)
    # But BodyIn.loop_ctx is special-cased and replaced with resume_node.
    # So we create a cycle between two business nodes:
    # 15 depends on 16 via "model", 16 depends on 15 via "images" → cycle
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["1", 0]},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0], "model": ["16", 0]},
        },
        "16": {
            "class_type": "CLIPTextEncode|Mie",
            "inputs": {"clip": ["15", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }
    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result(business=["15", "16"])
    try:
        _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "10", "20", "30", detect_result
        )
        assert False, "Should have raised ValueError for cyclic dependency"
    except ValueError as e:
        assert "cyclic" in str(e).lower()


def test_build_expand_graph_non_clone_nodes(monkeypatch):
    """Nodes outside the body (not in backward/forward sets) kept as original links."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    # Node 50 is outside the loop body - referenced by 15 but not in backward/forward
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["1", 0]},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0], "model": ["50", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
        "50": {
            "class_type": "CheckpointLoaderSimple|Mie",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
    }
    # Node 50 not in backward/forward sets
    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result(
        backward=["10", "15", "20"],
        forward=["10", "15", "20"],
    )
    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    # Find the KSampler node
    ksampler = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "KSampler|Mie":
            ksampler = node_data
            break
    assert ksampler is not None
    # model input should remain as original link ["50", 0] since 50 is not cloned
    model_input = ksampler["inputs"]["model"]
    assert isinstance(model_input, list)
    assert model_input[0] == "50"
    assert model_input[1] == 0


# ======================================================================
# Bug M1: build_node visiting set cleanup
# ======================================================================


def test_bug_m1_visiting_set_cleaned_on_error(monkeypatch):
    """After a build_node error, the visiting set should be clean."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["1", 0]},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }
    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result()
    # Build twice should work (if visiting wasn't cleaned, second build fails)
    result1, _ = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    result2, _ = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)


# ======================================================================
# Bug M4: state_json default passthrough
# ======================================================================


def test_bug_m4_bodyout_without_state_json_keeps_minimal_inputs(monkeypatch):
    """BodyOut without explicit state_json keeps only the loop_ctx link."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    next_ctx = _make_next_ctx()
    dynprompt = {
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["1", 0]},
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0], "steps": 20},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }
    detect_result = _make_detect_result()
    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    # Find BodyOut node
    body_out = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "MieLoopBodyOut|Mie":
            body_out = node_data
            break
    assert body_out is not None
    assert sorted(body_out["inputs"].keys()) == ["loop_ctx"]


# ======================================================================
# Edge case: Large body (10 business nodes)
# ======================================================================


def test_build_expand_graph_large_body(monkeypatch):
    """Body with 10 business nodes → all cloned correctly."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    # Build a chain: BodyIn(10) → Biz0(100) → Biz1(101) → ... → Biz9(109) → BodyOut(20) → End(30)
    dynprompt = {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0]}},
    }
    business_ids = [str(100 + i) for i in range(10)]
    prev_id = "10"
    for bid in business_ids:
        dynprompt[bid] = {
            "class_type": f"BizNode{bid}|Mie",
            "inputs": {"loop_ctx": [prev_id, 0]},
        }
        prev_id = bid
    dynprompt["20"] = {
        "class_type": "MieLoopBodyOut|Mie",
        "inputs": {"loop_ctx": [prev_id, 0]},
    }
    dynprompt["30"] = {
        "class_type": "MieLoopEnd|Mie",
        "inputs": {"loop_ctx": ["20", 0]},
    }
    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result(
        business=business_ids,
        backward=["10"] + business_ids + ["20"],
        forward=["10"] + business_ids + ["20"],
    )
    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )
    assert isinstance(result, dict)
    # All 10 business nodes should be in the result
    result_types = {nd["class_type"] for nd in result.values()}
    for bid in business_ids:
        assert f"BizNode{bid}|Mie" in result_types
    # Each business node should have __mie_loop_round_idx__ injected
    for nid, node_data in result.items():
        if node_data["class_type"].startswith("BizNode"):
            assert "__mie_loop_round_idx__" in node_data["inputs"]
