"""
Tests for loop optimization changes (T1 dead code removal, T2 simplified _build_expand_graph_for_next_round).

Covers:
- T1: walk_from_body_in removal
- T2: simplified should_clone using protocol_nodes | business_set
- T2: uses body_nodes_business from detect_result
- T2: resume_node.out(0) wiring to BodyIn.loop_ctx
- T2: __mie_loop_round_idx__ injection for business nodes
"""

import sys

import loop as loop_module
from loop import (
    _build_expand_graph_for_next_round,
)

# conftest.py is auto-loaded by pytest, access FakeGraphBuilder and FakeGraphNode via sys.modules
_conftest_mod = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
FakeGraphBuilder = _conftest_mod.FakeGraphBuilder if _conftest_mod else None
FakeGraphNode = _conftest_mod.FakeGraphNode if _conftest_mod else None


# ======================================================================
# Helpers (same pattern as test_loop_graph.py)
# =====================================================================


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


# ======================================================================
# T1: Dead code removal - walk_from_body_in should not exist
# ======================================================================


def test_walk_from_body_in_removed():
    """Verify walk_from_body_in function no longer exists in loop.py.

    T1 removed this function as part of dead code elimination.
    """
    assert not hasattr(loop_module, "walk_from_body_in"), (
        "walk_from_body_in should have been removed as dead code"
    )


# ======================================================================
# T2: Simplified _build_expand_graph_for_next_round tests
# ======================================================================


def test_should_clone_simplified(monkeypatch):
    """Verify should_clone checks protocol_nodes | business_set.

    T2 simplified the cloning logic to only consider:
    - protocol_nodes (body_in_id, body_out_id, end_id)
    - business_nodes (from detect_result['body_nodes_business'])
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    # Build a graph where node "50" is a non-business, non-protocol node
    # that is referenced by a business node
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

    next_ctx = _make_next_ctx()
    # Only "15" is a business node, "50" is outside the body
    detect_result = _make_detect_result(business=["15"])

    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )

    # Node "50" should NOT be cloned (it's not protocol, not business)
    result_ids = set(result.keys())
    assert "50" not in result_ids, (
        "Node 50 should not be cloned - it's not in protocol_nodes or business_set"
    )

    # Business node 15 should be cloned
    ksampler = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "KSampler|Mie":
            ksampler = node_data
            break
    assert ksampler is not None, "KSampler (business node) should be cloned"

    # Node 50's link should remain as original ["50", 0]
    assert ksampler["inputs"]["model"] == ["50", 0], (
        "Non-cloned node reference should remain as original link"
    )


def test_body_nodes_business_used(monkeypatch):
    """Verify _build_expand_graph_for_next_round uses body_nodes_business.

    T2 changed from using body_nodes_filtered to body_nodes_business
    for determining which nodes to clone.
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    # Create a scenario where body_nodes_business differs from body_nodes_filtered
    # SaveImage (17) should be excluded from business but in filtered
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
            "class_type": "SaveImage|Mie",  # excluded from business
            "inputs": {"images": ["15", 0]},
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["15", 0], "state_json": ["17", 0]},
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {"loop_ctx": ["20", 0]},
        },
    }

    # Only "15" is the true business node, SaveImage (17) is excluded
    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result(
        business=["15"],  # only KSampler is business
        backward=["10", "15", "17", "20"],  # SaveImage reachable
        forward=["10", "15", "17", "20"],  # SaveImage reachable
    )

    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )

    # KSampler (15) should be cloned and have __mie_loop_round_idx__
    ksampler = None
    for nid, node_data in result.items():
        if node_data["class_type"] == "KSampler|Mie":
            ksampler = node_data
            break
    assert ksampler is not None
    assert "__mie_loop_round_idx__" in ksampler["inputs"]

    # SaveImage (17) should NOT be in result (not business, not protocol)
    save_image_found = any(
        node_data["class_type"] == "SaveImage|Mie" for node_data in result.values()
    )
    assert not save_image_found, (
        "SaveImage should not be cloned - it's excluded from body_nodes_business"
    )


def test_expand_produces_correct_nodes(monkeypatch):
    """Verify expand graph produces body_in, body_out, end, and business nodes.

    T2 ensures only these 4 types of nodes appear in the expanded graph:
    - Protocol nodes: body_in, body_out, end
    - Business nodes: from body_nodes_business
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    dynprompt = _make_simple_dynprompt()
    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result()

    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )

    # Check expected node types (BodyIn is NOT cloned — it's the anchor)
    result_types = {node_data["class_type"] for node_data in result.values()}

    assert "MieLoopBodyIn|Mie" not in result_types, (
        "BodyIn should NOT be cloned (anchor)"
    )
    assert "MieLoopBodyOut|Mie" in result_types, "BodyOut should be in result"
    assert "MieLoopEnd|Mie" in result_types, "End should be in result"
    assert "KSampler|Mie" in result_types, (
        "Business node (KSampler) should be in result"
    )

    # No other node types should be present (Resume is also expected)
    extra_types = result_types - {
        "MieLoopBodyOut|Mie",
        "MieLoopEnd|Mie",
        "KSampler|Mie",
        "MieLoopResume|Mie",  # Resume node is created by the expand graph
    }
    assert not extra_types, f"Unexpected node types in result: {extra_types}"


def test_resume_node_loop_ctx_wiring(monkeypatch):
    """Verify resume_node.out(0) correctly connects to business nodes.

    Since BodyIn is NOT cloned (it's the anchor), business nodes that
    previously got loop_ctx from BodyIn now get it from resume_node.out(0).
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    next_ctx = _make_next_ctx()
    dynprompt = _make_simple_dynprompt()
    detect_result = _make_detect_result()

    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )

    # BodyIn should NOT be in the result
    body_in_found = any(
        node_data["class_type"] == "MieLoopBodyIn|Mie" for node_data in result.values()
    )
    assert not body_in_found, "BodyIn should NOT be cloned"

    # KSampler's loop_ctx should come from resume_node.out(0) — a list [id, 0]
    ksampler_found = False
    for nid, node_data in result.items():
        if node_data["class_type"] == "KSampler|Mie":
            ksampler_found = True
            loop_ctx_val = node_data["inputs"]["loop_ctx"]
            assert isinstance(loop_ctx_val, list), (
                f"KSampler.loop_ctx should be a list link, got {type(loop_ctx_val)}"
            )
            assert loop_ctx_val[1] == 0, (
                f"KSampler.loop_ctx should connect to out(0), got index {loop_ctx_val[1]}"
            )
            # Should NOT be the original ["10", 0] link to BodyIn
            assert loop_ctx_val[0] != "10", (
                "KSampler.loop_ctx should NOT point to BodyIn (should be resume_node)"
            )

    assert ksampler_found, "KSampler node should be in result"


def test_mie_loop_round_idx_injected(monkeypatch):
    """Verify business nodes get __mie_loop_round_idx__ widget injected.

    T2 ensures __mie_loop_round_idx__ is injected into business nodes
    using next_ctx['index'].
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    # Test with index=5
    next_ctx = _make_next_ctx(index=5)
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

    assert business_node is not None, "Business node should be in result"
    assert "__mie_loop_round_idx__" in business_node["inputs"], (
        "Business node should have __mie_loop_round_idx__ injected"
    )
    assert business_node["inputs"]["__mie_loop_round_idx__"] == 5, (
        f"__mie_loop_round_idx__ should be 5 (from next_ctx['index']), "
        f"got {business_node['inputs']['__mie_loop_round_idx__']}"
    )


# ======================================================================
# Additional edge case tests for T2 optimizations
# ======================================================================


def test_multiple_business_nodes_all_get_round_idx(monkeypatch):
    """When multiple business nodes exist, all should get __mie_loop_round_idx__."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    dynprompt = {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0]}},
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {"loop_ctx": ["10", 0], "steps": 20},
        },
        "16": {
            "class_type": "CLIPTextEncode|Mie",
            "inputs": {"loop_ctx": ["15", 0], "text": "prompt"},
        },
        "20": {"class_type": "MieLoopBodyOut|Mie", "inputs": {"loop_ctx": ["16", 0]}},
        "30": {"class_type": "MieLoopEnd|Mie", "inputs": {"loop_ctx": ["20", 0]}},
    }

    next_ctx = _make_next_ctx(index=3)
    detect_result = _make_detect_result(business=["15", "16"])

    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )

    # Both business nodes should have __mie_loop_round_idx__
    business_count = 0
    for nid, node_data in result.items():
        if node_data["class_type"] in ("KSampler|Mie", "CLIPTextEncode|Mie"):
            business_count += 1
            assert "__mie_loop_round_idx__" in node_data["inputs"]
            assert node_data["inputs"]["__mie_loop_round_idx__"] == 3

    assert business_count == 2, "Both business nodes should be cloned"


def test_protocol_nodes_always_cloned(monkeypatch):
    """Protocol nodes (BodyIn, BodyOut, End) should always be cloned."""
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)

    # Even with no business nodes
    dynprompt = {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0]}},
        "20": {"class_type": "MieLoopBodyOut|Mie", "inputs": {"loop_ctx": ["10", 0]}},
        "30": {"class_type": "MieLoopEnd|Mie", "inputs": {"loop_ctx": ["20", 0]}},
    }

    next_ctx = _make_next_ctx()
    detect_result = _make_detect_result(business=[])  # no business nodes

    result, _end_node = _build_expand_graph_for_next_round(
        next_ctx, dynprompt, "10", "20", "30", detect_result
    )

    # BodyOut and End should be present; BodyIn is NOT cloned (anchor)
    result_types = {node_data["class_type"] for node_data in result.values()}
    assert "MieLoopBodyIn|Mie" not in result_types, "BodyIn should NOT be cloned"
    assert "MieLoopBodyOut|Mie" in result_types
    assert "MieLoopEnd|Mie" in result_types
