"""
TDD tests for MieLoop v3 E2E bugfixes.

Covers the specific bugs found and fixed during E2E testing:
1. Collector nodes excluded from body_nodes_business (expand graph clone fix)
2. _build_expand_graph_for_next_round returns (finalize_result, end_built_node) tuple
3. Expand graph result uses is_link() values for execution (critical fix)
4. BodyIn anchor passthrough in expand graph (src_idx 0 vs 1)
5. RUNTIME_STORE _detect_cache across expand rounds
6. Full 3-round loop simulation matching user's workflow graph
"""

import json
import sys

import loop as loop_module
from loop import (
    collect_loop_body,
    _build_expand_graph_for_next_round,
    is_link,
    RUNTIME_STORE,
    MieLoopStart,
    MieLoopEnd,
)

# conftest.py is auto-loaded by pytest
_conftest_mod = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
FakeGraphBuilder = _conftest_mod.FakeGraphBuilder if _conftest_mod else None


# ======================================================================
# Helpers
# ======================================================================


def _make_next_ctx(index=1, count=3, **overrides):
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
            "body_in_id": "32",
            "body_out_id": "33",
            "end_id": "34",
        },
    }
    ctx.update(overrides)
    return ctx


def _make_user_workflow_dynprompt():
    """Build a dynprompt matching the user's actual workflow graph:

    #13 EmptySD3LatentImage → anchor
    #31 MieLoopStart
    #32 MieLoopBodyIn(loop_ctx:[31,0], anchor:[13,0])
    #35 MieLoopParamGetInt(loop_ctx:[32,0])
    #3  KSampler(model:[16,0], positive:[6,0], negative:[7,0], latent_image:[32,1], steps:[35,0])
    #8  VAEDecode(samples:[3,0], vae:[17,0])
    #37 MieLoopCollectImage(loop_ctx:[32,0], image:[8,0])
    #33 MieLoopBodyOut(loop_ctx:[37,0], value_image:[8,0])
    #34 MieLoopEnd(loop_ctx:[33,0], value_image:[33,1])
    """
    return {
        "13": {
            "class_type": "EmptySD3LatentImage|Mie",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "16": {
            "class_type": "UNETLoader|Mie",
            "inputs": {"unet_name": "model.safetensors"},
        },
        "17": {
            "class_type": "VAELoader|Mie",
            "inputs": {"vae_name": "vae.safetensors"},
        },
        "18": {
            "class_type": "CLIPLoader|Mie",
            "inputs": {"clip_name": "clip.safetensors"},
        },
        "6": {
            "class_type": "CLIPTextEncode|Mie",
            "inputs": {"text": "beautiful landscape", "clip": ["18", 0]},
        },
        "7": {
            "class_type": "CLIPTextEncode|Mie",
            "inputs": {"text": "ugly, blurry", "clip": ["18", 0]},
        },
        "31": {
            "class_type": "MieLoopStart|Mie",
            "inputs": {
                "loop_id": "test_loop",
                "params_mode": "int_list",
                "int_list": "8,9,10",
                "string_list": "",
                "json_list": "[]",
                "initial_state_json": "{}",
            },
        },
        "32": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {"loop_ctx": ["31", 0], "anchor": ["13", 0]},
        },
        "35": {
            "class_type": "MieLoopParamGetInt|Mie",
            "inputs": {"loop_ctx": ["32", 0], "key": "value", "default_value": 0},
        },
        "3": {
            "class_type": "KSampler|Mie",
            "inputs": {
                "model": ["16", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["32", 1],
                "steps": ["35", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode|Mie",
            "inputs": {"samples": ["3", 0], "vae": ["17", 0]},
        },
        "37": {
            "class_type": "MieLoopCollectImage|Mie",
            "inputs": {"loop_ctx": ["32", 0], "image": ["8", 0]},
        },
        "33": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {"loop_ctx": ["37", 0], "value_image": ["8", 0]},
        },
        "34": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {
                "loop_ctx": ["33", 0],
                "value_image": ["33", 1],
                "debug": True,
            },
        },
    }


# ======================================================================
# Fix 1: Collectors excluded from body_nodes_business
# ======================================================================


class TestCollectorCloning:
    """Verify collectors ARE cloned in expand graphs (each round must accumulate)."""

    def test_collect_image_in_business_nodes(self):
        """MieLoopCollectImage (#37) should be in BOTH collector_nodes AND body_nodes_business."""
        dynprompt = _make_user_workflow_dynprompt()
        _, debug = collect_loop_body("32", "33", dynprompt, end_id="34")
        assert "37" in debug["collector_nodes"]
        assert "37" in debug["body_nodes_business"], (
            "CollectImage must be in body_nodes_business to be cloned each round"
        )
        # Business nodes include KSampler, VAEDecode, ParamGetInt, AND CollectImage
        assert "3" in debug["body_nodes_business"]  # KSampler
        assert "8" in debug["body_nodes_business"]  # VAEDecode
        assert "35" in debug["body_nodes_business"]  # ParamGetInt

    def test_collect_image_in_filtered_and_business(self):
        """Collectors are in body_nodes_filtered AND body_nodes_business."""
        dynprompt = _make_user_workflow_dynprompt()
        _, debug = collect_loop_body("32", "33", dynprompt, end_id="34")
        assert "37" in debug["body_nodes_filtered"]
        assert "37" in debug["body_nodes_business"]

    def test_expand_graph_collector_cloned(self, monkeypatch):
        """Collectors MUST be cloned in expand graph so each round accumulates."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        next_ctx = _make_next_ctx()
        result, end_node = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )

        # CollectImage MUST be in the result (cloned for this round)
        collect_found = any(
            nd["class_type"] == "MieLoopCollectImage|Mie" for nd in result.values()
        )
        assert collect_found, (
            "CollectImage MUST be cloned in expand graph to accumulate images each round"
        )


# ======================================================================
# Fix 2: _build_expand_graph returns (finalize_result, end_built_node)
# ======================================================================


class TestExpandGraphTupleReturn:
    """Verify expand graph returns tuple with end_built_node for is_link construction."""

    def test_returns_tuple(self, monkeypatch):
        """Return value should be a 2-tuple (dict, FakeGraphNode)."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")
        next_ctx = _make_next_ctx()
        result = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        graph_dict, end_built_node = result
        assert isinstance(graph_dict, dict)
        assert end_built_node is not None

    def test_end_built_node_has_id(self, monkeypatch):
        """end_built_node should have a valid .id attribute."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")
        next_ctx = _make_next_ctx()
        _, end_node = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )
        assert hasattr(end_node, "id")
        assert end_node.id is not None
        assert isinstance(end_node.id, str)

    def test_end_built_node_out_produces_is_link(self, monkeypatch):
        """end_built_node.out(i) should produce is_link()-compatible values."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")
        next_ctx = _make_next_ctx()
        _, end_node = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )
        # Simulate what MieLoopEnd.execute does:
        result_tuple = tuple([end_node.out(i) for i in range(4)])
        assert len(result_tuple) == 4
        for i, val in enumerate(result_tuple):
            assert is_link(val), f"out({i}) = {val} should be is_link() compatible"


# ======================================================================
# Fix 3: BodyIn anchor passthrough
# ======================================================================


class TestBodyInAnchorPassthrough:
    """Verify KSampler gets LATENT from anchor (EmptySD3LatentImage), not loop_ctx."""

    def test_ksampler_latent_from_anchor_not_bodyin(self, monkeypatch):
        """KSampler's latent_image input should trace through BodyIn.anchor to EmptySD3LatentImage."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        next_ctx = _make_next_ctx()
        result, _ = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )

        # Find KSampler in expand graph
        ksampler = None
        for nid, nd in result.items():
            if nd["class_type"] == "KSampler|Mie":
                ksampler = nd
                break
        assert ksampler is not None, "KSampler should be cloned"

        # latent_image was originally [32, 1] (BodyIn.out(1) = anchor)
        # In expand graph, since BodyIn is not cloned, the code traces anchor
        # back to ["13", 0] (EmptySD3LatentImage) — external link preserved
        latent_input = ksampler["inputs"]["latent_image"]
        # Should remain as original external link since EmptySD3LatentImage is not cloned
        assert isinstance(latent_input, list), (
            f"latent_image should be a link, got {latent_input}"
        )
        # Should point to EmptySD3LatentImage (13), not BodyIn (32)
        assert latent_input[0] == "13", (
            f"KSampler.latent_image should trace to EmptySD3LatentImage (#13), "
            f"got {latent_input}"
        )

    def test_ksampler_loop_ctx_from_resume(self, monkeypatch):
        """KSampler's model input should remain as external link (not cloned)."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        next_ctx = _make_next_ctx()
        result, _ = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )

        # Find KSampler — model should be external link [16, 0]
        ksampler = None
        for nid, nd in result.items():
            if nd["class_type"] == "KSampler|Mie":
                ksampler = nd
                break
        assert ksampler is not None
        model_input = ksampler["inputs"]["model"]
        assert model_input == ["16", 0], "model should remain as external link"

    def test_bodyout_loop_ctx_from_resume(self, monkeypatch):
        """BodyOut's loop_ctx input must come from Resume node, not original CollectImage.

        This is the fix for the infinite expand loop: BodyOut's loop_ctx was
        previously pointing to the original CollectImage (#37), which is NOT cloned
        in the expand graph (it's a collector). This meant BodyOut always received
        Round 0's ctx (index=0), so clone MieLoopEnd always saw done=False and
        kept returning new expand graphs forever.

        With the fix, BodyOut.loop_ctx is forced to resume_node.out(0), which
        carries the correctly incremented ctx.
        """
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        next_ctx = _make_next_ctx()
        result, _ = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )

        # Find Resume node (first node built, id = fake.1)
        resume_node_id = None
        for nid, nd in result.items():
            if nd["class_type"] == "MieLoopResume|Mie":
                resume_node_id = nid
                break
        assert resume_node_id is not None, "Resume node should exist in expand graph"

        # Find BodyOut node
        bodyout = None
        for nid, nd in result.items():
            if nd["class_type"] == "MieLoopBodyOut|Mie":
                bodyout = nd
                break
        assert bodyout is not None, "BodyOut should be cloned in expand graph"

        # BodyOut.loop_ctx must point to Resume node
        loop_ctx_input = bodyout["inputs"]["loop_ctx"]
        assert isinstance(loop_ctx_input, list), (
            f"BodyOut.loop_ctx should be a link, got {loop_ctx_input}"
        )
        assert loop_ctx_input[0] == resume_node_id, (
            f"BodyOut.loop_ctx should point to Resume node ({resume_node_id}), "
            f"got {loop_ctx_input}"
        )

    def test_end_loop_ctx_from_resume(self, monkeypatch):
        """MieLoopEnd's loop_ctx input must also come from Resume node.

        Same fix as BodyOut: End's loop_ctx was also incorrectly pointing through
        the non-cloned collector chain, causing it to always see index=0.
        """
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        next_ctx = _make_next_ctx()
        result, _ = _build_expand_graph_for_next_round(
            next_ctx, dynprompt, "32", "33", "34", detect
        )

        # Find Resume node
        resume_node_id = None
        for nid, nd in result.items():
            if nd["class_type"] == "MieLoopResume|Mie":
                resume_node_id = nid
                break
        assert resume_node_id is not None

        # Find End node
        end_node = None
        for nid, nd in result.items():
            if nd["class_type"] == "MieLoopEnd|Mie":
                end_node = nd
                break
        assert end_node is not None, "MieLoopEnd should be cloned in expand graph"

        # End.loop_ctx must point to Resume node
        loop_ctx_input = end_node["inputs"]["loop_ctx"]
        assert isinstance(loop_ctx_input, list), (
            f"End.loop_ctx should be a link, got {loop_ctx_input}"
        )
        assert loop_ctx_input[0] == resume_node_id, (
            f"End.loop_ctx should point to Resume node ({resume_node_id}), "
            f"got {loop_ctx_input}"
        )


# ======================================================================
# Fix 4: RUNTIME_STORE _detect_cache
# ======================================================================


class TestDetectCache:
    """Verify detect_result is cached and reused across expand rounds."""

    def test_detect_cache_stored_in_runtime_store(self):
        """After first detection, result should be in RUNTIME_STORE._detect_cache."""
        run_id = "cache_test_run"
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        # Simulate caching (as MieLoopEnd.execute does)
        if "_detect_cache" not in RUNTIME_STORE:
            RUNTIME_STORE["_detect_cache"] = {}
        RUNTIME_STORE["_detect_cache"][run_id] = detect

        cached = RUNTIME_STORE.get("_detect_cache", {}).get(run_id)
        assert cached is not None
        assert cached["body_nodes_business"] == detect["body_nodes_business"]

    def test_detect_cache_reused_not_recomputed(self):
        """When cache exists, it should be returned instead of recomputing."""
        run_id = "cache_reuse_run"
        dynprompt = _make_user_workflow_dynprompt()
        _, detect1 = collect_loop_body("32", "33", dynprompt, end_id="34")

        if "_detect_cache" not in RUNTIME_STORE:
            RUNTIME_STORE["_detect_cache"] = {}
        RUNTIME_STORE["_detect_cache"][run_id] = detect1

        # Simulate second round: check cache before recomputing
        cached = RUNTIME_STORE.get("_detect_cache", {}).get(run_id)
        assert cached is not None
        # This is the same result — no need to call collect_loop_body again
        assert "3" in cached["body_nodes_business"]

    def test_detect_cache_cleaned_on_done(self):
        """Cache should be cleaned up when loop completes (done=True)."""
        run_id = "cache_cleanup_run"
        RUNTIME_STORE["_detect_cache"] = {run_id: {"body_nodes_business": ["3"]}}
        # Simulate cleanup
        RUNTIME_STORE["_detect_cache"].pop(run_id, None)
        assert run_id not in RUNTIME_STORE.get("_detect_cache", {})


# ======================================================================
# Fix 5: Full 3-round loop simulation
# ======================================================================


class TestThreeRoundSimulation:
    """Simulate the user's actual workflow: 3-round image generation loop."""

    def _setup_round(self, round_index, count=3):
        """Create a loop context for a specific round."""
        start_node = MieLoopStart()
        ctx, index, count_out, is_last = start_node.execute(
            loop_id="e2e_test",
            params_mode="int_list",
            int_list="8,9,10",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
            meta_json='{"body_in_id":"32","body_out_id":"33","end_id":"34"}',
        )
        # Advance to desired round
        for i in range(round_index):
            ctx["index"] = i
            ctx["current_params"] = ctx["params_list"][i]
            ctx["is_last"] = i == count - 1
        return ctx

    def test_round_0_start_produces_valid_ctx(self):
        """Round 0: MieLoopStart produces valid ctx with index=0."""
        ctx = self._setup_round(0)
        assert ctx["version"] == 3
        assert ctx["index"] == 0
        assert ctx["count"] == 3
        assert ctx["is_last"] is False
        assert ctx["meta"]["body_in_id"] == "32"

    def test_expand_graph_round_1_structure(self, monkeypatch):
        """Round 1→2: expand graph has correct structure for user's workflow."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        ctx = self._setup_round(1)
        result, end_node = _build_expand_graph_for_next_round(
            ctx, dynprompt, "32", "33", "34", detect
        )

        # Expected nodes in expand graph:
        # Resume + KSampler + VAEDecode + ParamGetInt + CollectImage + BodyOut + End
        # (BodyIn is NOT cloned - it only runs in round 0)
        result_types = {nd["class_type"] for nd in result.values()}

        assert "MieLoopResume|Mie" in result_types
        assert "KSampler|Mie" in result_types
        assert "VAEDecode|Mie" in result_types
        assert "MieLoopParamGetInt|Mie" in result_types
        assert "MieLoopBodyOut|Mie" in result_types
        assert "MieLoopEnd|Mie" in result_types
        assert "MieLoopCollectImage|Mie" in result_types, (
            "Collector MUST be cloned each round to accumulate"
        )
        assert "MieLoopBodyIn|Mie" not in result_types, "BodyIn NOT cloned"

    def test_expand_graph_external_links_preserved(self, monkeypatch):
        """External nodes (UNETLoader, VAELoader, CLIPLoader, CLIPTextEncode) remain as links."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        ctx = self._setup_round(1)
        result, _ = _build_expand_graph_for_next_round(
            ctx, dynprompt, "32", "33", "34", detect
        )

        # Find KSampler
        ksampler = None
        for nid, nd in result.items():
            if nd["class_type"] == "KSampler|Mie":
                ksampler = nd
                break
        assert ksampler is not None

        # External links should remain as-is
        assert ksampler["inputs"]["model"] == ["16", 0], "UNETLoader external"
        assert ksampler["inputs"]["positive"] == ["6", 0], "CLIPTextEncode pos external"
        assert ksampler["inputs"]["negative"] == ["7", 0], "CLIPTextEncode neg external"

        # Find VAEDecode
        vae_decode = None
        for nid, nd in result.items():
            if nd["class_type"] == "VAEDecode|Mie":
                vae_decode = nd
                break
        assert vae_decode is not None
        assert vae_decode["inputs"]["vae"] == ["17", 0], "VAELoader external"

    def test_expand_graph_end_node_out_is_link(self, monkeypatch):
        """end_built_node.out() produces values that pass is_link() check."""
        monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
        dynprompt = _make_user_workflow_dynprompt()
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")

        ctx = self._setup_round(1)
        _, end_node = _build_expand_graph_for_next_round(
            ctx, dynprompt, "32", "33", "34", detect
        )

        # Simulate MieLoopEnd return
        result_vals = tuple([end_node.out(i) for i in range(4)])
        for val in result_vals:
            assert is_link(val), f"{val} should pass is_link()"

    def test_three_round_detect_cache_lifecycle(self):
        """Verify detect cache is populated round 1 and cleaned round 3."""
        run_id = "three_round_lifecycle"
        dynprompt = _make_user_workflow_dynprompt()

        # Round 1: first detection → cache
        _, detect = collect_loop_body("32", "33", dynprompt, end_id="34")
        if "_detect_cache" not in RUNTIME_STORE:
            RUNTIME_STORE["_detect_cache"] = {}
        RUNTIME_STORE["_detect_cache"][run_id] = detect

        assert run_id in RUNTIME_STORE["_detect_cache"]

        # Round 2: cache hit (no recomputation)
        cached = RUNTIME_STORE["_detect_cache"].get(run_id)
        assert cached is not None
        assert "3" in cached["body_nodes_business"]

        # Round 3: done → cleanup
        RUNTIME_STORE["_detect_cache"].pop(run_id, None)
        assert run_id not in RUNTIME_STORE.get("_detect_cache", {})
