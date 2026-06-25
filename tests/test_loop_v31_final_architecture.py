import json
import sys

import torch

import loop as loop_module
from loop import MieLoopBodyOut, MieLoopEnd, _ensure_collectors

_conftest_mod = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
FakeGraphBuilder = _conftest_mod.FakeGraphBuilder if _conftest_mod else None


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "final_arch_loop",
        "run_id": "run_final_arch_1",
        "mode": "for_each",
        "index": 0,
        "count": 2,
        "is_last": False,
        "params_list": [{"value": 1}, {"value": 2}],
        "current_params": {"value": 1},
        "state": {},
        "collectors": {"image": {"ref": None, "count": 0}, "text": {"ref": None, "count": 0}},
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    ctx.update(overrides)
    return ctx


def _make_dynprompt():
    return {
        "10": {"class_type": "MieLoopBodyIn|Mie", "inputs": {"loop_ctx": ["1", 0]}},
        "15": {"class_type": "KSampler|Mie", "inputs": {"loop_ctx": ["10", 0]}},
        "20": {"class_type": "MieLoopBodyOut|Mie", "inputs": {"loop_ctx": ["15", 0], "state_json": "{}"}},
        "30": {"class_type": "MieLoopEnd|Mie", "inputs": {"loop_ctx": ["20", 0], "state_json": "{}"}},
    }


def test_ensure_collectors_initializes_image_and_text():
    ctx = {}
    _ensure_collectors(ctx)
    assert ctx["collectors"]["image"] == {"ref": None, "count": 0}
    assert ctx["collectors"]["text"] == {"ref": None, "count": 0}
    assert ctx["collectors"]["json"] == {"ref": None, "count": 0}


def test_collectors_are_not_output_nodes():
    assert getattr(loop_module.MieLoopCollectImage, "OUTPUT_NODE", False) is False
    assert getattr(loop_module.MieLoopCollectText, "OUTPUT_NODE", False) is False
    assert getattr(loop_module.MieLoopCollectJSON, "OUTPUT_NODE", False) is False


def test_loop_end_done_true_returns_only_ctx_and_done():
    ctx = _make_ctx(count=1, is_last=True, params_list=[{"value": 1}], current_params={"value": 1})
    body_ctx = MieLoopBodyOut().execute(ctx, state_json="{}")[0]
    result = MieLoopEnd().execute(body_ctx, "{}")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] is True


def test_loop_end_expand_result_uses_only_two_outputs(monkeypatch):
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    end = MieLoopEnd()
    ctx = _make_ctx()
    result = end.execute(ctx, "{}", dynprompt=_make_dynprompt(), unique_id="30")
    assert isinstance(result, dict)
    assert len(result["result"]) == 2


def test_expand_root_lazy_init(monkeypatch):
    """Plan A: execute pins expand_root into the ctx that flows to the next round.

    execute deep-copies the input ctx and serializes the copy into the expand
    graph's Resume node (loop_ctx_json). expand_root must be present there —
    lazily initialized to end_id on first expand — so the flat prefix stays
    stable for the whole run_id lifetime, including across resume.
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    ctx = _make_ctx()
    assert "expand_root" not in ctx["meta"], "precondition: no expand_root yet"
    end = MieLoopEnd()
    result = end.execute(ctx, "{}", dynprompt=_make_dynprompt(), unique_id="30")
    expand_graph = result["expand"]
    resume = next(
        d for d in expand_graph.values() if d["class_type"] == "MieLoopResume|Mie"
    )
    carried_ctx = json.loads(resume["inputs"]["loop_ctx_json"])
    assert carried_ctx["meta"].get("expand_root") == "30", (
        f"expand_root should be lazily initialized to end_id '30' and carried into "
        f"the next-round ctx, got {carried_ctx['meta'].get('expand_root')}"
    )


def test_resume_preserves_expand_root(monkeypatch):
    """Plan A: Start.resume_loop_ctx keeps expand_root; next End expand uses same prefix.

    Simulates a mid-run interrupt: the saved loop_ctx already carries a pinned
    expand_root from an earlier expand. After resume, flat node ids must stay on
    that root (not re-init from end_id or drift to nested .0.0. paths).
    """
    monkeypatch.setattr(loop_module, "GraphBuilder", FakeGraphBuilder)
    start = loop_module.MieLoopStart()
    end = MieLoopEnd()
    dynprompt = _make_dynprompt()

    def _resume_and_expand(expand_root: str):
        saved_ctx = _make_ctx(
            index=5,
            count=49,
            is_last=False,
            params_list=[{"value": i} for i in range(49)],
            current_params={"value": 5},
        )
        saved_ctx["meta"]["expand_root"] = expand_root
        resume_json = json.dumps(saved_ctx)

        resumed_ctx, index, count, _ = start.execute(
            loop_id="final_arch_loop",
            params_mode="int_list",
            int_list="",
            string_list="",
            json_list="[]",
            initial_state_json="{}",
            resume_loop_ctx=resume_json,
        )
        assert index == 5
        assert count == 49
        assert resumed_ctx["meta"]["expand_root"] == expand_root

        result = end.execute(resumed_ctx, "{}", dynprompt=dynprompt, unique_id="30")
        expand_graph = result["expand"]
        assert expand_graph, "expected expand graph after resume"
        expected_prefix = f"{expand_root}.r6."
        for nid in expand_graph:
            assert nid.startswith(expected_prefix), (
                f"resumed expand must keep expand_root={expand_root!r}, got {nid}"
            )
        resume_node = next(
            d for d in expand_graph.values() if d["class_type"] == "MieLoopResume|Mie"
        )
        carried = json.loads(resume_node["inputs"]["loop_ctx_json"])
        assert carried["meta"]["expand_root"] == expand_root

    # Default: expand_root equals template end_id (typical first-run pin).
    _resume_and_expand("30")
    # Explicit root differs from end_id — must not be overwritten on resume expand.
    _resume_and_expand("999")


def test_finalize_images_and_text_are_independent():
    image = torch.ones((1, 4, 4, 3))
    ctx = _make_ctx(count=1, is_last=True, params_list=[{"value": 1}], current_params={"value": 1})

    ctx = loop_module.MieLoopCollectImage().execute(ctx, image)[0]
    ctx = loop_module.MieLoopCollectText().execute(ctx, "hello")[0]

    final_images = loop_module.MieLoopFinalizeImages().execute(ctx, True)[0]
    final_text = loop_module.MieLoopFinalizeTextList().execute(ctx, True)[0]

    assert tuple(final_images.shape) == (1, 4, 4, 3)
    assert json.loads(final_text) == ["hello"]


def test_cleanup_images_and_text_reset_slots():
    image = torch.ones((1, 2, 2, 3))
    ctx = _make_ctx()
    ctx = loop_module.MieLoopCollectImage().execute(ctx, image)[0]
    ctx = loop_module.MieLoopCollectText().execute(ctx, "hello")[0]

    ctx, image_cleaned = loop_module.MieLoopCleanupImages().execute(ctx)
    ctx, text_cleaned = loop_module.MieLoopCleanupText().execute(ctx)

    assert image_cleaned is True
    assert text_cleaned is True
    assert ctx["collectors"]["image"] == {"ref": None, "count": 0}
    assert ctx["collectors"]["text"] == {"ref": None, "count": 0}
