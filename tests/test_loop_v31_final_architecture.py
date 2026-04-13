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
