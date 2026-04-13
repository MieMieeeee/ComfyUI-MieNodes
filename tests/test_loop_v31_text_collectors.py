import json

import torch

import loop as loop_module


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "text_loop",
        "run_id": "run_text_1",
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


def test_runtime_store_uses_collectors_namespace():
    assert "collectors" in loop_module.RUNTIME_STORE
    assert "image" in loop_module.RUNTIME_STORE["collectors"]


def test_collect_image_uses_unified_collector_store():
    collector = loop_module.MieLoopCollectImage()
    ctx = _make_ctx()
    image = torch.ones((1, 4, 4, 3))
    out_ctx = collector.execute(ctx, image)[0]
    ref = out_ctx["collectors"]["image"]["ref"]
    assert ref in loop_module.RUNTIME_STORE["collectors"]["image"]
    assert out_ctx["collectors"]["image"]["count"] == 1


def test_collect_and_finalize_text_list():
    collect_text = loop_module.MieLoopCollectText()
    finalize_text = loop_module.MieLoopFinalizeTextList()
    cleanup_text = loop_module.MieLoopCleanupText()
    ctx = _make_ctx()

    ctx = collect_text.execute(ctx, "hello")[0]
    ctx = collect_text.execute(ctx, "world")[0]

    pending_json = finalize_text.execute(ctx, False)[0]
    assert json.loads(pending_json) == []
    ref = ctx["collectors"]["text"]["ref"]
    assert ref in loop_module.RUNTIME_STORE["collectors"]["text"]

    text_json = finalize_text.execute(ctx, True)[0]
    assert json.loads(text_json) == ["hello", "world"]
    assert ref not in loop_module.RUNTIME_STORE["collectors"]["text"]

    cleaned_ctx, cleaned = cleanup_text.execute(ctx)
    assert cleaned is False or cleaned is True
    assert isinstance(cleaned_ctx, dict)


def test_text_finalize_and_cleanup_are_excluded_from_body_detection():
    assert loop_module._is_excluded_output_class("MieLoopFinalizeTextList") is True
    assert loop_module._is_excluded_output_class("MieLoopCleanupText") is True


def test_text_finalize_and_cleanup_log(monkeypatch):
    logs = []
    monkeypatch.setattr(loop_module, "mie_log", logs.append)

    collect_text = loop_module.MieLoopCollectText()
    finalize_text = loop_module.MieLoopFinalizeTextList()
    cleanup_text = loop_module.MieLoopCleanupText()

    ctx = _make_ctx()
    ctx = collect_text.execute(ctx, "hello")[0]
    finalize_text.execute(ctx, True)
    cleanup_text.execute(ctx)

    assert any("LoopFinalizeTextList:" in line for line in logs)
    assert any("LoopCleanupText:" in line for line in logs)
