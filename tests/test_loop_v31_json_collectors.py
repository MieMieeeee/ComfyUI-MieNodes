import json

import loop as loop_module


def _make_ctx(**overrides):
    ctx = {
        "version": 3,
        "loop_id": "json_loop",
        "run_id": "run_json_1",
        "mode": "for_each",
        "index": 0,
        "count": 2,
        "is_last": False,
        "params_list": [{"value": 1}, {"value": 2}],
        "current_params": {"value": 1},
        "state": {},
        "collectors": {
            "image": {"ref": None, "count": 0},
            "text": {"ref": None, "count": 0},
            "json": {"ref": None, "count": 0},
        },
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    ctx.update(overrides)
    return ctx


def test_ensure_collectors_initializes_json_slot():
    ctx = {}
    loop_module._ensure_collectors(ctx)
    assert ctx["collectors"]["json"] == {"ref": None, "count": 0}


def test_collect_and_finalize_json_list():
    collect_json = loop_module.MieLoopCollectJSON()
    finalize_json = loop_module.MieLoopFinalizeJSONList()
    cleanup_json = loop_module.MieLoopCleanupJSON()

    ctx = _make_ctx()
    ctx = collect_json.execute(ctx, '{"seed": 1, "score": 0.5}')[0]
    ctx = collect_json.execute(ctx, '{"seed": 2, "score": 0.8}')[0]

    pending = finalize_json.execute(ctx, False)[0]
    assert json.loads(pending) == []

    ref = ctx["collectors"]["json"]["ref"]
    final = finalize_json.execute(ctx, True)[0]
    assert json.loads(final) == [
        {"seed": 1, "score": 0.5},
        {"seed": 2, "score": 0.8},
    ]
    assert ref not in loop_module.RUNTIME_STORE["collectors"]["json"]

    cleaned_ctx, cleaned = cleanup_json.execute(ctx)
    assert cleaned is False or cleaned is True
    assert cleaned_ctx["collectors"]["json"] == {"ref": None, "count": 0}


def test_json_finalize_and_cleanup_are_excluded_from_body_detection():
    assert loop_module._is_excluded_output_class("MieLoopFinalizeJSONList") is True
    assert loop_module._is_excluded_output_class("MieLoopCleanupJSON") is True


def test_json_finalize_and_cleanup_log(monkeypatch):
    logs = []
    monkeypatch.setattr(loop_module, "mie_log", logs.append)

    collect_json = loop_module.MieLoopCollectJSON()
    finalize_json = loop_module.MieLoopFinalizeJSONList()
    cleanup_json = loop_module.MieLoopCleanupJSON()

    ctx = _make_ctx()
    ctx = collect_json.execute(ctx, '{"seed": 1}')[0]
    finalize_json.execute(ctx, True)
    cleanup_json.execute(ctx)

    assert any("LoopFinalizeJSONList:" in line for line in logs)
    assert any("LoopCleanupJSON:" in line for line in logs)
