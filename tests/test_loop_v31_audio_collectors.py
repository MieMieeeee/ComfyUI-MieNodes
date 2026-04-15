import pytest
import torch

from loop import (
    _ensure_collectors,
    MieLoopCollectAudio,
    MieLoopFinalizeAudio,
    MieLoopCleanupAudio,
    RUNTIME_STORE,
)


def _make_audio(length, sample_rate=24000, channels=2):
    return {
        "waveform": torch.arange(length * channels, dtype=torch.float32).reshape(
            1, channels, length
        ),
        "sample_rate": sample_rate,
    }


def _make_ctx():
    return {
        "version": 3,
        "loop_id": "audio_loop",
        "run_id": "audio_run_1",
        "mode": "for_each",
        "index": 0,
        "count": 3,
        "is_last": False,
        "params_list": [{"value": 0}, {"value": 1}, {"value": 2}],
        "current_params": {"value": 0},
        "state": {},
        "collectors": {
            "image": {"ref": None, "count": 0},
            "text": {"ref": None, "count": 0},
            "json": {"ref": None, "count": 0},
        },
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }


def test_ensure_collectors_adds_audio_slot():
    ctx = {}
    _ensure_collectors(ctx)
    assert ctx["collectors"]["audio"] == {"ref": None, "count": 0}


def test_collect_audio_appends_items_to_runtime_store():
    ctx = _make_ctx()
    node = MieLoopCollectAudio()
    audio1 = _make_audio(4)
    audio2 = _make_audio(6)

    ctx = node.execute(ctx, audio1)[0]
    ref = ctx["collectors"]["audio"]["ref"]
    assert ref is not None
    assert ctx["collectors"]["audio"]["count"] == 1

    ctx = node.execute(ctx, audio2)[0]
    assert ctx["collectors"]["audio"]["ref"] == ref
    assert ctx["collectors"]["audio"]["count"] == 2
    assert len(RUNTIME_STORE["collectors"]["audio"][ref]) == 2


def test_finalize_audio_done_false_returns_empty_audio():
    ctx = _make_ctx()
    node = MieLoopFinalizeAudio()
    audio = node.execute(ctx, False)[0]
    assert audio["sample_rate"] == 0
    assert audio["waveform"].shape[-1] == 0


def test_finalize_audio_concats_in_collection_order():
    ctx = _make_ctx()
    collect = MieLoopCollectAudio()
    finalize = MieLoopFinalizeAudio()
    audio1 = _make_audio(4, sample_rate=24000)
    audio2 = _make_audio(6, sample_rate=24000)

    ctx = collect.execute(ctx, audio1)[0]
    ctx = collect.execute(ctx, audio2)[0]
    ref = ctx["collectors"]["audio"]["ref"]

    merged = finalize.execute(ctx, True)[0]
    assert merged["sample_rate"] == 24000
    assert merged["waveform"].shape == (1, 2, 10)
    assert ref not in RUNTIME_STORE["collectors"]["audio"]


def test_finalize_audio_raises_on_sample_rate_mismatch():
    ctx = _make_ctx()
    collect = MieLoopCollectAudio()
    finalize = MieLoopFinalizeAudio()

    ctx = collect.execute(ctx, _make_audio(4, sample_rate=24000))[0]
    ctx = collect.execute(ctx, _make_audio(6, sample_rate=44100))[0]

    with pytest.raises(ValueError, match="sample rate mismatch"):
        finalize.execute(ctx, True)


def test_cleanup_audio_resets_slot_and_runtime_store():
    ctx = _make_ctx()
    collect = MieLoopCollectAudio()
    cleanup = MieLoopCleanupAudio()

    ctx = collect.execute(ctx, _make_audio(4))[0]
    ref = ctx["collectors"]["audio"]["ref"]
    assert ref in RUNTIME_STORE["collectors"]["audio"]

    cleaned_ctx = cleanup.execute(ctx)[0]
    assert cleaned_ctx["collectors"]["audio"] == {"ref": None, "count": 0}
    assert ref not in RUNTIME_STORE["collectors"]["audio"]
