"""Tests for incremental Finalize merge (Plan: PLAN_LOOP_FINALIZE_INCREMENTAL_MERGE).

Locks the zero-regression contract (incremental merge == one-shot torch.cat) plus
the new failure-path behavior (disk cache preserved on merge failure) and the
start/failed logging.
"""

from pathlib import Path

import pytest
import torch

import loop as loop_module
from loop import (
    MieLoopCollectImage,
    MieLoopFinalizeImages,
    MieLoopCollectAudio,
    MieLoopFinalizeAudio,
    RUNTIME_STORE,
    _merge_tensor_batches_incremental,
)


# ----------------------------------------------------------------------
# Helper: _merge_tensor_batches_incremental (new, generic tensor merge)
# ----------------------------------------------------------------------

def test_merge_tensor_batches_incremental_memory_matches_cat():
    """Memory batches merged incrementally must equal one-shot torch.cat."""
    batches = [
        torch.rand(2, 4, 4, 3),
        torch.rand(3, 4, 4, 3),
        torch.rand(1, 4, 4, 3),
        torch.rand(4, 4, 4, 3),
        torch.rand(2, 4, 4, 3),
    ]

    def load_disk_item(item):  # pragma: no cover - memory path must not call this
        raise AssertionError("load_disk_item must not be called for memory batches")

    def validate_batch(batch, idx, merged):
        if merged is not None:
            assert tuple(batch.shape[1:]) == tuple(merged.shape[1:])

    result = _merge_tensor_batches_incremental(
        batches, load_disk_item=load_disk_item, validate_batch=validate_batch
    )
    assert torch.equal(result, torch.cat(batches, dim=0))
    assert result.shape == (12, 4, 4, 3)
    assert result.dtype == batches[0].dtype


def test_merge_tensor_batches_incremental_loads_disk_items(tmp_path):
    """Disk-cache items are loaded via load_disk_item and catted incrementally."""
    t1 = torch.rand(2, 4, 4, 3)
    t2 = torch.rand(3, 4, 4, 3)
    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save(t1, str(p1))
    torch.save(t2, str(p2))
    items = [{"disk_path": str(p1), "ref": "r1"}, {"disk_path": str(p2), "ref": "r2"}]

    def load_disk_item(item):
        loaded = torch.load(item["disk_path"], map_location="cpu")
        assert isinstance(loaded, torch.Tensor)
        return loaded

    def validate_batch(batch, idx, merged):
        if merged is not None:
            assert tuple(batch.shape[1:]) == tuple(merged.shape[1:])

    result = _merge_tensor_batches_incremental(
        items, load_disk_item=load_disk_item, validate_batch=validate_batch
    )
    assert torch.equal(result, torch.cat([t1, t2], dim=0))
    assert result.shape == (5, 4, 4, 3)


# ----------------------------------------------------------------------
# Images: FinalizeImages incremental merge + cache-preserve-on-failure
# ----------------------------------------------------------------------

def test_finalize_images_many_disk_batches_matches_one_shot(sample_loop_ctx, tmp_path):
    """20 disk batches merge to the same tensor as one-shot cat; success cleans files."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    expected = []
    ctx = sample_loop_ctx
    for _ in range(20):
        img = torch.rand(1, 4, 4, 3)
        expected.append(img)
        ctx = collect.execute(ctx, img, True, str(tmp_path))[0]
    ref = ctx["collectors"]["image"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["image"][ref]]

    merged = finalize.execute(ctx, True)[0]

    assert torch.equal(merged, torch.cat(expected, dim=0))
    assert merged.shape == (20, 4, 4, 3)
    assert all(not p.exists() for p in paths), "success must clean disk cache"


def test_finalize_images_disk_failure_preserves_files(sample_loop_ctx, monkeypatch, tmp_path):
    """If torch.load fails mid-merge, disk cache files are preserved (not cleaned)."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    ctx = collect.execute(sample_loop_ctx, torch.rand(1, 4, 4, 3), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, torch.rand(1, 4, 4, 3), True, str(tmp_path))[0]
    ref = ctx["collectors"]["image"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["image"][ref]]
    assert all(p.exists() for p in paths)

    real_load = torch.load
    calls = {"n": 0}

    def flaky_load(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated load failure")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loop_module.torch, "load", flaky_load)

    with pytest.raises(RuntimeError):
        finalize.execute(ctx, True)

    assert all(p.exists() for p in paths), "disk cache must be preserved on merge failure"


def test_finalize_images_shape_mismatch_preserves_cache(sample_loop_ctx, tmp_path):
    """Shape mismatch raises ValueError and preserves disk cache for manual recovery."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    ctx = collect.execute(sample_loop_ctx, torch.rand(1, 8, 8, 3), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, torch.rand(1, 16, 16, 3), True, str(tmp_path))[0]
    ref = ctx["collectors"]["image"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["image"][ref]]

    with pytest.raises(ValueError):
        finalize.execute(ctx, True)

    assert all(p.exists() for p in paths), "disk cache must be preserved on shape mismatch"


# ----------------------------------------------------------------------
# Audio: FinalizeAudio incremental merge + cache-preserve-on-failure
# ----------------------------------------------------------------------

def _make_audio(length, sample_rate=24000, channels=2):
    return {
        "waveform": torch.arange(length * channels, dtype=torch.float32).reshape(
            1, channels, length
        ),
        "sample_rate": sample_rate,
    }


def test_finalize_audio_disk_failure_preserves_files(monkeypatch, tmp_path):
    """Audio: torch.load failure mid-merge preserves disk cache."""
    collect = MieLoopCollectAudio()
    finalize = MieLoopFinalizeAudio()
    ctx = {
        "version": 3, "loop_id": "audio_inc", "run_id": "audio_inc_run",
        "mode": "for_each", "index": 0, "count": 2, "is_last": False,
        "params_list": [{}, {}], "current_params": {}, "state": {},
        "collectors": {"audio": {"ref": None, "count": 0}},
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    ctx = collect.execute(ctx, _make_audio(4), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, _make_audio(6), True, str(tmp_path))[0]
    ref = ctx["collectors"]["audio"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["audio"][ref]]
    assert all(p.exists() for p in paths)

    real_load = torch.load
    calls = {"n": 0}

    def flaky_load(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated audio load failure")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loop_module.torch, "load", flaky_load)

    with pytest.raises(RuntimeError):
        finalize.execute(ctx, True)

    assert all(p.exists() for p in paths), "audio disk cache must be preserved on failure"


def test_finalize_audio_sample_rate_mismatch_preserves_cache(tmp_path):
    """Audio: sample_rate mismatch raises and preserves disk cache."""
    collect = MieLoopCollectAudio()
    finalize = MieLoopFinalizeAudio()
    ctx = {
        "version": 3, "loop_id": "audio_rate", "run_id": "audio_rate_run",
        "mode": "for_each", "index": 0, "count": 2, "is_last": False,
        "params_list": [{}, {}], "current_params": {}, "state": {},
        "collectors": {"audio": {"ref": None, "count": 0}},
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    ctx = collect.execute(ctx, _make_audio(4, sample_rate=24000), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, _make_audio(6, sample_rate=44100), True, str(tmp_path))[0]
    ref = ctx["collectors"]["audio"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["audio"][ref]]

    with pytest.raises(ValueError):
        finalize.execute(ctx, True)

    assert all(p.exists() for p in paths), "audio disk cache must be preserved on rate mismatch"


def test_finalize_audio_many_disk_batches_matches_one_shot(tmp_path):
    """Audio: 20 disk batches merge to one-shot cat (dim=-1); success cleans files."""
    collect = MieLoopCollectAudio()
    finalize = MieLoopFinalizeAudio()
    ctx = {
        "version": 3, "loop_id": "audio_many", "run_id": "audio_many_run",
        "mode": "for_each", "index": 0, "count": 20, "is_last": False,
        "params_list": [{}] * 20, "current_params": {}, "state": {},
        "collectors": {"audio": {"ref": None, "count": 0}},
        "meta": {"body_in_id": "10", "body_out_id": "20", "end_id": "30"},
    }
    expected = []
    for _ in range(20):
        a = _make_audio(3)
        expected.append(a["waveform"])
        ctx = collect.execute(ctx, a, True, str(tmp_path))[0]
    ref = ctx["collectors"]["audio"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["audio"][ref]]

    merged = finalize.execute(ctx, True)[0]

    assert torch.equal(merged["waveform"], torch.cat(expected, dim=-1))
    assert merged["waveform"].shape == (1, 2, 60)  # 20 * 3
    assert merged["sample_rate"] == 24000
    assert all(not p.exists() for p in paths), "audio success must clean disk cache"


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def test_finalize_logs_cache_dir_on_start(sample_loop_ctx, tmp_path, capsys):
    """FinalizeImages emits a merge-start log with run_id and cache_dir."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    ctx = collect.execute(sample_loop_ctx, torch.rand(1, 4, 4, 3), True, str(tmp_path))[0]
    finalize.execute(ctx, True)
    out = capsys.readouterr().out
    assert "LoopFinalizeMergeStart" in out
    assert str(tmp_path) in out  # cache_dir
    assert sample_loop_ctx["run_id"] in out


def test_finalize_logs_failed_preserves_files_on_error(sample_loop_ctx, monkeypatch, tmp_path, capsys):
    """On merge failure, a LoopFinalizeMergeFailed log names the cache_dir and files are kept."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    ctx = collect.execute(sample_loop_ctx, torch.rand(1, 4, 4, 3), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, torch.rand(1, 4, 4, 3), True, str(tmp_path))[0]
    ref = ctx["collectors"]["image"]["ref"]
    paths = [Path(x["disk_path"]) for x in RUNTIME_STORE["collectors"]["image"][ref]]

    def boom_load(*args, **kwargs):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(loop_module.torch, "load", boom_load)

    with pytest.raises(RuntimeError):
        finalize.execute(ctx, True)

    out = capsys.readouterr().out
    assert "LoopFinalizeMergeFailed" in out
    assert "disk_files_preserved=true" in out
    assert str(tmp_path) in out  # cache_dir printed for manual recovery
    assert all(p.exists() for p in paths)
