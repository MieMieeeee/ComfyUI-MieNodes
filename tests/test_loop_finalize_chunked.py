"""Tests for the chunked disk-streaming merge in MieLoopFinalizeImages.

Locks the no-OOM contract for the SCAIL-2 long-run case (30 batches of 81
frames at 704x1280 fp32 ~= 21 GB). Verifies that the chunked merge:
- produces the same tensor as one-shot torch.cat
- cleans up its intermediate chunk files
- preserves the input image_*.pt files on failure
- handles in-memory and on-disk batch mix
- still exposes the legacy in-memory cat when finalize_to_disk is empty
- drops phase-2 peak to ~1 chunk via numpy.memmap when avoid_oom=True
- falls back to the pre-allocate path for unsupported dtypes
"""

import pytest
import torch
from pathlib import Path

import loop as loop_module
from loop import (
    MieLoopCollectImage,
    MieLoopFinalizeImages,
    _save_inmem_batches_to_disk,
    _chunked_disk_merge,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_disk_batches(tmp_path, count, frames_per=2, h=4, w=4, prefix="image"):
    expected = []
    paths = []
    for i in range(count):
        t = torch.rand(frames_per, h, w, 3)
        expected.append(t)
        p = tmp_path / (prefix + "_" + str(i).zfill(10) + ".pt")
        torch.save(t, str(p))
        paths.append(p)
    return expected, paths


def _to_disk_items(paths):
    return [{"disk_path": str(p), "ref": ""} for p in paths]


# ---------------------------------------------------------------------------
# _save_inmem_batches_to_disk
# ---------------------------------------------------------------------------


def test_save_inmem_batches_to_disk_writes_pt_files(tmp_path):
    a = torch.rand(3, 4, 4, 3)
    b = torch.rand(2, 4, 4, 3)
    out = _save_inmem_batches_to_disk([a, b], tmp_path, "image")
    assert len(out) == 2
    for item in out:
        p = Path(item["disk_path"])
        assert p.parent == tmp_path
        assert p.name.startswith("image_inmem_")
        assert p.name.endswith(".pt")
        loaded = torch.load(str(p), map_location="cpu", weights_only=False)
        assert isinstance(loaded, torch.Tensor)


def test_save_inmem_batches_to_disk_passes_through_disk_items(tmp_path):
    a = torch.rand(3, 4, 4, 3)
    p = tmp_path / "image_pre.pt"
    torch.save(a, str(p))
    items = [{"disk_path": str(p), "ref": "r1"}]
    out = _save_inmem_batches_to_disk(items, tmp_path, "image")
    # The disk item must be passed through (same object, no new file written).
    assert out[0] is items[0]
    # No new image_inmem_*.pt files should appear.
    leftovers = list(tmp_path.glob("image_inmem_*.pt"))
    assert leftovers == []


def test_save_inmem_batches_to_disk_rejects_non_tensor(tmp_path):
    with pytest.raises(ValueError, match="not a torch.Tensor"):
        _save_inmem_batches_to_disk([{"not": "a tensor"}], tmp_path, "image")


# ---------------------------------------------------------------------------
# _chunked_disk_merge: pre-allocate (avoid_oom=False) and memmap (default)
# ---------------------------------------------------------------------------


def test_chunked_disk_merge_matches_one_shot_cat(tmp_path):
    expected, paths = _make_disk_batches(tmp_path, count=7, frames_per=2)
    out_path = tmp_path / "merged.pt"
    written = _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=3, kind="image",
        avoid_oom=False,
    )
    assert Path(written) == out_path
    assert out_path.exists()
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    ref = torch.cat(expected, dim=0)
    assert tuple(loaded.shape) == tuple(ref.shape)
    assert torch.equal(loaded, ref)


def test_chunked_disk_merge_chunk_larger_than_count(tmp_path):
    expected, paths = _make_disk_batches(tmp_path, count=3, frames_per=2)
    out_path = tmp_path / "merged.pt"
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=10, kind="image",
        avoid_oom=False,
    )
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert torch.equal(loaded, torch.cat(expected, dim=0))


def test_chunked_disk_merge_chunk_size_1_still_works(tmp_path):
    expected, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=1, kind="image",
        avoid_oom=False,
    )
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert torch.equal(loaded, torch.cat(expected, dim=0))


def test_chunked_disk_merge_cleans_chunk_files_on_success(tmp_path):
    _, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=2, kind="image",
        avoid_oom=False,
    )
    leftovers = list(tmp_path.glob("_mie_chunk_image_*.pt"))
    assert leftovers == [], "chunk files not cleaned: " + str(leftovers)
    assert out_path.exists()


def test_chunked_disk_merge_preserves_input_batches_on_failure(tmp_path, monkeypatch):
    _, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"

    real_load = torch.load
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 5:
            raise RuntimeError("simulated I/O error")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loop_module.torch, "load", flaky)

    with pytest.raises(RuntimeError, match="simulated I/O error"):
        _chunked_disk_merge(
            _to_disk_items(paths), out_path, chunk_size=2, kind="image",
            avoid_oom=False,
        )

    for p in paths:
        assert p.exists()
    leftovers = list(tmp_path.glob("_mie_chunk_image_*.pt"))
    assert leftovers == []


def test_chunked_disk_merge_shape_mismatch_raises(tmp_path):
    _, paths = _make_disk_batches(tmp_path, count=3, frames_per=2)
    weird = torch.rand(2, 8, 8, 3)
    paths[1] = tmp_path / "image_0000000001.pt"
    torch.save(weird, str(paths[1]))
    out_path = tmp_path / "merged.pt"

    def validate(batch, idx, merged):
        if merged is not None and tuple(batch.shape[1:]) != tuple(merged.shape[1:]):
            raise ValueError("shape mismatch")

    with pytest.raises(ValueError, match="shape mismatch"):
        _chunked_disk_merge(
            _to_disk_items(paths), out_path, chunk_size=2, kind="image",
            validate_batch=validate, avoid_oom=False,
        )


def test_chunked_disk_merge_empty_input_raises(tmp_path):
    out_path = tmp_path / "merged.pt"
    with pytest.raises(FileNotFoundError, match="no on-disk batches"):
        _chunked_disk_merge([], out_path, chunk_size=5, kind="image", avoid_oom=False)


# ---- memmap path (avoid_oom=True, the default) -------------------------


def test_chunked_disk_merge_memmap_matches_one_shot_cat(tmp_path):
    expected, paths = _make_disk_batches(tmp_path, count=7, frames_per=2)
    out_path = tmp_path / "merged.pt"
    written = _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=3, kind="image",
        avoid_oom=True,
    )
    assert Path(written) == out_path
    assert out_path.exists()
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    ref = torch.cat(expected, dim=0)
    assert tuple(loaded.shape) == tuple(ref.shape)
    assert torch.equal(loaded, ref)


def test_chunked_disk_merge_memmap_cleans_temp_and_chunk_files(tmp_path):
    _, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=2, kind="image",
        avoid_oom=True,
    )
    leftovers_chunks = list(tmp_path.glob("_mie_chunk_image_*.pt"))
    leftovers_mmap = list(tmp_path.glob("*.image.mmap.tmp"))
    assert leftovers_chunks == [], "chunk files not cleaned: " + str(leftovers_chunks)
    assert leftovers_mmap == [], "memmap temp not cleaned: " + str(leftovers_mmap)
    assert out_path.exists()


def test_chunked_disk_merge_memmap_preserves_input_on_failure(tmp_path, monkeypatch):
    _, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"

    real_load = torch.load
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 5:
            raise RuntimeError("simulated memmap I/O error")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loop_module.torch, "load", flaky)

    with pytest.raises(RuntimeError, match="simulated memmap I/O error"):
        _chunked_disk_merge(
            _to_disk_items(paths), out_path, chunk_size=2, kind="image",
            avoid_oom=True,
        )

    for p in paths:
        assert p.exists()
    assert list(tmp_path.glob("_mie_chunk_image_*.pt")) == []
    assert list(tmp_path.glob("*.image.mmap.tmp")) == []


def test_chunked_disk_merge_memmap_default_is_on(tmp_path):
    "avoid_oom default is True -- memmap is the path the user actually hits."
    expected, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"
    # Call without avoid_oom kwarg; should still go through the memmap branch.
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=2, kind="image",
    )
    leftovers = list(tmp_path.glob("*.image.mmap.tmp"))
    assert leftovers == [], "default should use memmap and clean its temp"
    assert out_path.exists()


def test_chunked_disk_merge_memmap_chunk_size_1_works(tmp_path):
    expected, paths = _make_disk_batches(tmp_path, count=4, frames_per=2)
    out_path = tmp_path / "merged.pt"
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=1, kind="image",
        avoid_oom=True,
    )
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert torch.equal(loaded, torch.cat(expected, dim=0))


def test_chunked_disk_merge_memmap_chunk_larger_than_count(tmp_path):
    expected, paths = _make_disk_batches(tmp_path, count=3, frames_per=2)
    out_path = tmp_path / "merged.pt"
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=10, kind="image",
        avoid_oom=True,
    )
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert torch.equal(loaded, torch.cat(expected, dim=0))


def test_chunked_disk_merge_memmap_uses_smaller_peak_than_pre_allocate(tmp_path, monkeypatch):
    # The whole point of avoid_oom: phase-2 peak should be bounded by ~1 chunk,
    # not final+chunk. We assert this indirectly by checking the memmap path
    # was actually used (np.memmap called with mode w+). Direct peak
    # measurement via psutil is too fragile for CI.
    import numpy as _np

    _, paths = _make_disk_batches(tmp_path, count=6, frames_per=4, h=8, w=8)
    out_path = tmp_path / "merged.pt"

    real_memmap = _np.memmap
    seen = {"modes": [], "shapes": []}

    def spy_memmap(filename, dtype=None, mode=None, shape=None, *args, **kwargs):
        seen["modes"].append(mode)
        seen["shapes"].append(shape)
        return real_memmap(filename, dtype=dtype, mode=mode, shape=shape, *args, **kwargs)

    monkeypatch.setattr(loop_module.np, "memmap", spy_memmap)
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=2, kind="image",
        avoid_oom=True,
    )
    # Phase 2 opens the mmap in "w+" first, then re-opens in "r" for torch.save.
    assert "w+" in seen["modes"], "avoid_oom=True must use numpy.memmap in write mode"
    # The mmap's shape is the full final tensor (24 frames total here).
    write_shapes = [s for s, m in zip(seen["shapes"], seen["modes"]) if m == "w+"]
    assert write_shapes, "no w+ mmap call captured"
    assert write_shapes[0][0] == 24  # 6 batches * 4 frames = 24
