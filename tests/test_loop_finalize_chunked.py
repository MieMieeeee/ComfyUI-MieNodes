"""Tests for the single-pass disk-streaming merge in MieLoopFinalizeImages.

Locks the no-OOM + capped-file contract for the SCAIL-2 long-run case (30
batches of 81 frames at 704x1280 fp32 ~= 21 GB). Verifies that the merge:
- produces the same tensor as one-shot torch.cat
- keeps every serialized file under the shard cap (2^31-byte zip64 guard)
- writes no intermediate chunk files / np.memmap staging (single pass)
- preserves the input image_*.pt / image_*.shards batches on failure
- handles in-memory and on-disk batch mixes, and mixed old/new formats
- returns the ACTUAL artifact path (merged.pt or merged.shards/)
"""

import glob
import json
import os
import shutil

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


def test_chunked_disk_merge_single_pass_no_memmap_no_chunk_files(tmp_path, monkeypatch):
    # The two-phase machinery (np.memmap staging + _mie_chunk_*.pt
    # intermediates) existed only to feed one giant torch.save. With the
    # byte-capped part writer it must be GONE: no memmap call, no chunk
    # files, no hidden staging leftovers -- and a correct merged result.
    import numpy as _np

    _, paths = _make_disk_batches(tmp_path, count=6, frames_per=4, h=8, w=8)
    out_path = tmp_path / "merged.pt"

    seen = {"memmap": 0}
    real_memmap = _np.memmap

    def spy_memmap(*args, **kwargs):
        seen["memmap"] += 1
        return real_memmap(*args, **kwargs)

    monkeypatch.setattr(_np, "memmap", spy_memmap)
    _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=2, kind="image",
        avoid_oom=True,
    )
    assert seen["memmap"] == 0, "merge must not use np.memmap anymore"
    assert not list(tmp_path.glob("_mie_chunk_*")), "no intermediate chunk files"
    assert not list(tmp_path.glob(".*staging*")), "no staging leftovers"
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert torch.equal(loaded, torch.cat(
        [torch.load(str(p), map_location="cpu", weights_only=False) for p in paths],
        dim=0,
    ))


def test_chunked_disk_merge_forced_cap_emits_parts_under_cap(tmp_path, monkeypatch):
    # The core guarantee the rework was built for: with a forced-tiny cap the
    # merge emits a .shards directory whose parts each stay under the cap and
    # reassemble into exactly torch.cat of the inputs.
    _, paths = _make_disk_batches(tmp_path, count=4, frames_per=3, h=8, w=8)
    monkeypatch.setattr(
        loop_module._tensor_store, "SHARD_FILE_CAP_BYTES", 800
    )  # frame = 8*8*3*4 = 768 B -> 1 frame per part
    out_path = tmp_path / "merged.pt"
    actual = _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=5, kind="image", avoid_oom=True
    )
    assert actual.endswith("merged.shards")
    assert not Path(out_path).exists()
    meta = json.loads((Path(actual) / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["parts"]) == 12  # 4 batches x 3 frames
    for entry in meta["parts"]:
        assert entry["encoding"] == "pt" and entry["frames"] == 1
    back = loop_module._tensor_store.load_tensor(actual)
    expected = torch.cat(
        [torch.load(str(p), map_location="cpu", weights_only=False) for p in paths], dim=0
    )
    assert torch.equal(back, expected)


def test_chunked_disk_merge_accepts_mixed_pt_and_shards_inputs(tmp_path, monkeypatch):
    # An offload dir written across the rework (or recovered manually) can mix
    # legacy image_*.pt files with image_*.shards directories; the merge must
    # stream both part-by-part in order.
    batches = [torch.rand(2, 4, 4, 3) for _ in range(4)]
    paths = []
    for i, t in enumerate(batches):
        base = tmp_path / f"image_{str(i).zfill(10)}"
        if i % 2 == 0:
            p = Path(str(base) + ".pt")
            torch.save(t, str(p))
        else:
            p = Path(loop_module._tensor_store.save_tensor(t, str(base) + ".pt"))
        paths.append(p)
    monkeypatch.setattr(loop_module._tensor_store, "SHARD_FILE_CAP_BYTES", 400)
    out_path = tmp_path / "merged.pt"
    actual = _chunked_disk_merge(
        _to_disk_items(paths), out_path, chunk_size=5, kind="image", avoid_oom=True
    )
    assert torch.equal(
        loop_module._tensor_store.load_tensor(actual), torch.cat(batches, dim=0)
    )

# ---------------------------------------------------------------------------
# MieLoopFinalizeImages contract: the only user-facing knob is avoid_oom.
# ---------------------------------------------------------------------------


def _clean_offload(merged_path):
    """Remove the merged artifact (file or .shards dir) + the run dir."""
    if not merged_path:
        return
    target = Path(merged_path)
    if target.is_dir():
        shutil.rmtree(str(target), ignore_errors=True)
    elif target.is_file():
        os.remove(merged_path)
    run_dir = os.path.dirname(merged_path)
    for leftover in glob.glob(os.path.join(run_dir, "*")):
        if os.path.isdir(leftover):
            shutil.rmtree(leftover, ignore_errors=True)
        else:
            try:
                os.remove(leftover)
            except OSError:
                pass
    try:
        os.rmdir(run_dir)
    except OSError:
        pass


def test_finalize_images_contract_only_avoid_oom_is_optional():
    """The user cannot misconfigure the merge into an OOM-prone state -- the
    node only exposes the avoid_oom BOOLEAN (default True). Everything else
    is hard-coded so a long-running loop can never OOM-crash the merge."""
    spec = MieLoopFinalizeImages.INPUT_TYPES()
    opt = spec.get("optional") or {}
    assert list(opt.keys()) == ["avoid_oom"], (
        f"MieLoopFinalizeImages must expose ONLY avoid_oom as an optional "
        f"input (got {list(opt.keys())}). Hiding chunk_size and "
        f"finalize_to_disk is intentional: the user cannot pick an OOM-prone "
        f"value for a long-running loop."
    )
    assert (opt["avoid_oom"][1]).get("default") is True


def test_finalize_images_auto_derives_merged_path_under_offload_dir(
    sample_loop_ctx, tmp_path
):
    """finalize_to_disk is not exposed -- the node auto-derives the output
    path under the loop's offload dir. This keeps the per-run artefacts
    (image_*.pt + merged.pt) co-located and discoverable."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    expected = []
    ctx = sample_loop_ctx
    for _ in range(4):
        img = torch.rand(2, 4, 4, 3)
        expected.append(img)
        ctx = collect.execute(ctx, img, True, str(tmp_path))[0]

    images, merged_path = finalize.execute(ctx, True)
    try:
        assert merged_path, "merged_path should be populated"
        assert merged_path.endswith("merged.pt")
        # The auto-derived path should live in an offload dir that contains
        # the per-batch image_*.pt files for this run.
        run_dir = os.path.dirname(merged_path)
        leftover_batches = glob.glob(os.path.join(run_dir, "image_*.pt"))
        # Success path cleans up the per-batch .pt; just verify run_dir exists.
        assert os.path.isdir(run_dir)
        assert isinstance(images, torch.Tensor)
        assert torch.equal(images, torch.cat(expected, dim=0))
    finally:
        _clean_offload(merged_path)


def test_finalize_images_merged_path_always_set_on_done(
    sample_loop_ctx, tmp_path
):
    """When done=True and there are batches, merged_path is the auto-derived
    .pt path. When done=False, merged_path is the empty string (loop still
    running). The path is always present and stable so the user can wire
    it into LoadAny|Mie for recovery."""
    finalize = MieLoopFinalizeImages()
    images, merged_path = finalize.execute(sample_loop_ctx, False)
    assert merged_path == ""
    assert images is loop_module.EMPTY_IMAGES


def test_finalize_images_avoid_oom_false_still_loads_image_when_possible(
    sample_loop_ctx, tmp_path
):
    """The avoid_oom=False path is the legacy pre-allocate. For small runs
    that fit in RAM, it should still produce the merged tensor and the
    auto-derived .pt. (Locks that the public contract is identical between
    avoid_oom=True and avoid_oom=False.)"""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    expected = []
    ctx = sample_loop_ctx
    for _ in range(3):
        img = torch.rand(2, 4, 4, 3)
        expected.append(img)
        ctx = collect.execute(ctx, img, True, str(tmp_path))[0]
    images, merged_path = finalize.execute(ctx, True, avoid_oom=False)
    try:
        assert isinstance(images, torch.Tensor)
        assert torch.equal(images, torch.cat(expected, dim=0))
        assert merged_path.endswith("merged.pt")
    finally:
        _clean_offload(merged_path)


def test_finalize_images_preserves_per_batch_files_on_failure(
    sample_loop_ctx, tmp_path, monkeypatch
):
    """Hard contract: if the merge fails, the per-batch image_*.pt files
    MUST survive. This is the only safety net for a multi-hour loop run
    -- losing them is losing the entire run."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    ctx = collect.execute(sample_loop_ctx, torch.rand(2, 4, 4, 3), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, torch.rand(2, 4, 4, 3), True, str(tmp_path))[0]

    real_load = torch.load
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("simulated disk error")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loop_module.torch, "load", flaky)

    with pytest.raises(RuntimeError):
        finalize.execute(ctx, True)
    leftovers = list(Path(tmp_path).glob("image_*.pt"))
    assert leftovers, "input batches must be preserved on failure"



# ---------------------------------------------------------------------------
# capped-shard end-to-end through the public nodes (forced tiny cap)
# ---------------------------------------------------------------------------


def test_finalize_images_forced_cap_end_to_end_shards(
    sample_loop_ctx, tmp_path, monkeypatch
):
    """With the cap forced tiny, the whole public path must shard: collect
    offloads each batch into image_*.shards/, finalize merges to
    merged.shards/, the IMAGE output still equals torch.cat, the merged_path
    probe resolves for LoadImageBatch|Mie, and success cleans the batch dirs.
    """
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    monkeypatch.setattr(loop_module._tensor_store, "SHARD_FILE_CAP_BYTES", 300)
    expected = []
    ctx = sample_loop_ctx
    for _ in range(3):
        img = torch.rand(2, 4, 4, 3)  # 384 B/frame -> every batch shards
        expected.append(img)
        ctx = collect.execute(ctx, img, True, str(tmp_path))[0]
    batch_dirs = list(Path(tmp_path).glob("image_*.shards"))
    assert len(batch_dirs) == 3, "offloaded batches must be shard dirs"

    images, merged_path = finalize.execute(ctx, True)
    try:
        assert merged_path.endswith("merged.shards")
        assert not Path(merged_path).parent.joinpath("merged.pt").exists()
        assert isinstance(images, torch.Tensor)
        assert torch.equal(images, torch.cat(expected, dim=0))
        # The recovery probe LoadImageBatch|Mie uses finds and reassembles it.
        probe = loop_module._tensor_store.find_tensor_path(
            str(Path(merged_path).parent / "merged.pt")
        )
        assert probe == merged_path
        assert torch.equal(
            loop_module._tensor_store.load_tensor(probe), torch.cat(expected, dim=0)
        )
    finally:
        _clean_offload(merged_path)
    # Success path cleans the per-batch shard dirs (dir-aware cleanup).
    assert not list(Path(tmp_path).glob("image_*.shards"))


def test_finalize_images_failure_preserves_sharded_batches(
    sample_loop_ctx, tmp_path, monkeypatch
):
    """The preserve-on-failure contract extends to image_*.shards batch dirs:
    a merge that dies mid-way must leave them on disk for the manual script.
    """
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    monkeypatch.setattr(loop_module._tensor_store, "SHARD_FILE_CAP_BYTES", 300)
    ctx = collect.execute(sample_loop_ctx, torch.rand(2, 4, 4, 3), True, str(tmp_path))[0]
    ctx = collect.execute(ctx, torch.rand(2, 4, 4, 3), True, str(tmp_path))[0]

    real_load = torch.load
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("simulated disk error")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", flaky)
    with pytest.raises(RuntimeError):
        finalize.execute(ctx, True)
    leftovers = list(Path(tmp_path).glob("image_*.shards"))
    assert leftovers, "sharded input batches must be preserved on failure"
    assert not list(Path(tmp_path).glob(".merged*staging*")), "staging must be cleaned"


def test_finalize_images_recovery_hint_points_at_loadimagebatch():
    """Lock the recovery-node fix: the post-merge-load failure hint must say
    LoadImageBatch|Mie (LoadAny|Mie pickles and can never read these files).
    """
    import inspect

    src = inspect.getsource(loop_module)
    assert "LoadImageBatch|Mie" in src
    for line in src.splitlines():
        if "post-merge load failed" in line:
            continue
        if "LoadAny|Mie" in line and "not" not in line and "never" not in line:
            # No stray positive references to LoadAny as the recovery node.
            assert "recover" not in line, line


def test_finalize_images_avoid_oom_false_logs_deprecation(
    sample_loop_ctx, tmp_path, capsys
):
    """avoid_oom is a documented no-op now; flipping it must at least say so
    in the log instead of silently doing nothing."""
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()
    ctx = collect.execute(sample_loop_ctx, torch.rand(2, 4, 4, 3), True, str(tmp_path))[0]
    images, merged_path = finalize.execute(ctx, True, avoid_oom=False)
    try:
        out = capsys.readouterr().out
        assert "avoid_oom=False is deprecated" in out
        assert merged_path.endswith("merged.pt")
    finally:
        _clean_offload(merged_path)
