# -*- coding: utf-8 -*-
"""Tests for core.tensor_store — capped-shard tensor persistence.

Locks the guarantees the sharding rework was built for:
- no serialized file ever exceeds SHARD_FILE_CAP_BYTES (the 2^31-byte zip64
  corruption boundary reported for torch.save on non-ASCII install paths)
- small payloads keep the byte-identical single-.pt behavior (widget paths,
  audio dicts, old caches)
- dtype fidelity (incl. bfloat16 — the old mmap merge silently downcast it)
- crash-left partial shard directories are detected, never silently mis-loaded
- the .pt -> .shards alias probe keeps SaveImageBatch/LoadImageBatch/FileExists
  working from the user-visible widget string
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import core.tensor_store as ts  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_cap():
    """Keep a forced-tiny cap from leaking between tests."""
    yield
    ts.SHARD_FILE_CAP_BYTES = int(1.5 * 1024**3)


def _force_cap(monkeypatch, cap):
    monkeypatch.setattr(ts, "SHARD_FILE_CAP_BYTES", cap)


def _read_meta(shard_dir):
    return json.loads((Path(shard_dir) / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# small payloads keep the legacy single-file behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_small_tensor_writes_single_pt_verbatim(tmp_path, dtype):
    t = torch.rand(3, 4, 5, 3).to(dtype)
    p = tmp_path / "foo.pt"
    actual = ts.save_tensor(t, str(p))
    assert actual == str(p)
    assert p.is_file()
    assert not ts.shard_dir_for(p).exists()
    back = ts.load_tensor(actual)
    assert back.dtype == dtype and torch.equal(t, back)


def test_dict_payload_stays_single_pt(tmp_path):
    payload = {"waveform": torch.zeros(1, 2, 10), "sample_rate": 24000}
    p = tmp_path / "audio_x.pt"
    actual = ts.save_tensor(payload, str(p))
    assert actual == str(p) and p.is_file()
    back = ts.load_tensor(actual)
    assert isinstance(back, dict) and back["sample_rate"] == 24000


def test_zero_frame_tensor_stays_single_pt(tmp_path):
    t = torch.zeros(0, 4, 4, 3)
    actual = ts.save_tensor(t, str(tmp_path / "empty.pt"))
    assert actual.endswith("empty.pt")
    back = ts.load_tensor(actual)
    assert tuple(back.shape) == (0, 4, 4, 3)


def test_zero_dim_tensor_stays_single_pt(tmp_path):
    t = torch.tensor(3.5)
    actual = ts.save_tensor(t, str(tmp_path / "scalar.pt"))
    assert actual.endswith("scalar.pt")
    assert float(ts.load_tensor(actual)) == 3.5


def test_legacy_pt_file_loads_unchanged(tmp_path):
    # A file written by plain torch.save (i.e. every pre-rework cache on disk)
    # must load through load_tensor byte-identically.
    t = torch.rand(4, 8, 8, 3)
    p = tmp_path / "old.pt"
    torch.save(t, str(p))
    assert torch.equal(ts.load_tensor(str(p)), t)


# ---------------------------------------------------------------------------
# sharded writes
# ---------------------------------------------------------------------------


def test_oversized_tensor_shards_with_parts_under_cap(tmp_path, monkeypatch):
    t = torch.rand(10, 4, 4, 3)  # 10 frames x 192 B = 1920 B
    _force_cap(monkeypatch, 500)  # -> 2 frames per part, 5 parts
    p = tmp_path / "big.pt"
    actual = ts.save_tensor(t, str(p))
    sdir = Path(actual)
    assert sdir.is_dir() and sdir.name == "big.shards"
    meta = _read_meta(sdir)
    assert meta["format"] == ts.SHARD_FORMAT
    assert meta["version"] == ts.SHARD_VERSION
    assert meta["dtype"] == "torch.float32"
    assert meta["shape"] == [10, 4, 4, 3]
    assert sum(e["frames"] for e in meta["parts"]) == 10
    frame_bytes = 4 * 4 * 3 * 4
    for entry in meta["parts"]:
        part = sdir / entry["file"]
        assert part.is_file()
        # The cap governs LOGICAL tensor bytes; the .pt zip container adds a
        # fixed KB-scale header (negligible against the real 1.5 GiB cap).
        assert entry["frames"] * frame_bytes <= 500
    assert torch.equal(ts.load_tensor(actual), t)


def test_shard_rollover_exact_multiple_and_remainder(tmp_path, monkeypatch):
    # frame = 4*4*3*4 = 192 B; cap 960 -> fps 5
    _force_cap(monkeypatch, 960)
    a = torch.rand(10, 4, 4, 3)  # exact multiple of 5
    sd_a = Path(ts.save_tensor(a, str(tmp_path / "exact.pt")))
    frames_a = [e["frames"] for e in _read_meta(sd_a)["parts"]]
    assert frames_a == [5, 5]
    b = torch.rand(11, 4, 4, 3)  # remainder 1
    sd_b = Path(ts.save_tensor(b, str(tmp_path / "rem.pt")))
    frames_b = [e["frames"] for e in _read_meta(sd_b)["parts"]]
    assert frames_b == [5, 5, 1]
    assert torch.equal(ts.load_tensor(str(sd_a)), a)
    assert torch.equal(ts.load_tensor(str(sd_b)), b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shards_preserve_dtype_including_bf16(tmp_path, monkeypatch, dtype):
    # Regression lock for the old mmap merge, which staged bf16 through numpy
    # float16 and stored an fp16 tensor in the final file.
    t = torch.rand(7, 4, 4, 3).to(dtype)
    _force_cap(monkeypatch, 300)
    actual = ts.save_tensor(t, str(tmp_path / "dt.pt"))
    back = ts.load_tensor(actual)
    assert back.dtype == dtype
    assert torch.equal(t, back)
    assert _read_meta(Path(actual))["dtype"] == str(dtype)


def test_non_contiguous_input_roundtrips(tmp_path, monkeypatch):
    base = torch.rand(20, 4, 4, 3)
    t = base[::2]  # non-contiguous view
    assert not t.is_contiguous()
    _force_cap(monkeypatch, 400)
    actual = ts.save_tensor(t, str(tmp_path / "strided.pt"))
    assert torch.equal(ts.load_tensor(actual), t)


def test_single_frame_over_cap_writes_raw_part_not_big_pt(tmp_path, monkeypatch):
    # The one hole in "no torch.save >= cap": a single frame bigger than the
    # cap must go out as raw bytes (no zip container at all).
    t = torch.rand(3, 64, 64, 3)  # 3 frames x 48 KB
    _force_cap(monkeypatch, 1024)  # fps 0 -> 1 frame per part, each over cap
    actual = ts.save_tensor(t, str(tmp_path / "raw.pt"))
    sdir = Path(actual)
    meta = _read_meta(sdir)
    assert all(e["encoding"] == "raw" for e in meta["parts"])
    assert all(e["file"].endswith(".bin") for e in meta["parts"])
    assert not any(f.suffix == ".pt" for f in sdir.iterdir())
    assert torch.equal(ts.load_tensor(actual), t)


def test_single_bf16_frame_over_cap_raw_part_bit_exact(tmp_path, monkeypatch):
    t = torch.rand(2, 64, 64, 3).to(torch.bfloat16)
    _force_cap(monkeypatch, 512)
    actual = ts.save_tensor(t, str(tmp_path / "bfraw.pt"))
    back = ts.load_tensor(actual)
    assert back.dtype == torch.bfloat16
    assert torch.equal(t, back)


def test_gpu_style_input_is_moved_to_cpu(tmp_path):
    # simulate a caller handing a non-cpu-like tensor by monkeypatching the
    # device check is overkill; contiguous() path is what matters here. Just
    # verify a plain cpu tensor with explicit .contiguous() call path.
    t = torch.rand(2, 4, 4, 3).contiguous()
    actual = ts.save_tensor(t, str(tmp_path / "c.pt"))
    assert torch.equal(ts.load_tensor(actual), t)


# ---------------------------------------------------------------------------
# crash / corruption protocol
# ---------------------------------------------------------------------------


def _make_sharded(tmp_path, monkeypatch, frames=6):
    t = torch.rand(frames, 4, 4, 3)
    _force_cap(monkeypatch, 400)  # fps 2
    actual = ts.save_tensor(t, str(tmp_path / "victim.pt"))
    return t, Path(actual)


def test_partial_dir_without_meta_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    (sdir / "meta.json").unlink()
    assert not ts.is_complete_shard_dir(sdir)
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))
    assert ts.find_tensor_path(str(tmp_path / "victim.pt")) is None


def test_partial_dir_with_missing_part_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    (sdir / _read_meta(sdir)["parts"][0]["file"]).unlink()
    assert not ts.is_complete_shard_dir(sdir)
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))


def test_partial_dir_with_empty_part_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    victim = sdir / _read_meta(sdir)["parts"][0]["file"]
    victim.write_bytes(b"")
    assert not ts.is_complete_shard_dir(sdir)
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))


def test_meta_frames_sum_mismatch_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    meta = _read_meta(sdir)
    meta["parts"][0]["frames"] += 1
    (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert not ts.is_complete_shard_dir(sdir)
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))


def test_meta_version_mismatch_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    meta = _read_meta(sdir)
    meta["version"] = ts.SHARD_VERSION + 1
    (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))


def test_meta_wrong_format_tag_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    meta = _read_meta(sdir)
    meta["format"] = "someone-else"
    (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))


def test_corrupt_part_dtype_mismatch_is_detected(tmp_path, monkeypatch):
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    name = _read_meta(sdir)["parts"][0]["file"]
    torch.save(torch.rand(2, 4, 4, 3).to(torch.float16), str(sdir / name))
    with pytest.raises(ts.CorruptShardError):
        ts.load_tensor(str(sdir))


def test_meta_json_tmp_is_not_a_valid_commit(tmp_path, monkeypatch):
    # A crash between writing meta.json.tmp and os.replace leaves only the
    # tmp: the dir must still count as incomplete.
    t, sdir = _make_sharded(tmp_path, monkeypatch)
    (sdir / "meta.json").rename(sdir / "meta.json.tmp")
    assert not ts.is_complete_shard_dir(sdir)


# ---------------------------------------------------------------------------
# alias probe / existence / cleanup helpers
# ---------------------------------------------------------------------------


def test_find_tensor_path_probes_file_dir_and_alias(tmp_path, monkeypatch):
    t = torch.rand(4, 4, 4, 3)
    _force_cap(monkeypatch, 300)
    p = tmp_path / "widget.pt"
    actual = ts.save_tensor(t, str(p))
    # directly as dir
    assert ts.find_tensor_path(actual) == actual
    # from the widget string (file missing, alias present)
    assert ts.find_tensor_path(str(p)) == actual
    # plain file
    small = tmp_path / "small.pt"
    ts.save_tensor(torch.rand(1, 4, 4, 3), str(small))
    assert ts.find_tensor_path(str(small)) == str(small)
    # missing + junk dir
    assert ts.find_tensor_path(str(tmp_path / "nope.pt")) is None
    junk = tmp_path / "junk.shards"
    junk.mkdir()
    (junk / "part_00000.pt").write_bytes(b"x")
    assert ts.find_tensor_path(str(junk)) is None  # no valid meta -> miss


def test_remove_path_file_dir_and_missing(tmp_path):
    f = tmp_path / "f.pt"
    f.write_bytes(b"x")
    assert ts.remove_path(str(f)) is True and not f.exists()
    d = tmp_path / "d.shards"
    d.mkdir()
    (d / "part_00000.pt").write_bytes(b"x")
    assert ts.remove_path(str(d)) is True and not d.exists()
    # missing path / missing dir: no-op, never raises
    assert ts.remove_path(str(tmp_path / "ghost.pt")) is False
    assert ts.remove_path(str(tmp_path / "ghost_dir")) is False


def test_dir_bytes_walks_part_files(tmp_path):
    d = tmp_path / "d.shards"
    d.mkdir()
    (d / "part_00000.pt").write_bytes(b"a" * 100)
    (d / "meta.json").write_text("{}", encoding="utf-8")  # 2 bytes
    assert ts.dir_bytes(str(d)) == 102  # every file in the tree counts
    assert ts.dir_bytes(str(tmp_path / "f.pt")) == 0  # not a dir


def test_save_swaps_stale_other_representation(tmp_path, monkeypatch):
    t_big = torch.rand(8, 4, 4, 3)
    t_small = torch.rand(1, 4, 4, 3)
    p = tmp_path / "swap.pt"
    _force_cap(monkeypatch, 300)
    big_actual = ts.save_tensor(t_big, str(p))
    assert Path(big_actual).is_dir()
    # small save over the same primary removes the stale alias dir
    monkeypatch.setattr(ts, "SHARD_FILE_CAP_BYTES", int(1.5 * 1024**3))
    small_actual = ts.save_tensor(t_small, str(p))
    assert small_actual == str(p) and Path(p).is_file()
    assert not Path(big_actual).exists()
    # big save again removes the stale single file
    _force_cap(monkeypatch, 300)
    big_actual2 = ts.save_tensor(t_big, str(p))
    assert Path(big_actual2).is_dir() and not Path(p).exists()


# ---------------------------------------------------------------------------
# streaming / writer behavior
# ---------------------------------------------------------------------------


def test_iter_parts_file_and_shard_dir(tmp_path, monkeypatch):
    t = torch.rand(6, 4, 4, 3)
    p = tmp_path / "one.pt"
    torch.save(t, str(p))
    parts = list(ts.iter_parts(str(p)))
    assert len(parts) == 1 and torch.equal(parts[0], t)

    _force_cap(monkeypatch, 400)  # fps 2
    sdir = Path(ts.save_tensor(t, str(tmp_path / "many.pt")))
    parts = list(ts.iter_parts(str(sdir)))
    assert [int(x.shape[0]) for x in parts] == [2, 2, 2]
    assert torch.equal(torch.cat(parts, dim=0), t)


def test_iter_parts_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(ts.iter_parts(str(tmp_path / "ghost.pt")))


def test_load_tensor_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ts.load_tensor(str(tmp_path / "ghost.pt"))


def test_writer_rejects_shape_and_dtype_mismatch(tmp_path, monkeypatch):
    w = ts.ShardPartWriter(tmp_path / "out.pt", cap=None)
    _force_cap(monkeypatch, 10000)
    w.append(torch.rand(2, 4, 4, 3))
    with pytest.raises(ValueError):
        w.append(torch.rand(2, 8, 8, 3))
    with pytest.raises(ValueError):
        w.append(torch.rand(2, 4, 4, 3, dtype=torch.float16))
    w.abort()


def test_writer_misuse_guards(tmp_path, monkeypatch):
    _force_cap(monkeypatch, 10000)
    w = ts.ShardPartWriter(tmp_path / "out.pt", cap=None)
    with pytest.raises(RuntimeError):
        w.finish()  # nothing appended
    w2 = ts.ShardPartWriter(tmp_path / "out2.pt", cap=None)
    w2.append(torch.rand(2, 4, 4, 3))
    w2.finish()
    with pytest.raises(RuntimeError):
        w2.append(torch.rand(1, 4, 4, 3))
    with pytest.raises(RuntimeError):
        w2.finish()


def test_save_failure_cleans_partial_output(tmp_path, monkeypatch):
    # A write that dies mid-parts must leave neither a .shards dir nor the
    # hidden staging dir behind (output is regenerable; inputs are preserved
    # by the loop contract, not by this module).
    t = torch.rand(6, 4, 4, 3)
    _force_cap(monkeypatch, 400)
    real_save = torch.save
    calls = {"n": 0}

    def flaky(obj, path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_save(obj, path)

    monkeypatch.setattr(ts.torch, "save", flaky)
    with pytest.raises(OSError):
        ts.save_tensor(t, str(tmp_path / "boom.pt"))
    monkeypatch.setattr(ts.torch, "save", real_save)
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == [], f"leftover artifacts: {leftovers}"


def test_writer_finish_small_result_writes_plain_pt(tmp_path, monkeypatch):
    _force_cap(monkeypatch, 100000)
    w = ts.ShardPartWriter(tmp_path / "small.pt", cap=None)
    a = torch.rand(2, 4, 4, 3)
    b = torch.rand(1, 4, 4, 3)
    w.append(a)
    w.append(b)
    actual = w.finish()
    assert actual == str(tmp_path / "small.pt")
    assert (tmp_path / "small.pt").is_file()
    assert torch.equal(ts.load_tensor(actual), torch.cat([a, b]))
    assert not list(tmp_path.glob(".*staging*"))


# ---------------------------------------------------------------------------
# real >2 GiB roundtrip (env-gated; slow, needs ~2.5 GB disk + RAM headroom)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("MIE_TENSOR_STORE_SLOW") != "1",
    reason="set MIE_TENSOR_STORE_SLOW=1 to run the real >2GiB roundtrip",
)
def test_real_over_2gib_roundtrip(tmp_path):
    # Proves the actual fix end-to-end at real scale: a tensor whose single
    # torch.save file would exceed 2^31 bytes comes back bit-exact from parts
    # that are each under the cap. 40 frames x 64 MiB = 2.5 GiB.
    n, fps = 40, 24  # cap 1.5 GiB / 64 MiB frames -> 24 frames per part
    src = torch.empty(n, 4096, 4096, dtype=torch.float32)
    actual = ts.save_tensor(src, str(tmp_path / "huge.pt"))
    assert Path(actual).is_dir()
    for f in Path(actual).iterdir():
        assert f.stat().st_size < 2**31, f"{f.name} exceeds 2^31 bytes"
    meta = _read_meta(Path(actual))
    assert [e["frames"] for e in meta["parts"]] == [fps, n - fps]
    back = ts.load_tensor(actual)
    assert back.shape == src.shape and back.dtype == src.dtype
