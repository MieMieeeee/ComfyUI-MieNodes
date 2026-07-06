#!/usr/bin/env python3
"""Chunked + memmap image/audio batch merge shared by the loop node and recovery tools.

This module is intentionally free of any ComfyUI / custom_nodes imports so
that both the live ComfyUI loop node (nodes/loop/loop.py) and the offline
recovery script (scripts/manual_merge_offloaded_images.py) can call the
exact same code path. The previous live-only torch.cat loop in the manual
merge script OOMed on long-running loops (SCAIL-2 case: 30 batches of 81
frames at 704x1280 fp32 ~= 25 GB final). Going through this module keeps
peak memory bounded by the chunked+memmap path (~7 GB end-to-end for the
same workload).

Public API:
  - chunked_disk_merge(disk_items, out_path, *, chunk_size=5, kind="image",
                          validate_batch=None, log_progress=None, avoid_oom=True)
  - build_disk_item(path) -> {"disk_path": str, "ref": ""}
  - is_disk_cache_item(item) -> bool
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch


# Torch -> numpy dtype map for the memmap phase-2 path. bfloat16 / float16
# round-trip through float16 in numpy (the small precision difference is
# acceptable for an intermediate staging tensor -- the final torch.save
# re-encodes to the original dtype on disk).
_TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}


PathLike = Union[str, Path]
DiskItem = dict  # {"disk_path": str, ...}


def is_disk_cache_item(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("disk_path"), str)
        and item.get("disk_path") != ""
    )


def build_disk_item(path: PathLike) -> DiskItem:
    return {"disk_path": str(path), "ref": ""}


def chunked_disk_merge(
    disk_items,
    out_path: PathLike,
    *,
    chunk_size: int = 5,
    kind: str = "image",
    validate_batch: Optional[Callable] = None,
    log_progress: Optional[Callable] = None,
    avoid_oom: bool = True,
) -> str:
    """Stream-merge on-disk per-batch .pt files to a single .pt at out_path.

    Two-phase to keep peak memory bounded:

    Phase 1: build intermediate chunk files. Each chunk cat-merges up to
    ``chunk_size`` input .pt files into a single chunk tensor and writes it to
    ``<run_dir>/_mie_chunk_{kind}_NNNN.pt``. Peak here is roughly
    ``2 * chunk_size * per_batch_bytes`` (load + cat). With the SCAIL-2 30-batch
    case (chunk_size=5) that is ~7 GB peak.

    Phase 2 has two strategies selected by ``avoid_oom``:

    - ``avoid_oom=False`` (legacy): pre-allocate the final tensor on CPU and
      copy each chunk into it. Peak is ``final_bytes + max_chunk_bytes`` (no
      2x torch.cat blow-up). For the SCAIL-2 case that is ~24.5 GB peak.

    - ``avoid_oom=True`` (default): write the final directly to a
      ``numpy.memmap`` file (one mmap file per merge, deleted right after the
      final ``torch.save``). Peak during phase 2 is roughly 1 chunk (~3.5 GB
      for SCAIL-2). The OS page cache bounds the ``torch.save`` step, so the
      whole merge uses roughly the phase-1 peak (~7 GB) end-to-end.
      Unsupported dtypes (anything not in ``_TORCH_TO_NUMPY_DTYPE``) silently
      fall back to the pre-allocate path so a weird dtype never OOMs harder
      than the legacy code.

    On any failure the chunk files (and the memmap file, if any) are removed
    but the original per-batch .pt files are NOT touched -- matches the
    preserve-on-failure contract so the user can still recover via
    ``scripts/manual_merge_offloaded_images.py``.
    """
    paths = [
        Path(item["disk_path"])
        for item in disk_items
        if is_disk_cache_item(item)
    ]
    if not paths:
        raise FileNotFoundError(
            f"chunked_disk_merge: no on-disk batches under {out_path.parent}"
        )
    n = len(paths)
    chunk_size = max(1, int(chunk_size))
    n_chunks = (n + chunk_size - 1) // chunk_size
    run_dir = paths[0].parent
    out_path = Path(out_path)
    chunk_paths = []
    mmap_temp_paths = []
    total_frames = 0
    ref_hwc = None
    ref_dtype = None
    try:
        # Phase 1: build chunk files.
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n)
            chunk_batch = None
            for idx in range(start, end):
                batch = torch.load(
                    str(paths[idx]), map_location="cpu", weights_only=False
                )
                if not isinstance(batch, torch.Tensor):
                    raise ValueError(
                        f"{paths[idx]} is not a torch.Tensor "
                        f"(got {type(batch).__name__})"
                    )
                if ref_hwc is None:
                    ref_hwc = tuple(batch.shape[1:])
                    ref_dtype = batch.dtype
                if validate_batch is not None:
                    validate_batch(batch, idx, chunk_batch)
                if chunk_batch is None:
                    chunk_batch = batch
                else:
                    chunk_batch = torch.cat([chunk_batch, batch], dim=0)
                del batch
                gc.collect()
            chunk_path = run_dir / f"_mie_chunk_{kind}_{ci:04d}.pt"
            torch.save(chunk_batch, str(chunk_path))
            chunk_paths.append(chunk_path)
            total_frames += chunk_batch.shape[0]
            if log_progress is not None:
                log_progress(end, n)
            del chunk_batch
            gc.collect()

        if ref_hwc is None:
            raise RuntimeError("ref_hwc not set; no batches processed")

        use_mmap = bool(avoid_oom) and ref_dtype in _TORCH_TO_NUMPY_DTYPE
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_mmap:
            np_dtype = _TORCH_TO_NUMPY_DTYPE[ref_dtype]
            mmap_path = out_path.with_name(f".{out_path.stem}.{kind}.mmap.tmp")
            if mmap_path.exists():
                mmap_path.unlink()
            mmap = np.memmap(
                str(mmap_path),
                dtype=np_dtype,
                mode="w+",
                shape=(total_frames,) + ref_hwc,
            )
            mmap_temp_paths.append(mmap_path)
            try:
                offset = 0
                for ci, chunk_path in enumerate(chunk_paths):
                    chunk = torch.load(
                        str(chunk_path), map_location="cpu", weights_only=False
                    )
                    n_chunk = chunk.shape[0]
                    mmap[offset : offset + n_chunk] = chunk.numpy()
                    offset += n_chunk
                    if log_progress is not None:
                        log_progress(n, n)
                    del chunk
                    gc.collect()
                mmap.flush()
            finally:
                del mmap
                gc.collect()
            # Reopen in read mode and torch.save from the mmap storage.
            # OS page cache bounds the RAM used here; we never materialize
            # the full tensor in Python space.
            mmap_read = np.memmap(
                str(mmap_path),
                dtype=np_dtype,
                mode="r",
                shape=(total_frames,) + ref_hwc,
            )
            try:
                tensor = torch.from_numpy(mmap_read)
                torch.save(tensor, str(out_path))
            finally:
                del tensor
                del mmap_read
                gc.collect()
        else:
            # Legacy pre-allocate path. Use this when the dtype is unsupported
            # by the memmap bridge or when the user explicitly opts out.
            out = torch.empty(
                (total_frames,) + ref_hwc, dtype=ref_dtype, device="cpu"
            )
            offset = 0
            for ci, chunk_path in enumerate(chunk_paths):
                chunk = torch.load(
                    str(chunk_path), map_location="cpu", weights_only=False
                )
                n_chunk = chunk.shape[0]
                out[offset : offset + n_chunk] = chunk
                offset += n_chunk
                if log_progress is not None:
                    log_progress(n, n)
                del chunk
                gc.collect()
            torch.save(out, str(out_path))
            del out
            gc.collect()
        return str(out_path)
    finally:
        for cp in chunk_paths:
            try:
                cp.unlink()
            except FileNotFoundError:
                pass
        for mp in mmap_temp_paths:
            try:
                mp.unlink()
            except FileNotFoundError:
                pass


# Backward-compat alias for the previous private name. The live node and
# existing tests import `_chunked_disk_merge`; keep that working without
# forcing every caller to be rewritten at once.
_chunked_disk_merge = chunked_disk_merge


__all__ = [
    "chunked_disk_merge",
    "_chunked_disk_merge",
    "is_disk_cache_item",
    "build_disk_item",
    "_TORCH_TO_NUMPY_DTYPE",
]