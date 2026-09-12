#!/usr/bin/env python3
"""Single-pass streaming merge of on-disk tensor batches, shared by the loop
node and recovery tools.

This module is intentionally free of any ComfyUI / custom_nodes imports so that
both the live ComfyUI loop node (nodes/loop/loop.py) and the offline recovery
script (scripts/manual_merge_offloaded_images.py) call the exact same code path.

Design (since the 2 GiB sharding rework): every batch is streamed part-by-part
through a :class:`~core.tensor_store.ShardPartWriter` rolling buffer, so no
serialized file ever exceeds ``tensor_store.SHARD_FILE_CAP_BYTES`` (1.5 GiB,
safely under the 2^31-byte zip64 corruption boundary reported for
``torch.save`` with non-ASCII install paths). The previous two-phase design
(``_mie_chunk_*.pt`` intermediates + ``numpy.memmap`` staging feeding one giant
``torch.save``) is gone: it existed only to feed that single oversized write,
wrote the payload twice, and its dtype map silently downcast bfloat16 to
float16 in the final file.

Peak memory is roughly one input batch + one output part (~2.3 GB for the
SCAIL-2 reference case: 81-frame 704x1280 fp32 batches, ~0.8 GB each, plus a
1.5 GiB part) versus ~7 GB for the old phase-1. Inputs may be plain ``.pt``
files or sharded ``.shards`` directories, and a folder may mix both (an
upgrade-in-place offload dir); sharded inputs stream part-by-part with no
whole-batch reassembly.

``chunk_size`` and ``avoid_oom`` are accepted for API compatibility with old
call sites and have no effect: the byte cap governs buffering.

On any failure the staging output is removed but the original per-batch files
are NOT touched -- the preserve-on-failure contract, so the user can still
recover via ``scripts/manual_merge_offloaded_images.py``.

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

import torch

try:
    from .tensor_store import ShardPartWriter, iter_parts
except Exception:  # pragma: no cover - flat-import fallback for odd loaders
    from tensor_store import ShardPartWriter, iter_parts


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
    """Stream-merge on-disk tensor batches into one artifact at ``out_path``.

    Returns the ACTUAL artifact path: ``out_path`` itself (a plain ``.pt``,
    used whenever the merged tensor fits under the shard cap — byte-identical
    to the historical output) or the sibling ``<stem>.shards/`` directory for
    oversized results. Callers must thread this return value into any reload
    and any user-facing path output.

    ``validate_batch(part, idx, ref)`` runs before each part is appended, where
    ``ref`` is the first part ever seen (``None`` for the very first), matching
    the historical ``(batch, idx, merged)`` contract.

    ``log_progress(current, total)`` fires once per input item.
    """
    paths = [
        Path(item["disk_path"])
        for item in disk_items
        if is_disk_cache_item(item)
    ]
    if not paths:
        raise FileNotFoundError(
            f"chunked_disk_merge: no on-disk batches under {Path(out_path).parent}"
        )
    writer = ShardPartWriter(out_path)
    ref = None
    try:
        for idx, path in enumerate(paths):
            for part in iter_parts(path):
                if not isinstance(part, torch.Tensor):
                    raise ValueError(
                        f"{path} is not a torch.Tensor "
                        f"(got {type(part).__name__})"
                    )
                if validate_batch is not None:
                    validate_batch(part, idx, ref)
                if ref is None:
                    # Kept alive for shape/dtype validation of later batches;
                    # bounded by one part (~cap) and released on return.
                    ref = part
                writer.append(part)
                del part
            if log_progress is not None:
                log_progress(idx + 1, len(paths))
            gc.collect()
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


# Backward-compat alias for the previous private name. The live node and
# existing tests import `_chunked_disk_merge`; keep that working without
# forcing every caller to be rewritten at once.
_chunked_disk_merge = chunked_disk_merge


__all__ = [
    "chunked_disk_merge",
    "_chunked_disk_merge",
    "is_disk_cache_item",
    "build_disk_item",
]
