#!/usr/bin/env python3
"""Sharded tensor persistence: no single ``torch.save`` file may exceed the cap.

Why this exists
---------------
Community-reported corruption (2026-09): when ``torch.save`` writes a file larger
than 2 GiB (its zip container switches to zip64) from an install whose paths
contain non-ASCII characters — Chinese-path ComfyUI integration packages, or even
a Chinese Windows username making ``%TEMP%`` itself non-ASCII — data corrupts at
the 2^31-byte boundary (reported symptom: a generated video collapsing to a
single frame). Every tensor persisted by the loop nodes therefore goes through
this module, which guarantees each serialized file stays under
``SHARD_FILE_CAP_BYTES`` (1.5 GiB binary, ~512 MiB of margin; the cap checks the
logical ``tensor.nbytes`` and the zip overhead for one tensor is KB-scale).

Format (used only when a tensor exceeds the cap)
------------------------------------------------
``<stem>.shards/`` directory next to the intended ``<path>.pt``:

- ``part_00000.pt``, ``part_00001.pt``, ... — frame-aligned slices along dim 0,
  each with ``nbytes <= cap``. Parts are plain ``torch.save`` tensors, so dtype
  (including ``bfloat16``) round-trips natively and users can ``torch.load``
  individual parts by hand.
- ``part_NNNNN.bin`` — rare degenerate case: a single frame larger than the cap
  is stored as raw bytes (no zip container at all) instead of an uncapped
  ``torch.save``; ``meta.json`` records ``"encoding": "raw"``.
- ``meta.json`` — the commit marker, written LAST via ``os.replace``. A process
  killed mid-write leaves a directory without a valid ``meta.json``;
  ``load_tensor`` / ``is_complete_shard_dir`` treat that as missing/corrupt and
  raise / return False — a partial directory is never silently mis-loaded.

Loading validates ``format`` / ``version``, that every listed part exists with
``st_size > 0``, that ``sum(frames) == shape[0]``, and each part's dtype and
frame count, then reassembles into one pre-allocated tensor (peak RAM = final
tensor + one part).

All ``torch.load`` calls use ``weights_only=False``, matching the loop nodes'
pre-existing behavior: these files are written by this plugin itself.

Non-goals: non-tensor payloads (the audio dict stays a single small ``.pt`` via
``torch.save``; a multi-hour waveform could in theory grow large — accepted),
and pickle-based caches (``SaveAny|Mie``) are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch


_log = logging.getLogger(__name__)

# Hard cap for any single serialized file, in bytes. 1.5 GiB (binary:
# 1,610,612,736 bytes) leaves ~512 MiB under the 2^31-byte zip64 corruption
# boundary. Tests force sharding by monkeypatching this module attribute.
SHARD_FILE_CAP_BYTES = int(1.5 * 1024**3)

SHARD_FORMAT = "mie-tensor-shards"
SHARD_VERSION = 1
META_NAME = "meta.json"

PathLike = Union[str, Path]


class CorruptShardError(RuntimeError):
    """A ``.shards`` directory is incomplete or inconsistent with its meta."""


def shard_dir_for(path: PathLike) -> Path:
    """Return the sibling shard-directory path for a primary ``path``.

    ``merged.pt`` -> ``merged.shards``; ``foo`` (no extension) -> ``foo.shards``.
    """
    s = str(path)
    if s.endswith(".pt"):
        s = s[: -len(".pt")]
    return Path(s + ".shards")


def _parse_dtype(name: Any) -> torch.dtype:
    if not isinstance(name, str) or not name.startswith("torch."):
        raise CorruptShardError(f"bad dtype tag in meta: {name!r}")
    dtype = getattr(torch, name[len("torch.") :], None)
    if not isinstance(dtype, torch.dtype):
        raise CorruptShardError(f"unknown dtype tag in meta: {name!r}")
    return dtype


def _read_meta(shard_dir: Path) -> dict:
    meta_path = shard_dir / META_NAME
    if not meta_path.is_file():
        raise CorruptShardError(f"no {META_NAME} in {shard_dir} (incomplete write?)")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CorruptShardError(f"unreadable {META_NAME} in {shard_dir}: {e}") from e
    if not isinstance(meta, dict):
        raise CorruptShardError(f"{meta_path} is not a JSON object")
    if meta.get("format") != SHARD_FORMAT:
        raise CorruptShardError(f"bad format tag in {meta_path}: {meta.get('format')!r}")
    if int(meta.get("version", -1)) != SHARD_VERSION:
        raise CorruptShardError(f"unsupported shard version in {meta_path}: {meta.get('version')!r}")
    return meta


def _validate_meta(shard_dir: Path, meta: dict) -> tuple:
    """Check the meta against the files on disk. Returns (dtype, shape, parts)."""
    shape = meta.get("shape")
    if not isinstance(shape, list) or len(shape) < 1 or not all(isinstance(x, int) and x >= 0 for x in shape):
        raise CorruptShardError(f"bad shape in {shard_dir}/{META_NAME}: {shape!r}")
    parts = meta.get("parts")
    if not isinstance(parts, list) or len(parts) == 0:
        raise CorruptShardError(f"no parts listed in {shard_dir}/{META_NAME}")
    dtype = _parse_dtype(meta.get("dtype"))
    total = 0
    for entry in parts:
        if not isinstance(entry, dict):
            raise CorruptShardError(f"bad part entry in {shard_dir}/{META_NAME}: {entry!r}")
        name = entry.get("file")
        enc = entry.get("encoding", "pt")
        frames = entry.get("frames")
        if not isinstance(name, str) or enc not in ("pt", "raw") or not isinstance(frames, int) or frames <= 0:
            raise CorruptShardError(f"bad part entry in {shard_dir}/{META_NAME}: {entry!r}")
        part_path = shard_dir / name
        if not part_path.is_file() or part_path.stat().st_size <= 0:
            raise CorruptShardError(f"missing or empty part file: {part_path}")
        total += frames
    if total != shape[0]:
        raise CorruptShardError(
            f"frames mismatch in {shard_dir}/{META_NAME}: parts sum {total} != shape[0] {shape[0]}"
        )
    return dtype, shape, parts


def is_complete_shard_dir(path: PathLike) -> bool:
    """True iff ``path`` is a fully-committed, self-consistent shard directory.

    Deliberately cheap (reads meta + stats parts; never loads tensors) so
    ``FileExists|Mie`` can use it as a gate: a crash-left partial directory
    counts as a MISS so an IfElse recompute branch still fires.
    """
    try:
        p = Path(path)
        if not p.is_dir():
            return False
        meta = _read_meta(p)
        _validate_meta(p, meta)
        return True
    except (CorruptShardError, OSError):
        return False


def find_tensor_path(path: PathLike) -> Optional[str]:
    """Resolve a user-visible path to a loadable artifact, or None.

    Probes in order: the file itself, a complete shard directory at ``path``
    (callers may pass the ``.shards`` dir directly), then the sibling
    ``<stem>.shards`` alias — so a widget string still saying ``foo.pt`` finds
    the ``foo.shards/`` written for an oversized tensor.
    """
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir() and is_complete_shard_dir(p):
        return str(p)
    alias = shard_dir_for(p)
    if alias.is_dir() and is_complete_shard_dir(alias):
        return str(alias)
    return None


def remove_path(path: PathLike) -> bool:
    """Delete a file or directory tree; missing paths and errors are swallowed.

    Returns True when something was removed. Mirrors the loop cleanup helpers'
    swallow-all semantics so cleanup can never mask the real exception.
    """
    try:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(str(p))
            return True
        if p.is_file():
            p.unlink()
            return True
    except OSError:
        return False
    return False


def dir_bytes(path: PathLike) -> int:
    """Summed size of a directory's files (0 for files/missing paths).

    Reporting helper: ``Path.stat().st_size`` on a directory is 0 on Windows
    and only the entry's own size on POSIX, so size reports must walk the tree.
    """
    p = Path(path)
    if not p.is_dir():
        return 0
    total = 0
    for child in p.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _write_raw_part(tensor: torch.Tensor, path: Path) -> None:
    # Bit-exact raw bytes for a dtype numpy does not know (bfloat16): view the
    # storage as uint8 (contiguous tensor, last dim contiguous) and dump it.
    u8 = tensor.view(torch.uint8)
    u8.numpy().tofile(str(path))


def _read_raw_part(path: Path, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
    arr = np.fromfile(str(path), dtype=np.uint8)
    t = torch.from_numpy(arr).view(dtype)
    return t.reshape(shape)


def _write_meta_atomic(shard_dir: Path, meta: dict) -> None:
    tmp = shard_dir / (META_NAME + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(shard_dir / META_NAME))


def iter_parts(path: PathLike) -> Iterator[torch.Tensor]:
    """Yield a persisted tensor's parts one at a time (streaming input).

    Plain ``.pt`` files yield their single object (which may be any pickled
    payload — callers type-check). Shard directories are validated first and
    then yield each part; the full tensor is never materialized, so a sharded
    merge input streams into the writer without a whole-batch reassembly.
    """
    p = Path(path)
    if p.is_file():
        yield torch.load(str(p), map_location="cpu", weights_only=False)
        return
    if p.is_dir():
        meta = _read_meta(p)
        dtype, shape, parts = _validate_meta(p, meta)
        for entry in parts:
            frames = int(entry["frames"])
            part_shape = (frames,) + tuple(shape[1:])
            if entry.get("encoding", "pt") == "raw":
                yield _read_raw_part(p / entry["file"], dtype, part_shape)
            else:
                part = torch.load(str(p / entry["file"]), map_location="cpu", weights_only=False)
                if not isinstance(part, torch.Tensor):
                    raise CorruptShardError(f"part {entry['file']} is not a torch.Tensor")
                yield part
        return
    raise FileNotFoundError(str(p))


def load_tensor(path: PathLike) -> Any:
    """Load what :func:`save_tensor` wrote at/for ``path``.

    - plain file -> ``torch.load(weights_only=False)`` (any payload, including
      the audio dict — the tensor-only restriction applies to shard dirs only)
    - shard directory -> validated reassembly into one tensor
    - anything else -> ``FileNotFoundError`` / :class:`CorruptShardError`
    """
    p = Path(path)
    if p.is_file():
        return torch.load(str(p), map_location="cpu", weights_only=False)
    if p.is_dir():
        meta = _read_meta(p)
        dtype, shape, parts = _validate_meta(p, meta)
        out = torch.empty(tuple(shape), dtype=dtype)
        offset = 0
        for entry in parts:
            frames = int(entry["frames"])
            part_shape = (frames,) + tuple(shape[1:])
            if entry.get("encoding", "pt") == "raw":
                part = _read_raw_part(p / entry["file"], dtype, part_shape)
            else:
                part = torch.load(str(p / entry["file"]), map_location="cpu", weights_only=False)
                if not isinstance(part, torch.Tensor):
                    raise CorruptShardError(f"part {entry['file']} is not a torch.Tensor")
                if part.dtype != dtype or tuple(part.shape) != part_shape:
                    raise CorruptShardError(
                        f"part {entry['file']} disagrees with meta: "
                        f"{part.dtype}{tuple(part.shape)} != {dtype}{part_shape}"
                    )
            out[offset : offset + frames] = part
            offset += frames
            del part
        return out
    raise FileNotFoundError(str(p))


class ShardPartWriter:
    """Streaming frame-aligned part writer behind both save paths.

    ``append()`` accumulates tensors in a rolling buffer bounded by the cap and
    flushes ``part_*`` files into a hidden staging directory as it fills;
    ``finish()`` commits either a single small ``<path>.pt`` (byte-identical to
    a plain ``torch.save``) or the ``<stem>.shards/`` directory with ``meta.json``
    written last. ``abort()`` removes the staging tree — output is regenerable,
    so a failed write is deleted rather than preserved (per-batch *inputs* are
    what the loop's preserve-on-failure contract keeps).

    Peak RAM during a merge is roughly one input batch + one part.
    """

    def __init__(self, out_path: PathLike, cap: Optional[int] = None):
        self.out_path = Path(out_path)
        self._cap_override = cap
        self._hwc: Optional[tuple] = None
        self._dtype: Optional[torch.dtype] = None
        self._frame_bytes = 0
        self._fps = 0  # frames per part
        self._buf: Optional[torch.Tensor] = None
        self._frames = 0
        self._parts: list = []
        self._finished = False
        stem = self.out_path.name
        if stem.endswith(".pt"):
            stem = stem[: -len(".pt")]
        # Dot-prefixed so batch-discovery globs (image_*) never pick it up.
        self._staging = self.out_path.parent / f".{stem}.staging"

    def _cap(self) -> int:
        return self._cap_override if self._cap_override is not None else SHARD_FILE_CAP_BYTES

    def _set_ref(self, tensor: torch.Tensor) -> None:
        self._hwc = tuple(tensor.shape[1:])
        self._dtype = tensor.dtype
        el = tensor.element_size()
        self._frame_bytes = el * int(np.prod(self._hwc)) if self._hwc else el
        if self._frame_bytes > 0:
            self._fps = max(1, self._cap() // self._frame_bytes)
        else:
            # Degenerate zero-sized frame (e.g. shape (N, 0, 3)): every frame
            # is 0 bytes, so a single part holds all of them.
            self._fps = 2**31

    def append(self, tensor: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("ShardPartWriter already finished")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"expected a torch.Tensor, got {type(tensor).__name__}")
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.to("cpu")
        if not t.is_contiguous():
            t = t.contiguous()
        if t.ndim < 1 or t.shape[0] == 0:
            return
        if self._hwc is None:
            self._set_ref(t)
        elif t.dtype != self._dtype or tuple(t.shape[1:]) != self._hwc:
            raise ValueError(
                f"shape/dtype mismatch: {t.dtype}{tuple(t.shape[1:])} != "
                f"{self._dtype}{self._hwc}"
            )
        self._buf = t if self._buf is None else torch.cat([self._buf, t])
        while self._buf is not None and self._fps > 0 and self._buf.shape[0] >= self._fps:
            self._write_part(self._buf[: self._fps].clone())
            rest = self._buf[self._fps :]
            self._buf = rest if rest.shape[0] > 0 else None

    def _write_part(self, part: torch.Tensor) -> None:
        self._staging.mkdir(parents=True, exist_ok=True)
        idx = len(self._parts)
        frames = int(part.shape[0])
        if part.nbytes <= self._cap():
            name = f"part_{idx:05d}.pt"
            torch.save(part, str(self._staging / name))
            enc = "pt"
        else:
            # Degenerate single frame larger than the cap: raw bytes, never an
            # uncapped torch.save zip.
            if frames != 1:
                raise CorruptShardError(f"multi-frame part exceeds cap: {part.shape}")
            name = f"part_{idx:05d}.bin"
            _write_raw_part(part, self._staging / name)
            enc = "raw"
            _log.warning(
                "tensor_store: single frame (%d bytes) exceeds cap (%d); "
                "written as raw part %s",
                part.nbytes,
                self._cap(),
                name,
            )
        self._parts.append({"file": name, "encoding": enc, "frames": frames})
        self._frames += frames

    def _meta(self) -> dict:
        return {
            "format": SHARD_FORMAT,
            "version": SHARD_VERSION,
            "dtype": str(self._dtype),
            "shape": [self._frames] + list(self._hwc or ()),
            "parts": self._parts,
            "created": datetime.now(timezone.utc).isoformat(),
        }

    def finish(self) -> str:
        """Commit and return the actual artifact path (file or shard dir)."""
        if self._finished:
            raise RuntimeError("ShardPartWriter already finished")
        self._finished = True
        if self._hwc is None:
            raise RuntimeError("no frames appended")
        if self._buf is not None and self._buf.shape[0] > 0:
            self._write_part(self._buf.clone())
            self._buf = None
        if not self._parts:
            raise RuntimeError("no parts written")
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            total_bytes = self._frames * self._frame_bytes
            if total_bytes <= self._cap() and len(self._parts) == 1 and self._parts[0]["encoding"] == "pt":
                # Small result: emit one plain .pt at exactly the caller's path.
                part = torch.load(
                    str(self._staging / self._parts[0]["file"]), map_location="cpu", weights_only=False
                )
                torch.save(part, str(self.out_path))
                remove_path(self._staging)
                remove_path(shard_dir_for(self.out_path))  # stale alias from an older big save
                return str(self.out_path)
            # Big result: commit the shard directory. meta.json last = commit marker.
            _write_meta_atomic(self._staging, self._meta())
            target = shard_dir_for(self.out_path)
            remove_path(target)
            remove_path(self.out_path)  # stale single file from an older small save
            os.replace(str(self._staging), str(target))
            return str(target)
        except BaseException:
            remove_path(self._staging)
            raise

    def abort(self) -> None:
        """Remove the staging tree (failed write; output is regenerable)."""
        self._finished = True
        remove_path(self._staging)


def save_tensor(obj: Any, path: PathLike) -> str:
    """Persist ``obj`` and return the actual artifact path.

    Non-tensor payloads and tensors at or below the cap go to a single
    ``torch.save`` at ``path`` verbatim (byte-identical to the old behavior, so
    widget strings like ``output/cache/foo.pt`` keep working). Oversized tensors
    land in the sibling ``<stem>.shards/`` directory instead. A previous
    artifact of the *other* representation is removed so the alias probe in
    :func:`find_tensor_path` can never resurrect stale data.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(obj, torch.Tensor):
        torch.save(obj, str(p))
        remove_path(shard_dir_for(p))
        return str(p)
    t = obj.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    if not t.is_contiguous():
        t = t.contiguous()
    if t.ndim < 1 or t.shape[0] == 0 or t.nbytes <= SHARD_FILE_CAP_BYTES:
        torch.save(t, str(p))
        remove_path(shard_dir_for(p))
        return str(p)
    writer = ShardPartWriter(p)
    try:
        writer.append(t)
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


__all__ = [
    "SHARD_FILE_CAP_BYTES",
    "SHARD_FORMAT",
    "SHARD_VERSION",
    "CorruptShardError",
    "ShardPartWriter",
    "save_tensor",
    "load_tensor",
    "iter_parts",
    "find_tensor_path",
    "is_complete_shard_dir",
    "shard_dir_for",
    "remove_path",
    "dir_bytes",
]
