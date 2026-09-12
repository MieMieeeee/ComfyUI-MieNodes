#!/usr/bin/env python3
"""Manual merge for MieLoopCollectImage disk cache (single-pass, byte-capped).

When a ComfyUI loop run produces too many on-disk IMAGE batches and the
in-process `MieLoopFinalizeImages` merge fails mid-way, the per-batch files
(`image_*.pt` and/or `image_*.shards/` directories) are intentionally
preserved (preserve-on-failure contract). This script reads them from
`<comfyui>/temp/mie_loop_offload/<run_id>/`, runs them through the same
single-pass merge the live node uses, and writes an artifact that
`LoadImageBatch|Mie` can load directly (a `merged.pt` when it fits under the
~1.5 GiB shard cap, otherwise a `merged.shards/` directory).

Memory profile: peak is roughly one input batch + one output part (~2.3 GB
for the SCAIL-2 30-batch 81-frame 704x1280 fp32 case), not the final tensor
size (~25 GB). No single serialized file ever exceeds the shard cap, which
also makes this script safe on non-ASCII (e.g. Chinese) install paths where
oversized torch.save zip files have been reported to corrupt.

Output tensor matches the contract of `MieLoopCollectImage`: shape
(N, H, W, C) float32 in [0, 1], i.e. a standard ComfyUI IMAGE batch.

Usage (run on the ComfyUI machine where the batch files live):

    # Auto-discover the latest run_id under temp/mie_loop_offload
    python scripts/manual_merge_offloaded_images.py

    # Explicit run_id
    python scripts/manual_merge_offloaded_images.py 5e81e5f1a4b24e58a721371f

    # Explicit offload dir + output path
    python scripts/manual_merge_offloaded_images.py ^
        --offload-dir F:/ComfyUI_Mie_2026_V8.0_Base/ComfyUI/temp/mie_loop_offload/5e81e5f1a4b24e58a721371f ^
        --out F:/ComfyUI_Mie_2026_V8.0_Base/ComfyUI/temp/mie_loop_offload/merged_5e81e5f1.pt

    # Dry-run: just print what would be merged
    python scripts/manual_merge_offloaded_images.py --dry-run

After it finishes, wire `LoadImageBatch|Mie` with `file_path` pointed at the
output path to bypass the failed in-process merge.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import torch

# Use the same single-pass implementation the live node uses, so the recovery
# path cannot OOM harder than the live path. The shared module lives in core/
# (same package as the live node), so we add the repo root (parent of
# scripts/) to sys.path and import as a normal package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from core.chunked_merge import (  # noqa: E402
    build_disk_item,
    chunked_disk_merge,
    is_disk_cache_item,
)
from core.tensor_store import dir_bytes, remove_path  # noqa: E402


DEFAULT_OFFLOAD_ROOT = Path(
    r"F:\ComfyUI_Mie_2026_V8.0_Base\ComfyUI\temp\mie_loop_offload"
)


def _discover_latest_run(offload_root: Path) -> Path:
    if not offload_root.is_dir():
        raise FileNotFoundError(f"offload root not found: {offload_root}")
    run_dirs = sorted(
        (p for p in offload_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"no run_id subdirs under {offload_root}")
    return run_dirs[0]


def _list_batches(run_dir: Path) -> list[Path]:
    """Per-batch artifacts: image_*.pt files plus image_*.shards/ directories.

    Deliberately does NOT glob bare ``image_*``: that would also sweep up junk
    names. Hidden staging dirs (dot-prefixed) never match either pattern.
    """
    files = [Path(p) for p in glob.glob(str(run_dir / "image_*.pt"))]
    dirs = [Path(p) for p in glob.glob(str(run_dir / "image_*.shards")) if Path(p).is_dir()]
    return sorted(files + dirs)


def _artifact_bytes(path: Path) -> int:
    """File size, or summed part sizes for a .shards directory (dir st_size
    is 0 on Windows and entry-only on POSIX, which would undercount)."""
    return path.stat().st_size if path.is_file() else dir_bytes(path)


def _format_eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0:
        return "--:--"
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"{int(remain // 60):02d}:{int(remain % 60):02d}"


def _log_progress(stage: str, current: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    eta = _format_eta(elapsed, current, total)
    print(
        f"  [{stage} {current:>3}/{total}]  "
        f"elapsed={elapsed:5.1f}s  eta={eta}",
        flush=True,
    )


def merge(
    run_dir: Path,
    out_path: Path,
    *,
    cleanup: bool,
    dry_run: bool,
    chunk_size: int = 5,
    avoid_oom: bool = True,
) -> dict:
    paths = _list_batches(run_dir)
    if not paths:
        raise FileNotFoundError(f"no image_*.pt / image_*.shards batches in {run_dir}")
    sizes = [_artifact_bytes(p) for p in paths]
    total_bytes = sum(sizes)
    summary = {
        "run_dir": str(run_dir),
        "out_path": str(out_path),
        "batch_count": len(paths),
        "total_input_bytes": total_bytes,
        "dry_run": dry_run,
    }
    print(
        f"[merge] run_dir={run_dir}\n"
        f"       batch_count={len(paths)}  total_input={total_bytes / 1e9:.2f} GB",
        flush=True,
    )
    if dry_run:
        for p, s in zip(paths, sizes):
            print(f"  - {p.name}  {s / 1e6:.1f} MB")
        return summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    disk_items = [build_disk_item(p) for p in paths]
    if not all(is_disk_cache_item(it) for it in disk_items):
        # Defensive: build_disk_item always produces a valid item, but assert
        # here so a future regression in the helper does not silently fall
        # back to an empty merge.
        raise RuntimeError("internal: build_disk_item produced invalid items")

    t0 = time.time()

    def _progress(stage: str):
        def _cb(current: int, total: int) -> None:
            _log_progress(stage, current, total, t0)

        return _cb

    progress_cb = _progress("merge")

    try:
        written = chunked_disk_merge(
            disk_items,
            out_path,
            chunk_size=chunk_size,
            kind="image",
            log_progress=progress_cb,
            avoid_oom=avoid_oom,
        )
    except Exception:
        # Mirror the in-process contract: on failure, keep input batches so
        # the user can re-run the script after fixing env (more RAM, more
        # free disk, ...).
        print("[merge] FAILED -- input batches preserved for re-run", file=sys.stderr)
        raise

    written_path = Path(written)
    out_size = _artifact_bytes(written_path)
    print(
        f"[merge] wrote {written}\n"
        f"       size={out_size / 1e9:.2f} GB",
        flush=True,
    )
    if cleanup:
        removed = 0
        for p in paths:
            if remove_path(p):
                removed += 1
        print(f"[merge] cleaned {removed}/{len(paths)} input batch artifacts")
    summary.update(
        {
            "out_bytes": out_size,
            "chunk_size": chunk_size,
            "avoid_oom": avoid_oom,
        }
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Manually merge MieLoopCollectImage offloaded batches (image_*.pt files "
            "and image_*.shards/ directories) left behind when MieLoopFinalizeImages "
            "fails. Uses the same single-pass, byte-capped path as the live node: "
            "peak memory ~ one batch + one part (~2.3 GB for the SCAIL-2 30-batch "
            "case) and no serialized file above ~1.5 GiB."
        )
    )
    ap.add_argument(
        "run_id",
        nargs="?",
        help="Offload subdir name (defaults to the most recently modified one).",
    )
    ap.add_argument(
        "--offload-root",
        default=str(DEFAULT_OFFLOAD_ROOT),
        help=f"Parent offload dir (default: {DEFAULT_OFFLOAD_ROOT}).",
    )
    ap.add_argument(
        "--offload-dir",
        default=None,
        help="Explicit full offload dir; overrides --offload-root + run_id.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output path (default: <run_dir>/merged.pt). Small results land at "
            "this path as a plain .pt; oversized results land in the sibling "
            "merged.shards/ directory instead."
        ),
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help=(
            "Deprecated, no effect (kept for CLI compatibility). The merge is "
            "byte-capped by core.tensor_store.SHARD_FILE_CAP_BYTES now."
        ),
    )
    ap.add_argument(
        "--no-avoid-oom",
        action="store_true",
        help=(
            "Deprecated, no effect (kept for CLI compatibility). The merge is "
            "always single-pass and memory-bounded now."
        ),
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the input image_*.pt / image_*.shards batches after a successful merge.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be merged without writing anything.",
    )
    args = ap.parse_args()

    if args.offload_dir:
        run_dir = Path(args.offload_dir)
    elif args.run_id:
        run_dir = Path(args.offload_root) / args.run_id
    else:
        run_dir = _discover_latest_run(Path(args.offload_root))

    if not run_dir.is_dir():
        print(f"error: offload dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else run_dir / "merged.pt"

    try:
        merge(
            run_dir,
            out_path,
            cleanup=args.cleanup,
            dry_run=args.dry_run,
            chunk_size=max(1, int(args.chunk_size)),
            avoid_oom=not bool(args.no_avoid_oom),
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())