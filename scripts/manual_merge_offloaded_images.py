#!/usr/bin/env python3
"""Manual merge for MieLoopCollectImage disk cache (chunked + memmap).

When a ComfyUI loop run produces too many on-disk IMAGE batches and the
in-process `MieLoopFinalizeImages` merge OOMs mid-way, the per-batch
`image_*.pt` files are intentionally preserved (preserve-on-failure contract).
This script reads them from `<comfyui>/temp/mie_loop_offload/<run_id>/`,
runs them through the same chunked+memmap merge the live node uses, and
writes a single `.pt` file that can be loaded back via `LoadAny|Mie`.

Memory profile: peak is bounded by phase-1 chunk size (~7 GB for the SCAIL-2
30-batch 81-frame 704x1280 fp32 case at chunk_size=5), not by the final
tensor size (~25 GB). The old in-memory torch.cat loop in this script could
not be used for that workload.

Output tensor matches the contract of `MieLoopCollectImage`: shape
(N, H, W, C) float32 in [0, 1], i.e. a standard ComfyUI IMAGE batch.

Usage (run on the ComfyUI machine where the .pt files live):

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

After it finishes, wire `LoadAny|Mie` with `file_path` pointed at the output
`.pt` to bypass the failed in-process merge.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import torch

# Use the same chunked+memmap implementation the live node uses, so the
# recovery path cannot OOM harder than the live path. The shared module
# lives in core/ (same package as the live node), so we add the repo
# root (parent of scripts/) to sys.path and import as a normal package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from core.chunked_merge import (  # noqa: E402
    build_disk_item,
    chunked_disk_merge,
    is_disk_cache_item,
)


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
    return sorted(Path(p) for p in glob.glob(str(run_dir / "image_*.pt")))


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
        raise FileNotFoundError(f"no image_*.pt files in {run_dir}")
    sizes = [p.stat().st_size for p in paths]
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
    n = len(paths)
    n_chunks = (n + chunk_size - 1) // chunk_size

    def _progress_chunk(stage: str):
        done = {"v": 0}

        def _cb(current: int, total: int) -> None:
            done["v"] = current
            _log_progress(stage, current, total, t0)

        return _cb

    # Phase-1 + phase-2 progress share the same callback shape; the chunked
    # merge calls it once per input batch in phase 1 and once per chunk in
    # phase 2 (with current == total at the end of phase 2).
    progress_cb = _progress_chunk("merge")

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
        # the user can re-run the script after fixing env (more RAM, fewer
        # batches per chunk).
        print("[merge] FAILED -- input batches preserved for re-run", file=sys.stderr)
        raise

    out_size = Path(written).stat().st_size
    print(
        f"[merge] wrote {written}\n"
        f"       size={out_size / 1e9:.2f} GB",
        flush=True,
    )
    if cleanup:
        for p in paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        print(f"[merge] cleaned {len(paths)} input batch files")
    summary.update(
        {
            "out_bytes": out_size,
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "avoid_oom": avoid_oom,
        }
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Manually merge MieLoopCollectImage offloaded .pt batches produced by "
            "MieLoopFinalizeImages when the in-process merge OOMs. Uses the same "
            "chunked+memmap path as the live node, so peak memory is bounded by "
            "the chunk size (~7 GB for the SCAIL-2 30-batch case at chunk_size=5)."
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
        help="Output .pt path (default: <run_dir>/merged.pt).",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help=(
            "Phase-1 chunk size. Lower = less phase-1 peak memory but more "
            "intermediate chunk files. Default 5 (matches the live node)."
        ),
    )
    ap.add_argument(
        "--no-avoid-oom",
        action="store_true",
        help=(
            "Disable the numpy.memmap phase-2 path. Phase 2 will pre-allocate "
            "the full final tensor on CPU; only safe if you have enough RAM "
            "for the whole result (~25 GB for SCAIL-2 30-batch)."
        ),
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the input image_*.pt files after a successful merge.",
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