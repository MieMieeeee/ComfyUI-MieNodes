#!/usr/bin/env python3
"""Manual merge for MieLoopCollectImage disk cache.

When a ComfyUI loop run produces too many on-disk IMAGE batches and the in-process
`MieLoopFinalizeImages` merge OOMs mid-way, the per-batch `image_*.pt` files are
intentionally preserved (see `_merge_tensor_batches_incremental` -> preserve-on-failure
contract). This script reads them from `<comfyui>/temp/mie_loop_offload/<run_id>/`,
concatenates them incrementally (same contract as the in-process merge), and writes a
single `.pt` file that can be loaded back via `LoadAny|Mie`.

Output tensor matches the contract of `MieLoopCollectImage`: shape (N, H, W, C) float32
in [0, 1], i.e. a standard ComfyUI IMAGE batch.

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

After it finishes, wire `LoadAny|Mie` with `file_path` pointed at the output `.pt`
to bypass the failed in-process merge.
"""

from __future__ import annotations

import argparse
import gc
import glob
import sys
import time
from pathlib import Path

import torch


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


def merge(
    run_dir: Path,
    out_path: Path,
    *,
    cleanup: bool,
    dry_run: bool,
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
        f"       batch_count={len(paths)}  total_input={total_bytes / 1e9:.2f} GB"
    )
    if dry_run:
        for p, s in zip(paths, sizes):
            print(f"  - {p.name}  {s / 1e6:.1f} MB")
        return summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = None
    ref_hwc = None
    t0 = time.time()
    try:
        for idx, p in enumerate(paths):
            batch = torch.load(str(p), map_location="cpu", weights_only=False)
            if not isinstance(batch, torch.Tensor):
                raise ValueError(
                    f"{p} is not a torch.Tensor (got {type(batch).__name__})"
                )
            hwc = tuple(batch.shape[1:])
            if ref_hwc is None:
                ref_hwc = hwc
            elif hwc != ref_hwc:
                raise ValueError(
                    f"shape mismatch at batch {idx}: {hwc} != {ref_hwc}"
                )
            if merged is None:
                merged = batch
            else:
                merged = torch.cat([merged, batch], dim=0)
            del batch
            gc.collect()
            elapsed = time.time() - t0
            done = idx + 1
            eta = _format_eta(elapsed, done, len(paths))
            print(
                f"  [{done:>3}/{len(paths)}]  "
                f"merged.shape={tuple(merged.shape)}  "
                f"peak~{merged.numel() * merged.element_size() / 1e9:.2f} GB  "
                f"elapsed={elapsed:5.1f}s  eta={eta}"
            )
    except Exception:
        # Mirror the in-process contract: on failure, keep input batches so the
        # user can re-run the script after fixing env (more RAM, smaller batch).
        print("[merge] FAILED -- input batches preserved for re-run", file=sys.stderr)
        raise

    torch.save(merged, str(out_path))
    out_size = out_path.stat().st_size
    print(
        f"[merge] wrote {out_path}\n"
        f"       shape={tuple(merged.shape)}  dtype={merged.dtype}  "
        f"size={out_size / 1e9:.2f} GB"
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
            "out_shape": tuple(merged.shape),
            "out_dtype": str(merged.dtype),
            "out_bytes": out_size,
        }
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Manually merge MieLoopCollectImage offloaded .pt batches produced by "
            "MieLoopFinalizeImages when the in-process merge OOMs."
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
        merge(run_dir, out_path, cleanup=args.cleanup, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())