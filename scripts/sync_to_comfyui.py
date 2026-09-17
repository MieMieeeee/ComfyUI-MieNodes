#!/usr/bin/env python3
"""Sync ComfyUI-MieNodes dev tree into live ComfyUI custom_nodes installs.

Built-in targets (both are disposable local test environments; code may be
fully overwritten — only the destination ``mie_llm_keys.json`` is preserved):

  local  C:\\PP\\V9\\V9-Large\\ComfyUI_Mie_2026_V9.0_Large\\ComfyUI\\custom_nodes\\comfyui_mienodes
  lan    \\\\192.168.31.150\\f\\V9_cu130\\V9-Large_cu130\\ComfyUI_Mie_2026_V9.0_Large\\ComfyUI\\custom_nodes\\comfyui_mienodes

Default behaviour: sync BOTH targets with mirror mode (stale destination
files that no longer exist in src are deleted). If one target is
unreachable (e.g. the LAN box is offline) the other still syncs; the exit
code reports the failure.

Usage:
  python scripts/sync_to_comfyui.py                # both targets
  python scripts/sync_to_comfyui.py --target local # one built-in target
  python scripts/sync_to_comfyui.py --dry-run
  python scripts/sync_to_comfyui.py --dst "D:\\other\\ComfyUI\\custom_nodes\\ComfyUI-MieNodes"
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Named deployment targets (test environments; full overwrite, keys kept).
TARGETS = {
    "local": Path(
        r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\custom_nodes\comfyui_mienodes"
    ),
    "lan": Path(
        r"\\192.168.31.150\f\V9_cu130\V9-Large_cu130\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\custom_nodes\comfyui_mienodes"
    ),
}

# Directory names skipped entirely during copy (and never mirror-deleted).
IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".idea",
    ".ruff_cache",
    ".pytest_cache",
    ".trae",
    ".claude",
    ".omc",
    ".sisyphus",
    ".zcode",
    "debug_logs",
    "logs",
    "%TEMP%",
    "notes",
}

# File names never copied from src; dst copies are preserved when present.
PRESERVE_DST_FILES = {
    "mie_llm_keys.json",
}

# Glob patterns for files never copied from src.
IGNORE_FILE_PATTERNS = (
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
    "*.excalidraw",
)


def _should_ignore_dir(name: str) -> bool:
    return name in IGNORE_DIRS


def _should_ignore_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORE_FILE_PATTERNS)


def _iter_source_files(src: Path) -> list[Path]:
    files: list[Path] = []
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if any(_should_ignore_dir(part) for part in rel.parts):
            continue
        if path.name in PRESERVE_DST_FILES:
            continue
        if _should_ignore_file(path.name):
            continue
        files.append(path)
    return files


def _backup_text(path: Path) -> str | None:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def _restore_text(path: Path, content: str | None) -> None:
    if content is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def sync(
    src: Path,
    dst: Path,
    *,
    mirror: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    src = src.resolve()
    dst = dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"source not found: {src}")

    stats = {"copied": 0, "removed": 0, "skipped": 0, "preserved": 0}

    keys_path = dst / "mie_llm_keys.json"
    keys_backup = _backup_text(keys_path) if keys_path.exists() else None

    source_files = _iter_source_files(src)
    source_rels = {f.relative_to(src) for f in source_files}

    if mirror:
        if dst.exists():
            for path in dst.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(dst)
                if any(_should_ignore_dir(part) for part in rel.parts):
                    continue
                if path.name in PRESERVE_DST_FILES:
                    continue
                if rel not in source_rels:
                    stats["removed"] += 1
                    if dry_run:
                        print(f"[dry-run] remove stale: {rel}")
                    else:
                        path.unlink()
                        print(f"removed stale: {rel}")

    for src_file in sorted(source_files):
        rel = src_file.relative_to(src)
        dst_file = dst / rel

        if rel.name in PRESERVE_DST_FILES and dst_file.exists():
            stats["preserved"] += 1
            print(f"preserved: {rel}")
            continue

        if dst_file.exists() and dst_file.read_bytes() == src_file.read_bytes():
            stats["skipped"] += 1
            continue

        stats["copied"] += 1
        if dry_run:
            print(f"[dry-run] copy: {rel}")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"copied: {rel}")

    if not dry_run:
        _restore_text(keys_path, keys_backup)
        if keys_backup is not None:
            print("restored mie_llm_keys.json in destination")

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync ComfyUI-MieNodes source into ComfyUI custom_nodes."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=REPO_ROOT,
        help=f"source repo root (default: {REPO_ROOT})",
    )
    parser.add_argument(
        "--target",
        choices=[*TARGETS, "all"],
        default="all",
        help="built-in deployment target(s) (default: all)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        help="custom single target (overrides --target), e.g. another ComfyUI install",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not delete stale files in destination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print actions without writing",
    )
    args = parser.parse_args(argv)

    if args.dst is not None:
        targets: list[tuple[str, Path]] = [("custom", args.dst)]
    elif args.target == "all":
        targets = list(TARGETS.items())
    else:
        targets = [(args.target, TARGETS[args.target])]

    print(f"src: {args.src.resolve()}")
    if args.dry_run:
        print("mode: dry-run")
    print()

    failed: list[str] = []
    for name, dst in targets:
        print(f"=== target: {name} -> {dst}")
        try:
            stats = sync(
                args.src,
                dst,
                mirror=not args.no_mirror,
                dry_run=args.dry_run,
            )
        except (FileNotFoundError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            failed.append(name)
            print()
            continue
        print(
            f"done ({name}). "
            f"copied={stats['copied']} "
            f"skipped={stats['skipped']} "
            f"removed={stats['removed']} "
            f"preserved={stats['preserved']}"
        )
        print()

    if failed:
        print(f"FAILED targets: {', '.join(failed)}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("restart ComfyUI or reload custom nodes to pick up Python changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
