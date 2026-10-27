"""Copy assets from the machine library into the plugin's bundled_assets/.

Assets under ``<plugin>/bundled_assets/`` travel with the plugin package:
every install gets them without importing anything, and the loader falls
back to them for names the user library does not have (a user library
entry always wins, so people can replace a bundled character with their
own).

    python scripts/bundle_assets.py --list
    python scripts/bundle_assets.py characters/Claude娘 characters/Claude娘-日式赛璐璐
    python scripts/bundle_assets.py --all-characters "Claude娘*"
    python scripts/bundle_assets.py --all-storyboards
    python scripts/bundle_assets.py --dry-run characters/Claude娘

Source is the machine's asset library (default: <plugin>/asset_library,
override with --library). Storyboards live under ``storyboards/``,
characters under ``characters/``, scenes under ``scenes/``.
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "nodes"))

from assets import library as _library  # noqa: E402
from assets import storyboard as _storyboard  # noqa: E402

KINDS = {
    "characters": ("characters", "character.json"),
    "scenes": ("scenes", "scene.json"),
    "storyboards": ("storyboards", "storyboard.json"),
}


def library_root(args) -> Path:
    if args.library:
        return Path(args.library).expanduser().resolve()
    return _library.fixed_library_root(PLUGIN_ROOT)


def bundled_root() -> Path:
    return PLUGIN_ROOT / _library.BUNDLED_DIRNAME


def matching_names(source: Path, kind: str, patterns) -> list[str]:
    if not source.is_dir():
        return []
    names = sorted(entry.name for entry in source.iterdir() if entry.is_dir())
    if not patterns:
        return names
    out = []
    for name in names:
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            out.append(name)
    return out


def copy_one(source_dir: Path, target_root: Path, kind: str, name: str, dry: bool) -> str:
    subdir, json_name = KINDS[kind]
    src = source_dir / name
    dst = target_root / subdir / name
    if dry:
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        return "would copy %s/%s (%.1f MB)" % (subdir, name, size / 1048576)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    return "bundled %s/%s (%.1f MB)" % (subdir, name, size / 1048576)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="asset names to bundle")
    ap.add_argument("--library", default="", help="source library root")
    ap.add_argument("--all-characters", default="", help="glob of character names")
    ap.add_argument("--all-storyboards", action="store_true", help="bundle every storyboard")
    ap.add_argument("--all-scenes", action="store_true", help="bundle every scene")
    ap.add_argument("--list", action="store_true", help="list what would be bundled")
    ap.add_argument("--dry-run", action="store_true", help="do not write anything")
    args = ap.parse_args(argv)

    root = library_root(args)
    print("source library:", root)
    print("bundled dir  :", bundled_root())
    if not root.is_dir():
        print("source library does not exist", file=sys.stderr)
        return 2

    # Names may be given bare ("Claude娘" - searched in every kind) or
    # qualified ("characters/Claude娘" - only that kind).
    plan = []
    for kind in ("characters", "scenes", "storyboards"):
        source = root / kind
        patterns = [n for n in args.names if "/" not in n]
        qualified = [n.split("/", 1)[1] for n in args.names
                     if "/" in n and n.split("/", 1)[0] in KINDS
                     and n.split("/", 1)[0] == kind]
        if kind == "characters" and args.all_characters:
            patterns = [args.all_characters]
        if kind == "storyboards" and args.all_storyboards:
            patterns = ["*"]
        if kind == "scenes" and args.all_scenes:
            patterns = ["*"]
        patterns += qualified
        if not patterns:
            continue
        for name in matching_names(source, kind, patterns):
            plan.append((kind, name))

    if args.list:
        for kind, name in plan:
            print("  %-12s %s" % (kind, name))
        print("%d asset(s) match" % len(plan))
        return 0

    if not plan:
        print("nothing to bundle (give names or use --all-*)", file=sys.stderr)
        return 2

    for kind, name in plan:
        print(copy_one(root / kind, bundled_root(), kind, name, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
