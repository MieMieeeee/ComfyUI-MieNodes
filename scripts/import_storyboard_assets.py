"""Import storyboard asset JSON files into the machine's asset library.

Pure filesystem: no ComfyUI, no torch. Each JSON is validated by the
plugin's own ``save_storyboard`` (which re-checks the v2 contract) and
written atomically to ``<library>/storyboards/<name>/storyboard.json``.

Usage:
    python scripts/import_storyboard_assets.py docs/storyboard-assets
    python scripts/import_storyboard_assets.py docs/storyboard-assets \
        --library "C:/path/to/asset_library" --overwrite
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types


def _load_plugin():
    """Import the plugin's asset modules without importing ComfyUI."""
    package = types.ModuleType("_mie_storyboard_cli")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "nodes" / "assets")]
    sys.modules[package.__name__] = package
    return (importlib.import_module(package.__name__ + ".library"),
            importlib.import_module(package.__name__ + ".storyboard"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="a JSON file or a directory of storyboard JSON files")
    parser.add_argument("--library", default="",
                        help="asset library root (default: the machine's fixed root)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing storyboard with the same name")
    args = parser.parse_args(argv)

    library, storyboard = _load_plugin()
    library_dir = library.resolve_library_dir(args.library)

    files = [args.source] if args.source.is_file() else sorted(args.source.glob("*.json"))
    if not files:
        print("No storyboard JSON files found in %s" % args.source, file=sys.stderr)
        return 2

    failures = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            existing = storyboard.list_storyboards(library_dir).get("names", [])
            names = [existing] if isinstance(existing, str) else list(existing)
            if payload.get("name") in names and not args.overwrite:
                print("skip (exists): %s" % payload.get("name"))
                continue
            cleaned, saved, status = storyboard.save_storyboard(library_dir, payload)
            print("%s: %s -> %s" % (status, cleaned["name"], saved))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            failures += 1
            print("failed: %s: %s" % (path.name, exc), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
