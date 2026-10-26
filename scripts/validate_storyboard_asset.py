"""Validate an external storyboard without importing ComfyUI, torch or the plugin."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output", type=Path, help="write canonical asset to this file")
    args = parser.parse_args(argv)
    package = types.ModuleType("_mie_storyboard_cli")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "nodes" / "assets")]
    sys.modules[package.__name__] = package
    contract = importlib.import_module(package.__name__ + ".storyboard_contract")
    try:
        payload = json.loads(args.asset.read_text(encoding="utf-8-sig"))
        asset = contract.validate_v2(payload)
        frames = contract.allocate_frames(asset)
        seconds = (sum(frames) - asset["continuation_overlap_frames"] * (len(frames) - 1)) / 24
        warnings = contract.storyboard_warnings(asset)
        if args.output:
            args.output.write_text(json.dumps(asset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"valid": True, "name": asset["name"], "roles": len(asset["roles"]),
                          "shots": len(frames), "frames": frames, "delivered_seconds": seconds,
                          "continuation_overlap_frames": asset["continuation_overlap_frames"],
                          "warnings": warnings},
                         ensure_ascii=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print("Invalid storyboard asset: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
