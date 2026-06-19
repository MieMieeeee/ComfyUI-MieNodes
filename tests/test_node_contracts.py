"""Contract safety net for the prompt-externalization refactor.

ComfyUI reloads a saved workflow by: class name -> INPUT_TYPES widget names ->
FUNCTION -> RETURN_TYPES. If those stay stable, existing workflows keep loading.

On first run this records the current contracts of every registered node to
``_node_contracts_baseline.json`` and skips. From then on it asserts every
*baseline* node's contract is unchanged — new nodes may be added freely, but an
existing node drifting its required fields / FUNCTION / RETURN_TYPES turns RED.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module  # noqa: E402

BASELINE_FILE = Path(__file__).resolve().parent / "_node_contracts_baseline.json"


def _collect_contracts(plugin) -> dict:
    contracts: dict = {}
    for key, cls in sorted(plugin.NODE_CLASS_MAPPINGS.items()):
        try:
            inputs = cls.INPUT_TYPES()
        except Exception as exc:  # noqa: BLE001
            contracts[key] = {"_error": str(exc)}
            continue
        contracts[key] = {
            "required": sorted((inputs.get("required") or {}).keys()),
            "optional": sorted((inputs.get("optional") or {}).keys()),
            "hidden": sorted((inputs.get("hidden") or {}).keys()),
            "function": getattr(cls, "FUNCTION", None),
            "return_types": list(getattr(cls, "RETURN_TYPES", []) or []),
            "return_names": list(getattr(cls, "RETURN_NAMES", []) or []),
            "category": getattr(cls, "CATEGORY", None),
        }
    return contracts


def _canonical(obj) -> str:
    """Stable comparison form.

    Node categories use 🐑 which loads as a UTF-16 surrogate pair under
    SourceFileLoader on Windows but round-trips through JSON as a single
    codepoint — semantically equal, repr-different. Canonical JSON treats
    both as ``\\ud83d\\udc11``, so this absorbs the env difference while
    still flagging any real field change.
    """
    return json.dumps(obj, ensure_ascii=True, sort_keys=True)


def test_existing_node_contracts_unchanged():
    plugin = load_plugin_module()
    current = _collect_contracts(plugin)
    if not BASELINE_FILE.exists():
        # ensure_ascii=True: node categories carry emoji (e.g. 🐑) that can
        # surface as lone surrogates through the loader on Windows, which utf-8
        # refuses to encode. ASCII-escaped JSON stores/reloads them losslessly.
        BASELINE_FILE.write_text(
            json.dumps(current, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pytest.skip(f"recorded baseline -> {BASELINE_FILE.name}; re-run to enforce")
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    missing = [k for k in baseline if k not in current]
    drifted = {
        k: {"baseline": baseline[k], "current": current.get(k)}
        for k in baseline
        if _canonical(current.get(k)) != _canonical(baseline[k])
    }
    assert not missing, f"nodes disappeared from NODE_CLASS_MAPPINGS: {missing}"
    assert not drifted, (
        "node contract drift detected (would break saved workflows):\n"
        + json.dumps(drifted, ensure_ascii=False, indent=2)
    )
