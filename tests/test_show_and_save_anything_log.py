# -*- coding: utf-8 -*-
"""Tests for the ShowAndSaveAnything node."""
import importlib.util
import json
import logging
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
GENERAL_PATH = PROJECT_DIR / "nodes" / "common" / "general.py"
UTILS_PATH = PROJECT_DIR / "core" / "utils.py"


def _load_module():
    """Load ``nodes/common/general.py`` with a stubbed ``folder_paths``."""
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.base_path = str(PROJECT_DIR)
        fp.models_dir = str(PROJECT_DIR / "models")
        fp.output_directory = str(PROJECT_DIR / "output")
        fp.input_directory = str(PROJECT_DIR / "input")
        fp.get_filename_list = lambda *_a, **_k: []
        fp.get_full_path = lambda *_a, **_k: ""
        fp.get_output_directory = lambda: fp.output_directory
        fp.get_input_directory = lambda: fp.input_directory
        sys.modules["folder_paths"] = fp

    if "_mienodes_internal" not in sys.modules:
        ip = types.ModuleType("_mienodes_internal")
        ip.__path__ = [str(PROJECT_DIR)]
        ip.__package__ = "_mienodes_internal"
        sys.modules["_mienodes_internal"] = ip
    if "_mienodes_internal.core" not in sys.modules:
        core = types.ModuleType("_mienodes_internal.core")
        core.__path__ = [str(PROJECT_DIR / "core")]
        core.__package__ = "_mienodes_internal.core"
        sys.modules["_mienodes_internal.core"] = core
    if "_mienodes_internal.core.utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "_mienodes_internal.core.utils", str(UTILS_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_mienodes_internal.core.utils"] = mod
        spec.loader.exec_module(mod)

    # Fresh load
    saved = list(sys.modules)
    for name in list(sys.modules):
        if name.startswith("general_for_test"):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        "general_for_test", str(GENERAL_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close_loggers(general):
    """Close file handles so temp dirs can be removed on Windows."""
    cls = getattr(general, "ShowAndSaveAnythingMie", None)
    if cls is None:
        return
    for logger in list(cls._LOGGER_CACHE.values()):
        for h in list(logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
    cls._LOGGER_CACHE.clear()


@pytest.fixture
def general():
    mod = _load_module()
    _close_loggers(mod)
    yield mod
    _close_loggers(mod)


@pytest.fixture
def tmp_log_dir(general, tmp_path, monkeypatch):
    """Redirect the plugin's logs folder to a per-test tmp directory."""
    log_dir = tmp_path / "logs"
    cls = general.ShowAndSaveAnythingMie
    monkeypatch.setattr(cls, "LOG_DIR_NAME", "logs")
    # Patch the helper so it lands in tmp_path
    monkeypatch.setattr(general, "_plugin_root_dir", lambda: str(tmp_path))
    yield log_dir
    _close_loggers(general)


def test_resolve_log_path_uses_default_name(general, tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_plugin_root_dir", lambda: str(tmp_path))
    p = general.ShowAndSaveAnythingMie._resolve_log_path("")
    assert p == str(tmp_path / "logs" / "show_anything.log")
    assert os.path.isdir(os.path.dirname(p))


def test_resolve_log_path_uses_supplied_name(general, tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_plugin_root_dir", lambda: str(tmp_path))
    p = general.ShowAndSaveAnythingMie._resolve_log_path("my_run.log")
    assert p == str(tmp_path / "logs" / "my_run.log")


def test_resolve_log_path_strips_directory_components(general, tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_plugin_root_dir", lambda: str(tmp_path))
    p = general.ShowAndSaveAnythingMie._resolve_log_path("../escape/foo.log")
    assert p == str(tmp_path / "logs" / "foo.log")


def test_resolve_log_path_whitespace_falls_back(general, tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_plugin_root_dir", lambda: str(tmp_path))
    p = general.ShowAndSaveAnythingMie._resolve_log_path("   ")
    assert p == str(tmp_path / "logs" / "show_anything.log")


def test_extract_upstream_finds_linked_node(general):
    extra_pnginfo = {
        "workflow": {
            "nodes": [
                {"id": 1, "type": "CLIPTextEncode", "title": "Prompt"},
                {
                    "id": 2,
                    "type": "ShowAndSaveAnything|Mie",
                    "title": "Show",
                    "inputs": [{"name": "anything", "type": "*", "link": 10}],
                },
            ],
            "links": [[10, 1, 0, 2, 0, "*"]],
        }
    }
    meta = general._extract_upstream(extra_pnginfo, 2)
    assert meta["node_id"] == 2
    assert meta["node_title"] == "Show"
    assert meta["node_type"] == "ShowAndSaveAnything|Mie"
    assert meta["upstream_id"] == 1
    assert meta["upstream_title"] == "Prompt"
    assert meta["upstream_type"] == "CLIPTextEncode"


def test_extract_upstream_handles_list_link(general):
    extra_pnginfo = {
        "workflow": {
            "nodes": [
                {"id": 7, "type": "Foo", "title": "Foo"},
                {"id": 9, "type": "ShowAndSaveAnything|Mie", "title": "Show", "inputs": [{"name": "anything", "link": [99]}]},
            ],
            "links": [[99, 7, 0, 9, 0, "*"]],
        }
    }
    meta = general._extract_upstream(extra_pnginfo, 9)
    assert meta["upstream_id"] == 7
    assert meta["upstream_type"] == "Foo"


def test_extract_upstream_handles_missing_workflow(general):
    assert general._extract_upstream(None, 1) == {}
    assert general._extract_upstream({}, 1) == {}
    assert general._extract_upstream({"workflow": {}}, 1) == {}
    assert general._extract_upstream({"workflow": {"nodes": "nope"}}, 1) == {}


def test_extract_upstream_handles_missing_node(general):
    extra_pnginfo = {
        "workflow": {
            "nodes": [{"id": 1, "type": "X", "title": "X"}],
            "links": [],
        }
    }
    meta = general._extract_upstream(extra_pnginfo, 999)
    # the current node is not in the workflow, so we get no info at all
    assert meta == {}


def test_execute_writes_log_entry(general, tmp_log_dir):
    extra_pnginfo = {
        "workflow": {
            "nodes": [
                {"id": 1, "type": "CLIPTextEncode", "title": "Prompt"},
                {
                    "id": 2,
                    "type": "ShowAndSaveAnything|Mie",
                    "title": "Show",
                    "inputs": [{"name": "anything", "link": 10}],
                },
            ],
            "links": [[10, 1, 0, 2, 0, "*"]],
        }
    }
    node = general.ShowAndSaveAnythingMie()
    out = node.execute(
        "hello world",
        log_file_name="run.log",
        unique_id=2,
        extra_pnginfo=extra_pnginfo,
    )
    assert out["result"] == ("hello world",)
    log_path = tmp_log_dir / "run.log"
    assert log_path.is_file()
    entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    # log entry is intentionally minimal: just ts, upstream_title, result
    assert set(entry.keys()) == {"ts", "upstream_title", "result"}
    assert entry["ts"]
    assert entry["result"] == "hello world"
    assert entry["upstream_title"] == "Prompt"


def test_execute_skips_log_when_disabled(general, tmp_log_dir):
    node = general.ShowAndSaveAnythingMie()
    out = node.execute("payload", save_to_log=False, log_file_name="skip.log")
    assert out["result"] == ("payload",)
    assert not (tmp_log_dir / "skip.log").exists()


def test_execute_swallows_log_errors(general, tmp_log_dir, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("nope")

    monkeypatch.setattr(general.ShowAndSaveAnythingMie, "_resolve_log_path", boom)
    node = general.ShowAndSaveAnythingMie()
    out = node.execute("payload")
    assert out["result"] == ("payload",)


def test_safe_repr_truncates_long_values(general):
    s = general._safe_repr("x" * 5000, limit=100)
    assert len(s) < 200
    assert s.startswith("x" * 100)
    assert "truncated" in s


def test_safe_repr_handles_none(general):
    assert general._safe_repr(None) == ""


def test_log_rotation_creates_backups(general, tmp_log_dir, monkeypatch):
    monkeypatch.setattr(general.ShowAndSaveAnythingMie, "LOG_MAX_BYTES", 200)
    monkeypatch.setattr(general.ShowAndSaveAnythingMie, "LOG_BACKUP_COUNT", 1)
    node = general.ShowAndSaveAnythingMie()
    payload = "x" * 150
    for _ in range(8):
        node.execute(payload, log_file_name="rot.log")

def test_execute_default_saves_log(general, tmp_log_dir):
    # The new node's whole point is logging; save_to_log defaults to True.
    node = general.ShowAndSaveAnythingMie()
    node.execute("data")
    log_path = tmp_log_dir / "show_anything.log"
    assert log_path.is_file()
    entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["result"] == "data"

def test_show_anything_node_unchanged(general):
    # The original ShowAnythingMie still has its original single input.
    inputs = general.ShowAnythingMie.INPUT_TYPES()
    assert list(inputs["required"].keys()) == ["anything"]
    assert "optional" not in inputs
    assert "hidden" not in inputs
    node = general.ShowAnythingMie()
    out = node.execute("hello")
    assert out["result"] == ("hello",)
