"""
Pytest conftest for ComfyUI-MieNodes loop module tests.

Strategy: The project directory (ComfyUI-MieNodes) has hyphens in its name,
so it cannot be imported as a Python package directly. We use importlib to
load loop.py into a fake package context, after mocking all ComfyUI runtime
dependencies.
"""

import sys
import os
import types
import importlib.util
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Pre-register the root package and all its submodules as stubs so that
# pytest can import the root __init__.py (which uses relative imports)
# without failing.  This must happen before any conftest-driven import.
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The normalized package name (hyphens → underscores)
_PKG = "comfyui_mienodes"
if _PKG not in sys.modules:
    _root = types.ModuleType(_PKG)
    _root.__path__ = [_PROJECT_DIR]
    _root.__package__ = _PKG
    sys.modules[_PKG] = _root
# Stub every sub-module that __init__.py tries to import via relative imports
for _sub in (
    "common",
    "caption_file_operator",
    "downloader",
    "loop",
    "utils",
    "set_general_llm_service_connector",
    "llm_prompt_generator",
    "llm_service_connector",
    "general_llm_service_connector",
    "image_nodes",
    "format_nodes",
):
    _fqn = f"{_PKG}.{_sub}"
    if _fqn not in sys.modules:
        sys.modules[_fqn] = MagicMock()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 1. Mock ComfyUI runtime modules (not available outside ComfyUI)
# ---------------------------------------------------------------------------
for _mod_name in [
    "comfy_execution",
    "comfy_execution.graph_utils",
    "comfy",
    "comfy.graph_utils",
]:
    sys.modules.setdefault(_mod_name, MagicMock())

# ---------------------------------------------------------------------------
# 2. Load utils.py as a standalone module
#    loop.py does: from .utils import any_typ, mie_log, add_suffix
#    We load utils.py first so it can be wired into the fake package.
# ---------------------------------------------------------------------------
_utils_spec = importlib.util.spec_from_file_location(
    "utils",
    os.path.join(PROJECT_DIR, "utils.py"),
    submodule_search_locations=[],
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# ---------------------------------------------------------------------------
# 3. Create a fake parent package so relative imports in loop.py work
#    loop.py uses: from .utils import ...
#    Python resolves ".utils" relative to __package__.
#    We create "mie_pkg" as a fake package with __path__ pointing to PROJECT_DIR.
# ---------------------------------------------------------------------------
_fake_pkg = types.ModuleType("mie_pkg")
_fake_pkg.__path__ = [PROJECT_DIR]
_fake_pkg.__package__ = "mie_pkg"
sys.modules["mie_pkg"] = _fake_pkg
sys.modules["mie_pkg.utils"] = _utils_mod

# ---------------------------------------------------------------------------
# 4. Load loop.py with the fake package context
# ---------------------------------------------------------------------------
_loop_spec = importlib.util.spec_from_file_location(
    "mie_pkg.loop",
    os.path.join(PROJECT_DIR, "loop.py"),
    submodule_search_locations=[],
)
loop = importlib.util.module_from_spec(_loop_spec)
loop.__package__ = "mie_pkg"
sys.modules["mie_pkg.loop"] = loop
sys.modules["loop"] = loop  # also accessible as plain "loop"

_loop_spec.loader.exec_module(loop)

# ---------------------------------------------------------------------------
# Expose commonly-used items for easy test imports
# ---------------------------------------------------------------------------
EMPTY_IMAGE = loop.EMPTY_IMAGE
EMPTY_IMAGES = loop.EMPTY_IMAGES
RUNTIME_STORE = loop.RUNTIME_STORE

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
import pytest


@pytest.fixture(autouse=True)
def reset_runtime_store():
    """Reset RUNTIME_STORE before and after each test."""
    store = loop.RUNTIME_STORE
    store["collectors"] = {"image": {}, "text": {}, "json": {}}
    store["state_objects"] = {"image": {}}
    store["meta"] = {}
    store["_detect_cache"] = {}
    yield
    store["collectors"] = {"image": {}, "text": {}, "json": {}}
    store["state_objects"] = {"image": {}}
    store["meta"] = {}
    store["_detect_cache"] = {}


@pytest.fixture
def sample_loop_ctx():
    """Return a valid loop_ctx dict for testing."""
    return {
        "version": 3,
        "loop_id": "test_loop",
        "run_id": "test_run_123",
        "mode": "for_each",
        "index": 0,
        "count": 3,
        "is_last": False,
        "params_list": [{"value": 1}, {"value": 2}, {"value": 3}],
        "current_params": {"value": 1},
        "state": {},
        "collectors": {
            "image": {"ref": None, "count": 0},
            "text": {"ref": None, "count": 0},
            "json": {"ref": None, "count": 0},
        },
        "meta": {
            "body_in_id": "10",
            "body_out_id": "20",
            "end_id": "30",
        },
    }


@pytest.fixture
def sample_dynprompt():
    """Return a dict representing a simple loop graph for testing."""
    return {
        "1": {
            "class_type": "MieLoopStart|Mie",
            "inputs": {
                "loop_id": "test_loop",
                "params_mode": "int_list",
                "int_list": "8,9,10",
                "string_list": "",
                "json_list": "[]",
                "initial_state_json": "{}",
            },
        },
        "10": {
            "class_type": "MieLoopBodyIn|Mie",
            "inputs": {
                "loop_ctx": ["1", 0],
                "anchor": ["1", 0],
            },
        },
        "15": {
            "class_type": "KSampler|Mie",
            "inputs": {
                "model": ["5", 0],
                "steps": ["12", 0],
                "loop_ctx": ["10", 0],
            },
        },
        "20": {
            "class_type": "MieLoopBodyOut|Mie",
            "inputs": {
                "loop_ctx": ["15", 0],
                "value_string": "output",
                "state_json": "{}",
            },
        },
        "30": {
            "class_type": "MieLoopEnd|Mie",
            "inputs": {
                "loop_ctx": ["20", 0],
                "state_json": "{}",
            },
        },
    }


# ---------------------------------------------------------------------------
# Fake graph helpers – mimic GraphBuilder / GraphNode for unit tests
# ---------------------------------------------------------------------------
class FakeGraphNode:
    """Mimics GraphBuilder's Node class for testing.

    Matches comfy_execution.graph_utils.Node interface:
    - .id: str — the full node ID (prefix + id)
    - .class_type: str
    - .inputs: dict
    - .override_display_id: str | None
    - .out(index) → [self.id, index]  (a list, matching real Node.out)
    - .set_override_display_id(id)
    - .serialize() → dict with class_type, inputs, optional override_display_id
    """

    def __init__(self, node_id, class_type, inputs):
        self.id = str(node_id)
        self.class_type = class_type
        self.inputs = inputs if isinstance(inputs, dict) else {}
        self.override_display_id = None

    def out(self, index):
        return [self.id, index]

    def set_input(self, key, value):
        if value is None:
            self.inputs.pop(key, None)
        else:
            self.inputs[key] = value

    def get_input(self, key):
        return self.inputs.get(key)

    def set_override_display_id(self, override_display_id):
        self.override_display_id = override_display_id

    def serialize(self):
        serialized = {
            "class_type": self.class_type,
            "inputs": self.inputs,
        }
        if self.override_display_id is not None:
            serialized["override_display_id"] = self.override_display_id
        return serialized


class FakeGraphBuilder:
    """Mimics GraphBuilder for testing _build_expand_graph_for_next_round.

    Matches comfy_execution.graph_utils.GraphBuilder interface:
    - node(class_type, id=None, **kwargs) → FakeGraphNode
    - finalize() → dict of node_id → serialized node
    """

    def __init__(self):
        self.nodes = {}
        self.id_gen = 1
        self.prefix = "fake."

    def node(self, class_type, node_id=None, **kwargs):
        if node_id is None:
            node_id = str(self.id_gen)
            self.id_gen += 1
        full_id = self.prefix + str(node_id)
        if full_id in self.nodes:
            return self.nodes[full_id]
        n = FakeGraphNode(full_id, class_type, kwargs)
        self.nodes[full_id] = n
        return n

    def finalize(self):
        return {nid: n.serialize() for nid, n in self.nodes.items()}


@pytest.fixture
def fake_graph_builder():
    """Provide a FakeGraphBuilder instance for testing."""
    return FakeGraphBuilder()
