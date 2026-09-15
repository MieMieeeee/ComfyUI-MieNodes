# -*- coding: utf-8 -*-
"""End-to-end tests for the ``MiniMaxH3StoryboardGenerator`` node with a
stubbed LLM connector (FakeConnector pattern from
``test_h3_prompt_generator.py``)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
LLM_DIR = PROJECT_DIR / "nodes" / "llm"


def _ensure_pkg(fqn: str, path: Path | None = None):
    if fqn in sys.modules:
        return sys.modules[fqn]
    mod = types.ModuleType(fqn)
    if path is not None:
        mod.__path__ = [str(path)]
    mod.__package__ = fqn
    sys.modules[fqn] = mod
    return mod


def _load_file(fqn: str, path: Path):
    if fqn in sys.modules:
        del sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sb():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"
    )
    _load_file(
        "_mienodes_internal.nodes.llm.h3_prompts", LLM_DIR / "h3_prompts.py"
    )
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
        LLM_DIR / "minimax_h3_storyboard_prompts.py",
    )
    gen = _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_generator",
        LLM_DIR / "minimax_h3_storyboard_generator.py",
    )
    return gen


def _shot_payload(i: int, **over):
    shot = {
        "id": f"scene_{i:02d}",
        "description": f"Shot {i}: the courtyard moment develops.",
        "shot_type": "medium_shot" if i % 2 else "close_up",
        "camera_movement": "slow_push_in",
        "transition_in": "fade_from_black" if i == 1 else "hard_cut",
        "duration_seconds": 8,
        "narrative_beat": "establish" if i == 1 else "rising_action",
        "characters": ["young_woman"],
        "props": ["porcelain_bowl"],
        "notes": "Carry the same bowl.",
    }
    shot.update(over)
    return shot


class FakeConnector:
    model = "fake-model"

    def get_state(self):
        return "fake-state"


class ScriptedConnector(FakeConnector):
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("scripted connector exhausted")
        return self.replies.pop(0)


def _board(n=3):
    return json.dumps([_shot_payload(i) for i in range(1, n + 1)])


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_e2e_returns_three_outputs(sb):
    conn = ScriptedConnector([_board(3)])
    enhancer = sb.H3StoryboardEnhancer(conn)
    out = enhancer(
        "A Jiangnan courtyard in high summer",
        3,
        style="narrative_arc",
        genre="cinematic-story - 电影短片/MV/戏剧",
        language="en",
        output_format="table",
        seed=1,
    )
    assert len(conn.calls) == 1
    system = conn.calls[0][0]["content"]
    user = conn.calls[0][1]["content"]
    assert "storyboard" in system.lower()
    assert "Jiangnan courtyard" in user
    assert "exactly 3 shots" in user
    shots = json.loads(out["shots_json"])
    assert len(shots) == 3 and out["shot_count"] == 3
    assert out["storyboard_text"].startswith("# Storyboard")
    assert "scene_01" in out["storyboard_text"]


def test_e2e_all_five_styles(sb):
    for style in sb.STYLES:
        conn = ScriptedConnector([_board(2)])
        out = sb.H3StoryboardEnhancer(conn)(
            "a lighthouse keeper's morning",
            2,
            style=style,
            genre="none - 不指定",
            language="en",
            output_format="table",
        )
        assert out["shot_count"] == 2, style


def test_e2e_empty_concept_uses_default(sb):
    conn = ScriptedConnector([_board(1)])
    out = sb.H3StoryboardEnhancer(conn)(
        "   ",
        1,
        style="narrative_arc",
        genre="none - 不指定",
        language="en",
    )
    assert out["shot_count"] == 1
    # Default concept text reached the user turn.
    assert "cinematic moment" in conn.calls[0][1]["content"]


def test_e2e_count_mismatch_keeps_actual(sb):
    conn = ScriptedConnector([_board(2)])  # asked 5, got 2
    out = sb.H3StoryboardEnhancer(conn)(
        "concept", 5, style="narrative_arc", language="en"
    )
    assert out["shot_count"] == 2
    assert "kept actual" in out["storyboard_text"]


def test_e2e_output_formats_render(sb):
    for fmt, marker in (
        ("table", "| # | ID |"),
        ("detailed", "### 1."),
        ("minimal", "1. `scene_01`"),
    ):
        conn = ScriptedConnector([_board(2)])
        out = sb.H3StoryboardEnhancer(conn)(
            "concept", 2, style="narrative_arc", language="en", output_format=fmt
        )
        assert marker in out["storyboard_text"], fmt


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_parse_retry_then_success(sb):
    conn = ScriptedConnector(["sorry, I cannot", _board(2)])
    out = sb.H3StoryboardEnhancer(conn)(
        "concept", 2, style="narrative_arc", language="en"
    )
    assert out["shot_count"] == 2
    assert len(conn.calls) == 2  # one retry
    # The retry carries a corrective user turn, not a plain repeat.
    assert len(conn.calls[1]) == 3
    assert conn.calls[1][2]["role"] == "user"
    retry_text = conn.calls[1][2]["content"].lower()
    assert "json array" in retry_text and "only" in retry_text


def test_parse_failure_raises_after_retry(sb):
    conn = ScriptedConnector(["sorry, I cannot do JSON", "still not json"])
    with pytest.raises(RuntimeError, match="reply head"):
        sb.H3StoryboardEnhancer(conn)(
            "concept", 2, style="narrative_arc", language="en"
        )
    assert len(conn.calls) == 2


def test_think_block_stripped(sb):
    fenced = "<think>reasoning...</think>\n" + _board(1)
    conn = ScriptedConnector([fenced])
    out = sb.H3StoryboardEnhancer(conn)(
        "concept", 1, style="narrative_arc", language="en"
    )
    assert out["shot_count"] == 1


# --------------------------------------------------------------------------- #
# is_changed
# --------------------------------------------------------------------------- #
def test_is_changed_stable_and_seed_sensitive(sb):
    node = sb.MiniMaxH3StoryboardGenerator()
    kwargs = dict(
        concept="hello",
        shot_count=5,
        style="narrative_arc - 叙事弧光/三幕",
        genre="cinematic-story - 电影短片/MV/戏剧",
        language="en",
        output_format="table - Markdown表格",
        seed=0,
    )
    a = node.is_changed(FakeConnector(), **kwargs)
    b = node.is_changed(FakeConnector(), **kwargs)
    assert a == b
    kwargs["seed"] = 1
    assert node.is_changed(FakeConnector(), **kwargs) != a
    kwargs["seed"] = 0
    kwargs["shot_count"] = 6
    assert node.is_changed(FakeConnector(), **kwargs) != a


def test_node_generate_wraps_enhancer(sb):
    conn = ScriptedConnector([_board(2)])
    node = sb.MiniMaxH3StoryboardGenerator()
    text, shots_json, count = node.generate_storyboard(
        conn,
        "concept",
        2,
        "narrative_arc - 叙事弧光/三幕",
        "none - 不指定",
        "en",
        "minimal - 仅id+描述",
        seed=0,
        timeout=60,
    )
    assert count == 2
    assert len(json.loads(shots_json)) == 2
    assert "1. `scene_01`" in text
