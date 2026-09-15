# -*- coding: utf-8 -*-
"""Tests for ``minimax_h3_storyboard_prompts`` (whitelists, JSON
extraction, normalization, markdown rendering).

Uses the same ``_mienodes_internal`` loader-stub injection as
``test_h3_prompt_generator.py`` (the project root has a hyphen and
cannot be imported as a normal package).
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"


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
def sbp():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", PROJECT_DIR / "nodes" / "llm")
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"
    )
    _load_file(
        "_mienodes_internal.nodes.llm.h3_prompts",
        PROJECT_DIR / "nodes" / "llm" / "h3_prompts.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
        PROJECT_DIR / "nodes" / "llm" / "minimax_h3_storyboard_prompts.py",
    )


# --------------------------------------------------------------------------- #
# Dropdown parsing + advice coverage
# --------------------------------------------------------------------------- #
def test_parse_style_and_output_format(sbp):
    for s in sbp.STYLES:
        code = sbp.parse_style(s)
        assert code in sbp.STYLE_CODES
    assert sbp.parse_style("narrative_arc") == "narrative_arc"  # bare code
    assert sbp.parse_style("") == ""
    for f in sbp.OUTPUT_FORMATS:
        assert sbp.parse_output_format(f) in sbp.OUTPUT_FORMAT_CODES
    assert sbp.parse_output_format("detailed") == "detailed"


def test_style_advice_covers_all_codes(sbp):
    for code in sbp.STYLE_CODES:
        assert sbp.STYLE_ADVICE[code].strip()
        assert sbp.style_advice(code) == sbp.STYLE_ADVICE[code]
    # Unknown style falls back to the default style's advice.
    assert sbp.style_advice("nope") == sbp.STYLE_ADVICE[sbp.DEFAULT_STYLE]


def test_language_name(sbp):
    assert sbp.language_name("en") == "English"
    assert sbp.language_name("zh") == "Chinese"
    assert sbp.language_name(None) == "en"


def test_build_user_text_contains_everything(sbp):
    text = sbp.build_user_text(
        "一个江南院落的夏天",
        5,
        "narrative_arc",
        "action - 动作戏/打斗/飙车",
        "zh",
    )
    assert "一个江南院落的夏天" in text
    assert "exactly 5 shots" in text
    assert "narrative_arc" in text
    assert "motion blur" in text  # action-category advice reaches the template
    assert "zh" in text


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def test_extract_json_array_bare(sbp):
    data = sbp.extract_json_array('[{"id": "a"}, {"id": "b"}]')
    assert data == [{"id": "a"}, {"id": "b"}]


def test_extract_json_array_fenced(sbp):
    text = '```json\n[{"id": "a"}]\n```'
    assert sbp.extract_json_array(text) == [{"id": "a"}]
    text2 = '```\n[{"id": "a"}]\n```'
    assert sbp.extract_json_array(text2) == [{"id": "a"}]


def test_extract_json_array_prose_wrapped(sbp):
    text = 'Here is your board:\n\n[{"id": "a"}, {"id": "b"}]\n\nGood luck!'
    assert len(sbp.extract_json_array(text)) == 2


def test_extract_json_array_nested_brackets(sbp):
    text = '[{"id": "a", "notes": "array [x] inside"}, {"id": "b"}]'
    out = sbp.extract_json_array(text)
    assert out[0]["notes"] == "array [x] inside"


def test_extract_json_array_invalid(sbp):
    with pytest.raises(ValueError):
        sbp.extract_json_array("no json here at all")
    with pytest.raises(ValueError):
        sbp.extract_json_array('{"object": "not an array"}')
    with pytest.raises(ValueError):
        sbp.extract_json_array("   ")


def test_extract_json_array_object_with_shots(sbp):
    """Models sometimes wrap the board in {"shots": [...]} despite the
    prompt — unwrap instead of failing."""
    text = '{"shots": [{"id": "a"}, {"id": "b"}]}'
    assert sbp.extract_json_array(text) == [{"id": "a"}, {"id": "b"}]


def test_extract_json_array_repairs_trailing_commas(sbp):
    text = '[{"id": "a",}, {"id": "b"},]'
    assert sbp.extract_json_array(text) == [{"id": "a"}, {"id": "b"}]


def test_extract_json_array_repairs_fullwidth_quotes(sbp):
    text = '[{"id": “a”, "description": "d"}]'
    assert sbp.extract_json_array(text) == [{"id": "a", "description": "d"}]


def test_repair_helper_idempotent_on_clean_json(sbp):
    clean = '[{"id": "a"}, {"id": "b"}]'
    assert sbp._repaired(clean) == clean


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _good_shot(**over):
    shot = {
        "id": "scene_01",
        "description": "A woman sets a porcelain bowl on a wooden table.",
        "shot_type": "medium_shot",
        "camera_movement": "slow_push_in",
        "transition_in": "fade_from_black",
        "duration_seconds": 8,
        "narrative_beat": "establish",
        "characters": ["young_woman"],
        "props": ["porcelain_bowl"],
        "notes": "Anchor her appearance.",
    }
    shot.update(over)
    return shot


def test_normalize_shots_passes_valid(sbp):
    shots, warnings = sbp.normalize_shots([_good_shot()], 1)
    assert len(shots) == 1 and not warnings
    assert shots[0]["id"] == "scene_01"
    assert shots[0]["duration_seconds"] == 8


def test_normalize_shots_whitelist_fallbacks(sbp):
    raw = _good_shot(shot_type="macro_lens_shot", transition_in="jump_flip")
    shots, warnings = sbp.normalize_shots([raw], 1)
    assert shots[0]["shot_type"] == "medium_shot"
    assert shots[0]["transition_in"] == "fade_from_black"  # first shot default
    assert any("shot_type" in w for w in warnings)
    # Non-first shot falls back to hard_cut.
    raw2 = _good_shot(id="scene_02", transition_in="jump_flip")
    shots2, _ = sbp.normalize_shots(
        [_good_shot(), raw2], 2
    )
    assert shots2[1]["transition_in"] == "hard_cut"


def test_normalize_shots_ids_default_slug_and_dedupe(sbp):
    raw = [
        {"description": "one"},
        {"description": "two", "id": "Shot Two!"},
        {"description": "three", "id": "shot_two"},
    ]
    shots, warnings = sbp.normalize_shots(raw, 3)
    assert shots[0]["id"] == "scene_01"
    assert shots[1]["id"] == "shot_two"
    assert shots[2]["id"] == "shot_two_2"
    assert any("renamed" in w for w in warnings)


def test_normalize_shots_duration_clamp_and_str_lists(sbp):
    raw = _good_shot(duration_seconds=999, characters="young_woman", props=None)
    shots, warnings = sbp.normalize_shots([raw], 1)
    assert shots[0]["duration_seconds"] == 60
    assert shots[0]["characters"] == ["young_woman"]
    assert shots[0]["props"] == []
    assert any("clamped" in w for w in warnings)


def test_normalize_shots_count_reconciliation(sbp):
    # More than expected -> trimmed.
    raw = [_good_shot(id=f"s{i}") for i in range(5)]
    shots, warnings = sbp.normalize_shots(raw, 3)
    assert len(shots) == 3
    assert any("trimmed" in w for w in warnings)
    # Fewer than expected -> kept, warned.
    shots2, warnings2 = sbp.normalize_shots(raw[:2], 5)
    assert len(shots2) == 2
    assert any("kept actual" in w for w in warnings2)


def test_normalize_shots_rejects_empty_description(sbp):
    with pytest.raises(ValueError):
        sbp.normalize_shots([{"id": "x", "description": "  "}], 1)
    with pytest.raises(ValueError):
        sbp.normalize_shots([], 3)
    with pytest.raises(ValueError):
        sbp.normalize_shots(["   "], 1)


def test_normalize_shots_accepts_bare_strings(sbp):
    shots, _ = sbp.normalize_shots(["A quiet courtyard at noon."], 1)
    assert shots[0]["description"] == "A quiet courtyard at noon."


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_markdown_table(sbp):
    shots = [
        _good_shot(),
        _good_shot(id="scene_02", description="Ice | clink with a | pipe char"),
    ]
    text = sbp.render_storyboard_markdown(shots, "table")
    assert text.startswith("# Storyboard")
    assert "| # | ID |" in text
    assert "\\|" in text  # pipes escaped
    assert text.count("\n| 2 |") == 1


def test_render_markdown_detailed_and_minimal(sbp):
    shots = [_good_shot(), _good_shot(id="scene_02")]
    detailed = sbp.render_storyboard_markdown(shots, "detailed")
    assert "### 1. `scene_01`" in detailed
    assert "**Description:**" in detailed
    assert "Notes:" in detailed
    minimal = sbp.render_storyboard_markdown(shots, "minimal")
    assert "1. `scene_01`" in minimal
    assert "**Description:**" not in minimal
    # Unknown format falls back to table.
    fb = sbp.render_storyboard_markdown(shots, "bogus")
    assert "| # | ID |" in fb


def test_render_markdown_includes_warnings(sbp):
    text = sbp.render_storyboard_markdown(
        [_good_shot()], "minimal", warnings=["trimmed to 3"]
    )
    assert "**Warnings:**" in text
    assert "- trimmed to 3" in text


def test_shots_to_json_string_roundtrip(sbp):
    shots, _ = sbp.normalize_shots([_good_shot()], 1)
    text = sbp.shots_to_json_string(shots)
    parsed = json.loads(text)
    assert parsed == shots
    # Round-trips through the loop node's parser contract (JSON array).
    assert isinstance(parsed, list) and parsed[0]["id"] == "scene_01"
