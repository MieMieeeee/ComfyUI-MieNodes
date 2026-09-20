# -*- coding: utf-8 -*-
"""Tests for reference / keyframe modes (t2va / i2va / fl2va / ref2va)
on MiniMaxH3LoopPromptGenerator and minimax_h3_loop_prompts.

Covers manifest parsing + validation, six-section split, schema-aware
plan validation, per-mode label policy, the official workflow round-trip
ground truth lock, and the full generator E2E paths for each keyframe
mode (including real-image grounding against the user's ComfyUI input
directory when present)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"
LLM_DIR = PROJECT_DIR / "nodes" / "llm"
PLUGIN_DIR = Path(
    r"C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-Context-Loop"
)
COMFYUI_INPUT_DIR = Path(
    r"C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/input"
)


def _ensure_pkg(fqn, path=None):
    if fqn in sys.modules:
        return sys.modules[fqn]
    mod = types.ModuleType(fqn)
    if path is not None:
        mod.__path__ = [str(path)]
    mod.__package__ = fqn
    sys.modules[fqn] = mod
    return mod


def _load_file(fqn, path):
    if fqn in sys.modules:
        del sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lp():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.llm", LLM_DIR)
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"
    )
    _load_file("_mienodes_internal.nodes.llm.h3_prompts", LLM_DIR / "h3_prompts.py")
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
        LLM_DIR / "minimax_h3_storyboard_prompts.py")
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py")


@pytest.fixture(scope="module")
def lg(lp):
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py")


# --------------------------------------------------------------------------- #
# Reference mode dropdown / helpers
# --------------------------------------------------------------------------- #
def test_reference_mode_dropdown_format(lp):
    assert lp.REFERENCE_MODES == (
        "t2va - 文生视频链(默认)",
        "i2va - 首帧关键帧(文生视频+首帧图)",
        "fl2va - 首尾帧关键帧(首帧+逐场尾帧)",
        "ref2va - 参考图(N张/全场景)")
    assert lp.REFERENCE_MODE_CODES == ("t2va", "i2va", "fl2va", "ref2va")


def test_parse_reference_mode(lp):
    assert lp.parse_reference_mode("t2va - 文生视频链(默认)") == "t2va"
    assert lp.parse_reference_mode("ref2va - 参考图(N张/全场景)") == "ref2va"
    assert lp.parse_reference_mode("weird") == "t2va"
    assert lp.parse_reference_mode("") == "t2va"


def test_schema_for_mode(lp):
    assert lp.schema_for_mode("t2va") == lp.SCHEMA_THREE
    assert lp.schema_for_mode("i2va") == lp.SCHEMA_THREE
    assert lp.schema_for_mode("fl2va") == lp.SCHEMA_THREE
    assert lp.schema_for_mode("ref2va") == lp.SCHEMA_SIX

# --------------------------------------------------------------------------- #
# Manifest parsing
# --------------------------------------------------------------------------- #
def test_parse_references_text_json_form(lp):
    manifest = lp.parse_references_text(json.dumps([
        {"slot": "Picture 1", "about": "courier face, yellow jacket",
         "role": "identity"},
        {"slot": "Picture 2", "about": "wet greenhouse entrance",
         "role": "destination"},
    ], ensure_ascii=False))
    assert len(manifest) == 2
    assert manifest[0]["slot"] == "Picture 1"
    assert manifest[0]["about"] == "courier face, yellow jacket"
    assert manifest[0]["role"] == "identity"
    assert manifest[1]["role"] == "destination"


def test_parse_references_text_natural_line(lp):
    manifest = lp.parse_references_text(
        "Picture 1: courier face, yellow jacket\n"
        "Picture 2: wet greenhouse entrance\n"
    )
    assert len(manifest) == 2
    assert manifest[0]["slot"] == "Picture 1"
    # role default: slot 1 -> identity; others -> destination
    assert manifest[0]["role"] == "identity"
    assert manifest[1]["role"] == "destination"


def test_parse_references_text_bracket_slot(lp):
    manifest = lp.parse_references_text("<Picture 1>: courier face")
    assert manifest[0]["slot"] == "Picture 1"
    assert manifest[0]["role"] == "identity"


def test_parse_references_text_empty(lp):
    assert lp.parse_references_text("") == []
    assert lp.parse_references_text("   \n  ") == []


def test_parse_references_text_rejects_video_slot(lp):
    # P0 video extension: Video slots are now LEGAL in ref2va manifests
    # (Pictures-then-Videos order); Audio slots remain rejected.
    m = lp.parse_references_text(json.dumps([
        {"slot": "Video 1", "about": "x"},
    ]))
    assert m[0]["slot"] == "Video 1"
    with pytest.raises(ValueError, match="Audio"):
        lp.parse_references_text(json.dumps([
            {"slot": "Audio 1", "about": "x"},
        ]))


def test_parse_references_text_rejects_gap_numbering(lp):
    # JSON form skips line context but still rejects; natural-line form
    # includes line number.
    with pytest.raises(ValueError, match="contiguous"):
        lp.parse_references_text("Picture 1: a\nPicture 3: c\n")
    with pytest.raises(ValueError, match="contiguous"):
        lp.parse_references_text(json.dumps([
            {"slot": "Picture 1", "about": "a"},
            {"slot": "Picture 3", "about": "c"},
        ]))


def test_parse_references_text_rejects_unknown_role(lp):
    with pytest.raises(ValueError, match="role"):
        lp.parse_references_text(json.dumps([
            {"slot": "Picture 1", "about": "a", "role": "bogus"},
        ]))


def test_parse_references_text_natural_line_line_number(lp):
    with pytest.raises(ValueError, match="line 2"):
        lp.parse_references_text("Picture 1: a\nnot a valid line\n")


def test_parse_references_text_alias_keys(lp):
    manifest = lp.parse_references_text(json.dumps([
        {"picture": "Picture 1", "description": "face"},
    ]))
    assert manifest[0]["slot"] == "Picture 1"
    assert manifest[0]["about"] == "face"


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "manifest,mode,expect_error",
    [
        # t2va: must be empty
        ([], "t2va", False),
        ([{"slot": "Picture 1", "about": "x", "role": "identity"}], "t2va", True),
        # i2va: exactly [Picture 1]
        ([{"slot": "Picture 1", "about": "x", "role": "identity"}], "i2va", False),
        ([], "i2va", True),
        ([{"slot": "Picture 2", "about": "x", "role": "destination"}], "i2va", True),
        # fl2va: exactly [Picture 1, Picture 2]
        (
            [
                {"slot": "Picture 1", "about": "x", "role": "identity"},
                {"slot": "Picture 2", "about": "y", "role": "destination"},
            ],
            "fl2va",
            False),
        ([{"slot": "Picture 1", "about": "x", "role": "identity"}], "fl2va", True),
        ([], "fl2va", True),
        # ref2va: 1..9 contiguous pictures
        (
            [
                {"slot": "Picture 1", "about": "x", "role": "identity"},
                {"slot": "Picture 2", "about": "y", "role": "destination"},
            ],
            "ref2va",
            False),
        (
            [{"slot": f"Picture {i}", "about": "z", "role": "destination"} for i in range(1, 10)],
            "ref2va",
            False),
        # >9 pictures rejected
        (
            [{"slot": f"Picture {i}", "about": "z", "role": "destination"} for i in range(1, 11)],
            "ref2va",
            True),
        ([], "ref2va", True),
    ])
def test_validate_manifest_matrix(lp, manifest, mode, expect_error):
    errors = lp.validate_manifest(manifest, mode)
    if expect_error:
        assert errors, f"expected errors for mode={mode}, got none"
    else:
        assert not errors, f"unexpected errors for mode={mode}: {errors}"


# --------------------------------------------------------------------------- #
# Label scanners
# --------------------------------------------------------------------------- #
def test_find_native_labels(lp):
    labels = lp.find_native_labels("<Picture 1> matches <Subject 2>; ignore <Picture 11>")
    assert ("Picture", 1) in labels
    assert ("Subject", 2) in labels
    # 11 is two digits and matches the regex (cap is 99 in our regex).
    assert ("Picture", 11) in labels


def test_find_alias_tokens(lp):
    assert lp.find_alias_tokens("@hero_face and @performance") == ["hero_face", "performance"]
    # Not an email address.
    assert lp.find_alias_tokens("foo@bar.com") == []


def test_find_semantic_anchors(lp):
    out = lp.find_semantic_anchors("#dialogue[3.5s] then #warp")
    assert out == [("dialogue", "3.5"), ("warp", "")]
    # No leading identifier char.
    assert lp.find_semantic_anchors("foo#bar") == []


# --------------------------------------------------------------------------- #
# Six-section split
# --------------------------------------------------------------------------- #
def _ref2va_clip_reply(scene_id=1, with_subject=True):
    parts = [
        "subject_definitions:",
        (
            "<Picture 1> defines <Subject 1>, the exact adult courier, "
            "including mustard jacket and bicycle." if with_subject else
            "<Picture 1> defines the courier scene."
        ),
        "",
        "summary:",
        "[reference generation] One continuous delivery moves the courier through the greenhouse.",
        "",
        "retention_analysis:",
        "<Subject 1>: fully_preserved - retain face, jacket, bicycle across the shot.",
        "<Picture 1>: reference - use opening identity.",
        "",
        "detailed_description:",
        f"[Shot 1] The courier enters, the camera tracks him laterally. "
        "End mid-action so the next clip can continue it.",
        "",
        "overall_soundscape:",
        "Light rain on greenhouse glass, wet tire roll, footsteps.",
        "",
        "non_diegetic_music:",
        "No non-diegetic music.",
    ]
    return "\n".join(parts)


def test_split_six_sections_canonical(lp):
    lines = lp.split_six_sections(_ref2va_clip_reply(lp))
    # Six headers in order with one blank line between each section.
    headers = [ln for ln in lines if lp.SIX_SECTION_FIELDS and ln.endswith(":")
               and ln.rstrip(":").strip() in lp.SIX_SECTION_FIELDS]
    assert headers == [f"{h}:" for h in lp.SIX_SECTION_FIELDS]
    assert lines[0] == "subject_definitions:"
    assert "Picture 1" in lines[1]
    # One blank line between sections.
    for h_idx in range(1, 6):
        line_idx = next(i for i, ln in enumerate(lines)
                        if ln == f"{lp.SIX_SECTION_FIELDS[h_idx]}:")
        assert lines[line_idx - 1] == ""
    # ends with non_diegetic_music
    assert lines[-2] == "non_diegetic_music:"
    assert lines[-1] == "No non-diegetic music."
    # The detailed_description body is somewhere in the lines after the
    # detailed_description: header.
    body_start = lines.index("detailed_description:") + 1
    assert any("[Shot 1]" in ln for ln in lines[body_start:])


def test_split_six_sections_tolerant_preamble(lp):
    text = "Sure! Here is the scene:\n\n" + _ref2va_clip_reply(lp)
    lines = lp.split_six_sections(text)
    assert lines[0] == "subject_definitions:"


def test_split_six_sections_missing_music_default(lp):
    text = _ref2va_clip_reply(lp).rsplit("non_diegetic_music:", 1)[0].rstrip()
    lines = lp.split_six_sections(text)
    assert lines[-2] == "non_diegetic_music:"
    assert lines[-1] == lp.DEFAULT_MUSIC_LINE


def test_split_six_sections_missing_section_raises(lp):
    text = _ref2va_clip_reply(lp)
    text = text.replace("subject_definitions:\n", "", 1)
    text = text.replace(
        "<Picture 1> defines <Subject 1>, the exact adult courier, "
        "including mustard jacket and bicycle.\n",
        "",
        1)
    with pytest.raises(ValueError, match="subject_definitions"):
        lp.split_six_sections(text)


def test_split_six_sections_empty_body_raises(lp):
    text = _ref2va_clip_reply(lp).replace(
        "<Picture 1> defines <Subject 1>, the exact adult courier, including mustard jacket and bicycle.",
        "")
    with pytest.raises(ValueError, match="empty subject_definitions"):
        lp.split_six_sections(text)


def test_description_body_schema_dispatch(lp):
    three = lp.split_three_sections(
        "integrated_multimodal_description:\n[Shot 1] she stirs.\n\n"
        "overall_soundscape:\nclink.\n\nnon_diegetic_music:\nno music.\n"
    )
    assert lp.description_body(three) == "[Shot 1] she stirs."
    assert lp.description_body(three, schema="three_section") == "[Shot 1] she stirs."

    six = lp.split_six_sections(_ref2va_clip_reply(lp))
    # six -> detailed_description body
    assert "camera tracks" in lp.description_body(six, schema="six_section")
    # six with bad schema raises (no join-everything fallback).
    with pytest.raises(ValueError, match="unknown schema"):
        lp.description_body(six, schema="bogus")


# --------------------------------------------------------------------------- #
# validate_plan(schema=)
# --------------------------------------------------------------------------- #
def _three_shot_plan(lp):
    return lp.build_plan(
        [
            {
                "id": "s1",
                "prompt": [
                    "integrated_multimodal_description:",
                    "[Shot 1] first.",
                    "",
                    "overall_soundscape:",
                    "rain.",
                    "",
                    "non_diegetic_music:",
                    "no music.",
                ],
                "length": 243,
                "seed": "1001",
            },
            {
                "id": "s2",
                "prompt": [
                    "integrated_multimodal_description:",
                    "[Shot 1] second.",
                    "",
                    "overall_soundscape:",
                    "rain.",
                    "",
                    "non_diegetic_music:",
                    "no music.",
                ],
                "length": 243,
                "seed": "1002",
            },
        ],
        ["Shared prefix."])


def _six_shot_plan(lp):
    return lp.build_plan(
        [
            {
                "id": "s1",
                "prompt": lp.split_six_sections(_ref2va_clip_reply(1)),
                "length": 243,
                "seed": "1001",
            },
            {
                "id": "s2",
                "prompt": lp.split_six_sections(_ref2va_clip_reply(2)),
                "length": 243,
                "seed": "1002",
            },
        ],
        [])


def test_validate_plan_three_section_clean(lp):
    plan = _three_shot_plan(lp)
    assert lp.validate_plan(plan, schema=lp.SCHEMA_THREE) == []
    assert lp.validate_plan(plan) == []  # default schema


def test_validate_plan_three_section_rejects_six_shape(lp):
    plan = _three_shot_plan(lp)
    # First shot prompt doesn't start with detailed_description
    plan["shots"][0]["prompt"] = [
        "detailed_description:", "[Shot 1] x", "", "overall_soundscape:", "y",
        "", "non_diegetic_music:", "n",
    ]
    errs = lp.validate_plan(plan, schema=lp.SCHEMA_SIX)
    assert any("detailed_description" in e for e in errs)


def test_validate_plan_six_section_clean(lp):
    plan = _six_shot_plan(lp)
    assert lp.validate_plan(plan, schema=lp.SCHEMA_SIX) == []


def test_validate_plan_six_section_rejects_bad_order(lp):
    plan = _six_shot_plan(lp)
    # Reorder: put summary before subject_definitions
    s = plan["shots"][0]["prompt"]
    subj_idx = s.index("subject_definitions:")
    sum_idx = s.index("summary:")
    s[subj_idx], s[sum_idx] = s[sum_idx], s[subj_idx]
    errs = lp.validate_plan(plan, schema=lp.SCHEMA_SIX)
    assert any("headers must be in order" in e for e in errs)


def test_validate_plan_six_section_rejects_missing_header(lp):
    plan = _six_shot_plan(lp)
    s = plan["shots"][0]["prompt"]
    del s[s.index("retention_analysis:")]
    errs = lp.validate_plan(plan, schema=lp.SCHEMA_SIX)
    assert any("retention_analysis" in e for e in errs)


def test_validate_plan_prompt_prefix_optional(lp):
    plan = _three_shot_plan(lp)
    del plan["prompt_prefix"]
    errs = lp.validate_plan(plan)
    assert errs == []  # prompt_prefix is optional (D3)
    # An UNEXPECTED extra key still errors.
    plan["bogus"] = "x"
    errs = lp.validate_plan(plan)
    assert any("top-level keys" in e for e in errs)


# --------------------------------------------------------------------------- #
# validate_label_policy per-mode matrix
# --------------------------------------------------------------------------- #
def test_validate_label_policy_aliases_rejected_everywhere(lp):
    for mode in lp.REFERENCE_MODE_CODES:
        plan = _three_shot_plan(lp)
        plan["prompt_prefix"] = ["@hero_face."]
        errs = lp.validate_label_policy(plan, mode, [])
        assert errs, f"expected alias error in {mode}"
        assert any("@alias" in e for e in errs)


def test_validate_label_policy_anchors_rejected_everywhere(lp):
    for mode in lp.REFERENCE_MODE_CODES:
        plan = _three_shot_plan(lp)
        plan["prompt_prefix"] = ["#dialogue"]
        errs = lp.validate_label_policy(plan, mode, [])
        assert errs, f"expected anchor error in {mode}"
        assert any("#tag" in e for e in errs)


def test_validate_label_policy_t2va_rejects_any_native(lp):
    plan = _three_shot_plan(lp)
    plan["prompt_prefix"] = ["Use <Picture 1> here."]
    errs = lp.validate_label_policy(plan, "t2va", [])
    assert any("not valid in t2va" in e for e in errs)


def test_validate_label_policy_i2va_scene1_picture1_only(lp):
    plan = _three_shot_plan(lp)
    plan["shots"][0]["prompt"][1] = "[Shot 1] <Picture 1> opens the scene."
    plan["shots"][1]["prompt"][1] = "[Shot 1] continuation, no labels."
    errs = lp.validate_label_policy(plan, "i2va",
        [{"slot": "Picture 1", "about": "x", "role": "identity"}])
    assert errs == []


def test_validate_label_policy_i2va_scene1_forbids_picture2(lp):
    plan = _three_shot_plan(lp)
    plan["shots"][0]["prompt"][1] = "[Shot 1] <Picture 2> is not allowed."
    errs = lp.validate_label_policy(plan, "i2va",
        [{"slot": "Picture 1", "about": "x", "role": "identity"}])
    assert any("i2va allows <Picture 1> only" in e for e in errs)


def test_validate_label_policy_i2va_scene2_forbids_any_label(lp):
    plan = _three_shot_plan(lp)
    plan["shots"][1]["prompt"][1] = "[Shot 1] <Picture 1> uses forbidden label."
    errs = lp.validate_label_policy(plan, "i2va",
        [{"slot": "Picture 1", "about": "x", "role": "identity"}])
    assert any("hides Picture 1 from continuation scenes" in e for e in errs)


def test_validate_label_policy_fl2va_scene1_allows_both(lp):
    plan = _three_shot_plan(lp)
    plan["shots"][0]["prompt"][1] = (
        "[Shot 1] <Picture 1> opens; reach <Picture 2> only on final frame."
    )
    plan["shots"][1]["prompt"][1] = "[Shot 1] converge on <Picture 1>."
    errs = lp.validate_label_policy(plan, "fl2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert errs == []


def test_validate_label_policy_fl2va_scene2_only_picture1(lp):
    plan = _three_shot_plan(lp)
    plan["shots"][0]["prompt"][1] = "[Shot 1] start at <Picture 1>."
    plan["shots"][1]["prompt"][1] = "[Shot 1] converge on <Picture 2>."  # WRONG
    errs = lp.validate_label_policy(plan, "fl2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert any("fl2va end target is <Picture 1>" in e for e in errs)


def test_validate_label_policy_fl2va_scene3_targets_picture1(lp):
    # Per-scene end target is <Picture 1> for EVERY scene N>=2
    # (verified against the upstream gate wiring — FrameIndexSwitch
    # exposes a single per-scene image under Picture 1).
    plan = lp.build_plan(
        [
            {
                "id": f"s{i}",
                "prompt": lp.split_three_sections(
                    "integrated_multimodal_description:\n[Shot 1] frame.\n\n"
                    "overall_soundscape:\nrain.\n\nnon_diegetic_music:\nno music.\n"
                ),
                "length": 243,
                "seed": f"100{i}",
            }
            for i in range(1, 4)
        ],
        ["prefix"])
    # Scene 3 with <Picture 1> PASSES (every scene 2+ targets Picture 1).
    plan["shots"][2]["prompt"][1] = "[Shot 1] converge on <Picture 1>."
    errs = lp.validate_label_policy(plan, "fl2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert errs == []


def test_validate_label_policy_fl2va_scene3_rejects_picture2(lp):
    # Scene 3 referencing <Picture 2> is an ERROR: the plugin exposes
    # no second picture in scenes 2+.  The fl2va end-target is
    # exclusively <Picture 1> there.
    plan = lp.build_plan(
        [
            {
                "id": f"s{i}",
                "prompt": lp.split_three_sections(
                    "integrated_multimodal_description:\n[Shot 1] frame.\n\n"
                    "overall_soundscape:\nrain.\n\nnon_diegetic_music:\nno music.\n"
                ),
                "length": 243,
                "seed": f"100{i}",
            }
            for i in range(1, 4)
        ],
        ["prefix"])
    plan["shots"][2]["prompt"][1] = "[Shot 1] converge on <Picture 2>."
    errs = lp.validate_label_policy(plan, "fl2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert any("fl2va end target is <Picture 1>" in e for e in errs), errs


def test_validate_label_policy_ref2va_subject_needs_paired_picture(lp):
    plan = _six_shot_plan(lp)
    # Use <Subject 1> in scene 1 without its paired <Picture 1>.
    body = plan["shots"][0]["prompt"]
    subj_line = next(i for i, ln in enumerate(body) if "Picture 1> defines <Subject 1>" in ln)
    body[subj_line] = body[subj_line].replace("<Picture 1>", "<Picture 9>")
    errs = lp.validate_label_policy(plan, "ref2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert any("not in the manifest" in e for e in errs)


def test_validate_label_policy_ref2va_subject_without_any_picture(lp):
    """Self-review finding 6: exercise the REAL paired-picture rule — a shot
    mentioning <Subject 1> while NO <Picture 1> label appears anywhere in
    that scene (not even in retention lines)."""
    plan = _six_shot_plan(lp)
    body = plan["shots"][0]["prompt"]
    # Scrub every <Picture N> label from the whole shot; keep <Subject 1>.
    scrubbed = [ln.replace("<Picture ", "(picture ") for ln in body]
    assert any("<Subject 1>" in ln for ln in scrubbed)
    assert not any("<Picture " in ln for ln in scrubbed)
    plan["shots"][0]["prompt"] = scrubbed
    errs = lp.validate_label_policy(plan, "ref2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert any(
        "needs its paired <Picture 1> in the same scene" in e for e in errs
    ), errs


def test_validate_label_policy_ref2va_subject_must_bind_to_identity(lp):
    # Manifest where slot 1 is non-identity -> <Subject 1> shouldn't be allowed.
    plan = _six_shot_plan(lp)
    body = plan["shots"][0]["prompt"]
    subj_line = next(i for i, ln in enumerate(body) if "Picture 1> defines <Subject 1>" in ln)
    body[subj_line] = body[subj_line].replace(
        "defines <Subject 1>, the exact adult courier", "defines <Subject 1>"
    )
    errs = lp.validate_label_policy(plan, "ref2va", [
        {"slot": "Picture 1", "about": "x", "role": "destination"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert any("no matching manifest" in e for e in errs)


# --------------------------------------------------------------------------- #
# build_reference_directive
# --------------------------------------------------------------------------- #
def test_build_reference_directive_t2va_empty(lp):
    assert lp.build_reference_directive("t2va", [], 1, 10) == ""


def test_build_reference_directive_i2va_scene1(lp):
    text = lp.build_reference_directive("i2va", [
        {"slot": "Picture 1", "about": "courier face, yellow jacket", "role": "identity"}
    ], 1, 10)
    assert text.startswith("At 0.00 seconds, <Picture 1> is fully referenced as the opening frame.")
    assert "courier face, yellow jacket" in text
    assert "smooth visual path" in text


def test_build_reference_directive_i2va_scene2_empty(lp):
    text = lp.build_reference_directive("i2va", [
        {"slot": "Picture 1", "about": "x", "role": "identity"}
    ], 2, 10)
    assert text == ""  # continuation must have zero native labels


def test_build_reference_directive_fl2va_scene1(lp):
    text = lp.build_reference_directive("fl2va", [
        {"slot": "Picture 1", "about": "courier at door", "role": "identity"},
        {"slot": "Picture 2", "about": "parcel on table", "role": "destination"},
    ], 1, 10)
    assert "<Picture 1> is the opening frame" in text
    assert "<Picture 2> is the closing frame" in text
    assert "courier at door" in text
    assert "closing pose the next scene arrives at" in text


def test_build_reference_directive_fl2va_scene2_picture1(lp):
    text = lp.build_reference_directive("fl2va", [
        {"slot": "Picture 1", "about": "outside", "role": "destination"},
        {"slot": "Picture 2", "about": "inside", "role": "destination"},
    ], 2, 10)
    assert "outside" in text
    assert "shown in <Picture 1>" in text
    assert "final frame without a cut" in text
    assert "<Picture 1>" in text
    assert "outside" in text
    assert "final frame without a cut" in text


def test_build_reference_directive_fl2va_scene3_picture1_fallback(lp):
    # Scene 3 in fl2va targets <Picture 1> — the FL2VA gate exposes
    # a single per-scene image under Picture 1 in every scene N>=2.
    # With a 2-entry manifest (legacy layout) scene 3 is an odd
    # scene, so the about text falls back to manifest[1].
    text = lp.build_reference_directive("fl2va", [
        {"slot": "Picture 1", "about": "outside", "role": "destination"},
        {"slot": "Picture 2", "about": "inside", "role": "destination"},
    ], 3, 10)
    assert "<Picture 1>" in text
    assert "<Picture 2>" not in text
    assert "inside" in text  # odd scene fallback to manifest[1]
    assert "final frame without a cut" in text


def test_build_reference_directive_fl2va_5entry_uses_manifest(lp):
    # 5-entry manifest (Picture 1 = opening; Pictures 2..5 =
    # per-scene end targets for scenes 1..4).  Scenes 2/3/4
    # must reference <Picture 1> with the per-scene end-target
    # about (manifest[2], manifest[3], manifest[4]).  Scene 5
    # is past the end of the manifest — odd scene → fallback
    # to manifest[1] ("END1").
    manifest = [
        {"slot": "Picture 1", "about": "OPEN", "role": "identity"},
        {"slot": "Picture 2", "about": "END1", "role": "destination"},
        {"slot": "Picture 3", "about": "END2", "role": "destination"},
        {"slot": "Picture 4", "about": "END3", "role": "destination"},
        {"slot": "Picture 5", "about": "END4", "role": "destination"},
    ]
    for s_idx, expect_about in [(2, "END2"), (3, "END3"), (4, "END4")]:
        text = lp.build_reference_directive("fl2va", manifest, s_idx, 10)
        assert "<Picture 1>" in text, f"scene {s_idx}: missing <Picture 1> label"
        assert "<Picture 2>" not in text, f"scene {s_idx}: stray <Picture 2> label"
        assert expect_about in text, f"scene {s_idx}: missing about {expect_about!r}"
        assert "OPEN" not in text, f"scene {s_idx}: leaked OPEN (opening) about"
    # Scene 5 falls back to manifest[1] (odd scene, past manifest end).
    text = lp.build_reference_directive("fl2va", manifest, 5, 10)
    assert "<Picture 1>" in text
    assert "END1" in text  # odd-scene fallback


def test_validate_manifest_fl2va_5_entries(lp):
    # 5-entry fl2va manifest validates (was rejected by the old
    # "exactly 2" rule).
    manifest = [
        {"slot": f"Picture {i}", "about": f"x{i}", "role": "destination"}
        for i in range(1, 6)
    ]
    errs = lp.validate_manifest(manifest, "fl2va")
    assert errs == [], errs


def test_validate_manifest_fl2va_too_many_rejected(lp):
    # fl2va caps at 9 pictures (MAX_MANIFEST_PICTURES).
    manifest = [
        {"slot": f"Picture {i}", "about": f"x{i}", "role": "destination"}
        for i in range(1, 11)
    ]
    errs = lp.validate_manifest(manifest, "fl2va")
    assert errs, errs
    assert any("supports up to 9 pictures" in e for e in errs), errs


# --------------------------------------------------------------------------- #
# Deterministic keyframe-idiom enforcement (live E2E finding: LLM drops
# angle brackets / skips the end-target sentence)
# --------------------------------------------------------------------------- #
_FL2VA_MANIFEST = [
    {"slot": "Picture 1", "about": "opening stance", "role": "identity"},
    {"slot": "Picture 2", "about": "clash impact", "role": "destination"},
]


def _three_reply(first_desc_line, extra="He swings; sparks fly."):
    return (
        "integrated_multimodal_description:\n"
        f"{first_desc_line}\n{extra}\n"
        "\n"
        "overall_soundscape:\nRain and impacts.\n"
        "\n"
        "non_diegetic_music:\nNo non-diegetic music.\n"
    )


def test_ensure_keyframe_idiom_repairs_bracketless_i2va(lp):
    lines = lp.split_three_sections(_three_reply(
        "At 0.00 seconds, Picture 1 is fully referenced as the opening frame. "
        "[Shot 1] Cinematic realism."
    ))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="i2va", manifest=_FL2VA_MANIFEST[:1], clip_index=1
    )
    body = "\n".join(fixed)
    assert "At 0.00 seconds, <Picture 1> is fully referenced as the opening frame." in body
    # Following LLM content on the same line survives.
    assert "[Shot 1] Cinematic realism." in body


def test_ensure_keyframe_idiom_prepends_missing_i2va(lp):
    lines = lp.split_three_sections(_three_reply("[Shot 1] The warrior waits."))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="i2va", manifest=_FL2VA_MANIFEST[:1], clip_index=1
    )
    desc_i = fixed.index("integrated_multimodal_description:")
    assert fixed[desc_i + 1].startswith("At 0.00 seconds, <Picture 1>")
    assert "[Shot 1] The warrior waits." in fixed


def test_ensure_keyframe_idiom_i2va_scene2_untouched(lp):
    lines = lp.split_three_sections(_three_reply("[Shot 1] Continuation, no labels."))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="i2va", manifest=_FL2VA_MANIFEST[:1], clip_index=2
    )
    assert fixed == lines  # scenes 2+ must stay label-free


def test_ensure_keyframe_idiom_fl2va_scene1_full_repair(lp):
    lines = lp.split_three_sections(_three_reply(
        "Picture 1 aligns with 0.00 seconds and Picture 2 aligns with the "
        "final target frame. Begin from the stance."
    ))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="fl2va", manifest=_FL2VA_MANIFEST, clip_index=1
    )
    body = "\n".join(fixed)
    assert ("<Picture 1> aligns with 0.00 seconds and <Picture 2> aligns "
            "with the final target frame.") in body
    # Reach sentence appended at the end of the description body.
    assert "Reach <Picture 2> only on the final frame; do not freeze early or cut." in body
    assert "overall_soundscape:" in body  # section structure intact


def test_ensure_keyframe_idiom_fl2va_scene2_appends_end_target(lp):
    lines = lp.split_three_sections(_three_reply("[Shot 1] The rival counters hard."))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="fl2va", manifest=_FL2VA_MANIFEST, clip_index=2
    )
    body = "\n".join(fixed)
    assert "Reach <Picture 1> only on the final frame; do not freeze early or cut." in body
    # Appended INSIDE the description section, before the soundscape header.
    assert fixed.index("overall_soundscape:") > next(
        i for i, ln in enumerate(fixed) if "Reach <Picture 1> only on the final frame" in ln
    )


def test_ensure_keyframe_idiom_fl2va_scene3_targets_picture1_fallback(lp):
    # Scene 3 in fl2va always uses the <Picture 1> label (per-scene
    # end target exposed by FrameIndexSwitch).  With a 2-entry
    # manifest scene 3 is an odd scene — about text falls back to
    # manifest[1] so legacy 2-image workflows keep their A->B->A
    # rhythm.
    lines = lp.split_three_sections(_three_reply("[Shot 1] The tide turns again."))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="fl2va", manifest=_FL2VA_MANIFEST, clip_index=3
    )
    body = "\n".join(fixed)
    assert "Reach <Picture 1> only on the final frame" in body
    assert "shown in <Picture 2>" not in body


def test_ensure_keyframe_idiom_fl2va_scene3_5entry_uses_manifest(lp):
    # 5-entry manifest: scene 3 uses manifest[3] as the per-scene
    # end target.  The appended end-target sentence must reference
    # <Picture 1> with manifest[3].about (not the legacy fallback).
    manifest5 = [
        {"slot": "Picture 1", "about": "opening state", "role": "identity"},
        {"slot": "Picture 2", "about": "end-target 1", "role": "destination"},
        {"slot": "Picture 3", "about": "end-target 2", "role": "destination"},
        {"slot": "Picture 4", "about": "end-target 3", "role": "destination"},
        {"slot": "Picture 5", "about": "end-target 4", "role": "destination"},
    ]
    lines = lp.split_three_sections(_three_reply("[Shot 1] The tide turns again."))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="fl2va", manifest=manifest5, clip_index=3
    )
    body = "\n".join(fixed)
    assert "Reach <Picture 1> only on the final frame; do not freeze early or cut." in body


def test_ensure_keyframe_idiom_t2va_ref2va_noop(lp):
    lines = lp.split_three_sections(_three_reply("[Shot 1] Plain t2va text."))
    assert lp.ensure_keyframe_idiom(
        lines, mode="t2va", manifest=[], clip_index=1
    ) == lines
    six = lp.split_six_sections(
        "subject_definitions:\n<Picture 1> defines <Subject 1>.\n\n"
        "summary:\n[reference generation] One move.\n\n"
        "retention_analysis:\n<Subject 1>: fully_preserved - keep.\n\n"
        "detailed_description:\n[Shot 1] The fight continues.\n\n"
        "overall_soundscape:\nImpacts.\n\n"
        "non_diegetic_music:\nNo non-diegetic music.\n"
    )
    assert lp.ensure_keyframe_idiom(
        six, mode="ref2va", manifest=_FL2VA_MANIFEST, clip_index=1
    ) == six


# --------------------------------------------------------------------------- #
# Video references (ref2va, P0 extension: motion/performance refs)
# --------------------------------------------------------------------------- #
def test_parse_references_video_slot(lp):
    m = lp.parse_references_text(json.dumps([
        {"slot": "Video 1", "about": "revenge-dance choreography, full body"},
    ]))
    assert m[0]["slot"] == "Video 1"
    assert m[0]["role"] == "destination"  # videos never default to identity


def test_parse_references_video_rejects_audio_and_order(lp):
    with pytest.raises(ValueError, match="Audio"):
        lp.parse_references_text(json.dumps([
            {"slot": "Audio 1", "about": "x"},
        ]))
    with pytest.raises(ValueError, match="after a Video slot"):
        lp.parse_references_text(
            "Video 1: dance\nPicture 1: face"
        )
    with pytest.raises(ValueError, match="contiguous Video"):
        lp.parse_references_text("Video 2: dance")


def test_validate_manifest_ref2va_video(lp):
    vid = [{"slot": "Video 1", "about": "dance", "role": "destination"}]
    assert lp.validate_manifest(vid, "ref2va") == []
    # Video count cap (3) enforced.
    three = [f"Video {i}" for i in (1, 2, 3)]
    ok = lp.validate_manifest(
        [{"slot": s, "about": "x", "role": "destination"} for s in three],
        "ref2va")
    assert ok == []
    four = lp.validate_manifest(
        [{"slot": f"Video {i}", "about": "x", "role": "destination"}
         for i in (1, 2, 3, 4)],
        "ref2va")
    assert any("up to 3 video" in e for e in four)
    # Pictures-then-videos combination is valid; video numbering must
    # restart at Video 1.
    combo = lp.validate_manifest([
        {"slot": "Picture 1", "about": "face", "role": "identity"},
        {"slot": "Video 1", "about": "dance", "role": "destination"},
    ], "ref2va")
    assert combo == []
    # i2va/fl2va still reject video slots.
    assert lp.validate_manifest(vid, "i2va")
    assert lp.validate_manifest(vid, "fl2va")
    assert lp.validate_manifest(vid, "t2va")


def test_validate_label_policy_ref2va_video_labels(lp):
    manifest = [{"slot": "Video 1", "about": "dance", "role": "destination"}]
    six = (
        "subject_definitions:\n"
        "The dancer's every movement reproduces the choreography of <Video 1>.\n"
        "\n"
        "summary:\n"
        "[reference generation] She performs the full routine.\n"
        "\n"
        "retention_analysis:\n"
        "<Video 1>: reference - choreography and timing.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] She dances the routine from the top.\n"
        "\n"
        "overall_soundscape:\nMusic and steps.\n"
        "\n"
        "non_diegetic_music:\nNo non-diegetic music.\n"
    )
    plan = lp.build_plan(
        [{"id": "s1", "prompt": lp.split_six_sections(six),
          "length": 175, "seed": "1"}],
        ["prefix"])
    assert lp.validate_label_policy(plan, "ref2va", manifest) == []
    # <Video 2> is not in the manifest -> error.
    bad = json.loads(json.dumps(plan))
    bad["shots"][0]["prompt"][1] += " Also mirror <Video 2>."
    errs = lp.validate_label_policy(bad, "ref2va", manifest)
    assert any("<Video 2> is not in the manifest" in e for e in errs)


def test_build_reference_directive_ref2va_video(lp):
    text = lp.build_reference_directive("ref2va", [
        {"slot": "Video 1", "about": "revenge-dance choreography"},
    ], 1, 10)
    assert "<Video 1>" in text
    assert "VIDEO MOTION REFERENCE" in text
    assert "choreography" in text
    assert "retention_analysis" in text


def test_ensure_keyframe_idiom_token_free_paraphrase_still_appends(lp):
    """Live E2E regression: the LLM writes token-free paraphrases like
    'aligning with the warrior's opening stance on the final frame' —
    those must NOT count as the end-target being present."""
    paraphrase = (
        "[Shot 1] The rival counters. The camera settles as the warrior "
        "returns to his stance, aligning with the watermelon warrior's "
        "opening stance on the final frame."
    )
    lines = lp.split_three_sections(_three_reply(paraphrase))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="fl2va", manifest=_FL2VA_MANIFEST, clip_index=2
    )
    body = "\n".join(fixed)
    assert "Reach <Picture 1> only on the final frame; do not freeze early or cut." in body


def test_ensure_keyframe_idiom_real_token_present_not_duplicated(lp):
    """When the LLM DID include the real bracketed token near a final-frame
    phrase, the canonical sentence is not duplicated."""
    good = (
        "[Shot 1] The rival counters hard, converging on the stance shown "
        "in <Picture 1> and reaching that picture only on the final frame."
    )
    lines = lp.split_three_sections(_three_reply(good))
    fixed = lp.ensure_keyframe_idiom(
        lines, mode="fl2va", manifest=_FL2VA_MANIFEST, clip_index=2
    )
    body = "\n".join(fixed)
    assert body.count("progressively align the visible scene") == 0


def test_build_reference_directive_ref2va_deterministic(lp):
    text = lp.build_reference_directive("ref2va", [
        {"slot": "Picture 1", "about": "courier face", "role": "identity"},
        {"slot": "Picture 2", "about": "greenhouse", "role": "destination"},
    ], 1, 10)
    assert "Deterministic subject binding" in text
    assert "<Subject 1>" in text
    assert "<Picture 1>" in text
    assert "courier face" in text
    assert "[reference generation]" in text
    assert "fully_preserved" in text
    assert "reference -" in text


# --------------------------------------------------------------------------- #
# OFFICIAL WORKFLOW ROUND-TRIP (ground truth lock)
# --------------------------------------------------------------------------- #
def _read_workflow_plan_json(workflow_path: Path) -> dict:
    """Extract the plan JSON widget value from the upstream plugin's
    example workflow and normalize each shot's ``prompt`` string into a
    line array (the upstream workflow stores it as a single ``\\n``-
    joined string; the Production Plan runtime expects a JSON array of
    strings). Returns ``{}`` if the file is missing or the plan widget
    cannot be located."""
    if not workflow_path.is_file():
        return {}
    raw = json.loads(workflow_path.read_text(encoding="utf-8"))
    for node in raw.get("nodes") or []:
        if str(node.get("type") or "").endswith("MiniMaxH3ChainPlanModern"):
            widgets = node.get("widgets_values") or []
            if widgets and isinstance(widgets[0], str) and widgets[0].lstrip().startswith("{"):
                plan = json.loads(widgets[0])
                plan.pop("defaults", None)
                for shot in plan.get("shots") or []:
                    shot.pop("steps", None)
                    if isinstance(shot.get("prompt"), str):
                        shot["prompt"] = shot["prompt"].split("\n")
                return plan
    return {}


@pytest.mark.skipif(not PLUGIN_DIR.is_dir(), reason="upstream plugin dir not present")
@pytest.mark.parametrize(
    "filename,schema,expected_id",
    [
        ("I2V Normal - MiniMax H3 0.6.json", "three_section", "greenhouse_arrival"),
        ("FL2V Normal - MiniMax H3 0.6.json", "three_section", "arrival_to_delivery"),
        ("Ref2V Basic - MiniMax H3 0.6.json", "six_section", "reference_delivery"),
    ])
def test_official_workflow_round_trip_validates(lp, filename, schema, expected_id):
    plan = _read_workflow_plan_json(PLUGIN_DIR / "example_workflows" / filename)
    assert plan, f"no plan JSON found in {filename}"
    assert any(s["id"] == expected_id for s in plan.get("shots") or [])
    errs = lp.validate_plan(plan, schema=schema)
    assert not errs, f"official workflow failed validation: {errs[:5]}"


@pytest.mark.skipif(not PLUGIN_DIR.is_dir(), reason="upstream plugin dir not present")
def test_official_i2v_uses_picture1_label(lp):
    plan = _read_workflow_plan_json(
        PLUGIN_DIR / "example_workflows" / "I2V Normal - MiniMax H3 0.6.json"
    )
    first_prompt = plan["shots"][0]["prompt"]
    # _read_workflow_plan_json normalizes the prompt to a line array.
    assert isinstance(first_prompt, list)
    joined = "\n".join(first_prompt)
    assert "<Picture 1>" in joined
    # Opening idiom (official wording, may be split across lines in JSON).
    assert "At 0.00 seconds, <Picture 1> is fully referenced as the opening frame." in joined


@pytest.mark.skipif(not PLUGIN_DIR.is_dir(), reason="upstream plugin dir not present")
def test_official_fl2v_alternation(lp):
    plan = _read_workflow_plan_json(
        PLUGIN_DIR / "example_workflows" / "FL2V Normal - MiniMax H3 0.6.json"
    )
    first = "\n".join(plan["shots"][0]["prompt"])
    second = "\n".join(plan["shots"][1]["prompt"])
    # Scene 1: both pictures + the "Reach <Picture 2> only on the final frame" line.
    assert "<Picture 1>" in first and "<Picture 2>" in first
    assert "Reach <Picture 2> only on the final frame" in first
    # Scene 2 (L2VA): end target reverts to Picture 1.
    assert "<Picture 1>" in second
    assert "progressively align" in second.lower()


@pytest.mark.skipif(not PLUGIN_DIR.is_dir(), reason="upstream plugin dir not present")
def test_official_ref2v_six_headers_in_order(lp):
    plan = _read_workflow_plan_json(
        PLUGIN_DIR / "example_workflows" / "Ref2V Basic - MiniMax H3 0.6.json"
    )
    first = plan["shots"][0]["prompt"]
    expected_headers = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    positions = [first.index(h) for h in expected_headers]
    assert positions == sorted(positions)
    first_joined = "\n".join(first)
    assert "[reference generation]" in first_joined
    assert "fully_preserved" in first_joined


# --------------------------------------------------------------------------- #
# Generator E2E: helpers
# --------------------------------------------------------------------------- #
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


def _fake_images(n=1):
    return np.zeros((n, 4, 4, 3), dtype=np.float32)


# Stub helpers for the v4 plan pipeline (extract_dialogue +
# _auto_storyboard + prefix + per-shot). The v4 plan path always
# runs these in order for natural-language inputs.
_EXTRACT_DIALOGUE_EMPTY = json.dumps({"turns": []})


def _auto_storyboard_reply(n=2, *, per_shot_seconds=10):
    """Storyboard LLM reply used by v4 plan tests. Mirrors the helper
    in test_h3_loop_prompt_generator.py."""
    shots = []
    for i in range(1, n + 1):
        shots.append(
            {
                "id": f"scene_{i:02d}",
                "description": f"Shot {i} description.",
                "shot_type": "medium_shot",
                "camera_movement": "slow_push_in",
                "transition_in": "fade_from_black" if i == 1 else "hard_cut",
                "duration_seconds": per_shot_seconds,
                "narrative_beat": "establish",
                "characters": ["young_woman"],
                "props": ["porcelain_bowl"],
                "notes": "Carry the bowl.",
            }
        )
    return json.dumps(shots, ensure_ascii=False)


def _storyboard_json(n=2, *, per_shot_seconds=10):
    shots = []
    for i in range(1, n + 1):
        shots.append(
            {
                "id": f"scene_{i:02d}",
                "description": f"Shot {i} description.",
                "shot_type": "medium_shot",
                "camera_movement": "slow_push_in",
                "transition_in": "fade_from_black" if i == 1 else "hard_cut",
                "duration_seconds": per_shot_seconds,
                "characters": [],
                "props": [],
            }
        )
    return json.dumps(shots, ensure_ascii=False)


def _i2va_scene1_reply():
    return (
        "integrated_multimodal_description:\n"
        "At 0.00 seconds, <Picture 1> is fully referenced as the opening frame. "
        "Animate the exact courier face, yellow jacket shown in <Picture 1>. "
        "He enters through the doorway and walks inside.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, footsteps, the door hinge.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _i2va_scene2_reply():
    return (
        "integrated_multimodal_description:\n"
        "[Shot 1] Continue directly from the incoming H3 Motion Context. "
        "Preserve the courier identity. He crosses the greenhouse.\n"
        "\n"
        "overall_soundscape:\n"
        "Continue the greenhouse room tone.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _fl2va_scene1_reply():
    return (
        "integrated_multimodal_description:\n"
        "<Picture 1> aligns with 0.00 seconds and <Picture 2> aligns with the "
        "final target frame. Begin from the exact courier at the door shown in "
        "<Picture 1>. Progressively match the parcel placement in <Picture 2>. "
        "Reach <Picture 2> only on the final frame; do not freeze early or cut.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, freewheel tick, footsteps.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _fl2va_scene2_reply():
    return (
        "integrated_multimodal_description:\n"
        "[Shot 1] Continue from the incoming delivery context. "
        "During the final seconds, progressively align the visible scene with "
        "the exterior doorway shown in <Picture 1>, reaching that picture "
        "only on the final frame without a cut or early hold.\n"
        "\n"
        "overall_soundscape:\n"
        "Continue greenhouse room tone and rain.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _fl2va_scene3_reply():
    # Scene 3 of the 2-entry legacy layout: odd scene → about
    # falls back to manifest[1] ("parcel on table"); the label
    # is <Picture 1> for every scene 2+ (verified against the
    # upstream gate wiring).
    return (
        "integrated_multimodal_description:\n"
        "[Shot 1] Continue from the previous clip. During the final seconds, "
        "progressively align the visible scene with the parcel on table "
        "shown in <Picture 1>, reaching that picture only on the "
        "final frame without a cut or early hold.\n"
        "\n"
        "overall_soundscape:\n"
        "Continue greenhouse room tone and rain.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _ref2va_scene1_reply():
    return (
        "subject_definitions:\n"
        "<Picture 1> defines <Subject 1>, the exact courier face, mustard jacket.\n"
        "<Picture 2> defines the greenhouse interior with wooden potting table.\n"
        "<Subject 1> is the single courier shown by both references.\n"
        "\n"
        "summary:\n"
        "[reference generation] The continuous delivery moves the courier "
        "from the door to the table.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain face, jacket, bicycle.\n"
        "<Picture 1>: fully_preserved - opening state.\n"
        "<Picture 2>: reference - destination only.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] The courier enters, walks to the table, places the parcel.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, freewheel tick, footsteps.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _ref2va_scene2_reply():
    return (
        "subject_definitions:\n"
        "<Picture 1> preserves the courier identity carried from the boundary.\n"
        "<Picture 2> defines the same greenhouse interior.\n"
        "<Subject 1> is the same courier shown by both references.\n"
        "\n"
        "summary:\n"
        "[reference generation] The delivery continues after the carried overlap.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain identity.\n"
        "<Picture 1>: reference - cool exterior.\n"
        "<Picture 2>: fully_preserved - greenhouse interior.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] Continue directly from the incoming H3 Motion Context. "
        "Roll the bicycle to the table, place the parcel.\n"
        "\n"
        "overall_soundscape:\n"
        "Continue rain-muted greenhouse ambience.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _ref2va_scene2_bad_timestamp_reply():
    return (
        "subject_definitions:\n"
        "<Picture 1> preserves the courier identity carried from the boundary.\n"
        "<Picture 2> defines the same greenhouse interior.\n"
        "<Subject 1> is the same courier shown by both references.\n"
        "\n"
        "summary:\n"
        "[reference generation] The delivery continues after the carried overlap.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain identity.\n"
        "<Picture 1>: reference - cool exterior.\n"
        "<Picture 2>: fully_preserved - greenhouse interior.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] Continue directly from the incoming H3 Motion Context; "
        "from 0.0s to 5.2s, the courier rolls the bicycle to the table.\n"
        "\n"
        "overall_soundscape:\n"
        "The exact bed from the previous clip carries across the boundary, "
        "then rain-muted ambience continues.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


_PREFIX_REPLY = "Always the same young woman in a Jiangnan courtyard at high summer."


# --------------------------------------------------------------------------- #
# Generator E2E: i2va
# --------------------------------------------------------------------------- #
def test_e2e_i2va_happy_path(lg):
    manifest_json = json.dumps([
        {"slot": "Picture 1", "about": "courier face, yellow jacket", "role": "identity"},
    ], ensure_ascii=False)
    # caption stage is stubbed (spy); v4 plan chain order:
    # extract_dialogue (narrator fallback) -> _auto_storyboard -> prefix
    # -> 2 per-shot calls (i2va first-scene idiom).
    conn = ScriptedConnector([
        _EXTRACT_DIALOGUE_EMPTY,
        _auto_storyboard_reply(2),
        _PREFIX_REPLY,
        _i2va_scene1_reply(),
        _i2va_scene2_reply(),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (json.loads(manifest_json), [])
    out = enh(user_input="concept",
        generation_mode="per_shot",
        reference_mode="i2va - 首帧关键帧(文生视频+首帧图)",
        images=_fake_images(1),
        seed=1)
    plan = json.loads(out["plan_json"])
    # Strict upstream contract: NO `defaults` key, NO `steps` on shots.
    assert set(plan.keys()) == {"shots", "prompt_prefix"}
    assert len(plan["shots"]) == 2
    # Scene 1 contains the official opening idiom + <Picture 1>.
    s1_prompt = plan["shots"][0]["prompt"]
    assert any("At 0.00 seconds, <Picture 1> is fully referenced as the opening frame." in ln for ln in s1_prompt)
    assert any("<Picture 1>" in ln for ln in s1_prompt)
    # Scene 2 has ZERO native labels.
    s2_text = " ".join(plan["shots"][1]["prompt"])
    assert "<Picture" not in s2_text and "<Video" not in s2_text and "<Audio" not in s2_text
    assert "<Subject" not in s2_text
    # Summary carries the mode (the old preflight D10 wiring hints were
    # dropped with the two-output reduction).
    assert "reference=i2va" in out["summary"]
    # Plan validates three_section.
    plan_for_validate = {
        "shots": [
            {k: v for k, v in shot.items() if k in {"id", "prompt", "length", "seed"}}
            for shot in plan["shots"]
        ],
        "prompt_prefix": plan.get("prompt_prefix", []),
    }
    errs = lg.validate_plan(plan_for_validate, schema=lg.SCHEMA_THREE)
    assert errs == []
    # Stage 2 call injected the reference_directive (D6) and the
    # mode_note for the per-clip system policy.
    shot_user = conn.calls[3][1]["content"]
    assert "At 0.00 seconds, <Picture 1> is fully referenced as the opening frame." in shot_user
    assert "courier face, yellow jacket" in shot_user
    assert "Reference manifest (binding):" in shot_user


def test_e2e_i2va_single_call_rejected(lg):
    enh = lg.H3LoopPromptEnhancer(ScriptedConnector([]))
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (
        [{"slot": "Picture 1", "about": "x", "role": "identity"}],
        [])
    with pytest.raises(RuntimeError, match="single_call generation mode is not supported"):
        enh(user_input="concept",
            generation_mode="single_call - 单次调用(快/省)",
            reference_mode="i2va",
            images=_fake_images(1),
            seed=1)


# --------------------------------------------------------------------------- #
# Generator E2E: fl2va
# --------------------------------------------------------------------------- #
def test_e2e_fl2va_3shot_alternation(lg):
    manifest_json = json.dumps([
        {"slot": "Picture 1", "about": "courier at door", "role": "identity"},
        {"slot": "Picture 2", "about": "parcel on table", "role": "destination"},
    ], ensure_ascii=False)
    conn = ScriptedConnector([
        '{"turns": []}',
        _storyboard_json(3),
        _PREFIX_REPLY,
        _fl2va_scene1_reply(),
        _fl2va_scene2_reply(),
        _fl2va_scene3_reply(),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (json.loads(manifest_json), [])
    out = enh(user_input="concept",
        generation_mode="per_shot",
        reference_mode="fl2va - 首尾帧关键帧(首帧+逐场尾帧)",
        images=_fake_images(2),
        seed=1)
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 3
    # Scene 1: both pictures + Reach <Picture 2> only on the final frame.
    s1 = " ".join(plan["shots"][0]["prompt"])
    assert "<Picture 1>" in s1 and "<Picture 2>" in s1
    assert "Reach <Picture 2> only on the final frame" in s1
    # Scene 2: end target <Picture 1>.
    s2 = " ".join(plan["shots"][1]["prompt"])
    assert "<Picture 1>" in s2 and "<Picture 2>" not in s2
    assert "progressively align" in s2.lower()
    # Scene 3 (3-shot board, 2-entry legacy manifest): end
    # target stays <Picture 1>; about falls back to manifest[1]
    # ("parcel on table") because scene 3 is an odd scene.
    s3 = " ".join(plan["shots"][2]["prompt"])
    assert "<Picture 1>" in s3
    assert "<Picture 2>" not in s3
    assert "parcel on table" in s3
    # Validates three_section + label policy clean.
    plan_for_validate = {
        "shots": [
            {k: v for k, v in shot.items() if k in {"id", "prompt", "length", "seed"}}
            for shot in plan["shots"]
        ],
        "prompt_prefix": plan.get("prompt_prefix", []),
    }
    assert lg.validate_plan(plan_for_validate, schema=lg.SCHEMA_THREE) == []
    errs = lg.validate_label_policy(plan, "fl2va", json.loads(manifest_json))
    assert errs == []
    assert "reference=fl2va" in out["summary"]


def test_e2e_fl2va_single_call_rejected(lg):
    enh = lg.H3LoopPromptEnhancer(ScriptedConnector([]))
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (
        [
            {"slot": "Picture 1", "about": "x", "role": "identity"},
            {"slot": "Picture 2", "about": "y", "role": "destination"},
        ],
        [])
    with pytest.raises(RuntimeError, match="single_call"):
        enh(user_input="c",
            generation_mode="single_call",
            reference_mode="fl2va",
            images=_fake_images(2),
            seed=1)


# --------------------------------------------------------------------------- #
# Generator E2E: ref2va
# --------------------------------------------------------------------------- #
def test_e2e_ref2va_six_sections(lg):
    manifest_json = json.dumps([
        {"slot": "Picture 1", "about": "courier face, yellow jacket", "role": "identity"},
        {"slot": "Picture 2", "about": "greenhouse interior", "role": "destination"},
    ], ensure_ascii=False)
    conn = ScriptedConnector([
        '{"turns": []}',
        _storyboard_json(2),
        _PREFIX_REPLY,
        _ref2va_scene1_reply(),
        _ref2va_scene2_reply(),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (json.loads(manifest_json), [])
    out = enh(user_input="concept",
        generation_mode="per_shot",
        reference_mode="ref2va - 参考图(N张/全场景)",
        images=_fake_images(2),
        seed=1)
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 2
    # All shots have six sections in order.
    for shot in plan["shots"]:
        s = shot["prompt"]
        positions = [s.index(f"{h}:") for h in lg.SIX_SECTION_FIELDS]
        assert positions == sorted(positions)
    # Summary lines start with [reference generation].
    for shot in plan["shots"]:
        text = " ".join(shot["prompt"])
        assert "[reference generation]" in text
    # subject_definitions references manifest slots + Subject binding.
    s1 = " ".join(plan["shots"][0]["prompt"])
    assert "<Picture 1>" in s1 and "<Picture 2>" in s1 and "<Subject 1>" in s1
    # retention uses fully_preserved + reference.
    assert "fully_preserved" in s1 and "reference -" in s1
    # prefix contains NO native labels (ref2va policy + D3).
    assert plan.get("prompt_prefix")
    # Validates six_section.
    plan_for_validate = {
        "shots": [
            {k: v for k, v in shot.items() if k in {"id", "prompt", "length", "seed"}}
            for shot in plan["shots"]
        ],
        "prompt_prefix": plan.get("prompt_prefix", []),
    }
    errs = lg.validate_plan(plan_for_validate, schema=lg.SCHEMA_SIX)
    assert errs == []
    errs = lg.validate_label_policy(plan, "ref2va", json.loads(manifest_json))
    assert errs == []
    assert "reference=ref2va" in out["summary"]
    # Stage 1 call used the ref2va mode_note.
    prefix_user = conn.calls[2][1]["content"]
    assert "whole-video invariants only" in prefix_user.lower()
    # Stage 2 call used shot_system_prompt_ref2v (six-section rules).
    shot_system = conn.calls[3][0]["content"]
    assert "Ref2VA addendum" in shot_system
    # Continuation uses ref2v template.
    cont_user = conn.calls[4][1]["content"]
    assert "Ref2VA clip CONTINUES" in cont_user


def test_ref2va_retry_on_timestamp_contract(lg):
    """Ref2VA timestamp ban (clock/seconds notation) is the only remaining
    stage-2 retry contract — sound-recap rule was removed alongside the
    H0/spoken-scene contracts."""
    manifest_json = json.dumps([
        {"slot": "Picture 1", "about": "courier face, yellow jacket", "role": "identity"},
        {"slot": "Picture 2", "about": "greenhouse interior", "role": "destination"},
    ], ensure_ascii=False)
    conn = ScriptedConnector([
        '{"turns": []}',
        _storyboard_json(2),
        _PREFIX_REPLY,
        _ref2va_scene1_reply(),
        _ref2va_scene2_bad_timestamp_reply(),
        _ref2va_scene2_reply(),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (json.loads(manifest_json), [])
    out = enh(user_input="concept",
        generation_mode="per_shot",
        reference_mode="ref2va - 参考图(N张/全场景)",
        images=_fake_images(2),
        output_language="en",
        seed=1)
    plan = json.loads(out["plan_json"])
    # extractor + storyboard + prefix + scene1 + scene2 attempt1 + scene2 retry
    assert len(conn.calls) == 6
    retry_call = conn.calls[5]
    assert len(retry_call) == 3
    assert "forbidden timestamp" in retry_call[2]["content"]
    s2_text = " ".join(plan["shots"][1]["prompt"])
    assert "0.0s" not in s2_text


def test_ref2va_continuation_carries_subject_body_only(lg):
    """Regression (self-review finding 1): the {previous_subject_definitions}
    slice must end at the summary header — summary and retention_analysis
    bodies must NOT be smuggled under the bindings heading."""
    manifest_json = json.dumps([
        {"slot": "Picture 1", "about": "courier face, yellow jacket", "role": "identity"},
        {"slot": "Picture 2", "about": "greenhouse interior", "role": "destination"},
    ], ensure_ascii=False)
    conn = ScriptedConnector([
        '{"turns": []}',
        _storyboard_json(2),
        _PREFIX_REPLY,
        _ref2va_scene1_reply(),
        _ref2va_scene2_reply(),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (json.loads(manifest_json), [])
    out = enh(user_input="concept",
        reference_mode="ref2va",
        images=_fake_images(2),
        seed=1)
    assert json.loads(out["plan_json"])
    cont_user = conn.calls[4][1]["content"]
    # The previous subject_definitions BODY is present...
    assert "defines <Subject 1>, the exact courier face" in cont_user
    # ...but the summary / retention_analysis bodies are NOT (the old slice
    # ended at detailed_description and leaked both under the heading).
    assert "The continuous delivery moves" not in cont_user
    assert "<Subject 1>: fully_preserved - retain face" not in cont_user


def test_e2e_ref2va_single_call_rejected(lg):
    enh = lg.H3LoopPromptEnhancer(ScriptedConnector([]))
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (
        [{"slot": "Picture 1", "about": "x", "role": "identity"}],
        [])
    with pytest.raises(RuntimeError, match="single_call"):
        enh(user_input="c",
            generation_mode="single_call",
            reference_mode="ref2va",
            images=_fake_images(1),
            seed=1)


def test_e2e_ref2va_missing_images_raises(lg):
    with pytest.raises(RuntimeError, match="ref2va requires images"):
        lg.H3LoopPromptEnhancer(ScriptedConnector([]))(
            user_input="a quiet scene",
            scene_count=1,
            reference_mode="ref2va",
            seed=1)


def test_e2e_ref2va_label_violation_raises(lg, lp):
    """A plan that violates D5 (subject_definitions mentions <Subject 1>
    but the manifest has no identity role) must be caught by
    validate_label_policy."""
    plan = lp.build_plan(  # type: ignore[union-attr]
        [
            {
                "id": "s1",
                "prompt": lp.split_six_sections(_ref2va_scene1_reply()),
                "length": 243,
                "seed": "1001",
            },
        ],
        ["p"])
    errs = lg.validate_label_policy(plan, "ref2va", [
        {"slot": "Picture 1", "about": "x", "role": "destination"},
        {"slot": "Picture 2", "about": "y", "role": "destination"},
    ])
    assert errs
    assert any("no matching manifest" in e for e in errs)


def test_e2e_t2va_regression(lg):
    """Default call (no reference_mode / no references_text) unchanged."""
    t2va_reply = (
        "integrated_multimodal_description:\n"
        "[Shot 1] The courier enters the greenhouse.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, footsteps.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )
    conn = ScriptedConnector(
        [
            '{"intent": "narration"}',  # classifier (no dialogue lines)
            _storyboard_json(1),
            _PREFIX_REPLY,
            t2va_reply,
        ]
    )
    # Without setting reference_mode, t2va is the default.
    out = lg.H3LoopPromptEnhancer(conn)(user_input="concept",
        seed=1)
    plan = json.loads(out["plan_json"])
    assert plan["prompt_prefix"]
    assert "reference=t2va" in out["summary"]
    plan_for_validate = {
        "shots": [
            {k: v for k, v in shot.items() if k in {"id", "prompt", "length", "seed"}}
            for shot in plan["shots"]
        ],
        "prompt_prefix": plan.get("prompt_prefix", []),
    }
    errs = lg.validate_plan(plan_for_validate)
    assert errs == []


# --------------------------------------------------------------------------- #
# is_changed stability + sensitivity
# --------------------------------------------------------------------------- #
def _base_kwargs():
    return dict(
        user_input="c",
        seed=0,
        scene_count=0,
        total_duration_seconds=10,
        generation_mode="per_shot - 逐场生成(推荐)",
        category="none - 不指定",
        output_language="en",
                temperature=0.4,
        max_tokens=8192,
        timeout=120,
        reference_mode="t2va - 文生视频链(默认)")


def test_is_changed_reference_mode_affects_hash(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    a = node.is_changed(FakeConnector(), **base)
    base_b = dict(base, reference_mode="i2va - 首帧关键帧(文生视频+首帧图)")
    b = node.is_changed(FakeConnector(), **base_b)
    assert a != b


def test_is_changed_whitespace_only_manifest_no_op(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    a = node.is_changed(FakeConnector(), **dict(base, images=None))
    b = node.is_changed(FakeConnector(), **dict(base, images=None))
    assert a == b


def test_is_changed_image_shape_affects_hash(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    a = node.is_changed(FakeConnector(), **dict(base, reference_mode="i2va", images=_fake_images(1)))
    b = node.is_changed(FakeConnector(), **dict(base, reference_mode="i2va", images=_fake_images(2)))
    assert a != b


def test_is_changed_image_presence_affects_hash(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    a = node.is_changed(FakeConnector(), **dict(base, reference_mode="i2va", images=None))
    b = node.is_changed(FakeConnector(), **dict(base, reference_mode="i2va", images=_fake_images(1)))
    assert a != b


def test_is_changed_references_text_affects_hash(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    a = node.is_changed(FakeConnector(), **dict(base, references_text=""))
    b = node.is_changed(
        FakeConnector(),
        **dict(base, references_text="Picture 1: orange tabby"),
    )
    assert a != b


def test_is_changed_same_shape_different_pixels_affects_hash(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    zeros = _fake_images(1)
    ones = zeros + 1
    a = node.is_changed(
        FakeConnector(), **dict(base, reference_mode="i2va", images=zeros)
    )
    b = node.is_changed(
        FakeConnector(), **dict(base, reference_mode="i2va", images=ones)
    )
    assert a != b


def test_is_changed_category_affects_hash(lg):
    """The category widget hashes into is_changed since it drives the
    spoken-scene / genre contract in ref2va."""
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = _base_kwargs()
    a = node.is_changed(FakeConnector(), **dict(base, category="none - 不指定"))
    b = node.is_changed(FakeConnector(), **dict(base, category="dialogue - 对白/对话/相声"))
    assert a != b


# --------------------------------------------------------------------------- #
# Real-image grounding (user requirement)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not COMFYUI_INPUT_DIR.is_dir(), reason="ComfyUI input dir not present")
def test_real_images_present():
    assert (COMFYUI_INPUT_DIR / "First-Frame.png").is_file()
    assert (COMFYUI_INPUT_DIR / "Last-Frame.jpeg").is_file()


@pytest.mark.skipif(not COMFYUI_INPUT_DIR.is_dir(), reason="ComfyUI input dir not present")
def test_e2e_fl2va_real_image_grounding(lg):
    """The fl2va manifest's about text MUST mention the real filenames
    so the loader can bind the right pictures when the user wires up
    LoadImage nodes in ComfyUI."""
    manifest_json = json.dumps([
        {
            "slot": "Picture 1",
            "about": "Courier at door, from First-Frame.png (refer to ComfyUI input)",
            "role": "identity",
        },
        {
            "slot": "Picture 2",
            "about": "Parcel on table, from Last-Frame.jpeg (refer to ComfyUI input)",
            "role": "destination",
        },
    ], ensure_ascii=False)
    conn = ScriptedConnector([
        '{"turns": []}',
        _storyboard_json(2),
        _PREFIX_REPLY,
        _fl2va_scene1_reply(),
        _fl2va_scene2_reply(),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    enh._caption_images = lambda *, images, ref_code, seed, **kwargs: (json.loads(manifest_json), [])
    out = enh(user_input="concept",
        generation_mode="per_shot",
        reference_mode="fl2va",
        images=_fake_images(2),
        seed=1)
    # The real filenames are passed to the LLM via the per-clip user
    # template "Reference manifest (binding)" block + the reference
    # directive that weaves the manifest about text into the opening
    # idiom. A downstream LLM MUST echo them into the clip text.
    shot1_user = conn.calls[2][1]["content"]
    assert "First-Frame.png" in shot1_user
    assert "Last-Frame.jpeg" in shot1_user
    assert "reference=fl2va" in out["summary"]


# --------------------------------------------------------------------------- #
# Misc node surface
# --------------------------------------------------------------------------- #
def test_node_input_types_includes_reference_mode(lg):
    inputs = lg.MiniMaxH3LoopPromptGenerator.INPUT_TYPES()
    opt = inputs["optional"]
    assert "reference_mode" in opt
    assert opt["reference_mode"][0] == list(lg.REFERENCE_MODES)
    assert "references_text" not in opt
    assert "references_text" not in opt


# --------------------------------------------------------------------------- #
# Fix 0b — parse_references_text ref2va default role
# --------------------------------------------------------------------------- #
def test_parse_references_text_ref2va_defaults_every_slot_to_identity(lp):
    """Regression: previously Picture 2+ defaulted to role='destination' in
    all modes. For ref2va every reference picture is an identity anchor."""
    manifest = lp.parse_references_text(
        "Picture 1: tabby in suit\n"
        "Picture 2: cream cat with dress\n",
        reference_mode="ref2va")
    assert len(manifest) == 2
    assert manifest[0]["role"] == "identity"
    assert manifest[1]["role"] == "identity"


def test_parse_references_text_i2va_keeps_legacy_other_destination(lp):
    """Non-ref2va modes keep the i==1→identity, others→destination legacy
    default so FL2VA/I2VA contracts stay intact."""
    manifest = lp.parse_references_text(
        "Picture 1: tabby\nPicture 2: cream cat\n",
        reference_mode="fl2va")
    assert manifest[0]["role"] == "identity"
    assert manifest[1]["role"] == "destination"


def test_parse_references_text_ref2va_explicit_role_still_honored(lp):
    """When the user explicitly writes role='environment' in a ref2va JSON
    manifest we keep it — the override only fills *default* roles."""
    manifest = lp.parse_references_text(json.dumps([
        {"slot": "Picture 1", "about": "tabby"},
        {"slot": "Picture 2", "about": "cafe room", "role": "environment"},
    ]), reference_mode="ref2va")
    assert manifest[0]["role"] == "identity"
    assert manifest[1]["role"] == "environment"


def test_parse_references_text_ref2va_no_h0_defaults_every_slot_to_identity(lp):
    """Ref2VA still defaults every Picture slot to role='identity' even
    without any H0-style named_identity_slots input. This is the only
    role-defaulting rule that survives the H0 removal."""
    manifest = lp.parse_references_text(
        "Picture 1: tabby\nPicture 2: cream cat\n",
        reference_mode="ref2va")
    assert manifest[0]["role"] == "identity"
    assert manifest[1]["role"] == "identity"


# --------------------------------------------------------------------------- #
# Category-driven contract injection in build_reference_directive.
# The category widget (e.g. "dialogue") is the only trigger for the SPOKEN
# SCENE CONTRACT / GENRE CONTRACT blocks; only `dialogue` does so today.
# --------------------------------------------------------------------------- #
def test_build_reference_directive_ref2va_no_category_has_no_contract(lp):
    text = lp.build_reference_directive(
        "ref2va",
        [{"slot": "Picture 1", "about": "tabby", "role": "identity"}],
        1,
        10,
        category="none - 不指定")
    assert "GENRE CONTRACT" not in text
    assert "SPOKEN SCENE CONTRACT" not in text
    assert "[reference generation]" in text


def test_build_reference_directive_ref2va_action_has_no_contract(lp):
    """`action` is a visual-styling category; it does NOT unlock any
    spoken-scene / genre contract."""
    text = lp.build_reference_directive(
        "ref2va",
        [{"slot": "Picture 1", "about": "tabby", "role": "identity"}],
        1,
        10,
        category="action - 动作戏/打斗/飙车")
    assert "GENRE CONTRACT" not in text
    assert "SPOKEN SCENE CONTRACT" not in text


def test_build_reference_directive_ref2va_dialogue_injects_contracts(lp):
    text = lp.build_reference_directive(
        "ref2va",
        [
            {"slot": "Picture 1", "about": "tabby", "role": "identity"},
            {"slot": "Picture 2", "about": "cream cat", "role": "identity"},
        ],
        1,
        10,
        category="dialogue - 对白/对话/相声")
    assert "SPOKEN SCENE CONTRACT" in text
    assert "never in the middle of a spoken sentence" in text.lower()
    assert "GENRE CONTRACT" in text
    # Dialogue-as-data: the node appends the verbatim blocks; the
    # model writes none of them (see assemble_dialogue_line_blocks).
    assert "Dialogue on this clip is LOCKED as data" in text
    assert "ONE input dialogue line = ONE appended block" in text
    assert "[reference generation]" in text


def test_build_reference_directive_i2va_fl2va_dialogue_no_contract(lp):
    """The SPOKEN/GENRE contract only fires in ref2va; i2va / fl2va
    pass the category through unchanged."""
    for mode in ("i2va", "fl2va"):
        manifest = (
            [{"slot": "Picture 1", "about": "x", "role": "identity"}]
            if mode == "i2va"
            else [
                {"slot": "Picture 1", "about": "x", "role": "identity"},
                {"slot": "Picture 2", "about": "y", "role": "destination"},
            ]
        )
        clip_index = 1
        text = lp.build_reference_directive(
            mode,
            manifest,
            clip_index,
            10,
            category="dialogue - 对白/对话/相声")
        assert "GENRE CONTRACT" not in text, (mode, text)
        assert "SPOKEN SCENE CONTRACT" not in text, (mode, text)
