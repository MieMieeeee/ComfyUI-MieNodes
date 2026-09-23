# -*- coding: utf-8 -*-
"""Tests for ``minimax_h3_loop_prompts`` (17k+5 grid, three-section
split, plan assembly + validation, reports, template builders).

NOTE (test drift fix, 2026-09-14):
  * Removed test cases for APIs deleted in the one-node-rewrite:
      - parse_shots_text family            (now auto-storyboard inside the enhancer)
      - parse_per_shot_overrides family    (overrides dropped from the node)
      - apply_overrides                    (overrides dropped from the node)
  * build_plan no longer emits a top-level ``defaults`` key and no longer
    accepts a per-shot ``steps`` field — sampling parameters live on the
    Production Plan widgets.
  * build_shot_user_text / build_single_call_user_text dropped the
    ``width`` / ``height`` arguments (canvas stays on the Plan node).
  * build_shots_digest now appends ``# all_shots=.../first_seen=...``
    roster hints so the prefix-synthesis LLM can tell recurring vs.
    one-off subjects apart.
"""
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
        LLM_DIR / "minimax_h3_storyboard_prompts.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )


# --------------------------------------------------------------------------- #
# 17k+5 grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "seconds,frames",
    [
        (5, 124),
        (10, 243),
        (15, 362),
        (20, 481),
        (30, 736),
        (60, 1450),
        (1, 39),    # 24 frames -> next grid point 39
        (0.25, 22), # 6 frames -> next grid point 22
        (149.667, 3592),
        (149.7, 3592),  # clamped to the max grid value
        (10.0, 243),
        ("10", 243),  # numeric strings tolerated
    ],
)
def test_seconds_to_length(lp, seconds, frames):
    assert lp.seconds_to_length(seconds) == frames


def test_seconds_to_length_rejects_bad(lp):
    for bad in (0, -1, "abc", None):
        with pytest.raises(ValueError):
            lp.seconds_to_length(bad)


def test_is_valid_length(lp):
    for good in (5, 22, 39, 243, 3592):
        assert lp.is_valid_length(good)
    for bad in (4, 6, 21, 23, 242, 244, 3593, -5, "243x", None):
        assert not lp.is_valid_length(bad)


# --------------------------------------------------------------------------- #
# Three-section split
# --------------------------------------------------------------------------- #
def _clip_reply(desc="The bowl settles on the wood.", sound="Cicadas hold the air.", music="No non-diegetic music."):
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] {desc}\n"
        "\n"
        "overall_soundscape:\n"
        f"{sound}\n"
        "\n"
        "non_diegetic_music:\n"
        f"{music}\n"
    )


def test_split_three_sections_canonical(lp):
    lines = lp.split_three_sections(_clip_reply())
    assert lines[0] == "integrated_multimodal_description:"
    assert lines[1] == "[Shot 1] The bowl settles on the wood."
    assert lines[2] == ""
    assert lines[3] == "overall_soundscape:"
    assert lines[4] == "Cicadas hold the air."
    assert lines[5] == ""
    assert lines[6] == "non_diegetic_music:"
    assert lines[7] == "No non-diegetic music."
    # Matches the Production Plan prompt-array shape exactly.
    assert len(lines) == 8


def test_split_three_sections_tolerant(lp):
    text = (
        "Sure! Here is the clip:\n\n"
        "integrated_multimodal_description:\n"
        "\n[Shot 1] A first line.\nA second line.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass.\n"
        "\n"
        "non_diegetic_music:\n"
        "\nA lone piano note.\n"
    )
    lines = lp.split_three_sections(text)
    assert lines[0] == "integrated_multimodal_description:"
    assert lines[1:3] == ["[Shot 1] A first line.", "A second line."]
    assert lines[-1] == "A lone piano note."


def test_split_three_sections_missing_music_uses_default(lp):
    text = (
        "integrated_multimodal_description:\n[Shot 1] She stirs.\n\n"
        "overall_soundscape:\nOne bright clink.\n"
    )
    lines = lp.split_three_sections(text)
    assert lines[-2] == "non_diegetic_music:"
    assert lines[-1] == lp.DEFAULT_MUSIC_LINE


def test_split_three_sections_missing_sections_raise(lp):
    with pytest.raises(ValueError, match="integrated_multimodal_description"):
        lp.split_three_sections("overall_soundscape:\nonly sound\n")
    with pytest.raises(ValueError, match="overall_soundscape"):
        lp.split_three_sections("integrated_multimodal_description:\nonly desc\n")
    with pytest.raises(ValueError, match="empty"):
        lp.split_three_sections("integrated_multimodal_description:\n\n\noverall_soundscape:\nx\n")
    with pytest.raises(ValueError):
        lp.split_three_sections("")
    with pytest.raises(ValueError):
        lp.split_three_sections("   ")


def test_split_three_sections_chinese(lp):
    text = (
        "integrated_multimodal_description:\n"
        "[Shot 1] 江南院落，青瓷碗里的冰块开始滑动。\n"
        "\n"
        "overall_soundscape:\n"
        "蝉声持续，冰块轻碰碗壁。\n"
        "\n"
        "non_diegetic_music:\n"
        "无。\n"
    )
    lines = lp.split_three_sections(text)
    assert any("江南院落，青瓷碗里的冰块开始滑动。" in ln for ln in lines)


def test_previous_tail(lp):
    lines = lp.split_three_sections(
        _clip_reply(desc="[Shot 1] First beat.\nShe lifts the bowl with both hands and drinks.")
    )
    tail = lp.previous_tail(lines)
    assert "She lifts the bowl" in tail


def test_description_and_sound_bodies(lp):
    lines = lp.split_three_sections(_clip_reply())
    assert (
        lp.description_body(lines)
        == "[Shot 1] The bowl settles on the wood."
    )
    assert lp.sound_body(lines) == "Cicadas hold the air."
    # Mid-paragraph anchors survive the full-description handoff.
    long_desc = (
        "A cat with a red collar watches from the left ridge. "
        "The dog below wears nothing. " + "padding " * 120
    )
    lines2 = lp.split_three_sections(_clip_reply(desc=long_desc, sound="Night wind."))
    assert "red collar" in lp.description_body(lines2)
    assert lp.sound_body(lines2) == "Night wind."


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def test_derive_seed(lp):
    assert lp.derive_seed(1000, 1) == "1001"
    assert lp.derive_seed(1000, 3) == "1003"
    # Unified: every clip shares the base seed.
    assert lp.derive_seed(1000, 1, unified=True) == "1000"
    assert lp.derive_seed(1000, 3, unified=True) == "1000"
    base = lp.derive_seed_base(0)
    assert base > 0  # wall-clock derived
    assert lp.derive_seed_base(777) == 777


# --------------------------------------------------------------------------- #
# Plan assembly + validation
# --------------------------------------------------------------------------- #
# A canonical split prompt (the exact Production Plan line-array shape).
_CLIP_PROMPT_LINES = [
    "integrated_multimodal_description:",
    "[Shot 1] The bowl settles on the wood.",
    "",
    "overall_soundscape:",
    "Cicadas hold the air.",
    "",
    "non_diegetic_music:",
    "No non-diegetic music.",
]


def _plan_entry(i, prompt=None, length=243, seed=None):
    return {
        "id": f"scene_{i:02d}",
        "prompt": prompt or list(_CLIP_PROMPT_LINES),
        "length": length,
        "seed": seed or str(1926 + i),
    }


def test_build_plan_and_validate_clean(lp):
    plan = lp.build_plan(
        [_plan_entry(1), _plan_entry(2)], ["Shared prefix line."]
    )
    assert lp.validate_plan(plan) == []
    # ``defaults`` lives on the Production Plan node now — the prompt
    # generator only emits shots + prompt_prefix.
    assert set(plan.keys()) == {"shots", "prompt_prefix"}
    assert list(plan["shots"][0].keys()) == ["id", "prompt", "length", "seed"]
    assert isinstance(plan["shots"][0]["seed"], str)


def test_validate_plan_catches_drift(lp):
    good = lp.build_plan([_plan_entry(1)], ["prefix"])
    # Bad top-level key.
    bad = dict(good, extra="x")
    assert any("top-level" in e for e in lp.validate_plan(bad))
    # Seed as int, not string.
    bad_seed = json.loads(json.dumps(good))
    bad_seed["shots"][0]["seed"] = 123
    assert any("seed" in e for e in lp.validate_plan(bad_seed))
    # Prompt as plain string, not array.
    bad_prompt = json.loads(json.dumps(good))
    bad_prompt["shots"][0]["prompt"] = "flat string"
    assert any("prompt" in e for e in lp.validate_plan(bad_prompt))
    # Off-grid length.
    bad_len = json.loads(json.dumps(good))
    bad_len["shots"][0]["length"] = 240
    assert any("grid" in e for e in lp.validate_plan(bad_len))
    # Empty prompt_prefix is tolerated (absent OR empty are both fine);
    # only absent-from-required or malformed entries error.
    bad_prefix = json.loads(json.dumps(good))
    bad_prefix["prompt_prefix"] = []
    assert lp.validate_plan(bad_prefix) == []
    del bad_prefix["prompt_prefix"]
    assert lp.validate_plan(bad_prefix) == []
    bad_prefix["prompt_prefix"] = [""]  # empty string entry still errors
    assert any("prompt_prefix" in e for e in lp.validate_plan(bad_prefix))
    bad_prefix["prompt_prefix"] = "not a list"
    assert any("prompt_prefix" in e for e in lp.validate_plan(bad_prefix))
    # Continuation_mode / width / steps must never appear in a shot.
    bad_extra = json.loads(json.dumps(good))
    bad_extra["shots"][0]["continuation_mode"] = "guide"
    assert any("unexpected keys" in e for e in lp.validate_plan(bad_extra))
    bad_extra2 = json.loads(json.dumps(good))
    bad_extra2["shots"][0]["steps"] = 30
    assert any("unexpected keys" in e for e in lp.validate_plan(bad_extra2))


def test_plan_json_serialization_matches_reference_shape(lp):
    """The serialized plan matches the real Production Plan JSON shape
    from the user's workflow (top keys, prompt line-array, seed string)."""
    plan = lp.build_plan(
        [_plan_entry(1, seed="1926")], ["Always the same young woman."]
    )
    text = lp.plan_to_json_string(plan)
    parsed = json.loads(text)
    # Sampling parameters live on the Plan node now — no ``defaults``.
    assert set(parsed.keys()) == {"shots", "prompt_prefix"}
    shot = parsed["shots"][0]
    assert shot["prompt"][0] == "integrated_multimodal_description:"
    assert shot["prompt"][3] == "overall_soundscape:"
    assert shot["prompt"][6] == "non_diegetic_music:"
    assert isinstance(shot["seed"], str) and shot["seed"] == "1926"
    assert parsed["prompt_prefix"] == ["Always the same young woman."]


# --------------------------------------------------------------------------- #
# Reports + template builders
# --------------------------------------------------------------------------- #
def test_reports(lp):
    plan = lp.build_plan([_plan_entry(1), _plan_entry(2)], ["prefix"])
    report = lp.build_preflight_report(plan, warnings=["watch out"])
    assert "Scenes: 2" in report
    assert "486" in report  # total frames
    assert "watch out" in report
    preview = lp.build_plan_preview(plan)
    assert "| 1 | `scene_01` | 243 |" in preview
    assert "| 2 | `scene_02` | 243 |" in preview


def test_shot_system_prompt_reuses_h3_guide(lp):
    system = lp.shot_system_prompt()
    assert "MiniMax H3" in system  # official t2v guide prepended
    assert "Loop-plan addendum" in system
    assert "integrated_multimodal_description:" in system


# --------------------------------------------------------------------------- #
# Pacing directive (beat density scales with duration)
# --------------------------------------------------------------------------- #
def test_pacing_directive_bands(lp):
    short = lp.pacing_directive(2.5)
    assert "single clear gesture" in short
    assert "real-time" in short
    assert "slow motion" in short
    mid = lp.pacing_directive(10.1)
    assert "2-3 distinct action beats" in mid
    long_clip = lp.pacing_directive(20.04)  # 481 frames = the 20s grid value
    assert "3-4 distinct action beats" in long_clip
    very_long = lp.pacing_directive(30)
    assert "4-6 distinct action beats" in very_long
    # Bands are the pacing contract: every 2-3 seconds a visible change.
    for seconds in (2.5, 5, 10.1, 15, 30):
        directive = lp.pacing_directive(seconds)
        assert "2-3 seconds" in directive
        # Burst beats + camera energy matching ride along on every band.
        assert "short burst with a clear impact instant" in directive
        assert "never take slow camera adjectives" in directive


def test_pacing_directive_continued_clip(lp):
    plain = lp.pacing_directive(10)
    cont = lp.pacing_directive(10, continued=True)
    assert "overlap" not in plain
    assert "overlap" in cont


def test_shot_user_text_carries_pacing(lp):
    # NOTE: width/height arguments were removed — canvas now lives on
    # the Production Plan node.
    user = lp.build_shot_user_text(
        concept="courtyard summer",
        prefix_text="Same woman, white blouse.",
        category="none - 不指定",
        continuation_block="CONTINUATION RULES",
        shot={"id": "scene_01", "description": "she stirs"},
        clip_index=2,
        clip_count=3,
        duration_seconds=10.13,  # actual grid length 243 frames
        language_name="English",
    )
    # Actual duration is surfaced with one decimal.
    assert "10.1 seconds" in user
    # Band for ~10s and the anti-slow-mo contract are both present.
    assert "2-3 distinct action beats" in user
    assert "real-time" in user
    # Clip 2+ notes the carried overlap does not consume the beat budget.
    assert "overlap" in user


def test_shot_user_text_omits_private_shot_keys(lp):
    user = lp.build_shot_user_text(
        concept="courtyard summer",
        prefix_text="Same woman, white blouse.",
        category="none - 不指定",
        continuation_block="CONTINUATION RULES",
        shot={
            "id": "scene_01",
            "description": "she stirs",
            "_dialogue_lines": ["secret line"],
            "_turn_speaker": "Sahli",
        },
        clip_index=1,
        clip_count=1,
        duration_seconds=10.13,
        language_name="English",
    )
    assert "_dialogue_lines" not in user
    assert "secret line" not in user


def test_shot_system_prompt_has_anti_slow_motion(lp):
    system = lp.shot_system_prompt()
    assert "Anti-slow-motion rule" in system
    assert "words per second" in system
    # Slow camera adjectives bleed into subject motion — banned outright.
    assert "Slow-word ban" in system
    assert "never combine slow camera with fast action" in system
    # The base guide's "At 00:03.500" form must not leak into chain clips.
    assert "Timestamp ban" in system
    assert "never clock times" in system
    # Burst-style action verbs.
    assert "Strike verbs" in system


def test_template_builders(lp):
    # NOTE: build_shot_user_text / build_single_call_user_text no longer
    # take width/height — canvas lives on the Production Plan widgets.
    user = lp.build_shot_user_text(
        concept="courtyard summer",
        prefix_text="Same woman, white blouse.",
        category="cinematic-story - 电影短片/MV/戏剧",
        continuation_block="CONTINUATION RULES",
        shot={"id": "scene_01", "description": "she stirs"},
        clip_index=1,
        clip_count=3,
        duration_seconds=10,
        language_name="English",
    )
    assert "courtyard summer" in user
    assert "Same woman, white blouse." in user
    assert "CONTINUATION RULES" in user
    assert "clip 1 of 3" in user

    single = lp.build_single_call_user_text(
        concept="c",
        prefix_text="p",
        category="none - 不指定",
        shots=[{"id": "a", "description": "d"}],
        duration_seconds=10,
        language_name="English",
    )
    assert "SINGLE-CALL FORMAT OVERRIDE" in single
    assert '"id"' in single
    # Single-call mode also carries the density-vs-duration rule.
    assert "Pacing" in single
    assert "slow motion" in single


def test_single_call_user_text_carries_spatial_layout(lp):
    """Single-call mode must inject the same binding spatial-layout
    directive the per-shot path injects — the prefix alone is not the
    binding channel for positions."""
    layout = {"Subject 1": "left of frame", "Subject 2": "right of frame"}
    single = lp.build_single_call_user_text(
        concept="c",
        prefix_text="p",
        category="none - 不指定",
        shots=[{"id": "a", "description": "d"}],
        duration_seconds=10,
        language_name="English",
        spatial_layout=layout,
    )
    assert "Spatial layout" in single
    assert "Subject 1 stays at left of frame" in single
    assert "Subject 2 stays at right of frame" in single
    # No layout declared -> the neutral fallback line fills the slot
    # (mirrors build_spatial_layout_directive's contract).
    bare = lp.build_single_call_user_text(
        concept="c",
        prefix_text="p",
        category="none - 不指定",
        shots=[{"id": "a", "description": "d"}],
        duration_seconds=10,
        language_name="English",
    )
    assert "no spatial_layout declared" in bare


def test_continuation_block_carries_full_context(lp):
    block = lp.build_continuation_block(
        "scene_01",
        "A cat with a red collar perches left; the dog wears nothing.",
        "Night wind over tiles; a low creak.",
    )
    assert "scene_01" in block
    # Full previous description is embedded, not a truncated tail.
    assert "red collar" in block
    # The exact previous soundscape bed is embedded for carrying.
    assert "Night wind over tiles" in block
    # Anti-drift rules present.
    assert "who-wears/holds-what" in block
    assert "do not introduce new background elements" in block
    # Empty bed gets an explicit fallback instead of a blank section.
    block2 = lp.build_continuation_block("scene_01", "desc", "")
    assert "no explicit bed" in block2
    assert lp.split_prefix_paragraphs("A.\n\nB.\n") == ["A.", "B."]
    assert lp.split_prefix_paragraphs("  \n ") == []


def test_build_shots_digest(lp):
    shots = [
        {"id": "scene_01", "description": "First scene description."},
        {"id": "scene_02", "description": "x" * 300},
    ]
    digest = lp.build_shots_digest(shots)
    lines = digest.split("\n")
    # New format appends roster-hint flags (``# all_shots=...,
    # first_seen=...``) to the end of each line when the heuristic
    # roster extractor finds subject-name tokens.  Assert on
    # starts-with so the test stays stable whether tokens are found.
    assert lines[0].startswith("- scene_01: First scene description.")
    assert lines[1].startswith("- scene_02: ")
    # Long description is still capped with the ellipsis marker before
    # any trailing flag comment.
    body = lines[1].split("  # ", 1)[0] if "  # " in lines[1] else lines[1]
    assert body.endswith("...")
    assert len(body) <= 160 + len("- scene_02: ") + 3


def test_prefix_user_text_carries_storyboard(lp):
    user = lp.build_prefix_user_text(
        "concept",
        "cinematic-story - 电影短片/MV/戏剧",
        "English",
        shots_digest="- scene_01: a cat and a dog duel on a rooftop",
    )
    assert "- scene_01: a cat and a dog duel on a rooftop" in user
    # The prefix synth template always ships the genre advice block for
    # non-"none" categories and the two-artifact footer (prefix paragraph
    # then the CAST sheet header).  Assert on the actual template text
    # instead of the obsolete "art style" phrasing.
    assert "cinematic-story" in user or "cinematic" in user.lower()
    assert "CAST sheet" in user
    # Without a digest the placeholder keeps the template well-formed.
    user_default = lp.build_prefix_user_text("c", "none - 不指定", "English")
    assert "(no storyboard provided)" in user_default


def test_genre_advice_block(lp):
    user = lp.build_prefix_user_text(
        "concept", "action - 动作戏/打斗/飙车", "English"
    )
    assert "action" in user
    user_none = lp.build_prefix_user_text("concept", "none - 不指定", "English")
    assert "none" in user_none


# --------------------------------------------------------------------- #
# Speaker-ID contract (official H3 rule: a speaker keeps the same
# (S<n>) across shots; non-vocal characters get no ID).
# --------------------------------------------------------------------- #
def test_build_speaker_id_map_orders_by_first_appearance(lp):
    shots = [
        {"id": "scene_01", "_turn_speaker": "莎莉猫"},
        {"id": "scene_02", "_turn_speaker": "哈利猫"},
        {"id": "scene_03", "_turn_speaker": "莎莉猫"},  # repeat, no new slot
        {"id": "scene_04"},  # narration shot, ignored
        {"id": "scene_05", "_turn_speaker": "教授"},
    ]
    assert lp.build_speaker_id_map(shots) == {
        "莎莉猫": "S1",
        "哈利猫": "S2",
        "教授": "S3",
    }


def test_build_speaker_id_map_empty_for_narration(lp):
    assert lp.build_speaker_id_map([{"id": "scene_01"}, {"id": "scene_02"}]) == {}
    assert lp.build_speaker_id_map([]) == {}
    assert lp.build_speaker_id_map(None) == {}


def test_format_speaker_id_map_text(lp):
    text = lp.format_speaker_id_map_text({"莎莉猫": "S1", "哈利猫": "S2"})
    assert text == "莎莉猫=(S1), 哈利猫=(S2)"
    assert lp.format_speaker_id_map_text({}) == ""


def test_repair_speaker_ids_rewrites_wrong_number(lp):
    lines = [
        "integrated_multimodal_description:",
        "[Shot 1] The cream cat, adult female, soft timbre (S2), says "
        "<d>[Chinese] 今天天气真好。</d>",
        "",
        "overall_soundscape:",
        "quiet.",
    ]
    out, problems = lp.repair_speaker_ids(lines, "莎莉猫", "S1")
    assert problems == []
    assert "(S1)" in out[1]
    assert "(S2)" not in out[1]
    # Non-speaking lines untouched, structure preserved.
    assert out[0] == lines[0]
    assert out[3] == "overall_soundscape:"


def test_repair_speaker_ids_flags_missing_tag(lp):
    lines = [
        "integrated_multimodal_description:",
        "[Shot 1] The cat says <d>[Chinese] 今天天气真好。</d>",
    ]
    out, problems = lp.repair_speaker_ids(lines, "莎莉猫", "S1")
    assert out == lines  # nothing repairable without a tag
    assert len(problems) == 1
    assert "no (S<n>) tag" in problems[0]


def test_repair_speaker_ids_first_appearance_requires_gender(lp):
    lines_with_gender = [
        "The cat (S1) says <d>[Chinese] 你好。</d> with an adult female voice"
    ]
    out, problems = lp.repair_speaker_ids(
        lines_with_gender, "莎莉猫", "S1", first_appearance=True
    )
    assert problems == []
    lines_without_gender = [
        "The cat (S1) says <d>[Chinese] 你好。</d> quietly"
    ]
    out, problems = lp.repair_speaker_ids(
        lines_without_gender, "莎莉猫", "S1", first_appearance=True
    )
    assert len(problems) == 1
    assert "voice identity" in problems[0]
    # Non-first appearances do not need the gender restated.
    out, problems = lp.repair_speaker_ids(
        lines_without_gender, "莎莉猫", "S1", first_appearance=False
    )
    assert problems == []


def test_repair_speaker_ids_no_dialogue_is_noop(lp):
    lines = [
        "integrated_multimodal_description:",
        "[Shot 1] A cat sleeps. The other cat (S2) watches.",
    ]
    out, problems = lp.repair_speaker_ids(lines, "莎莉猫", "S1")
    assert out == lines
    assert problems == []


def test_shot_user_text_carries_speaker_id_directive(lp):
    user = lp.build_shot_user_text(
        concept="coffee shop cats",
        prefix_text="Warm amber coffee shop.",
        category="dialogue - 对白/对话/相声",
        continuation_block="CONTINUATION RULES",
        shot={"id": "scene_02", "_turn_speaker": "哈利猫"},
        clip_index=2,
        clip_count=3,
        duration_seconds=6.0,
        language_name="English",
        dialogue_lines=["好？你凭什么定义好。"],
        turn_index=2,
        turn_speaker="哈利猫",
        speaker_id_map={"莎莉猫": "S1", "哈利猫": "S2"},
    )
    # Dialogue-as-data contract: the lock instruction, the fixed map,
    # and the performance-phrase tag rule (tags allowed ONLY inside
    # the required voice phrases).
    assert "Dialogue is LOCKED" in user
    assert "REPLACES the description body" in user
    assert "verbatim <d> tag" in user
    assert "莎莉猫=(S1), 哈利猫=(S2)" in user
    assert "Speaker tags are owned by the node" in user
    assert "Write NO (S<n>) tags in your prose" in user
    assert "non-vocal on-screen characters get NO tag" in user
    # Voice performance phrases: exact descriptor, heuristic fallback
    # when no sheet given (哈利猫 has no gender marker -> stable
    # neutral adult descriptor).
    assert "Voice performance binding" in user
    assert "- 哈利猫 as (S2) adult, mid-range pitch, natural timbre" in user
    # The locked line rides as CONTEXT ONLY — a bare line, not a <d>
    # block the model could copy.
    assert "1. 哈利猫 (S2): 好？你凭什么定义好。" in user


def test_shot_user_text_without_map_omits_directive(lp):
    user = lp.build_shot_user_text(
        concept="courtyard summer",
        prefix_text="Same woman, white blouse.",
        category="none - 不指定",
        continuation_block="CONTINUATION RULES",
        shot={"id": "scene_01", "description": "she stirs"},
        clip_index=1,
        clip_count=1,
        duration_seconds=5.0,
        language_name="English",
    )
    assert "Speaker ID map" not in user


def test_single_call_user_text_carries_speaker_map(lp):
    user = lp.build_single_call_user_text(
        concept="coffee shop cats",
        prefix_text="Warm amber.",
        category="dialogue - 对白/对话/相声",
        shots=[{"id": "scene_01", "_turn_speaker": "莎莉猫"}],
        duration_seconds=20,
        language_name="English",
        speaker_id_map={"莎莉猫": "S1", "哈利猫": "S2"},
    )
    assert "莎莉猫=(S1), 哈利猫=(S2)" in user
    # No dialogue on the board -> no lock block.
    assert "Dialogue is LOCKED" not in user


def test_single_call_user_text_locks_dialogue(lp):
    user = lp.build_single_call_user_text(
        concept="coffee shop cats",
        prefix_text="Warm amber.",
        category="dialogue - 对白/对话/相声",
        shots=[{"id": "scene_01", "_turn_speaker": "莎莉猫",
                "_dialogue_lines": ["你好。"]}],
        duration_seconds=20,
        language_name="English",
        speaker_id_map={"莎莉猫": "S1"},
    )
    assert "Dialogue is LOCKED as data" in user
    assert "no <d> blocks, no quoted or unquoted spoken words" in user
    assert "the node attaches each speaker's (s<n>) tag" in user.lower()


def test_system_prompts_carry_speaker_id_rules(lp):
    # Three-section (t2va/i2va/fl2va) addendum.
    system = lp.shot_system_prompt()
    assert "Speaker IDs `(S1)`, `(S2)`" in system
    assert "never renumber, never swap, never invent new IDs" in system
    assert "get NO `(S<n>)` tag" in system
    # Six-section (ref2va) addendum.
    system6 = lp.shot_system_prompt_ref2v()
    assert "Speaker IDs `(S1)`, `(S2)`" in system6
    assert "never renumber, never swap, never invent new IDs" in system6
    assert "must not suddenly carry" in system6


# --------------------------------------------------------------------- #
# repair_speaker_ids_for_lines — multi-speaker variant used by packed
# scenes (split_bias != aggressive). Each <d> block has its own
# speaker; the function rewrites the (S<n>) tag in the delivery prose
# BEFORE that block to that speaker's mapped ID.
# --------------------------------------------------------------------- #
def test_repair_speaker_ids_for_lines_rewrites_two_blocks(lp):
    """The model emits Sahli with the wrong (S9) tag and Harry with
    (S2) in a packed scene — only the FIRST block's delivery prose
    carries the wrong number; the second block is already correct.
    The function must rewrite the wrong tag to Sahli's mapped S1,
    leave Harry's correct S2 alone, and not flag anything."""
    lines = [
        "integrated_multimodal_description:",
        "The cream cat, adult female, soft timbre (S9), says "
        "<d>[Chinese] 今天天气真好。</d> The orange cat, adult male, "
        "mid pitch (S2), then says <d>[Chinese] 好？你凭什么定义好。</d>",
        "",
        "overall_soundscape:",
        "Cafe hush.",
    ]
    sid_map = {"莎莉猫": "S1", "哈利猫": "S2"}
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["莎莉猫", "哈利猫"], sid_map
    )
    assert problems == [], f"unexpected problems: {problems}"
    prompt_text = "\n".join(out)
    # Sahli's delivery prose now carries the FIXED (S1) tag.
    assert "(S1)" in prompt_text
    assert "(S9)" not in prompt_text
    # Harry's (S2) survives untouched (no false rewrite).
    assert "(S2)" in prompt_text
    # Both <d> blocks kept verbatim.
    assert "<d>[Chinese] 今天天气真好。</d>" in prompt_text
    assert "<d>[Chinese] 好？你凭什么定义好。</d>" in prompt_text


def test_repair_speaker_ids_for_lines_flags_mismatch(lp):
    """If the LLM reply has 2 <d> blocks but the caller only declared
    1 speaker, the contract cannot be enforced — flag and bail."""
    lines = [
        "[Shot 1] Beat one: <d>[Chinese] 第一句。</d> "
        "Beat two: <d>[Chinese] 第二句。</d>"
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["only_one_speaker"], {"only_one_speaker": "S1"}
    )
    assert any("mismatch" in p and "2 <d>" in p for p in problems), (
        f"expected mismatch problem; got {problems}"
    )


def test_repair_speaker_ids_for_lines_missing_tag_first_block(lp):
    """A speaker's FIRST block in this shot must carry their tag. If
    the delivery prose has no (S<n>) tag, the function flags it."""
    lines = [
        "[Shot 1] The cream cat says <d>[Chinese] 第一句。</d> "
        "Harry, adult male, mid pitch (S2), then says "
        "<d>[Chinese] 第二句。</d>"
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["莎莉猫", "哈利猫"],
        {"莎莉猫": "S1", "哈利猫": "S2"},
    )
    assert any("carries no" in p and "莎莉猫" in p for p in problems), (
        f"expected missing-tag problem for 莎莉猫; got {problems}"
    )


def test_repair_speaker_ids_for_lines_first_appearance_needs_gender(lp):
    """A speaker making their FIRST spoken appearance in the video
    must state gender/pitch/timbre beside the tag at their first
    block in this shot. The tag is present but no gender word — flag."""
    lines = [
        "[Shot 1] Sahli quietly (S1) says <d>[Chinese] 你好。</d>"
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["Sahli"], {"Sahli": "S1"},
        first_appearance_speakers={"Sahli"},
    )
    assert any("first spoken appearance" in p and "Sahli" in p for p in problems), (
        f"expected first-appearance gender problem; got {problems}"
    )


def test_repair_speaker_ids_for_lines_first_appearance_with_gender_passes(lp):
    """Same setup but gender IS stated → no first-appearance problem."""
    lines = [
        "[Shot 1] Sahli, adult female, soft timbre (S1), says "
        "<d>[Chinese] 你好。</d>"
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["Sahli"], {"Sahli": "S1"},
        first_appearance_speakers={"Sahli"},
    )
    assert problems == [], f"unexpected problems: {problems}"
    assert "(S1)" in "\n".join(out)


def test_repair_speaker_ids_for_lines_tail_not_rewritten(lp):
    """Text AFTER the last <d> block on a line is the closing reaction
    — its tag ownership is ambiguous, so the function must NOT rewrite
    it (only the segment before the next block counts as 'delivery
    prose' for that line)."""
    lines = [
        "[Shot 1] Sahli, adult female (S1), says "
        "<d>[Chinese] 你好。</d> The cat then turns to Harry (S9) "
        "and waits.",
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["Sahli"], {"Sahli": "S1"},
    )
    # The (S9) tag in the tail is left untouched.
    assert "(S9)" in "\n".join(out), (
        "tail-text tags must not be rewritten; got:\n" + "\n".join(out)
    )
    # And the first-block delivery prose was rewritten to S1.
    assert "Sahli, adult female (S1)" in "\n".join(out)


def test_repair_speaker_ids_for_lines_no_dialogue_is_noop(lp):
    """Lines without <d> blocks pass through untouched and produce
    no problems."""
    lines = [
        "integrated_multimodal_description:",
        "[Shot 1] A cat sleeps on the chair.",
        "",
        "overall_soundscape:",
        "Cafe hush.",
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, [], {"Sahli": "S1"},
    )
    assert out == lines
    assert problems == []


def test_repair_speaker_ids_for_lines_block_count_match_no_problems(lp):
    """Sanity: balanced two-speaker packed scene with two well-formed
    <d> blocks (each carrying the correct tag) produces no problems."""
    lines = [
        "Sahli, adult female, soft timbre (S1), says "
        "<d>[Chinese] 你好。</d> Harry, adult male, mid pitch (S2), "
        "then says <d>[Chinese] 你也好。</d>"
    ]
    out, problems = lp.repair_speaker_ids_for_lines(
        lines, ["Sahli", "Harry"],
        {"Sahli": "S1", "Harry": "S2"},
        first_appearance_speakers={"Sahli", "Harry"},
    )
    assert problems == [], f"unexpected problems: {problems}"


def test_build_tempo_directive(lp):
    fast = lp.build_tempo_directive("fast")
    assert "BRISK" in fast and "chain" in fast
    slow = lp.build_tempo_directive("slow")
    assert "MEASURED" in slow and "never slow motion" in slow
    assert "NATURAL" in lp.build_tempo_directive("normal")
    # Unknown keys fall back to natural tempo; the directive is binding
    # prose, injected verbatim into every per-shot template.
    assert "NATURAL" in lp.build_tempo_directive("")
    assert "NATURAL" in lp.build_tempo_directive("weird")


# --------------------------------------------------------------------- #
# Dialogue-as-data: assemble / scrub / append helpers
# --------------------------------------------------------------------- #
def test_assemble_dialogue_line_blocks_first_appearance_identity(lp):
    blocks = lp.assemble_dialogue_line_blocks(
        ["你好。", "好的。"],
        line_speakers=["莎莉猫", "哈利猫"],
        turn_speaker="",
        speaker_id_map={"莎莉猫": "S1", "哈利猫": "S2"},
        speaker_identities={"莎莉猫": "cream cat", "哈利猫": "tabby"},
        first_appearance_speakers={"莎莉猫", "哈利猫"},
    )
    voice = "adult, mid-range pitch, natural timbre"
    assert blocks == [
        f"莎莉猫 speaks as (S1) {voice} <d>[Chinese] 你好。</d>",
        f"哈利猫 speaks as (S2) {voice} <d>[Chinese] 好的。</d>",
    ]
    for speech in blocks:
        assert ":" not in speech.split("<d>")[0]
        assert "cream cat" not in speech and "tabby" not in speech


def test_assemble_dialogue_line_blocks_later_appearance_bare(lp):
    blocks = lp.assemble_dialogue_line_blocks(
        ["再见。"],
        line_speakers=["莎莉猫"],
        turn_speaker="莎莉猫",
        speaker_id_map={"莎莉猫": "S1"},
        speaker_identities={"莎莉猫": "cream cat"},
        first_appearance_speakers=set(),
    )
    assert blocks == [
        "莎莉猫 speaks as (S1) adult, mid-range pitch, natural timbre "
        "<d>[Chinese] 再见。</d>"
    ]


def test_assemble_dialogue_line_blocks_english_tag_and_no_map(lp):
    blocks = lp.assemble_dialogue_line_blocks(
        ["hello there."],
        line_speakers=["Sahli"],
        turn_speaker="Sahli",
        first_appearance_speakers=set(),
    )
    assert blocks == [
        "Sahli speaks, adult, mid-range pitch, natural timbre "
        "<d>[English] hello there.</d>"
    ]
    blocks = lp.assemble_dialogue_line_blocks(
        ["hello."], line_speakers=["Sahli"], turn_speaker="Sahli",
        speaker_id_map={"Sahli": "S1"}, first_appearance_speakers=set(),
    )
    assert blocks == [
        "Sahli speaks as (S1) adult, mid-range pitch, natural timbre "
        "<d>[English] hello.</d>"
    ]


def test_vocative_facing_uses_role_binding_and_layout(lp):
    """Mommy, ... faces 妈妈's character on her side; a later "daddy"
    in the same line does not steal the addressee. Daddy, ... faces 爸爸."""
    concept = (
        "图1的黑猫是爸爸，图2的白猫是妈妈，图3的小猫是女儿。\n"
        "小猫：Mommy, daddy is so ugly, why did you marry him.\n"
        "小猫：Daddy, why did you marry a blind lady?"
    )
    bindings = lp.extract_role_bindings(concept)
    assert bindings == {"爸爸": "黑猫", "妈妈": "白猫", "女儿": "小猫"}
    layout = {
        "黑猫": "left of frame",
        "白猫": "right of frame",
        "小猫": "center of frame",
    }
    voice = "child voice, high pitch, bright timbre"
    blocks = lp.assemble_dialogue_line_blocks(
        [
            "Mommy, daddy is so ugly, why did you marry him.",
            "I guess I was blind.",
            "Daddy, why did you marry a blind lady?",
        ],
        line_speakers=["小猫", "白猫", "小猫"],
        speaker_id_map={"小猫": "S1", "白猫": "S2", "黑猫": "S3"},
        speaker_voices={"小猫": voice, "白猫": "adult female, warm mid-range pitch"},
        concept=concept,
        spatial_layout=layout,
    )
    assert "turns to face 白猫 on the RIGHT" in blocks[0]
    assert "turns to face 黑猫" not in blocks[0]
    assert f"mouth opening as (S1) {voice} <d>" in blocks[0]
    assert "turns to face" not in blocks[1]
    assert "白猫 speaks as (S2) adult female, warm mid-range pitch <d>" in blocks[1]
    assert "turns to face 黑猫 on the LEFT" in blocks[2]
    assert ":" not in blocks[2].split("<d>")[0]


def test_vocative_without_binding_or_self_address_has_no_facing(lp):
    concept = "黑猫是爸爸。"
    assert lp.extract_role_bindings(concept) == {"爸爸": "黑猫"}
    blocks = lp.assemble_dialogue_line_blocks(
        ["Daddy, hi.", "Hello."],
        line_speakers=["黑猫", "白猫"],
        speaker_id_map={"黑猫": "S1", "白猫": "S2"},
        concept=concept,
        spatial_layout={"黑猫": "left of frame"},
    )
    # 黑猫 would be told to face himself.
    assert "turns to face" not in blocks[0]
    assert "speaks as (S1) " in blocks[0]
    # 白猫's line has no vocative.
    assert "turns to face" not in blocks[1]


def test_paren_role_and_negated_copula(lp):
    bound = lp.extract_role_bindings(
        "参考图 1 → 黑猫（猫爸爸，画面右边）；参考图 2 → 白猫（猫妈妈，画面左边）。"
        "他不是爸爸。"
    )
    assert bound["爸爸"] == "黑猫"
    assert bound["妈妈"] == "白猫"
    assert "不" not in bound.values()
    assert "他不" not in bound.values()


def test_offscreen_claim_is_removed_not_rewritten(lp):
    lines = [
        "summary:",
        "First-person POV from the mother, who stays behind the camera.",
        "retention_analysis:",
        "<Subject 2> serves as the locked-off camera off-screen.",
        "detailed_description:",
        "白猫 speaks as (S2) adult female, warm mid-range pitch <d>[English] Hi.</d>",
    ]
    out = lp.neutralize_offscreen_claims(lines)
    text = "\n".join(out)
    assert "behind the camera" not in text
    assert "off-screen" not in text
    assert "POV" not in text
    assert "None of them is the camera." in text
    assert "<d>[English] Hi.</d>" in text


def test_voice_descriptor_drops_trailing_period(lp):
    blocks = lp.assemble_dialogue_line_blocks(
        ["Oh my god."],
        line_speakers=["白猫"],
        speaker_id_map={"白猫": "S2"},
        speaker_voices={"白猫": "adult female, warm mid-range pitch, soft rounded timbre."},
    )
    assert blocks[0].endswith(
        "as (S2) adult female, warm mid-range pitch, soft rounded timbre "
        "<d>[English] Oh my god.</d>"
    )
    assert "timbre." not in blocks[0]


def test_locked_line_marks_vocative_addressee(lp):
    user = lp.build_shot_user_text(
        concept="图2的白猫是妈妈，图1的黑猫是爸爸。",
        prefix_text="p",
        category="dialogue - 对白/对话/相声",
        continuation_block="",
        shot={"id": "scene_01"},
        clip_index=1,
        clip_count=1,
        duration_seconds=6.0,
        language_name="English",
        dialogue_lines=["Daddy, why did you marry a blind lady?"],
        turn_index=1,
        turn_speaker="小猫",
        line_speakers=["小猫"],
        speaker_id_map={"小猫": "S1", "黑猫": "S2", "白猫": "S3"},
        spatial_layout={"黑猫": "left of frame", "白猫": "right of frame"},
    )
    assert "小猫 (S1) -> 黑猫 (LEFT of frame): Daddy, why" in user


def test_scrub_dialogue_from_prompt_text(lp):
    text = "he says: <d>[Chinese] 你好。</d> and again 你好。 plus ok"
    out, notes = lp.scrub_dialogue_from_prompt_text(text, ["你好。"])
    assert "<d>" not in out
    assert "你好。" not in out
    assert "ok" in out and "…" in out
    assert notes


def test_scrub_dialogue_from_prompt_text_clean_passthrough(lp):
    out, notes = lp.scrub_dialogue_from_prompt_text(
        "she gestures, ambient rain", ["你好。"]
    )
    assert out == "she gestures, ambient rain"
    assert notes == []


def test_append_dialogue_blocks_to_sections_three(lp):
    lines = [
        "integrated_multimodal_description:", "body",
        "", "overall_soundscape:", "s",
        "", "non_diegetic_music:", "m",
    ]
    out = lp.append_dialogue_blocks_to_sections(
        lines, ["B1"], schema="three_section"
    )
    assert out == [
        "integrated_multimodal_description:", "body", "B1",
        "", "overall_soundscape:", "s",
        "", "non_diegetic_music:", "m",
    ]


def test_append_dialogue_blocks_to_sections_six(lp):
    lines = [
        "subject_definitions:", "x", "summary:", "y",
        "retention_analysis:", "z", "detailed_description:", "d1",
        "", "overall_soundscape:", "s",
        "", "non_diegetic_music:", "m",
    ]
    out = lp.append_dialogue_blocks_to_sections(
        lines, ["B1"], schema="six_section"
    )
    assert out[6:10] == ["detailed_description:", "d1", "B1", ""]
