# -*- coding: utf-8 -*-
"""§7.1 golden-equivalence test (PLAN_STORYBOARD_ASSET.md).

The golden fixture (``tests/fixtures/plan_english_kitten_grandpa.json``)
was the reference target when the plan was authored: an LLM-track
plan that the zero-LLM builder must reproduce structurally (the prose
will be flatter by design — PLAN §6.6). This test hand-authors a
storyboard equivalent to the golden plan and runs ``assemble_plan``
against it, then asserts every structural invariant the plan lists
under §7.1:

  1. 台词逐字、说话人正确
  2. 每场都有每个角色的 ``<Picture i>`` 和场景的 Picture
  3. 每场主体行都是 ``<Subject k> -> <Picture m>:``，格式全片一致
  4. 空间钉是 ``名字 stays ...``
  5. 六段 header 顺序齐全
  6. 没有 ``[Shot 2]``、``At 00:``、``@alias``
  7. Setting 文本等于分镜的英文 setting_line
  8. retention 等于 build_fallback_shot_lines 的固定句
  9. ``<d>`` 标签等于分镜 ``language``
  10. ``(Sn)`` 等于 character_slot

The fixture deliberately DIVERGES from the golden plan in two places
the plan explicitly calls out as "we don't replicate":

  * the LLM-track golden plan has irregular `<Subject K>` formatting
    across scenes (scene_01 binds `<Subject 1> -> <Picture 1>` but
    scene_02/03/04 leave the binding off subject_definitions and
    splice it into retention). The mechanical builder is consistent
    across every scene (§7.1 "金样自己在 scene_01 和 scene_02 上格式
    不一致，机械结果不复制这种不一致").
  * the LLM-track golden plan smuggles the Chinese Setting sentence
    into `Setting: 客厅里，猫爷爷和小猫妹妹坐在沙发上聊天。`. The
    mechanical builder uses the author-supplied English setting_line
    verbatim.

The test asserts the structural invariants the plan demands AND
asserts these two known divergences so the suite documents them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
GOLDEN_PLAN = PROJECT_DIR / "tests" / "fixtures" / "plan_english_kitten_grandpa.json"


# --------------------------------------------------------------------------- #
# Module loaders (no ComfyUI)
# --------------------------------------------------------------------------- #
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
def sb_modules():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.assets", PROJECT_DIR / "nodes" / "assets")
    _load_file(
        "_mienodes_internal.nodes.assets.library",
        PROJECT_DIR / "nodes" / "assets" / "library.py",
    )
    _load_file(
        "_mienodes_internal.nodes.assets.storyboard",
        PROJECT_DIR / "nodes" / "assets" / "storyboard.py",
    )
    _ensure_pkg("_mienodes_internal.nodes.llm", PROJECT_DIR / "nodes" / "llm")
    _ensure_pkg(
        "_mienodes_internal.nodes.llm.prompts",
        PROJECT_DIR / "nodes" / "llm" / "prompts",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.prompts.loader",
        PROJECT_DIR / "nodes" / "llm" / "prompts" / "loader.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.h3_prompts",
        PROJECT_DIR / "nodes" / "llm" / "h3_prompts.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
        PROJECT_DIR / "nodes" / "llm" / "minimax_h3_storyboard_prompts.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        PROJECT_DIR / "nodes" / "llm" / "minimax_h3_loop_prompts.py",
    )
    _load_file(
        "_mienodes_internal.nodes.llm.dialogue_segmenter",
        PROJECT_DIR / "nodes" / "llm" / "dialogue_segmenter.py",
    )
    builder = _load_file(
        "_mienodes_internal.nodes.llm.storyboard_builder",
        PROJECT_DIR / "nodes" / "llm" / "storyboard_builder.py",
    )
    return {
        "builder": builder,
        "storyboard_lib": sys.modules[
            "_mienodes_internal.nodes.assets.storyboard"
        ],
    }


# --------------------------------------------------------------------------- #
# The storyboard equivalent of the golden plan
# --------------------------------------------------------------------------- #
# Authors manually inspected ``plan_english_kitten_grandpa.json`` (4
# shots: kitten talks to grandfather about a childhood hedgehog memory).
# This storyboard is the PLAN §3.1 example shape: cast by character_slot,
# English setting_line + picture cue on shot 2 line 1, language=zh to
# match the LLM track.
def _golden_storyboard():
    return {
        "name": "板栗对话金样",
        "version": 1,
        "language": "zh",
        "camera_discipline": "locked",
        "pacing": "normal",
        "seed_policy": "per_scene_increment",
        "prompt_prefix": (
            "A cozy cottage living room, polished 3D, warm light. "
            "The camera stays locked off in one continuous medium two-shot."
        ),
        "cast": {
            "小猫妹妹": {"character_slot": 1},
            "猫爷爷": {"character_slot": 2},
        },
        "setting_line": (
            "Grandpa and the kitten girl sit together on the sofa, talking."
        ),
        "spatial_layout": {
            "小猫妹妹": "left of frame",
            "猫爷爷": "right of frame",
        },
        "ambience": "Quiet room tone with soft indoor air.",
        "shots": [
            {"lines": [
                {"speaker": "小猫妹妹",
                 "text": "爷爷，我一直有个心理阴影。"},
                {"speaker": "猫爷爷", "text": "是什么呢。"},
            ]},
            {"lines": [
                {"speaker": "小猫妹妹",
                 "text": "小时候我还没上小学。",
                 "picture": "She keeps both paws in her lap."},
                {"speaker": "小猫妹妹",
                 "text": "咱家院子里有几只小刺猬。"},
            ]},
            {"lines": [
                {"speaker": "小猫妹妹",
                 "text": "我把剩饭给他们，他们又不吃，也不跑。"},
            ]},
            {"lines": [
                {"speaker": "猫爷爷", "text": "那些刺猬现在还在吗？"},
                {"speaker": "小猫妹妹", "text": "还在。"},
            ]},
        ],
    }


def _golden_bundles(sb_modules):
    """Slot-keyed dict: {1: kitten bundle, 2: grandpa bundle, 3: scene}."""
    import torch
    return {
        1: {
            "kind": "character",
            "name": "小猫妹妹",
            "description": (
                "The character is calico_kitten_sister, a young female "
                "calico kitten with fluffy cream, orange, and charcoal-black "
                "fur. She wears a cream-colored short-sleeved floral dress."
            ),
            "voice": (
                "young female, soft and sweet timbre, mid register, gentle pace"
            ),
            "image": torch.zeros(1, 8, 8, 3),
            "audio": {
                "waveform": torch.zeros(1, 1, 1600),
                "sample_rate": 16000,
            },
        },
        2: {
            "kind": "character",
            "name": "猫爷爷",
            "description": (
                "grandfather_cat is an elderly male domestic longhair cat "
                "with fluffy orange tabby fur. He wears a plaid shirt beneath "
                "a dark forest-green cable-knit cardigan and a brown tweed cap."
            ),
            "voice": (
                "elderly male, soft and mildly gravelly timbre, low register, "
                "slow and unhurried pace"
            ),
            "image": torch.zeros(1, 8, 8, 3),
            "audio": {
                "waveform": torch.zeros(1, 1, 1600),
                "sample_rate": 16000,
            },
        },
        3: {
            "kind": "scene",
            "name": "客厅",
            "description": (
                "A cozy cottage living room arranged around a cream-and-olive "
                "loveseat against floral wallpaper; warm golden sunlight."
            ),
            "ambience": "",
            "image": torch.zeros(1, 8, 8, 3),
        },
    }


# --------------------------------------------------------------------------- #
# §7.1 assertions
# --------------------------------------------------------------------------- #
class TestGoldenEquivalence:
    def test_run_assembles_without_error(self, sb_modules):
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=611937399731,
        )
        assert "plan" in out
        assert len(out["plan"]["shots"]) == 4

    def test_dialogue_lines_are_byte_identical(self, sb_modules):
        """§7.1 #1: 台词逐字、说话人正确."""
        builder = sb_modules["builder"]
        storyboard = _golden_storyboard()
        out = builder.assemble_plan(
            storyboard=storyboard,
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot, sl in zip(out["plan"]["shots"], storyboard["shots"]):
            text = "\n".join(shot["prompt"])
            for ln in sl["lines"]:
                tag = "[Chinese]" if storyboard["language"] == "zh" else "[English]"
                assert "<d>%s %s</d>" % (tag, ln["text"]) in text

    def test_picture_labels_every_shot(self, sb_modules):
        """§7.1 #2: 每场都有每个角色的 <Picture i> 和场景的 Picture."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            for pic in ("Picture 1", "Picture 2", "Picture 3"):
                assert pic in text, (
                    "shot %s missing %s" % (shot["id"], pic)
                )

    def test_subject_binding_is_consistent_across_shots(self, sb_modules):
        """§7.1 #3: 每场主体行都是 <Subject k> -> <Picture m>:，格式全片一致.

        The golden plan is irregular across scenes (scene_01's
        retention lines lack the ``-> <Picture N>`` binding, while
        scene_02 carries it). The mechanical builder produces the
        consistent ``<Subject K> -> <Picture M>`` shape in
        subject_definitions on every shot — that is the §7.1
        acceptance bar (PLAN §7.1 "金样自己在 scene_01 和 scene_02 上
        格式不一致，机械结果不复制这种不一致")."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            assert "<Subject 1> -> <Picture 1>" in text
            assert "<Subject 2> -> <Picture 2>" in text
        # The golden fixture has at least one scene whose retention
        # line drops the ``-> <Picture N>`` binding — confirm the
        # divergence is documented (not enforced as a check).
        golden = json.loads(GOLDEN_PLAN.read_text(encoding="utf-8"))
        irregular_found = False
        for scene in golden["shots"]:
            retention = next(
                (
                    blk for blk in scene["prompt"]
                    if blk.startswith("retention_analysis:")
                ),
                None,
            )
            if retention is None:
                continue
            body = "\n".join(
                scene["prompt"][
                    scene["prompt"].index(retention) + 1:
                ]
            ).split("detailed_description:")[0]
            for line in body.splitlines():
                if line.startswith("<Subject") and "->" not in line:
                    irregular_found = True
                    break
            if irregular_found:
                break
        assert irregular_found, (
            "expected the LLM-track golden fixture to have at least one "
            "scene whose retention omits the <Subject K> -> <Picture M> "
            "binding (the mechanical builder's consistency bar exists "
            "because of this irregularity)"
        )

    def test_spatial_pin_format(self, sb_modules):
        """§7.1 #4: 空间钉是 '名字 stays ...'."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            assert "小猫妹妹 stays at left of frame" in text
            assert "猫爷爷 stays at right of frame" in text

    def test_six_headers_in_order_every_shot(self, sb_modules):
        """§7.1 #5: 六段 header 顺序齐全."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        required = [
            "subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music",
        ]
        for shot in out["plan"]["shots"]:
            stripped = [
                (ln.strip().rstrip(":").lower() if isinstance(ln, str) else "")
                for ln in shot["prompt"]
            ]
            positions = [(stripped.index(h), h) for h in required if h in stripped]
            assert [h for _, h in sorted(positions)] == required

    def test_no_forbidden_markers(self, sb_modules):
        """§7.1 #6: 没有 [Shot 2] / At 00: / @alias."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            assert "[Shot 2]" not in text
            assert "At 00:" not in text
            assert "@alias" not in text

    def test_setting_line_is_authored_english(self, sb_modules):
        """§7.1 #7: Setting 文本等于分镜的英文 setting_line. The golden
        plan's Chinese Setting is NOT replicated."""
        builder = sb_modules["builder"]
        storyboard = _golden_storyboard()
        out = builder.assemble_plan(
            storyboard=storyboard,
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            assert "Setting: %s" % storyboard["setting_line"] in text
        # The golden plan smuggles Chinese in Setting — confirm we did
        # NOT inherit it.
        golden = json.loads(GOLDEN_PLAN.read_text(encoding="utf-8"))
        golden_setting = next(
            ln for ln in golden["shots"][0]["prompt"]
            if ln.startswith("Setting:")
        )
        assert "客厅" in golden_setting  # documents the divergence
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            for ln in text.splitlines():
                if ln.startswith("Setting:"):
                    assert "客厅" not in ln

    def test_retention_is_the_fixed_line(self, sb_modules):
        """§7.1 #8: retention 等于 build_fallback_shot_lines 的固定句.

        The mechanical retention line is
        '<Subject K> -> <Picture M>: fully_preserved - preserve face
        identity, hair, wardrobe, and scale; do not preserve the
        reference head angle or eye direction.'"""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        fixed_phrase = (
            "fully_preserved - preserve face identity, hair, wardrobe, "
            "and scale; do not preserve the reference head angle or "
            "eye direction."
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            assert fixed_phrase in text

    def test_dialogue_tag_matches_storyboard_language(self, sb_modules):
        """§7.1 #9: <d> 标签等于分镜 language."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            text = "\n".join(shot["prompt"])
            assert "<d>[Chinese]" in text
            assert "<d>[English]" not in text

    def test_speaker_tag_matches_character_slot(self, sb_modules):
        """§7.1 #10: (Sn) 等于 character_slot.

        Specifically: in the golden storyboard, 小猫妹妹 is on slot 1
        and 猫爷爷 is on slot 2. The mechanical builder must mirror
        that. Crucially, in shot 3 the only speaker is 小猫妹妹, so
        that scene carries (S1) only — there is no slot 2 dialog in
        that shot."""
        builder = sb_modules["builder"]
        out = builder.assemble_plan(
            storyboard=_golden_storyboard(),
            slot_bundles=_golden_bundles(sb_modules),
            seed_base=1,
        )
        # shot_01: 小猫妹妹 → (S1), 猫爷爷 → (S2).
        text0 = "\n".join(out["plan"]["shots"][0]["prompt"])
        assert "小猫妹妹 speaks as (S1)" in text0
        assert "猫爷爷 speaks as (S2)" in text0
        # shot_03: 小猫妹妹 alone → (S1); (S2) absent (no grandpa dialog).
        text2 = "\n".join(out["plan"]["shots"][2]["prompt"])
        assert "小猫妹妹 speaks as (S1)" in text2
        assert "speaks as (S2)" not in text2

    def test_byte_reproducible(self, sb_modules):
        """§6.2: byte-identical plan_json on a repeat call."""
        builder = sb_modules["builder"]
        sb = _golden_storyboard()
        bu = _golden_bundles(sb_modules)
        out_a = builder.assemble_plan(
            storyboard=sb, slot_bundles=bu, seed_base=42,
        )
        out_b = builder.assemble_plan(
            storyboard=sb, slot_bundles=bu, seed_base=42,
        )
        a = json.dumps(out_a["plan"], ensure_ascii=False, sort_keys=True)
        b = json.dumps(out_b["plan"], ensure_ascii=False, sort_keys=True)
        assert a == b


# --------------------------------------------------------------------------- #
# Bonus: when ``shots`` is omitted and ``lines`` is provided, the
# dialogue_segmenter's minimal-cut rule takes over (PLAN §3.1).
# --------------------------------------------------------------------------- #
class TestAutoPackFromLines:
    def test_lines_field_auto_packs_into_scenes(self, sb_modules):
        builder = sb_modules["builder"]
        storyboard = {
            "name": "autopack",
            "version": 1,
            "language": "zh",
            "camera_discipline": "locked",
            "pacing": "normal",
            "seed_policy": "per_scene_increment",
            "prompt_prefix": "global style.",
            "cast": {
                "甲": {"character_slot": 1},
                "乙": {"character_slot": 2},
            },
            "lines": [
                {"speaker": "甲", "text": "一"},
                {"speaker": "乙", "text": "二"},
                {"speaker": "甲", "text": "三"},
                {"speaker": "乙", "text": "四"},
                {"speaker": "甲", "text": "五"},
            ],
        }
        bundles = _golden_bundles(sb_modules)  # not used for slot mapping,
        # but we need the slot-keyed shape with empty slots 1 & 2 since
        # the test uses 甲 and 乙 names — keep the same shape.
        import torch
        bundles_for_autopack = {
            1: {
                "kind": "character", "name": "甲",
                "description": "甲 desc.",
                "voice": "young female, soft timbre",
                "image": torch.zeros(1, 8, 8, 3),
                "audio": {
                    "waveform": torch.zeros(1, 1, 1600),
                    "sample_rate": 16000,
                },
            },
            2: {
                "kind": "character", "name": "乙",
                "description": "乙 desc.",
                "voice": "elderly male, low voice",
                "image": torch.zeros(1, 8, 8, 3),
                "audio": {
                    "waveform": torch.zeros(1, 1, 1600),
                    "sample_rate": 16000,
                },
            },
            3: {
                "kind": "scene", "name": "Room",
                "description": "A room.",
                "image": torch.zeros(1, 8, 8, 3),
            },
        }
        out = builder.assemble_plan(
            storyboard=storyboard,
            slot_bundles=bundles_for_autopack,
            seed_base=1,
        )
        # Five lines packed 2-per-scene → 3 scenes (2+2+1).
        assert len(out["plan"]["shots"]) == 3

    def test_both_shots_and_lines_is_rejected(self, sb_modules):
        builder = sb_modules["builder"]
        storyboard = {
            "name": "x", "version": 1, "language": "en",
            "camera_discipline": "locked", "pacing": "normal",
            "seed_policy": "per_scene_increment",
            "prompt_prefix": "p",
            "cast": {"A": {"character_slot": 1}},
            "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
            "lines": [{"speaker": "A", "text": "hi"}],
        }
        bundles = _golden_bundles(sb_modules)
        with pytest.raises(ValueError, match="both"):
            builder.assemble_plan(
                storyboard=storyboard,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_neither_shots_nor_lines_is_rejected(self, sb_modules):
        builder = sb_modules["builder"]
        # Single-character storyboard: max cast slot = 1, scene at slot 2.
        # Provide a scene bundle at slot 2 so the assembler reaches the
        # "neither shots nor lines" branch instead of failing on the
        # scene-slot check first.
        import torch
        storyboard = {
            "name": "x", "version": 1, "language": "en",
            "camera_discipline": "locked", "pacing": "normal",
            "seed_policy": "per_scene_increment",
            "prompt_prefix": "p",
            "cast": {"A": {"character_slot": 1}},
        }
        bundles = {
            1: {
                "kind": "character", "name": "A",
                "description": "A desc.",
                "voice": "young female, soft timbre",
                "image": torch.zeros(1, 8, 8, 3),
                "audio": {
                    "waveform": torch.zeros(1, 1, 1600),
                    "sample_rate": 16000,
                },
            },
            2: {
                "kind": "scene", "name": "Room",
                "description": "A room.",
                "image": torch.zeros(1, 8, 8, 3),
            },
        }
        with pytest.raises(ValueError, match="must declare"):
            builder.assemble_plan(
                storyboard=storyboard,
                slot_bundles=bundles,
                seed_base=1,
            )