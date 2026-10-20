# -*- coding: utf-8 -*-
"""V3 zero-LLM generator + the first real storyboard assets
(docs/PLAN_STORYBOARD_FIRST_ASSET_V3.md §6).

Covers the six acceptance items:

  §6.1  en storyboard reproduces the golden plan's structure
  §6.2  V3 and Build emit byte-identical ``plan_json``
  §6.3  zh storyboard: 7 verbatim Chinese lines + ``[Chinese]`` tags
  §6.4  error text passes through V3 unchanged from Build
  §6.5  V3 is registered with a bilingual display name
  §6.6  ``IS_CHANGED`` tracks storyboard + asset content

The fixtures (``tests/fixtures/storyboard_banli_{en,zh}.json``) were
harvested from ``plan_english_kitten_grandpa.json`` by
``scripts/harvest_storyboard_from_plan.py`` and then hand-completed:
the harvest refuses to inherit the golden plan's Chinese ``Setting``
line, so ``setting_line`` is an authored English sentence
(PLAN_STORYBOARD_FIRST_ASSET_V3 §3.1).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_DIR / "tests" / "fixtures"
GOLDEN_PLAN = FIXTURES / "plan_english_kitten_grandpa.json"
EN_STORYBOARD = FIXTURES / "storyboard_banli_en.json"
ZH_STORYBOARD = FIXTURES / "storyboard_banli_zh.json"


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
        "prompts": sys.modules["_mienodes_internal.nodes.llm.minimax_h3_loop_prompts"],
        "storyboard_lib": sys.modules["_mienodes_internal.nodes.assets.storyboard"],
    }


# --------------------------------------------------------------------------- #
# Fixtures + bundle helpers
# --------------------------------------------------------------------------- #
def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def en_storyboard():
    return _load(EN_STORYBOARD)


@pytest.fixture(scope="module")
def zh_storyboard():
    return _load(ZH_STORYBOARD)


def _bundles() -> dict:
    """Slot-keyed bundles for 小猫妹妹(1) / 猫爷爷(2) / 客厅(3)."""
    import torch
    return {
        1: {
            "kind": "character",
            "name": "小猫妹妹",
            "description": (
                "A young female calico kitten with fluffy cream, orange and "
                "charcoal-black fur, large dark eyes and a pink nose. She wears "
                "a cream short-sleeved floral dress with a Peter Pan collar."
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
                "An elderly male domestic longhair cat with fluffy orange tabby "
                "fur, white cheeks and a long ringed orange tail. He wears a "
                "plaid shirt under a dark green cable-knit cardigan and a tweed cap."
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
                "A cozy cottage living room around a cream-and-olive loveseat "
                "against floral wallpaper, with warm golden sunlight."
            ),
            "ambience": "",
            "image": torch.zeros(1, 8, 8, 3),
        },
    }


def _socket_kwargs(bundles: dict) -> dict:
    """Bundle dict → the node's socket kwargs (character_1/2 + scene_1)."""
    return {
        "character_1": bundles[1],
        "character_2": bundles[2],
        "scene_1": bundles[3],
    }


def _structural_assertions(plan: dict, storyboard: dict, sb_modules) -> None:
    """The §7.1 structural set from PLAN_STORYBOARD_ASSET, applied to any
    assembled plan. Shared by §6.1 (en) and §6.3 (zh)."""
    prompts = sb_modules["prompts"]

    # Top-level keys: shots + prompt_prefix only.
    assert set(plan.keys()) == {"shots", "prompt_prefix"}

    shots = plan["shots"]
    assert len(shots) == len(storyboard["shots"])

    tag = "[Chinese]" if storyboard["language"] == "zh" else "[English]"
    required_headers = list(prompts.SIX_SECTION_FIELDS)
    fixed_retention = (
        "fully_preserved - preserve face identity, hair, wardrobe, and "
        "scale; do not preserve the reference head angle or eye direction."
    )

    for shot, sl in zip(shots, storyboard["shots"]):
        text = "\n".join(shot["prompt"])

        # §7.1 #5: six headers, in order.
        stripped = [
            (ln.strip().rstrip(":").lower() if isinstance(ln, str) else "")
            for ln in shot["prompt"]
        ]
        positions = [(stripped.index(h), h) for h in required_headers if h in stripped]
        assert [h for _, h in sorted(positions)] == required_headers, shot["id"]

        # §7.1 #2: every picture on every shot.
        for pic in ("Picture 1", "Picture 2", "Picture 3"):
            assert pic in text, (shot["id"], pic)

        # §7.1 #3: consistent <Subject k> -> <Picture m>: bindings.
        assert "<Subject 1> -> <Picture 1>" in text, shot["id"]
        assert "<Subject 2> -> <Picture 2>" in text, shot["id"]

        # §7.1 #4: spatial pin grammar.
        assert "小猫妹妹 stays at left of frame" in text, shot["id"]
        assert "猫爷爷 stays at right of frame" in text, shot["id"]

        # §7.1 #6: no forbidden markers.
        assert "[Shot 2]" not in text, shot["id"]
        assert "At 00:" not in text, shot["id"]
        assert "@alias" not in text, shot["id"]

        # §7.1 #7: Setting == the authored English field.
        assert "Setting: %s" % storyboard["setting_line"] in text, shot["id"]
        for ln in text.splitlines():
            if ln.startswith("Setting:"):
                assert not any("\u4e00" <= c <= "\u9fff" for c in ln), ln

        # §7.1 #8: retention is build_fallback_shot_lines' fixed line.
        assert fixed_retention in text, shot["id"]

        # §7.1 #1 + #9 + #10: verbatim lines, tag from language, (Sn)==slot.
        for ln in sl["lines"]:
            assert "<d>%s %s</d>" % (tag, ln["text"]) in text, shot["id"]
        assert "<d>[Chinese]" not in text or tag == "[Chinese]", shot["id"]
        assert "<d>[English]" not in text or tag == "[English]", shot["id"]
        for name, info in storyboard["cast"].items():
            if any(ln["speaker"] == name for ln in sl["lines"]):
                assert "%s speaks as (S%d)" % (
                    name, info["character_slot"],
                ) in text, (shot["id"], name)

    # prompt_prefix came through verbatim (as lines).
    assert plan["prompt_prefix"] == storyboard["prompt_prefix"].split("\n")


# --------------------------------------------------------------------------- #
# §6.1 — en storyboard reproduces the golden structure
# --------------------------------------------------------------------------- #
class TestEnStoryboardGoldenStructure:
    def test_fixture_passes_validator(self, sb_modules, en_storyboard):
        cleaned = sb_modules["storyboard_lib"].validate_storyboard_payload(
            en_storyboard,
        )
        assert cleaned["language"] == "en"
        assert len(cleaned["shots"]) == 4
        assert sum(len(s["lines"]) for s in cleaned["shots"]) == 7

    def test_en_structure_assertions_via_v3(
        self, sb_modules, en_storyboard,
    ):
        builder = sb_modules["builder"]
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        plan_json, summary, board_kind, images, assets = node.generate_v3(
            en_storyboard, seed=611937399731, **_socket_kwargs(_bundles()),
        )
        assert board_kind == "dialogue"
        assert "mechanical · zero-LLM" in summary
        assert "板栗对话-EN" in summary
        assert images is not None and images.shape[0] == 3
        assert len(assets) == 3
        _structural_assertions(
            json.loads(plan_json), en_storyboard, sb_modules,
        )

    def test_en_lines_match_the_golden_plan_verbatim(
        self, sb_modules, en_storyboard,
    ):
        """The en storyboard's dialogue IS the golden plan's dialogue —
        harvested, not retyped."""
        golden = _load(GOLDEN_PLAN)
        golden_lines = []
        for shot in golden["shots"]:
            text = "\n".join(shot["prompt"])
            for chunk in text.split("<d>[English]")[1:]:
                golden_lines.append(chunk.split("</d>")[0].strip())
        sb_lines = [
            ln["text"] for sl in en_storyboard["shots"] for ln in sl["lines"]
        ]
        assert sb_lines == golden_lines

    def test_en_byte_reproducible(self, sb_modules, en_storyboard):
        builder = sb_modules["builder"]
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        kwargs = _socket_kwargs(_bundles())
        a = node.generate_v3(en_storyboard, seed=42, **kwargs)[0]
        b = node.generate_v3(en_storyboard, seed=42, **kwargs)[0]
        assert a == b

    def test_mechanical_word_never_reaches_the_plan(
        self, sb_modules, en_storyboard,
    ):
        builder = sb_modules["builder"]
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        plan_json = node.generate_v3(
            en_storyboard, seed=1, **_socket_kwargs(_bundles()),
        )[0]
        assert "mechanical" not in plan_json


# --------------------------------------------------------------------------- #
# §6.2 — V3 == Build byte-for-byte
# --------------------------------------------------------------------------- #
class TestV3MatchesBuild:
    @pytest.mark.parametrize(
        "fixture", ["storyboard_banli_en.json", "storyboard_banli_zh.json"],
    )
    def test_plan_json_identical(self, sb_modules, fixture):
        builder = sb_modules["builder"]
        storyboard = _load(FIXTURES / fixture)
        kwargs = _socket_kwargs(_bundles())
        build_out = builder.BuildPlanFromStoryboard().build(
            storyboard, seed=611937399731, **kwargs,
        )
        v3_out = builder.MiniMaxH3LoopPromptGeneratorV3().generate_v3(
            storyboard, seed=611937399731, **kwargs,
        )
        assert build_out[0] == v3_out[0]
        # summary + roster are shared too; only board_kind is V3's extra
        # middle wire.
        assert build_out[1] == v3_out[1]
        assert [b["name"] for b in build_out[3]] == [
            b["name"] for b in v3_out[4]
        ]

    def test_differing_seed_changes_only_the_seeds(self, sb_modules):
        builder = sb_modules["builder"]
        storyboard = _load(EN_STORYBOARD)
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        a = json.loads(
            node.generate_v3(
                storyboard, seed=100, **_socket_kwargs(_bundles()),
            )[0],
        )
        b = json.loads(
            node.generate_v3(
                storyboard, seed=200, **_socket_kwargs(_bundles()),
            )[0],
        )
        assert [s["prompt"] for s in a["shots"]] == [
            s["prompt"] for s in b["shots"]
        ]
        assert [s["seed"] for s in a["shots"]] == ["100", "101", "102", "103"]
        assert [s["seed"] for s in b["shots"]] == ["200", "201", "202", "203"]


# --------------------------------------------------------------------------- #
# §6.3 — zh storyboard renders
# --------------------------------------------------------------------------- #
class TestZhStoryboard:
    def test_zh_fixture_passes_validator(self, sb_modules, zh_storyboard):
        cleaned = sb_modules["storyboard_lib"].validate_storyboard_payload(
            zh_storyboard,
        )
        assert cleaned["language"] == "zh"

    def test_seven_lines_verbatim_with_chinese_tag(
        self, sb_modules, zh_storyboard,
    ):
        builder = sb_modules["builder"]
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        plan = json.loads(
            node.generate_v3(
                zh_storyboard, seed=611937399731,
                **_socket_kwargs(_bundles()),
            )[0],
        )
        sb_lines = [
            ln["text"] for sl in zh_storyboard["shots"] for ln in sl["lines"]
        ]
        assert len(sb_lines) == 7
        all_text = "\n".join(
            "\n".join(s["prompt"]) for s in plan["shots"]
        )
        for line in sb_lines:
            assert "<d>[Chinese] %s</d>" % line in all_text
        # No English tag leaks in from the shared English prefix.
        assert "<d>[English]" not in all_text

    def test_zh_structure_assertions(self, sb_modules, zh_storyboard):
        builder = sb_modules["builder"]
        plan_json = builder.MiniMaxH3LoopPromptGeneratorV3().generate_v3(
            zh_storyboard, seed=1, **_socket_kwargs(_bundles()),
        )[0]
        _structural_assertions(
            json.loads(plan_json), zh_storyboard, sb_modules,
        )

    def test_zh_dialogue_is_not_translated_at_assembly(
        self, sb_modules, zh_storyboard,
    ):
        """The contract that makes multi-language work: what the
        storyboard says is what the plan says. No detector, no
        translation step."""
        builder = sb_modules["builder"]
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        kwargs = _socket_kwargs(_bundles())
        first = node.generate_v3(zh_storyboard, seed=1, **kwargs)[0]
        second = node.generate_v3(zh_storyboard, seed=1, **kwargs)[0]
        assert first == second
        assert "爷爷，我一直有个心理阴影。" in first
        assert "但是你走过来，用脚一只一只踩死了，我记得内脏都出来了。" in first
        assert "傻孩子，那是板栗呀" in first


# --------------------------------------------------------------------------- #
# §6.4 — error text passes through V3 unchanged
# --------------------------------------------------------------------------- #
class TestErrorPassthrough:
    def _both_raises(self, sb_modules, storyboard, kwargs, match):
        builder = sb_modules["builder"]
        build_node = builder.BuildPlanFromStoryboard()
        v3_node = builder.MiniMaxH3LoopPromptGeneratorV3()
        with pytest.raises(ValueError) as build_exc:
            build_node.build(storyboard, seed=1, **kwargs)
        with pytest.raises(ValueError) as v3_exc:
            v3_node.generate_v3(storyboard, seed=1, **kwargs)
        assert match in str(build_exc.value)
        assert str(build_exc.value) == str(v3_exc.value)

    def test_gap_cast(self, sb_modules):
        storyboard = _load(EN_STORYBOARD)
        storyboard["cast"]["猫爷爷"] = {"character_slot": 3}
        self._both_raises(
            sb_modules, storyboard, _socket_kwargs(_bundles()), "contiguous",
        )

    def test_unwired_cast_slot(self, sb_modules):
        storyboard = _load(EN_STORYBOARD)
        kwargs = _socket_kwargs(_bundles())
        kwargs["character_2"] = None
        self._both_raises(
            sb_modules, storyboard, kwargs, "character_slot 2",
        )

    def test_empty_voice(self, sb_modules):
        storyboard = _load(EN_STORYBOARD)
        bundles = _bundles()
        bundles[1]["voice"] = ""
        self._both_raises(
            sb_modules, storyboard, _socket_kwargs(bundles), "empty voice",
        )

    def test_both_shots_and_lines(self, sb_modules):
        storyboard = _load(EN_STORYBOARD)
        storyboard["lines"] = [
            {"speaker": "小猫妹妹", "text": "重复字段"},
        ]
        self._both_raises(
            sb_modules, storyboard, _socket_kwargs(_bundles()), "both",
        )

    def test_no_scene_wired(self, sb_modules):
        storyboard = _load(EN_STORYBOARD)
        kwargs = _socket_kwargs(_bundles())
        kwargs["scene_1"] = None
        self._both_raises(
            sb_modules, storyboard, kwargs, "scene",
        )


# --------------------------------------------------------------------------- #
# §6.5 — registration + bilingual UI
# --------------------------------------------------------------------------- #
class TestRegistrationAndUI:
    def test_v3_registered_with_bilingual_name(self):
        plugin = _load_plugin()
        key = "MiniMaxH3LoopPromptGeneratorV3|Mie"
        assert key in plugin.NODE_CLASS_MAPPINGS
        display = plugin.NODE_DISPLAY_NAME_MAPPINGS[key]
        # 2026-10-11 ruling: 中文 · English, English after the separator.
        assert " · " in display, display
        zh_part, en_part = display.split(" · ", 1)
        assert any("\u4e00" <= c <= "\u9fff" for c in zh_part), display
        assert any(c.isascii() and c.isalpha() for c in en_part), display
        assert "V3" in en_part and "Zero LLM" in en_part, display

    def test_v3_class_in_the_llm_package(self):
        pkg = _load_llm_package()
        assert hasattr(pkg, "MiniMaxH3LoopPromptGeneratorV3")
        assert hasattr(pkg, "BuildPlanFromStoryboard")

    def test_v3_shape_matches_v2(self, sb_modules):
        """Swapping a V2 node for a V3 node must not break anything
        downstream of plan_json."""
        builder = sb_modules["builder"]
        v3 = builder.MiniMaxH3LoopPromptGeneratorV3
        assert v3.RETURN_NAMES == (
            "plan_json", "summary", "board_kind", "images", "assets",
        )
        assert v3.RETURN_TYPES == (
            "STRING", "STRING", "STRING", "IMAGE", "MIE_ASSET_ROSTER",
        )
        spec = v3.INPUT_TYPES()
        assert sorted(spec["required"]) == ["seed", "storyboard"]
        # Same socket names as the Build node (wire-compatible).
        assert sorted(spec["optional"]) == sorted(
            builder.BuildPlanFromStoryboard.INPUT_TYPES()["optional"],
        )

    def test_v3_has_no_llm_controls(self, sb_modules):
        """The zero-LLM red line: no connector socket, no sampling
        widgets, no prose knobs."""
        builder = sb_modules["builder"]
        spec = builder.MiniMaxH3LoopPromptGeneratorV3.INPUT_TYPES()
        all_keys = set(spec["required"]) | set(spec["optional"])
        for banned in (
            "llm_service_connector", "user_input", "temperature",
            "max_tokens", "timeout", "enhance_user_input",
            "camera_discipline", "pacing", "language", "scene_count",
            "category", "seed_mode",
        ):
            assert banned not in all_keys, banned

    def test_js_autogrow_table_covers_v3(self):
        js = (PROJECT_DIR / "js" / "assetLibraryUI.js").read_text(
            encoding="utf-8",
        )
        assert "MiniMaxH3LoopPromptGeneratorV3|Mie" in js
        assert "generatorSockets" in js


def _load_plugin():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_plugin_imports import load_plugin_module
    return load_plugin_module()


def _load_llm_package():
    return sys.modules.get("_mienodes_internal.nodes.llm") or __import__(
        "nodes.llm", fromlist=["MiniMaxH3LoopPromptGeneratorV3"],
    )


# --------------------------------------------------------------------------- #
# §6.6 — IS_CHANGED
# --------------------------------------------------------------------------- #
class TestIsChanged:
    def test_storyboard_content_change_moves_the_hash(self, sb_modules):
        builder = sb_modules["builder"]
        v3 = builder.MiniMaxH3LoopPromptGeneratorV3
        storyboard = _load(EN_STORYBOARD)
        kwargs = _socket_kwargs(_bundles())
        before = v3.IS_CHANGED(storyboard=storyboard, **kwargs)
        edited = json.loads(json.dumps(storyboard))
        edited["shots"][0]["lines"][0]["text"] = "Grandpa, something changed."
        after = v3.IS_CHANGED(storyboard=edited, **kwargs)
        assert before != after

    def test_asset_bundle_change_moves_the_hash(self, sb_modules):
        builder = sb_modules["builder"]
        v3 = builder.MiniMaxH3LoopPromptGeneratorV3
        storyboard = _load(EN_STORYBOARD)
        bundles = _bundles()
        before = v3.IS_CHANGED(
            storyboard=storyboard, **_socket_kwargs(bundles),
        )
        bundles[1]["voice"] = "elderly female, re-timed voice"
        after = v3.IS_CHANGED(
            storyboard=storyboard, **_socket_kwargs(bundles),
        )
        assert before != after

    def test_identical_inputs_are_stable(self, sb_modules):
        builder = sb_modules["builder"]
        v3 = builder.MiniMaxH3LoopPromptGeneratorV3
        storyboard = _load(EN_STORYBOARD)
        kwargs = _socket_kwargs(_bundles())
        assert v3.IS_CHANGED(storyboard=storyboard, **kwargs) == \
            v3.IS_CHANGED(storyboard=storyboard, **kwargs)

    def test_wiring_an_extra_character_moves_the_hash(self, sb_modules):
        """A third character wired into character_3 changes the roster the
        assembler would emit, so the cache key must move even though the
        storyboard file is untouched."""
        builder = sb_modules["builder"]
        v3 = builder.MiniMaxH3LoopPromptGeneratorV3
        storyboard = _load(EN_STORYBOARD)
        kwargs = _socket_kwargs(_bundles())
        before = v3.IS_CHANGED(storyboard=storyboard, **kwargs)
        kwargs["character_3"] = dict(
            _bundles()[1], name="新角色", voice="young female, re-timed",
        )
        after = v3.IS_CHANGED(storyboard=storyboard, **kwargs)
        assert before != after
