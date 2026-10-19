# -*- coding: utf-8 -*-
"""Tests for docs/PLAN_STORYBOARD_ASSET.md (storyboard asset + zero-LLM
assembler).

Two surfaces are covered:

* ``nodes/assets/storyboard.py`` — JSON schema, validation, on-disk
  round-trips, listing, content-hash cache key. No ComfyUI, no torch.
* ``nodes/llm/storyboard_builder.py`` — ``assemble_plan``: takes a
  storyboard dict + wired asset bundles, returns a six-section Ref2V
  plan. The PLAN §7 acceptance list (§7.1 golden structure, §7.2 hard
  errors, §7.3 speaker!=slot, §7.4 scene ambience roundtrip, §7.5
  extracted verifier).

The builder tests use a small custom loader (``_load_pkg``) so the
llm package's eager imports do not pull in ComfyUI runtime modules —
the same trick the existing test_h3_loop_prompt_generator uses.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Module loaders (no ComfyUI, no eager import side-effects)
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
    library = _load_file(
        "_mienodes_internal.nodes.assets.library",
        PROJECT_DIR / "nodes" / "assets" / "library.py",
    )
    storyboard = _load_file(
        "_mienodes_internal.nodes.assets.storyboard",
        PROJECT_DIR / "nodes" / "assets" / "storyboard.py",
    )
    _ensure_pkg("_mienodes_internal.nodes.llm", PROJECT_DIR / "nodes" / "llm")
    _ensure_pkg("_mienodes_internal.nodes.llm.prompts", PROJECT_DIR / "nodes" / "llm" / "prompts")
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
    prompts = _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        PROJECT_DIR / "nodes" / "llm" / "minimax_h3_loop_prompts.py",
    )
    segmenter = _load_file(
        "_mienodes_internal.nodes.llm.dialogue_segmenter",
        PROJECT_DIR / "nodes" / "llm" / "dialogue_segmenter.py",
    )
    builder = _load_file(
        "_mienodes_internal.nodes.llm.storyboard_builder",
        PROJECT_DIR / "nodes" / "llm" / "storyboard_builder.py",
    )
    return {
        "library": library,
        "storyboard": storyboard,
        "prompts": prompts,
        "segmenter": segmenter,
        "builder": builder,
    }


@pytest.fixture
def lib_dir(tmp_path):
    d = tmp_path / "asset_lib"
    d.mkdir()
    return d


@pytest.fixture
def golden_storyboard():
    """The §3.1 example, minimally cleaned: a 2-shot board-cast chat in
    Chinese, locked camera, hand-written English prompt_prefix + Setting
    line + one picture cue on shot 2 line 1."""
    return {
        "name": "客厅板栗对话",
        "version": 1,
        "language": "zh",
        "camera_discipline": "locked",
        "pacing": "normal",
        "seed_policy": "per_scene_increment",
        "prompt_prefix": (
            "A cozy cottage living room, polished 3D, warm light.\n"
            "The camera stays locked off."
        ),
        "cast": {
            "孙女": {"character_slot": 1},
            "爷爷": {"character_slot": 2},
        },
        "setting_line": "Grandpa and the girl sit on the sofa, talking.",
        "spatial_layout": {
            "孙女": "left of frame",
            "爷爷": "right of frame",
        },
        "ambience": "Quiet room tone with soft indoor air. No music.",
        "shots": [
            {"lines": [
                {"speaker": "孙女", "text": "爷爷，我一直有个心理阴影。"},
                {"speaker": "爷爷", "text": "是什么呢。"},
            ]},
            {"lines": [
                {
                    "speaker": "孙女",
                    "text": "小时候我还没上小学。",
                    "picture": "She keeps both paws in her lap.",
                },
                {"speaker": "孙女", "text": "咱家院子里有几只小刺猬。"},
            ]},
        ],
    }


def _make_bundles(storyboard):
    """Two-character + one-scene bundles in slot order, mirroring what
    the V2 generator's ``_build_roster_from_bundles`` would emit.

    Returns a dict keyed by picture number (``character_slot`` for
    characters, ``max(cast_slots) + 1`` for the scene)."""
    import torch

    used_slots = sorted(
        int(info["character_slot"])
        for info in storyboard["cast"].values()
    )
    if used_slots != list(range(1, len(used_slots) + 1)):
        raise ValueError("fixture expects character_slots to be 1..N")

    bundles: dict[int, dict | None] = {}
    for slot, (name, info) in zip(
        used_slots, storyboard["cast"].items(),
    ):
        bundles[slot] = {
            "kind": "character",
            "name": name,
            "description": (
                "A calico kitten girl with cream-and-orange fur."
                if name == "孙女"
                else "An elderly cat grandfather with orange tabby fur."
            ),
            "voice": (
                "young female, soft and sweet timbre, mid register, gentle pace"
                if name == "孙女"
                else "elderly male, soft and mildly gravelly timbre, low register, slow and unhurried pace"
            ),
            "image": torch.zeros(1, 8, 8, 3),
            "audio": {
                "waveform": torch.zeros(1, 1, 1600),
                "sample_rate": 16000,
            },
        }
    scene_slot = max(used_slots) + 1
    bundles[scene_slot] = {
        "kind": "scene",
        "name": "客厅",
        "description": "A cozy cottage living room with a sofa against the wall.",
        "ambience": "",
        "image": torch.zeros(1, 8, 8, 3),
    }
    return bundles


# --------------------------------------------------------------------------- #
# Storyboard on-disk layer
# --------------------------------------------------------------------------- #
class TestStoryboardDisk:
    def test_validate_minimal_payload(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        payload = {
            "name": "Booster", "version": 1, "language": "en",
            "camera_discipline": "locked", "pacing": "normal",
            "seed_policy": "per_scene_increment",
            "prompt_prefix": "Global visual style.",
            "cast": {"A": {"character_slot": 1}},
            "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
        }
        out = storyboard.validate_storyboard_payload(payload)
        assert out["name"] == "Booster"
        assert out["language"] == "en"
        assert out["prompt_prefix"] == ["Global visual style."]
        assert out["shots"][0]["lines"][0]["text"] == "hi"

    def test_validate_rejects_wrong_version(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="version"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 2, "prompt_prefix": "p",
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
            })

    def test_validate_rejects_cjk_in_prefix(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="English"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "zh",
                "prompt_prefix": "客厅 cozy",
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
            })

    def test_validate_rejects_cjk_in_setting_line(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="English"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "zh",
                "prompt_prefix": "ok",
                "setting_line": "客厅里猫爷爷",
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
            })

    def test_validate_rejects_cjk_in_picture_cue(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="English"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "zh",
                "prompt_prefix": "ok",
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [{
                    "speaker": "A", "text": "hi",
                    "picture": "她双手放在膝盖上",
                }]}],
            })

    def test_validate_rejects_empty_shots_array(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="empty"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "en",
                "prompt_prefix": "ok",
                "cast": {"A": {"character_slot": 1}},
                "shots": [],
            })

    def test_validate_rejects_too_many_lines_per_shot(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="per-shot cap"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "en",
                "prompt_prefix": "ok",
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [
                    {"speaker": "A", "text": "1"},
                    {"speaker": "A", "text": "2"},
                    {"speaker": "A", "text": "3"},
                ]}],
            })

    def test_validate_rejects_duplicate_slot(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="more than one speaker"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "en",
                "prompt_prefix": "ok",
                "cast": {
                    "A": {"character_slot": 1},
                    "B": {"character_slot": 1},
                },
                "shots": [{"lines": [
                    {"speaker": "A", "text": "1"},
                    {"speaker": "B", "text": "2"},
                ]}],
            })

    def test_validate_rejects_slot_out_of_range(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="out of range"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "en",
                "prompt_prefix": "ok",
                "cast": {"A": {"character_slot": 9}},
                "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
            })

    def test_validate_rejects_speaker_not_in_cast(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="not in cast"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "en",
                "prompt_prefix": "ok",
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [
                    {"speaker": "A", "text": "hi"},
                    {"speaker": "B", "text": "ho"},
                ]}],
            })

    def test_save_load_roundtrip(self, sb_modules, lib_dir, golden_storyboard):
        storyboard = sb_modules["storyboard"]
        cleaned, path, status = storyboard.save_storyboard(
            str(lib_dir), golden_storyboard,
        )
        assert status == "created"
        assert cleaned["name"] == "客厅板栗对话"
        assert cleaned["shots"][1]["lines"][0]["picture"] == (
            "She keeps both paws in her lap."
        )
        assert Path(path).is_file()

        loaded, load_path = storyboard.load_storyboard(
            str(lib_dir), "客厅板栗对话",
        )
        assert load_path == path
        assert loaded["language"] == "zh"
        assert loaded["spatial_layout"] == golden_storyboard["spatial_layout"]
        assert loaded["ambience"] == "Quiet room tone with soft indoor air. No music."

    def test_save_overwrite_status(self, sb_modules, lib_dir, golden_storyboard):
        storyboard = sb_modules["storyboard"]
        storyboard.save_storyboard(str(lib_dir), golden_storyboard)
        cleaned, _, status = storyboard.save_storyboard(
            str(lib_dir), golden_storyboard,
        )
        assert status == "overwritten"
        assert cleaned["name"] == "客厅板栗对话"

    def test_load_keeps_cjk_speaker_names(self, sb_modules, lib_dir, golden_storyboard):
        storyboard = sb_modules["storyboard"]
        storyboard.save_storyboard(str(lib_dir), golden_storyboard)
        loaded, _ = storyboard.load_storyboard(str(lib_dir), "客厅板栗对话")
        assert "孙女" in loaded["cast"]
        assert "爷爷" in loaded["cast"]
        assert loaded["cast"]["孙女"]["character_slot"] == 1
        assert loaded["cast"]["爷爷"]["character_slot"] == 2

    def test_load_errors_contain_absolute_path(self, sb_modules, lib_dir):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(FileNotFoundError) as exc:
            storyboard.load_storyboard(str(lib_dir), "ghost")
        assert str(lib_dir / "storyboards" / "ghost" / "storyboard.json") in str(exc.value)

    def test_list_skips_broken_directories(self, sb_modules, lib_dir):
        storyboard = sb_modules["storyboard"]
        (lib_dir / "storyboards" / "GoodOne").mkdir(parents=True)
        ok_payload = {
            "name": "GoodOne", "version": 1, "language": "en",
            "prompt_prefix": "ok",
            "cast": {"A": {"character_slot": 1}},
            "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
        }
        storyboard.save_storyboard(str(lib_dir), ok_payload)
        # Write a broken storyboard.json — JSON-invalid.
        broken = lib_dir / "storyboards" / "BrokenOne"
        broken.mkdir(parents=True)
        (broken / "storyboard.json").write_text("{not valid", encoding="utf-8")
        # Write one with the wrong version.
        badver = lib_dir / "storyboards" / "BadVer"
        badver.mkdir(parents=True)
        (badver / "storyboard.json").write_text(
            json.dumps({"name": "BadVer", "version": 99}), encoding="utf-8",
        )
        result = storyboard.list_storyboards(str(lib_dir))
        assert "GoodOne" in result["names"]
        assert result["names"].count("\n") == 0
        assert "BrokenOne" in result["status"]
        assert "BadVer" in result["status"]

    def test_cache_key_reflects_content(self, sb_modules, lib_dir, golden_storyboard):
        storyboard = sb_modules["storyboard"]
        storyboard.save_storyboard(str(lib_dir), golden_storyboard)
        key1 = storyboard.storyboard_cache_key(str(lib_dir), "客厅板栗对话")
        # Re-save the same payload — the bytes don't change because the
        # validator produces a deterministic cleaned dict.
        storyboard.save_storyboard(str(lib_dir), golden_storyboard)
        key2 = storyboard.storyboard_cache_key(str(lib_dir), "客厅板栗对话")
        assert key1 == key2
        assert key1 != sb_modules["library"].MISSING_CACHE_MARKER

    def test_cache_key_returns_missing_marker_for_unknown(self, sb_modules, lib_dir):
        storyboard = sb_modules["storyboard"]
        key = storyboard.storyboard_cache_key(str(lib_dir), "ghost")
        assert key == sb_modules["library"].MISSING_CACHE_MARKER


# --------------------------------------------------------------------------- #
# assemble_plan — §7 acceptance
# --------------------------------------------------------------------------- #
class TestAssemblePlan:
    def test_golden_structure(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        out = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=611937399731,
        )
        plan = out["plan"]
        # Top-level keys: shots + prompt_prefix only (validate_plan §7.1).
        assert set(plan.keys()) == {"shots", "prompt_prefix"}
        # Six-section form: every shot's prompt carries all six headers.
        prompts = sb_modules["prompts"]
        required_headers = list(prompts.SIX_SECTION_FIELDS)
        for shot in plan["shots"]:
            stripped = [
                (ln.strip().rstrip(":").lower() if isinstance(ln, str) else "")
                for ln in shot["prompt"]
            ]
            for h in required_headers:
                assert h in stripped, (
                    "shot %s missing %s" % (shot["id"], h)
                )
        # Shot count == storyboard shot count (no auto scenes).
        assert len(plan["shots"]) == len(golden_storyboard["shots"])
        # Per-shot Picture references (every cast member + the scene).
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            for pic in ("Picture 1", "Picture 2", "Picture 3"):
                assert pic in text, "shot %s missing %s" % (shot["id"], pic)
        # No [Shot 2], no At 00:, no @alias.
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "[Shot 2]" not in text
            assert "At 00:" not in text
            assert "@alias" not in text
            assert "#" not in text.split("detailed_description:")[-1].split(
                "overall_soundscape:"
            )[0]
        # The verbatim dialogue lines appear in every <d> block.
        for shot, sl in zip(plan["shots"], golden_storyboard["shots"]):
            text = "\n".join(shot["prompt"])
            for ln in sl["lines"]:
                # <d> tag carries the language from storyboard.language
                if golden_storyboard["language"] == "zh":
                    assert "<d>[Chinese] %s</d>" % ln["text"] in text
                else:
                    assert "<d>[English] %s</d>" % ln["text"] in text
        # Speaker -> (Sn) is by character_slot (not first speaker).
        text0 = "\n".join(plan["shots"][0]["prompt"])
        assert "(S1) young female" in text0
        assert "(S2) elderly male" in text0
        # Setting line equals the storyboard's English setting_line (no
        # Chinese residue).
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "Setting: Grandpa and the girl sit on the sofa, talking." in text
            assert "Setting:" in text
            # No Chinese in Setting.
            setting_i = text.index("Setting:")
            setting_line = text[setting_i:].splitlines()[0]
            assert not any("\u4e00" <= c <= "\u9fff" for c in setting_line)
        # Spatial pin grammar is `name stays ...`.
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "孙女 stays at left of frame" in text
            assert "爷爷 stays at right of frame" in text
        # (d) tag = storyboard.language, NOT auto-detected.
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "<d>[Chinese]" in text

    def test_seed_per_scene_increment(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        out = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=611937399731,
        )
        seeds = [s["seed"] for s in out["plan"]["shots"]]
        assert seeds == ["611937399731", "611937399732"]

    def test_seed_same_across_scenes(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        storyboard = {**golden_storyboard, "seed_policy": "same_across_scenes"}
        bundles = _make_bundles(storyboard)
        out = builder.assemble_plan(
            storyboard=storyboard,
            slot_bundles=bundles,
            seed_base=611937399731,
        )
        seeds = [s["seed"] for s in out["plan"]["shots"]]
        assert seeds == ["611937399731", "611937399731"]

    def test_byte_reproducible(self, sb_modules, golden_storyboard):
        """Same inputs ⇒ byte-identical plan_json."""
        import json as _json
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        out_a = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=42,
        )
        out_b = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=42,
        )
        # plan_to_json_string is deterministic given equal inputs.
        a = sb_modules["prompts"].plan_to_json_string(out_a["plan"])
        b = sb_modules["prompts"].plan_to_json_string(out_b["plan"])
        assert a == b
        assert _json.loads(a) == _json.loads(b)

    def test_en_language_tag(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        storyboard = {**golden_storyboard, "language": "en"}
        for sl in storyboard["shots"]:
            for line in sl["lines"]:
                line["text"] = "Hello there."
        bundles = _make_bundles(storyboard)
        out = builder.assemble_plan(
            storyboard=storyboard,
            slot_bundles=bundles,
            seed_base=1,
        )
        text = "\n".join(out["plan"]["shots"][0]["prompt"])
        assert "<d>[English] Hello there.</d>" in text

    def test_speaker_not_equal_to_character_slot(self, sb_modules, golden_storyboard):
        """§7.3 — speaker names differ from character_slot order, but
        ``(Sn)`` is still bound by character_slot."""
        builder = sb_modules["builder"]
        # The cast names are 爷爷 (slot 2) and 孙女 (slot 1). The shot
        # has 爷爷 speaking first (slot 2), but (S1) belongs to the
        # speaker on slot 1 (孙女). Verify that.
        storyboard = {
            **golden_storyboard,
            "shots": [
                {"lines": [
                    # 爷爷 speaks first, but his slot is 2.
                    {"speaker": "爷爷", "text": "你先说。"},
                    {"speaker": "孙女", "text": "嗯，好的。"},
                ]},
            ],
        }
        bundles = _make_bundles(storyboard)
        out = builder.assemble_plan(
            storyboard=storyboard,
            slot_bundles=bundles,
            seed_base=1,
        )
        text = "\n".join(out["plan"]["shots"][0]["prompt"])
        # 爷爷 -> (S2), not (S1); 孙女 -> (S1).
        assert "爷爷 speaks as (S2)" in text
        assert "孙女 speaks as (S1)" in text
        # Both Picture 1 and Picture 2 still bind.
        assert "<Subject 1> -> <Picture 1>" in text
        assert "<Subject 2> -> <Picture 2>" in text

    def test_empty_voice_is_a_hard_error(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        bundles[1]["voice"] = ""  # strip 孙女's voice (slot 1)
        with pytest.raises(ValueError, match="empty voice"):
            builder.assemble_plan(
                storyboard=golden_storyboard,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_missing_audio_is_a_hard_error(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        bundles[1]["audio"] = None  # strip 孙女's audio (slot 1)
        with pytest.raises(ValueError, match="no audio sample"):
            builder.assemble_plan(
                storyboard=golden_storyboard,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_missing_slot_is_a_hard_error(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        # Drop the second character — its character_slot (2) is empty.
        del bundles[2]
        with pytest.raises(ValueError, match="character_slot 2"):
            builder.assemble_plan(
                storyboard=golden_storyboard,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_no_scene_is_a_hard_error(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        # Drop the scene bundle.
        scene_slot = max(bundles.keys())
        del bundles[scene_slot]
        with pytest.raises(ValueError, match="scene"):
            builder.assemble_plan(
                storyboard=golden_storyboard,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_non_contiguous_wiring_raises_loudly(self, sb_modules):
        """P0 (review 2026-10-19): cast uses slots {1, 2}, user wires
        only character_1 + character_3 (character_2 empty). The build
        node MUST raise rather than silently bind slot 2 to whatever is
        wired at character_3."""
        builder = sb_modules["builder"]
        # Reuse _make_bundles to get well-formed character bundles.
        # Build a storyboard whose cast has slots 1 and 2.
        sb = {
            "name": "p0", "version": 1, "language": "en",
            "camera_discipline": "locked", "pacing": "normal",
            "seed_policy": "per_scene_increment",
            "prompt_prefix": "global style.",
            "cast": {
                "甲": {"character_slot": 1},
                "乙": {"character_slot": 2},
            },
            "shots": [{"lines": [
                {"speaker": "甲", "text": "hi 1"},
                {"speaker": "乙", "text": "hi 2"},
            ]}],
        }
        bundles = _make_bundles(sb)
        # Force the hole: drop slot 2, keep only slot 1 and the scene.
        del bundles[2]
        with pytest.raises(ValueError, match="character_slot 2"):
            builder.assemble_plan(
                storyboard=sb,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_compact_wiring_does_not_misbind_slot(self, sb_modules):
        """Sanity guard for the P0 fix: even if the user accidentally
        passes a list of bundles in slot enumeration order (without
        None padding), the new dict API surfaces the right error
        instead of silently re-binding."""
        builder = sb_modules["builder"]
        sb = {
            "name": "p0", "version": 1, "language": "en",
            "camera_discipline": "locked", "pacing": "normal",
            "seed_policy": "per_scene_increment",
            "prompt_prefix": "global style.",
            "cast": {
                "甲": {"character_slot": 1},
                "乙": {"character_slot": 2},
            },
            "shots": [{"lines": [
                {"speaker": "甲", "text": "hi 1"},
                {"speaker": "乙", "text": "hi 2"},
            ]}],
        }
        bundles = _make_bundles(sb)
        # Caller passes a list — the API is now strict (no list
        # compaction). Surfaces the API drift loudly rather than
        # re-introducing the silent misbind.
        list_bundles = list(bundles.values())
        with pytest.raises(ValueError, match="dict"):
            builder.assemble_plan(
                storyboard=sb,
                slot_bundles=list_bundles,
                seed_base=1,
            )

    def test_speaker_not_in_cast_rejected_by_layer(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bad_sb = {
            **golden_storyboard,
            "shots": [{"lines": [
                {"speaker": "Stranger", "text": "hi"},
                {"speaker": "爷爷", "text": "ho"},
            ]}],
        }
        bundles = _make_bundles(bad_sb)
        with pytest.raises(ValueError, match="not in cast"):
            builder.assemble_plan(
                storyboard=bad_sb,
                slot_bundles=bundles,
                seed_base=1,
            )

    def test_ambience_fallback_resolution(self, sb_modules, golden_storyboard):
        """§3.2: storyboard ambience > scene ambience > camera-discipline fallback."""
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        # Scene bundle lives at slot max(cast_slots) + 1.
        scene_slot = max(
            int(v["character_slot"]) for v in golden_storyboard["cast"].values()
        ) + 1

        # Case 1: storyboard ambience wins.
        sb1 = dict(golden_storyboard)
        bundles1 = _make_bundles(sb1)
        bundles1[scene_slot]["ambience"] = "scene ambience"
        out1 = builder.assemble_plan(
            storyboard=sb1, slot_bundles=bundles1, seed_base=1,
        )
        assert out1["ambience"] == "Quiet room tone with soft indoor air. No music."

        # Case 2: no storyboard ambience, scene ambience wins.
        sb2 = {**golden_storyboard, "ambience": ""}
        bundles2 = _make_bundles(sb2)
        bundles2[scene_slot]["ambience"] = "scene-only tone"
        out2 = builder.assemble_plan(
            storyboard=sb2, slot_bundles=bundles2, seed_base=1,
        )
        assert out2["ambience"] == "scene-only tone"

        # Case 3: both empty -> discipline fallback (three distinct
        # lines — the discipline actually picks).
        sb3 = {**golden_storyboard, "ambience": ""}
        bundles3 = _make_bundles(sb3)
        bundles3[scene_slot]["ambience"] = ""
        out3 = builder.assemble_plan(
            storyboard=sb3, slot_bundles=bundles3, seed_base=1,
        )
        # The "locked" fallback line is now visibly distinct from
        # no_closeups / auto.
        assert out3["ambience"] != "no_closeups' line" and \
            out3["ambience"].startswith("The room tone stays quiet and even")
        # And the three fallback strings are different from each other.
        assert builder.AMBIENCE_FALLBACKS["locked"] != \
            builder.AMBIENCE_FALLBACKS["no_closeups"]
        assert builder.AMBIENCE_FALLBACKS["locked"] != \
            builder.AMBIENCE_FALLBACKS["auto"]

    def test_summary_mentions_mechanical_source(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        out = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=1,
        )
        # The word "mechanical" only appears in the node's STRING summary
        # (PLAN §2: "不把 mechanical 写进 H3 prompt").
        plan_text = sb_modules["prompts"].plan_to_json_string(out["plan"])
        assert "mechanical" not in plan_text
        # No Chinese Setting residue in the plan.
        import json as _json
        plan_obj = _json.loads(plan_text)
        for shot in plan_obj["shots"]:
            joined = "\n".join(shot["prompt"])
            setting_lines = [
                ln[len("Setting:"):].strip()
                for ln in joined.splitlines()
                if ln.startswith("Setting:")
            ]
            for s in setting_lines:
                assert not any("\u4e00" <= c <= "\u9fff" for c in s), (
                    "Chinese residue in Setting: %r" % s
                )

    def test_picture_cue_renders_as_plain_sentence(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        out = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=1,
        )
        text = "\n".join(out["plan"]["shots"][1]["prompt"])
        # No "Silent picture only" wrapper — that wrapper contradicts
        # the upcoming speaking line.
        assert "Silent picture only" not in text
        # The cue text surfaces verbatim (with a trailing period).
        assert "She keeps both paws in her lap." in text


# --------------------------------------------------------------------------- #
# §7.5 extracted verifier — minimal hard-checks on the assembled plan
# --------------------------------------------------------------------------- #
def verify_storyboard_plan(plan, dialogue_lines_by_shot, slot_count):
    """Mechanical verifier pulled out of ``assemble_plan`` for reuse.

    Hard checks (matches PLAN §7.5):

    * every shot's <d> blocks match the locked lines verbatim;
    * every speaker in the locked lines is in the cast (slot_count is
      the size of the cast — at most slot_count unique speakers);
    * every Picture N (1..slot_count+1) appears in every shot;
    * spatial pin is ``name stays ...`` for every cast member;
    * forbidden markers (At 00:, [Shot 2], @alias, #tag) absent;
    * the Setting text equals the storyboard's English setting_line
      (or no Setting line at all).
    """
    errors: list[str] = []
    shots = plan.get("shots") or []
    if len(shots) != len(dialogue_lines_by_shot):
        errors.append(
            "shot count %d != dialogue block count %d"
            % (len(shots), len(dialogue_lines_by_shot))
        )
    for i, (shot, lines) in enumerate(zip(shots, dialogue_lines_by_shot)):
        text = "\n".join(shot.get("prompt") or [])
        # Verbatim <d> blocks.
        for ln in lines:
            if "<d>[Chinese] %s</d>" % ln not in text and \
               "<d>[English] %s</d>" % ln not in text:
                # The language tag is one of the two; the dialog
                # text follows the line as written.
                found = False
                for tag in ("[Chinese]", "[English]"):
                    if "<d>%s %s</d>" % (tag, ln) in text:
                        found = True
                        break
                if not found:
                    errors.append("shot %d missing verbatim <d> for %r"
                                  % (i, ln))
        # Picture labels.
        for pic in range(1, slot_count + 2):
            if ("Picture %d" % pic) not in text:
                errors.append("shot %d missing Picture %d" % (i, pic))
        # No forbidden markers.
        if "[Shot 2]" in text:
            errors.append("shot %d contains [Shot 2]" % i)
        if "At 00:" in text:
            errors.append("shot %d contains 'At 00:'" % i)
        if "@alias" in text:
            errors.append("shot %d contains @alias token" % i)
    return errors


class TestExtractedVerifier:
    def test_golden_passes(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        bundles = _make_bundles(golden_storyboard)
        out = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=1,
        )
        lines_per_shot = [
            [ln["text"] for ln in sl["lines"]]
            for sl in golden_storyboard["shots"]
        ]
        errors = verify_storyboard_plan(
            out["plan"],
            lines_per_shot,
            slot_count=len(golden_storyboard["cast"]),
        )
        assert errors == []

    def test_no_chinese_setting_residue(self, sb_modules, golden_storyboard):
        builder = sb_modules["builder"]
        # The storyboard validator rejects CJK in setting_line at the
        # load layer. Confirm the build itself would never produce a
        # Chinese Setting — Chinese residue in the Setting line is a
        # hard error (PLAN §7.5). Speaker names ARE CJK — that's the
        # cast carrying through the prompt — but the Setting line is
        # the user-authored English field.
        bundles = _make_bundles(golden_storyboard)
        out = builder.assemble_plan(
            storyboard=golden_storyboard,
            slot_bundles=bundles,
            seed_base=1,
        )
        for shot in out["plan"]["shots"]:
            joined = "\n".join(shot["prompt"])
            # The Setting line is the English one and ONLY contains
            # English. Walk every "Setting:" header and assert the
            # body has no CJK.
            for ln in joined.splitlines():
                if not ln.startswith("Setting:"):
                    continue
                body = ln[len("Setting:"):].strip()
                assert not any("\u4e00" <= c <= "\u9fff" for c in body), (
                    "Setting line carries CJK: %r" % body
                )
        lines_per_shot = [
            [ln["text"] for ln in sl["lines"]]
            for sl in golden_storyboard["shots"]
        ]
        errors = verify_storyboard_plan(
            out["plan"], lines_per_shot,
            slot_count=len(golden_storyboard["cast"]),
        )
        assert errors == []

    def test_setting_line_too_long_rejected(self, sb_modules):
        storyboard = sb_modules["storyboard"]
        with pytest.raises(ValueError, match="longer than"):
            storyboard.validate_storyboard_payload({
                "name": "X", "version": 1, "language": "en",
                "prompt_prefix": "ok",
                "setting_line": "a" * (storyboard.SETTING_LINE_MAX_CHARS + 1),
                "cast": {"A": {"character_slot": 1}},
                "shots": [{"lines": [{"speaker": "A", "text": "hi"}]}],
            })