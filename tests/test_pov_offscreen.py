# -*- coding: utf-8 -*-
"""First-person POV voiceover + per-shot presence
(docs/PLAN_POV_OFFSCREEN.md §6).

Covers all seven acceptance items:

  §6.1  three-form line templates, pinned verbatim
  §6.2  byte-stability red line — non-POV plans unchanged (VETO item)
  §6.3  the POV fixture: mixed / pure-listen / absent shots
  §6.4  validator: offscreen in layout / absent, all-empty shot, bad lang
  §6.5  neutralize_offscreen_claims conditionalisation
  §6.6  V2 （画外） suffix parsing, incl. the mood-parenthetical trap
  §6.7  (Sn) / audio slot alignment is unaffected by presence
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_DIR / "tests" / "fixtures"
POV_FIXTURE = FIXTURES / "storyboard_pov_mixed.json"


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
def mods():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.assets", PROJECT_DIR / "nodes" / "assets")
    _load_file(
        "_mienodes_internal.nodes.assets.library",
        PROJECT_DIR / "nodes" / "assets" / "library.py",
    )
    storyboard = _load_file(
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
        "prompts": prompts,
        "segmenter": segmenter,
        "builder": builder,
        "storyboard": storyboard,
    }


def _char(name, description, voice):
    import torch
    return {
        "kind": "character",
        "name": name,
        "description": description,
        "voice": voice,
        "image": torch.zeros(1, 8, 8, 3),
        "audio": {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000},
    }


@pytest.fixture(scope="module")
def pov_plan(mods):
    doc = json.loads(POV_FIXTURE.read_text(encoding="utf-8"))
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    out = node.generate_v3(
        doc,
        seed=611937399731,
        character_1=_char(
            "叙述者", "An unseen narrator with no on-screen body.",
            "adult female, calm narrative timbre, even pace",
        ),
        character_2=_char(
            "小猫妹妹", "A young female calico kitten in a cream dress.",
            "young female, soft and sweet timbre, gentle pace",
        ),
        character_3=_char(
            "猫爷爷", "An elderly orange tabby cat in a green cardigan.",
            "elderly male, soft gravelly timbre, low register",
        ),
        scene_1={
            "kind": "scene", "name": "客厅",
            "description": "A cozy living room with a table.",
            "ambience": "", "image": __import__("torch").zeros(1, 8, 8, 3),
        },
    )
    return doc, out


# --------------------------------------------------------------------------- #
# §6.1 — three-form line templates, pinned verbatim
# --------------------------------------------------------------------------- #
class TestThreeFormTemplates:
    def test_onscreen_form_without_pov_board_is_unchanged(self, mods):
        p = mods["prompts"]
        assert p.on_screen_speech_line(
            "甲", "S1", "young female, soft timbre", "hi", "[Chinese]",
        ) == "甲 speaks as (S1) young female, soft timbre <d>[Chinese] hi</d>"

    def test_onscreen_form_on_pov_board_appends_lip_sentence(self, mods):
        p = mods["prompts"]
        assert p.on_screen_speech_line(
            "甲", "S1", "young female, soft timbre", "hi", "[Chinese]",
            pov_board=True,
        ) == (
            "甲 speaks as (S1) young female, soft timbre "
            "<d>[Chinese] hi</d> 甲's lips move naturally with the line."
        )

    def test_offscreen_form_has_all_three_parts(self, mods):
        p = mods["prompts"]
        got = p.offscreen_speech_line(
            "S1", "adult female, calm timbre", "hello", "[English]",
            listener="甲", reaction="lowers her ears and glances down.",
        )
        assert got == (
            "An off-screen voice (S1), adult female, calm timbre, says in "
            "an off-screen voiceover: <d>[English] hello</d> while 甲's lips "
            "remain completely closed; lowers her ears and glances down."
        )

    def test_offscreen_default_reaction_is_degendered(self, mods):
        """No pronoun inference — the name is repeated (§8 decision)."""
        p = mods["prompts"]
        got = p.offscreen_speech_line(
            "S1", "adult female, calm timbre", "hello", "[English]",
            listener="甲",
        )
        assert got.endswith("while 甲's lips remain completely closed; "
                            "甲 only listens with a slight nod.")
        assert " she " not in got and " he " not in got

    def test_offscreen_form_without_reaction_or_listener(self, mods):
        p = mods["prompts"]
        got = p.offscreen_speech_line(
            "S1", "adult female, calm timbre", "hello", "[Chinese]",
        )
        assert got == (
            "An off-screen voice (S1), adult female, calm timbre, says in "
            "an off-screen voiceover: <d>[Chinese] hello</d>"
        )

    def test_absent_declarations(self, mods):
        p = mods["prompts"]
        assert p.absent_declarations(["猫爷爷"]) == [
            "猫爷爷 is not in the frame in this shot."
        ]

    def test_pov_shot1_agrees_in_number(self, mods):
        p = mods["prompts"]
        assert "甲 faces the camera" in p.pov_shot1_coverage(["甲"])
        assert "甲 and 乙 face the camera" in p.pov_shot1_coverage(["甲", "乙"])


# --------------------------------------------------------------------------- #
# §6.2 — byte-stability red line (VETO)
# --------------------------------------------------------------------------- #
class TestByteStabilityRedLine:
    def test_banli_fixtures_assemble_to_the_recorded_bytes(self, mods):
        """The two shipped storyboards were captured BEFORE this feature
        landed. If presence plumbing leaks into a non-POV board, these
        digests move and the veto fires."""
        import torch
        builder = mods["builder"]
        node = builder.MiniMaxH3LoopPromptGeneratorV3()
        bundles = {
            "character_1": _char(
                "小猫妹妹",
                "A young female calico kitten with fluffy cream, orange and "
                "charcoal-black fur, large dark eyes and a pink nose. She wears "
                "a cream short-sleeved floral dress with a Peter Pan collar.",
                "young female, soft and sweet timbre, mid register, gentle pace",
            ),
            "character_2": _char(
                "猫爷爷",
                "An elderly male domestic longhair cat with fluffy orange tabby "
                "fur, white cheeks and a long ringed orange tail. He wears a "
                "plaid shirt under a dark green cable-knit cardigan and a tweed cap.",
                "elderly male, soft and mildly gravelly timbre, low register, "
                "slow and unhurried pace",
            ),
            "scene_1": {
                "kind": "scene", "name": "客厅",
                "description": (
                    "A cozy cottage living room around a cream-and-olive "
                    "loveseat against floral wallpaper, with warm golden sunlight."
                ),
                "ambience": "", "image": torch.zeros(1, 8, 8, 3),
            },
        }
        # Recorded from the pre-POV implementation.
        expected = {
            ("storyboard_banli_en.json", 0): "c5cdb234bccb3f75",
            ("storyboard_banli_en.json", 611937399731): "7af1a721ae5f24ba",
            ("storyboard_banli_zh.json", 0): "dc4e36c5cc0bc70d",
            ("storyboard_banli_zh.json", 611937399731): "9dd8410e08b4c487",
        }
        for (fx, seed), want in expected.items():
            doc = json.loads(
                (FIXTURES / fx).read_text(encoding="utf-8"),
            )
            plan_json = node.generate_v3(doc, seed=seed, **bundles)[0]
            got = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()[:16]
            assert got == want, (
                "%s seed=%s changed under the POV work: %s != %s"
                % (fx, seed, got, want)
            )


# --------------------------------------------------------------------------- #
# §6.3 — the POV fixture
# --------------------------------------------------------------------------- #
class TestPovFixture:
    def test_mixed_shot_has_both_forms(self, mods, pov_plan):
        _, out = pov_plan
        plan = json.loads(out[0])
        shot1 = "\n".join(plan["shots"][0]["prompt"])
        # Off-screen form with the reaction taken from `picture`.
        assert (
            "An off-screen voice (S1), adult female, calm narrative timbre, "
            "even pace, says in an off-screen voiceover: <d>[English] "
            "Before you were even in primary school.</d> while 小猫妹妹's lips "
            "remain completely closed; The kitten girl lowers her ears and "
            "glances down." in shot1
        )
        # On-screen form with the positive lip sentence.
        assert (
            "小猫妹妹 speaks as (S2) young female, soft and sweet timbre, "
            "gentle pace <d>[Chinese] 我那时候还不认识字呢。</d> 小猫妹妹's lips "
            "move naturally with the line." in shot1
        )

    def test_per_line_lang_override(self, mods, pov_plan):
        _, out = pov_plan
        plan = json.loads(out[0])
        # Every narration line in this fixture is English, so each one must
        # declare lang=en explicitly — the board default is zh.
        assert "<d>[English] Before you were even in primary school.</d>" in \
            "\n".join(plan["shots"][0]["prompt"])
        assert ("<d>[English] There were a few hedgehogs in the yard at "
                "home.</d>") in "\n".join(plan["shots"][1]["prompt"])
        assert ("<d>[English] And then you walked over and stepped on them "
                "one by one.</d>") in "\n".join(plan["shots"][2]["prompt"])
        # The kitten's Chinese lines still inherit the board default.
        assert "<d>[Chinese] 我那时候还不认识字呢。</d>" in \
            "\n".join(plan["shots"][0]["prompt"])

    def test_every_line_tag_matches_its_declared_language(self, pov_plan):
        """Fixture-level integrity check.

        ``lang`` is DECLARED, never detected — the assembler copies it
        verbatim (that ruling is what makes the plan byte-stable), so a
        mistyped fixture produces a confidently wrong <d> tag. Catching
        that here is the fixture test's job, not the assembler's: no
        language probing belongs in the assembly path.

        Every narration line in this fixture is English and the kitten's
        are Chinese, so the expectation is exact rather than heuristic.
        """
        import re
        doc, out = pov_plan
        plan = json.loads(out[0])
        cjk = re.compile(r"[\u4e00-\u9fff]")
        for shot_out, shot_in in zip(plan["shots"], doc["shots"]):
            text = "\n".join(shot_out["prompt"])
            for line_in in shot_in["lines"]:
                declared = line_in.get("lang") or doc["language"]
                tag = "[Chinese]" if declared == "zh" else "[English]"
                assert "<d>%s %s</d>" % (tag, line_in["text"]) in text, (
                    "line %r declared %s but text is %s; a mistyped fixture "
                    "yields a confidently wrong <d> tag"
                    % (line_in["text"], declared,
                       "CJK" if cjk.search(line_in["text"]) else "latin")
                )

    def test_pov_shot1_replaces_the_two_person_template(self, mods, pov_plan):
        _, out = pov_plan
        plan = json.loads(out[0])
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "First-person point of view: the lens sits where the " \
                "off-screen speaker is." in text, shot["id"]
            assert "The camera sits beside the pair" not in text, shot["id"]

    def test_offscreen_speaker_gets_no_frame_pin(self, mods, pov_plan):
        """A frame pin says "visible at <position>" — the exact opposite of
        an off-screen voice. Layout COMPLETION must skip them too, not just
        a hand-written pin."""
        _, out = pov_plan
        plan = json.loads(out[0])
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "叙述者 stays" not in text, shot["id"]
            assert "叙述者 stays at left of frame" not in text, shot["id"]

    def test_pure_listen_summary_wording(self, mods, pov_plan):
        """No on-screen speaker in frame → the listening form."""
        _, out = pov_plan
        plan = json.loads(out[0])
        summaries = []
        for s in plan["shots"]:
            lines = s["prompt"]
            i = lines.index("summary:")
            summaries.append(lines[i + 1])
        # Every fixture shot has the kitten speaking, so the mixed form is
        # used (verb agrees with the count: "are" for two, "is" for one);
        # assert the listening variant separately.
        assert all("visible in the frame and speaking" in s
                   for s in summaries)
        assert "小猫妹妹 and 猫爷爷 are visible in the frame and speaking" \
            in summaries[0]
        assert "小猫妹妹 is visible in the frame and speaking" in summaries[2]
        listening = mods["prompts"].presence_summary(
            on_screen_names=["小猫妹妹"], speaking_names=[],
            offscreen_names=["叙述者"],
        )
        assert "小猫妹妹 is visible in the frame, listening; 叙述者's " \
            "voice is off-screen." in listening

    def test_absent_declaration_and_narrowed_pin(self, mods, pov_plan):
        _, out = pov_plan
        plan = json.loads(out[0])
        shot3 = "\n".join(plan["shots"][2]["prompt"])
        assert "猫爷爷 is not in the frame in this shot." in shot3
        # The absent character's pin is gone…
        assert "猫爷爷 stays" not in shot3
        # …but its identity anchors and picture binding SURVIVE (§4.3).
        assert "<Picture 3>" in shot3
        assert "猫爷爷" in shot3
        # And the summary names only who is present.
        summary = shot3.split("summary:")[1].splitlines()[1]
        assert "小猫妹妹 is visible in the frame" in summary
        assert "猫爷爷" not in summary

    def test_identity_pins_keep_full_cast(self, mods, pov_plan):
        """Omitting an identity anchor is the historical defect source —
        the absent character keeps its subject/picture/retention lines."""
        _, out = pov_plan
        plan = json.loads(out[0])
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            for k in (1, 2, 3):
                assert "<Subject %d>" % k in text, (shot["id"], k)
            assert "fully_preserved" in text, shot["id"]

    def test_offscreen_speaker_keeps_its_slot_and_voice(self, mods, pov_plan):
        """Presence changes the SENTENCE FORM, never the slot numbering."""
        _, out = pov_plan
        plan = json.loads(out[0])
        text = "\n".join(plan["shots"][0]["prompt"])
        assert "An off-screen voice (S1)" in text
        # Voice descriptor present on the off-screen line too.
        assert "adult female, calm narrative timbre, even pace" in text


# --------------------------------------------------------------------------- #
# §6.4 — validator
# --------------------------------------------------------------------------- #
def _base_board():
    return {
        "name": "X", "version": 1, "language": "zh",
        "camera_discipline": "locked", "pacing": "normal",
        "seed_policy": "per_scene_increment",
        "prompt_prefix": "global style.",
        "cast": {
            "叙述者": {"character_slot": 1, "offscreen": True},
            "甲": {"character_slot": 2},
            "乙": {"character_slot": 3},
        },
        "spatial_layout": {"甲": "left of frame", "乙": "right of frame"},
        "shots": [{"lines": [
            {"speaker": "叙述者", "text": "hi"},
            {"speaker": "甲", "text": "yo"},
        ]}],
    }


class TestValidator:
    def test_baseline_board_is_valid(self, mods):
        out = mods["storyboard"].validate_storyboard_payload(_base_board())
        assert out["cast"]["叙述者"] == {
            "character_slot": 1, "offscreen": True,
        }

    def test_offscreen_in_spatial_layout_rejected(self, mods):
        board = _base_board()
        board["spatial_layout"]["叙述者"] = "left of frame"
        with pytest.raises(ValueError, match="offscreen speaker"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_absent_containing_offscreen_rejected(self, mods):
        board = _base_board()
        board["shots"][0]["absent"] = ["叙述者"]
        with pytest.raises(ValueError, match="offscreen speaker"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_absent_not_in_cast_rejected(self, mods):
        board = _base_board()
        board["shots"][0]["absent"] = ["路人"]
        with pytest.raises(ValueError, match="not in cast"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_all_empty_shot_rejected(self, mods):
        """Marking every on-screen character absent leaves an empty room —
        the known empty-plate render defect."""
        board = _base_board()
        board["shots"][0]["absent"] = ["甲", "乙"]
        with pytest.raises(ValueError, match="no on-screen"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_every_cast_offscreen_rejected(self, mods):
        board = _base_board()
        board["cast"]["甲"] = {"character_slot": 2, "offscreen": True}
        board["cast"]["乙"] = {"character_slot": 3, "offscreen": True}
        # Nobody is in frame, so no frame pin may remain either.
        board["spatial_layout"] = {}
        with pytest.raises(ValueError, match="every cast member is offscreen"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_bad_line_lang_rejected(self, mods):
        board = _base_board()
        board["shots"][0]["lines"][0]["lang"] = "fr"
        with pytest.raises(ValueError, match="lang must be one of"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_non_bool_offscreen_rejected(self, mods):
        board = _base_board()
        board["cast"]["叙述者"]["offscreen"] = "yes"
        with pytest.raises(ValueError, match="must be a bool"):
            mods["storyboard"].validate_storyboard_payload(board)

    def test_absent_and_lang_omitted_from_cleaned_when_unset(self, mods):
        """Cleaned output keeps the no-POV payload byte-identical."""
        out = mods["storyboard"].validate_storyboard_payload(_base_board())
        assert "absent" not in out["shots"][0]
        assert "lang" not in out["shots"][0]["lines"][0]
        assert "offscreen" not in out["cast"]["甲"]


# --------------------------------------------------------------------------- #
# §6.5 — neutralize_offscreen_claims conditionalisation
# --------------------------------------------------------------------------- #
class TestNeutralizeConditional:
    LINES = [
        "subject_definitions:",
        "甲 stays at left of frame.",
        "The camera is first-person POV.",
        "summary:",
        "ok",
    ]

    def test_default_keeps_the_visible_cast_assertion(self, mods):
        out = mods["prompts"].neutralize_offscreen_claims(self.LINES)
        assert "All speaking characters are visible in the frame. " \
            "None of them is the camera." in out
        assert "first-person POV" not in "\n".join(out)

    def test_declared_drops_assertion_but_still_scrubs_prose(self, mods):
        out = mods["prompts"].neutralize_offscreen_claims(
            self.LINES, declared_offscreen=True,
        )
        joined = "\n".join(out)
        # The prose claim is STILL a defect and still goes…
        assert "first-person POV" not in joined
        # …but the contradictory assertion is not injected.
        assert "All speaking characters are visible in the frame." not in joined
        assert "None of them is the camera." not in joined

    def test_default_is_byte_identical_to_the_old_one_arg_call(self, mods):
        assert mods["prompts"].neutralize_offscreen_claims(
            self.LINES, declared_offscreen=False,
        ) == mods["prompts"].neutralize_offscreen_claims(self.LINES)

    def test_pov_board_does_not_receive_the_assertion(self, mods, pov_plan):
        _, out = pov_plan
        plan = json.loads(out[0])
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            assert "None of them is the camera." not in text, shot["id"]


# --------------------------------------------------------------------------- #
# §6.6 — V2 suffix parsing
# --------------------------------------------------------------------------- #
class TestOffscreenSuffixParsing:
    @pytest.mark.parametrize("label", [
        "小D（画外）", "小D(画外)", "小D（画外音）", "小D（画外声）",
        "小D(off-screen)", "小D (offscreen)", "小D(off screen)",
        "小D（画外人声）", "Narrator(voiceover)", "Narrator（voice-over）",
    ])
    def test_whitelisted_markers_detected(self, mods, label):
        seg = mods["segmenter"]
        bare, off = seg.split_offscreen_marker(label)
        assert off is True, label
        assert bare == label.rsplit("（", 1)[0].rsplit("(", 1)[0].strip()

    @pytest.mark.parametrize("label", [
        "小猫妹妹（嫌弃）", "小D（医生）", "妈妈（生气）", "爷爷",
    ])
    def test_mood_parentheticals_are_not_markers(self, mods, label):
        """The trap: a mood qualifier must never be read as presence."""
        seg = mods["segmenter"]
        bare, off = seg.split_offscreen_marker(label)
        assert off is False, label
        assert bare == label
        # And the mood strip still works exactly as before.
        assert seg.normalize_speaker_with_presence(label) == (
            seg.normalize_speaker_token(label), False,
        )

    def test_normalize_with_presence_combines_both_strips(self, mods):
        seg = mods["segmenter"]
        assert seg.normalize_speaker_with_presence("小D（画外）") == ("小D", True)
        assert seg.normalize_speaker_with_presence("小猫妹妹哭着说") == (
            "小猫妹妹", False,
        )
        assert seg.normalize_speaker_with_presence("小猫妹妹（画外）哭着说") == (
            "小猫妹妹", True,
        )

    def test_parsed_turn_carries_presence(self, mods):
        seg = mods["segmenter"]
        concept = "小D（画外）：你听我说\n小猫妹妹：我在听"
        turns, _prologue = seg.parse_structured_dialogue_turns(concept)
        assert turns is not None
        by_speaker = {t.speaker: t for t in turns}
        assert by_speaker["小D"].offscreen is True
        assert by_speaker["小猫妹妹"].offscreen is False

    def test_plain_board_turns_all_default_false(self, mods):
        seg = mods["segmenter"]
        turns, _ = seg.parse_structured_dialogue_turns("小猫妹妹：你好\n爷爷：嗯")
        assert all(t.offscreen is False for t in turns)


# --------------------------------------------------------------------------- #
# §6.7 — (Sn) / audio slots unaffected by presence
# --------------------------------------------------------------------------- #
class TestSlotAlignment:
    def test_offscreen_keeps_its_character_slot_and_roster_order(self, mods, pov_plan):
        _, out = pov_plan
        roster = out[4]
        names = [b["name"] for b in roster]
        # The off-screen speaker occupies slot 1 like anyone else.
        assert names == ["叙述者", "小猫妹妹", "猫爷爷", "客厅"]
        assert out[3] is not None and out[3].shape[0] == 4
        # audio comes off the same bundles — an off-screen voice still
        # needs its sample to be heard.
        assert all(
            b.get("audio") is not None for b in roster
            if b.get("kind") == "character"
        )

    def test_picture_numbering_is_contiguous(self, mods, pov_plan):
        _, out = pov_plan
        plan = json.loads(out[0])
        for shot in plan["shots"]:
            text = "\n".join(shot["prompt"])
            for n in (1, 2, 3, 4):
                assert "<Picture %d>" % n in text, (shot["id"], n)

    def test_gap_cast_still_rejected_on_a_pov_board(self, mods):
        """Presence must not relax the slot contract: {1,2,4} is a gap."""
        builder = mods["builder"]
        doc = json.loads(POV_FIXTURE.read_text(encoding="utf-8"))
        doc["cast"]["猫爷爷"] = {"character_slot": 4}
        with pytest.raises(ValueError, match="contiguous"):
            builder.assemble_plan(
                storyboard=doc,
                slot_bundles={
                    1: _char("叙述者", "d", "v"),
                    2: _char("小猫妹妹", "d", "v"),
                    4: _char("猫爷爷", "d", "v"),
                    5: {"kind": "scene", "name": "客厅", "description": "d",
                        "image": None},
                },
                seed_base=1,
            )


# --------------------------------------------------------------------------- #
# §4.6 — the addendum contract and the re-roll advice
# --------------------------------------------------------------------------- #
class TestAddendumAndAdvice:
    def test_addendum_carries_the_three_part_contract(self):
        raw = (
            PROJECT_DIR / "nodes" / "llm" / "prompts" / "h3_loop"
            / "ref2v_addendum.txt"
        ).read_text(encoding="utf-8")
        # The prompt is hard-wrapped, so match on whitespace-collapsed
        # text rather than on the literal line breaks.
        text = " ".join(raw.split())
        assert "says in an off-screen voiceover" in text
        assert "lips remain completely closed" in text
        assert "one beat of separation" in text
        assert "(S<n>)" in text

    def test_pov_advice_is_advisory_only(self, mods):
        seg = mods["segmenter"]
        import importlib.util as _ilu
        path = (
            PROJECT_DIR / "nodes" / "llm" / "minimax_h3_loop_prompt_generator.py"
        )
        spec = _ilu.spec_from_file_location(
            "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator", path,
        )
        gen = _ilu.module_from_spec(spec)
        sys.modules[spec.name] = gen
        spec.loader.exec_module(gen)
        # No off-screen turns -> silent.
        plain = [seg.DialogueTurn(speaker="甲", lines=["hi"])]
        assert gen._pov_offscreen_advice(plain) == []
        # With one -> the re-roll recipe, and it does NOT touch packing.
        pov = [seg.DialogueTurn(speaker="小D", lines=["hi"], offscreen=True)]
        out = gen._pov_offscreen_advice(pov)
        assert len(out) == 1
        assert "re-roll" in out[0]
        assert "scene_range" in out[0]


# --------------------------------------------------------------------------- #
# Community-recipe equivalence fixture: the 小D/Claude first-person prompt
# (2026-10-20 user validation) — pins that our assembler reproduces the
# community-verified POV dialogue shape from a storyboard asset.
# --------------------------------------------------------------------------- #
CLAUDE_001_FIXTURE = FIXTURES / "storyboard_pov_claude_001.json"


@pytest.fixture(scope="module")
def claude_001_plan(mods):
    doc = json.loads(CLAUDE_001_FIXTURE.read_text(encoding="utf-8"))
    node = mods["builder"].MiniMaxH3LoopPromptGeneratorV3()
    out = node.generate_v3(
        doc,
        seed=20261020,
        character_1=_char(
            "小D", "An off-screen male caller with no on-screen body.",
            "adult male, casual and clear timbre, mid register",
        ),
        character_2=_char(
            "Claude", "A calm young woman with long dark hair, seated facing the camera.",
            "adult female, soft and playful timbre, mid-high register",
        ),
        scene_1={
            "kind": "scene", "name": "室内",
            "description": "A cozy modern room with natural indoor lighting.",
            "ambience": "", "image": __import__("torch").zeros(1, 8, 8, 3),
        },
    )
    return doc, out


class TestClaude001CommunityRecipe:
    def test_six_lines_three_scenes(self, claude_001_plan):
        _doc, out = claude_001_plan
        full = "\n".join("\n".join(s["prompt"]) for s in json.loads(out[0])["shots"])
        assert full.count("<d>") == 6
        assert len(json.loads(out[0])["shots"]) == 3

    def test_community_three_part_offscreen_form(self, claude_001_plan):
        _doc, out = claude_001_plan
        full = "\n".join("\n".join(s["prompt"]) for s in json.loads(out[0])["shots"])
        # fixed phrase + embedded mute clause, three times
        assert full.count("says in an off-screen voiceover:") == 3
        assert full.count("lips remain completely closed") == 3
        # reactions are the authored picture cues, not the default
        assert "slight tilt of the head" in full
        assert "eyebrows in mild surprise" in full
        assert "amused expression" in full
        # on-screen lines carry the positive lip sentence
        assert full.count("lips move naturally with the line.") == 3

    def test_speaker_ids_and_languages(self, claude_001_plan):
        _doc, out = claude_001_plan
        full = "\n".join("\n".join(s["prompt"]) for s in json.loads(out[0])["shots"])
        assert "off-screen voice (S1)" in full
        assert "Claude speaks as (S2)" in full
        # 小D speaks EN twice then ZH; Claude is ZH throughout
        assert "<d>[English] Claude, help me resolve this document.</d>" in full
        assert "<d>[English] wait, i'm not Chinese.</d>" in full
        assert "[Chinese] 真，真的吗？</d>" in full

    def test_offscreen_speaker_has_no_frame_pin(self, claude_001_plan):
        _doc, out = claude_001_plan
        full = "\n".join("\n".join(s["prompt"]) for s in json.loads(out[0])["shots"])
        assert "小D stays" not in full
        assert full.count("Claude stays at center of frame") == 3
        assert "小D: " in full and "Claude: " in full  # aliases keep full cast

    def test_pov_opener_and_community_soundscape(self, claude_001_plan):
        _doc, out = claude_001_plan
        full = "\n".join("\n".join(s["prompt"]) for s in json.loads(out[0])["shots"])
        assert full.count("First-person point of view") == 3
        assert "Quiet indoor room tone, soft fabric movement, natural breathing." in full
