"""Regression cases from the offscreen/storyboard audit."""
import json

import pytest
from .test_pov_offscreen import mods, _char
from .test_h3_loop_prompt_generator import lg, ScriptedConnector, _clip_reply, _single_call_reply


def board():
    return {
        "version": 1, "name": "Presence-audit", "language": "zh",
        "prompt_prefix": ["Natural indoor lighting."],
        "cast": {"N": {"character_slot": 1, "offscreen": True},
                 "A": {"character_slot": 2}, "B": {"character_slot": 3}},
        "ambience": "Quiet room tone.",
        "shots": [{"lines": [{"speaker": "N", "text": "你好。"},
                              {"speaker": "A", "text": "好的。"}]}],
    }


def bundles():
    import torch
    return {
        1: _char("N", "An unseen caller.", "adult male, low pitch"),
        2: _char("A", "A young woman.", "adult female, soft"),
        3: _char("B", "An old man.", "elderly male, gravelly"),
        4: {"kind": "scene", "name": "room", "description": "A plain room.",
            "ambience": "Quiet room tone.", "image": torch.zeros(1, 8, 8, 3)},
    }


def assemble(mods, doc):
    return mods["builder"].assemble_plan(storyboard=doc, slot_bundles=bundles(), seed_base=1)


def test_absent_speaker_is_rejected(mods):
    doc = board()
    doc["shots"][0]["absent"] = ["A"]
    with pytest.raises(ValueError, match="absent.*A|A.*absent"):
        mods["storyboard"].validate_storyboard_payload(doc)


def test_single_explicit_offscreen_line_is_structured_dialogue(mods):
    turns, _ = mods["segmenter"].parse_structured_dialogue_turns("N(offscreen)：你好。")
    assert turns is not None
    assert turns[0].offscreen


def test_silent_cast_stays_visible_without_authored_layout(mods):
    text = "\n".join(assemble(mods, board())["plan"]["shots"][0]["prompt"])
    assert "A and B are visible in the frame" in text
    assert "B stays" in text


def test_offscreen_does_not_imply_first_person(mods):
    text = "\n".join(assemble(mods, board())["plan"]["shots"][0]["prompt"])
    assert "First-person point of view" not in text
    assert "lens sits where the off-screen speaker is" not in text


def test_first_person_requires_explicit_field_and_is_preserved(mods):
    doc = board()
    doc["camera_perspective"] = "first_person"
    cleaned = mods["storyboard"].validate_storyboard_payload(doc)
    assert cleaned["camera_perspective"] == "first_person"
    assert "First-person point of view" in "\n".join(assemble(mods, doc)["plan"]["shots"][0]["prompt"])
    doc["camera_perspective"] = "sideways"
    with pytest.raises(ValueError, match="camera_perspective"):
        mods["storyboard"].validate_storyboard_payload(doc)


def test_listen_only_shot_never_claims_visible_speech(mods):
    doc = board()
    doc["camera_perspective"] = "first_person"
    doc["shots"][0]["lines"] = doc["shots"][0]["lines"][:1]
    text = "\n".join(assemble(mods, doc)["plan"]["shots"][0]["prompt"])
    assert "speaking to the unseen speaker" not in text
    assert "A's lips remain completely closed" in text
    assert "B's lips remain completely closed" in text


def test_mixed_presence_inserts_an_explicit_turn_boundary(mods):
    text = "\n".join(assemble(mods, board())["plan"]["shots"][0]["prompt"])
    assert "After one silent beat, the next speaker begins." in text


def test_same_language_warning_does_not_demand_rewriting_dialogue(mods):
    out = assemble(mods, board())
    assert not any("Give the off-screen speaker a different" in x for x in out["summary_lines"])


def test_declared_presence_removes_stale_all_visible_assertion(mods):
    got = mods["prompts"].neutralize_offscreen_claims(
        ["All speaking characters are visible in the frame. None of them is the camera."],
        declared_offscreen=True,
    )
    assert not any("All speaking characters are visible" in x for x in got)


@pytest.mark.parametrize("mode", ["per_shot", "single_call"])
def test_v2_explicit_presence_reaches_final_prompt(lg, mode):
    reply = _single_call_reply(1) if mode == "single_call" else _clip_reply(1)
    connector = ScriptedConnector([reply])
    out = lg.MiniMaxH3LoopPromptGeneratorV2().generate_v2(
        connector, user_input="N(offscreen)：你好。\nA：好的。",
        enhance_user_input="off", generation_mode=mode, seed=1,
    )
    text = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    assert "says in an off-screen voiceover" in text
    assert "N speaks as" not in text
    assert "All speaking characters are visible" not in text
    assert "N stays" not in text
    assert "A's lips remain completely closed" in text
    assert "A's lips move naturally with the line" in text


def test_speaker_normalization_keeps_offscreen_flag(lg):
    turn = lg._DLG_DialogueTurn(speaker="N", lines=["你好。"], offscreen=True)
    got = lg._normalize_turn_speakers([turn], ["A"])
    assert got[0].speaker == "N"
    assert got[0].offscreen


@pytest.mark.parametrize("declared", [False, True])
def test_enhancement_cannot_invent_or_erase_presence(lg, monkeypatch, declared):
    original = "N(offscreen)：你好。\nA：好的。" if declared else "N：你好。\nA：好的。"
    rewritten = "N：你好。\nA：好的。" if declared else "N(offscreen)：你好。\nA：好的。"
    monkeypatch.setattr(lg.H3LoopPromptEnhancer, "enhance_user_input_draft",
                        lambda *args, **kwargs: (rewritten, "test rewrite"))
    out = lg.H3LoopPromptEnhancer(ScriptedConnector([_clip_reply(1)]))(
        user_input=original, enhance_user_input=True, seed=1,
    )
    text = "\n".join(json.loads(out["plan_json"])["shots"][0]["prompt"])
    assert ("says in an off-screen voiceover" in text) is declared


def test_v2_roster_has_separate_picture_and_audio_spaces(lg):
    bs = bundles()
    roster, manifest, images, _ = lg.MiniMaxH3LoopPromptGeneratorV2._build_roster_from_bundles(
        [bs[1], bs[2]], [bs[4]], offscreen_names={"N"},
    )
    assert images.shape[0] == 2
    assert [x["name"] for x in json.loads(roster)] == ["A", "room"]
    assert len(manifest) == 2


@pytest.mark.parametrize("wire_offscreen", [True, False])
def test_v2_asset_to_slots_matches_actual_audio_order(lg, mods, monkeypatch, wire_offscreen):
    import importlib
    bs = bundles()
    chars = [bs[1], bs[2]] if wire_offscreen else [bs[2]]
    _, manifest, _, _ = lg.MiniMaxH3LoopPromptGeneratorV2._build_roster_from_bundles(
        chars, [bs[4]], offscreen_names={"N"},
    )
    reply = "\n".join(lg.build_fallback_shot_lines(lg.SCHEMA_SIX, manifest=manifest))
    monkeypatch.setattr(lg.H3LoopPromptEnhancer, "extract_board_facts",
                        lambda *args, **kwargs: ([], {"layout": {"N": "left of frame", "A": "center of frame"}, "bindings": []}))
    monkeypatch.setattr(lg.H3LoopPromptEnhancer, "_synth_prefix",
                        lambda *args, **kwargs: (["Natural indoor lighting."], {"N": "An unseen caller.", "A": "A young woman."}, {}))
    kwargs = {"character_2": bs[2], "scene_1": bs[4]}
    if wire_offscreen:
        kwargs["character_1"] = bs[1]
    out = lg.MiniMaxH3LoopPromptGeneratorV2().generate_v2(
        ScriptedConnector([reply]), user_input="N(offscreen)：你好。\nA：好的。",
        enhance_user_input="off", reference_mode="ref2va", seed=1, **kwargs,
    )
    text = "\n".join(json.loads(out[0])["shots"][0]["prompt"])
    assets = out[4]
    slots_mod = importlib.import_module("_mienodes_internal.nodes.assets.nodes")
    slots = slots_mod.AssetToSlots().split(assets)
    assert slots[0] == 2
    assert slots[1] is bs[2]["image"]
    assert slots[2] is bs[4]["image"]
    assert slots[10] == (2 if wire_offscreen else 1)
    assert slots[11] is bs[1 if wire_offscreen else 2]["audio"]
    assert "<Subject 1> -> <Picture 1>: A young woman." in text
    assert "<Picture 2>: A plain room." in text
    assert "N stays" not in text
    if wire_offscreen:
        assert "off-screen voice (S1), adult male" in text
        assert "A speaks as (S2) adult female" in text
        assert assets[0]["image"] is None
    else:
        assert "An off-screen voice says" in text
        assert "A speaks as (S1) adult female" in text
