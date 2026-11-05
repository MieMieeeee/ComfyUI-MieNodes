# -*- coding: utf-8 -*-
"""The continuity critic (2026-11-05).

Two halves, both locked here:

* the DETERMINISTIC cross-shot checks — mechanical, free, always on:
  a garment that changes colour, a prop in two pairs of hands, a hand
  that is empty and holding at once, a character who acts where the
  visibility replay has them off stage, an hour that contradicts the
  light preset;
* the MODEL critic — a second reader over the finished asset. Its
  protocol is the part worth locking: a finding WITHOUT a path is
  discarded, an unreadable reply is "unsure" and unsure never blocks,
  only CRITICAL findings buy a repair round, and a critic that cannot
  be satisfied still delivers the contract-valid asset with the notes
  attached (a quality gate must not become a hard failure).
"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from .critic_connector import NO_FINDINGS, critic_reply, is_critic_call
from .test_pov_offscreen import mods
from .test_h3_loop_prompt_generator import lg
from .test_h3_storyboard_generator import sb
from .test_storyboard_generation_contract import apple as apple_asset

PROJECT_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def deterministic(mods):
    """The mechanical critic module."""
    return importlib.import_module(mods["storyboard"].__package__ + ".storyboard_critic")


@pytest.fixture(scope="module")
def model_critic(mods):
    """The model critic module."""
    return importlib.import_module(mods["builder"].__package__ + ".storyboard_critic")


@pytest.fixture(scope="module")
def generator(mods):
    """The asset generator (the node's brain)."""
    fqn = "_mienodes_internal.nodes.llm.storyboard_asset_generator"
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(
        fqn, str(PROJECT_DIR / "nodes" / "llm" / "storyboard_asset_generator.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = module
    spec.loader.exec_module(module)
    return module


class CriticConnector:
    """Serves the asset reply from one queue and the critic from another.

    The critic is a SECOND READER, not a pipeline stage: its reply must
    not eat the next scripted reply.
    """

    def __init__(self, replies, critic_replies=None):
        self.replies = list(replies)
        self.critic_replies = list(critic_replies or [])
        self.calls = []
        self.critic_calls = []

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        if is_critic_call(messages):
            self.critic_calls.append(messages)
            return (self.critic_replies.pop(0) if self.critic_replies
                    else NO_FINDINGS)
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("scripted connector exhausted")
        return self.replies.pop(0)

    def get_state(self):
        return "critic-test"


def _kinds(issues):
    return {issue["kind"] for issue in issues}


def _tiny_asset():
    """A minimal v2 asset: Character 2 has not entered yet."""
    return {
        "version": 2, "name": "tiny", "language": "en",
        "roles": [
            {"id": "character_1", "name": "A", "presence": "on_screen",
             "initially_visible": True, "position": "left", "voice": ""},
            {"id": "character_2", "name": "B", "presence": "on_screen",
             "initially_visible": False, "position": "none", "voice": ""}],
        "camera": {"framing": "medium_shot", "movement": "locked",
                   "perspective": "observer"},
        "look": "", "setting": "", "constraints": "",
        "target_duration_seconds": 8, "continuation_overlap_frames": 22,
        "source_script": "A: hi",
        "shots": [{"id": "scene_01", "duration_seconds": 8, "events": [
            {"id": "event_001", "type": "action", "character": "character_1",
             "text": "turns", "language": "en"}]}],
        "execution": {"look": "", "setting": "",
                      "voices": {"character_1": "", "character_2": ""},
                      "visible_state": "Character 1 stands in the present.",
                      "directions": {"event_001": "Character 1 turns in place."}},
        "light": "follow_image",
    }


# --------------------------------------------------------------------------- #
# the deterministic half
# --------------------------------------------------------------------------- #
def test_the_verified_apple_asset_is_clean(deterministic):
    """No false positives on the asset our own prompt ships."""
    assert deterministic.review_asset_critique(apple_asset()) == []


def test_the_contract_copy_empty_hands_beside_a_held_object_is_clean(deterministic):
    """The refusal copy the contract DEMANDS is not a contradiction:
    the refusing actor's hands are empty while the other one holds the
    single apple."""
    issues = deterministic.review_asset_critique(apple_asset())
    assert "prop_holds_and_empty_hands" not in _kinds(issues)


def test_one_garment_keeps_one_colour(deterministic):
    raw = apple_asset()
    raw["execution"]["directions"]["event_001"] = (
        "Character 2 retains the single apple in a red coat while extending "
        "it toward Character 1.")
    raw["execution"]["directions"]["event_004"] = (
        "Character 2 still holds the apple and moves half a step closer to "
        "Character 1 in a blue coat.")
    issues = deterministic.review_asset_critique(raw)
    assert "appearance_conflict" in _kinds(issues)


def test_a_hand_cannot_hold_and_be_empty_at_once(deterministic):
    raw = apple_asset()
    raw["execution"]["directions"]["event_004"] = (
        "Character 2 holds the single apple with both hands visibly empty "
        "while moving closer to Character 1.")
    issues = deterministic.review_asset_critique(raw)
    hard = [i for i in issues if i["kind"] == "prop_holds_and_empty_hands"]
    assert hard and hard[0]["severity"] == deterministic.HARD


def test_the_single_prop_cannot_be_in_two_pairs_of_hands(deterministic):
    raw = apple_asset()
    raw["execution"]["directions"]["event_004"] = (
        "Character 2 holds the single apple and Character 1 holds the single "
        "apple at the same time.")
    issues = deterministic.review_asset_critique(raw)
    double = [i for i in issues if i["kind"] == "prop_double_holder"]
    assert double and double[0]["severity"] == deterministic.HARD


def test_a_character_who_is_not_on_stage_is_reported(deterministic):
    raw = _tiny_asset()
    raw["execution"]["directions"]["event_001"] = (
        "Character 1 turns toward Character 2 and waits for an answer.")
    issues = deterministic.review_asset_critique(raw)
    assert "character_absent" in _kinds(issues)


def test_the_opening_state_may_not_name_an_absent_character(deterministic):
    raw = _tiny_asset()
    raw["execution"]["visible_state"] = (
        "Character 1 and Character 2 stand together in the present.")
    issues = deterministic.review_asset_critique(raw)
    assert "opening_state_absent" in _kinds(issues)


def test_the_hour_must_match_the_light_anchor(deterministic):
    raw = _tiny_asset()
    raw["light"] = "night_lamp"
    raw["execution"]["directions"]["event_001"] = (
        "Character 1 steps into the bright daylight by the window.")
    issues = deterministic.review_asset_critique(raw)
    assert "light_conflict" in _kinds(issues)


def test_one_segment_carries_one_beat(deterministic):
    raw = _tiny_asset()
    raw["shots"][0]["events"] = [
        {"id": "event_001", "type": "action", "character": "character_1",
         "text": "a", "language": "en"},
        {"id": "event_002", "type": "action", "character": "character_1",
         "text": "b", "language": "en"},
        {"id": "event_003", "type": "action", "character": "character_1",
         "text": "c", "language": "en"}]
    raw["execution"]["directions"] = {
        "event_001": "Character 1 turns.", "event_002": "Character 1 steps.",
        "event_003": "Character 1 stops."}
    issues = deterministic.review_asset_critique(raw)
    assert "multi_action_segment" in _kinds(issues)


def test_a_version_one_asset_is_not_reviewed(deterministic):
    assert deterministic.review_asset_critique({"version": 1}) == []


def test_the_legacy_board_review_looks_at_repeated_props(deterministic):
    shots = [
        {"id": "scene_01", "characters": ["A"], "props": ["bowl"]},
        {"id": "scene_02", "characters": ["B"], "props": ["bowl"]},
    ]
    issues = deterministic.review_shot_list_critique(shots)
    assert "prop_cast_conflict" in _kinds(issues)


# --------------------------------------------------------------------------- #
# the model half: the protocol
# --------------------------------------------------------------------------- #
def test_a_finding_without_a_path_is_discarded(model_critic):
    conn = CriticConnector([], [critic_reply([
        {"severity": "critical", "issue": "the coat changes", "fix": "keep one coat"}])])
    result = model_critic.run_asset_critic(conn, apple_asset())
    assert result["findings"] == []
    assert result["unsure"] is False


def test_an_unreadable_reply_is_unsure_and_blocks_nothing(model_critic):
    conn = CriticConnector([], ["not json at all", "still not json"])
    result = model_critic.run_asset_critic(conn, apple_asset())
    assert result["findings"] == []
    assert result["unsure"] is True


def test_a_critic_that_cannot_reach_the_provider_never_breaks_generation(model_critic):
    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("provider down")

    result = model_critic.run_asset_critic(Broken(), apple_asset())
    assert result["findings"] == [] and result["unsure"] is True


def test_only_critical_findings_ask_for_a_repair(model_critic):
    conn = CriticConnector([], [critic_reply([
        {"severity": "critical", "path": "execution.directions.event_004",
         "issue": "the coat changes", "fix": "keep one coat"},
        {"severity": "minor", "path": "execution.visible_state",
         "issue": "the room drifts", "fix": "name the room"}])])
    result = model_critic.run_asset_critic(conn, apple_asset())
    assert len(result["findings"]) == 2
    assert [f["path"] for f in model_critic.critical_findings(result)] == [
        "execution.directions.event_004"]


def test_the_critic_reads_the_asset_not_the_render(model_critic):
    conn = CriticConnector([])
    model_critic.run_asset_critic(conn, apple_asset())
    system = conn.critic_calls[0][0]["content"]
    assert "SKEPTICAL QA reviewer" in system
    payload = json.loads(conn.critic_calls[0][1]["content"])
    assert payload["source_script"] == apple_asset()["source_script"]
    assert payload["shots"][0]["id"] == "scene_01"


# --------------------------------------------------------------------------- #
# the repair loop
# --------------------------------------------------------------------------- #
def _asset_options(raw):
    from .test_storyboard_generation_node import options
    settings = options(raw)
    settings.pop("critic", None)
    return settings


def test_a_critical_finding_buys_a_repair_round(generator):
    raw = apple_asset()
    conn = CriticConnector(
        [json.dumps(raw), json.dumps(raw)],
        [critic_reply([{"severity": "critical", "path": "execution.directions.event_004",
                         "issue": "the coat changes", "fix": "keep one coat"}]),
         NO_FINDINGS])
    planner = generator.AssetStoryboardEnhancer(conn)
    asset = planner(raw["source_script"], 2, **_asset_options(raw))
    assert asset["name"] == raw["name"]
    assert len(conn.critic_calls) == 2
    feedback = conn.calls[1][-1]["content"]
    assert "critic" in feedback.lower() and "keep one coat" in feedback


def test_an_unrepairable_critic_still_delivers_the_contract_valid_asset(generator):
    raw = apple_asset()
    conn = CriticConnector([json.dumps(raw)] * 3,
                            [critic_reply([{"severity": "critical",
                                            "path": "execution.visible_state",
                                            "issue": "the room drifts",
                                            "fix": "name the room"}])] * 3)
    planner = generator.AssetStoryboardEnhancer(conn)
    asset = planner(raw["source_script"], 2, **_asset_options(raw))
    assert asset["version"] == 2
    assert any("critic" in warning for warning in planner.warnings)


def test_critic_off_makes_no_extra_call(generator):
    raw = apple_asset()
    conn = CriticConnector([json.dumps(raw)])
    planner = generator.AssetStoryboardEnhancer(conn)
    planner(raw["source_script"], 2, critic="off", **_asset_options(raw))
    assert conn.critic_calls == []
    assert len(conn.calls) == 1


def test_the_critic_notes_reach_the_node_preview(generator):
    raw = apple_asset()
    conn = CriticConnector([json.dumps(raw)],
                            [critic_reply([{"severity": "minor", "path": "execution.visible_state",
                                            "issue": "the room drifts", "fix": "name the room"}])])
    planner = generator.AssetStoryboardEnhancer(conn)
    planner(raw["source_script"], 2, **_asset_options(raw))
    assert any("critic" in warning for warning in planner.warnings)


def test_the_node_widget_defaults_to_on(sb):
    spec = sb.MiniMaxH3StoryboardGenerator.INPUT_TYPES()["optional"]["critic"]
    assert spec[0] == ["on", "off"]
    assert spec[1]["default"] == "on"


# --------------------------------------------------------------------------- #
# the zero-LLM track: the V3 assembler reports the notes
# --------------------------------------------------------------------------- #
def test_the_v3_assembler_reports_the_critic_notes_without_a_model(mods):
    raw = apple_asset()
    raw["execution"]["directions"]["event_001"] = (
        "Character 2 retains the single apple in a red coat while extending "
        "it toward Character 1.")
    raw["execution"]["directions"]["event_004"] = (
        "Character 2 still holds the apple and moves half a step closer to "
        "Character 1 in a blue coat.")
    out = mods["builder"].MiniMaxH3LoopPromptGeneratorV3().generate_v3(raw)
    summary = out[1]
    assert "critic" in summary
    assert "连续性复核" in summary


# --------------------------------------------------------------------------- #
# the legacy board (V1/V2 auto-storyboard) gets the same second reader
# --------------------------------------------------------------------------- #
def test_the_legacy_board_also_gets_a_repair_round(lg):
    from .test_h3_loop_prompt_generator import ScriptedConnector, _storyboard_json
    conn = ScriptedConnector([_storyboard_json(2), _storyboard_json(2)],
                             critic_replies=[
                                 critic_reply([{"severity": "critical", "path": "shots[1]",
                                                "issue": "the prop changed hands",
                                                "fix": "keep the bowl with the first actor"}]),
                                 NO_FINDINGS])
    enhancer = lg.H3LoopPromptEnhancer(conn)
    shots, warnings = enhancer._auto_storyboard(
        "a courtyard in summer", 2, 20, "cinematic-story", "en", seed=1)
    assert len(shots) == 2
    assert len(conn.critic_calls) == 2
    assert "critic" in conn.calls[1][-1]["content"].lower()


def test_the_legacy_critic_can_be_switched_off(lg):
    from .test_h3_loop_prompt_generator import ScriptedConnector, _storyboard_json
    conn = ScriptedConnector([_storyboard_json(1)])
    enhancer = lg.H3LoopPromptEnhancer(conn)
    enhancer.critic_enabled = False  # what the node's widget does
    shots, _warnings = enhancer._auto_storyboard(
        "a courtyard in summer", 1, 20, "cinematic-story", "en", seed=1)
    assert len(shots) == 1
    assert conn.critic_calls == []
