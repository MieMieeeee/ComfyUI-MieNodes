# -*- coding: utf-8 -*-
"""Tests for the authored asset roster on the H3 Loop prompt generator.

The roster is the contract between the asset library and this pipeline: a
character's name, appearance, voice and picture slot arrive as DATA. The
pipeline's job is then to stop guessing — no vision caption for a roster
slot, no CAST line invented from a name, no timbre guessed from a gender
keyword.

Headline assertion (test_roster_skips_the_vision_caption_entirely): a
fully-rostered ref2va run makes ZERO multimodal calls. Everything else
locks the edges: mixed batches, reconciliation, voice priority, and
byte-identical output when the roster is empty.
"""
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
        LLM_DIR / "minimax_h3_storyboard_prompts.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )


@pytest.fixture(scope="module")
def lg(lp):
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )


# --------------------------------------------------------------------------- #
# Roster parsing (pure)
# --------------------------------------------------------------------------- #
def test_parse_asset_roster_orders_by_slot(lp):
    entries = lp.parse_asset_roster(
        json.dumps(
            [
                {"name": "Bob", "kind": "character",
                 "appearance": " a man ", "voice": "low", "slot": 2},
                {"name": "Alice", "kind": "character",
                 "appearance": "a woman", "voice": "clear", "slot": 1},
                {"name": "Alley", "kind": "scene",
                 "description": "a rainy alley", "slot": 3},
            ]
        )
    )
    assert [e["name"] for e in entries] == ["Alice", "Bob", "Alley"]
    assert entries[0]["appearance"] == "a woman"  # stripped
    assert lp.roster_appearance_by_slot(entries) == {
        1: "a woman", 2: "a man", 3: "a rainy alley",
    }
    # Scenes never feed the identity pin or the voice sheet.
    assert lp.roster_bindings(entries) == [("Alice", 1), ("Bob", 2)]
    assert lp.roster_voices(entries) == {"Alice": "clear", "Bob": "low"}
    assert lp.roster_scene_slots(entries) == {3}


def test_parse_asset_roster_empty_is_not_an_error(lp):
    assert lp.parse_asset_roster("") == []
    assert lp.parse_asset_roster("   \n ") == []


@pytest.mark.parametrize(
    "payload,message",
    [
        ("{ not json", "not valid JSON"),
        ('{"name": "x"}', "must be a JSON list"),
        ('["x"]', "must be an object"),
        ('[{"kind": "character", "appearance": "a", "voice": "v", "slot": 1}]',
         "non-empty 'name'"),
        ('[{"name": "x", "voice": "v", "slot": 1, "kind": "character"}]',
         "non-empty 'appearance'"),
        ('[{"name": "x", "appearance": "a", "slot": 1, "kind": "character"}]',
         "non-empty 'voice'"),
        ('[{"name": "x", "appearance": "a", "kind": "scene"}]',
         "non-empty 'description'"),
        ('[{"name": "x", "appearance": "a", "voice": "v", "slot": 1}]',
         "'kind' to be"),
        ('[{"name": "x", "appearance": "a", "voice": "v", "slot": 1, "kind": "prop"}]',
         "'kind' to be"),
        ('[{"name": "x", "kind": "character", "appearance": "a", "voice": "v"}]',
         "integer 'slot'"),
        ('[{"name": "x", "kind": "character", "appearance": "a", "voice": "v", "slot": "1"}]',
         "integer 'slot'"),
        ('[{"name": "x", "kind": "character", "appearance": "a", "voice": "v", "slot": 0}]',
         "1..9"),
        ('[{"name": "x", "kind": "character", "appearance": "a", "voice": "v", "slot": 10}]',
         "1..9"),
        (
            '[{"name": "x", "kind": "character", "appearance": "a", "voice": "v", "slot": 1},'
            ' {"name": "x", "kind": "character", "appearance": "b", "voice": "w", "slot": 2}]',
            "twice",
        ),
        (
            '[{"name": "x", "kind": "scene", "description": "a", "slot": 1},'
            ' {"name": "y", "kind": "scene", "description": "b", "slot": 1}]',
            "slot 1 to both",
        ),
    ],
)
def test_parse_asset_roster_rejects_bad_shapes(lp, payload, message):
    with pytest.raises(ValueError, match=message):
        lp.parse_asset_roster(payload)


def test_roster_block_names_pictures_but_never_subjects(lp):
    """<Subject N> numbering belongs to _identity_subject_slots and the
    reference directive; restating it here would hand the model two
    numbering systems to reconcile."""
    entries = lp.parse_asset_roster(
        json.dumps([
            {"name": "Alice", "kind": "character", "appearance": "a woman",
             "voice": "clear", "slot": 1},
            {"name": "Alley", "kind": "scene", "description": "a rainy alley",
             "slot": 2},
        ])
    )
    block = lp.build_asset_roster_block(entries)
    assert "<Picture 1> is Alice: a woman" in block
    assert "<Picture 2> is Alley (scene): a rainy alley" in block
    assert "ENVIRONMENT reference" in block
    assert "<Subject" not in block
    # A scene line does not inherit the character fidelity rules.
    char_rules = block.split("Scene roster")[0]
    assert "wardrobe" in char_rules
    assert "wardrobe" not in block.split("Scene roster")[1]


def test_roster_block_is_empty_without_entries(lp):
    assert lp.build_asset_roster_block([]) == ""


# --------------------------------------------------------------------------- #
# Generator E2E helpers
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


def _vision_call_count(conn):
    """How many recorded calls carried an image (a multimodal turn)."""
    total = 0
    for messages in conn.calls:
        blob = json.dumps(messages, ensure_ascii=False, default=str)
        if "image_url" in blob:
            total += 1
    return total


def _fake_images(n=1):
    return np.zeros((n, 4, 4, 3), dtype=np.float32)


ENV_CAPTION = (
    "A greenhouse interior at dusk, condensation on the glass roof, rusted "
    "shelving packed with seedling trays, wet concrete floor."
)
CAPTION = (
    "A woman in a yellow courier jacket stands at a greenhouse door, "
    "short dark hair, warm brown eyes, canvas messenger bag."
)
EXTRACT_DIALOGUE_EMPTY = json.dumps({"turns": []})
# The semantic-facts gate runs only on DIALOGUE boards, so a narration
# concept spends no extra call even when a roster is present.
SEMANTIC_FACTS = json.dumps(
    {"turns": [], "layout": [], "bindings": []}, ensure_ascii=False
)
PREFIX_REPLY = "Always the same young woman in a Jiangnan courtyard at high summer."


def _storyboard(n=1):
    """The storyboard LLM answers with a bare JSON array of shots."""
    shots = []
    for i in range(1, n + 1):
        shots.append(
            {
                "id": "scene_%02d" % i,
                "description": "Shot %d description." % i,
                "shot_type": "medium_shot",
                "camera_movement": "slow_push_in",
                "transition_in": "fade_from_black" if i == 1 else "hard_cut",
                "duration_seconds": 10,
                "narrative_beat": "establish",
                "characters": ["Alice"],
                "props": [],
                "notes": "",
            }
        )
    return json.dumps(shots, ensure_ascii=False)


def _ref2va_scene1_reply():
    return (
        "subject_definitions:\n"
        "<Picture 1> defines <Subject 1>, the courier waiting by the door.\n"
        "\n"
        "summary:\n"
        "[reference generation] The courier waits.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain face, jacket.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] The courier stands still and looks toward the door.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, quiet room tone.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _ref2va_environment_second_reply():
    return (
        "subject_definitions:\n"
        "<Picture 1> defines <Subject 1>, the courier waiting by the door.\n"
        "<Picture 2> defines the greenhouse interior (environment).\n"
        "\n"
        "summary:\n"
        "[reference generation] The courier waits inside the greenhouse.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain face, jacket.\n"
        "<Picture 2>: reference - environment where it happens.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] The courier stands among the seedling trays.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, quiet room tone.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _ref2va_two_identity_reply():
    return (
        "subject_definitions:\n"
        "<Picture 1> defines <Subject 1>, the courier waiting by the door.\n"
        "<Picture 2> defines <Subject 2>, the greenhouse keeper.\n"
        "\n"
        "summary:\n"
        "[reference generation] Two people are in the greenhouse.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain face, jacket.\n"
        "<Subject 2>: fully_preserved - retain face, apron.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] The courier stands among the seedling trays.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, quiet room tone.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _t2va_scene_reply():
    return (
        "integrated_multimodal_description:\n"
        "[Shot 1] A woman waits by a greenhouse door in the rain.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _prefix_reply_with_cast_and_voice(voice_line):
    return (
        "Hand-drawn 2D animation, warm greenhouse interior at dusk.\n"
        "\n"
        "CAST:\n"
        "Alice: a young woman with short dark hair, wearing a navy coat.\n"
        "\n"
        "VOICE:\n"
        "Alice: %s\n" % voice_line
    )


REF2VA = "ref2va - 参考图(N张/全场景)"
T2VA = "t2va - 文生视频链(默认)"
NO_CACHE = "no_cache - 禁用缓存"

ALICE = {
    "name": "Alice",
    "kind": "character",
    "appearance": (
        "A young woman with short dark hair and warm brown eyes, wearing a "
        "yellow courier jacket and a canvas messenger bag."
    ),
    "voice": "adult female, clear and slightly warm, mid register, unhurried",
    "slot": 1,
}
STORE = {
    "name": "Greenhouse",
    "kind": "scene",
    "description": "A greenhouse interior at dusk with a wet concrete floor.",
    "slot": 2,
}
GUESSED_VOICE = "adult female, bright and slightly husky, mid register, fast"
# A canonical 角色：台词 board parses deterministically. The roster opens
# the semantic-facts gate on dialogue boards, so that reply is spent first.
DIALOGUE_BOARD = "Alice：I'm here." + chr(10) + "Alice：I'm still here."


def _roster(entries):
    return json.dumps(entries, ensure_ascii=False)


def _run(lg, conn, **kwargs):
    return lg.H3LoopPromptEnhancer(conn)(**kwargs)


def _shot_text(out, index=0):
    return "\n".join(json.loads(out["plan_json"])["shots"][index]["prompt"])


def _shot_prompts(conn):
    """The user messages of every per-shot LLM call.

    The node hands the model the authored description; what the model then
    writes is the model's job. Asserting on the prompt is the honest
    contract — a fake connector cannot be made to "copy" anything.
    """
    return [
        json.dumps(m, ensure_ascii=False)
        for m in conn.calls
        if "Write this clip's prompt now" in json.dumps(m, ensure_ascii=False)
    ]


# --------------------------------------------------------------------------- #
# The headline: a fully-rostered run makes no vision call at all
# --------------------------------------------------------------------------- #
def test_roster_skips_the_vision_caption_entirely(lg):
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_scene1_reply(),
    ])
    out = _run(
        lg, conn,
        user_input="Alice waits by the greenhouse door.",
        reference_mode=REF2VA,
        images=_fake_images(1),
        caption_mode=NO_CACHE,
        asset_roster=_roster([ALICE]),
        seed=1,
    )
    assert _vision_call_count(conn) == 0, (
        "a roster-covered slot must not spend a multimodal call: its "
        "description is already authored"
    )
    assert len(json.loads(out["plan_json"])["shots"]) == 1
    # The authored description is handed to the model verbatim...
    prompts = _shot_prompts(conn)
    assert prompts
    for blob in prompts:
        assert ALICE["appearance"] in blob
    # ...and the preflight report says where that description came from.
    assert "authored description" in out["summary"]


def test_roster_appearance_replaces_the_manifest_caption(lg):
    """The prefix LLM invents appearance from the NAME; the identity pin
    overwrites it with the slot's description. With a roster that
    description is authored, so the pin is what the model writes from."""
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        _prefix_reply_with_cast_and_voice(GUESSED_VOICE),
        _ref2va_scene1_reply(),
    ])
    out = _run(
        lg, conn,
        user_input="Alice waits by the greenhouse door.",
        reference_mode=REF2VA,
        images=_fake_images(1),
        caption_mode=NO_CACHE,
        asset_roster=_roster([ALICE]),
        seed=1,
    )
    prompts = _shot_prompts(conn)
    assert prompts
    for blob in prompts:
        assert ALICE["appearance"] in blob
        assert "navy coat" not in blob  # the name-derived CAST line loses
    assert "identity pin" in out["summary"]
    assert "character asset" in out["summary"]


def test_mixed_batch_captions_only_the_slots_the_roster_does_not_cover(lg):
    conn = ScriptedConnector([
        ENV_CAPTION,  # slot 2 is outside the roster -> one vision call
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_two_identity_reply(),  # slot 2 captioned => identity role
    ])
    out = _run(
        lg, conn,
        user_input="Alice waits in the greenhouse.",
        reference_mode=REF2VA,
        images=_fake_images(2),
        caption_mode=NO_CACHE,
        asset_roster=_roster([ALICE]),  # only slot 1
        seed=1,
    )
    assert _vision_call_count(conn) == 1
    blob = _shot_prompts(conn)[0]
    assert ALICE["appearance"] in blob
    assert ENV_CAPTION[:40] in blob


def test_roster_slot_beyond_the_batch_fails_loudly(lg):
    conn = ScriptedConnector([])
    with pytest.raises(RuntimeError, match="slot 2, but only 1 reference slot"):
        _run(
            lg, conn,
            user_input="Alice waits.",
            reference_mode=REF2VA,
            images=_fake_images(1),  # only one portrait wired
            caption_mode=NO_CACHE,
            asset_roster=_roster([ALICE, dict(ALICE, name="Bob", slot=2)]),
            seed=1,
        )
    assert conn.calls == []


def test_roster_without_images_fails_loudly(lg):
    conn = ScriptedConnector([])
    with pytest.raises(RuntimeError, match="images socket is empty"):
        _run(
            lg, conn,
            user_input="Alice waits.",
            reference_mode=REF2VA,
            images=None,
            asset_roster=_roster([ALICE]),
            seed=1,
        )
    assert conn.calls == []


def test_roster_under_t2va_is_reported_not_silently_dropped(lg):
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _t2va_scene_reply(),
    ])
    out = _run(
        lg, conn,
        user_input="Alice waits by a greenhouse door.",
        reference_mode=T2VA,
        images=_fake_images(1),
        asset_roster=_roster([ALICE]),
        seed=1,
    )
    assert "t2va has no reference slots" in out["summary"]


def test_malformed_roster_json_raises_before_any_llm_call(lg):
    conn = ScriptedConnector([])
    with pytest.raises(ValueError, match="not valid JSON"):
        _run(
            lg, conn,
            user_input="x",
            asset_roster="{ nope",
            reference_mode=REF2VA,
            images=_fake_images(1),
            seed=1,
        )
    assert conn.calls == []


# --------------------------------------------------------------------------- #
# Voice priority
# --------------------------------------------------------------------------- #
def test_roster_voice_outranks_the_prefix_llm_sheet(lg):
    conn = ScriptedConnector([
        SEMANTIC_FACTS,
        _prefix_reply_with_cast_and_voice(GUESSED_VOICE),
        _ref2va_scene1_reply(),
        _ref2va_scene1_reply(),  # the shot gate retries a malformed reply
    ])
    out = _run(
        lg, conn,
        # A canonical 角色：台词 board parses deterministically, so the LLM
        # calls are the facts (which the roster opens the gate for),
        # storyboard, prefix and the shot.
        user_input=DIALOGUE_BOARD,
        reference_mode=REF2VA,
        images=_fake_images(1),
        caption_mode=NO_CACHE,
        asset_roster=_roster([ALICE]),
        seed=1,
    )
    text = _shot_text(out)
    assert ALICE["voice"] in text
    assert "slightly husky" not in text
    assert "asset voice" in out["summary"]


def test_speaker_missing_from_the_roster_falls_back_silently(lg):
    conn = ScriptedConnector([
        SEMANTIC_FACTS,
        _prefix_reply_with_cast_and_voice("adult male, low and gravelly, slow"),
        _ref2va_scene1_reply(),
        _ref2va_scene1_reply(),
    ])
    out = _run(
        lg, conn,
        user_input=DIALOGUE_BOARD,
        reference_mode=REF2VA,
        images=_fake_images(1),
        caption_mode=NO_CACHE,
        asset_roster=_roster([dict(ALICE, name="Bob", voice="a different voice")]),
        seed=1,
    )
    assert "asset voice" not in out["summary"]
    assert "gravelly" in _shot_text(out)


# --------------------------------------------------------------------------- #
# Regression anchor: an empty roster changes nothing
# --------------------------------------------------------------------------- #
def test_empty_roster_is_byte_identical_to_the_pre_roster_path(lg):
    # No roster -> the vision captioner runs, so its reply leads the script.
    replies = [
        CAPTION,
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_scene1_reply(),
    ]
    plain = ScriptedConnector(list(replies))
    empty = ScriptedConnector(list(replies))
    shared = dict(
        user_input="Alice waits by the greenhouse door.",
        reference_mode=REF2VA,
        images=_fake_images(1),
        caption_mode=NO_CACHE,
        seed=7,
    )
    out_plain = _run(lg, plain, **shared)
    out_empty = _run(lg, empty, **shared, asset_roster="")
    assert out_empty["plan_json"] == out_plain["plan_json"]
    assert out_empty["summary"] == out_plain["summary"]
    # The template must not grow an empty roster section.
    shot_calls = [
        m for m in empty.calls
        if "Write this clip's prompt now" in json.dumps(m, ensure_ascii=False)
    ]
    assert shot_calls
    for messages in shot_calls:
        assert "Authored character roster" not in json.dumps(
            messages, ensure_ascii=False
        )


def test_scene_entry_in_roster_reaches_the_shot_prompt(lg):
    """A scene asset in the roster is where the story happens: its
    description reaches the clip prompt and its slot is an environment
    reference (t2va has no picture slots, so this must use ref2va)."""
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_environment_second_reply(),
    ])
    out = _run(
        lg, conn,
        user_input="Alice waits in the greenhouse.",
        reference_mode=REF2VA,
        images=_fake_images(2),
        caption_mode=NO_CACHE,
        asset_roster=_roster([ALICE, STORE]),
        seed=1,
    )
    prompts = _shot_prompts(conn)
    assert prompts
    for blob in prompts:
        assert STORE["description"] in blob
        assert "ENVIRONMENT reference" in blob
    # Zero vision calls: the scene slot is authored too.
    assert _vision_call_count(conn) == 0
    # The scene never feeds the identity machinery.
    assert "identity pin" not in out["summary"]
    assert "asset voice" not in out["summary"]


# --------------------------------------------------------------------------- #
# V2 node shell
# --------------------------------------------------------------------------- #
def test_v2_node_extends_v1_widgets_with_socket_slots(lp, lg):
    v1 = lg.MiniMaxH3LoopPromptGenerator.INPUT_TYPES()
    v2 = lg.MiniMaxH3LoopPromptGeneratorV2.INPUT_TYPES()
    assert set(v1) == set(v2)
    assert set(v1["required"]) == set(v2["required"])
    for key, spec in v1["optional"].items():
        assert v2["optional"][key] == spec, key
    opt = v2["optional"]
    # Slot SOCKETS (no selection widgets left on the generator).
    assert "asset_roster" not in opt
    assert opt["character_1"][0] == "MIE_ASSET"
    assert opt["scene_1"][0] == "MIE_ASSET"
    assert "character_9" in opt and "scene_9" in opt
    # Outputs EXTEND V1's by one: plan_json/summary/board_kind keep their
    # slots (the Production Plan wiring is unchanged) and the appended
    # `images` batch feeds the video chain.
    assert (
        lg.MiniMaxH3LoopPromptGeneratorV2.RETURN_TYPES[:3]
        == lg.MiniMaxH3LoopPromptGenerator.RETURN_TYPES
    )
    assert lg.MiniMaxH3LoopPromptGeneratorV2.RETURN_TYPES[3] == "IMAGE"
    assert lg.MiniMaxH3LoopPromptGeneratorV2.RETURN_NAMES[3] == "images"



# --------------------------------------------------------------------------- #
# Template regression: an empty roster renders the PRE-ROSTER template
# --------------------------------------------------------------------------- #
_B1DF1F6_SHOT_USER_TEMPLATE = """Concept (whole production):
{concept}

Shared identity / style prefix (binding for every clip):
{prompt_prefix}

Spatial layout for THIS clip (binding — must match the prefix above and must be preserved exactly from any prior clip that named these positions):
{spatial_layout_directive}

{tempo_directive}

{genre_advice}

{continuation_block}

Cast on screen in THIS clip (identity binding — weave each member's exact identity into the description, using these exact names):
{cast_block}

Cast block rule: copy each named character's CAST identity line
VERBATIM into this clip's integrated_multimodal_description. Do not
paraphrase ("navy wool coat" must not become "dark jacket"); the model
treats paraphrases as different subjects. Wardrobe is described as a
`color + material + item` triple (e.g. `mustard-yellow waterproof
jacket, dark teal trousers, charcoal helmet, olive messenger bag`) —
see official H3 example_workflows `I2V Normal - MiniMax H3 0.6.json`
shot 0 for the canonical wording.

Storyboard entry for THIS clip (clip {clip_index} of {clip_count}):
{shot_json}

{reference_directive}

Reference manifest (binding):
{manifest_digest}

Dialogue lines for this shot (turn {turn_index}, speaker {turn_speaker}):
{dialogue_lines_block}
{speaker_id_directive}
Clip duration: {duration_seconds:.1f} seconds (exact raw length on the H3 grid). Output language: {language_name}.
Language policy: narrative prose follows {language_name}. {dialogue_language_policy}

{pacing_directive}

Write this clip's prompt now: exactly the three bare-header sections, nothing else.
"""


def test_empty_roster_template_is_byte_identical_to_pre_roster(lp, monkeypatch):
    """The shot template's ONLY roster hook is the former blank line.

    Substituting an empty roster block must yield the b1df1f6 template
    exactly. This is the honest version of the "V1 unchanged" claim: the
    template rendering is locked byte-for-byte; what IS different for all
    generations (documented in the guide) is two shared system prompts'
    wording.
    """
    from pathlib import Path

    current = Path(
        lp.__file__ if hasattr(lp, "__file__") else "."
    )
    template_path = (
        PROJECT_DIR / "nodes" / "llm" / "prompts" / "h3_loop" / "shot_user_template.txt"
    )
    new_text = template_path.read_text(encoding="utf-8")
    substituted = new_text.replace("{asset_roster_block}", "")
    assert substituted == _B1DF1F6_SHOT_USER_TEMPLATE, (
        "the shot template changed beyond the single roster hook; V1's "
        "prompt shape for roster-less runs is no longer byte-identical"
    )


def test_v2_consumes_batch_roster_with_v2_counts(lp, lg):
    """LoadAssetBatch's V2 roster_json (image_count / audio_count /
    character description mirror) feeds the generator UNCHANGED: counts are
    preserved on the parsed entries, the run still makes zero vision calls,
    and nothing about the extra fields disturbs the label policy."""
    roster = json.dumps(
        [
            {"name": "Alice", "kind": "character", "slot": 1,
             "appearance": "A courier in a yellow jacket.", "voice": "adult female, clear",
             "description": "A courier in a yellow jacket.",
             "image_count": 3, "audio_count": 2},
            {"name": "Store", "kind": "scene", "slot": 2,
             "description": "A midnight convenience store.", "image_count": 2},
        ],
        ensure_ascii=False,
    )
    entries = lp.parse_asset_roster(roster)
    assert entries[0]["image_count"] == 3 and entries[0]["audio_count"] == 2
    assert entries[1]["image_count"] == 2 and "audio_count" not in entries[1]
    assert entries[0]["description"] == "A courier in a yellow jacket."

    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_environment_second_reply(),
    ])
    alice = dict(ALICE)
    alice["appearance"] = "A courier in a yellow jacket."
    out = _run(
        lg, conn,
        user_input="Alice waits in the store.",
        reference_mode=REF2VA,
        images=_fake_images(2),
        caption_mode=NO_CACHE,
        asset_roster=roster,
        seed=1,
    )
    assert _vision_call_count(conn) == 0
    prompts = _shot_prompts(conn)
    assert prompts
    for blob in prompts:
        assert "A courier in a yellow jacket." in blob
        assert "A midnight convenience store." in blob


# --------------------------------------------------------------------------- #
# Generator dropdown slots (2026-10-10 scheme): character_1 / scene_1 read the
# fixed library themselves — no images socket, no roster JSON, no vision call.
# --------------------------------------------------------------------------- #
def _slot_lib(tmp_path, monkeypatch):
    """One simple character (1 img / 1 audio) + one scene in a tmp fixed lib."""
    import wave as _wave

    import numpy as np
    from PIL import Image

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nodes.assets import library as asset_library

    monkeypatch.setenv(asset_library.LIBRARY_DIR_ENV, str(tmp_path))

    def _png(color):
        buf = __import__("io").BytesIO()
        Image.new("RGB", (64, 96), color).save(buf, format="PNG")
        return buf.getvalue()

    def _wav(freq):
        t = np.arange(1600) / 16000
        data = (np.sin(2 * np.pi * freq * t) * 20000).astype("<i2")
        buf = __import__("io").BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(data.tobytes())
        return buf.getvalue()

    asset_library.save_character(
        str(tmp_path), "Alice", "A courier in a yellow jacket.", "adult female, clear",
        [_png((200, 40, 40))], [_wav(440.0)],
    )
    asset_library.save_scene(
        str(tmp_path), "Store", "A midnight convenience store.", [_png((40, 40, 200))]
    )
    return asset_library


def test_generator_socket_bundles_full_path(lp, lg, tmp_path, monkeypatch):
    """Load nodes' asset outputs wire straight into the generator slots:
    no library reads at generation time, no vision calls, slot order =
    characters then scenes, and the images output carries the batch."""
    lib = _slot_lib(tmp_path, monkeypatch)
    from nodes.assets.nodes import LoadCharacterAssetV2, LoadSceneAssetV2

    char_bundle = LoadCharacterAssetV2().load_character("", "Alice")[5]
    scene_bundle = LoadSceneAssetV2().load_scene("", "Store")[3]
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_environment_second_reply(),
    ])
    node = lg.MiniMaxH3LoopPromptGeneratorV2()
    plan_json, summary, board_kind, images, assets = node.generate_v2(
        conn,
        user_input="Alice waits in the store.",
        reference_mode=REF2VA,
        seed=1,
        character_1=char_bundle,
        scene_1=scene_bundle,
    )
    assert _vision_call_count(conn) == 0  # bundles replaced the captioner
    assert images is not None and images.shape[0] == 2  # char then scene
    assert [b["name"] for b in assets] == ["Alice", "Store"]  # slot order
    prompts = _shot_prompts(conn)
    assert prompts
    for blob in prompts:
        assert "A courier in a yellow jacket." in blob
        assert "A midnight convenience store." in blob
    assert "Picture 1" in prompts[0]


def test_generator_socket_rules(lg, tmp_path, monkeypatch):
    lib = _slot_lib(tmp_path, monkeypatch)
    from nodes.assets.nodes import LoadCharacterAssetV2, LoadSceneAssetV2

    node = lg.MiniMaxH3LoopPromptGeneratorV2()
    bundle = LoadCharacterAssetV2().load_character("", "Alice")[5]

    # A non-bundle value (e.g. a plain string) is rejected with wiring help.
    with pytest.raises(RuntimeError, match="not a character asset bundle"):
        node.generate_v2(ScriptedConnector([]), user_input="x",
                         reference_mode=REF2VA, seed=1, character_1="Alice")
    # Same asset in two slots -> error.
    with pytest.raises(RuntimeError, match="more than one slot"):
        node.generate_v2(ScriptedConnector([]), user_input="x",
                         reference_mode=REF2VA, seed=1,
                         character_1=bundle, character_2=bundle)
    # A scene bundle in a character slot -> kind mismatch error.
    scene_bundle = LoadSceneAssetV2().load_scene("", "Store")[3]
    with pytest.raises(RuntimeError, match="not a character asset bundle"):
        node.generate_v2(ScriptedConnector([]), user_input="x",
                         reference_mode=REF2VA, seed=1, character_1=scene_bundle)


# --------------------------------------------------------------------------- #
# Camera discipline (2026-10-10 ruling): close-up switching is an OPTION, and
# the concept's locked-camera prose must not lose to the storyboard's
# SHOT-SIZE VARIETY rule.
# --------------------------------------------------------------------------- #
def test_camera_discipline_pure_helpers(lp):
    assert lp.parse_camera_discipline("locked - 固定机位(构图全场不变)") == "locked"
    assert lp.parse_camera_discipline("no_closeups - 禁特写(中景以上)") == "no_closeups"
    assert lp.parse_camera_discipline("auto - 自动(分镜自由构图)") == "auto"
    assert lp.parse_camera_discipline("garbage") == "auto"
    directive = lp.build_camera_discipline_directive("locked")
    assert "LOCKED OFF" in directive and "no close-ups" in directive.lower()
    assert lp.build_camera_discipline_directive("auto") == ""

    shots = [
        {"id": "scene_01", "shot_type": "close_up", "camera_movement": "slow_push_in"},
        {"id": "scene_02", "shot_type": "two_shot", "camera_movement": "tracking_left"},
        {"id": "scene_03", "shot_type": "medium_close_up", "camera_movement": "static_lockoff"},
    ]
    lp.apply_camera_discipline(shots, "locked")
    assert all(sh["camera_movement"] == "static_lockoff" for sh in shots)
    assert all(sh["shot_type"] not in ("close_up", "medium_close_up") for sh in shots)
    assert shots[1]["shot_type"] == "two_shot"  # non-closeup framing survives
    assert "camera locked" in shots[0]["notes"]

    shots2 = [
        {"id": "scene_01", "shot_type": "extreme_close_up", "camera_movement": "handheld"},
        {"id": "scene_02", "shot_type": "two_shot", "camera_movement": "crane_up"},
    ]
    lp.apply_camera_discipline(shots2, "no_closeups")
    assert shots2[0]["shot_type"] == "medium_shot"
    assert shots2[0]["camera_movement"] == "handheld"  # camera stays free
    assert shots2[1]["shot_type"] == "two_shot"

    shots3 = [{"id": "s", "shot_type": "close_up"}]
    lp.apply_camera_discipline(shots3, "auto")
    assert shots3[0]["shot_type"] == "close_up"  # auto: untouched


def test_generator_camera_discipline_clamps_storyboard(lg):
    """The storyboard LLM returns close-ups; locked widens them and locks
    the camera, and the summary reports the clamp."""
    closeup_board = json.dumps([
        {"id": "scene_01", "description": "d1", "shot_type": "close_up",
         "camera_movement": "slow_push_in", "duration_seconds": 6,
         "characters": ["孙女"], "props": []},
        {"id": "scene_02", "description": "d2", "shot_type": "extreme_close_up",
         "camera_movement": "handheld", "duration_seconds": 6,
         "characters": ["孙女"], "props": []},
    ], ensure_ascii=False)
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        closeup_board,
        PREFIX_REPLY,
        _ref2va_scene1_reply(),
        _ref2va_scene1_reply(),
    ])
    out = _run(
        lg, conn,
        user_input="孙女和爷爷坐在客厅沙发上，镜头固定，对称双人中景。",
        reference_mode=REF2VA,
        images=_fake_images(1),
        caption_mode=NO_CACHE,
        asset_roster=_roster([ALICE]),
        camera_discipline="locked - 固定机位(构图全场不变)",
        seed=1,
    )
    assert "camera discipline locked: 2 shot(s) widened" in out["summary"]
    # The clamp shows up in the storyboard user turn the LLM saw? The shot
    # turn's shot_json carries the CLAMPED types (downstream inherits them).
    shots = json.loads(out["plan_json"])["shots"]
    prompts = [ln for s in shots for ln in s["prompt"]]
    assert not any("close_up" in ln for ln in prompts)


def test_camera_discipline_widget_present(lg):
    spec = lg.MiniMaxH3LoopPromptGenerator.INPUT_TYPES()["optional"]
    assert "camera_discipline" in spec
    assert spec["camera_discipline"][0][0].startswith("auto")
    v2 = lg.MiniMaxH3LoopPromptGeneratorV2.INPUT_TYPES()["optional"]
    assert "camera_discipline" in v2  # derived from V1


def test_generator_mixed_picture_sizes_are_legal(lp, lg):
    """H3 takes each reference on its OWN socket and resizes independently
    (match/max), so mixed sizes across ASSETS must not fail the run — only
    the convenience images batch steps aside with a clear note."""
    import torch

    alice = dict(ALICE)
    alice["description"] = alice.get("description", alice["appearance"])
    alice["image"] = torch.rand(1, 96, 48, 3)   # portrait
    scene_bundle = {
        "kind": "scene", "name": "Store",
        "description": "A midnight convenience store.",
        "image": torch.rand(1, 48, 96, 3),       # landscape
    }
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_environment_second_reply(),
    ])
    node = lg.MiniMaxH3LoopPromptGeneratorV2()
    plan_json, summary, board_kind, images, assets = node.generate_v2(
        conn,
        user_input="Alice waits in the store.",
        reference_mode=REF2VA,
        seed=1,
        character_1=alice,
        scene_1=scene_bundle,
    )
    assert images is None  # batch off — but the plan built fine
    assert "picture sizes differ" in summary
    assert "picture_1" in summary  # the note says how to wire
    assert _vision_call_count(conn) == 0


def test_asset_to_slots_carries_native_shapes():
    """Asset → Slots never cats: each picture output keeps its asset's own
    [1,H,W,3] — mixed-size rosters work there natively."""
    import torch

    from nodes.assets.nodes import AssetToSlots, png_bytes_list_to_image_tensor
    from tests.test_asset_library import make_png

    tall = png_bytes_list_to_image_tensor([make_png((1, 2, 3), 8, 16)])
    wide = png_bytes_list_to_image_tensor([make_png((4, 5, 6), 16, 8)])
    outs = AssetToSlots().split([
        {"kind": "character", "name": "A", "description": "d", "voice": "v",
         "image": tall, "audio": None},
        {"kind": "scene", "name": "S", "description": "d", "image": wide},
    ])
    assert outs[1].shape == (1, 16, 8, 3)   # tall native
    assert outs[2].shape == (1, 8, 16, 3)   # wide native


def _wav_bytes():
    import wave

    import numpy as np

    t = np.arange(800) / 16000
    data = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    buf = __import__("io").BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data.tobytes())
    return buf.getvalue()


def test_camera_discipline_violation_detector(lp):
    # 2026-10-11 live failure on the LAN box: locked board, every shot said
    # "The camera holds still", yet Shot 2 cut to a tight close-up.
    bad = ("[Shot 2] At 00:02.961, the camera cuts to a tight close-up of "
           "爷爷. The camera holds still.")
    assert "close-up framing" in lp.camera_discipline_violation(bad, "locked")
    assert "camera move/cut" in lp.camera_discipline_violation(bad, "locked")
    assert lp.camera_discipline_violation(bad, "no_closeups")  # close-up only
    # negations the directives themselves inject must NOT fire
    ok = ("No shot/reverse-shot cutaways, no close-ups, no slow motion. "
          "The camera holds still; the courtyard walls stay behind them.")
    assert lp.camera_discipline_violation(ok, "locked") == ""
    # free camera wording stays legal under auto / no_closeups (no close-up)
    ok2 = "The camera pans to follow her across the room."
    assert lp.camera_discipline_violation(ok2, "auto") == ""
    assert lp.camera_discipline_violation(ok2, "no_closeups") == ""
    assert lp.camera_discipline_violation(ok2, "locked")  # locked bans pans
    # Chinese prose: 切到/特写 fire, negated 特写 does not
    assert lp.camera_discipline_violation("镜头切到爷爷的特写。", "locked")
    assert "爷爷特写" not in lp.camera_discipline_violation("无特写，全程固定机位。", "locked")


def test_reaction_cuts_clamped_under_locked(lp, lg):
    # 2026-10-11 ordering hole: reaction cuts are inserted AFTER the clamp,
    # so their hard-coded close_up sailed through a locked board. The clamp
    # now runs after the last board edit.
    warnings = []
    shots = [
        {"id": "scene_01", "description": "A speaks.", "shot_type": "medium_shot",
         "camera_movement": "static_lockoff", "characters": ["A", "B"],
         "narrative_beat": "", "props": [], "notes": "",
         "duration_seconds": 5.0, "_line_speakers": ["A"]},
        {"id": "scene_02", "description": "B answers.", "shot_type": "medium_shot",
         "camera_movement": "static_lockoff", "characters": ["A", "B"],
         "narrative_beat": "", "props": [], "notes": "",
         "duration_seconds": 5.0, "_line_speakers": ["B"]},
    ]
    out = lg._insert_reaction_cuts(shots, 1, warnings)
    assert out[1]["shot_type"] == "close_up"  # inserted raw...
    lp.apply_camera_discipline(out, "locked")  # ...and the clamp reaches it
    assert out[1]["shot_type"] == "medium_shot"
    assert out[1]["camera_movement"] == "static_lockoff"
    assert "camera locked" in out[1]["notes"]


def _ref2va_cut_violation_reply():
    # The 2026-10-11 live failure shape: shot obeys "camera holds still"
    # AND cuts to a tight close-up in the same breath.
    return (
        "subject_definitions:\n"
        "<Picture 1> defines <Subject 1>, the courier waiting by the door.\n"
        "<Picture 2> defines the greenhouse interior (environment).\n"
        "\n"
        "summary:\n"
        "[reference generation] The courier waits inside the greenhouse.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - retain face, jacket.\n"
        "<Picture 2>: reference - environment where it happens.\n"
        "\n"
        "detailed_description:\n"
        "[Shot 1] The courier stands among the seedling trays.\n"
        "The camera cuts to a tight close-up of the courier; "
        "the greenhouse walls stay behind her. The camera holds still.\n"
        "\n"
        "overall_soundscape:\n"
        "Rain on glass, quiet room tone.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def test_camera_discipline_retry_rewrites_cut_prose(lp, lg, tmp_path, monkeypatch):
    """locked board + a shot reply that cuts to a close-up: the mechanical
    guard rejects it, the corrective retry turn runs, and only the clean
    rewrite reaches the plan (2026-10-11 live hole on the LAN box)."""
    lib = _slot_lib(tmp_path, monkeypatch)
    from nodes.assets.nodes import LoadCharacterAssetV2, LoadSceneAssetV2

    char_bundle = LoadCharacterAssetV2().load_character("", "Alice")[5]
    scene_bundle = LoadSceneAssetV2().load_scene("", "Store")[3]
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_cut_violation_reply(),   # attempt 1: violating prose
        _ref2va_environment_second_reply(),  # attempt 2: clean rewrite
    ])
    node = lg.MiniMaxH3LoopPromptGeneratorV2()
    plan_json, summary, board_kind, images, assets = node.generate_v2(
        conn,
        user_input="Alice waits in the store.",
        reference_mode=REF2VA,
        camera_discipline="locked - 固定机位(构图全场不变)",
        seed=1,
        character_1=char_bundle,
        scene_1=scene_bundle,
    )
    prompts = _shot_prompts(conn)
    joined = "\n".join(prompts[0])
    assert "camera cuts to" not in joined
    assert "close-up of" not in joined
    # the corrective retry turn taught the model WHY (the violation text
    # rides in the retry turn's content)
    blob = json.dumps(conn.calls, ensure_ascii=False, default=str)
    assert "camera discipline violated" in blob
    assert "camera LOCKED" in blob


def test_fallback_shot_lines_keep_identity_bindings(lp):
    """2026-10-11 live failure: the fallback capped about at 400 chars and
    wrote bare 'Picture N:' lines with no <Subject> binding — a ref2va
    continuation that fell back lost its identity anchors (grandfather
    morphed into the girl). The fallback must be identity-complete."""
    long_about_girl = "A cute dazed girl. " + "detail " * 120  # ~900 chars
    long_about_man = "A kind elderly grandfather. " + "detail " * 110
    manifest = [
        {"slot": "Picture 1", "about": long_about_girl, "role": "identity"},
        {"slot": "Picture 2", "about": long_about_man, "role": "identity"},
        {"slot": "Picture 3", "about": "A cozy living room at night.",
         "role": "environment"},
    ]
    lines = lp.build_fallback_shot_lines(
        lp.SCHEMA_SIX, manifest=manifest,
        spatial_layout={"爷爷": "left of frame", "孙女": "right of frame"},
    )
    body = "\n".join(lines)
    # FULL about text — no 400-char cap on authored ground truth
    assert "detail detail detail" in body
    assert len([ln for ln in lines if "detail" in ln and "Picture 1" in ln
                or "Subject 1" in ln and "detail" in ln]) >= 1
    assert len(body) > 1600  # two ~800-char abouts survive whole
    # identity slots carry the binding syntax
    assert "<Subject 1> -> <Picture 1>:" in body
    assert "<Subject 2> -> <Picture 2>:" in body
    # environment stays a plain picture line, unbound
    assert "<Picture 3>: A cozy living room at night." in body
    # per-label retention lines
    assert ("<Subject 1> -> <Picture 1>: fully_preserved" in body)
    assert ("<Picture 3>: reference - use as the destination" in body)
    # layout preserved
    assert "爷爷 stays left of frame" in body


def test_fallback_shot_lines_without_manifest(lp):
    lines = lp.build_fallback_shot_lines(lp.SCHEMA_SIX)
    body = "\n".join(lines)
    assert "already established" in body
    assert "fully_preserved" in body


def test_shot_stage_survives_two_bad_replies(lp, lg, tmp_path, monkeypatch):
    """_SHOT_PARSE_RETRIES = 2: two unparseable replies then a clean one
    must NOT fall back (the fallback costs the most at this stage)."""
    lib = _slot_lib(tmp_path, monkeypatch)
    from nodes.assets.nodes import LoadCharacterAssetV2, LoadSceneAssetV2

    char_bundle = LoadCharacterAssetV2().load_character("", "Alice")[5]
    scene_bundle = LoadSceneAssetV2().load_scene("", "Store")[3]
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        "garbage — no sections at all",
        "still not a sectioned reply",
        _ref2va_environment_second_reply(),
    ])
    node = lg.MiniMaxH3LoopPromptGeneratorV2()
    plan_json, summary, board_kind, images, assets = node.generate_v2(
        conn,
        user_input="Alice waits in the store.",
        reference_mode=REF2VA,
        seed=1,
        character_1=char_bundle,
        scene_1=scene_bundle,
    )
    plan = json.loads(plan_json)
    shot_lines = "\n".join(plan["shots"][0]["prompt"])
    assert "seedling trays" in shot_lines  # the clean third reply was used
    assert "<Subject 1>" in shot_lines      # real binding, not the fallback
    # three shot-stage calls happened (2 bad + 1 good), not a fallback path
    assert len(_shot_prompts(conn)) == 3


def test_assembled_plan_has_no_mechanical_cut_or_clock(lp, lg, tmp_path, monkeypatch):
    """2026-10-11 root cause of the recurring 'camera cuts + At 00:0X.XXX'
    in every generated plan: the node-owned description assembly inserted
    a mechanical '[Shot 2] At clock, the camera cuts to a tight
    close-up' line between the two speech lines of a two-line scene —
    AFTER the per-shot validation, so the prose guards never saw it, and
    in direct violation of the shot addendum + timestamp ban + locked
    discipline. The assembly must keep one continuous held shot."""
    lib = _slot_lib(tmp_path, monkeypatch)
    from nodes.assets.nodes import LoadCharacterAssetV2, LoadSceneAssetV2

    char_bundle = LoadCharacterAssetV2().load_character("", "Alice")[5]
    scene_bundle = LoadSceneAssetV2().load_scene("", "Store")[3]
    conn = ScriptedConnector([
        EXTRACT_DIALOGUE_EMPTY,
        _storyboard(1),
        PREFIX_REPLY,
        _ref2va_environment_second_reply(),
    ])
    node = lg.MiniMaxH3LoopPromptGeneratorV2()
    plan_json, _summary, _bk, _img, _assets = node.generate_v2(
        conn,
        user_input="Alice waits in the store.",
        reference_mode=REF2VA,
        camera_discipline="locked - 固定机位(构图全场不变)",
        seed=1,
        character_1=char_bundle,
        scene_1=scene_bundle,
    )
    blob = json.dumps(json.loads(plan_json), ensure_ascii=False)
    assert "[Shot 2]" not in blob
    assert "camera cuts to" not in blob
    import re as _re
    assert not _re.search(r"\b\d{1,2}:\d{2}\.\d{3}\b", blob), \
        "no MM:SS.mmm clock anywhere in the assembled plan"
