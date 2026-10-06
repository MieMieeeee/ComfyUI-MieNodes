# -*- coding: utf-8 -*-
"""Content-contract tests for the MiniMax H3 Loop user-input enhancer.

``test_h3_loop_user_input_enhancer.py`` covers PLUMBING: block
extraction, widget forwarding, timeouts, seeds. None of those tests
would notice a reply that keeps the BEGIN/END markers and silently
rewrites the user's dialogue — which is the live 2026-10-05 failure
(「老板，你这里剩了那么多，怎么办？」shipped as 「老板，你今天剩了那么多，
要怎么处理？」).

The contract this file pins down:

* spoken lines are word-perfect data and survive verbatim;
* a rewrite that drops or paraphrases them is retried with a
  CORRECTIVE prompt, and pinned back if the retries run out — the
  node never raises over a missing word;
* a ``镜头：`` line is legal ABOVE the first spoken line and illegal
  inside the dialogue block, where it breaks the downstream parser;
* narration / action boards and free-form prose drafts are NOT subject
  to the dialogue contract (no false positives).
"""
from __future__ import annotations

import importlib.util
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
def enh():
    # Load under a private package alias. The other enhancer test file
    # loads the same source under ``_mienodes_internal.nodes.llm.*``; a
    # shared module object would make their module-level globals (the
    # scripted-reply queue) leak across files, so each file gets its
    # own instance of the code under test.
    root = "_mie_enh_content"
    _ensure_pkg(root, PROJECT_DIR)
    _ensure_pkg(f"{root}.core", PROJECT_DIR / "core")
    _load_file(f"{root}.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg(f"{root}.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg(f"{root}.nodes.llm", LLM_DIR)
    _ensure_pkg(f"{root}.nodes.llm.prompts", PROMPTS_DIR)
    _load_file(f"{root}.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py")
    _load_file(
        f"{root}.nodes.llm.dialogue_segmenter", LLM_DIR / "dialogue_segmenter.py"
    )
    _load_file(
        f"{root}.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    _load_file(
        f"{root}.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )
    return _load_file(
        f"{root}.nodes.llm.minimax_h3_loop_user_input_enhancer",
        LLM_DIR / "minimax_h3_loop_user_input_enhancer.py",
    )


@pytest.fixture(autouse=True)
def _isolate_enhancer_cache(enh, tmp_path):
    enh.clear_enhancer_cache()
    previous = enh._CACHE_ROOT_OVERRIDE
    enh._CACHE_ROOT_OVERRIDE = str(tmp_path)
    yield
    enh.clear_enhancer_cache()
    enh._CACHE_ROOT_OVERRIDE = previous


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _ScriptedConnector:
    """Returns canned replies in order; running past the script fails the
    test loudly rather than silently returning an empty reply."""

    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.timeout = 120

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        self.calls.append({
            "messages": list(messages),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if not self.replies:
            raise AssertionError("scripted connector exhausted")
        return self.replies.pop(0)

    def get_state(self):
        return "fake-state"


DRAFT_DIALOGUE = (
    "图1的黑猫：老板，你这里剩了那么多，怎么办？\n"
    "图2的白猫：倒掉。\n"
    "图1的黑猫：明天怎么办？\n"
    "图2的白猫：先做。"
)

GOOD_BLOCK = "场景设定：厨房。\n\n" + DRAFT_DIALOGUE

# The exact live failure: markers intact, two lines reworded.
PARAPHRASED_BLOCK = (
    "场景设定：厨房。\n"
    "\n"
    "图1的黑猫：老板，你今天剩了那么多，要怎么处理？\n"
    "图2的白猫：倒掉。\n"
    "图1的黑猫：那明天还做这么多吗？\n"
    "图2的白猫：先做。"
)


def reply(body: str, classification: str = "Dialogue") -> str:
    return (
        f"Classification: {classification}\n"
        "Notes for the user: ok\n"
        "\n"
        "--- BEGIN user_input ---\n"
        f"{body}\n"
        "--- END user_input ---\n"
    )


def run(enh, conn, draft, category="dialogue - 对白/对话/相声", **kw):
    kw.setdefault("cache", False)
    kw.setdefault("reference_mode", "t2va - 文生视频链(默认)")
    return enh.run_enhancer(
        conn,
        draft=draft,
        category=category,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Verbatim contract
# --------------------------------------------------------------------------- #
def test_paraphrased_line_is_pinned_on_the_first_reply(enh):
    """A paraphrase used to cost a second corrective call. The lock path
    pins the original lines immediately — the second scripted reply must
    stay unused."""
    conn = _ScriptedConnector([
        reply(PARAPHRASED_BLOCK),
        reply(GOOD_BLOCK),
    ])

    block, advice = run(enh, conn, DRAFT_DIALOGUE)

    assert len(conn.calls) == 1, "a paraphrase is pinned, not retried"
    assert "老板，你这里剩了那么多，怎么办？" in block
    assert "老板，你今天剩了那么多" not in block
    assert "Pinned" in advice


def test_structured_draft_uses_the_lock_prompt(enh):
    """The system prompt for an already-structured draft must not teach
    the model to invent six lines and call them verbatim."""
    conn = _ScriptedConnector([reply(GOOD_BLOCK)])

    run(enh, conn, DRAFT_DIALOGUE)

    system = conn.calls[0]["messages"][0]["content"]
    assert "Camera DIRECTION" in system
    assert "all verbatim" not in system.lower()
    assert "Six back-and-forth" not in system
    assert conn.calls[0]["temperature"] == 0.0


def test_content_failure_does_not_spend_a_second_seed(enh):
    conn = _ScriptedConnector([reply(PARAPHRASED_BLOCK)])

    run(enh, conn, DRAFT_DIALOGUE, seed=0)

    assert len(conn.calls) == 1
    assert conn.calls[0]["seed"] == 0


def test_missing_block_retry_does_advance_the_seed(enh):
    """The opposite case: an empty/blockless reply IS transport-shaped,
    so a fresh seed is the right lever there."""
    conn = _ScriptedConnector([
        "no markers at all",
        reply(GOOD_BLOCK),
    ])

    run(enh, conn, DRAFT_DIALOGUE, seed=0)

    assert conn.calls[0]["seed"] == 0
    assert conn.calls[1]["seed"] == 1


# --------------------------------------------------------------------------- #
# Holes found in review (2026-10-05): the contract skipped boards the
# node accepts, and the matcher was too loose to notice.
# --------------------------------------------------------------------------- #
def test_single_line_draft_is_still_protected(enh):
    """`parse_structured_dialogue_turns` needs two or more lines, so a
    one-line draft used to bypass the contract entirely — and a one-line
    paraphrase shipped verbatim-unverified."""
    draft = "黑猫：今天天气真好。"
    bad = reply("场景设定：厨房。\n\n黑猫：今天天气不错啊。")
    good = reply("场景设定：厨房。\n\n黑猫：今天天气真好。")
    conn = _ScriptedConnector([bad, good])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1, "a one-line paraphrase is pinned immediately"
    assert "黑猫：今天天气真好。" in block


def test_line_swallowed_by_a_longer_neighbour_is_caught(enh):
    """A substring test passes when a short utterance sits inside a
    longer one — but that is a changed line, not a preserved one."""
    draft = "图1的黑猫：明天怎么办？\n图2的白猫：先做。"
    bad = reply("场景设定：厨房。\n\n图1的黑猫：那明天怎么办，明天怎么办？\n图2的白猫：先做。")
    good = reply("场景设定：厨房。\n\n" + draft)
    conn = _ScriptedConnector([bad, good])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1
    assert "图1的黑猫：明天怎么办？" in block


def test_speaker_swap_is_caught(enh):
    """Right words, wrong character: the substring test passed this, and
    the finished video would give one cat another's line."""
    draft = "图1的黑猫：明天怎么办？\n图2的白猫：先做。"
    bad = reply("场景设定：厨房。\n\n图1的黑猫：先做。\n图2的白猫：明天怎么办？")
    good = reply("场景设定：厨房。\n\n" + draft)
    conn = _ScriptedConnector([bad, good])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1
    assert "图1的黑猫：明天怎么办？" in block
    assert "图2的白猫：先做。" in block


def test_draft_rejected_by_parser_still_protects_its_lines(enh):
    """A draft with prose after the dialogue is rejected wholesale by the
    downstream parser — but the lines BEFORE that point are still the
    user's words and must not become editable."""
    draft = (
        "场景设定：厨房。\n"
        "图1的黑猫：明天怎么办？\n"
        "图2的白猫：先做。\n"
        "（这里用户自己写了一段散文）"
    )
    bad = reply(
        "场景设定：厨房。\n\n图1的黑猫：那明天还做这么多吗？\n图2的白猫：先做。"
    )
    good = reply(
        "场景设定：厨房。\n\n图1的黑猫：明天怎么办？\n图2的白猫：先做。"
    )
    conn = _ScriptedConnector([bad, good])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1, "lines before the prose are pinned, not retried"
    assert "图1的黑猫：明天怎么办？" in block


def test_pin_keeps_camera_lines_from_the_dialogue_region(enh):
    """The pin rebuilds the dialogue region; the staging the rewrite
    produced must survive even when the model put it in the wrong place.

    A `镜头：` line IS hoistable because the downstream parser lists it
    in its production-directive blacklist and reads it as prologue."""
    polluted = (
        "场景设定：厨房。\n\n"
        "图1的黑猫：老板，今天剩了这么多，怎么处理？\n"
        "镜头：正反打对话。\n"
        "图2的白猫：倒掉。\n"
        "图1的黑猫：那明天还做这么多吗？\n"
        "图2的白猫：先做。"
    )
    conn = _ScriptedConnector([reply(polluted)] * 3)

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    assert "镜头：正反打对话。" in block, "camera line lost by the pin"
    turns, _pro = enh._dlg_parse_structured_dialogue_turns(block)
    assert turns, "pinned block must still parse"
    assert [ln for t in turns for ln in t.lines] == [
        "老板，你这里剩了那么多，怎么办？",
        "倒掉。",
        "明天怎么办？",
        "先做。",
    ]


def test_pin_does_not_hoist_a_line_the_downstream_would_read_as_speech(enh):
    """`参考图1：黑猫` is NOT in the downstream blacklist, so the parser
    reads it as a character named 参考图1 uttering 「黑猫」. Hoisting it
    into the prologue would not fix that — it would make the phantom the
    FIRST line, i.e. the S1 every later speaker ID counts from. The pin
    must therefore use the downstream's own speaker test for its region
    split and leave such lines where the rewrite put them."""
    polluted = (
        "场景设定：厨房。\n\n"
        "图1的黑猫：老板，今天剩了这么多，怎么处理？\n"
        "镜头：正反打对话。\n"
        "参考图1：黑猫\n"
        "图2的白猫：倒掉。\n"
        "图1的黑猫：那明天还做这么多吗？\n"
        "图2的白猫：先做。"
    )
    conn = _ScriptedConnector([reply(polluted)] * 3)

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    # It is NOT moved to the head of the block.
    assert not block.startswith("参考图1：黑猫")
    assert "镜头：正反打对话。" in block


def test_pin_handles_a_rewrite_with_no_dialogue_at_all(enh):
    """A rewrite that dropped every line leaves no speaker line to split
    on; the draft's lines must still come back."""
    prose_only = reply("场景设定：厨房，暖黄吊灯，料理台堆满食材。\n镜头：正反打。")
    conn = _ScriptedConnector([prose_only] * 3)

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    for line in DRAFT_DIALOGUE.split("\n"):
        assert line in block
    assert "暖黄吊灯" in block, "setting prose must survive the pin"


def test_scene_advice_does_not_double_count_multi_line_turns(enh):
    """A two-line turn's duration was being added once per line inside a
    scene, roughly doubling the suggested total."""
    draft = (
        "图1的黑猫：老板，你这里剩了那么多，怎么办？\n"
        "图1的黑猫：明天怎么办？\n"
        "图2的白猫：先做。\n"
        "图1的黑猫：还是你这儿好，吃了不烧心。"
    )
    conn = _ScriptedConnector([reply("场景设定：厨房。\n\n" + draft)])

    _block, advice = run(enh, conn, draft)

    line = next(
        ln for ln in advice.split("\n") if "Suggest scene_count" in ln
    )
    total = int(
        line.split("total_duration_seconds≈")[1].split(" ")[0]
    )
    # 4 lines at normal pacing: ~4.5 + 2.3 + 1.6 + 3.8 ≈ 12s of speech.
    # A doubled count would land well above 30s for a 3-scene suggestion.
    assert total <= 30, f"suggested total looks double-counted: {line}"


def test_attempts_one_pins_immediately(enh):
    """With no retries available the pin is the only path — it must not
    raise."""
    conn = _ScriptedConnector([reply(PARAPHRASED_BLOCK)])

    block, advice = run(enh, conn, DRAFT_DIALOGUE, attempts=1)

    assert "老板，你这里剩了那么多，怎么办？" in block
    assert "Pinned" in advice


def test_exhausted_retries_pin_original_lines_back(enh):
    """Never raise over a missing word: pin the draft's own lines back so
    the board still runs and nothing was lost."""
    conn = _ScriptedConnector([reply(PARAPHRASED_BLOCK)] * 3)

    block, advice = run(enh, conn, DRAFT_DIALOGUE)

    for line in DRAFT_DIALOGUE.split("\n"):
        spoken = line.split("：", 1)[1]
        assert spoken in block, f"pinned block lost a line: {spoken!r}"
    assert "Pinned" in advice or "WARNING" in advice


def test_speaker_normalisation_is_allowed_but_words_are_not(enh):
    """Stripping a picture prefix from a speaker name is this node's job
    (one stable key for the speaker map), so `图3的小猫` -> `小猫` must
    pass. The WORDS still have to match exactly — that is what the
    actor says out loud."""
    draft = "图3的小猫：Mommy,daddy is so ugly,why did you marry him.\n图2的白猫：oh my god"
    normalised = reply(
        "场景设定：客厅。\n\n"
        "小猫：Mommy,daddy is so ugly,why did you marry him.\n"
        "白猫：oh my god"
    )
    conn = _ScriptedConnector([normalised])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1, "roster normalisation is not a violation"
    assert "Mommy,daddy is so ugly,why did you marry him." in block


def test_pinned_block_is_still_parseable(enh):
    """The pin must produce a block the DOWNSTREAM parser accepts. A
    restored line missing its ``speaker：`` prefix would silently become
    prose, which is the exact failure the contract exists to prevent."""
    polluted = (
        "场景设定：厨房。\n"
        "镜头：正反打。\n"
        "\n"
        "图1的黑猫：老板，今天剩了这么多，怎么处理？\n"
        "图2的白猫：倒掉。\n"
        "图1的黑猫：那明天还做这么多吗？\n"
        "图2的白猫：先做。"
    )
    conn = _ScriptedConnector([reply(polluted)] * 3)

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    turns, _prologue = enh._dlg_parse_structured_dialogue_turns(block)
    assert turns, "pinned block must still parse as dialogue"
    assert [ln for t in turns for ln in t.lines] == [
        "老板，你这里剩了那么多，怎么办？",
        "倒掉。",
        "明天怎么办？",
        "先做。",
    ]
    # The rewrite's own camera line survives the rebuild.
    assert "镜头：正反打。" in block


def test_verbatim_pass_through_does_not_pin_or_retry(enh):
    """The happy path must stay cheap: a faithful rewrite is returned on
    the first attempt with no pinning."""
    conn = _ScriptedConnector([reply(GOOD_BLOCK)])

    block, advice = run(enh, conn, DRAFT_DIALOGUE)

    assert len(conn.calls) == 1
    assert block == GOOD_BLOCK
    assert "Pinned" not in advice and "WARNING" not in advice


def test_verify_block_content_reports_dropped_lines(enh):
    block = "场景设定：厨房。\n\n图1的黑猫：完全不一样的话。\n图2的白猫：倒掉。"
    errors, _notices = enh.verify_block_content(
        block, DRAFT_DIALOGUE, "Dialogue"
    )
    assert errors, "dropped lines must be reported"
    assert any("verbatim" in e for e in errors)


def test_draft_dialogue_lines_only_for_canonical_drafts(enh):
    assert enh.draft_dialogue_lines("一段散文，没有任何对白。") == []
    assert enh.draft_dialogue_lines(DRAFT_DIALOGUE) == [
        "老板，你这里剩了那么多，怎么办？",
        "倒掉。",
        "明天怎么办？",
        "先做。",
    ]


# --------------------------------------------------------------------------- #
# Camera line placement
# --------------------------------------------------------------------------- #
def test_camera_line_inside_dialogue_block_is_hoisted_on_pin(enh):
    """A `镜头：` line inside the dialogue block breaks the downstream
    parser. The lock path pins on that first reply and keeps the camera
    line above the restored utterances."""
    polluted = (
        "场景设定：厨房。\n"
        "\n"
        "图1的黑猫：老板，你这里剩了那么多，怎么办？\n"
        "镜头：切到白猫近景\n"
        "图2的白猫：倒掉。\n"
        "图1的黑猫：明天怎么办？\n"
        "图2的白猫：先做。"
    )
    conn = _ScriptedConnector([reply(polluted)])

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    assert len(conn.calls) == 1
    assert block.find("镜头：切到白猫近景") < block.find("老板，你这里剩了那么多")
    turns, _prologue = enh._dlg_parse_structured_dialogue_turns(block)
    assert turns


def test_camera_line_in_prologue_is_accepted(enh):
    """The same line ABOVE the first spoken line is legal prologue and
    must NOT be stripped — it is the only position where camera
    direction survives into the per-clip concept slot."""
    with_camera = (
        "场景设定：厨房。\n"
        "镜头：正反打对话，说话方给中近景。\n"
        "\n" + DRAFT_DIALOGUE
    )
    conn = _ScriptedConnector([reply(with_camera)])

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    assert len(conn.calls) == 1, "prologue camera line must not trigger a retry"
    assert "镜头：正反打对话，说话方给中近景。" in block


# --------------------------------------------------------------------------- #
# No false positives
# --------------------------------------------------------------------------- #
def test_narration_board_parse_failure_is_not_an_error(enh):
    """Narration is not supposed to parse as dialogue; failing it would
    reject a valid rewrite."""
    draft = "一个老钟表匠在午后的光里举起一只黄铜怀表，机位固定。"
    body = (
        "A grey-bearded watchmaker in a faded grey wool vest sits at a worn "
        "wooden workbench, holding a brass pocket watch up to the slanting "
        "light. Camera: locked-off medium close-up."
    )
    conn = _ScriptedConnector([reply(body, classification="Narration")])

    block, _advice = run(enh, conn, draft, category="none - 不指定")

    assert len(conn.calls) == 1
    assert "watchmaker" in block


def test_freeform_prose_draft_skips_verbatim_contract(enh):
    """A prose draft has no locatable source lines, so the rewrite is free
    to smooth it — there is no verbatim contract to enforce."""
    draft = "two cats in a cafe argue about food, one says regular meals are better"
    body = "场景设定：咖啡店。\n\n白猫：规律进食比较好。\n黑猫：那叫闹钟。"
    conn = _ScriptedConnector([reply(body)])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1
    assert "白猫：规律进食比较好。" in block


def test_misclassification_is_a_notice_not_a_failure(enh):
    """Classification disagreeing with the block shape is advisory —
    mixed boards are common and a hard failure would misfire."""
    conn = _ScriptedConnector([reply(GOOD_BLOCK, classification="Narration")])

    block, _advice = run(enh, conn, DRAFT_DIALOGUE, category="none - 不指定")

    assert len(conn.calls) == 1
    assert "图2的白猫：先做。" in block


# --------------------------------------------------------------------------- #
# Advice output
# --------------------------------------------------------------------------- #
def test_advice_output_carries_classification_and_cut_advice(enh):
    """The second output surfaces the Classification plus this module's
    own findings. Previously the header was dropped on the standalone
    path and only reached the preflight in inline (loop-node) mode."""
    conn = _ScriptedConnector([reply(GOOD_BLOCK)])
    node = enh.MiniMaxH3LoopUserInputEnhancer()

    out = node.enhance(
        conn,
        draft=DRAFT_DIALOGUE,
        category="dialogue - 对白/对话/相声",
        reference_mode="t2va - 文生视频链(默认)",
    )
    _block, advice = out["result"]

    assert "reference_mode: t2va" in advice
    assert "Classification" in advice
    assert "scene_count" in advice
    assert "total_duration_seconds" in advice
    assert DRAFT_DIALOGUE.split("\n", 1)[0] in out["ui"]["text"]


def test_scene_advice_empty_for_narration(enh):
    """No cut-budget advice on a board that has no dialogue to budget."""
    draft = "一个老钟表匠在午后的光里举起一只黄铜怀表。"
    body = "A watchmaker holds up a brass pocket watch. Camera: locked-off."
    conn = _ScriptedConnector([reply(body, classification="Narration")])

    _block, advice = run(enh, conn, draft, category="none - 不指定")

    assert "scene_count" not in advice


def test_scene_advice_respects_the_longest_unsplittable_turn(enh):
    """The suggested total must leave room for the longest single turn:
    a turn can never be split across scenes, so a share below its own
    duration truncates the line."""
    long_line = (
        "图1的黑猫：怕，但更怕账不清。宁可少挣点，也不能坏了规矩。"
        "规矩立住了，路才会越走越宽。"
    )
    draft = (
        "图1的黑猫：老板，你这里剩了那么多，怎么办？\n"
        "图2的白猫：倒掉。\n"
        f"图2的白猫：{long_line.split('：', 1)[1]}\n"
        "图1的黑猫：还是你这儿好，吃了不烧心。"
    )
    block = "场景设定：厨房。\n\n" + draft
    conn = _ScriptedConnector([reply(block)])

    _b, advice = run(enh, conn, draft)

    assert "cannot be split" in advice


# --------------------------------------------------------------------------- #
# Prompt contract (the rules the LLM is asked to follow)
# --------------------------------------------------------------------------- #
def test_prompt_requires_camera_direction_preservation(enh):
    sys_text = enh.load_prompt_text(enh._PROMPT_NAME)
    assert "Camera DIRECTION" in sys_text
    assert "PRESERVE" in sys_text.upper() or "preserve" in sys_text


def test_prompt_forbids_non_dialogue_inside_the_dialogue_block(enh):
    sys_text = enh.load_prompt_text(enh._PROMPT_NAME)
    assert "ONLY dialogue" in sys_text


def test_colon_label_inside_prose_composes_and_is_not_pinned(enh):
    """A colon line in ordinary prose is not a spoken line.

    Locking it used to pin a phantom speaker (要求 / Goal / 小猫说)
    after the model correctly folded the label into the setting.
    """
    drafts = (
        "拍一只小猫在厨房里走。\n要求：视频里不要出现人类",
        "Make a short clip.\nGoal: a funny cat video",
        "小猫趴在碗边，看起来没精神。\n小猫说：我今天不想吃饭。",
    )
    for draft in drafts:
        conn = _ScriptedConnector([
            reply(
                "场景设定：厨房，一只小猫趴在碗边。不要出现人类。",
                classification="Narration",
            )
        ])
        block, advice = run(
            enh, conn, draft, temperature=0.7, category="none - 不指定",
        )
        assert conn.calls[0]["temperature"] == 0.7, draft
        system = conn.calls[0]["messages"][0]["content"]
        assert "newly written" in system, draft
        assert "Pinned" not in advice, draft
        assert "要求：" not in block
        assert "Goal:" not in block
        assert "小猫说：" not in block


def test_single_spoken_line_beside_a_setting_header_stays_locked(enh):
    draft = "场景设定：厨房。\n黑猫：今天天气真好。"
    conn = _ScriptedConnector([
        reply("场景设定：厨房。\n\n黑猫：今天天气不错。")
    ])

    block, _advice = run(enh, conn, draft)

    assert conn.calls[0]["temperature"] == 0.0
    assert "黑猫：今天天气真好。" in block
    assert "今天天气不错" not in block


def test_invented_spoken_line_is_a_notice_not_a_pin(enh):
    extra = DRAFT_DIALOGUE + "\n路人：我也要一份。"
    conn = _ScriptedConnector([reply("场景设定：厨房。\n\n" + extra)])

    block, advice = run(enh, conn, DRAFT_DIALOGUE)

    assert len(conn.calls) == 1
    assert "路人：我也要一份。" in block
    assert "Pinned" not in advice
    assert "not in the draft" in advice


def test_lock_cache_misses_when_provider_identity_changes(enh, tmp_path):
    """Same model name, different endpoint, must not share a rewrite."""

    class _Endpoint(_ScriptedConnector):
        def __init__(self, replies, state):
            super().__init__(replies)
            self._state = state

        def get_state(self):
            return self._state

    root = str(tmp_path)
    local = _Endpoint(
        [reply(GOOD_BLOCK)],
        "http://127.0.0.1:8080/v1local-tokenfake-model",
    )
    official = _Endpoint(
        [reply(GOOD_BLOCK)],
        "https://api.example/v1official-tokenfake-model",
    )
    run(enh, local, DRAFT_DIALOGUE, cache=True, cache_root=root)
    run(enh, official, DRAFT_DIALOGUE, cache=True, cache_root=root)
    assert len(local.calls) == 1
    assert len(official.calls) == 1


def test_lines_after_a_camera_note_stay_protected(enh):
    draft = (
        "图1的黑猫：明天怎么办？\n"
        "镜头：正反打\n"
        "图2的白猫：先做。\n"
    )
    bad = reply("场景设定：厨房。\n\n图1的黑猫：明天怎么办？\n图2的白猫：先做吧。")
    conn = _ScriptedConnector([bad])

    block, _advice = run(enh, conn, draft)

    assert len(conn.calls) == 1
    assert "图2的白猫：先做。" in block
    assert "先做吧" not in block


def test_pin_drops_prose_that_restates_a_rewritten_line(enh):
    polluted = (
        "场景设定：厨房。\n"
        "老板，你今天剩了那么多，要怎么处理？\n"
        "图1的黑猫：完全不同的一句。\n"
    )
    conn = _ScriptedConnector([reply(polluted)])

    block, _advice = run(enh, conn, DRAFT_DIALOGUE)

    assert "要怎么处理" not in block
    assert "老板，你这里剩了那么多，怎么办？" in block
    assert "场景设定：厨房。" in block


def test_freeform_draft_uses_compose_prompt_and_keeps_temperature(enh):
    draft = "two cats in a cafe argue about food"
    body = "场景设定：咖啡店。\n\n白猫：规律进食比较好。\n黑猫：那叫闹钟。"
    conn = _ScriptedConnector([reply(body)])

    run(enh, conn, draft, temperature=0.7)

    system = conn.calls[0]["messages"][0]["content"]
    assert "## (A) Dialogue" in system
    assert "newly written" in system
    assert conn.calls[0]["temperature"] == 0.7


def test_reference_modes_keep_separate_picture_contracts(enh):
    phrases = {
        "t2va - 文生视频链(默认)": "Do not write a Picture-to-character binding line.",
        "i2va - 首帧关键帧(文生视频+首帧图)": "Picture 1 is the opening frame of scene 1 only.",
        "fl2va - 首尾帧关键帧(首帧+逐场尾帧)": "Pictures form a frame cycle",
        "ref2va - 参考图(N张/全场景)": "A Picture-to-character binding line is required.",
    }
    absent = {
        "t2va - 文生视频链(默认)": "A Picture-to-character binding line is required.",
        "i2va - 首帧关键帧(文生视频+首帧图)": "Pictures form a frame cycle",
        "fl2va - 首尾帧关键帧(首帧+逐场尾帧)": "opening frame of scene 1 only",
        "ref2va - 参考图(N张/全场景)": "Do not write a Picture-to-character binding line.",
    }
    for mode, phrase in phrases.items():
        conn = _ScriptedConnector([reply(GOOD_BLOCK)])
        run(enh, conn, DRAFT_DIALOGUE, reference_mode=mode)
        system = conn.calls[0]["messages"][0]["content"]
        assert phrase in system, mode
        assert absent[mode] not in system, mode


def test_lock_cache_ignores_seed_and_mode_change_misses(enh, tmp_path):
    conn = _ScriptedConnector([reply(GOOD_BLOCK), reply(GOOD_BLOCK), reply(GOOD_BLOCK)])
    root = str(tmp_path)

    run(enh, conn, DRAFT_DIALOGUE, seed=1, cache=True, cache_root=root)
    run(enh, conn, DRAFT_DIALOGUE, seed=9, cache=True, cache_root=root)
    assert len(conn.calls) == 1

    run(
        enh, conn, DRAFT_DIALOGUE, seed=9, cache=True, cache_root=root,
        reference_mode="ref2va - 参考图(N张/全场景)",
    )
    assert len(conn.calls) == 2


def test_missing_block_is_not_cached(enh, tmp_path):
    root = str(tmp_path)
    failing = _ScriptedConnector(["no markers"])
    try:
        run(enh, failing, DRAFT_DIALOGUE, attempts=1, cache=True, cache_root=root)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a markerless reply must raise")

    conn = _ScriptedConnector([reply(GOOD_BLOCK)])
    run(enh, conn, DRAFT_DIALOGUE, cache=True, cache_root=root)
    assert len(conn.calls) == 1


def test_scene_advice_matches_the_shared_ceil_helper(enh):
    conn = _ScriptedConnector([reply(GOOD_BLOCK)])
    _block, advice = run(enh, conn, DRAFT_DIALOGUE)
    line = next(ln for ln in advice.split("\n") if "Suggest scene_count" in ln)
    suggested = int(line.split("total_duration_seconds≈")[1].split(" ")[0])
    count = int(line.split("scene_count=")[1].split(" ")[0])
    turns, _prologue = enh._dlg_parse_structured_dialogue_turns(GOOD_BLOCK)
    preset = enh._DLG_PACING_PRESETS["normal"]
    fine, _report = enh._dlg_estimate_shot_budget(turns, preset, max_shot_seconds=0.0)
    expected = enh._dlg_min_total_seconds_for_scene_count(fine, count)
    assert suggested == expected
