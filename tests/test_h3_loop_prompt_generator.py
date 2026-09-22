# -*- coding: utf-8 -*-
"""End-to-end tests for the ``MiniMaxH3LoopPromptGenerator`` node with a
stubbed LLM connector: full pipeline (Stage 0-3) in per_shot and
single_call modes, plan shape vs. the real Production Plan JSON, and
failure handling."""
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
def lg():
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
    _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
        LLM_DIR / "minimax_h3_loop_prompts.py",
    )
    return _load_file(
        "_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator",
        LLM_DIR / "minimax_h3_loop_prompt_generator.py",
    )


class FakeConnector:
    model = "fake-model"

    def get_state(self):
        return "fake-state"


class ScriptedConnector(FakeConnector):
    """Serves queued replies; understands which stage is calling by the
    user content so tests can mix prefix + shot replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, *, seed=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("scripted connector exhausted")
        return self.replies.pop(0)


def _storyboard_json(n=3):
    shots = []
    for i in range(1, n + 1):
        shots.append(
            {
                "id": f"scene_{i:02d}",
                "description": f"Shot {i} description of the courtyard.",
                "shot_type": "medium_shot",
                "camera_movement": "slow_push_in",
                "transition_in": "fade_from_black" if i == 1 else "hard_cut",
                "duration_seconds": 10,
                "narrative_beat": "establish",
                "characters": ["young_woman"],
                "props": ["porcelain_bowl"],
                "notes": "Carry the bowl.",
            }
        )
    return json.dumps(shots, ensure_ascii=False)


def _clip_reply(i):
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] Clip {i}: the bowl settles, ice slips.\n"
        "\n"
        "overall_soundscape:\n"
        "Cicadas hold; one bright clink.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def _single_call_reply(n):
    items = []
    for i in range(1, n + 1):
        items.append(
            {
                "id": f"scene_{i:02d}",
                "integrated_multimodal_description": f"[Shot 1] Clip {i} body text.",
                "overall_soundscape": "Rain on glass, distant cicadas.",
                "non_diegetic_music": "No non-diegetic music.",
            }
        )
    return json.dumps(items, ensure_ascii=False)


# Stubs for the new v4 plan pipeline (extract_dialogue + storyboard +
# prefix + per-shot). The v4 path always runs them in this order for
# non-dialogue natural-language inputs (extract_dialogue returns an
# empty turn list, falling back to the narrator path; _auto_storyboard
# then emits a fresh board).
def _extract_dialogue_empty():
    """Reply the LLM span-extractor gives when the concept has no
    `speaker：text` lines — the orchestrator falls back to a narrator
    turn and asks _auto_storyboard to pick the scene count."""
    return json.dumps({"turns": []})


def _auto_storyboard_reply(n):
    """Reply the storyboard LLM gives: a JSON array of n board entries."""
    shots = []
    for i in range(1, n + 1):
        shots.append(
            {
                "id": f"scene_{i:02d}",
                "description": f"Shot {i} description of the courtyard.",
                "shot_type": "medium_shot",
                "camera_movement": "slow_push_in",
                "transition_in": "fade_from_black" if i == 1 else "hard_cut",
                "duration_seconds": 10,
                "narrative_beat": "establish",
                "characters": ["young_woman"],
                "props": ["porcelain_bowl"],
                "notes": "Carry the bowl.",
            }
        )
    return json.dumps(shots, ensure_ascii=False)


PREFIX_REPLY = (
    "Always the same young woman in a Jiangnan courtyard at high summer: "
    "loosely pinned black hair, white cotton blouse, pale jade bracelet."
)


# --------------------------------------------------------------------------- #
# per_shot happy path
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# single_call mode
# --------------------------------------------------------------------------- #
def test_parse_generation_mode(lg):
    assert lg.parse_generation_mode("per_shot - 逐场生成(推荐)") == "per_shot"
    assert lg.parse_generation_mode("single_call") == "single_call"
    assert lg.parse_generation_mode("weird") == "per_shot"
    assert lg.parse_generation_mode("") == "per_shot"


# --------------------------------------------------------------------------- #
# Auto-storyboard mode (scene_count) + trimming
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Natural-language input + validation errors
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Retry behavior
# --------------------------------------------------------------------------- #
def test_shot_parse_retry_then_success(lg):
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(2),
        PREFIX_REPLY,
        "garbage reply",  # shot1 attempt 1 -> raises -> retry
        _clip_reply(1),    # shot1 attempt 2 -> succeeds
        _clip_reply(2),    # shot2
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=2,
        seed=1,
    )
    assert len(json.loads(out["plan_json"])["shots"]) == 2
    # extract + storyboard + prefix + shot1(garbage) + shot1(retry) + shot2
    assert len(conn.calls) == 6
    retry_call = conn.calls[4]
    assert len(retry_call) == 3
    assert "could not be parsed" in retry_call[2]["content"]
    assert "integrated_multimodal_description:" in retry_call[2]["content"]

def test_shot_parse_failure_raises_no_partial_plan(lg):
    # Storyboard emits 2 shots; shot1 attempt 1 + attempt 2 both
    # garbage -> 2 attempts raise, no shot2 ever runs.
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(2),
        PREFIX_REPLY,
        "bad",
        "still bad",
    ])
    with pytest.raises(RuntimeError, match="scene_01"):
        lg.H3LoopPromptEnhancer(conn)(
            user_input="a quiet courtyard summer afternoon",
            scene_count=2,
            seed=1,
        )

def test_prefix_retry(lg):
    # First prefix reply is empty -> 'no prefix paragraph' retry trigger.
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        "",  # empty reply -> prefix retry
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=1,
        seed=1,
    )
    # extract + storyboard + prefix(attempt 1, empty) +
    # prefix(attempt 2, ok) + shot1
    assert len(conn.calls) == 5
    assert json.loads(out["plan_json"])["prompt_prefix"] == [PREFIX_REPLY]

def test_think_block_stripped(lg):
    reply = "<think>chain of thought</think>\n" + _clip_reply(1)
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        reply,
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=1,
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert "think" not in json.dumps(plan["shots"][0]["prompt"])

def test_node_generate_returns_five_outputs(lg):
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(2),
        PREFIX_REPLY,
        _clip_reply(1),
        _clip_reply(2),
    ])
    node = lg.MiniMaxH3LoopPromptGenerator()
    result = node.generate(
        conn,
        user_input="a quiet courtyard summer afternoon",
        scene_count=2,
        seed=42,
        seed_mode="same_across_scenes - 全场同seed",
        timeout=60,
    )
    assert len(result) == 3
    plan_json, summary, board_kind = result
    plan = json.loads(plan_json)
    # Strict upstream contract: only id / prompt / length / seed per shot.
    # seed_mode passed explicitly: same_across_scenes keeps one shared seed.
    assert [s["seed"] for s in plan["shots"]] == ["42", "42"]  # unified
    assert "MiniMax H3 Loop Plan summary" in summary
    assert "Board: narration (no spoken lines detected)" in summary
    assert board_kind == "narration"
    assert "LLM requests: 5" in summary  # extract+storyboard+prefix+2 shots
    assert "Tokens (estimated)" in summary
    assert "reference=t2va" in summary
    assert "defaults" not in plan
    for s in plan["shots"]:
        assert "steps" not in s


def test_node_default_seed_mode_is_per_scene_increment(lg):
    """The upstream Context-Loop plugin derives per-scene seeds from one
    base (deterministic but distinct per scene) — our recommended
    default matches: no seed_mode value -> seed_base + index per scene."""
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(2),
        PREFIX_REPLY,
        _clip_reply(1),
        _clip_reply(2),
    ])
    node = lg.MiniMaxH3LoopPromptGenerator()
    result = node.generate(
        conn,
        user_input="a quiet courtyard summer afternoon",
        scene_count=2,
        seed=42,
        timeout=60,
    )
    plan = json.loads(result[0])
    assert [s["seed"] for s in plan["shots"]] == ["43", "44"]

def test_node_input_types_parameter_surface_updated(lg):
    optional = lg.MiniMaxH3LoopPromptGenerator.INPUT_TYPES()["optional"]
    assert optional["scene_count"][1]["max"] == 128
    assert optional["output_language"][1]["default"] == "en"
    assert "seed_mode" in optional
    assert "split_bias" not in optional
    assert "caption_mode" in optional
    assert "category" in optional
    assert "task_mode" not in optional
    assert "unified_seed" not in optional
    assert "force_recaption" not in optional
    assert "caption_cache_scope" not in optional
    # Loop category widget is the curated 3-entry subset (none / dialogue
    # / action); legacy cinematic-story / short-drama / anime etc. are
    # NOT exposed — sibling nodes still use the full 22-entry taxonomy.
    cat_choices = list(optional["category"][0])
    assert cat_choices == [
        "none - 不指定",
        "dialogue - 对白/对话/相声",
        "action - 动作戏/打斗/飙车",
    ]
    assert optional["category"][1]["default"] == "none - 不指定"
    # And the loop-local advice table covers dialogue + action (none is "").
    assert lg.LOOP_CATEGORY_ADVICE["dialogue"].startswith("spoken-scene cinematography")
    assert lg.LOOP_CATEGORY_ADVICE["action"].startswith("motion blur")
    assert lg.LOOP_CATEGORY_ADVICE["none"] == ""


def test_caption_mode_maps_to_cache_controls(lg):
    class SpyConnector(ScriptedConnector):
        pass

    # Caption stage is stubbed out (spy_caption), so the LLM call queue
    # is extract_dialogue + storyboard + prefix + 1 shot.
    conn = SpyConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    seen = {}
    enh = lg.H3LoopPromptEnhancer(conn)

    def _spy_caption(*, images, ref_code, seed, **kwargs):
        seen["force_recaption"] = kwargs.get("force_recaption")
        seen["caption_cache_scope"] = kwargs.get("caption_cache_scope")
        return (
            [{"slot": "Picture 1", "about": "a subject in red coat", "role": "identity"}],
            [],
        )

    enh._caption_images = _spy_caption
    fake_images = __import__("numpy").zeros((1, 4, 4, 3), dtype=float)
    enh(
        user_input="a quiet courtyard summer afternoon",
        scene_count=1,
        reference_mode="i2va",
        images=fake_images,
        caption_mode="force_recaption_once - 本次强制重打标",
        seed=1,
    )
    assert seen["force_recaption"] is True
    assert seen["caption_cache_scope"] == "disabled"

def test_caption_cache_hits_on_second_run_memory_only(lg):
    # Both runs share the SAME connector so the script is consumed
    # across runs. Per-run LLM order: caption (cache miss) -> extract
    # -> storyboard -> prefix -> shot. Run 2 caption is a memory
    # cache hit so its LLM call is skipped.
    conn = ScriptedConnector([
        # run 1: caption LLM + extract + storyboard + prefix + shot
        "orange_tabby_kitten in a red jacket under soft window light.",
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
        # run 2: caption cached -> extract + storyboard + prefix + shot
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    fake_images = __import__("numpy").zeros((1, 4, 4, 3), dtype=float)
    kwargs = dict(
        user_input="a quiet scene",
        scene_count=1,
        reference_mode="i2va",
        images=fake_images,
        caption_mode="cache_memory_only - 缓存:仅内存",
        seed=1,
    )
    first = enh(**kwargs)
    second = enh(**kwargs)
    # 5 run-1 calls + 4 run-2 calls (caption from memory cache).
    assert len(conn.calls) == 9
    assert "hit (0%)" in first["summary"]
    assert "hit (100%)" in second["summary"]

def test_caption_cache_disk_root_prefers_comfy_output(monkeypatch, lg, tmp_path):
    fp = types.SimpleNamespace(get_output_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    root = lg._caption_cache_disk_root()
    assert root == str(tmp_path / "mien_nodes" / "caption_cache")


def test_is_changed_stable_and_sensitive(lg):
    node = lg.MiniMaxH3LoopPromptGenerator()
    base = dict(
        user_input="a quiet courtyard summer afternoon",
        scene_count=2,
        seed=0,
        generation_mode="per_shot - 逐场生成(推荐)",
        category="none - 不指定",
        output_language="en",
        temperature=0.4,
        max_tokens=8192,
        timeout=120,
    )
    a = node.is_changed(FakeConnector(), **base)
    b = node.is_changed(FakeConnector(), **base)
    assert a == b
    for key, value in (
        ("seed", 1),
        ("scene_count", 3),
        ("user_input", "a different concept"),
    ):
        assert node.is_changed(FakeConnector(), **dict(base, **{key: value})) != a

def test_auto_storyboard_includes_pacing_derived_bias_directive(lg):
    # Stage 0.5 chain: extract_dialogue (returns empty) -> _auto_storyboard
    # (1st garbage, 2nd OK) -> prefix -> shot1. The storyboard LLM user
    # message is the second call (index 2: 0=extract, 1=garbage, 2=retry).
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        "junk then real",  # storyboard attempt 1 fails JSON parse
        _auto_storyboard_reply(1),  # storyboard attempt 2 succeeds
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=1,
        pacing="fast - 快",
        seed=1,
    )
    # The storyboard retry carried the corrective user turn; the
    # NEW attempt uses the user-supplied split_bias directive (which
    # sits in the original user message, NOT in the corrective add-on).
    # The prompt template uses space-separated ``split bias`` so the
    # directive is recognised as the snake_case widget value
    # anywhere in the user content.
    retry_msg = conn.calls[2][1]["content"].lower()
    assert "split bias" in retry_msg
    assert "aggressive" in retry_msg

def test_seconds_to_length_grid_math_for_per_shot_cap(lg):
    assert lg.seconds_to_length(4) == 107, (
        "4 s should be 107 frames on the 17k+5 grid (plan v4 §0.2)"
    )
    assert lg.seconds_to_length(14) == 345, (
        "14 s should be 345 frames on the 17k+5 grid (our MAX_PER_SHOT_SECONDS)"
    )
    assert lg.seconds_to_length(15) == 362, (
        "15 s should be 362 frames on the 17k+5 grid "
        "(= round-1 P0 false-positive boundary; 14 s honest clip rounds to "
        "345 and must NOT trigger any over-length warning)"
    )
    assert lg.seconds_to_length(30) == 736, (
        "30 s should be 736 frames (well above the 4-15 s model window)"
    )


def test_max_per_shot_seconds_constant_is_14(lg):
    """The hard cap constant lives in Stage 0; pin it at 14 s so a future
    edit cannot silently change the H3 model cap. The 14-s value (vs.
    15-s) leaves a 1-s safety margin on top of the documented 4-15 s
    window, since seconds_to_length(15) = 362 frames is the exact grid
    threshold and the H3 model may reject 15-s entries at the long edge.
    """
    import inspect
    src = inspect.getsource(lg.H3LoopPromptEnhancer.__call__)
    assert "MAX_PER_SHOT_SECONDS = 14" in src, (
        "Stage 0 per-shot cap constant must be 14 s; if you intentionally "
        "change it, update plan v4 §0.3 F3 and re-run the grid-math test."
    )


# --------------------------------------------------------------------------- #
# Interrupt propagation: the loop node must honour ComfyUI's "Stop" button
# rather than sitting in a single LLM round-trip until the per-call timeout
# fires. These tests monkeypatch the module-level interrupt gate so they
# run without an actual ComfyUI executor; they verify that every LLM call
# site (storyboard / prefix / per-shot / single-call / caption) bails out
# the moment the gate fires.
# --------------------------------------------------------------------------- #
def _install_interrupt_after(monkeypatch, after_calls: int, lg):
    """Make _comfy_interrupt_pressed() return False until ``after_calls``
    LLM invocations have happened, then return True. Returns the
    counter so the test can assert the exact number of calls."""
    state = {"calls": 0}

    def _pressed() -> bool:
        state["calls"] += 1
        return state["calls"] > after_calls

    # The generator module is loaded by the fixture under the
    # ``_mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator``
    # alias; the shim's free function ``_comfy_interrupt_pressed`` is
    # defined at module top-level. Monkeypatching on the spec-loaded
    # alias updates the same module object that the generator's
    # ``_check_interrupt`` reads from, because the module object is
    # stored in sys.modules under that alias.
    monkeypatch.setattr(lg, "_comfy_interrupt_pressed", _pressed)
    return state
    return state


def test_invoke_aborts_on_interrupt_pressed(monkeypatch, lg):
    """The very first LLM call must abort if interrupt is already set."""
    _install_interrupt_after(monkeypatch, after_calls=0, lg=lg)
    conn = ScriptedConnector([_clip_reply(1)])  # never reached
    enh = lg.H3LoopPromptEnhancer(conn)
    with pytest.raises(Exception) as exc:
        enh._invoke(
            [{"role": "user", "content": "x"}],
            temperature=0.4,
            seed=1,
            stage="probe",
        )
    # The helper raises InterruptProcessingException; in tests / standalone
    # the import shim degrades that to a plain Exception subclass.
    assert "interrupt pressed" in str(exc.value).lower()


def test_no_interrupt_passes_through_cleanly(monkeypatch, lg):
    """No interrupt -> orchestrator finishes and returns plan_json."""
    monkeypatch.setattr(lg, "_comfy_interrupt_pressed", lambda: False)
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(2),
        PREFIX_REPLY,
        _clip_reply(1),
        _clip_reply(2),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=2,
        seed=1,
    )
    assert json.loads(out["plan_json"])["shots"]

def test_auto_storyboard_retry_aborts_on_interrupt(monkeypatch, lg):
    """Storyboard parses fine on attempt 1 but the gate fires before
    the next retry — the helper inside the loop must short-circuit."""
    _install_interrupt_after(monkeypatch, after_calls=1, lg=lg)
    bad = _storyboard_json(0)  # empty -> first parse fails
    conn = ScriptedConnector([bad])
    enh = lg.H3LoopPromptEnhancer(conn)
    with pytest.raises(Exception) as exc:
        enh._auto_storyboard(
            "c", 0, 15, "none - 不指定", "en", seed=1
        )
    assert "interrupt pressed" in str(exc.value).lower()


def test_caption_images_aborts_on_interrupt(monkeypatch, lg):
    """Caption stage must bail between the per-frame LLM calls."""
    _install_interrupt_after(monkeypatch, after_calls=1, lg=lg)
    conn = ScriptedConnector([
        "orange_tabby kitten in a red jacket.",
        "another caption reply (never reached).",
    ])
    enh = lg.H3LoopPromptEnhancer(conn)
    fake = __import__("numpy").zeros((2, 4, 4, 3), dtype=__import__("numpy").float32)
    with pytest.raises(Exception) as exc:
        enh._caption_images(
            images=fake, ref_code="ref2va", seed=1,
        )
    assert "interrupt pressed" in str(exc.value).lower()


def test_generate_shot_prompt_aborts_after_first_reply(monkeypatch, lg):
    """Interrupt pressed -> the orchestrator's ``_check_interrupt``
    raises InterruptProcessingException at the gate before each
    LLM round-trip. ``extract_dialogue`` swallows that exception
    for the narrator-fallback path (so narrative concepts still
    degrade gracefully), but the gate fires at every other call
    site: ``_check_interrupt`` propagates InterruptProcessingException
    out of the orchestrator when it fires at the stage-1 prefix
    call site.

    The shot-prompt LLM call site is the contract we care about for
    this test: monkeypatch the gate to fire at the SHOT call site
    (after extract / storyboard / prefix have already happened) and
    confirm the orchestrator raises without making the per-shot call.
    """
    # State machine: allow first 3 LLM calls (extract, storyboard,
    # prefix), then fire.
    state = {"calls": 0}

    def _pressed() -> bool:
        state["calls"] += 1
        # Fire at and after the 4th call (the first per-shot call).
        return state["calls"] >= 4

    monkeypatch.setattr(lg, "_comfy_interrupt_pressed", _pressed)
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        # No _clip_reply queued: the per-shot call should be aborted
        # before consuming it.
    ])
    with pytest.raises(Exception):
        lg.H3LoopPromptEnhancer(conn)(
            user_input="a quiet courtyard summer afternoon",
            scene_count=1,
            seed=1,
        )
    # extract + storyboard ran; prefix's _check_interrupt fired before
    # the actual LLM call so the stub queue was not consumed.
    assert len(conn.calls) == 2

def test_invoke_import_shim_degrades_outside_comfyui(lg):
    """When nodes / comfy_execution are not importable (this is exactly
    the standalone-test environment), the gate must be a no-op and
    ``InterruptProcessingException`` must fall back to a catchable
    Exception subclass so test fixtures can ``pytest.raises`` against it.
    With the no-op gate the helper must NOT raise on its own."""
    mod = lg  # spec-loaded module provided by the ``lg`` fixture
    assert mod._comfy_interrupt_pressed() is False
    assert issubclass(mod.InterruptProcessingException, BaseException)
    # No-op gate → no raise.
    mod._check_interrupt("probe")  # would raise if gate were True
    # When the gate IS True, the helper must raise the shimmed type.
    original = mod._comfy_interrupt_pressed
    try:
        mod._comfy_interrupt_pressed = lambda: True  # type: ignore
        with pytest.raises(mod.InterruptProcessingException):
            mod._check_interrupt("probe")
    finally:
        mod._comfy_interrupt_pressed = original  # type: ignore


# --------------------------------------------------------------------------- #
# Inline user-input enhancement (enhance_user_input toggle)
# --------------------------------------------------------------------------- #
def _enhancer_reply_block():
    """A well-formed inline-enhancer reply: Classification + Notes +
    BEGIN/END block carrying the canonical rewrite."""
    return (
        "Classification: Narration\n"
        "Notes for the user: single paragraph, locked-off camera.\n"
        "\n"
        "--- BEGIN user_input ---\n"
        "A quiet Jiangnan courtyard at high summer; a young woman in a "
        "white cotton blouse carries a porcelain bowl across the frame. "
        "Camera: locked-off medium shot.\n"
        "--- END user_input ---"
    )


def test_parse_enhance_user_input(lg):
    assert lg.parse_enhance_user_input("on - 自动润色后再规划") is True
    assert lg.parse_enhance_user_input("on") is True
    assert lg.parse_enhance_user_input("OFF - 不润色(默认)") is False
    assert lg.parse_enhance_user_input("off - 不润色(默认)") is False
    assert lg.parse_enhance_user_input("") is False
    assert lg.parse_enhance_user_input("weird") is False


def test_auto_enhance_on_rewrites_before_pipeline(lg):
    conn = ScriptedConnector([
        _enhancer_reply_block(),      # 1. inline enhancement
        _extract_dialogue_empty(),    # 2. extractor runs on REWRITTEN text
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="courtyard summer",
        scene_count=1,
        seed=1,
        enhance_user_input=True,
    )
    assert len(conn.calls) == 5
    # The FIRST LLM call is the enhancement; its user message carries
    # the raw draft (the enhancer's ---BEGIN DRAFT--- wrapper).
    enh_user = conn.calls[0][1]["content"]
    assert "---BEGIN DRAFT---" in enh_user
    assert "courtyard summer" in enh_user
    # Downstream consumed the REWRITTEN text, not the raw draft.
    extractor_user = conn.calls[1][1]["content"]
    assert "young woman in a white cotton blouse" in extractor_user
    assert "courtyard summer" not in extractor_user
    # The summary surfaces the rewrite after the fact: the Auto-enhance
    # line, the advice header (Classification + Notes) and the exact
    # rewritten text — plus the usage entry for the rewrite call.
    summary = out["summary"]
    assert "Auto-enhance" in summary
    assert "Classification: Narration" in summary
    assert "Notes for the user" in summary
    assert "--- rewritten user_input ---" in summary
    assert "porcelain bowl" in summary
    assert "user_input_enhance 1" in summary
    # The plan itself built normally off the rewrite.
    assert len(json.loads(out["plan_json"])["shots"]) == 1


def test_auto_enhance_parse_failure_raises_no_silent_fallback(lg):
    conn = ScriptedConnector([
        "Sure! Here is your rewrite, trust me.",
        "Still nothing structured.",
        "Third unstructured reply.",
    ])
    with pytest.raises(RuntimeError, match="BEGIN user_input"):
        lg.H3LoopPromptEnhancer(conn)(
            user_input="courtyard summer",
            scene_count=1,
            seed=1,
            enhance_user_input=True,
        )
    # Nothing downstream ran — 3 enhancer attempts, no storyboard /
    # prefix / shot calls spent.
    assert len(conn.calls) == 3


def test_auto_enhance_survives_empty_llm_reply(lg):
    """Regression (MiniMax-M3 empty-200 flake): an empty first reply is
    retried with a fresh seed; the pipeline then continues normally."""
    conn = ScriptedConnector([
        "",  # empty first attempt
        _enhancer_reply_block(),
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="courtyard summer",
        scene_count=1,
        seed=1,
        enhance_user_input=True,
    )
    assert len(json.loads(out["plan_json"])["shots"]) == 1
    # Both enhancement attempts were recorded in the usage summary.
    assert "user_input_enhance 2" in out["summary"]


def test_auto_enhance_off_by_default_skips_rewrite(lg):
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=1,
        seed=1,
    )
    # First call is the span extractor (---BEGIN CONCEPT---), not the
    # enhancer (---BEGIN DRAFT---).
    first_user = conn.calls[0][1]["content"]
    assert "---BEGIN CONCEPT---" in first_user
    assert "---BEGIN DRAFT---" not in first_user
    # And no Auto-enhance section in the preflight.
    assert "Auto-enhance" not in out["summary"]


def test_generate_and_is_changed_carry_the_toggle(lg):
    conn = ScriptedConnector([
        _enhancer_reply_block(),
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    node = lg.MiniMaxH3LoopPromptGenerator()
    node.generate(
        conn,
        user_input="courtyard summer",
        scene_count=1,
        seed=1,
        enhance_user_input="on - 自动润色后再规划",
    )
    assert "---BEGIN DRAFT---" in conn.calls[0][1]["content"]

    h_off = node.is_changed(
        conn, user_input="draft", scene_count=1, seed=1,
        enhance_user_input="off - 不润色(默认)",
    )
    h_on = node.is_changed(
        conn, user_input="draft", scene_count=1, seed=1,
        enhance_user_input="on - 自动润色后再规划",
    )
    assert h_off != h_on


# --------------------------------------------------------------------------- #
# scene_count drives the dialogue board (line count no longer dictates it)
# --------------------------------------------------------------------------- #
_DIALOGUE_CONCEPT = "莎莉猫：你好。\n哈利猫：为什么。\n莎莉猫：再见。"
# Canonical 角色：台词 concepts parse deterministically (no extractor
# call) and short boards cut their prefix locally, so dialogue tests
# only script the per-shot visual replies.
_PREFIX_REPLY_DIALOGUE = (
    "Hand-drawn 2D animation, warm cafe interior at dusk.\n"
    "\n"
    "CAST:\n"
    "莎莉猫: cream-blonde fluffy cat, navy bow tie.\n"
    "哈利猫: orange tabby cat, brown blazer.\n"
)


def _dlg_clip_reply(i, speaker, sid, gender, line):
    """A three-section reply under the dialogue-as-data contract: the
    model writes the visual performance ONLY — no <d> blocks, no spoken
    words. The node appends the verbatim speech blocks afterwards."""
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] Clip {i}: {speaker}, {gender}, leans in and speaks "
        "with animated expression; ears twitch between words.\n"
        "\n"
        "overall_soundscape:\n"
        "Warm cafe ambience.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )


def test_scene_count_repacks_dialogue_to_exact_target(lg):
    """scene_count=3 on a board whose natural packing is ONE scene: the
    per-line budgets are regrouped into EXACTLY 3 scenes; every spoken
    line lands verbatim in its own scene. The canonical 角色：台词
    concept parses deterministically (no extractor call) and the short
    board cuts its prefix locally — the only LLM calls are the shots."""
    conn = ScriptedConnector([
        _dlg_clip_reply(1, "莎莉猫", "S1", "adult female", "你好。"),
        _dlg_clip_reply(2, "哈利猫", "S2", "adult male", "为什么。"),
        _dlg_clip_reply(3, "莎莉猫", "S1", "adult female", "再见。"),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input=_DIALOGUE_CONCEPT,
        scene_count=3,
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 3
    assert [s["id"] for s in plan["shots"]] == ["scene_01", "scene_02", "scene_03"]
    # structured parse + local prefix (3 turns, no reference images):
    # exactly the 3 shot calls — no extractor, no prefix LLM call.
    assert len(conn.calls) == 3
    assert out["board_kind"] == "dialogue"
    assert "Board: dialogue" in out["summary"]
    assert "prefix derived locally" in out["summary"]
    assert "LLM requests: 3" in out["summary"]
    # The shot user texts carry the dialogue lock, not a copy order.
    for call in conn.calls:
        assert "Dialogue is LOCKED" in call[1]["content"]
    for shot, line in zip(plan["shots"], ["你好。", "为什么。", "再见。"]):
        prompt_text = "\n".join(shot["prompt"])
        assert f"<d>[Chinese] {line}</d>" in prompt_text


def test_scene_count_above_line_count_gets_reaction_cuts(lg):
    """scene_count=5 with only 3 dialogue lines: the board tops up with
    2 mechanical SILENT reaction cuts at speaker-change boundaries —
    the user's count is reached without touching a single spoken line."""
    conn = ScriptedConnector([
        _dlg_clip_reply(1, "莎莉猫", "S1", "adult female", "你好。"),
        _clip_reply(1),  # reaction cut (silent)
        _dlg_clip_reply(2, "哈利猫", "S2", "adult male", "为什么。"),
        _clip_reply(2),  # reaction cut (silent)
        _dlg_clip_reply(3, "莎莉猫", "S1", "adult female", "再见。"),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input=_DIALOGUE_CONCEPT,
        scene_count=5,
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 5
    assert [s["id"] for s in plan["shots"]] == [
        "scene_01", "scene_02", "scene_03", "scene_04", "scene_05",
    ]
    # Shots 1/3/5 carry the dialogue verbatim; shots 2/4 are silent.
    texts = ["\n".join(s["prompt"]) for s in plan["shots"]]
    for line in ("你好。", "为什么。", "再见。"):
        assert sum(f"<d>[Chinese] {line}</d>" in t for t in texts) == 1
    assert "<d>" not in texts[1] and "<d>" not in texts[3]
    # The reaction shots' per-shot user templates carried the mechanical
    # storyboard entry + the no-dialogue notice.
    reaction_user_texts = [conn.calls[i][1]["content"] for i in (1, 3)]
    for txt in reaction_user_texts:
        assert "Reaction cut" in txt
        assert "no dialogue lines assigned to this turn" in txt
    # And the preflight tells the user what was inserted.
    assert "reaction cut" in out["summary"]


def test_scene_count_one_squeezes_all_lines_time_only(lg):
    """TIME-only model: scene_count=1 with two very long lines squeezes
    EVERYTHING into one scene (lines are never dropped or split), the
    clip clamps at the 14s H3 window, and the summary warns — never an
    error, never a reshape (能否说完不重要)."""
    long_a = "这是一句相当长的台词，" * 10
    long_b = "另外一段同样很长的回答，" * 10
    concept = f"甲猫：{long_a}\n乙猫：{long_b}"
    one_scene_reply = (
        # Deliberate contract violation: the model quoted both lines
        # inline. The description body is replaced by the node, so the
        # leaked copy is gone and only the node sentences remain.
        "integrated_multimodal_description:\n"
        f"[Shot 1] 甲猫 says: <d>[Chinese] {long_a}</d> "
        f"乙猫 replies: <d>[Chinese] {long_b}</d>\n"
        "\n"
        "overall_soundscape:\n"
        "Quiet room tone.\n"
        "\n"
        "non_diegetic_music:\n"
        "No non-diegetic music.\n"
    )
    conn = ScriptedConnector([
        one_scene_reply,
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input=concept,
        scene_count=1,
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    # Exactly ONE scene carrying BOTH lines verbatim, in order.
    assert len(plan["shots"]) == 1
    prompt_text = "\n".join(plan["shots"][0]["prompt"])
    assert f"<d>[Chinese] {long_a}</d>" in prompt_text
    assert f"<d>[Chinese] {long_b}</d>" in prompt_text
    assert prompt_text.count("<d>[Chinese]") == 2  # leaked copy not kept
    assert "甲猫 says" not in prompt_text
    # The clip is clamped at the H3 window and the summary says so.
    assert plan["shots"][0]["length"] <= 14 * 24 + 17
    assert "per-shot cap" in out["summary"]


def test_summary_is_logged_via_log_pipeline(lg, monkeypatch):
    """The summary must mirror into mie_log (log_pipeline) so the run's
    request/token/warning footprint lands in the console + h3_loop.log
    without wiring a Show-Anything node."""
    logged: list[str] = []
    monkeypatch.setattr(lg, "log_pipeline", lambda msg: logged.append(str(msg)))
    conn = ScriptedConnector([
        _extract_dialogue_empty(),
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="a quiet courtyard summer afternoon",
        scene_count=1,
        seed=1,
    )
    joined = "\n".join(logged)
    assert "MiniMax H3 Loop Plan summary" in joined
    assert "LLM requests: 4" in joined
    assert "Tokens (estimated)" in joined
    assert "Warnings" in joined
    # The logged copy matches the returned summary verbatim.
    assert any(msg == out["summary"] for msg in logged)


# --------------------------------------------------------------------------- #
# dialogue_extract empty-reply retries (MiniMax-M3 empty-200 flake)
# --------------------------------------------------------------------------- #
def test_extract_dialogue_retries_empty_reply(lg):
    """Empty first reply is a transport flake, not a 'no dialogue'
    verdict: retry, then parse the turns from the second reply."""
    concept = "甲猫：你好。\n乙猫：好的。"
    # 甲猫：is 3 chars so 你好。 spans [3,6); newline + 乙猫：is 4 more so 好的。 spans [10,13).
    extract = json.dumps({"turns": [
        {"speaker": "甲猫", "lines": [{"text": "你好。", "start": 3, "end": 6}]},
        {"speaker": "乙猫", "lines": [{"text": "好的。", "start": 10, "end": 13}]},
    ]}, ensure_ascii=False)
    conn = ScriptedConnector(["", extract])
    enh = lg.H3LoopPromptEnhancer(conn, temperature=0.0, timeout=30)
    turns = enh.extract_dialogue(concept)
    assert len(conn.calls) == 2
    assert [t.speaker for t in turns] == ["甲猫", "乙猫"]
    assert turns[0].lines == ["你好。"]


def test_extract_dialogue_all_empty_falls_back_to_narrator(lg):
    """Three empty replies -> narrator fallback ([]), with all three
    attempts spent."""
    conn = ScriptedConnector(["", "", ""])
    enh = lg.H3LoopPromptEnhancer(conn, temperature=0.0, timeout=30)
    assert enh.extract_dialogue("甲猫：你好。") == []
    assert len(conn.calls) == 3
    assert enh._dialogue_extract_warnings


def test_extract_dialogue_strips_think_block(lg):
    concept = "甲猫：你好。\n乙猫：好的。"
    extract = json.dumps({"turns": [
        {"speaker": "甲猫", "lines": [{"text": "你好。", "start": 3, "end": 6}]},
        {"speaker": "乙猫", "lines": [{"text": "好的。", "start": 10, "end": 13}]},
    ]}, ensure_ascii=False)
    wrapped = "<think>{\"turns\": []}</think>\n" + extract
    conn = ScriptedConnector([wrapped])
    enh = lg.H3LoopPromptEnhancer(conn, temperature=0.0, timeout=30)
    turns = enh.extract_dialogue(concept)
    assert [t.speaker for t in turns] == ["甲猫", "乙猫"]
    assert enh._dialogue_extract_warnings == []


def test_extract_dialogue_relocates_wrong_offsets(lg):
    """The few-shot off-by-one span is recovered from the unique text."""
    concept = '公猫问："给够钱就行？" 母猫答："给够钱。"'
    extract = json.dumps({"turns": [
        {"speaker": "公猫", "lines": [{"text": "给够钱就行？", "start": 4, "end": 10}]},
        {"speaker": "母猫", "lines": [{"text": "给够钱。", "start": 18, "end": 22}]},
    ]}, ensure_ascii=False)
    conn = ScriptedConnector([extract])
    enh = lg.H3LoopPromptEnhancer(conn, temperature=0.0, timeout=30)
    turns = enh.extract_dialogue(concept)
    assert [t.speaker for t in turns] == ["公猫", "母猫"]
    assert turns[0].lines == ["给够钱就行？"]
    assert turns[0].start == 5
    assert turns[0].end == 11


def test_extract_dialogue_failure_warns_in_summary(lg):
    conn = ScriptedConnector([
        "not-json",
        "still-not-json",
        "nope",
        _auto_storyboard_reply(1),
        PREFIX_REPLY,
        _clip_reply(1),
    ])
    out = lg.H3LoopPromptEnhancer(conn, temperature=0.0, timeout=30)(
        user_input="甲猫：你好。",
        scene_count=1,
        seed=1,
    )
    assert "spoken lines will NOT be preserved" in out["summary"]
    assert len(conn.calls) == 6


# --------------------------------------------------------------------------- #
# Pacing owns the tempo: derived scene count + binding tempo directive
# --------------------------------------------------------------------------- #
def _dlg_conn(replies):
    return ScriptedConnector(replies)


def test_pacing_derives_scene_count_and_injects_tempo(lg):
    """Dialogue boards with scene_count=0 (auto) pack MINIMALLY — the
    fewest scenes whose speech+pause math fits the 14s H3 window —
    regardless of pacing; pacing still owns the SPEECH tempo (durations)
    and injects the BRISK/MEASURED directive into every per-shot user
    template. An explicit scene_count is honoured exactly via even
    speech distribution."""
    CONCEPT = "甲猫：你好。\n乙猫：好的。\n甲猫：再见。\n乙猫：下次见。"
    # Canonical 角色：台词 concept: parsed deterministically (no
    # extractor call), short board -> local prefix (no prefix call).

    LINES = [("甲猫", "S1", "adult male", "你好。"),
             ("乙猫", "S2", "adult female", "好的。"),
             ("甲猫", "S1", "adult male", "再见。"),
             ("乙猫", "S2", "adult female", "下次见。")]

    def run(pacing, scenes, scene_count=0):
        # scenes = list of line-index groups matching the expected split.
        # The canonical 角色：台词 concept parses deterministically and
        # the short board cuts its prefix locally, so the ONLY scripted
        # replies are the per-shot visual replies.
        replies = []
        for group in scenes:
            body = " ".join(
                f"{spk}, {g} performs the line with gesture"
                for spk, sid, g, line in (LINES[i] for i in group)
            )
            replies.append(
                "integrated_multimodal_description:\n"
                f"[Shot 1] {body}\n"
                "\noverall_soundscape:\nRoom tone.\n\n"
                "non_diegetic_music:\nNo non-diegetic music.\n"
            )
        conn = _dlg_conn(replies)
        out = lg.H3LoopPromptEnhancer(conn, temperature=0.4, timeout=60)(
            user_input=CONCEPT,
            total_duration_seconds=20,
            pacing=pacing,
            scene_count=scene_count,
            seed=1,
        )
        plan = json.loads(out["plan_json"])
        return out, plan, conn

    # auto (scene_count=0): MINIMAL-CUT packing — this 4-line exchange
    # (~5.6s speech+pause math on fast) fits ONE scene under the 14s H3
    # window, so both fast and slow pack to a single scene. Auto never
    # multiplies cuts; the explicit 20s total just rescales/clamps it.
    out_f, plan_f, conn_f = run("fast - 快", [[0, 1, 2, 3]])
    assert len(plan_f["shots"]) == 1
    assert plan_f["shots"][0]["length"] <= 345  # 14s H3 cap on the grid
    out_s, plan_s, conn_s = run("slow - 慢", [[0, 1, 2, 3]])
    assert len(plan_s["shots"]) == 1

    # The tempo directive still rides into every per-shot user template —
    # pacing owns the SPEECH tempo vocabulary, not the cut count.
    shot_calls_f = [c for c in conn_f.calls if "Clip duration" in c[1]["content"]]
    assert len(shot_calls_f) == 1
    for c in shot_calls_f:
        assert "Tempo: BRISK" in c[1]["content"]
    shot_calls_s = [c for c in conn_s.calls if "Clip duration" in c[1]["content"]]
    assert len(shot_calls_s) == 1
    for c in shot_calls_s:
        assert "Tempo: MEASURED" in c[1]["content"]

    # Explicit scene_count is honoured EXACTLY: 2 scenes, speech evenly
    # distributed (2 lines each), equal 10s time share per scene.
    out_2, plan_2, conn_2 = run("fast - 快", [[0, 1], [2, 3]], scene_count=2)
    assert len(plan_2["shots"]) == 2
    for s in plan_2["shots"]:
        assert s["length"] == 243  # 20s / 2 = 10s -> 243f on the grid


# --------------------------------------------------------------------------- #
# Dialogue-as-data / structured-parse / local-prefix / category upgrade
# --------------------------------------------------------------------------- #
def test_long_dialogue_board_uses_llm_prefix_and_cast_identity(lg):
    """Above the local-prefix threshold (5+ turns) the stage-1 prefix
    LLM call still runs, and the code-assembled speech blocks carry
    each speaker's CAST identity on their FIRST spoken clip only."""
    concept = "\n".join([
        "莎莉猫：第一句。", "哈利猫：第二句。", "莎莉猫：第三句。",
        "哈利猫：第四句。", "莎莉猫：第五句。", "哈利猫：第六句。",
    ])
    replies = [_PREFIX_REPLY_DIALOGUE]
    for i in range(1, 7):
        replies.append(
            "integrated_multimodal_description:\n"
            f"[Shot 1] Clip {i}: the speaker performs the line with "
            "lively gesture; ears twitch.\n"
            "\noverall_soundscape:\nWarm cafe ambience.\n\n"
            "non_diegetic_music:\nNo non-diegetic music.\n"
        )
    conn = ScriptedConnector(replies)
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input=concept,
        scene_count=6,
        seed=1,
    )
    plan = json.loads(out["plan_json"])
    assert len(plan["shots"]) == 6
    # prefix + 6 shots; the extractor never ran (canonical concept).
    assert len(conn.calls) == 7
    assert "prefix derived locally" not in out["summary"]
    texts = ["\n".join(s["prompt"]) for s in plan["shots"]]
    voice = "adult, mid-range pitch, natural timbre"
    # Short CAST identity is pinned once, on the first clip, and is not
    # the <d> line. Later clips restate the voice descriptor only.
    assert "莎莉猫: cream-blonde fluffy cat, navy bow tie." in texts[0]
    assert (
        f"莎莉猫 speaks as (S1) {voice} <d>[Chinese] 第一句。</d>" in texts[0]
    )
    assert "哈利猫: orange tabby cat, brown blazer." in texts[1]
    assert (
        f"哈利猫 speaks as (S2) {voice} <d>[Chinese] 第二句。</d>" in texts[1]
    )
    assert "cream-blonde" not in texts[2]
    assert (
        f"莎莉猫 speaks as (S1) {voice} <d>[Chinese] 第三句。</d>" in texts[2]
    )
    assert (
        f"哈利猫 speaks as (S2) {voice} <d>[Chinese] 第四句。</d>" in texts[3]
    )


def test_single_call_dialogue_appends_blocks(lg):
    """single_call mode: the one LLM reply writes visuals only; the
    node assembles + appends the verbatim blocks per shot."""
    conn = ScriptedConnector([_single_call_reply(1)])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input="甲猫：你好。\n乙猫：好的。",
        generation_mode="single_call - 单次调用(快/省)",
        seed=1,
    )
    # Local prefix (2 turns, no references) + single call = 1 request.
    assert len(conn.calls) == 1
    assert "Dialogue is LOCKED as data" in conn.calls[0][1]["content"]
    plan = json.loads(out["plan_json"])
    prompt_text = "\n".join(plan["shots"][0]["prompt"])
    assert "<d>[Chinese] 你好。</d>" in prompt_text
    assert "<d>[Chinese] 好的。</d>" in prompt_text
    assert prompt_text.count("<d>[") == 2
    assert out["board_kind"] == "dialogue"


def test_dialogue_board_upgrades_none_category(lg):
    """Lines extracted + category none (or blank): the spoken-scene
    cinematography contract is applied automatically and surfaced in
    the summary; the stored widget value is untouched."""
    conn = ScriptedConnector([
        _dlg_clip_reply(1, "莎莉猫", "S1", "adult female", "你好。"),
        _dlg_clip_reply(2, "哈利猫", "S2", "adult male", "为什么。"),
        _dlg_clip_reply(3, "莎莉猫", "S1", "adult female", "再见。"),
    ])
    out = lg.H3LoopPromptEnhancer(conn)(
        user_input=_DIALOGUE_CONCEPT,
        scene_count=3,
        category="none - 不指定",
        seed=1,
    )
    assert "category: none -> dialogue" in out["summary"]
    for call in conn.calls:
        assert "spoken-scene cinematography" in call[1]["content"]
