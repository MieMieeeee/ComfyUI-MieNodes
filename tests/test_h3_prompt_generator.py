# -*- coding: utf-8 -*-
"""Tests for the H3PromptEnhancer behavior and the node's ``is_changed``.

Uses the same ``_mienodes_internal`` loader-stub injection as
``test_scail2_prompts.py`` (the project root has a hyphen and cannot be
imported as a normal package).

Coverage:
* Frame-sampling helpers (``_sample_indices`` / ``_sample_urls``).
* ``is_changed`` is a stable hash that varies when meaningful inputs
  change but is robust to identical re-runs.
* ``H3PromptEnhancer.__call__`` graceful degradation: returns the original
  user_prompt on empty idea, on missing required media per task, and on
  an unknown task code.
* ``__call__`` happy path: T2V runs a single ``invoke`` (no caption);
  ``reference`` runs two ``invoke`` calls in caption-then-enhance order
  and returns the stage-2 text.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
H3_GEN_PATH = PROJECT_DIR / "nodes" / "llm" / "h3_prompt_generator.py"
H3_PROMPTS_PATH = PROJECT_DIR / "nodes" / "llm" / "h3_prompts.py"
UTILS_PATH = PROJECT_DIR / "core" / "utils.py"
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"


def _load_h3():
    """Inject a fake ``_mienodes_internal`` package tree and load the two
    h3 modules. Mirrors ``test_h3_prompts._load_h3``."""
    if "_mienodes_internal" not in sys.modules:
        ip = types.ModuleType("_mienodes_internal")
        ip.__path__ = [str(PROJECT_DIR)]
        ip.__package__ = "_mienodes_internal"
        sys.modules["_mienodes_internal"] = ip
    if "_mienodes_internal.core" not in sys.modules:
        core = types.ModuleType("_mienodes_internal.core")
        core.__path__ = [str(PROJECT_DIR / "core")]
        core.__package__ = "_mienodes_internal.core"
        sys.modules["_mienodes_internal.core"] = core
    if "_mienodes_internal.core.utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "_mienodes_internal.core.utils", str(UTILS_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_mienodes_internal.core.utils"] = mod
        spec.loader.exec_module(mod)
    if "_mienodes_internal.nodes" not in sys.modules:
        n = types.ModuleType("_mienodes_internal.nodes")
        n.__path__ = [str(PROJECT_DIR / "nodes")]
        n.__package__ = "_mienodes_internal.nodes"
        sys.modules["_mienodes_internal.nodes"] = n
    if "_mienodes_internal.nodes.llm" not in sys.modules:
        nllm = types.ModuleType("_mienodes_internal.nodes.llm")
        nllm.__path__ = [str(PROJECT_DIR / "nodes" / "llm")]
        nllm.__package__ = "_mienodes_internal.nodes.llm"
        sys.modules["_mienodes_internal.nodes.llm"] = nllm
    if "_mienodes_internal.nodes.llm.prompts" not in sys.modules:
        np_ = types.ModuleType("_mienodes_internal.nodes.llm.prompts")
        np_.__path__ = [str(PROMPTS_DIR)]
        sys.modules["_mienodes_internal.nodes.llm.prompts"] = np_
    if "_mienodes_internal.nodes.llm.prompts.loader" not in sys.modules:
        loader_spec = importlib.util.spec_from_file_location(
            "_mienodes_internal.nodes.llm.prompts.loader",
            str(PROMPTS_DIR / "loader.py"),
        )
        loader_mod = importlib.util.module_from_spec(loader_spec)
        sys.modules["_mienodes_internal.nodes.llm.prompts.loader"] = loader_mod
        loader_spec.loader.exec_module(loader_mod)
    for name in ("h3_prompts", "h3_prompt_generator"):
        full = f"_mienodes_internal.nodes.llm.{name}"
        if full in sys.modules:
            del sys.modules[full]
        path = H3_PROMPTS_PATH if name == "h3_prompts" else H3_GEN_PATH
        spec = importlib.util.spec_from_file_location(full, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return (
        sys.modules["_mienodes_internal.nodes.llm.h3_prompts"],
        sys.modules["_mienodes_internal.nodes.llm.h3_prompt_generator"],
    )


@pytest.fixture(scope="module")
def h3():
    return _load_h3()


class FakeConnector:
    """Minimal connector double: exposes ``get_state`` + ``model`` so
    ``is_changed`` and ``mie_log`` work without a real LLM backend."""

    model = "fake-model"

    def get_state(self):
        return "fake-state"


# --------------------------------------------------------------------------- #
# Frame-sampling helpers
# --------------------------------------------------------------------------- #
def test_sample_indices(h3):
    _, gen = h3
    assert gen._sample_indices(0, 5) == []
    assert gen._sample_indices(10, 1) == [5]
    assert gen._sample_indices(10, 2) == [0, 9]
    # Even spacing: total=10, n=5 -> [0, 2, 4, 7, 9].
    assert gen._sample_indices(10, 5) == [0, 2, 4, 7, 9]
    # Clamping.
    assert len(gen._sample_indices(3, 99)) == 3
    assert gen._sample_indices(1, 8) == [0]


def test_sample_urls_dedupes_endpoints(h3):
    _, gen = h3
    urls = [f"http://x/{i}" for i in range(4)]
    assert gen._sample_urls(urls, 1) == ["http://x/2"]
    assert gen._sample_urls(urls, 2) == ["http://x/0", "http://x/3"]
    assert gen._sample_urls([], 5) == []


# --------------------------------------------------------------------------- #
# is_changed
# --------------------------------------------------------------------------- #
def test_is_changed_stable_and_seed_sensitive(h3):
    _, gen = h3
    node = gen.MiniMaxH3PromptGenerator()
    kwargs = dict(
        task_type="t2v - 文生视频",
        user_prompt="hello",
        seed=0,
        width=1280,
        height=720,
        duration=6.0,
        category="none - 不指定",
        caption_sample_frames=8,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=4096,
        max_tokens_enhance=8192,
        timeout=120,
    )
    a = node.is_changed(FakeConnector(), **kwargs)
    b = node.is_changed(FakeConnector(), **kwargs)
    assert a == b  # same inputs -> same hash
    # Different seed -> different hash.
    kwargs["seed"] = 1
    c = node.is_changed(FakeConnector(), **kwargs)
    assert a != c


def test_is_changed_senses_category_duration_aspect(h3):
    """Changing category / duration / width / height must change the hash
    (these are output-affecting scalars)."""
    _, gen = h3
    node = gen.MiniMaxH3PromptGenerator()
    base = dict(
        llm_service_connector=FakeConnector(),
        task_type="t2v - 文生视频",
        user_prompt="hello",
        seed=0,
        width=1280,
        height=720,
        duration=6.0,
        category="none - 不指定",
        caption_sample_frames=8,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=4096,
        max_tokens_enhance=8192,
        timeout=120,
    )
    h0 = node.is_changed(**base)

    base2 = dict(base, category="cinematic-story - 电影短片/MV/戏剧")
    assert node.is_changed(**base2) != h0

    base3 = dict(base, duration=8.0)
    assert node.is_changed(**base3) != h0

    # width change that flips the ratio (16:9 -> 9:16) must change the hash.
    base4 = dict(base, width=720, height=1280)
    assert node.is_changed(**base4) != h0


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #
def test_enhancer_unknown_task_returns_original(h3):
    _, gen = h3
    enhancer = gen.H3PromptEnhancer(FakeConnector())
    out = enhancer("not_a_task - ???", "some idea")
    assert out == "some idea"


def test_default_idea_per_task(h3):
    """``_default_idea`` returns a non-empty, task-specific fallback."""
    _, gen = h3
    for code in ("t2v", "i2v_first", "i2v_first_last", "i2v_last", "reference", "s2v"):
        idea = gen._default_idea(code, "none - 不指定")
        assert idea and idea.strip(), f"empty default idea for {code!r}"
    # Category hint is appended when a non-'none' category is chosen.
    idea_cat = gen._default_idea("t2v", "action - 动作戏/打斗/飙车")
    assert "action" in idea_cat
    # 'none' / unknown categories produce no category hint.
    assert "for a none video" not in gen._default_idea("t2v", "none - 不指定")
    assert "for a not-a-cat video" not in gen._default_idea("t2v", "not-a-cat")


def test_enhancer_empty_prompt_runs_pipeline(h3, monkeypatch):
    """An empty user_prompt no longer short-circuits: the enhancer
    synthesizes a default idea and runs the full pipeline."""
    _, gen = h3
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(max_tokens)
            return "An H3 prompt synthesized from a default idea."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls", lambda t: [] if t is None else ["data:image/jpeg;base64,AAAA"]
    )
    enhancer = gen.H3PromptEnhancer(_Conn())
    # Empty string and whitespace-only both trigger the fallback.
    out = enhancer("t2v - 文生视频", "")
    assert out == "An H3 prompt synthesized from a default idea."
    assert len(invocations) == 1
    out_ws = enhancer("t2v - 文生视频", "   \n  ")
    assert out_ws == "An H3 prompt synthesized from a default idea."
    assert len(invocations) == 2


def test_enhancer_i2v_first_missing_first_frame(h3):
    _, gen = h3
    enhancer = gen.H3PromptEnhancer(FakeConnector())
    out = enhancer("i2v_first - 图生视频(首帧)", "she walks away", first_frame=None)
    assert out == "she walks away"


def test_enhancer_i2v_first_last_missing_one_frame(h3):
    _, gen = h3
    enhancer = gen.H3PromptEnhancer(FakeConnector())
    # Only first_frame supplied.
    out = enhancer(
        "i2v_first_last - 图生视频(首尾帧)", "transition",
        first_frame=object(), last_frame=None,
    )
    assert out == "transition"
    # Only last_frame supplied.
    out = enhancer(
        "i2v_first_last - 图生视频(首尾帧)", "transition",
        first_frame=None, last_frame=object(),
    )
    assert out == "transition"


def test_enhancer_i2v_last_missing_last_frame(h3):
    _, gen = h3
    enhancer = gen.H3PromptEnhancer(FakeConnector())
    out = enhancer("i2v_last - 图生视频(尾帧)", "approach the end", last_frame=None)
    assert out == "approach the end"


def test_enhancer_reference_missing_all_media(h3):
    _, gen = h3
    enhancer = gen.H3PromptEnhancer(FakeConnector())
    out = enhancer(
        "reference - 全能参考", "make them dance",
        reference_images=None, reference_video=None,
    )
    assert out == "make them dance"


def test_enhancer_s2v_missing_reference_images(h3):
    _, gen = h3
    enhancer = gen.H3PromptEnhancer(FakeConnector())
    out = enhancer("s2v - 主体参考", "she dances", reference_images=None)
    assert out == "she dances"


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
class _FakeTensor:
    """Stand-in "tensor": only truthiness matters once the data-url helper
    is monkeypatched to return canned URLs."""


def test_enhancer_t2v_single_invoke(h3, monkeypatch):
    """T2V has no media and no caption: exactly one ``invoke`` (enhance)."""
    _, gen = h3
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(max_tokens)
            return "An H3 T2VA structured prompt."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls", lambda t: [] if t is None else ["data:image/jpeg;base64,AAAA"]
    )
    enhancer = gen.H3PromptEnhancer(_Conn())
    out = enhancer("t2v - 文生视频", "a neon city", seed=7)
    assert out == "An H3 T2VA structured prompt."
    assert len(invocations) == 1


def test_enhancer_reference_two_stage_pipeline(h3, monkeypatch):
    """The ``reference`` path runs stage-1 caption then stage-2 enhance,
    in that order, and returns the stage-2 text."""
    _, gen = h3
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(max_tokens)
            if len(invocations) == 1:
                return "A caption of the reference material."
            return "An H3 Ref2VA structured prompt."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,AAAA"] * 4 if t is not None else [],
    )
    enhancer = gen.H3PromptEnhancer(_Conn(), caption_sample_frames=8)
    out = enhancer(
        "reference - 全能参考", "make them dance",
        reference_images=_FakeTensor(),
        reference_video=_FakeTensor(),
        seed=42,
    )
    assert out == "An H3 Ref2VA structured prompt."
    assert len(invocations) == 2


def test_enhancer_reference_empty_caption_short_circuits(h3, monkeypatch):
    """If stage-1 caption returns empty, the enhancer returns the original
    prompt instead of running stage 2."""
    _, gen = h3
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(max_tokens)
            return ""  # empty response

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,AAAA"] * 4 if t is not None else [],
    )
    enhancer = gen.H3PromptEnhancer(_Conn())
    out = enhancer(
        "reference - 全能参考", "make them dance",
        reference_images=_FakeTensor(), reference_video=_FakeTensor(),
    )
    assert out == "make them dance"
    assert len(invocations) == 1  # only the caption attempt


def test_enhancer_i2v_first_single_invoke(h3, monkeypatch):
    """I2V single-frame is single-stage (no caption): one ``invoke``."""
    _, gen = h3
    invocations = []

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            invocations.append(max_tokens)
            return "An H3 I2VA structured prompt."

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,AAAA"] if t is not None else [],
    )
    enhancer = gen.H3PromptEnhancer(_Conn())
    out = enhancer(
        "i2v_first - 图生视频(首帧)", "she walks away",
        first_frame=_FakeTensor(), seed=1,
    )
    assert out == "An H3 I2VA structured prompt."
    assert len(invocations) == 1


def test_enhancer_empty_llm_response_falls_back(h3, monkeypatch):
    """An empty stage-2 response falls back to the original user_prompt."""
    _, gen = h3

    class _Conn(FakeConnector):
        def invoke(self, messages, *, seed, temperature, max_tokens):
            return ""

    monkeypatch.setattr(
        gen, "image_tensor_batch_to_data_urls", lambda t: [] if t is None else ["data:image/jpeg;base64,AAAA"]
    )
    enhancer = gen.H3PromptEnhancer(_Conn())
    out = enhancer("t2v - 文生视频", "a neon city")
    assert out == "a neon city"
