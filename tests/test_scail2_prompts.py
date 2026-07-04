# -*- coding: utf-8 -*-
"""Tests for the Scail2 prompt templates and helpers.

Mirrors the loader-stubbing pattern from ``test_bernini_timeout_override.py``
so the project layout (root has hyphens, cannot be imported as a normal
package) keeps working under pytest.

Coverage:
* All 6 prompt files load via the project's ``load_prompt_text``.
* ``caption_replacement`` / ``enhance_replacement`` are byte-level copies
  of the upstream ``prompt_enhancer.py`` contents (as of 2026-06).
* ``enhance_replacement_prompt`` / ``enhance_motion_transfer_prompt``
  ``.format(...)`` cleanly with their expected kwargs and embed the
  supplied caption / hint.
* ``bundled_examples_replacement`` returns the upstream two examples.
* ``parse_task_code`` handles display strings, bare codes, and None.
* ``Scail2PromptGenerator`` is importable, has the right INPUT_TYPES
  shape, and exposes the expected ports.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCAIL2_GEN_PATH = PROJECT_DIR / "nodes" / "llm" / "scail2_prompt_generator.py"
SCAIL2_PROMPTS_PATH = PROJECT_DIR / "nodes" / "llm" / "scail2_prompts.py"
UTILS_PATH = PROJECT_DIR / "core" / "utils.py"
PROMPTS_DIR = PROJECT_DIR / "nodes" / "llm" / "prompts"

def _load_scail2():
    """Inject a fake ``_mienodes_internal`` package tree and load the two
    scail2 modules so tests can ``import`` them like the rest of the
    project does at runtime."""
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
    for name in ("scail2_prompts", "scail2_prompt_generator"):
        full = f"_mienodes_internal.nodes.llm.{name}"
        if full in sys.modules:
            del sys.modules[full]
        path = SCAIL2_PROMPTS_PATH if name == "scail2_prompts" else SCAIL2_GEN_PATH
        spec = importlib.util.spec_from_file_location(full, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return (
        sys.modules["_mienodes_internal.nodes.llm.scail2_prompts"],
        sys.modules["_mienodes_internal.nodes.llm.scail2_prompt_generator"],
    )


@pytest.fixture(scope="module")
def scail2():
    return _load_scail2()


# --------------------------------------------------------------------------- #
# Prompt file / template tests
# --------------------------------------------------------------------------- #
def test_prompt_files_exist():
    """All 6 prompt .txt files + UPSTREAM.md must be on disk."""
    expected = {
        "UPSTREAM.md",
        "caption_replacement.txt",
        "enhance_replacement.txt",
        "examples_replacement.txt",
        "caption_motion_transfer.txt",
        "enhance_motion_transfer.txt",
        "examples_motion_transfer.txt",
    }
    on_disk = {p.name for p in (PROMPTS_DIR / "scail2").iterdir()}
    missing = expected - on_disk
    assert not missing, f"missing prompt files: {missing}"


def test_examples_replacement_matches_upstream():
    """``examples_replacement.txt`` is a verbatim copy of the upstream
    2-example ``prompt_examples.txt`` (1584 bytes)."""
    text = (PROMPTS_DIR / "scail2" / "examples_replacement.txt").read_text(encoding="utf-8")
    assert "woodworking workshop" in text
    assert "medical worker is washing his hands" in text.lower()
    # Upstream file is 1584 bytes; allow a trailing-newline tolerance.
    assert 1580 <= len(text) <= 1590, f"unexpected size {len(text)}"


def test_caption_replacement_matches_upstream(scail2):
    """The replacement stage-1 caption system prompt is the upstream
    ``VIDEO_CAPTION_PROMPT`` verbatim."""
    prompts, _ = scail2
    text = prompts.caption_replacement_prompt()
    assert "You are captioning sampled frames" in text
    assert "character replacement video generation task" in text
    assert "Do not mention the replacement target image" in text
    assert text.strip() == text  # no trailing whitespace


def test_enhance_replacement_prompt_formatting(scail2):
    """The replacement stage-2 template formats cleanly and embeds inputs."""
    prompts, _ = scail2
    rendered = prompts.enhance_replacement_prompt(
        instruction="replace the man with the woman from the image",
        caption="A man is doing woodworking at a bench.",
        examples="example-one\nexample-two",
    )
    assert "replace the man with the woman from the image" in rendered
    assert "A man is doing woodworking at a bench." in rendered
    assert "example-one" in rendered
    # The 8 rules are preserved (upstream has 8 numbered rules).
    assert rendered.count("\n8. ") >= 1


def test_enhance_replacement_falls_back_when_examples_empty(scail2):
    """If ``examples`` is empty/falsy, the template substitutes the
    upstream default sentinel ``(No examples provided.)``."""
    prompts, _ = scail2
    rendered = prompts.enhance_replacement_prompt(
        instruction="x", caption="y", examples=""
    )
    assert "(No examples provided.)" in rendered


def test_enhance_motion_transfer_prompt_formatting(scail2):
    """The motion-transfer stage-2 template formats cleanly."""
    prompts, _ = scail2
    rendered = prompts.enhance_motion_transfer_prompt(
        caption="a person spins twice and smiles",
        user_hint="in slow motion",
        examples="example-a",
    )
    assert "a person spins twice and smiles" in rendered
    assert "in slow motion" in rendered
    assert "example-a" in rendered
    # 8 rules preserved.
    assert rendered.count("\n8. ") >= 1


def test_enhance_motion_transfer_empty_user_hint(scail2):
    """Empty user_hint must NOT be passed through as an empty literal;
    the template substitutes the default hint so the LLM knows to
    derive motion from the caption."""
    prompts, _ = scail2
    rendered = prompts.enhance_motion_transfer_prompt(
        caption="caption-text", user_hint="", examples="x"
    )
    assert "No additional user hint" in rendered
    assert "caption-text" in rendered


def test_enhance_motion_transfer_user_hint_whitespace_only(scail2):
    """Whitespace-only user_hint is treated the same as empty."""
    prompts, _ = scail2
    rendered = prompts.enhance_motion_transfer_prompt(
        caption="caption-text", user_hint="   \n  ", examples="x"
    )
    assert "No additional user hint" in rendered


def test_caption_motion_transfer_content(scail2):
    """The motion-transfer stage-1 caption system prompt exists and
    focuses on motion / body mechanics."""
    prompts, _ = scail2
    text = prompts.caption_motion_transfer_prompt()
    assert "driving video" in text
    assert "motion" in text.lower()
    assert "body mechanics" in text


def test_parse_task_code(scail2):
    prompts, _ = scail2
    assert prompts.parse_task_code("character_replacement - 角色替换") == "character_replacement"
    assert prompts.parse_task_code("motion_transfer - 动作迁移") == "motion_transfer"
    assert prompts.parse_task_code("character_replacement") == "character_replacement"
    assert prompts.parse_task_code("") == ""
    assert prompts.parse_task_code(None) is None


def test_bundled_examples_helpers(scail2):
    """``load_bundled_examples`` returns non-empty text for both tasks."""
    prompts, _ = scail2
    assert "woodworking" in prompts.load_bundled_examples("character_replacement")
    # motion_transfer ships with MieNodes-original few-shot examples.
    assert prompts.load_bundled_examples("motion_transfer")
    # Unknown code returns empty.
    assert prompts.load_bundled_examples("not_a_task") == ""


# --------------------------------------------------------------------------- #
# Frame-sampling tests
# --------------------------------------------------------------------------- #
def test_sample_indices(scail2):
    _, gen = scail2
    # Basic shape.
    assert gen._sample_indices(0, 5) == []
    assert gen._sample_indices(10, 1) == [5]
    assert gen._sample_indices(10, 2) == [0, 9]
    # Even spacing; for total=10, n=5 the rounded indices are
    # [0, round(9/4)=2, round(18/4)=4 (banker), round(27/4)=7, 9].
    assert gen._sample_indices(10, 5) == [0, 2, 4, 7, 9]
    # Clamping: requesting more indices than frames returns total frames.
    assert len(gen._sample_indices(3, 99)) == 3
    # Single frame: returns the only index.
    assert gen._sample_indices(1, 8) == [0]


def test_sample_urls_dedupes_endpoints(scail2):
    """``_sample_urls`` preserves order and de-duplicates indices that
    coincide at the endpoints (can happen when ``n >= total``)."""
    _, gen = scail2
    urls = [f"http://x/{i}" for i in range(4)]
    out = gen._sample_urls(urls, 1)
    assert out == ["http://x/2"]  # middle of 4
    out = gen._sample_urls(urls, 2)
    assert out == ["http://x/0", "http://x/3"]  # endpoints
    # Empty / None URLs return empty.
    assert gen._sample_urls([], 5) == []


# --------------------------------------------------------------------------- #
# ComfyUI-node tests
# --------------------------------------------------------------------------- #
def test_comfyui_node_input_types(scail2):
    """``Scail2PromptGenerator.INPUT_TYPES`` exposes the right ports."""
    _, gen = scail2
    spec = gen.Scail2PromptGenerator.INPUT_TYPES()
    required = spec["required"]
    assert "llm_service_connector" in required
    assert "task_type" in required
    assert "user_prompt" in required
    assert "seed" in required
    optional = spec["optional"]
    assert "driving_video" in optional
    assert "reference_images" in optional
    assert "num_frames" in optional
    assert "image_detail" in optional
    assert "temperature" in optional
    assert "timeout" in optional


def test_comfyui_node_return_types(scail2):
    """The node returns a single ``STRING`` like its sibling prompt
    generators (no source_caption port)."""
    _, gen = scail2
    assert gen.Scail2PromptGenerator.RETURN_TYPES == ("STRING",)
    assert gen.Scail2PromptGenerator.RETURN_NAMES == ("scail2_prompt",)


def test_comfyui_node_category(scail2):
    """Lives in the same Prompt Generator category as Bernini / Ideogram4."""
    _, gen = scail2
    cat = gen.Scail2PromptGenerator.CATEGORY
    assert "Prompt Generator" in cat


def test_is_changed_factors_in_seed(scail2):
    """``is_changed`` is a stable hash that varies when meaningful inputs
    change but is robust to identical re-runs."""
    _, gen = scail2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    node = gen.Scail2PromptGenerator()
    a = node.is_changed(
        FakeConnector(),
        task_type="character_replacement - 角色替换",
        user_prompt="hello",
        seed=0,
        num_frames=8,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=2048,
        max_tokens_enhance=512,
        timeout=120,
    )
    b = node.is_changed(
        FakeConnector(),
        task_type="character_replacement - 角色替换",
        user_prompt="hello",
        seed=0,
        num_frames=8,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=2048,
        max_tokens_enhance=512,
        timeout=120,
    )
    assert a == b  # same inputs -> same hash

    c = node.is_changed(
        FakeConnector(),
        task_type="character_replacement - 角色替换",
        user_prompt="hello",
        seed=1,  # different seed
        num_frames=8,
        image_detail="auto",
        temperature=0.4,
        max_tokens_caption=2048,
        max_tokens_enhance=512,
        timeout=120,
    )
    assert a != c


def test_enhancer_short_circuits_on_missing_media(scail2):
    """No driving_video / ref images -> returns the original user_prompt,
    matching Bernini's graceful-degradation behavior."""
    _, gen = scail2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    enhancer = gen.Scail2PromptEnhancer(FakeConnector(), num_frames=8)
    out = enhancer(
        "character_replacement - 角色替换",
        "replace the man",
        driving_video=None,
        reference_images=None,
    )
    assert out == "replace the man"


def test_enhancer_short_circuits_on_empty_prompt_replacement(scail2):
    """``character_replacement`` with empty user_prompt -> returns the
    empty original (the task requires a real instruction)."""
    _, gen = scail2

    class FakeConnector:
        def get_state(self):
            return "fake-state"

    enhancer = gen.Scail2PromptEnhancer(FakeConnector(), num_frames=8)
    out = enhancer(
        "character_replacement - 角色替换",
        "",
        driving_video=None,
        reference_images=None,
    )
    assert out == ""


def test_enhancer_happy_path_runs_full_pipeline(scail2, monkeypatch):
    """Regression for the ``source_urls`` NameError that the rename commit
    (58fafad) introduced on the success path: ``__call__`` must run both
    stages and return the enhanced prompt when media + user_prompt are
    present. The short-circuit tests above never reach the buggy log line,
    so this case is the only thing that would have caught it.

    We monkeypatch ``image_tensor_batch_to_data_urls`` on the generator
    module to return canned URLs (decoupling from the real image-encode
    path) and a ``FakeConnector`` whose ``invoke`` returns deterministic
    text per stage. We then assert:
      - the returned prompt is the stage-2 enhanced text, not the original;
      - both stages were invoked exactly once, in caption-then-enhance order.
    """
    _, gen = scail2

    invocations = []

    class FakeConnector:
        model = "fake-model"

        def get_state(self):
            return "fake-state"

        def invoke(self, messages, *, seed, temperature, max_tokens):
            # Stage 1 (caption) asks for more tokens than stage 2 (enhance).
            invocations.append(max_tokens)
            if max_tokens >= 1024:
                return "A caption of the driving video frames."
            return "An enhanced SCAIL-2 animation prompt."

    # Decouple from the real image-encoding path: any non-None media is
    # treated as a non-empty batch of frames / references.
    monkeypatch.setattr(
        gen,
        "image_tensor_batch_to_data_urls",
        lambda t: ["data:image/jpeg;base64,AAAA"] * 8 if t is not None else [],
    )

    enhancer = gen.Scail2PromptEnhancer(FakeConnector(), num_frames=8)

    # A stand-in "tensor": only truthiness matters once the data-url helper
    # is monkeypatched.
    class FakeTensor:
        pass

    out = enhancer(
        "motion_transfer - 动作迁移",
        "the girl is dancing",
        driving_video=FakeTensor(),
        reference_images=FakeTensor(),
        seed=42,
    )

    assert out == "An enhanced SCAIL-2 animation prompt."
    # First the high-token caption stage, then the lower-token enhance stage.
    assert invocations == [gen._DEFAULT_MAX_TOKENS_CAPTION, gen._DEFAULT_MAX_TOKENS_ENHANCE]
