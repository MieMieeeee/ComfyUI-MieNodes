# -*- coding: utf-8 -*-
"""Tests for the H3 prompt templates and helpers.

Mirrors the loader-stubbing pattern from ``test_scail2_prompts.py`` so the
project layout (root has hyphens, cannot be imported as a normal package)
keeps working under pytest.

Coverage:
* All 11 prompt files + UPSTREAM.md load via the project's
  ``load_prompt_text``.
* System / caption prompts load and contain their key phrases.
* Alignment directives substitute ``{duration}`` and produce the expected
  timestamps.
* ``user_t2v_prompt`` / ``user_i2v_prompt`` ``.format(...)`` cleanly with
  their expected kwargs and embed the supplied idea / caption / examples.
* ``parse_task_code`` / ``parse_category`` handle display strings, bare
  codes, and None.
* ``category_advice`` returns the right string per code (empty for none /
  unknown).
* ``CATEGORIES`` follows the ``" - "`` contract and every entry parses
  back to a known short code.
* ``bundled_examples_t2v`` / ``bundled_examples_i2v`` return non-empty
  text and truncate.
* ``MiniMaxH3PromptGenerator`` is importable, has the right INPUT_TYPES
  shape (incl. category / duration / width / height / first_frame /
  last_frame / reference_video), and exposes the expected ports.
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
    h3 modules so tests can ``import`` them like the rest of the project
    does at runtime. Mirrors ``test_scail2_prompts._load_scail2``."""
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


# --------------------------------------------------------------------------- #
# Prompt file existence
# --------------------------------------------------------------------------- #
def test_prompt_files_exist():
    """All 11 prompt .txt files + UPSTREAM.md must be on disk."""
    expected = {
        "UPSTREAM.md",
        "system_t2v.txt",
        "system_reference.txt",
        "caption_reference.txt",
        "user_t2v_template.txt",
        "user_i2v_template.txt",
        "align_i2v_first.txt",
        "align_i2v_first_last.txt",
        "align_i2v_last.txt",
        "examples_t2v.txt",
        "examples_i2v.txt",
    }
    on_disk = {p.name for p in (PROMPTS_DIR / "h3").iterdir()}
    missing = expected - on_disk
    assert not missing, f"missing prompt files: {missing}"


# --------------------------------------------------------------------------- #
# System / caption prompt content
# --------------------------------------------------------------------------- #
def test_system_t2v_content(h3):
    prompts, _ = h3
    text = prompts.system_t2v_prompt()
    assert "Five core principles" in text
    # Headers are now injected via {section_headers}; the system prompt
    # must tell the model to copy them verbatim and respect output_language.
    assert "output_language" in text
    assert "VERBATIM" in text or "verbatim" in text
    assert "seven" in text.lower()  # per-shot seven-element checklist
    assert "Output rules" in text
    assert text.rstrip("\n") == text.rstrip()  # no stray trailing whitespace


def test_system_reference_content(h3):
    prompts, _ = h3
    text = prompts.system_reference_prompt()
    assert "<Picture N>" in text
    assert "<Subject N>" in text
    assert "alignment directive" in text.lower()
    assert "Do NOT re-describe" in text or "do not re-describe" in text.lower()
    assert text.rstrip("\n") == text.rstrip()  # no stray trailing whitespace


def test_caption_reference_content(h3):
    prompts, _ = h3
    text = prompts.caption_reference_prompt()
    assert "reference" in text.lower()
    assert "preserve" in text.lower()
    # The captioner is explicitly told NOT to emit H3-specific tags. The
    # instruction itself names the forbidden tags ("do NOT write any
    # H3-specific tags (<Picture N>, <Subject N>, <d>...</d>)"), so we
    # assert the prohibition is present rather than that the tag string
    # is wholly absent.
    assert "do NOT write any H3-specific tags" in text
    assert "Output only the reference-material caption" in text


# --------------------------------------------------------------------------- #
# Alignment directives
# --------------------------------------------------------------------------- #
def test_align_i2v_first(h3):
    prompts, _ = h3
    text = prompts.align_i2v_first_snippet()
    assert "at 0.00 seconds" in text
    assert "<Picture 1>" in text
    assert "{duration}" not in text  # no placeholder in the first-frame snippet


def test_align_i2v_first_last_substitutes_duration(h3):
    prompts, _ = h3
    text = prompts.align_i2v_first_last_snippet(6)
    assert "<Picture 1>" in text
    assert "<Picture 2>" in text
    assert "6.000s" in text
    assert "{ts}" not in text


def test_align_i2v_last_substitutes_duration(h3):
    prompts, _ = h3
    text = prompts.align_i2v_last_snippet(8)
    assert "<Picture 1>" in text
    assert "8.000s" in text
    assert "{ts}" not in text


def test_align_i2v_first_last_accepts_float_duration(h3):
    """Fractional durations render as ``<secs>.000s`` with 3 decimals."""
    prompts, _ = h3
    assert "7.500s" in prompts.align_i2v_first_last_snippet(7.5)
    assert "12.250s" in prompts.align_i2v_last_snippet(12.25)


def test_align_directive_for_picks_right_snippet(h3):
    prompts, _ = h3
    assert "at 0.00 seconds" in prompts.align_directive_for("i2v_first", 6)
    assert "6.000s" in prompts.align_directive_for("i2v_first_last", 6)
    assert "8.000s" in prompts.align_directive_for("i2v_last", 8)
    # reference / s2v have no top-of-prompt directive.
    assert prompts.align_directive_for("reference", 6) == ""
    assert prompts.align_directive_for("s2v", 6) == ""
    # t2v and unknown -> empty.
    assert prompts.align_directive_for("t2v", 6) == ""
    assert prompts.align_directive_for("not_a_task", 6) == ""


# --------------------------------------------------------------------------- #
# User templates
# --------------------------------------------------------------------------- #
def test_user_t2v_prompt_formatting(h3):
    prompts, _ = h3
    rendered = prompts.user_t2v_prompt(
        idea="a neon city at night",
        duration=8,
        aspect_ratio="16:9",
        category_advice="cinematic color grading, 35-50mm focal length",
        examples="example-one\nexample-two",
    )
    assert "a neon city at night" in rendered
    assert "8" in rendered
    assert "16:9" in rendered
    assert "cinematic color grading, 35-50mm focal length" in rendered
    assert "example-one" in rendered
    assert "{idea}" not in rendered
    assert "{duration}" not in rendered


def test_user_i2v_prompt_formatting(h3):
    prompts, _ = h3
    rendered = prompts.user_i2v_prompt(
        align_directive="For the target video, at 0.00 seconds ... <Picture 1> ...",
        idea="she opens the trunk and takes out a bag",
        caption="A young woman in a white dress beside a vintage car.",
        duration=6,
        aspect_ratio="16:9",
        category_advice="product hero highlight",
        examples="example-a",
    )
    assert "at 0.00 seconds" in rendered
    assert "she opens the trunk" in rendered
    assert "A young woman in a white dress" in rendered
    assert "6" in rendered
    assert "product hero highlight" in rendered
    assert "example-a" in rendered
    for token in ("{align_directive}", "{idea}", "{caption}", "{duration}"):
        assert token not in rendered


def test_user_t2v_prompt_falls_back_when_examples_empty(h3):
    prompts, _ = h3
    rendered = prompts.user_t2v_prompt(
        idea="x", duration=6, aspect_ratio="16:9",
        category_advice="", examples="",
    )
    assert "(No examples provided.)" in rendered


def test_user_prompt_accepts_float_duration(h3):
    """``duration`` may be a float; integer-valued floats render without a
    decimal part (6.0 -> "6"), fractional values keep their decimals."""
    prompts, _ = h3
    int_rendered = prompts.user_t2v_prompt(
        idea="x", duration=6.0, aspect_ratio="16:9",
        category_advice="", examples="x",
    )
    assert "duration: 6 seconds" in int_rendered
    frac_rendered = prompts.user_t2v_prompt(
        idea="x", duration=7.5, aspect_ratio="16:9",
        category_advice="", examples="x",
    )
    assert "duration: 7.5 seconds" in frac_rendered


# --------------------------------------------------------------------------- #
# aspect_ratio_string helper (width x height -> "W:H")
# --------------------------------------------------------------------------- #
def test_aspect_ratio_string_reduces_pairs(h3):
    prompts, _ = h3
    assert prompts.aspect_ratio_string(1280, 720) == "16:9"
    assert prompts.aspect_ratio_string(720, 1280) == "9:16"
    assert prompts.aspect_ratio_string(1080, 1080) == "1:1"
    assert prompts.aspect_ratio_string(1024, 768) == "4:3"
    assert prompts.aspect_ratio_string(768, 1024) == "3:4"
    # Coprime / large pair reduces correctly.
    assert prompts.aspect_ratio_string(2560, 1080) == "64:27"


def test_aspect_ratio_string_handles_bad_input(h3):
    """Zero / negative / non-numeric inputs fall back to the project default."""
    prompts, _ = h3
    assert prompts.aspect_ratio_string(0, 1080) == "16:9"
    assert prompts.aspect_ratio_string(1920, 0) == "16:9"
    assert prompts.aspect_ratio_string(-1, 100) == "16:9"
    assert prompts.aspect_ratio_string("not a number", 100) == "16:9"


# --------------------------------------------------------------------------- #
# Task / category parsing + category advice
# --------------------------------------------------------------------------- #
def test_parse_task_code(h3):
    prompts, _ = h3
    assert prompts.parse_task_code("t2v - 文生视频") == "t2v"
    assert prompts.parse_task_code("i2v_first_last - 图生视频(首尾帧)") == "i2v_first_last"
    assert prompts.parse_task_code("reference") == "reference"  # bare code
    assert prompts.parse_task_code("") == ""
    assert prompts.parse_task_code(None) is None


def test_parse_category(h3):
    prompts, _ = h3
    assert prompts.parse_category("cinematic-story - 电影短片/MV/戏剧") == "cinematic-story"
    assert prompts.parse_category("none - 不指定") == "none"
    assert prompts.parse_category("action") == "action"  # bare code
    assert prompts.parse_category("") == ""
    assert prompts.parse_category(None) is None


def test_category_advice(h3):
    prompts, _ = h3
    assert prompts.category_advice("cinematic-story - 电影短片/MV/戏剧") != ""
    assert "motion blur" in prompts.category_advice("action")
    # none -> empty.
    assert prompts.category_advice("none - 不指定") == ""
    assert prompts.category_advice("none") == ""
    # unknown -> empty.
    assert prompts.category_advice("not-a-category") == ""


def test_categories_follow_separator_contract(h3):
    """Every CATEGORIES entry uses the literal " - " separator and parses
    back to a short code present in CATEGORY_CODES."""
    prompts, _ = h3
    assert len(prompts.CATEGORIES) == len(prompts.CATEGORY_CODES)
    for display, code in zip(prompts.CATEGORIES, prompts.CATEGORY_CODES):
        assert " - " in display, f"bad separator in {display!r}"
        assert prompts.parse_category(display) == code
    # Default dropdown value is "none".
    assert prompts.CATEGORIES[0] == "none - 不指定"


def test_category_advice_covers_all_codes(h3):
    """Every known category code has an advice entry (empty for none)."""
    prompts, _ = h3
    for code in prompts.CATEGORY_CODES:
        assert code in prompts.CATEGORY_ADVICE, f"missing advice for {code!r}"


# --------------------------------------------------------------------------- #
# Output language + section headers
# --------------------------------------------------------------------------- #
def test_section_headers_block_en(h3):
    prompts, _ = h3
    block = prompts.section_headers_block("en")
    assert "Core idea" in block
    assert "Soundscape" in block
    assert "Music" in block
    # Canonical English API field names are always present (parse anchor).
    assert "integrated_multimodal_description" in block
    assert "overall_soundscape" in block
    assert "non_diegetic_music" in block
    # No Chinese leaks into the English block.
    assert "核心创意" not in block


def test_section_headers_block_zh(h3):
    prompts, _ = h3
    block = prompts.section_headers_block("zh")
    assert "核心创意" in block
    assert "整体音效" in block
    assert "配乐" in block
    # Canonical English API field names stay even in zh (parse anchor).
    assert "integrated_multimodal_description" in block
    assert "overall_soundscape" in block
    assert "non_diegetic_music" in block


def test_section_headers_block_unknown_lang_falls_back(h3):
    prompts, _ = h3
    block = prompts.section_headers_block("fr")
    assert "Core idea" in block  # falls back to English


def test_negatives_header(h3):
    prompts, _ = h3
    assert "Do not include" in prompts.negatives_header("en")
    assert "不要出现" in prompts.negatives_header("zh")


def test_user_t2v_prompt_injects_language_and_headers(h3):
    """``output_language`` drives both the header block language and the
    ``Output language:`` line in the rendered user message."""
    prompts, _ = h3
    en = prompts.user_t2v_prompt(
        idea="x", duration=6, aspect_ratio="16:9",
        category_advice="", examples="ex", output_language="en",
    )
    assert "Output language: en" in en
    assert "Core idea" in en
    assert "Soundscape" in en
    assert "核心创意" not in en

    zh = prompts.user_t2v_prompt(
        idea="x", duration=6, aspect_ratio="16:9",
        category_advice="", examples="ex", output_language="zh",
    )
    assert "Output language: zh" in zh
    assert "核心创意" in zh
    assert "整体音效" in zh
    assert "配乐" in zh
    # Canonical field name survives in zh too.
    assert "integrated_multimodal_description" in zh


# --------------------------------------------------------------------------- #
# Few-shot examples
# --------------------------------------------------------------------------- #
def test_bundled_examples(h3):
    prompts, _ = h3
    t2v = prompts.bundled_examples_t2v()
    i2v = prompts.bundled_examples_i2v()
    assert t2v
    assert i2v
    # T2V example mentions the kinetic-typography sample.
    assert "SUMMER NEVER SLEEPS" in t2v
    # I2V example mentions the brand-film + five-character MV samples.
    assert "<Picture 1>" in i2v
    assert "<Picture 5>" in i2v


def test_load_bundled_examples_routes_by_task(h3):
    prompts, _ = h3
    assert prompts.load_bundled_examples("t2v") == prompts.bundled_examples_t2v()
    for code in prompts.IMAGE_TASK_CODES:
        assert prompts.load_bundled_examples(code) == prompts.bundled_examples_i2v()
    assert prompts.load_bundled_examples("not_a_task") == ""


def test_bundled_examples_truncation(h3):
    prompts, _ = h3
    full = prompts.bundled_examples_t2v()
    short = prompts.bundled_examples_t2v(max_chars=50)
    assert len(short) <= 50
    assert short == full[:50]


# --------------------------------------------------------------------------- #
# ComfyUI-node surface
# --------------------------------------------------------------------------- #
def test_comfyui_node_input_types(h3):
    _, gen = h3
    spec = gen.MiniMaxH3PromptGenerator.INPUT_TYPES()
    required = spec["required"]
    for name in ("llm_service_connector", "task_type", "user_prompt", "seed"):
        assert name in required, f"missing required port {name!r}"
    optional = spec["optional"]
    for name in (
        "first_frame", "last_frame", "reference_images", "reference_video",
        "width", "height", "duration", "category", "output_language",
        "caption_sample_frames", "image_detail", "temperature",
        "max_tokens_caption", "max_tokens_enhance", "timeout",
    ):
        assert name in optional, f"missing optional port {name!r}"
    # The old aspect_ratio dropdown and num_frames knob must be gone.
    assert "aspect_ratio" not in optional
    assert "num_frames" not in optional
    # output_language dropdown exposes en + zh, defaults to en.
    assert set(optional["output_language"][0]) == {"en", "zh"}
    assert optional["output_language"][1]["default"] == "en"


def test_comfyui_node_return_types(h3):
    _, gen = h3
    assert gen.MiniMaxH3PromptGenerator.RETURN_TYPES == ("STRING",)
    assert gen.MiniMaxH3PromptGenerator.RETURN_NAMES == ("h3_prompt",)


def test_comfyui_node_category(h3):
    _, gen = h3
    assert "Prompt Generator" in gen.MiniMaxH3PromptGenerator.CATEGORY
