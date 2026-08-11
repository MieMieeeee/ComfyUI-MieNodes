"""MiniMax H3 prompt templates and message-building helpers.

Supports six task paths derived from the official H3
VIDEO_PROMPT_WRITING_GUIDE (base + ref):

- ``t2v``            -- text-to-video (no reference image)
- ``i2v_first``      -- image-to-video, single first frame (I2VA)
- ``i2v_first_last`` -- image-to-video, first + last frame (FL2VA)
- ``i2v_last``       -- image-to-video, single last frame (L2VA)
- ``reference``      -- multimodal reference (Ref2VA): multiple images and/or
                        a reference video (as an IMAGE batch)
- ``s2v``            -- single-image subject reference (S2V)

A ``category`` dimension (19 use-case classes from the H3 scenario taxonomy,
plus ``none``) is exposed as a second dropdown; it does not change the task
path -- it only injects a short "default styling advice" string into the
stage-2 user template so the model can apply per-category defaults
(cinematic color grading for cinematic-story, motion blur for action, etc.).

The strings themselves live as ``.txt`` files under
``nodes/llm/prompts/h3/`` and are loaded via ``prompts/loader.py`` to keep
the Python surface small. Mirrors the layout of ``scail2_prompts.py``.
"""
from __future__ import annotations

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text


# --------------------------------------------------------------------------- #
# Task list (display strings for the ComfyUI dropdown).
#
# Display strings use the literal separator " - " (space-hyphen-space, ASCII
# U+002D) so ``parse_task_code`` can split them back into the short code.
# Every entry MUST follow "<code> - <label>" exactly; using a different
# separator (em-dash, colon, no spaces) silently breaks the split.
# --------------------------------------------------------------------------- #
TASK_TYPES = (
    "t2v - 文生视频",
    "i2v_first - 图生视频(首帧)",
    "i2v_first_last - 图生视频(首尾帧)",
    "i2v_last - 图生视频(尾帧)",
    "reference - 全能参考",
    "s2v - 主体参考",
)

TASK_CODES = (
    "t2v",
    "i2v_first",
    "i2v_first_last",
    "i2v_last",
    "reference",
    "s2v",
)

# Tasks that take reference images (everything except t2v). Used by the
# enhancer to pick the system prompt and decide whether to caption.
IMAGE_TASK_CODES = frozenset(
    c for c in TASK_CODES if c != "t2v"
)

# --------------------------------------------------------------------------- #
# Use-case category dimension (H3 scenario taxonomy, 19 classes + none).
#
# Display strings follow the same "<code> - <label>" contract as TASK_TYPES
# so ``parse_category`` reuses the split logic. Chinese sub-labels are
# separated by "/" (never " - ") so the split does not get truncated.
# --------------------------------------------------------------------------- #
CATEGORIES = (
    "none - 不指定",
    "cinematic-story - 电影短片/MV/戏剧",
    "product-commercial - 商业广告/UGC带货",
    "music-video - 音乐卡点MV",
    "action - 动作戏/打斗/飙车",
    "anime - 二次元/漫画风",
    "gameplay - 游戏/玩家视角",
    "fashion - 时装片/街拍",
    "animation - 3D/2D动画",
    "comedy - 喜剧/段子",
    "horror - 恐怖/惊悚",
    "cinematic-travel - 旅行/风景",
    "music-performance - 音乐演奏/演唱会",
    "vlog - Vlog/自拍",
    "brand-film - 品牌片/形象片",
    "commercial-ad - 硬广",
    "motion-graphics - 动态图形/MG",
    "viral-short - 病毒短视频",
    "title-sequence - 片头/字幕",
    "product-demo - 产品演示",
)

CATEGORY_CODES = (
    "none",
    "cinematic-story",
    "product-commercial",
    "music-video",
    "action",
    "anime",
    "gameplay",
    "fashion",
    "animation",
    "comedy",
    "horror",
    "cinematic-travel",
    "music-performance",
    "vlog",
    "brand-film",
    "commercial-ad",
    "motion-graphics",
    "viral-short",
    "title-sequence",
    "product-demo",
)

# Per-category default styling advice injected into the stage-2 user prompt.
# English keyword phrases so the model can apply them as concrete defaults.
# Empty string for "none" means "no category-specific advice". Sourced from
# the H3 scenario taxonomy's "default parameter advice" column.
CATEGORY_ADVICE = {
    "none": "",
    "cinematic-story": "cinematic color grading, 35-50mm focal length, soft directional lighting, restrained camera moves",
    "product-commercial": "product hero highlight, shallow depth of field, handheld realism, clean background",
    "music-video": "beat-synced editing, one cut per beat, subtitle/lyric punctuation, rhythmic camera pushes",
    "action": "motion blur, camera shake, low angle, quick cuts, handheld energy",
    "anime": "2D anime look, flat lighting, clean linework, limited but saturated palette",
    "gameplay": "player POV, HUD overlay feel, lower-fps texture, direct-control camera",
    "fashion": "editorial color grade, model-pose-led framing, crisp rim light, slow confident moves",
    "animation": "3D CG / Pixar-style or 2D cartoon look, soft global illumination, appealing shapes",
    "comedy": "exaggerated reactions, jump cuts, on-screen text punctuation, playful sound stings",
    "horror": "low-key lighting, deep shadows, slow creeping camera, timed jump-scare moments",
    "cinematic-travel": "drone / wide establishing shots, golden-hour light, smooth gimbal moves",
    "music-performance": "stage lighting, wide-to-closeup coverage, live-performance energy, audience ambience",
    "vlog": "iPhone front-facing feel, natural window light, casual handheld, spoken-word pacing",
    "brand-film": "cinematic look, abstract concept, slow motion, premium grade",
    "commercial-ad": "product high-key lighting, bold on-screen text, product center-framed",
    "motion-graphics": "flat 2D / motion-design, easing curves, loop-friendly timing",
    "viral-short": "hook in the first second, large subtitle text, a clear payoff/reversal",
    "title-sequence": "animated typography, cinematic mood, text-led composition",
    "product-demo": "centered product, clean rotation, white background, UI callouts",
}

# Mirrors scail2_prompts.MAX_EXAMPLE_CHARS (upstream prompt_enhancer default).
MAX_EXAMPLE_CHARS = 4000

# Default video resolution used when the node's optional width/height inputs
# are left unconnected. 1280x720 -> "16:9". Pure prompt-side metadata; the
# downstream H3 video node sets the real render size.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def gcd(a: int, b: int) -> int:
    """Greatest common divisor (Euclid). ``math.gcd`` would do, but inlining
    keeps the helper dependency-free and obvious."""
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a or 1


def aspect_ratio_string(width: int, height: int) -> str:
    """Reduce a ``width`` x ``height`` pair to a simplified ``W:H`` ratio
    string for the H3 prompt header.

    Returns the literal ``"16:9"`` for the project default 1280x720, and the
    exact ``"{w}:{h}"`` when either side is 0 or the inputs do not reduce to
    a clean small ratio (e.g. arbitrary pixel sizes). The string is pure
    prompt metadata -- the downstream H3 video node decides the real render
    size, so we only need a human-readable ratio here.
    """
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return "16:9"
    if w <= 0 or h <= 0:
        return "16:9"
    g = gcd(w, h)
    rw, rh = w // g, h // g
    return f"{rw}:{rh}"


# --------------------------------------------------------------------------- #
# Output language + section headers
#
# The final H3 prompt uses three named section headers plus optional ones
# (negatives, dialogue). Headers are computed here (single source of truth)
# and injected into the user template via ``{section_headers}``, so the LLM
# copies them verbatim instead of choosing its own translation. One header
# per line, ``"header: <label>"`` so the model treats them as section labels.
# --------------------------------------------------------------------------- #
OUTPUT_LANGUAGES = ("en", "zh")
DEFAULT_OUTPUT_LANGUAGE = "en"

# Per-language section labels. Keys are the canonical English field names
# the H3 API expects (integrated_multimodal_description etc.); values are
# the display labels written into the prompt.
_SECTION_LABELS = {
    "en": {
        "description": "Core idea",
        "sound": "Soundscape",
        "music": "Music",
        "negatives": "Do not include",
    },
    "zh": {
        "description": "核心创意",
        "sound": "整体音效",
        "music": "配乐",
        "negatives": "不要出现",
    },
}

# Canonical English field name for the description header (always English so
# the H3 API can still parse it): written as a parenthetical after the label.
_DESC_FIELD_NAME = "integrated_multimodal_description"
_SOUND_FIELD_NAME = "overall_soundscape"
_MUSIC_FIELD_NAME = "non_diegetic_music"


def _labels(output_language: str) -> dict:
    """Return the section-label dict for ``output_language`` (en/zh). Falls
    back to English for unknown languages."""
    return _SECTION_LABELS.get(output_language, _SECTION_LABELS["en"])


def section_headers_block(output_language: str = DEFAULT_OUTPUT_LANGUAGE) -> str:
    """Return the three required section headers (description / sound /
    music), one per line, formatted as ``"header: <label> (<field>)"``.

    The label is localized (中文/English) but the parenthetical field name
    is always the canonical English API token, so the H3 pipeline can still
    parse the structure regardless of the chosen output language.

    Example (en):
        Header: Core idea (integrated_multimodal_description)
        Header: Soundscape (overall_soundscape)
        Header: Music (non_diegetic_music)
    Example (zh):
        Header: 核心创意 (integrated_multimodal_description)
        Header: 整体音效 (overall_soundscape)
        Header: 配乐 (non_diegetic_music)
    """
    labels = _labels(output_language)
    return "\n".join(
        [
            f"Header: {labels['description']} ({_DESC_FIELD_NAME})",
            f"Header: {labels['sound']} ({_SOUND_FIELD_NAME})",
            f"Header: {labels['music']} ({_MUSIC_FIELD_NAME})",
        ]
    )


def negatives_header(output_language: str = DEFAULT_OUTPUT_LANGUAGE) -> str:
    """Return the optional negatives-section header line, or empty string."""
    labels = _labels(output_language)
    return f"Header: {labels['negatives']}"


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #
_SYSTEM_T2V_PATH = "h3/system_t2v"
_SYSTEM_REFERENCE_PATH = "h3/system_reference"
_CAPTION_REFERENCE_PATH = "h3/caption_reference"


def system_t2v_prompt() -> str:
    """Stage-2 system prompt for the text-to-video (T2VA) path.

    Encodes the official H3 base guide: three-field structure, per-shot
    seven-element checklist, dialogue rules, output rules.
    """
    return load_prompt_text(_SYSTEM_T2V_PATH)


def system_reference_prompt() -> str:
    """Stage-2 system prompt for all image-bearing paths
    (I2VA/FL2VA/L2VA/Ref2VA/S2V).

    Encodes the official H3 ref guide: material-role taxonomy, alignment
    directive rule, the "do not re-describe the reference image" rule, and
    label-consistency rule.
    """
    return load_prompt_text(_SYSTEM_REFERENCE_PATH)


def caption_reference_prompt() -> str:
    """Stage-1 system prompt: caption reference images / sampled video frames.

    Identifies identity, wardrobe, scene, and the features that must be
    preserved, so the stage-2 enhancer can lock them item by item instead
    of re-describing them.
    """
    return load_prompt_text(_CAPTION_REFERENCE_PATH)


# --------------------------------------------------------------------------- #
# Alignment directive snippets (prepended verbatim to the I2V/R2V user prompt)
# --------------------------------------------------------------------------- #
_ALIGN_I2V_FIRST_PATH = "h3/align_i2v_first"
_ALIGN_I2V_FIRST_LAST_PATH = "h3/align_i2v_first_last"
_ALIGN_I2V_LAST_PATH = "h3/align_i2v_last"


def align_i2v_first_snippet() -> str:
    """I2VA first-frame alignment directive (verbatim, no placeholder)."""
    return load_prompt_text(_ALIGN_I2V_FIRST_PATH).strip()


def _final_timestamp(duration) -> str:
    """Format the FL2VA/L2VA final timestamp as ``<seconds>.000s``.

    Accepts int or float. Integer durations (6) render as ``6.000s``;
    fractional durations (7.5) render as ``7.500s``. Negative or
    non-numeric values fall back to ``0.000s``.
    """
    try:
        d = float(duration)
    except (TypeError, ValueError):
        d = 0.0
    if d < 0:
        d = 0.0
    return f"{d:.3f}s"


def align_i2v_first_last_snippet(duration) -> str:
    """FL2VA first+last alignment directive; final timestamp substituted."""
    template = load_prompt_text(_ALIGN_I2V_FIRST_LAST_PATH)
    return template.format(ts=_final_timestamp(duration)).strip()


def align_i2v_last_snippet(duration) -> str:
    """L2VA last-frame alignment directive; final timestamp substituted."""
    template = load_prompt_text(_ALIGN_I2V_LAST_PATH)
    return template.format(ts=_final_timestamp(duration)).strip()


def align_directive_for(task_code: str, duration) -> str:
    """Pick the right alignment directive for an image-bearing task.

    Returns an empty string for ``t2v`` (no reference image, no directive)
    and for unknown codes (the enhancer short-circuits unknown codes
    earlier, so this is defensive only).
    """
    if task_code == "i2v_first":
        return align_i2v_first_snippet()
    if task_code == "i2v_first_last":
        return align_i2v_first_last_snippet(duration)
    if task_code == "i2v_last":
        return align_i2v_last_snippet(duration)
    if task_code in ("reference", "s2v"):
        # No single-frame anchor: the model uses <Subject N> tags inside
        # subject_definitions instead of a top-of-prompt <Picture N> line.
        return ""
    return ""


# --------------------------------------------------------------------------- #
# Stage-2 user templates (substituted via .format)
# --------------------------------------------------------------------------- #
_USER_T2V_PATH = "h3/user_t2v_template"
_USER_I2V_PATH = "h3/user_i2v_template"


def _duration_str(duration) -> str:
    """Format ``duration`` for the prompt header.

    Integer-valued floats (6.0) render without a decimal part (``6``);
    fractional values (7.5) keep one decimal (``7.5``). Non-numeric /
    negative values fall back to ``0``.
    """
    try:
        d = float(duration)
    except (TypeError, ValueError):
        d = 0.0
    if d < 0:
        d = 0.0
    if d.is_integer():
        return str(int(d))
    # Trim to a reasonable precision; stripf trailing zeros.
    return f"{d:.3f}".rstrip("0").rstrip(".")


def user_t2v_prompt(
    idea: str,
    duration,
    aspect_ratio: str,
    category_advice: str,
    examples: str,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """Stage-2 user text for the T2VA path.

    Substitutes ``{idea}``/``{duration}``/``{aspect_ratio}``/
    ``{category_advice}``/``{section_headers}``/``{output_language}``/
    ``{examples}`` into ``user_t2v_template.txt``.
    """
    template = load_prompt_text(_USER_T2V_PATH)
    return template.format(
        idea=(idea or "").strip(),
        duration=_duration_str(duration),
        aspect_ratio=(aspect_ratio or "").strip(),
        category_advice=(category_advice or "").strip(),
        section_headers=section_headers_block(output_language),
        negatives_header=negatives_header(output_language),
        output_language=output_language if output_language in OUTPUT_LANGUAGES else "en",
        examples=(examples or "(No examples provided.)").strip(),
    )


def user_i2v_prompt(
    align_directive: str,
    idea: str,
    caption: str,
    duration,
    aspect_ratio: str,
    category_advice: str,
    examples: str,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """Stage-2 user text for the image-bearing paths.

    Substitutes ``{align_directive}``/``{idea}``/``{caption}``/
    ``{duration}``/``{aspect_ratio}``/``{category_advice}``/
    ``{section_headers}``/``{negatives_header}``/``{output_language}``/
    ``{examples}`` into ``user_i2v_template.txt``.
    """
    template = load_prompt_text(_USER_I2V_PATH)
    return template.format(
        align_directive=(align_directive or "").strip(),
        idea=(idea or "").strip(),
        caption=(caption or "").strip(),
        duration=_duration_str(duration),
        aspect_ratio=(aspect_ratio or "").strip(),
        category_advice=(category_advice or "").strip(),
        section_headers=section_headers_block(output_language),
        negatives_header=negatives_header(output_language),
        output_language=output_language if output_language in OUTPUT_LANGUAGES else "en",
        examples=(examples or "(No examples provided.)").strip(),
    )


# --------------------------------------------------------------------------- #
# Few-shot examples
# --------------------------------------------------------------------------- #
_EXAMPLES_T2V_PATH = "h3/examples_t2v"
_EXAMPLES_I2V_PATH = "h3/examples_i2v"


def bundled_examples_t2v(max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Return the bundled T2VA few-shot examples, truncated to ``max_chars``."""
    text = load_prompt_text(_EXAMPLES_T2V_PATH).strip()
    return text[:max_chars]


def bundled_examples_i2v(max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Return the bundled I2V/R2V few-shot examples, truncated."""
    text = load_prompt_text(_EXAMPLES_I2V_PATH).strip()
    return text[:max_chars]


def load_bundled_examples(task_code: str, max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Return the bundled few-shot examples for the given task code.

    T2V uses the T2VA examples; every image-bearing task uses the I2V/R2V
    examples (they illustrate the alignment-directive + "describe the
    change" pattern that all of them share).
    """
    if task_code == "t2v":
        return bundled_examples_t2v(max_chars)
    if task_code in IMAGE_TASK_CODES:
        return bundled_examples_i2v(max_chars)
    return ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_task_code(task_type: str) -> str:
    """Extract the short task code from a display string.

    Accepts:
      - display strings like ``"i2v_first - 图生视频(首帧)"``
      - the bare code ``"i2v_first"`` (saved workflows)
      - None / empty (passed through unchanged)

    Separator contract: the display string splits on the literal substring
    ``" - "`` (space-hyphen-space, ASCII hyphen-minus U+002D). Every entry in
    ``TASK_TYPES`` MUST follow ``"<code> - <label>"`` exactly; using a
    different separator will silently break the split.
    """
    if not task_type:
        return task_type
    return task_type.split(" - ", 1)[0].strip()


def parse_category(category: str) -> str:
    """Extract the short category code from a display string.

    Same contract as ``parse_task_code``: splits on ``" - "``. Accepts
    display strings (``"cinematic-story - 电影短片/MV/戏剧"``), bare codes
    (``"cinematic-story"``), and None / empty.
    """
    if not category:
        return category
    return category.split(" - ", 1)[0].strip()


def category_advice(category: str) -> str:
    """Return the default styling advice for a category display string or
    bare code. Empty string for ``"none"`` or any unknown category."""
    code = parse_category(category)
    return CATEGORY_ADVICE.get(code, "")
