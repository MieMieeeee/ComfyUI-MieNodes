"""MiniMax H3 Loop plan-generator ComfyUI node.

Turns one free-form ``user_input`` (a concept paragraph OR one line per
beat) into a ``plan_json`` STRING that plugs straight into
ethanfel/ComfyUI-MiniMaxH3-Context-Loop's ``MiniMaxH3ChainPlanModern``
(Production Plan) ``plan_json_input`` socket — a non-empty upstream
string overrides the editor's stored plan.

Pipeline:
  * Stage 0.5 — span-anchored dialogue extraction (one LLM call, then
    mechanical ``concept[start:end] == text`` checks). Dialogue boards
    build the storyboard deterministically: auto ``scene_count`` packs
    consecutive turns into the fewest scenes that fit the 14 s H3
    window; an explicit count spreads speech evenly and may insert
    silent reaction cuts. Extraction failure is a summary warning and
    falls back to the legacy LLM storyboard (narration path). Pure
    narration (extractor returned ``{"turns": []}``) also uses the
    LLM storyboard. ``pacing`` owns speech tempo, not auto cut count.
  * Stage 0 (deterministic) — grid-convert durations, derive per-shot
    seeds, sanity-check the summed duration against the budget.
  * Stage 1 (LLM) — derive the shared ``prompt_prefix``.
  * Stage 2 (LLM) — write each scene's prompt in the three-section H3
    form; scenes 2+ get an explicit continuation directive referencing
    the previous scene's ending (unbroken motion chain, ambience bed
    carries across the boundary). ``per_shot`` mode = one LLM call per
    scene (best continuity); ``single_call`` = one call for the whole
    board (cheaper/faster).
  * Stage 3 (deterministic) — assemble + validate the strict plan shape
    (``shots``/``prompt_prefix`` only) and emit the summary.

Failures raise ``RuntimeError`` — a partial plan must never reach the
Production Plan node.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Interrupt support.
#
# ComfyUI exposes a global interrupt flag via ``nodes.interrupt_processing()``;
# nodes that want to honour the "stop" button check it and raise
# ``nodes.InterruptProcessingException`` (imported from
# ``comfy_execution.execution`` in modern ComfyUI). The enhancer's LLM
# round-trips run with the default 300 s timeout; without explicit
# interrupt checks the node can sit in a single ``self.llm.invoke(...)``
# for the full duration after the user clicks Stop. We check the flag
# before each LLM call (and once per retry iteration) so a queued
# round-trip aborts on its way out instead of dragging on.
#
# The import is wrapped in try/except so the module still imports in
# standalone / test contexts where ``nodes`` / ``comfy_execution`` are
# unavailable. In that case ``_comfy_interrupt_pressed`` is a no-op
# and ``InterruptProcessingException`` falls back to ``Exception`` so
# tests can ``pytest.raises`` against it.
# --------------------------------------------------------------------------- #
try:
    import nodes as _comfy_nodes  # type: ignore
    from comfy_execution.execution import InterruptProcessingException  # type: ignore

    def _comfy_interrupt_pressed() -> bool:
        try:
            return bool(_comfy_nodes.interrupt_processing())
        except Exception:  # pragma: no cover - defensive
            return False
except Exception:  # pragma: no cover - tests / standalone
    InterruptProcessingException = Exception  # type: ignore

    def _comfy_interrupt_pressed() -> bool:
        return False


def _check_interrupt(stage: str) -> None:
    """Raise ``InterruptProcessingException`` if the executor signalled
    a stop. No-op outside ComfyUI's runtime (tests / standalone)."""
    if _comfy_interrupt_pressed():
        mie_log(f"H3LOOP {stage}: interrupt pressed; aborting")
        raise InterruptProcessingException(
            f"H3LOOP {stage}: interrupt pressed by user"
        )


# ``normalize_shots`` lives in minimax_h3_storyboard_prompts; it pulls
# in nodes/llm/__init__.py at import time, which in turn requires the
# ComfyUI runtime (``folder_paths``). Lazy-import via module-level try
# so test stubs can exercise H3LoopPromptEnhancer without ComfyUI.
try:
    from .minimax_h3_storyboard_prompts import normalize_shots as _ns_lazy  # type: ignore
except (ImportError, ModuleNotFoundError, ValueError):
    # Production code path: ``from .minimax_h3_storyboard_prompts`` works.
    # Test / standalone path: import by file path so we don't drag in
    # the package __init__ (which requires folder_paths).
    import importlib.util as _il
    _storyboard_prompts_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "minimax_h3_storyboard_prompts.py",
    )
    _spec = _il.spec_from_file_location(
        "_mienodes_internal_ns_shim", _storyboard_prompts_path
    )
    _mod = _il.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _ns_lazy = _mod.normalize_shots

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:
        from core.utils import mie_log

try:
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        DEFAULT_MUSIC_LINE,
        DEFAULT_TOTAL_DURATION_SECONDS,
        PREFIX_SYNTH_SYSTEM,
        REFERENCE_MODES,
        REFERENCE_MODE_CODES,
        SCHEMA_SIX,
        append_dialogue_blocks_to_sections,
        assemble_dialogue_line_blocks,
        build_continuation_block,
        build_continuation_block_ref2v,
        build_prefix_user_text,
        build_reference_directive,
        build_shot_user_text,
        build_tempo_directive,
        build_shots_digest,
        build_single_call_user_text,
        build_spatial_layout_directive,
        build_cast_block,
        build_cast_sheet_text,
        extract_spatial_layout,
        scrub_dialogue_from_prompt_text,
        validate_spatial_layout_invariant,
        build_speaker_id_map,
        derive_seed,
        derive_seed_base,
        description_body,
        ensure_keyframe_idiom,
        length_to_seconds,
        log_pipeline,
        plan_to_json_string,
        SCHEMA_THREE,
        schema_for_mode,
        parse_reference_mode,
        seconds_to_length,
        shot_system_prompt,
        shot_system_prompt_ref2v,
        sound_body,
        split_prefix_and_cast,
        split_prefix_paragraphs,
        split_six_sections,
        split_three_sections,
        validate_label_policy,
        validate_manifest,
        validate_plan,
        validate_dialogue_invariantity,
        SIX_SECTION_FIELDS,
        _manifest_digest,
        _mode_note_for_prefix,
        parse_references_text,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts import (
        SYSTEM_STORYBOARD_PROMPT,
        PARSE_RETRY_CORRECTION,
        build_user_text as build_storyboard_user_text,
        extract_json_array,
        normalize_shots,
    )
    from _mienodes_internal.nodes.llm.h3_prompts import caption_reference_prompt
    from _mienodes_internal.core.utils import (
        image_tensor_batch_to_data_urls,
        mie_log,
    )
    from _mienodes_internal.nodes.llm.dialogue_segmenter import (
        PACING_PRESETS as _DLG_PACING_PRESETS,
        DialogueTurn as _DLG_DialogueTurn,
        ExtractedLine as _DLG_ExtractedLine,
        estimate_shot_budget as _dlg_estimate_shot_budget,
        narrator_fallback_turn as _dlg_narrator_fallback_turn,
        pacing_report_text as _dlg_pacing_report_text,
        distribute_lines_to_scenes as _dlg_distribute_lines_to_scenes,
        parse_structured_dialogue_turns as _dlg_parse_structured_dialogue_turns,
        scene_raw_seconds as _dlg_scene_raw_seconds,
        group_budgets_into_scenes as _dlg_group_budgets_into_scenes,
        relocate_extracted_lines as _dlg_relocate_extracted_lines,
        validate_extracted_lines as _dlg_validate_extracted_lines,
    )
except ImportError:
    from .minimax_h3_loop_prompts import (
        DEFAULT_MUSIC_LINE,
        DEFAULT_TOTAL_DURATION_SECONDS,
        PREFIX_SYNTH_SYSTEM,
        REFERENCE_MODES,
        REFERENCE_MODE_CODES,
        SCHEMA_SIX,
        append_dialogue_blocks_to_sections,
        assemble_dialogue_line_blocks,
        build_continuation_block,
        build_continuation_block_ref2v,
        build_prefix_user_text,
        build_reference_directive,
        build_shot_user_text,
        build_tempo_directive,
        build_shots_digest,
        build_single_call_user_text,
        build_spatial_layout_directive,
        build_cast_block,
        build_cast_sheet_text,
        extract_spatial_layout,
        scrub_dialogue_from_prompt_text,
        validate_spatial_layout_invariant,
        build_speaker_id_map,
        derive_seed,
        derive_seed_base,
        description_body,
        ensure_keyframe_idiom,
        length_to_seconds,
        log_pipeline,
        plan_to_json_string,
        SCHEMA_THREE,
        schema_for_mode,
        parse_reference_mode,
        seconds_to_length,
        shot_system_prompt,
        shot_system_prompt_ref2v,
        sound_body,
        split_prefix_and_cast,
        split_prefix_paragraphs,
        split_six_sections,
        split_three_sections,
        validate_label_policy,
        validate_manifest,
        validate_plan,
        validate_dialogue_invariantity,
        SIX_SECTION_FIELDS,
        _manifest_digest,
        _mode_note_for_prefix,
        parse_references_text,
    )
    from .minimax_h3_storyboard_prompts import (
        SYSTEM_STORYBOARD_PROMPT,
        PARSE_RETRY_CORRECTION,
        build_user_text as build_storyboard_user_text,
        extract_json_array,
        normalize_shots,
    )
    from .h3_prompts import caption_reference_prompt
    from ...core.utils import (
        image_tensor_batch_to_data_urls,
        mie_log,
    )
    from .dialogue_segmenter import (
        PACING_PRESETS as _DLG_PACING_PRESETS,
        DialogueTurn as _DLG_DialogueTurn,
        ExtractedLine as _DLG_ExtractedLine,
        estimate_shot_budget as _dlg_estimate_shot_budget,
        narrator_fallback_turn as _dlg_narrator_fallback_turn,
        pacing_report_text as _dlg_pacing_report_text,
        distribute_lines_to_scenes as _dlg_distribute_lines_to_scenes,
        parse_structured_dialogue_turns as _dlg_parse_structured_dialogue_turns,
        scene_raw_seconds as _dlg_scene_raw_seconds,
        group_budgets_into_scenes as _dlg_group_budgets_into_scenes,
        relocate_extracted_lines as _dlg_relocate_extracted_lines,
        validate_extracted_lines as _dlg_validate_extracted_lines,
    )


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Loop-node category widget — a curated 3-entry subset of the full
# ``h3_prompts.CATEGORIES`` taxonomy. We expose only the entries that
# drive the loop prompt-generation path end-to-end:
#   * none     — no category-specific advice / contract
#   * dialogue — drives the SPOKEN SCENE CONTRACT / GENRE CONTRACT
#                injection in build_reference_directive, plus a
#                visual-styling advice block injected into the shot
#                user template via ``{genre_advice}``.
#   * action   — visual-styling advice only (motion-blur / camera-shake);
#                does NOT trigger any contract, but the action-specific
#                cinematography advice meaningfully shapes the whole
#                prompt.
# Sibling nodes (``MiniMaxH3PromptGenerator``, ``MiniMaxH3StoryboardGenerator``)
# keep using the full 22-entry ``h3_prompts.CATEGORIES`` taxonomy —
# only the loop node widget is narrowed here. ``LOOP_CATEGORY_ADVICE``
# is intentionally a separate dict so that even if a caller passes
# one of the legacy 19 codes (e.g. ``"cinematic-story - ..."``) the
# advice lookup simply returns "".
LOOP_CATEGORIES = (
    "none - 不指定",
    "dialogue - 对白/对话/相声",
    "action - 动作戏/打斗/飙车",
)
LOOP_CATEGORY_ADVICE = {
    "none": "",
    "dialogue": (
        "spoken-scene cinematography: medium-close framing on speakers, "
        "shot/reverse-shot on turns, eyeline matching; natural room tone "
        "with breath and lip movement foregrounded; dialogue language "
        "defaults to Chinese unless the concept specifies otherwise; "
        "no on-screen subtitles, no captions, no watermark, no SFX stings "
        "during speech"
    ),
    "action": (
        "motion blur, camera shake, low angle, quick cuts, handheld energy; "
        "all action at real-time speed with no slow motion; snappy "
        "strike-and-recoil verbs (launch -> connect -> recoil); combat "
        "beats 4-6 seconds per clip"
    ),
}


# Structured output: 0.4 keeps the three-section contract stable (the
# h3 sibling uses the same value for stage-2 enhancement).
_DEFAULT_TEMPERATURE = 0.4
# Short-dialogue local-prefix path: skip the stage-1 prefix LLM call
# when the board is dialogue-only, small, and has no reference images —
# the prefix is cut deterministically from the concept's own
# scene-setting text instead.
_LOCAL_PREFIX_MAX_TURNS = 4
_LOCAL_PREFIX_MAX_LINES = 8
# One-stop pipeline: the storyboard stage now emits a full duration-
# budgeted board and every stage writes richer prose, so the budget and
# deadline both moved up from the old 8192 / 120s defaults.
_DEFAULT_MAX_TOKENS = 16384
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768
_DEFAULT_TIMEOUT = 300
# Caption stage: short reply (1-2 sentences per image); 4096 is enough headroom.
_DEFAULT_MAX_TOKENS_CAPTION = 4096

# Cap on the ref2va manifest (matches upstream MAX_MANIFEST_PICTURES).
_MAX_REFERENCE_IMAGES = 9

# Pacing labels for the auto-length estimator (Dialogue segmenter).
# Each maps 1:1 to a Pacing preset; default "normal - 正常(语速)(推荐)".
# The label says 语速 up front: on dialogue boards this widget is SPEECH
# tempo only — it never adds or removes cuts. Legacy stored values
# ("fast - 快" etc.) still parse: the key lookup splits on " - ".
_PACING_LABELS = (
    "fast - 快（语速）",
    "normal - 正常（语速·推荐）",
    "slow - 慢（语速）",
)
_PACING_LABEL_TO_KEY = {
    "fast": "fast",
    "normal": "normal",
    "slow": "slow",
}

# Inline user-input enhancement toggle. "on" runs the standalone
# ``MiniMaxH3LoopUserInputEnhancer`` rewrite once inside this node (one
# extra LLM call) before planning; "off" (default) consumes user_input
# verbatim. Default off because the rewrite's value is a visible,
# editable intermediate — wire the dedicated enhancer node upstream when
# you want that checkpoint, and keep this off so an already-canonical
# text is never re-rewritten (double-enhance footgun).
_ENHANCE_USER_INPUT_LABELS = (
    "off - 不润色(默认)",
    "on - 自动润色后再规划",
)


def parse_enhance_user_input(mode: str) -> bool:
    """``on`` -> True; anything else (including a blank widget) -> False."""
    return (mode or "").split(" - ", 1)[0].strip().lower() == "on"

GENERATION_MODES = (
    "per_shot - 逐场生成(推荐)",
    "single_call - 单次调用(快/省)",
)
GENERATION_MODE_CODES = ("per_shot", "single_call")

# Seed modes. per_scene_increment is the RECOMMENDED default: the
# upstream ComfyUI-MiniMaxH3-Context-Loop plugin (chain_nodes.py) itself
# derives per-scene seeds from one base — sha256(f"{base}:{index}:{shot_id}")
# — whenever the plan omits seeds, and its checkpoint-recovery logic relies
# on seeds being deterministic but distinct per scene. Identical seeds on
# every clip make consecutive clips sample near-identical noise, which
# reads as repeated motion rhythm across cuts.
SEED_MODES = (
    "per_scene_increment - 每场seed递增(推荐)",
    "same_across_scenes - 全场同seed",
)
SEED_MODE_CODES = ("per_scene_increment", "same_across_scenes")

# Storyboard split bias is DERIVED from the pacing preset — pacing owns
# the whole tempo story now (the old split_bias widget was merged into
# it): fast cuts aggressively (one beat per scene), slow packs long.
_PACING_TO_STORYBOARD_BIAS = {
    "fast": "aggressive",
    "normal": "balanced",
    "slow": "conservative",
}

CAPTION_MODES = (
    "cache_memory_disk - 缓存:内存+磁盘(推荐)",
    "cache_memory_only - 缓存:仅内存",
    "no_cache - 禁用缓存",
    "force_recaption_once - 本次强制重打标",
)
CAPTION_MODE_CODES = (
    "cache_memory_disk",
    "cache_memory_only",
    "no_cache",
    "force_recaption_once",
)

# Fixed storyboard style for the inline auto-storyboard (standalone mode).
# "single_continuous" is the only board shape that matches this node's
# chain contract (unbroken motion, invisible_cut handoffs, carried
# ambience) — cut-grammar styles like parallel_montage would fight the
# stage-2 continuation rules.
AUTO_STORYBOARD_STYLE = "single_continuous"

# Subject-name extractors used by the caption completeness check.
# M3 emits handles in three flavors across runs:
#   snake_case:        orange_tabby_kitten
#   CamelCase compound: CalicoMotherCat
#   Space-separated:    "ginger adult cat", "Calico Mother Cat"
# A caption with zero handles is a red flag (model fell back to prose
# like "three kittens" instead of enumerated names).
_NAMED_SUBJECT_RE = re.compile(r"\b([a-z]+(?:_[a-z]+)+)\b")
_NAMED_CAMEL_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
_NAMED_TITLE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+"
    r"(?i:cat|kitten|kittens|mother|father|family|cats|child|adult|warrior|person|figure|"
    r"girl|boy|man|woman)\b"
)

# Subject-name extractors used by the caption completeness check. Kept
# for the wardrobe-token warning path; the spoken-contract / H0 paths
# have been removed.
def _extract_named_subjects(about: str) -> list[str]:
    """Pull subject-name handles out of a caption, returning the unique
    handles in first-seen order. Captures snake_case ("orange_tabby_kitten"),
    CamelCase compounds ("CalicoMotherCat"), and Title-Case +
    role-noun ("Ginger Adult Cat")."""
    if not about:
        return []
    snake = list(dict.fromkeys(_NAMED_SUBJECT_RE.findall(about)))
    snake_low = {s.lower() for s in snake}
    camel = [
        m for m in _NAMED_CAMEL_RE.findall(about)
        if m not in snake and m.lower() not in snake_low
    ]
    titled = []
    for m in _NAMED_TITLE_RE.findall(about):
        name_part = m[0] if isinstance(m, tuple) else m
        norm = str(name_part).lower().replace(" ", "_")
        if norm not in snake and norm not in {c.lower() for c in camel}:
            titled.append(norm)
    return list(dict.fromkeys(snake + camel + titled))

_REF2VA_FORBIDDEN_TIMECODE_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b"
    r"|\b(?:at|from)\s+\d+(?:\.\d+)?\s*s\b"
    r"|\b\d+(?:\.\d+)?\s*s\s+to\s+\d+(?:\.\d+)?\s*s\b",
    re.IGNORECASE,
)

_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_PARSE_RETRIES = 1


def _caption_cache_key(image_url: str, prompt_text: str) -> str:
    """Hash a data URL + the caption prompt into a stable cache key.

    The image URL is the first (and usually only) element of the data-URL
    list returned by `image_tensor_batch_to_data_urls`; it embeds the full
    base64 of the pixel bytes, so hashing it is equivalent to hashing the
    pixel payload. The prompt-text hash invalidates the cache whenever
    `caption_reference_prompt()` is upgraded.
    """
    h = hashlib.sha256()
    h.update((image_url or "").encode("utf-8"))
    h.update(b"|")
    h.update(hashlib.sha256((prompt_text or "").encode("utf-8")).hexdigest().encode("ascii"))
    return h.hexdigest()


def _caption_cache_disk_root() -> str:
    """Resolve the on-disk caption cache directory.

    Resolution order:
      1) ``MIEN_NODES_CACHE_DIR`` override.
      2) ComfyUI's runtime output dir via ``folder_paths.get_output_directory()``.
      3) Fallback ``<repo>/output/mien_nodes/caption_cache`` for tests /
         standalone import contexts where ``folder_paths`` is unavailable.
    """
    override = os.environ.get("MIEN_NODES_CACHE_DIR")
    if override:
        return os.path.abspath(override)
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.join(
                os.path.abspath(str(output_dir)),
                "mien_nodes",
                "caption_cache",
            )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    # __file__ is nodes/llm/<thisfile>.py; walk up two to the repo root.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "output", "mien_nodes", "caption_cache")


def _read_caption_cache_disk(root: str, key: str) -> Optional[str]:
    path = os.path.join(root, f"{key}.txt")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _write_caption_cache_disk(root: str, key: str, about: str) -> Optional[str]:
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        return f"mkdir failed: {exc}"
    path = os.path.join(root, f"{key}.txt")
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(about)
        os.replace(tmp, path)
        return None
    except OSError as exc:
        # Cache write failures are non-fatal: the in-memory tier still
        # serves the current node lifetime.
        err = f"write failed: {exc}"
        try:
            os.unlink(tmp)
        except OSError as cleanup_exc:
            err = f"{err}; tmp cleanup failed: {cleanup_exc}"
        return err


def _emit_caption_manifest_entry(
    *,
    manifest: list[dict],
    slot_n: int,
    about: str,
    ref_code: str,
    warnings: list[str],
) -> None:
    """Append one slot to the manifest with the same role policy used by
    the uncached path, plus the no-wardrobe warning when applicable. Used
    by both cache-hit and cache-miss branches so behavior is identical.
    """
    subjects = _extract_named_subjects(about)
    if not subjects:
        warnings.append(
            f"caption[{slot_n}]: no underscored subject names found "
            f"(caption: {about[:120]}...)"
        )
    log_pipeline(
        f"caption[{slot_n}] enumerated {len(subjects)} subject(s): "
        f"{', '.join(subjects[:8])}"
    )
    role = (
        "identity"
        if ref_code in ("ref2va",) or slot_n == 1
        else "destination"
    )
    manifest.append({"slot": f"Picture {slot_n}", "about": about, "role": role})
    log_pipeline(
        f"stage0.6 captioned Picture {slot_n} ({len(about)} chars, "
        f"role={role})"
    )
    low = about.lower()
    wardrobe_tokens = (
        "suit", "blazer", "jacket", "shirt", "tie", "dress", "skirt",
        "pants", "trousers", "sweater", "coat", "vest", "bow", "ribbon",
        "collar", "hat", "cap", "scarf", "glasses", "shoes", "boots",
        "socks", "apron", "uniform", "kimono", "hoodie", "cardigan",
    )
    clothing_present = any(tok in low for tok in wardrobe_tokens)
    if not clothing_present and (
        " human " in low or " person " in low or " man " in low
        or " woman " in low or " cat " in low or " dog " in low
        or " character " in low or " anthropomorphic " in low
        or " bipedal " in low or " figure " in low
    ):
        warnings.append(
            f"caption[{slot_n}]: no wardrobe tokens found for an "
            f"anthropomorphic subject — the reference image may be "
            f"missing clothes, or the vision captioner dropped them. "
            f"Consider re-uploading an image that clearly shows the "
            f"outfit so the caption stage can recover it. "
            f"(caption preview: {about[:160]}...)"
        )


def parse_generation_mode(mode: str) -> str:
    code = (mode or "").split(" - ", 1)[0].strip()
    return code if code in GENERATION_MODE_CODES else GENERATION_MODE_CODES[0]


def parse_seed_mode(mode: str) -> str:
    code = (mode or "").split(" - ", 1)[0].strip()
    return code if code in SEED_MODE_CODES else ""


def parse_caption_mode(mode: str) -> str:
    code = (mode or "").split(" - ", 1)[0].strip()
    return code if code in CAPTION_MODE_CODES else ""


def resolve_seed_unified(seed_mode: str) -> bool:
    """Resolve whether every scene shares one seed or each scene gets a
    per-clip-incremented seed (recommended).

    Defaults to per-scene increment when the widget value is
    unrecognised — that matches the upstream Context-Loop plugin's own
    default (per-scene seeds derived from one base; deterministic for
    checkpoint recovery, distinct per scene so consecutive clips don't
    repeat the same noise/motion rhythm) and this widget's
    "per_scene_increment - 每场seed递增(推荐)" default label.
    """
    code = parse_seed_mode(seed_mode)
    if code == "same_across_scenes":
        return True
    return False


def resolve_caption_controls(
    caption_mode: str,
    force_recaption_compat: bool,
    caption_cache_scope_compat: str,
) -> tuple[bool, str]:
    code = parse_caption_mode(caption_mode)
    if code == "cache_memory_disk":
        return False, "memory_disk"
    if code == "cache_memory_only":
        return False, "memory_only"
    if code == "no_cache":
        return False, "disabled"
    if code == "force_recaption_once":
        return True, "disabled"
    scope = str(caption_cache_scope_compat or "memory_disk")
    if scope not in {"memory_only", "memory_disk", "disabled"}:
        scope = "memory_disk"
    return bool(force_recaption_compat), scope


def postprocess_reply(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = _THINK_BLOCK_RE.sub("", text, count=1).strip()
    return text


def _multimodal_user_content(text: str, image_urls: list[str]) -> list[dict]:
    """Build an OpenAI-style user content list mixing text + image_url
    parts. Mirrors ``core.utils.build_multimodal_user_content`` (kept
    inline to avoid dragging the optional core import path into a
    place the existing tests already stub)."""
    parts: list[dict] = []
    if image_urls:
        for url in image_urls:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "auto"},
                }
            )
    if text:
        parts.append({"type": "text", "text": text})
    elif not parts:
        parts.append({"type": "text", "text": ""})
    return parts


def _default_concept() -> str:
    return (
        "A quiet, visually striking short film with one clear protagonist, "
        "one hero prop, and a simple emotional arc told in a continuous "
        "chain of moments."
    )


def _build_shots_from_turns(
    turns: list,
    budgets: list,
    *,
    scenes: Optional[list] = None,
    pacing=None,
    scene_target_total_sec: Optional[float] = None,
) -> list[dict]:
    """Synthesise a storyboard straight from dialogue turns + pacing budgets.

    This is the dialogue-driven short-circuit used by ``_auto_storyboard``
    when the upstream parser produced turn structure. The output mirrors
    the schema the LLM-backed path produces, so downstream stages don't
    need to know which path ran.

    Important: a turn may produce MULTIPLE budgets when it splits per-line
    (see ``estimate_shot_budget`` overbudget handling). We iterate the
    budget list, NOT the turn list — a ``zip(turns, budgets)`` here
    silently drops sub-budgets and corrupts the remaining turn mapping.
    The originating turn is looked up by ``budget.turn_index``.

    ``scenes`` (from ``group_budgets_into_scenes``) packs consecutive
    budgets into multi-turn scenes — used by the scene builder.
    Each scene becomes ONE shot carrying every line of its turns in
    order (``_dialogue_lines`` + ``_line_speakers``), so all three
    dialogue invariants still hold; only the scene granularity changes.

    When the parser produced a single ``(narrator)`` fallback turn (no
    ``speaker：text`` lines were found), we leave ``_dialogue_lines``
    unset so the per-shot post-validator does not enforce the dialogue
    invariant — there is nothing to enforce.
    """
    shots: list[dict] = []
    is_fallback = (
        len(turns) == 1 and turns[0].speaker == "(narrator)"
    )
    if scenes and pacing is not None:
        return _build_shots_from_scenes(
            turns,
            scenes,
            pacing,
            is_fallback=is_fallback,
            target_total_sec=scene_target_total_sec,
        )
    # Group budgets by turn_index so we can compute an intra-turn
    # sub-budget index (1..N within the same turn). The first budget
    # of a turn is index 1, etc.
    intra_index_by_budget: dict[int, int] = {}
    per_turn_counts: dict[int, int] = {}
    for budget in budgets:
        ti = budget.turn_index
        per_turn_counts[ti] = per_turn_counts.get(ti, 0) + 1
    seen_counts: dict[int, int] = {}
    for budget in budgets:
        ti = budget.turn_index
        seen_counts[ti] = seen_counts.get(ti, 0) + 1
        intra_index_by_budget[id(budget)] = seen_counts[ti]

    for budget in budgets:
        shot_idx = len(shots) + 1
        shot_id = f"scene_{shot_idx:02d}"
        # Look up the originating turn by the budget's turn_index
        # (handles both 1:1 and split-per-line cases).
        turn = turns[budget.turn_index] if budget.turn_index < len(turns) else None
        if turn is None:
            continue
        first_line = budget.lines[0] if budget.lines else ""
        if is_fallback:
            description = first_line or "(empty concept)"
            narrative_beat = "concept beat (no dialogue)"
            notes = "no dialogue lines; dialogue invariant not enforced"
            characters: list[str] = []
        else:
            sub_idx = intra_index_by_budget[id(budget)]
            sub_total = per_turn_counts[budget.turn_index]
            is_only_budget = sub_total == 1
            if is_only_budget:
                # One budget per turn -> single line OR multi-line single shot.
                if budget.line_count == 1:
                    description = (
                        f"[Turn {turn.speaker}] {first_line}"
                    )
                    narrative_beat = (
                        f"{turn.speaker} delivers the next line."
                    )
                else:
                    description = (
                        f"[Turn {turn.speaker}] {turn.speaker} delivers "
                        f"{budget.line_count} lines in one continuous "
                        f"shot: " + " / ".join(budget.lines)
                    )
                    narrative_beat = (
                        f"{turn.speaker} continuous monologue."
                    )
            else:
                # Turn was split across multiple sub-budgets.
                description = (
                    f"[Turn {turn.speaker} part {sub_idx}/{sub_total}] "
                    f"{turn.speaker} delivers {budget.line_count} of "
                    f"{turn.line_count} lines: " + " / ".join(budget.lines)
                )
                narrative_beat = (
                    f"{turn.speaker} continuous monologue (part "
                    f"{sub_idx}/{sub_total})."
                )
            notes = (
                f"dialogue invariant: {budget.line_count} verbatim <d> "
                f"block(s) required; do not split, paraphrase, or merge."
            )
            characters = [turn.speaker]
        shot_dict = {
            "id": shot_id,
            "description": description,
            "shot_type": "medium_shot",
            "camera_movement": "locked_off",
            "transition_in": "invisible_cut",
            "duration_seconds": budget.duration_sec,
            "narrative_beat": narrative_beat,
            "characters": characters,
            "props": [],
            "notes": notes,
            "_turn_speaker": turn.speaker,
            "_turn_index": budget.turn_index,
        }
        if not is_fallback:
            shot_dict["_dialogue_lines"] = list(budget.lines)
            shot_dict["_line_speakers"] = [turn.speaker] * len(budget.lines)
        shots.append(shot_dict)
    if not is_fallback and shots:
        log_pipeline(
            f"shot->turn mapping: {len(turns)} turn(s) -> {len(shots)} shot(s) "
            f"({len(shots) - len(turns)} split-per-line sub-shots)"
        )
    return shots


def _build_shots_from_scenes(
    turns: list,
    scenes: list,
    pacing,
    *,
    is_fallback: bool = False,
    target_total_sec: Optional[float] = None,
) -> list[dict]:
    """One shot per packed scene (multi-turn packing).

    Every line of the scene's budgets lands in one shot, in order, with
    a per-line speaker list (``_line_speakers``) so the speaker-ID
    contract can be enforced per spoken line. The scene duration is
    recomputed from the packing math (speech + intra pauses + one
    inter-budget pause per boundary + a single head/tail pad) and
    grid-rounded; a scene holding a single over-cap budget keeps that
    budget's own duration. ``target_total_sec`` (explicit user total)
    rescales every scene proportionally inside the 4-14 s band.
    """
    # First pass: per-scene metadata + raw durations.
    meta: list[dict] = []
    for scene in scenes:
        if not scene:
            continue
        lines: list[str] = []
        line_speakers: list[str] = []
        turn_indices: list[int] = []
        speakers: list[str] = []
        for budget in scene:
            if budget.turn_index >= len(turns):
                continue
            turn = turns[budget.turn_index]
            lines.extend(budget.lines)
            line_speakers.extend([turn.speaker] * len(budget.lines))
            if turn.speaker not in speakers:
                speakers.append(turn.speaker)
            if budget.turn_index not in turn_indices:
                turn_indices.append(budget.turn_index)
        if not lines:
            continue
        raw = _dlg_scene_raw_seconds(scene, pacing)
        if len(scene) == 1 and scene[0].raw_seconds > raw:
            # Single-budget scene whose own (over-cap) budget is the
            # honest duration — keep it instead of the packing math.
            duration = scene[0].duration_sec
        else:
            duration = length_to_seconds(seconds_to_length(raw))
        meta.append(
            {
                "lines": lines,
                "line_speakers": line_speakers,
                "turn_indices": turn_indices,
                "speakers": speakers,
                "duration": duration,
            }
        )
    # Optional explicit total: scale scene durations proportionally,
    # clamped to the H3 4-14 s single-generation band.
    if target_total_sec and meta:
        current = sum(m["duration"] for m in meta)
        if current > 0:
            factor = float(target_total_sec) / current
            for m in meta:
                scaled = min(14.0, max(4.0, m["duration"] * factor))
                m["duration"] = length_to_seconds(seconds_to_length(scaled))
    shots: list[dict] = []
    for shot_idx, m in enumerate(meta, start=1):
        names = " / ".join(m["speakers"])
        lines = m["lines"]
        description = (
            f"[Exchange {names}] {len(lines)} lines in one continuous "
            f"shot: " + " / ".join(lines[:3])
            + (" ..." if len(lines) > 3 else "")
        )
        notes = (
            f"dialogue invariant: {len(lines)} verbatim <d> block(s) "
            f"required across speakers [{names}]; do not split, "
            f"paraphrase, or merge; keep the lines in this order."
        )
        shots.append(
            {
                "id": f"scene_{shot_idx:02d}",
                "description": description,
                "shot_type": "medium_shot",
                "camera_movement": "locked_off",
                "transition_in": "invisible_cut",
                "duration_seconds": m["duration"],
                "narrative_beat": f"{names} multi-turn exchange.",
                "characters": list(m["speakers"]),
                "props": [],
                "notes": notes,
                "_turn_speaker": m["line_speakers"][0],
                "_turn_index": m["turn_indices"][0],
                "_turn_indices": m["turn_indices"],
                "_dialogue_lines": list(lines),
                "_line_speakers": m["line_speakers"],
            }
        )
    if shots:
        total_turns = len({ti for m in meta for ti in m["turn_indices"]})
        log_pipeline(
            f"shot->turn mapping (conservative packing): {total_turns} "
            f"turn(s) -> {len(shots)} multi-turn shot(s), each within the "
            f"H3 single-generation window"
        )
    return shots


def _insert_reaction_cuts(
    shots: list[dict],
    needed: int,
    warnings: list[str],
) -> list[dict]:
    """Mechanically top up a dialogue board with silent reaction shots.

    Used when the user's ``scene_count`` exceeds the per-line ceiling
    (one dialogue line per scene): the extra scenes become 切换镜头 —
    the classic multi-cam "cut to the listener" beat. Insertion points
    are speaker-change boundaries AFTER the first spoken shot (a
    reaction before anyone speaks would react to nothing); the reaction
    shot shows the NEXT speaker listening silently.

    Contract guarantees:
      - a reaction shot carries NO dialogue lines (no
        ``_dialogue_lines`` / ``_line_speakers`` / ``_turn_speaker``
        keys), so the 1-line -> 1-<d>-block invariant is untouched, the
        speaker-ID map ignores it, and the per-shot prompt generator
        writes a silent shot ("no dialogue lines assigned");
      - boundaries cycle until ``needed`` cuts are placed;
      - IDs are renumbered ``scene_01..N`` afterwards so downstream
        id-keyed stages stay consistent.

    When no speaker-change boundary exists (single-speaker monologue
    squeezed into one scene), the board is returned unchanged with a
    warning — we never invent speech to fill a count.
    """
    if needed <= 0 or not shots:
        return shots

    def _last_speaker(shot: dict) -> str:
        spk = [s for s in (shot.get("_line_speakers") or []) if s]
        return spk[-1] if spk else ""

    def _first_speaker(shot: dict) -> str:
        spk = [s for s in (shot.get("_line_speakers") or []) if s]
        return spk[0] if spk else ""

    # Boundary after original index i exists when shot i ends a spoken
    # run and some LATER shot opens a different speaker's run.
    boundaries: list[tuple[int, str]] = []
    for i in range(len(shots) - 1):
        s_i = _last_speaker(shots[i])
        if not s_i:
            continue
        for j in range(i + 1, len(shots)):
            s_j = _first_speaker(shots[j])
            if s_j:
                if s_j != s_i:
                    boundaries.append((i, s_j))
                break
    if not boundaries:
        warnings.append(
            f"scene_count: could not insert {needed} reaction cut(s) — no "
            "speaker-change boundary on the board (single-speaker or "
            "single-scene dialogue); emitting the dialogue scenes only"
        )
        return shots

    cuts_after: dict[int, int] = {}
    listeners_after: dict[int, str] = {}
    for k in range(needed):
        idx, listener = boundaries[k % len(boundaries)]
        cuts_after[idx] = cuts_after.get(idx, 0) + 1
        listeners_after.setdefault(idx, listener)

    out: list[dict] = []
    for i, shot in enumerate(shots):
        out.append(shot)
        n = cuts_after.get(i, 0)
        for _ in range(n):
            listener = listeners_after[i]
            out.append(
                {
                    "id": "scene_??",  # renumbered below
                    "description": (
                        f"[Reaction cut] {listener} listens in silence as "
                        "the previous line lands — a micro-expression "
                        "shift, no speech, lips closed. Brief hold, then "
                        "the exchange continues."
                    ),
                    "shot_type": "close_up",
                    "camera_movement": "locked_off",
                    "transition_in": "invisible_cut",
                    "duration_seconds": 4.0,
                    "narrative_beat": f"Silent reaction cut to {listener}.",
                    "characters": [listener],
                    "props": [],
                    "notes": (
                        "mechanically inserted reaction shot; NO dialogue "
                        "lines — do not invent speech or lip movement"
                    ),
                }
            )
    for j, shot in enumerate(out, start=1):
        shot["id"] = f"scene_{j:02d}"
    warnings.append(
        f"scene_count: inserted {needed} mechanical silent reaction "
        f"cut(s) at speaker-change boundaries (no dialogue touched)"
    )
    log_pipeline(
        f"reaction cuts: +{needed} silent scene(s); board is now "
        f"{len(out)} scene(s)"
    )
    return out


def _parse_first_json_object(text: str):
    """Tolerantly extract the first balanced ``{...}`` JSON object.

    The LLM extractor is told to emit a single line of JSON, but in
    practice it may prefix a one-line preamble or wrap the reply in
    markdown fences. This helper finds the first balanced top-level
    ``{...}`` and returns ``json.loads(...)`` on it, or ``None``.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        # try next '{'
        start = text.find("{", start + 1)
    return None


class H3LoopPromptEnhancer:
    """Loop-plan generator that talks to the project's
    LLMServiceConnector (shares the family logging + timeout pattern).

    Historical note: the module/file name still carries ``minimax``
    because the surrounding H3 workflow and upstream node names were
    introduced around MiniMax H3. The enhancer logic itself is intended
    to remain provider-agnostic at the connector boundary; backend
    transport/auth should live in connector implementations, not in this
    prompt generator."""

    def __init__(
        self,
        llm_service_connector: Any,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: Optional[int] = None,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._timeout_override = int(timeout) if timeout else None
        # In-memory image-caption cache. Key is the sha256 of the image
        # pixel bytes combined with the sha256 of the caption prompt
        # text, so any prompt upgrade automatically invalidates the
        # cache. The cache lives only for the lifetime of this node
        # instance; ComfyUI restart or new workflow clears it. The
        # force_recaption widget on __call__ bypasses the cache.
        self._caption_cache: dict[str, str] = {}
        # On-disk cache directory is computed lazily on first miss; see
        # `_caption_cache_disk_path` for path resolution. Files are
        # named by sha256(pixels) + sha256(prompt) so a prompt upgrade
        # invalidates both tiers atomically.
        self._caption_cache_dir: Optional[str] = None
        # LLM usage ledger for the node's ``summary`` output: one entry
        # per connector invoke — stage label, prompt chars, reply chars.
        # Token counts are ESTIMATES from the text (the connector API
        # returns plain strings, no usage payload): ~4 ASCII chars per
        # token, ~1 token per CJK char.
        self._usage: list[dict] = []
        # Filled by extract_dialogue: empty for genuine narration
        # ({"turns": []}); non-empty when JSON/span failed and the
        # caller is about to fall back to the narrator storyboard.
        self._dialogue_extract_warnings: list[str] = []

    def _record_usage(self, stage: str, messages, reply: str) -> None:
        prompt_text = "\n".join(
            str(m.get("content") or "") for m in messages or []
        )
        reply_text = reply or ""
        # Aggregate on the BASE stage name -- per-attempt / per-scene
        # suffixes ("shot[scene_02][attempt 2]") would fragment the
        # summary line into one bucket per call.
        base_stage = str(stage or "unknown").split("[", 1)[0]
        self._usage.append(
            {
                "stage": base_stage,
                "prompt_chars": len(prompt_text),
                "reply_chars": len(reply_text),
                "prompt_tokens": self._estimate_tokens(prompt_text),
                "reply_tokens": self._estimate_tokens(reply_text),
            }
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: CJK chars ≈ 1 token each, everything
        else ≈ 4 chars per token. Good enough for a summary line — the
        connector gives us text, not billing data."""
        if not text:
            return 0
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return cjk + (len(text) - cjk) // 4

    def _usage_summary_lines(self) -> list[str]:
        if not self._usage:
            return ["LLM requests: 0"]
        by_stage: dict[str, int] = {}
        prompt_chars = reply_chars = 0
        prompt_tokens = reply_tokens = 0
        for entry in self._usage:
            by_stage[entry["stage"]] = (
                by_stage.get(entry["stage"], 0) + 1
            )
            prompt_chars += entry["prompt_chars"]
            reply_chars += entry["reply_chars"]
            prompt_tokens += entry["prompt_tokens"]
            reply_tokens += entry["reply_tokens"]
        total_tokens = prompt_tokens + reply_tokens
        stage_text = ", ".join(
            f"{name} {count}" for name, count in by_stage.items()
        )
        return [
            f"LLM requests: {len(self._usage)} ({stage_text})",
            f"Tokens (estimated): ~{total_tokens:,} "
            f"(prompt ~{prompt_tokens:,} + reply ~{reply_tokens:,}; "
            f"chars {prompt_chars:,} / {reply_chars:,})",
        ]

    def _invoke(
        self,
        messages: list[dict],
        *,
        temperature: float,
        seed: Optional[int],
        stage: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            _check_interrupt(stage)
            if self._timeout_override is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=temperature,
                max_tokens=int(max_tokens) if max_tokens else self.max_tokens,
            )
            self._record_usage(stage, messages, out or "")
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out:
                mie_log(
                    f"H3LOOP {stage}: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"H3LOOP {stage}: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
            )
            return out.strip()
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    @staticmethod
    def _messages(system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def enhance_user_input_draft(
        self,
        draft: str,
        *,
        category: str,
        reference_mode: str,
        pacing: str = "",
        seed: Optional[int] = None,
    ) -> tuple[str, str]:
        """Run the standalone enhancer's rewrite inline (toggle on).

        Reuses ``MiniMaxH3LoopUserInputEnhancer``'s ``run_enhancer``
        (single code path for the prompt contract AND for the
        empty-reply retry policy — providers sometimes answer HTTP 200
        with empty content, which must not kill a whole multi-round
        run). Imported lazily (both deployment paths) because that
        module imports ``LOOP_CATEGORIES`` from this one; a module-
        level import would be circular.

        Returns ``(rewritten_user_input, advice_header)``. Raises
        ``RuntimeError`` after the retries are exhausted — no silent
        fallback to the raw draft: an unverified rewrite must not flow
        into the pipeline, and a raw draft that was never meant to be
        canonical would silently degrade the board.
        """
        try:
            from _mienodes_internal.nodes.llm.minimax_h3_loop_user_input_enhancer import (
                run_enhancer as _run_input_enhancer,
            )
        except ImportError:
            from .minimax_h3_loop_user_input_enhancer import (
                run_enhancer as _run_input_enhancer,
            )
        _check_interrupt("user_input_enhance")
        try:
            return _run_input_enhancer(
                self.llm,
                draft,
                category=category,
                reference_mode=reference_mode,
                pacing=pacing,
                seed=seed,
                # The rewrite is tuned for the standalone node's
                # sampling defaults; only the per-call timeout follows
                # this node.
                timeout=int(self._timeout_override or _DEFAULT_TIMEOUT),
                usage_sink=self._record_usage,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "MiniMax H3 Loop Plan Generator (auto-enhance): "
                f"{exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Dialogue extraction (LLM with mechanical span verification)
    # ------------------------------------------------------------------ #
    # This single call replaces BOTH the old regex parser AND the
    # intent classifier. The LLM does the semantic judgement (which
    # characters speak, where each line lives in the source); the
    # node does the mechanical judgement (does each returned span
    # literally match the source). The invariant "1 line == 1 <d>
    # block, verbatim" is now guaranteed by span anchoring rather
    # than by any string-format convention on the user.
    _EXTRACT_SYSTEM = (
        "You are a span-anchored dialogue extractor for a "
        "video-prompt generator. The user pasted a concept (any "
        "language, free-form). Your ONLY job is to identify every "
        "character utterance and return their character-exact "
        "spans as JSON.\n\n"
        "Output schema (single line of JSON, nothing else):\n"
        '  {"turns": [\n'
        '    {"speaker": "<name>", "lines": [\n'
        '      {"text": "<spoken words only>", "start": N, "end": M}\n'
        "    ]},\n"
        "  ...]}\n\n"
        "HARD RULES — violation of any rule invalidates the reply:\n"
        "  1. ``text`` MUST be the spoken words only. Do NOT include "
        "the lead verb (said, 问, 说), the surrounding quotes, or "
        "any punctuation outside the utterance.\n"
        "  2. concept[start:end] MUST equal ``text`` exactly "
        "(Python character slicing, not UTF-8 byte offsets; "
        "whitespace outside the utterance is OK to drop on either "
        "side; if you trim, adjust start/end accordingly). The node "
        "verifies mechanically — paraphrases, translations, or "
        "summaries will be rejected.\n"
        "  3. A turn is one speaker's contiguous run of lines. "
        "Same-speaker lines that are adjacent in the source merge "
        "into one turn (sequential lines from the same speaker are "
        "ONE turn with multiple lines, not multiple turns).\n"
        "  4. ``start`` and ``end`` are 0-based character offsets "
        "into the original concept string (len(concept) counts "
        "Unicode characters, so CJK glyphs are 1 each). "
        "``start >= 0``; ``end <= len(concept)``; ``start < end`` "
        "for every line.\n"
        "  5. Spans are NON-OVERLAPPING and ORDERED. Do not reuse "
        "characters. A line that ends at offset 200 cannot be "
        "followed by another starting at offset 150.\n"
        "  6. If the concept has NO dialogue at all, return "
        '{"turns": []}.\n'
        "  7. Do NOT invent dialogue. If a sentence is narrator "
        "narration, leave it out. The user can always re-run.\n"
        "  8. Output ONLY the JSON object on a single line, no "
        "markdown, no commentary.\n"
        "Example (offsets are character indices; quotes are NOT "
        "part of the spoken span):\n"
        "  concept = 公猫问：\"给够钱就行？\" 母猫答：\"给够钱。\"\n"
        "  len(concept) = 23\n"
        "  reply:\n"
        '  {"turns":[{"speaker":"公猫","lines":[{"text":"给够钱就行？","start":5,"end":11}]},'
        '{"speaker":"母猫","lines":[{"text":"给够钱。","start":18,"end":22}]}]}'
    )

    def extract_dialogue(self, concept: str) -> list:
        """Run the LLM span extractor and return a list of
        ``DialogueTurn`` with verified spans.

        The LLM is the ONLY source of the speaker/line judgements;
        the node mechanically verifies the spans. Wrong offsets that
        still uniquely locate the spoken text are rewritten in place.
        JSON / span failures retry with a corrective turn. A genuine
        ``{"turns": []}`` is narration (empty list, no warning).
        Exhausted retries also return an empty list, but fill
        ``self._dialogue_extract_warnings`` so the caller can surface
        the fallback instead of pretending there was no dialogue.
        ``InterruptProcessingException`` is re-raised.
        """
        self._dialogue_extract_warnings = []
        if not concept or not concept.strip():
            return []
        user_prompt = (
            "---BEGIN CONCEPT---\n"
            f"{concept}"
            "\n---END CONCEPT---\n\n"
            "Reply with ONLY the single-line JSON object. Length of "
            f"concept: {len(concept)} characters. Offsets are 0-based "
            "Python character indices (CJK glyph = 1)."
        )
        messages = self._messages(self._EXTRACT_SYSTEM, user_prompt)
        last_reason = "no reply"
        for attempt in range(1, 4):
            try:
                raw = self._invoke(
                    messages,
                    temperature=0.0,
                    seed=None,
                    stage=f"dialogue_extract[attempt {attempt}]",
                    max_tokens=16384,
                )
            except InterruptProcessingException:
                raise
            except Exception as exc:
                last_reason = f"invoke failed ({exc!r})"
                log_pipeline(
                    f"dialogue extractor: {last_reason}; "
                    f"attempt {attempt}/3"
                )
                continue
            raw = postprocess_reply(raw)
            if not raw:
                last_reason = "empty reply"
                log_pipeline(
                    f"dialogue extractor: empty reply on attempt "
                    f"{attempt}/3; retrying"
                )
                continue
            parsed = _parse_first_json_object(raw)
            if parsed is None:
                last_reason = f"no JSON object (head={raw[:80]!r})"
                log_pipeline(f"dialogue extractor: {last_reason}")
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not a JSON object. "
                            "Reply with ONLY {\"turns\":[...]} using "
                            "0-based character offsets so that "
                            "concept[start:end] equals each line's text."
                        ),
                    }
                ]
                continue
            raw_turns = parsed.get("turns")
            if raw_turns is None:
                raw_turns = []
            if not isinstance(raw_turns, list):
                last_reason = "turns is not a list"
                continue
            if not raw_turns:
                return []
            rows: list[tuple[str, _DLG_ExtractedLine]] = []
            for t in raw_turns:
                if not isinstance(t, dict):
                    continue
                speaker = str(t.get("speaker") or "").strip()
                if not speaker:
                    continue
                for entry in t.get("lines") or []:
                    if not isinstance(entry, dict):
                        continue
                    text = str(entry.get("text") or "").strip()
                    try:
                        s = int(entry.get("start"))
                        e = int(entry.get("end"))
                    except (TypeError, ValueError):
                        s, e = 0, 0
                    if not text:
                        continue
                    rows.append(
                        (speaker, _DLG_ExtractedLine(text=text, start=s, end=e))
                    )
            if not rows:
                last_reason = "no usable lines in turns"
                continue
            relocated, relocate_notes = _dlg_relocate_extracted_lines(
                concept, [line for _, line in rows]
            )
            errors = _dlg_validate_extracted_lines(concept, relocated)
            if errors:
                last_reason = errors[0]
                log_pipeline(
                    f"dialogue extractor: span validation failed on "
                    f"attempt {attempt}/3 ({last_reason[:120]!r})"
                )
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Span validation failed: "
                            + "; ".join(errors[:4])
                            + ". Offsets are 0-based Python character "
                            "indices (not UTF-8 bytes). concept[start:end] "
                            "must equal text. Reply with ONLY the JSON."
                        ),
                    }
                ]
                continue
            out: list[_DLG_DialogueTurn] = []
            for (speaker, _), line in zip(rows, relocated):
                if out and out[-1].speaker == speaker and line.start >= out[-1].end:
                    out[-1].lines.append(line.text)
                    out[-1].end = line.end
                    continue
                out.append(
                    _DLG_DialogueTurn(
                        speaker=speaker,
                        lines=[line.text],
                        start=line.start,
                        end=line.end,
                    )
                )
            if relocate_notes:
                log_pipeline(
                    "dialogue extractor: "
                    + "; ".join(relocate_notes[:6])
                )
            if not out:
                last_reason = "0 valid turns after merge"
                continue
            return out
        warning = (
            "dialogue extractor failed after 3 attempts "
            f"({last_reason}); falling back to the narrator "
            "storyboard — spoken lines will NOT be preserved verbatim"
        )
        log_pipeline(warning)
        self._dialogue_extract_warnings.append(warning)
        return []

    # ------------------------------------------------------------------ #
    # Stage 1: style-only prompt_prefix + CAST sheet
    # ------------------------------------------------------------------ #
    def _derive_local_prefix(
        self,
        idea: str,
        turns: list,
        tempo_directive: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Deterministic stage-1 replacement for short dialogue boards.

        The prefix paragraph is the concept's own narration text (the
        scene-setting prose outside the dialogue spans); when the
        concept has none, a minimal whole-video invariant line stands
        in. The CAST sheet maps every speaker to a consistency
        instruction — thin by LLM-prefix standards, but the verbatim
        speech blocks carry each speaker's name + fixed (S<n>) tag +
        CAST identity on first appearance, which is where the voice
        binding actually rides on this path.
        """
        spans = sorted((t.start, t.end) for t in turns if t.end > t.start)
        pieces: list[str] = []
        cursor = 0
        for s, e in spans:
            if s > cursor:
                pieces.append(idea[cursor:s])
            cursor = max(cursor, e)
        if cursor < len(idea):
            pieces.append(idea[cursor:])
        narration = " ".join(
            p.strip() for p in pieces if p and p.strip()
        )
        narration = " ".join(narration.split())
        prefix_lines = [
            narration
            or (
                "Consistent visual style, setting and characters across "
                "every clip of this production."
            )
        ]
        if (tempo_directive or "").strip():
            prefix_lines.append(tempo_directive.strip())
        cast = {
            t.speaker: (
                "speaking character; keep appearance and voice "
                "consistent across all clips"
            )
            for t in turns
            if t.speaker and t.speaker != "(narrator)"
        }
        return prefix_lines, cast

    def _synth_prefix(
        self,
        concept: str,
        category: str,
        language_name: str,
        shots: list[dict],
        *,
        seed: Optional[int],
        mode: str = "t2va",
        manifest: Optional[list[dict]] = None,
        tempo_directive: str = "",
    ) -> tuple[list[str], dict[str, str]]:
        """One LLM call producing (a) the whole-video-invariant prefix
        (art style / setting / palette / tempo / exclusions — never any
        character) and (b) the CAST sheet (one identity line per named
        character) that per-clip cast blocks are deterministically cut
        from. The roster comes from the storyboard's structured
        ``characters`` arrays; a missing/incomplete CAST sheet triggers
        one corrective retry naming the expected names."""
        roster: list[str] = []
        seen: set[str] = set()
        for shot in shots:
            for raw_name in shot.get("characters") or []:
                name = str(raw_name).strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    roster.append(name)
        roster_text = "\n".join(f"- {name}" for name in roster) or "(none named)"
        messages = self._messages(
            PREFIX_SYNTH_SYSTEM,
            build_prefix_user_text(
                concept,
                category,
                language_name,
                shots_digest=build_shots_digest(shots),
                mode_note=_mode_note_for_prefix(mode)
                + ("\n" + tempo_directive.strip() if tempo_directive.strip() else ""),
                manifest_digest=_manifest_digest(manifest or []),
                cast_roster=roster_text,
            ),
        )
        prefix_lines: Optional[list[str]] = None
        cast: dict[str, str] = {}
        problems: list[str] = []
        for attempt in range(1 + _PARSE_RETRIES):
            _check_interrupt(f"prefix[attempt {attempt + 1}]")
            attempt_messages = messages
            if attempt > 0:
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was incomplete: "
                            + "; ".join(problems)
                            + ". Reply again with BOTH artifacts: the prefix "
                            "paragraph (whole-video invariants only, NO "
                            "characters), then a line reading exactly 'CAST:' "
                            "followed by one 'name: identity line' for EVERY "
                            "roster name, using these exact names: "
                            + ", ".join(roster)
                            + "."
                        ),
                    }
                ]
            raw = postprocess_reply(
                self._invoke(
                    attempt_messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"prefix[attempt {attempt + 1}]",
                )
            )
            prefix_lines, cast = split_prefix_and_cast(raw)
            problems = []
            if not prefix_lines:
                problems.append("no prefix paragraph")
            if roster and cast:
                missing = [n for n in roster if n.lower() not in cast]
                if missing:
                    problems.append("CAST sheet missing names: " + ", ".join(missing))
            if not problems:
                return prefix_lines, cast
        raise RuntimeError(
            f"prompt_prefix synthesis failed: {'; '.join(problems)}"
        )

    # ------------------------------------------------------------------ #
    # Stage 0.6: image captioning (only when reference_mode != t2va and
    # any IMAGE socket is connected). One LLM call per image produces a
    # short subject description; entries feed the JSON manifest that
    # parse_references_text + validate_manifest + build_reference_directive
    # already know how to consume.
    # ------------------------------------------------------------------ #
    def _caption_images(
        self,
        *,
        images: Any,
        ref_code: str,
        seed: Optional[int],
        **kwargs: Any,
    ) -> tuple[list[dict], list[str]]:
        # ``force_recaption`` is taken from kwargs (not a typed param) so
        # existing tests that stub _caption_images with positional args do
        # not break on the new keyword.
        """Caption the connected IMAGE batch per ``ref_code``. Returns
        ``(manifest, warnings)`` where manifest is a list ``[{slot, about,
        role}]`` ready for ``parse_references_text``, and warnings are
        human-readable notes that surface in the preflight report.

        Routing mirrors the upstream H3 Context Loop picture sinks:
          - t2va: images are ignored (no upstream image wiring).
          - i2va: only the FIRST frame is captioned (the upstream
            ``MiniMaxH3ChainFirstSceneImage`` only consumes the opening
            image). Any extra frames are dropped with a warning naming them.
          - fl2va: every frame becomes its own manifest slot in
            connection order. The upstream ``MiniMaxH3ChainFrameIndexSwitch``
            handles per-scene wrapping — the user writes the wrap
            intent ("从图一到图二" / "A->B->A->B") in user_input, and
            the storyboard LLM distributes the frames accordingly.
          - ref2va: every frame is active in every scene (six-section
            schema); one caption per frame in connection order.
        """
        warnings: list[str] = []
        if ref_code == "t2va":
            if images is not None:
                warnings.append(
                    "reference_mode=t2va ignores the images socket; switch "
                    "to i2va / fl2va / ref2va to use images"
                )
            return [], warnings
        if images is None:
            return [], warnings
        # Single un-batched image (H,W,C): normalize to a batch of 1 —
        # mirrors core.utils.image_tensor_batch_to_data_urls, which also
        # accepts ndim==3. Without this a lone (H,W,C) tensor would be
        # silently dropped.
        if hasattr(images, "ndim") and images.ndim == 3:
            images = images[None, ...]
        if not hasattr(images, "ndim") or images.ndim != 4:
            return [], warnings
        frame_count = int(images.shape[0])
        if frame_count == 0:
            return [], warnings
        # Honor MAX_REFERENCE_IMAGES: truncate excess with a warning
        # rather than failing — the user gets the most useful subset.
        used = min(frame_count, _MAX_REFERENCE_IMAGES)
        if frame_count > _MAX_REFERENCE_IMAGES:
            warnings.append(
                f"only the first {_MAX_REFERENCE_IMAGES} of "
                f"{frame_count} images were captioned (max {_MAX_REFERENCE_IMAGES})"
            )
            log_pipeline(
                f"stage0.6: {frame_count} images supplied; only the first "
                f"{_MAX_REFERENCE_IMAGES} will be captioned"
            )

        # Determine how many manifest entries to emit.
        if ref_code == "i2va":
            emit_count = 1  # extras dropped
            if frame_count > 1:
                dropped = frame_count - 1
                warnings.append(
                    f"reference_mode=i2va only uses the first frame; "
                    f"dropping {dropped} extra image(s) (the upstream "
                    "MiniMax H3 First-Scene Image Gate consumes one)"
                )
                log_pipeline(
                    f"stage0.6: i2va only uses the first frame; "
                    f"dropping {frame_count - 1} extra image(s) "
                    "(upstream First-Scene Image Gate consumes one)"
                )
        else:
            emit_count = used  # fl2va / ref2va: keep all (within cap)

        manifest: list[dict] = []
        caption_budget = max(
            int(self.max_tokens), _DEFAULT_MAX_TOKENS_CAPTION
        )
        # Caption-cache: hash on pixel bytes + caption-prompt version, so
        # any upgrade to caption_reference_prompt() invalidates the
        # cache automatically. force_recaption (passed via kwargs by
        # __call__) bypasses the cache entirely.
        force_recaption = bool(kwargs.get("force_recaption", False))
        requested_cache_scope = str(
            kwargs.get("caption_cache_scope", "memory_disk") or "memory_disk"
        )
        effective_cache_scope = "disabled" if force_recaption else requested_cache_scope
        if effective_cache_scope not in {"memory_only", "memory_disk", "disabled"}:
            effective_cache_scope = "memory_disk"
        if effective_cache_scope == "memory_disk" and self._caption_cache_dir is None:
            self._caption_cache_dir = _caption_cache_disk_root()
        cache_root_note = (
            os.path.abspath(self._caption_cache_dir)
            if self._caption_cache_dir
            else "<disabled>"
        )
        log_pipeline(
            f"stage0.6 caption cache scope={effective_cache_scope} root={cache_root_note}"
        )
        cache_hits = 0
        cache_misses = 0
        for slot_n in range(1, emit_count + 1):
            # Honour ComfyUI's interrupt between per-frame LLM calls.
            _check_interrupt(f"caption[{slot_n}]")
            slot_tensor = images[slot_n - 1 : slot_n]
            slot_urls = image_tensor_batch_to_data_urls(slot_tensor)
            if not slot_urls:
                raise RuntimeError(
                    f"{slot_n}: failed to encode image tensor to data URLs"
                )
            about = ""
            cache_key = _caption_cache_key(slot_urls[0], caption_reference_prompt())
            short_key = cache_key[:12]
            if effective_cache_scope != "disabled":
                # Tier 1: in-memory
                if cache_key in self._caption_cache:
                    about = self._caption_cache[cache_key]
                    log_pipeline(
                        f"stage0.6 caption[{slot_n}] cache hit (memory, key={short_key}, "
                        f"{len(about)} chars)"
                    )
                elif effective_cache_scope == "memory_disk":
                    # Tier 2: on-disk (persists across restarts / workflows)
                    disk_about = _read_caption_cache_disk(
                        self._caption_cache_dir, cache_key
                    )
                    if disk_about is not None:
                        about = disk_about
                        # Promote to in-memory for the rest of this run.
                        self._caption_cache[cache_key] = about
                        log_pipeline(
                            f"stage0.6 caption[{slot_n}] cache hit (disk, key={short_key}, "
                            f"{len(about)} chars)"
                        )
                if about:
                    cache_hits += 1
                    _emit_caption_manifest_entry(
                        manifest=manifest,
                        slot_n=slot_n,
                        about=about,
                        ref_code=ref_code,
                        warnings=warnings,
                    )
                    continue
            cache_misses += 1
            if effective_cache_scope != "disabled":
                log_pipeline(
                    f"stage0.6 caption[{slot_n}] cache miss (key={short_key})"
                )
            user_text = (
                f"{slot_n} reference image(s). Write ONE tight English "
                "paragraph (2-4 sentences, ~60-100 words) that names the "
                "subjects and every recurring visual feature an AI video "
                "generator must hold constant across all shots: identity, "
                "species/breed, build, wardrobe, accessories, palette, "
                "lighting direction, framing, and any distinctive marks. "
                "No mood abstractions, no shot grammar, no timestamps. "
                "Reply with the paragraph ONLY."
            )
            messages = [
                {"role": "system", "content": caption_reference_prompt()},
                {
                    "role": "user",
                    "content": _multimodal_user_content(user_text, slot_urls),
                },
            ]
            raw = postprocess_reply(
                self._invoke(
                    messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"caption[{slot_n}]",
                    max_tokens=caption_budget,
                )
            )
            if not raw:
                raise RuntimeError(
                    f"{slot_n}: captioning returned empty reply"
                )
            about = raw.strip()
            # Cache the new caption before appending to manifest; the
            # cache key already encodes the prompt version, so the next
            # call (with the same image + no force_recaption) hits.
            self._caption_cache[cache_key] = about
            if effective_cache_scope == "memory_disk":
                write_err = _write_caption_cache_disk(
                    self._caption_cache_dir,
                    cache_key,
                    about,
                )
                if write_err:
                    warnings.append(
                        f"caption[{slot_n}]: disk cache write failed ({write_err}); "
                        "memory cache remains available for this run"
                    )
                    log_pipeline(
                        f"stage0.6 caption[{slot_n}] disk cache write failed "
                        f"(key={short_key}): {write_err}"
                    )
            _emit_caption_manifest_entry(
                manifest=manifest,
                slot_n=slot_n,
                about=about,
                ref_code=ref_code,
                warnings=warnings,
            )
        # Cache summary: surface hit / miss counts in the preflight report
        # so users can confirm caches are working.
        total_captions = cache_hits + cache_misses
        if total_captions > 0:
            hit_pct = (100 * cache_hits) // total_captions if total_captions else 0
            label = "forced recaption" if force_recaption else "caption cache"
            warnings.append(
                f"{label}: {cache_hits}/{total_captions} frame(s) hit "
                f"({hit_pct}%); {cache_misses}/{total_captions} re-captioned"
            )
        return manifest, warnings
    def _auto_storyboard(
        self,
        concept: str,
        scene_count: int,
        total_duration_seconds: int,
        category: str,
        language: str,
        *,
        seed: Optional[int],
        reference_digest: str = "",
        storyboard_bias: str = "balanced",
        dialogue_turns: Optional[list] = None,
        shot_budgets: Optional[list] = None,
        turn_scenes: Optional[list] = None,
        pacing_obj=None,
        scene_target_total_sec: Optional[float] = None,
    ) -> tuple[list[dict], list[str]]:
        """Split user_input into a storyboard (one LLM call) whose
        per-shot durations sum close to the whole-board budget.

        When ``dialogue_turns`` + ``shot_budgets`` are supplied (the
        dialogue-driven path), we bypass the LLM entirely and synthesise
        a storyboard straight from the parser output. That guarantees the
        turn structure survives intact — the LLM is not given a chance to
        merge or split turns, so invariant (1) (1:1 line -> <d> block)
        and invariant (3) (same-speaker consecutive lines in one scene)
        are satisfied deterministically.

        ``turn_scenes`` + ``pacing_obj`` pack
        consecutive turns into multi-turn scenes; see
        ``group_budgets_into_scenes``.
        """
        if dialogue_turns and shot_budgets and not (
            len(dialogue_turns) == 1
            and dialogue_turns[0].speaker == "(narrator)"
        ):
            shots = _build_shots_from_turns(
                dialogue_turns,
                shot_budgets,
                scenes=turn_scenes,
                pacing=pacing_obj,
                scene_target_total_sec=scene_target_total_sec,
            )
            digest = ", ".join(s["id"] for s in shots)
            log_pipeline(
                f"auto-storyboard[turns]: built {len(shots)} shots from "
                f"{len(dialogue_turns)} dialogue turn(s): {digest}"
            )
            return shots, []
        # Empty concept or no dialogue lines: fall through to the LLM-backed
        # storyboard path (legacy behaviour).
        user_text = build_storyboard_user_text(
            (concept or "").strip(),
            int(scene_count or 0),
            AUTO_STORYBOARD_STYLE,
            category,
            language,
            total_duration_seconds=int(total_duration_seconds),
            reference_digest=reference_digest,
            split_bias=storyboard_bias,
        )
        messages = self._messages(SYSTEM_STORYBOARD_PROMPT, user_text)
        # The storyboard reply is the pipeline's longest single output; a
        # small widget value truncates the JSON array mid-object (live
        # failure: "no JSON array found"). Floor the budget for this stage.
        storyboard_budget = max(self.max_tokens, 16384)
        last_error: Optional[Exception] = None
        last_head = "<no reply>"
        for attempt in range(1 + _PARSE_RETRIES):
            _check_interrupt(f"storyboard[attempt {attempt + 1}]")
            attempt_messages = messages
            if attempt > 0:
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply could not be parsed. Reply with ONLY the JSON array. "
                            + PARSE_RETRY_CORRECTION
                        ),
                    }
                ]
            raw = postprocess_reply(
                self._invoke(
                    attempt_messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"storyboard[attempt {attempt + 1}]",
                    max_tokens=storyboard_budget,
                )
            )
            if not raw:
                last_error = ValueError("empty storyboard reply")
                last_head = "<empty reply>"
                continue
            last_head = raw[:200]
            try:
                raw_shots = extract_json_array(raw)
            except ValueError as exc:
                last_error = exc
                log_pipeline(
                    f"auto-storyboard parse failed ({exc}); retry head={last_head!r}"
                )
                continue
            shots, warnings = normalize_shots(raw_shots, int(scene_count))
            digest = ", ".join(s["id"] for s in shots)
            log_pipeline(f"auto-storyboard built {len(shots)} shots: {digest}")
            return shots, warnings
        raise RuntimeError(
            f"auto-storyboard reply unparseable after {1 + _PARSE_RETRIES} attempts: "
            f"{last_error}; last reply head: {last_head!r}"
        )

    # ------------------------------------------------------------------ #
    # Stage 2, per-shot mode
    # ------------------------------------------------------------------ #
    def _generate_shot_prompt(
    self,
    *,
    concept: str,
    prefix_text: str,
    category: str,
    shot: dict,
    clip_index: int,
    clip_count: int,
    prev_lines: Optional[list[str]],
    prev_id: str,
    prev_subject_definitions: Optional[list[str]],
    duration_seconds: int,
    language_name: str,
    seed: Optional[int],
    mode: str = "t2va",
    manifest: Optional[list[dict]] = None,
    prev_soundscape_ref2v: Optional[list[str]] = None,
    cast_block: str = "",
    dialogue_lines: Optional[list[str]] = None,
    turn_index: Optional[int] = None,
    turn_speaker: Optional[str] = None,
    speaker_id_map: Optional[dict] = None,
        line_speakers: Optional[list[str]] = None,
        first_appearance_speakers: Optional[set] = None,
        spatial_layout: Optional[dict] = None,
        tempo_directive: str = "",
        speaker_identities: Optional[dict] = None,
    ) -> list[str]:
        code = parse_reference_mode(mode)
        schema = schema_for_mode(mode)
        effective_manifest = list(manifest or [])
        # Reference directive (D6): first-sentence idiom / subject binding.
        # Category widget drives the spoken-scene / genre contract injection
        # in ref2va; other modes pass through unchanged.
        reference_directive = build_reference_directive(
            mode,
            effective_manifest,
            clip_index,
            duration_seconds,
            category=category,
        )
        # Continuation block: ref2va uses the six-section carry-over; the
        # default three-section template stays for t2va/i2va/fl2va.
        if clip_index > 1 and prev_lines:
            if schema == SCHEMA_SIX:
                continuation = build_continuation_block_ref2v(
                    prev_id,
                    prev_subject_definitions or "",
                    description_body(prev_lines, schema=schema),
                    sound_body(prev_lines),
                )
            else:
                continuation = build_continuation_block(
                    prev_id,
                    description_body(prev_lines, schema=schema),
                    sound_body(prev_lines),
                )
        else:
            continuation = (
                "This clip OPENS the chain: establish the subject's full "
                "appearance, the setting, and the visual style; end the clip "
                "mid-action so the next clip can continue it."
            )
        user_text = build_shot_user_text(
            concept=concept,
            prefix_text=prefix_text,
            category=category,
            continuation_block=continuation,
            shot=shot,
            clip_index=clip_index,
            clip_count=clip_count,
            duration_seconds=duration_seconds,
            language_name=language_name,
            reference_directive=reference_directive,
            manifest_digest=_manifest_digest(effective_manifest),
            cast_block=cast_block,
            reference_mode=mode,
            dialogue_lines=dialogue_lines,
            turn_index=turn_index,
            turn_speaker=turn_speaker,
            speaker_id_map=speaker_id_map,
            line_speakers=line_speakers,
            first_appearance_speakers=first_appearance_speakers,
            spatial_layout=spatial_layout,
            tempo_directive=tempo_directive,
        )
        # System prompt dispatch: ref2va uses the six-section addendum.
        system = (
            shot_system_prompt_ref2v() if schema == SCHEMA_SIX else shot_system_prompt()
        )
        messages = self._messages(system, user_text)
        last_error: Optional[Exception] = None
        for attempt in range(1 + _PARSE_RETRIES):
            _check_interrupt(f"shot[{shot['id']}][attempt {attempt + 1}]")
            # Retry carries a corrective user turn (mirrors the storyboard
            # pipeline): identical resend taught the model nothing.
            attempt_messages = messages
            if attempt > 0:
                expected = (
                    "six bare section headers in this exact order: "
                    "subject_definitions: / summary: / retention_analysis: / "
                    "detailed_description: / overall_soundscape: / "
                    "non_diegetic_music:"
                    if schema == SCHEMA_SIX
                    else "three bare section headers in this exact order: "
                    "integrated_multimodal_description: / overall_soundscape: / "
                    "non_diegetic_music:"
                )
                attempt_messages = messages + [{
                    "role": "user",
                    "content": (
                        f"Your previous reply could not be parsed / violated the shot contract ({last_error}). Try again. "
                        f"Reply with ONLY the {expected} "
                        "Each header on its own line, one blank line between "
                        "sections, no prose before or after, no markdown."
                    ),
                }]
            raw = postprocess_reply(
                self._invoke(
                    attempt_messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"shot[{shot['id']}][attempt {attempt + 1}]",
                )
            )
            try:
                if schema == SCHEMA_SIX:
                    lines_out = split_six_sections(raw)
                else:
                    lines_out = split_three_sections(raw)
                    if code in ("i2va", "fl2va"):
                        lines_out = ensure_keyframe_idiom(
                            lines_out,
                            mode=code,
                            manifest=manifest or [],
                            clip_index=clip_index,
                        )
                # Ref2VA timestamp contract: no clock/seconds notation in
                # detailed_description. Caught here so the retry can fix it.
                if schema == SCHEMA_SIX and _REF2VA_FORBIDDEN_TIMECODE_RE.search(
                    description_body(lines_out, schema=schema)
                ):
                    raise ValueError(
                        "ref2va contract: detailed_description contains forbidden "
                        "timestamp (clock/seconds notation like 'At 00:03.500', "
                        "'from 0.0s to 5.2s')."
                    )
                # Dialogue-as-data: the spoken words never pass through
                # the model on this path. Scrub anything it wrote
                # despite the lock, then append the verbatim <d> blocks
                # assembled from the locked lines. The invariant check
                # below is a self-check — it cannot fire by
                # construction, but stays as the hard gate.
                if dialogue_lines:
                    scrubbed_text, scrub_notes = scrub_dialogue_from_prompt_text(
                        "\n".join(lines_out), list(dialogue_lines)
                    )
                    if scrub_notes:
                        log_pipeline(
                            f"shot {shot['id']}: scrubbed model-written "
                            "dialogue ("
                            + "; ".join(scrub_notes[:3])
                            + ")"
                        )
                        lines_out = scrubbed_text.split("\n")
                    blocks = assemble_dialogue_line_blocks(
                        list(dialogue_lines),
                        line_speakers=line_speakers,
                        turn_speaker=turn_speaker or "",
                        speaker_id_map=speaker_id_map,
                        speaker_identities=speaker_identities,
                        first_appearance_speakers=first_appearance_speakers,
                    )
                    lines_out = append_dialogue_blocks_to_sections(
                        lines_out, blocks, schema=schema
                    )
                    errors = validate_dialogue_invariantity(
                        [{"prompt": lines_out}],
                        [_DLG_DialogueTurn(speaker=turn_speaker or "", lines=list(dialogue_lines))],
                    )
                    if errors:
                        raise ValueError(
                            "dialogue invariant violated for "
                            f"{shot['id']}: {errors[0]}"
                        )
                # Speaker-ID contract: with dialogue-as-data the tags
                # are code-owned — every appended block carries the
                # speaker's name, fixed (S<n>) tag, and CAST identity
                # (first appearance only) straight from the stage-0.5
                # turn data. The old post-hoc repair (rewriting tags the
                # model misplaced, retrying on genderless first
                # appearances) targeted model-written speech and is
                # obsolete on this path; the invariant check above is
                # the hard gate.
                return lines_out
            except ValueError as exc:
                last_error = exc
                log_pipeline(f"shot {shot['id']}: parse failed ({exc}); retrying")
        if schema == SCHEMA_SIX:
            raise RuntimeError(
                f"clip {shot['id']}: unparseable six-section reply after "
                f"{1 + _PARSE_RETRIES} attempts: {last_error}"
            )
        raise RuntimeError(
            f"clip {shot['id']}: unparseable three-section reply after "
            f"{1 + _PARSE_RETRIES} attempts: {last_error}"
        )

    # ------------------------------------------------------------------ #
    # Stage 2, single-call mode
    # ------------------------------------------------------------------ #
    def _generate_all_shots_single_call(
        self,
        *,
        concept: str,
        prefix_text: str,
        category: str,
        shots: list[dict],
        duration_seconds: int,
        language_name: str,
        seed: Optional[int],
        cast_sheet: str = "",
        dialogue_turns: Optional[list] = None,
        speaker_id_map: Optional[dict] = None,
        spatial_layout: Optional[dict] = None,
        tempo_directive: str = "",
        speaker_identities: Optional[dict] = None,
    ) -> dict[str, list[str]]:
        user_text = build_single_call_user_text(
            concept=concept,
            prefix_text=prefix_text,
            category=category,
            shots=shots,
            duration_seconds=duration_seconds,
            language_name=language_name,
            cast_sheet=cast_sheet,
            speaker_id_map=speaker_id_map,
            spatial_layout=spatial_layout,
            tempo_directive=tempo_directive,
        )
        messages = self._messages(shot_system_prompt(), user_text)
        last_error: Optional[Exception] = None
        last_head = "<no reply>"
        for attempt in range(1 + _PARSE_RETRIES):
            _check_interrupt(f"single_call[attempt {attempt + 1}]")
            attempt_messages = messages
            if attempt > 0:
                correction = PARSE_RETRY_CORRECTION
                if last_error and "dialogue invariant" in str(last_error):
                    correction += (
                        " Ensure each shot's integrated_multimodal_description "
                        "contains exactly the dialogue lines assigned to that shot "
                        "(verbatim, no split, no paraphrase)."
                    )
                attempt_messages = messages + [
                    {"role": "user", "content": correction}
                ]
            raw = postprocess_reply(
                self._invoke(
                    attempt_messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"single[attempt {attempt + 1}]",
                )
            )
            if not raw:
                last_error = ValueError("empty single-call reply")
                last_head = "<empty reply>"
                continue
            last_head = raw[:200]
            try:
                items = extract_json_array(raw)
            except ValueError as exc:
                last_error = exc
                log_pipeline(
                    f"single-call parse failed ({exc}); retry head={last_head!r}"
                )
                continue
            result = self._split_single_call_items(items, shots)
            if result is None:
                last_error = ValueError("single-call reply missing clips")
                continue
            # Dialogue-as-data: scrub any model-written dialogue, then
            # append the verbatim <d> blocks assembled from the locked
            # per-shot lines. First-appearance tracking walks the board
            # in order (single_call is t2va-only: three-section schema).
            if dialogue_turns:
                seen_firsts: set = set()
                for shot in shots:
                    shot_id = shot["id"]
                    if shot_id not in result:
                        continue
                    dl = shot.get("_dialogue_lines") or []
                    if not dl:
                        continue
                    speaker = str(shot.get("_turn_speaker") or "").strip()
                    spk_per_line = list(shot.get("_line_speakers") or [])
                    if len(spk_per_line) != len(dl):
                        spk_per_line = [speaker] * len(dl)
                    scrubbed, scrub_notes = scrub_dialogue_from_prompt_text(
                        "\n".join(result[shot_id]), list(dl)
                    )
                    if scrub_notes:
                        log_pipeline(
                            f"single-call {shot_id}: scrubbed model-written "
                            "dialogue (" + "; ".join(scrub_notes[:3]) + ")"
                        )
                    blocks = assemble_dialogue_line_blocks(
                        list(dl),
                        line_speakers=spk_per_line,
                        turn_speaker=speaker,
                        speaker_id_map=speaker_id_map,
                        speaker_identities=speaker_identities,
                        first_appearance_speakers=set(spk_per_line) - seen_firsts,
                    )
                    result[shot_id] = append_dialogue_blocks_to_sections(
                        (
                            scrubbed.split("\n")
                            if scrub_notes
                            else result[shot_id]
                        ),
                        blocks,
                        schema=SCHEMA_THREE,
                    )
                    seen_firsts.update(spk_per_line)
            # Dialogue invariant post-validate (per-shot).
            sid_errors: list[str] = []
            if dialogue_turns:
                errors: list[str] = []
                for shot in shots:
                    shot_id = shot["id"]
                    if shot_id not in result:
                        continue
                    lines = shot.get("_dialogue_lines") or []
                    if not lines:
                        continue
                    # Find the matching turn by index.
                    t_idx = shot.get("_turn_index")
                    if t_idx is None or t_idx >= len(dialogue_turns):
                        continue
                    turn = dialogue_turns[t_idx]
                    errs = validate_dialogue_invariantity(
                        [{"prompt": result[shot_id]}],
                        [_DLG_DialogueTurn(
                            speaker=turn.speaker,
                            lines=list(lines),
                        )],
                    )
                    errors.extend(errs)
                if errors:
                    last_error = ValueError(
                        f"dialogue invariant violated: {errors[0]}"
                    )
                    log_pipeline(
                        f"single-call: {last_error}; retrying"
                    )
                    continue
            # Speaker-ID contract: with dialogue-as-data the appended
            # blocks carry name + fixed (S<n>) tag + CAST identity
            # (first appearance) straight from the turn data — the old
            # post-hoc repair loop targeted model-written speech and is
            # obsolete here (see the per-shot path).
            return result
        raise RuntimeError(
            f"single-call reply unparseable after {1 + _PARSE_RETRIES} attempts: "
            f"{last_error}; last reply head: {last_head!r}"
        )

    @staticmethod
    def _split_single_call_items(
        items: list[Any], shots: list[dict]
    ) -> Optional[dict[str, list[str]]]:
        """Map single-call JSON entries onto the storyboard ids."""
        out: dict[str, list[str]] = {}
        by_id = {}
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                by_id[str(item["id"]).strip().lower()] = item
        for shot in shots:
            item = by_id.get(shot["id"].lower())
            if item is None:
                return None
            desc = str(item.get("integrated_multimodal_description") or "").strip()
            sound = str(item.get("overall_soundscape") or "").strip()
            music = str(item.get("non_diegetic_music") or "").strip()
            if not desc or not sound:
                return None
            music = music or DEFAULT_MUSIC_LINE
            out[shot["id"]] = (
                ["integrated_multimodal_description:"]
                + desc.split("\n")
                + ["", "overall_soundscape:"]
                + sound.split("\n")
                + ["", "non_diegetic_music:"]
                + music.split("\n")
            )
        return out

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        *,
        user_input: str = "",
        total_duration_seconds: int = 0,
        scene_count: int = 0,
        pacing: str = _PACING_LABELS[1],
        generation_mode: str = "per_shot",
        category: str = "",
        output_language: str = "en",
        seed: Optional[int] = None,
        seed_mode: str = "",
        reference_mode: str = REFERENCE_MODES[0],
        references_text: str = "",
        images: Any = None,
        caption_mode: str = "",
        force_recaption: bool = False,
        caption_cache_scope: str = "memory_disk",
        enhance_user_input: bool = False,
    ) -> dict:
        warnings: list[str] = []
        pacing_key = _PACING_LABEL_TO_KEY.get(
            (pacing or "").split(" - ", 1)[0].strip().lower(), "normal"
        )
        if not hasattr(self.llm, "invoke"):
            raise RuntimeError(
                "auto-storyboard requires an LLM connector with invoke(); "
                "connect a full LLMServiceConnector"
            )
        raw_input = (user_input or "").strip()
        # Auto-enhance (toggle on + non-empty draft): rewrite the rough
        # draft into the canonical user_input format first. The
        # category / reference_mode passed down are THIS node's widget
        # values — single source of truth, no mirrored widgets to keep
        # in sync. Failure raises (see enhance_user_input_draft); we
        # never silently plan from the raw draft after asking for a
        # rewrite.
        enhance_header: Optional[str] = None
        if raw_input and enhance_user_input:
            idea, enhance_header = self.enhance_user_input_draft(
                raw_input,
                category=category,
                reference_mode=reference_mode,
                pacing=pacing,
                seed=seed,
            )
            log_pipeline(
                "auto-enhance: user_input rewritten "
                f"({len(raw_input)} -> {len(idea)} chars)"
            )
        else:
            idea = raw_input or _default_concept()
            if not raw_input:
                warnings.append(
                    "user_input empty: used the built-in default concept"
                )

        # Single source of truth for the budget. 0 = auto: derived from
        # dialogue pacing + line count below.
        effective_total_duration = int(total_duration_seconds or 0)

        # Image socket detection (plan v4 surface).
        has_images = images is not None and (
            not hasattr(images, "ndim")
            or images.ndim != 4
            or images.shape[0] > 0
        )
        if (
            images is not None
            and hasattr(images, "ndim")
            and images.ndim == 4
            and int(images.shape[0]) == 0
        ):
            warnings.append(
                "images socket connected with an empty batch; no images were captioned"
            )

        ref_code = parse_reference_mode(reference_mode)
        gen_code = parse_generation_mode(generation_mode)
        seed_unified = resolve_seed_unified(seed_mode)
        effective_force_recaption, effective_caption_cache_scope = (
            resolve_caption_controls(
                caption_mode,
                bool(force_recaption),
                caption_cache_scope,
            )
        )
        schema = schema_for_mode(ref_code)
        if gen_code == "single_call" and ref_code != "t2va":
            raise RuntimeError(
                f"single_call generation mode is not supported for {ref_code}; "
                "use per_shot mode"
            )

        if ref_code != "t2va" and not has_images and not (references_text or "").strip():
            raise RuntimeError(
                f"{ref_code} requires images on the images socket or a "
                f"references_text manifest; provide one of them to run "
                f"{ref_code}"
            )

        manifest: list[dict] = []
        if references_text and not has_images:
            try:
                manifest = parse_references_text(
                    references_text, reference_mode=ref_code
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"references_text manifest invalid: {exc}"
                ) from exc
            manifest_errors = validate_manifest(manifest, ref_code)
            if manifest_errors:
                raise RuntimeError(
                    "references_text manifest failed reference_mode="
                    + ref_code
                    + " validation: "
                    + "; ".join(manifest_errors)
                )
        if has_images:
            try:
                captioned_manifest, caption_warnings = self._caption_images(
                    images=images,
                    ref_code=ref_code,
                    seed=seed,
                    force_recaption=effective_force_recaption,
                    caption_cache_scope=effective_caption_cache_scope,
                )
            except TypeError:
                # Backward-compat for tests/stubs monkeypatching the older
                # 3-arg signature: _caption_images(*, images, ref_code, seed).
                captioned_manifest, caption_warnings = self._caption_images(
                    images=images,
                    ref_code=ref_code,
                    seed=seed,
                )
            warnings.extend(caption_warnings)
            if captioned_manifest:
                manifest = captioned_manifest
                log_pipeline(
                    f"stage0.6 manifest from {len(captioned_manifest)} "
                    f"captioned image(s)"
                )
            else:
                if ref_code == "t2va":
                    manifest = []
                else:
                    raise RuntimeError(
                        f"{ref_code} requires images and the caption stage "
                        "returned no usable manifest; reconnect the image batch"
                    )

        # ---- Stage 0.5: auto storyboard ------------------------------ #
        if _ns_lazy is None:
            shots = []
            # Stage 2's single_call branch references the stage-0.5
            # extraction; keep the name bound on this (degenerate) path
            # too. The empty-entries RuntimeError below fires first in
            # practice, so this list is never consumed.
            turns: list = []
            warnings.append(
                "auto-storyboard: skipped normalize_shots because the "
                "storyboard-prompt module is unavailable in this runtime."
            )
        else:
            reference_digest = _manifest_digest(manifest) if manifest else ""
            # ---- Dialogue-driven auto storyboard ---------------------- #
            # 1) Extract dialogue turns. Canonical ``speaker：line``
            #    concepts (what the enhancer emits and the skill
            #    teaches) parse deterministically — no LLM call, exact
            #    spans by construction. Free-form prose falls back to
            #    the LLM span-anchored extractor. On extraction failure
            #    (LLM paraphrase, bad JSON, malformed spans) we fall
            #    back to a single narrator turn carrying the whole
            #    concept as one beat.
            turns, _structured_prologue = (
                _dlg_parse_structured_dialogue_turns(idea)
            )
            if turns:
                self._dialogue_extract_warnings = []
                log_pipeline(
                    "structured dialogue parsed deterministically "
                    f"({len(turns)} turn(s)) — LLM extractor skipped"
                )
            else:
                turns = self.extract_dialogue(idea)
                if self._dialogue_extract_warnings:
                    warnings.extend(self._dialogue_extract_warnings)
            pacing_obj = _DLG_PACING_PRESETS[pacing_key]
            if not turns:
                # Genuine {"turns": []} is narration. Exhausted extract
                # retries also return [] but already pushed a warning.
                turns = [_dlg_narrator_fallback_turn(idea)]
            is_narrator_fallback = (
                len(turns) == 1 and turns[0].speaker == "(narrator)"
            )
            # Dialogue board + category "none": the spoken-scene
            # cinematography contract (shot/reverse-shot, no subtitles)
            # is what the user expects once lines are extracted — the
            # widget keeps its stored value, only the effective
            # category used downstream upgrades.
            if (
                not is_narrator_fallback
                and (category or "").split(" - ", 1)[0].strip().lower()
                in ("", "none")
            ):
                category = next(
                    c for c in LOOP_CATEGORIES if c.startswith("dialogue")
                )
                upgrade_note = (
                    "category: none -> dialogue (spoken lines extracted; "
                    "spoken-scene cinematography contract auto-enabled)"
                )
                warnings.append(upgrade_note)
                log_pipeline(upgrade_note)
            budgets, pacing_report = _dlg_estimate_shot_budget(turns, pacing_obj)
            if is_narrator_fallback:
                # Pacing math is TTS-speech math — for a narration board
                # (the whole prose paragraph as one fallback "line") it
                # produces a meaningless number that must NOT be logged
                # like a duration estimate; users read it as the final
                # board length. The budget below is what actually rules.
                log_pipeline(
                    "pacing: narration board (no dialogue lines) — pacing "
                    "estimate not used; the total_duration_seconds budget "
                    "rules"
                )
            else:
                log_pipeline(_dlg_pacing_report_text(pacing_report))
            # 2) Resolve total_duration: 0 -> pacing estimate; >0 -> user override.
            #    Track which case ran so step 4 knows whether to scale.
            #    Narrator fallback (no dialogue lines): pacing math is
            #    meaningless — fall back to the legacy DEFAULT_TOTAL.
            if is_narrator_fallback and effective_total_duration <= 0:
                effective_total_duration = DEFAULT_TOTAL_DURATION_SECONDS
                log_pipeline(
                    f"auto-length (narrator fallback): no dialogue lines "
                    f"found; defaulting total_duration_seconds="
                    f"{effective_total_duration}s"
                )
                auto_total = effective_total_duration
            elif effective_total_duration <= 0:
                auto_total = int(round(pacing_report.total_sec))
                effective_total_duration = auto_total
                log_pipeline(
                    f"auto-length: derived total_duration_seconds={auto_total} "
                    f"from {len(turns)} turn(s) @ pacing={pacing_key}"
                )
            else:
                auto_total = 0  # signal: user gave an explicit total
            # 3) Resolve the scene count:
            #      - scene_count=N given (dialogue): honoured EXACTLY —
            #        distribute the SPEAKING evenly across N scenes by
            #        estimated speech time (台词平均分配); above the line
            #        count the extra scenes become silent reaction cuts
            #        (切换镜头), never a touched spoken line.
            #      - scene_count=0 (dialogue): MINIMAL-CUT packing —
            #        greedily pack consecutive turns into the fewest
            #        scenes whose speech+pause math fits the 14s H3
            #        single-generation window. Scene cuts are the chain's
            #        most expensive operation (each cut regenerates a
            #        carried overlap and risks identity/ambience drift),
            #        so auto never multiplies cuts; pacing owns the SPEECH
            #        tempo (rates + pauses that size each scene), NOT the
            #        cut count.
            #      - Narration boards (no dialogue): the LLM storyboard
            #        honours the count; its split-bias directive is
            #        derived from pacing (fast→aggressive etc.) — cut
            #        density stays a narration-only control.
            storyboard_bias = _PACING_TO_STORYBOARD_BIAS.get(
                pacing_key, "balanced"
            )
            pending_reaction_cuts = 0
            user_scene_count = int(scene_count or 0)
            even_time_split = False
            if is_narrator_fallback:
                # Narration: the LLM storyboard honours the count (or
                # decides it when scene_count=0). Leave as-is.
                turn_scenes = []
            else:
                if user_scene_count > 0:
                    # Explicit count: per-line granularity, then spread
                    # the SPEECH evenly across exactly N scenes.
                    fine_budgets, _fine_report = _dlg_estimate_shot_budget(
                        turns, pacing_obj, max_shot_seconds=0.0
                    )
                    total_lines = len(fine_budgets)
                    if user_scene_count > total_lines:
                        # Ceiling: one line per scene; the rest become
                        # silent reaction cuts after the board is built.
                        turn_scenes = [[b] for b in fine_budgets]
                        pending_reaction_cuts = (
                            user_scene_count - total_lines
                        )
                        log_pipeline(
                            f"scene_count={user_scene_count}: one line per "
                            f"scene ({total_lines}) + "
                            f"{pending_reaction_cuts} reaction cut(s)"
                        )
                    else:
                        turn_scenes = _dlg_distribute_lines_to_scenes(
                            fine_budgets, scene_count=user_scene_count
                        )
                        log_pipeline(
                            f"scene_count={user_scene_count}: distributed "
                            f"{total_lines} line(s) evenly by speech time"
                        )
                    # Whether the speech actually fits the budget is a
                    # WARNING, not a constraint (能否说完不重要).
                    speech_total = sum(
                        b.estimated_speech_sec for b in fine_budgets
                    )
                    if speech_total > effective_total_duration:
                        warnings.append(
                            f"dialogue speech ≈{speech_total:.0f}s exceeds "
                            f"the {effective_total_duration}s time budget; "
                            "lines were distributed evenly anyway — some "
                            "lines may not finish inside their clip"
                        )
                    scene_count = user_scene_count
                    even_time_split = True
                else:
                    # Auto: minimal-cut packing to the 14s H3 window.
                    # Scene durations come from the same speech+pause
                    # math, so the lines fit by construction — no
                    # fits-warning needed here.
                    turn_scenes = _dlg_group_budgets_into_scenes(
                        budgets, pacing_obj, bias="balanced"
                    )
                    natural = len(turn_scenes)
                    log_pipeline(
                        f"scene_count auto (minimal-cut packing): "
                        f"{len(budgets)} turn budget(s) -> {natural} "
                        f"scene(s) within the 14s H3 single-generation "
                        f"window (pacing={pacing_key} owns speech tempo, "
                        f"not cut count)"
                    )
                    scene_count = natural
            # 4) Duration model:
            #      - Explicit count (dialogue): after the board is built
            #        every scene gets an EQUAL share of the budget
            #        (T / scene_count); Stage 0 then grid-rounds and
            #        clamps to the H3 4..14s window with its own warnings.
            #      - Auto packing (dialogue): per-scene durations come
            #        from the packing math itself (speech + pauses +
            #        one head/tail pad per scene); an explicit user total
            #        rescales the packed scenes proportionally inside
            #        _build_shots_from_scenes.
            use_packed = not is_narrator_fallback
            # Note: there is no "rebalance shots down" helper anymore.
            # When a turn splits into multiple budgets (overbudget per-line
            # packing), we accept more shots than turns rather than merge
            # — merging would either cut a line in half (forbidden by the
            # 1-line -> 1-<d>-block invariant) or merge across speakers
            # (also forbidden). The scene count (given or pacing-derived)
            # is honoured via even speech distribution + reaction cuts,
            # never by clipping a line.
            # 5) Hand off: auto-storyboard now gets the dialogue-aware plan.
            #    For the packed-auto path, an EXPLICIT user total rescales
            #    the packed scenes proportionally (auto_total==0 means the
            #    user set total_duration_seconds); an auto total means the
            #    packing math already IS the budget — no rescale.
            packed_scale_total = (
                float(effective_total_duration)
                if (use_packed and not even_time_split and auto_total == 0)
                else None
            )
            shots, sb_warnings = self._auto_storyboard(
                idea,
                int(scene_count or 0),
                effective_total_duration,
                category,
                (output_language or "en").strip().lower(),
                seed=seed,
                reference_digest=reference_digest,
                storyboard_bias=storyboard_bias,
                dialogue_turns=turns,
                shot_budgets=budgets,
                turn_scenes=turn_scenes if use_packed else None,
                pacing_obj=pacing_obj if use_packed else None,
                scene_target_total_sec=packed_scale_total,
            )
            warnings.extend(f"auto-storyboard: {w}" for w in sb_warnings)
            # 5b) Even time split: every scene gets an equal share of
            #     the budget (T / scene_count). Whether a scene's speech
            #     fits its share was already warned about above; the
            #     split itself is unconditional — TIME is the only
            #     must.
            if even_time_split and shots:
                share = float(effective_total_duration) / max(
                    1, len(shots)
                )
                for shot in shots:
                    shot["duration_seconds"] = share
                log_pipeline(
                    f"time budget split evenly: {effective_total_duration}s "
                    f"/ {len(shots)} scene(s) = {share:.2f}s each (Stage 0 "
                    "clamps to the 4-14s H3 window)"
                )
            # 6) Reaction cuts: when the user's scene_count exceeded the
            #    per-line ceiling, top the board up with mechanically
            #    inserted silent reaction shots (切换镜头) at
            #    speaker-change boundaries — multi-cam dialogue rhythm.
            #    A reaction shot carries NO dialogue lines (the per-shot
            #    prompt generation then writes a silent shot), never
            #    touches the spoken text, and never joins the speaker-ID
            #    map (no _line_speakers / _turn_speaker keys).
            if pending_reaction_cuts > 0 and shots:
                shots = _insert_reaction_cuts(
                    shots, pending_reaction_cuts, warnings
                )

        # ---- Stage 0: per-shot duration guardrail (H3 4..14s) --------- #
        MIN_PER_SHOT_SECONDS = 4
        MAX_PER_SHOT_SECONDS = 14
        seed_base = derive_seed_base(seed)
        per_shot_fallback = max(
            MIN_PER_SHOT_SECONDS,
            min(
                MAX_PER_SHOT_SECONDS,
                int(round(effective_total_duration / max(1, len(shots)))),
            ),
        )
        over_length_shots: list[str] = []
        under_length_shots: list[str] = []
        entries: list[dict] = []
        for i, shot in enumerate(shots, start=1):
            dur = shot.get("duration_seconds") or per_shot_fallback
            if dur < MIN_PER_SHOT_SECONDS:
                under_length_shots.append(
                    f"{shot.get('id', f'shot_{i}')} ({dur:.1f}s -> "
                    f"{MIN_PER_SHOT_SECONDS}s)"
                )
                dur = MIN_PER_SHOT_SECONDS
            if dur > MAX_PER_SHOT_SECONDS:
                over_length_shots.append(
                    f"{shot.get('id', f'shot_{i}')} ({dur:.1f}s -> "
                    f"{MAX_PER_SHOT_SECONDS}s)"
                )
                dur = MAX_PER_SHOT_SECONDS
            length = (
                int(shot["length"])
                if "length" in shot
                else seconds_to_length(dur)
            )
            shot_seed = derive_seed(seed_base, i, unified=seed_unified)
            entries.append(
                {
                    "id": shot["id"],
                    "source": shot,
                    "length": length,
                    "seed": shot_seed,
                }
            )
        if under_length_shots:
            warnings.append(
                f"per-shot floor: each Scene should be at least "
                f"{MIN_PER_SHOT_SECONDS}s; raised "
                f"{', '.join(under_length_shots)}."
            )
        if over_length_shots:
            warnings.append(
                f"per-shot cap: MiniMax-H3 single-generation ceiling is "
                f"{MAX_PER_SHOT_SECONDS}s; clamped "
                f"{', '.join(over_length_shots)}. The original beats "
                f"are shorter than intended; raise scene_count to keep "
                f"the climax."
            )
            warnings.append(
                f"soft preflight: MiniMax-H3 single-generation ceiling is "
                f"4-15s; the upstream H3 API may reject or degrade entries "
                f"longer than 15s. After the per-shot cap above, every entry "
                f"fits within {MAX_PER_SHOT_SECONDS}s."
            )
        total_frames = sum(e["length"] for e in entries)
        total_seconds = total_frames / 24.0
        # Drift is only meaningful when the user actually set a budget
        # (effective_total_duration > 0). With the new 0 = auto default,
        # pre-built shots_text / unscaled storyboards would otherwise
        # always emit "drifts 520% from the 0s budget" garbage.
        if effective_total_duration > 0:
            drift = abs(total_seconds - effective_total_duration) / max(
                1, effective_total_duration
            )
            if drift > 0.2:
                warnings.append(
                    f"board duration {total_seconds:.1f}s drifts {drift:.0%} from the "
                    f"{effective_total_duration}s budget (each clip rounds up onto "
                    "the 17k+5 grid)"
                )
        if int(scene_count or 0) > 0 and int(scene_count or 0) * MIN_PER_SHOT_SECONDS > int(effective_total_duration):
            warnings.append(
                f"requested scene_count={int(scene_count)} with total_duration_seconds="
                f"{int(effective_total_duration)}s forces very short beats; with "
                f"{MIN_PER_SHOT_SECONDS}-{MAX_PER_SHOT_SECONDS}s per Scene, this "
                f"budget is better suited to <= {max(1, int(effective_total_duration) // MIN_PER_SHOT_SECONDS)} scenes."
            )
        log_pipeline(
            f"stage0 done: {len(entries)} clips, "
            f"total={total_frames}f ({total_seconds:.1f}s vs budget "
            f"{effective_total_duration}s), "
            f"mode={parse_generation_mode(generation_mode)}"
        )
        if not entries:
            raise RuntimeError("no usable shots produced from shots_text / storyboard")

        # ---- Stage 1: prefix + CAST ----------------------------------- #
        # The pacing preset's binding tempo directive rides into the
        # prefix note AND every per-shot / single-call prompt below, so
        # the WRITING tempo matches the board's pacing end to end.
        tempo_directive = build_tempo_directive(pacing_key)
        # Short dialogue boards without reference images skip the
        # stage-1 LLM call: the prefix is cut locally from the concept's
        # own scene-setting text (the "simple task, faster result" path
        # — fewer calls, contracts not relaxed: identity/voice binding
        # rides the CAST lines the dialogue blocks carry).
        dialogue_board = bool(turns) and not (
            len(turns) == 1 and turns[0].speaker == "(narrator)"
        )
        if (
            dialogue_board
            and not manifest
            and len(turns) <= _LOCAL_PREFIX_MAX_TURNS
            and sum(t.line_count for t in turns) <= _LOCAL_PREFIX_MAX_LINES
        ):
            prefix_lines, cast = self._derive_local_prefix(
                idea, turns, tempo_directive
            )
            local_prefix_note = (
                f"prefix derived locally (short dialogue board: "
                f"{len(turns)} turn(s), no reference images) — stage-1 "
                "LLM call skipped"
            )
            warnings.append(local_prefix_note)
            log_pipeline(local_prefix_note)
        else:
            prefix_lines, cast = self._synth_prefix(
                idea,
                category,
                    output_language,
                    shots,
                    seed=seed,
                    manifest=manifest,
                    tempo_directive=tempo_directive,
                )
        prefix_text = "\n".join(prefix_lines)

        # ---- Stage 1.5: extract stable spatial layout (deterministic) -- #
        # Pulls each subject's on-screen position out of the rewritten
        # user_input via regex; the result is injected into both the
        # prefix (via the synth system prompt's hard rule) AND every
        # per-shot user template via build_spatial_layout_directive. No
        # extra LLM call — this is a rule-based pass over the rewritten
        # concept the enhancer already produced.
        spatial_layout = extract_spatial_layout(idea)
        if spatial_layout:
            log_pipeline(
                "spatial layout (extracted from user_input): "
                + ", ".join(f"{n}={p}" for n, p in spatial_layout.items())
            )

        # ---- Stage 2: per-clip or single_call ------------------------- #
        # Deterministic speaker-ID map (official H3 rule: a speaker keeps
        # the same (S<n>) across shots). Derived from the storyboard turn
        # order so every per-shot LLM call sees the identical assignment.
        speaker_id_map = build_speaker_id_map(shots)
        if speaker_id_map:
            log_pipeline(
                "speaker ID map: "
                + ", ".join(f"{n}=({s})" for n, s in speaker_id_map.items())
            )
        if gen_code == "single_call" and ref_code == "t2va":
            all_shot_prompts = self._generate_all_shots_single_call(
                concept=idea,
                prefix_text=prefix_text,
                category=category,
                shots=shots,
                duration_seconds=effective_total_duration,
                language_name=output_language,
                seed=seed,
                cast_sheet=build_cast_sheet_text(cast),
                # Reuse the stage-0.5 extraction — a second LLM call
                # here could disagree with the board that was actually
                # built (and doubles the cost for no benefit).
                dialogue_turns=turns,
                speaker_id_map=speaker_id_map,
                spatial_layout=spatial_layout,
                tempo_directive=tempo_directive,
                speaker_identities=cast,
            )
            for entry in entries:
                entry["prompt"] = all_shot_prompts.get(entry["id"], [])
        else:
            # per_shot: one LLM call per entry. Baseline shape:
            # _generate_shot_prompt accepts prev_lines / prev_id /
            # prev_subject_definitions and builds the continuation
            # block + reference_directive internally.
            seen_speakers: set = set()
            for clip_index, entry in enumerate(entries, start=1):
                _check_interrupt(f"shot[{clip_index}/{len(entries)}]")
                prev_entry = entries[clip_index - 2] if clip_index > 1 else None
                prev_subject_definitions = None
                if prev_entry and schema == SCHEMA_SIX:
                    prev_prompt = prev_entry.get("prompt", []) or []
                    if isinstance(prev_prompt, list):
                        try:
                            i0 = prev_prompt.index("subject_definitions:")
                            i1 = prev_prompt.index("summary:")
                            prev_subject_definitions = prev_prompt[i0 + 1 : i1]
                        except ValueError:
                            prev_subject_definitions = None
                shot_prompt = self._generate_shot_prompt(
                    concept=idea,
                    prefix_text=prefix_text,
                    category=category,
                    shot=entry["source"],
                    clip_index=clip_index,
                    clip_count=len(entries),
                    prev_lines=(prev_entry.get("prompt", []) if prev_entry else None),
                    prev_id=(prev_entry["id"] if prev_entry else ""),
                    prev_subject_definitions=prev_subject_definitions,
                    duration_seconds=length_to_seconds(entry["length"]),
                    language_name=output_language,
                    seed=entry["seed"],
                    mode=ref_code,
                    manifest=manifest,
                    dialogue_lines=entry["source"].get("_dialogue_lines"),
                    turn_index=entry["source"].get("_turn_index"),
                    turn_speaker=entry["source"].get("_turn_speaker"),
                    speaker_id_map=speaker_id_map,
                    line_speakers=entry["source"].get("_line_speakers"),
                    tempo_directive=tempo_directive,
                    speaker_identities=cast,
                    first_appearance_speakers={
                        s for s in set(
                            entry["source"].get("_line_speakers")
                            or [entry["source"].get("_turn_speaker", "")]
                        )
                        if s and s not in seen_speakers
                    },
                    spatial_layout=spatial_layout,
                )
                entry["prompt"] = shot_prompt
                seen_speakers.update(
                    s for s in (
                        entry["source"].get("_line_speakers")
                        or [entry["source"].get("_turn_speaker", "")]
                    ) if s
                )

        # ---- Stage 2.5: spatial-layout drift preflight ---------------- #
        # Re-extract each generated shot's declared positions from its
        # emitted prompt text and compare — against the board layout
        # declared in user_input AND against every other shot — using
        # the same canonical buckets. Warning-level only: prose written
        # by the per-shot LLM is advisory, the directive + prefix are
        # the binding side, and a paraphrase here must not fail the run.
        if spatial_layout:
            scene_layouts = [{"spatial_layout": spatial_layout}]
            for entry in entries:
                prompt_text = "\n".join(entry.get("prompt") or [])
                scene_layouts.append(
                    {"spatial_layout": extract_spatial_layout(prompt_text)}
                )
            drift_errors = validate_spatial_layout_invariant(
                spatial_layout, scene_layouts
            )
            for err in drift_errors:
                warnings.append(f"spatial layout drift: {err}")

        # ---- Stage 3: assemble plan_json ------------------------------ #
        # Strict upstream contract: top-level keys are exactly
        # ``shots`` + ``prompt_prefix``; each shot carries exactly
        # ``id`` / ``prompt`` / ``length`` / ``seed``. Sampling
        # parameters (steps / cfg / sampler / canvas / context_length
        # / continuation_mode) live on the Production Plan widget and
        # never enter this JSON — the Production Plan node reads its
        # own widget values and the prompt array only.
        #
        # Note on ref2va: the LLM's ``subject_definitions`` body lands
        # inside ``shot["prompt"][0]`` (the section header line is
        # already ``subject_definitions:``) — the body itself follows
        # on subsequent prompt-array lines per the six-section schema.
        # We do NOT duplicate it as a top-level ``subject_definitions``
        # key on the shot; that would be data redundancy the upstream
        # compiler doesn't expect.
        shots_for_plan: list[dict] = [
            {
                "id": entry["id"],
                "prompt": entry.get("prompt", []),
                "length": entry["length"],
                "seed": str(entry["seed"]),
            }
            for entry in entries
        ]
        plan: dict[str, Any] = {
            "shots": shots_for_plan,
            "prompt_prefix": list(prefix_lines),
        }
        plan_errors = validate_plan(plan, schema=schema)
        if plan_errors:
            raise RuntimeError("plan validation failed: " + "; ".join(plan_errors))
        label_errors = validate_label_policy(
            plan,
            ref_code,
            manifest or [],
        )
        if label_errors:
            raise RuntimeError(
                "label policy failed: " + "; ".join(label_errors)
            )
        plan_json = plan_to_json_string(plan)
        # ---- Stage 4: summary (the node's only other output) -------- #
        # Human-readable run report: LLM request count + estimated token
        # usage + every warning. The old preflight / preview / prompts
        # outputs are gone — plan_json carries the machine-readable plan
        # and summary carries everything a human needs at a glance.
        summary_lines = ["MiniMax H3 Loop Plan summary"]
        # Board-kind marker (also the node's third output pin): a
        # dialogue run and a narrator-fallback run look identical in
        # the UI otherwise — this line says which promise applies.
        if dialogue_board:
            summary_lines.append(
                "Board: dialogue — "
                f"{len(turns)} turn(s), "
                f"{sum(t.line_count for t in turns)} spoken line(s) "
                "locked verbatim"
            )
        else:
            fallback_reason = (
                "extraction failed — spoken lines NOT preserved"
                if self._dialogue_extract_warnings
                else "no spoken lines detected"
            )
            summary_lines.append(f"Board: narration ({fallback_reason})")
        if enhance_header is not None:
            # Auto-enhance ran: surface the rewrite after the fact so
            # the intermediate stays inspectable without the two-node
            # checkpoint. The full advice header + rewritten text are
            # appended at the bottom of this summary.
            summary_lines.append(
                "Auto-enhance: user_input was rewritten before planning "
                "(see the rewrite appended below)"
            )
        summary_lines.extend(self._usage_summary_lines())
        total_frames = sum(int(s.get("length") or 0) for s in plan["shots"])
        summary_lines.append(
            f"Plan: {len(plan['shots'])} scene(s), {total_frames} frames "
            f"({total_frames / 24.0:.1f}s delivered vs "
            f"{effective_total_duration}s budget), mode={gen_code}, "
            f"reference={ref_code}"
        )
        if warnings:
            summary_lines.append(f"Warnings ({len(warnings)}):")
            summary_lines.extend(f"  - {w}" for w in warnings)
        else:
            summary_lines.append("Warnings: none")
        if enhance_header is not None:
            summary_lines.append("")
            summary_lines.append("Auto-enhance rewrite (what the pipeline consumed):")
            summary_lines.append(f"{enhance_header.strip()}")
            summary_lines.append("--- rewritten user_input ---")
            summary_lines.append(idea.strip())
            summary_lines.append("--- end rewritten user_input ---")
        summary = "\n".join(summary_lines)
        # Mirror the summary into mie_log (the ComfyUI console) so a
        # run's request/token/warning footprint is visible without a
        # Show-Anything node. mie_log only prints — it never writes
        # files; logs/*.log entries come solely from the Show node
        # writing whatever is wired into it.
        log_pipeline(summary)
        return {
            "plan_json": plan_json,
            "summary": summary,
            "board_kind": "dialogue" if dialogue_board else "narration",
        }



class MiniMaxH3LoopPromptGenerator:
    """ComfyUI node: free-form user_input -> Production Plan ``plan_json``.

    Paste a concept paragraph or one beat per line into ``user_input``;
    the node splits it into ``scene_count`` scenes whose durations sum
    close to ``total_duration_seconds``, then feeds ``plan_json`` into
    ``MiniMaxH3ChainPlanModern.plan_json_input``. The generated JSON only
    carries ``shots`` / ``prompt_prefix`` — sampler steps, canvas size,
    continuation mode, run name etc. stay on the Plan node's widgets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "user_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Everything you want, in any shape: a concept "
                            "paragraph, speaker：line dialogue, or mixed. "
                            "Dialogue is extracted then packed "
                            "deterministically; narration still goes through "
                            "the storyboard LLM. Blank = default concept."
                        ),
                    },
                ),
                "enhance_user_input": (
                    list(_ENHANCE_USER_INPUT_LABELS),
                    {
                        "default": _ENHANCE_USER_INPUT_LABELS[0],
                        "tooltip": (
                            "on: run the MiniMax H3 Loop User Input "
                            "Enhancer's rewrite once inside this node (one "
                            "extra LLM call) before planning — paste any "
                            "rough draft. off (default): consume user_input "
                            "verbatim. Keep off when you already write the "
                            "canonical format or when the standalone "
                            "Enhancer node is wired upstream (it would "
                            "re-rewrite an already-canonical text). The "
                            "rewrite (Classification + Notes + full text) "
                            "is surfaced at the top of the preflight "
                            "report. NOTE: with on, ANY widget change "
                            "re-runs the rewrite too (single-node caching)."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Seed base for the per-scene seed chain "
                            "(scene N gets base+N as a string). 0 = derive "
                            "from the clock each run."
                        ),
                    },
                ),
            },
            "optional": {
                "scene_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "tooltip": (
                            "Number of scenes.\n\n"
                            "0 = auto. Dialogue boards pack consecutive "
                            "turns into the fewest scenes that fit the 14s "
                            "H3 window (a speaker change does NOT force a "
                            "cut). Pacing does not change this cut count — "
                            "it only changes speech tempo and the BRISK/"
                            "MEASURED prompt directive. Narration boards "
                            "let the storyboard LLM decide, using pacing as "
                            "split bias (fast=aggressive, slow=conservative)."
                            "\n\n"
                            ">0 on a dialogue board: speech is spread evenly "
                            "across that many scenes; above one-line-per-"
                            "scene the extras become silent reaction cuts "
                            "(cut to the listener). A line is never split "
                            "mid-utterance.\n\n"
                            ">0 on a narration board: the storyboard LLM "
                            "produces exactly that many scenes."
                        ),
                    },
                ),
                "total_duration_seconds": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1800,
                        "tooltip": (
                            "Whole-board duration budget. The storyboard LLM "
                            "distributes it across scenes; each scene rounds up "
                            "onto the 17k+5 frame grid, so the delivered total "
                            "lands close to this (±20% tolerated silently).\n\n"
                            "0 = auto: derived from dialogue line count and the "
                            "selected pacing preset (TTS rate + per-turn pause). "
                            "For concepts without speaker:line dialogue the "
                            "auto budget falls back to 15s. Set >0 to override "
                            "the auto estimate. The override actually shrinks "
                            "shots (clamped to 4-14s each); the old '15 default' "
                            "no longer applies.\n\n"
                            "MiniMax-H3 single-generation window: 4-15 s per "
                            "upstream MiniMax-AI/MiniMax-H3 README. This node "
                            "hard-caps every per-shot duration_seconds at 14 s "
                            "and emits a preflight warning when clamping occurred."
                        ),
                    },
                ),
                "pacing": (
                    list(_PACING_LABELS),
                    {
                        "default": _PACING_LABELS[1],
                        "tooltip": (
                            "SPEECH TEMPO (语速) of the whole board — "
                            "this is the faster/slower rhythm control, "
                            "NOT a cut-density control.\n\n"
                            "On a dialogue board with scene_count=0, "
                            "pacing does NOT change how many scenes you "
                            "get (auto packing is always fewest cuts "
                            "inside the 14s window). It does:\n"
                            "1. Prompt tempo: BRISK / natural / MEASURED "
                            "injected into the prefix and every shot.\n"
                            "2. Auto length (total=0): TTS rate + turn "
                            "pause (fast CN 4.5 chars/s + 1.0s, normal "
                            "3.5 + 1.5s, slow 2.8 + 2.0s).\n"
                            "Narration boards also map fast/normal/slow "
                            "to the storyboard LLM's aggressive/balanced/"
                            "conservative split bias. An explicit "
                            "scene_count is the cut-count override; "
                            "total_duration_seconds stays the duration "
                            "constraint."
                        ),
                    },
                ),
                "generation_mode": (
                    list(GENERATION_MODES),
                    {
                        "default": GENERATION_MODES[0],
                        "tooltip": (
                            "per_shot: one LLM call per scene, best continuity "
                            "(recommended). single_call: one LLM call for the "
                            "whole board (cheaper/faster).",
                        ),
                    },
                ),
                "category": (
                    list(LOOP_CATEGORIES),
                    {
                        "default": LOOP_CATEGORIES[0],
                        "tooltip": (
                            "none: no extra cinematography contract — "
                            "but once dialogue lines are extracted the "
                            "board is automatically treated as dialogue "
                            "(spoken-scene framing, shot/reverse-shot, "
                            "no on-screen subtitles). "
                            "dialogue: the same contract, always. "
                            "action: motion-blur / camera-shake advice. "
                            "The dialogue verbatim contract runs "
                            "whenever lines are extracted, on any "
                            "category."
                        ),
                    },
                ),
                "output_language": (
                    ["en", "zh"],
                    {
                        "default": "en",
                        "tooltip": (
                            "Narrative language for generated scene prose. "
                            "English default is recommended for H3 stability; "
                            "spoken lines still follow dialogue-tag policy."
                        ),
                    },
                ),
                "seed_mode": (
                    list(SEED_MODES),
                    {
                        "default": SEED_MODES[0],
                        "tooltip": (
                            "per_scene_increment (recommended): seed_base+index "
                            "per scene — deterministic but DISTINCT per scene. "
                            "This mirrors the upstream ComfyUI-MiniMaxH3-Context-"
                            "Loop plugin's own default, which derives per-scene "
                            "seeds from one base (sha256(base:index:shot_id)) "
                            "when the plan omits seeds and relies on that "
                            "determinism for checkpoint recovery; identical "
                            "seeds on every clip make consecutive clips sample "
                            "near-identical noise (repeated motion rhythm). "
                            "same_across_scenes: every scene shares one seed — "
                            "use it only when you deliberately want the "
                            "identical-noise look."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Advanced tuning: LLM creativity/randomness. "
                            "Most workflows should keep the default."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _DEFAULT_MAX_TOKENS,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                        "tooltip": (
                            "Advanced tuning: upper token budget for each LLM call."
                        ),
                    },
                ),
                "timeout": (
                    [60, 120, 300, 600],
                    {
                        "default": _DEFAULT_TIMEOUT,
                        "tooltip": (
                            "Advanced tuning: per-call timeout (seconds)."
                        ),
                    },
                ),
                "reference_mode": (
                    list(REFERENCE_MODES),
                    {
                        "default": REFERENCE_MODES[0],
                        "tooltip": (
                            "t2va: text-only chain (default). "
                            "i2va: scene 1 anchored to one <Picture 1>. "
                            "fl2va: scene 1 + alternating end targets "
                            "<Picture (N % 2) + 1>. "
                            "ref2va: N pictures active for every scene "
                            "(six-section prompt contract)."
                        ),
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "One IMAGE batch, all modes. Wiring (mirror the "
                            "upstream H3 workflow): batch[0] (Picture 1) is "
                            "the OPENING frame -> MiniMax H3 First-Scene "
                            "Image Gate 'image' (i2va/fl2va). For fl2va, "
                            "batch[1..] (Picture 2..N) are per-scene end "
                            "targets -> Chain Frame Index Switch "
                            "frame_1..frame_{N-1} (scene j ends on "
                            "frame_j, wraps after the last slot). For "
                            "ref2va, the whole batch -> Reference to Video "
                            "images (all active every scene). i2va keeps "
                            "only batch[0] and drops the rest with a "
                            "warning; t2va ignores images (warning). "
                            "Scene-transition intent (e.g. '从图一到图二再"
                            "回到图一' / 'A→B→A') goes in user_input. "
                            "1-9 frames. REQUIRED for i2va / fl2va / ref2va."
                        ),
                    },
                ),
                "caption_mode": (
                    list(CAPTION_MODES),
                    {
                        "default": CAPTION_MODES[0],
                        "tooltip": (
                            "Caption cache strategy for images. "
                            "cache_memory_disk (recommended): persistent cache. "
                            "cache_memory_only: RAM-only cache. "
                            "no_cache: always bypass cache. "
                            "force_recaption_once: force fresh captions now."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "plan_json",
        "summary",
        "board_kind",
    )
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        *,
        user_input: str = "",
        seed=None,
        scene_count=0,
        total_duration_seconds=0,
        generation_mode=GENERATION_MODES[0],
        category="",
        output_language="en",
        seed_mode="",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        reference_mode=REFERENCE_MODES[0],
        references_text: str = "",
        images=None,
        caption_mode="",
        force_recaption: bool = False,
        caption_cache_scope: str = "memory_disk",
        pacing: str = _PACING_LABELS[1],
        enhance_user_input: str = _ENHANCE_USER_INPUT_LABELS[0],
    ):
        enhancer = H3LoopPromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            user_input=user_input,
            scene_count=scene_count,
            total_duration_seconds=total_duration_seconds,
            generation_mode=generation_mode,
            category=category,
            output_language=output_language,
            seed=seed,
            seed_mode=seed_mode,
            reference_mode=reference_mode,
            references_text=references_text,
            images=images,
            caption_mode=caption_mode,
            force_recaption=force_recaption,
            caption_cache_scope=caption_cache_scope,
            pacing=pacing,
            enhance_user_input=parse_enhance_user_input(enhance_user_input),
        )
        return (
            out["plan_json"],
            out["summary"],
            out["board_kind"],
        )

    def is_changed(
        self,
        llm_service_connector,
        *,
        user_input: str = "",
        seed=None,
        scene_count=0,
        total_duration_seconds=0,
        generation_mode=GENERATION_MODES[0],
        category="",
        output_language="en",
        seed_mode="",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        reference_mode=REFERENCE_MODES[0],
        images=None,
        references_text: str = "",
        caption_mode="",
        force_recaption: bool = False,
        caption_cache_scope: str = "memory_disk",
        pacing: str = _PACING_LABELS[1],
        enhance_user_input: str = _ENHANCE_USER_INPUT_LABELS[0],
    ):
        h = hashlib.md5()
        for part in (
            user_input,
            str(seed),
            str(scene_count),
            parse_seed_mode(seed_mode),
            str(total_duration_seconds),
            generation_mode,
            category,
            output_language,
            str(temperature),
            str(max_tokens),
            str(timeout),
            parse_reference_mode(reference_mode),
            parse_caption_mode(caption_mode),
            str(bool(force_recaption)),
            str(caption_cache_scope or "memory_disk"),
            pacing,
            str(parse_enhance_user_input(enhance_user_input)),
            (references_text or "").strip(),
        ):
            h.update((part or "").encode("utf-8"))
        # images: hash tensor shape plus head/mid/tail samples so a
        # same-shape swap still invalidates the node.
        if images is None:
            h.update(b"none")
        else:
            try:
                shape = tuple(images.shape)
            except AttributeError:
                shape = ()
            h.update(repr(shape).encode("utf-8"))
            h.update(str(getattr(images, "dtype", "")).encode("utf-8"))
            sample_bytes = b""
            try:
                if hasattr(images, "detach") and hasattr(images, "cpu"):
                    flat = images.detach().cpu().reshape(-1)
                elif hasattr(images, "reshape"):
                    flat = images.reshape(-1)
                else:
                    flat = None
                if flat is not None:
                    n = int(getattr(flat, "shape", [0])[0] if hasattr(flat, "shape") else len(flat))
                    chunks = []
                    if n > 0:
                        head = flat[:1024]
                        mid_start = max(0, (n // 2) - 512)
                        mid = flat[mid_start:mid_start + 1024]
                        tail = flat[-1024:] if n > 1024 else flat
                        for part in (head, mid, tail):
                            if hasattr(part, "numpy"):
                                chunks.append(bytes(part.numpy().tobytes()))
                            elif hasattr(part, "tobytes"):
                                chunks.append(bytes(part.tobytes()))
                    sample_bytes = b"".join(chunks)
            except Exception:
                sample_bytes = b""
            h.update(hashlib.md5(sample_bytes).hexdigest().encode("ascii"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
