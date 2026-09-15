"""MiniMax H3 Loop plan-generator ComfyUI node.

Turns one free-form ``user_input`` (a concept paragraph OR one line per
beat) into a ``plan_json`` STRING that plugs straight into
ethanfel/ComfyUI-MiniMaxH3-Context-Loop's ``MiniMaxH3ChainPlanModern``
(Production Plan) ``plan_json_input`` socket — a non-empty upstream
string overrides the editor's stored plan.

Pipeline:
  * Stage 0.5 (LLM) — split ``user_input`` into a storyboard. The LLM
    decides the per-shot ``duration_seconds`` so the whole board lands
    close to the ``total_duration_seconds`` budget (each scene later
    rounds UP onto the H3 17k+5 grid, so "close" is exact enough).
    ``shot_count`` pins the number of scenes; 0 lets the LLM decide.
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
    (``shots``/``prompt_prefix`` only) and emit the
    preflight report + markdown preview.

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
        build_continuation_block,
        build_continuation_block_ref2v,
        build_plan,
        build_plan_preview,
        build_prefix_user_text,
        build_preflight_report,
        build_reference_directive,
        build_shot_user_text,
        build_shots_digest,
        build_single_call_user_text,
        build_cast_block,
        build_cast_sheet_text,
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
except ImportError:
    from .minimax_h3_loop_prompts import (
        DEFAULT_MUSIC_LINE,
        DEFAULT_TOTAL_DURATION_SECONDS,
        PREFIX_SYNTH_SYSTEM,
        REFERENCE_MODES,
        REFERENCE_MODE_CODES,
        SCHEMA_SIX,
        build_continuation_block,
        build_continuation_block_ref2v,
        build_plan,
        build_plan_preview,
        build_prefix_user_text,
        build_preflight_report,
        build_reference_directive,
        build_shot_user_text,
        build_shots_digest,
        build_single_call_user_text,
        build_cast_block,
        build_cast_sheet_text,
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


def _loop_category_advice(category: str) -> str:
    """Localised category-advice lookup for the loop widget.

    Splits the ``"<code> - <label>"`` display string the same way
    ``h3_prompts.parse_category`` does, then returns the loop-specific
    advice (or "" for ``none`` / unknown codes)."""
    code = (category or "").split(" - ", 1)[0].strip()
    return LOOP_CATEGORY_ADVICE.get(code, "")


# Structured output: 0.4 keeps the three-section contract stable (the
# h3 sibling uses the same value for stage-2 enhancement).
_DEFAULT_TEMPERATURE = 0.4
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

GENERATION_MODES = (
    "per_shot - 逐场生成(推荐)",
    "single_call - 单次调用(快/省)",
)
GENERATION_MODE_CODES = ("per_shot", "single_call")

SEED_MODES = (
    "same_across_scenes - 全场同seed(推荐)",
    "per_scene_increment - 每场seed递增",
)
SEED_MODE_CODES = ("same_across_scenes", "per_scene_increment")

SPLIT_BIASES = (
    "balanced - 平衡(推荐)",
    "conservative - 更少分场/更长镜头",
    "aggressive - 更细分场/更高节奏",
)
SPLIT_BIAS_CODES = ("balanced", "conservative", "aggressive")

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


def parse_split_bias(bias: str) -> str:
    code = (bias or "").split(" - ", 1)[0].strip()
    return code if code in SPLIT_BIAS_CODES else ""


def parse_caption_mode(mode: str) -> str:
    code = (mode or "").split(" - ", 1)[0].strip()
    return code if code in CAPTION_MODE_CODES else ""


def resolve_seed_unified(seed_mode: str, unified_seed_compat: bool) -> bool:
    code = parse_seed_mode(seed_mode)
    if code == "same_across_scenes":
        return True
    if code == "per_scene_increment":
        return False
    return bool(unified_seed_compat)


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


def _parse_shots_text_fallback(shots_text: str) -> list[Any]:
    raw = str(shots_text or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("shots"), list):
        data = data.get("shots")
    if isinstance(data, list):
        return data
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return [
        {"id": f"clip_{i:04d}", "description": line}
        for i, line in enumerate(lines, start=1)
    ]


def _apply_per_shot_overrides_local(plan: dict, overrides_text: str) -> dict:
    out = dict(plan)
    shots = [dict(s) for s in (plan.get("shots") or [])]
    out["shots"] = shots
    by_id = {str(s.get("id", "")): s for s in shots}
    for raw in str(overrides_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^([^:\s]+)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$", line)
        if not m:
            raise RuntimeError(f"invalid override line: {line!r}")
        shot_id, field, value = m.group(1), m.group(2), m.group(3).strip()
        shot = by_id.get(shot_id)
        if shot is None:
            raise RuntimeError(f"unknown shot id in overrides: {shot_id}")
        if field == "length":
            try:
                length = int(value)
            except ValueError as exc:
                raise RuntimeError(f"invalid length override for {shot_id}: {value!r}") from exc
            if length % 17 != 5:
                raise RuntimeError(
                    f"length override for {shot_id} is off the H3 grid (17k+5): {length}"
                )
            shot["length"] = length
        elif field == "seed":
            shot["seed"] = str(value)
        elif field == "steps":
            try:
                shot["steps"] = int(value)
            except ValueError as exc:
                raise RuntimeError(f"invalid steps override for {shot_id}: {value!r}") from exc
        else:
            raise RuntimeError(f"unsupported override field for {shot_id}: {field}")
    return out


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

    # ------------------------------------------------------------------ #
    # Stage 1: style-only prompt_prefix + CAST sheet
    # ------------------------------------------------------------------ #
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
                mode_note=_mode_note_for_prefix(mode),
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
        shot_count: int,
        total_duration_seconds: int,
        category: str,
        language: str,
        *,
        seed: Optional[int],
        reference_digest: str = "",
        split_bias: str = "balanced",
    ) -> tuple[list[dict], list[str]]:
        """Split user_input into a storyboard (one LLM call) whose
        per-shot durations sum close to the whole-board budget."""
        user_text = build_storyboard_user_text(
            (concept or "").strip(),
            int(shot_count or 0),
            AUTO_STORYBOARD_STYLE,
            category,
            language,
            total_duration_seconds=int(total_duration_seconds),
            reference_digest=reference_digest,
            split_bias=split_bias,
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
            shots, warnings = normalize_shots(raw_shots, int(shot_count))
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
    reference_directive: str = "",
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
    ) -> dict[str, list[str]]:
        user_text = build_single_call_user_text(
            concept=concept,
            prefix_text=prefix_text,
            category=category,
            shots=shots,
            duration_seconds=duration_seconds,
            language_name=language_name,
            cast_sheet=cast_sheet,
        )
        messages = self._messages(shot_system_prompt(), user_text)
        last_error: Optional[Exception] = None
        last_head = "<no reply>"
        for attempt in range(1 + _PARSE_RETRIES):
            _check_interrupt(f"single_call[attempt {attempt + 1}]")
            attempt_messages = messages
            if attempt > 0:
                attempt_messages = messages + [
                    {"role": "user", "content": PARSE_RETRY_CORRECTION}
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
            if result is not None:
                return result
            last_error = ValueError("single-call reply missing clips")
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
        concept: str = "",
        shots_text: str = "",
        *,
        # Baseline-compatible positional/kwarg surface.
        duration_seconds: int = 0,
        width: int = 544,
        height: int = 960,
        prompt_prefix_input: str = "",
        per_shot_overrides: str = "",
        unified_seed: bool = True,
        seed_mode: str = "",
        # Plan v4 widget surface.
        user_input: str = "",
        total_duration_seconds: int = DEFAULT_TOTAL_DURATION_SECONDS,
        shot_count: int = 0,
        split_bias: str = SPLIT_BIASES[0],
        generation_mode: str = "per_shot",
        category: str = "",
        output_language: str = "en",
        seed: Optional[int] = None,
        reference_mode: str = REFERENCE_MODES[0],
        references_text: str = "",
        images: Any = None,
        caption_mode: str = "",
        force_recaption: bool = False,
        caption_cache_scope: str = "memory_disk",
    ) -> dict:
        warnings: list[str] = []
        # Resolve the two-API concept input.
        raw_concept = concept or user_input or shots_text
        idea = (raw_concept or "").strip() or _default_concept()
        if not (raw_concept or "").strip():
            warnings.append(
                "concept/user_input empty: used the built-in default concept"
            )

        # Resolve the duration budget (single source of truth is
        # total_duration_seconds; duration_seconds is legacy alias).
        effective_total_duration = int(
            total_duration_seconds or DEFAULT_TOTAL_DURATION_SECONDS
        )
        if duration_seconds and int(duration_seconds) > 0:
            if int(total_duration_seconds or 0) != DEFAULT_TOTAL_DURATION_SECONDS:
                if int(duration_seconds) != int(total_duration_seconds):
                    warnings.append(
                        f"both duration_seconds ({duration_seconds}s) and "
                        f"total_duration_seconds ({total_duration_seconds}s) were set; "
                        f"using total_duration_seconds"
                    )
            else:
                effective_total_duration = int(duration_seconds)
                warnings.append(
                    f"legacy alias: duration_seconds={duration_seconds}s was used; "
                    "prefer total_duration_seconds."
                )

        # Pre-built shots_text (baseline) wins; empty triggers Stage 0.5.
        if shots_text and shots_text.strip():
            canonical_shots_text = shots_text
        else:
            canonical_shots_text = "[]"
        if canonical_shots_text == "[]" and not hasattr(self.llm, "invoke"):
            raise RuntimeError(
                "auto-storyboard requires an LLM connector with invoke(); "
                "provide storyboard shots_text explicitly or connect a full "
                "LLMServiceConnector"
            )

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
        split_bias_code = parse_split_bias(split_bias)
        seed_unified = resolve_seed_unified(seed_mode, bool(unified_seed))
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

        # Width/height validation (baseline).
        if int(width) % 32 or int(height) % 32:
            raise RuntimeError(
                f"width/height must be multiples of 32 (got {width}x{height})"
            )

        # Parse references_text (baseline path; plan v4 callers skip).
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

        # ---- Stage 0.5: auto storyboard OR parse pre-built shots ----- #
        storyboard_reply: str = canonical_shots_text
        if _ns_lazy is None:
            shots = []
            warnings.append(
                "auto-storyboard: skipped normalize_shots because the "
                "storyboard-prompt module is unavailable in this runtime."
            )
        elif storyboard_reply == "[]":
            reference_digest = _manifest_digest(manifest) if manifest else ""
            shots, sb_warnings = self._auto_storyboard(
                idea,
                int(shot_count or 0),
                effective_total_duration,
                category,
                (output_language or "en").strip().lower(),
                seed=seed,
                reference_digest=reference_digest,
                split_bias=split_bias_code,
            )
            warnings.extend(f"auto-storyboard: {w}" for w in sb_warnings)
        else:
            raw_shots_obj = _parse_shots_text_fallback(storyboard_reply)
            try:
                expected_count = int(shot_count) if int(shot_count or 0) > 0 else len(raw_shots_obj)
                normalized, ns_warnings = _ns_lazy(raw_shots_obj, expected_count)
            except TypeError:
                # Older normalize_shots that needs a different signature.
                normalized, ns_warnings = _ns_lazy(raw_shots_obj)
            warnings.extend(f"auto-storyboard: {w}" for w in ns_warnings)
            if int(shot_count or 0) > 0 and len(raw_shots_obj) > int(shot_count):
                warnings.append("trimmed incoming storyboard to requested shot_count")
            explicit_default = int(duration_seconds) if int(duration_seconds or 0) > 0 else 10
            for idx, shot in enumerate(normalized):
                if idx >= len(raw_shots_obj):
                    break
                raw_item = raw_shots_obj[idx] if isinstance(raw_shots_obj[idx], dict) else {}
                if "duration_seconds" not in raw_item:
                    shot["duration_seconds"] = explicit_default
            shots = normalized

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
                    "steps": 20,
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
                f"are shorter than intended; raise shot_count to keep "
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
        drift = abs(total_seconds - effective_total_duration) / max(
            1, effective_total_duration
        )
        if drift > 0.2:
            warnings.append(
                f"board duration {total_seconds:.1f}s drifts {drift:.0%} from the "
                f"{effective_total_duration}s budget (each clip rounds up onto "
                "the 17k+5 grid)"
            )
        if int(shot_count or 0) > 0 and int(shot_count or 0) * MIN_PER_SHOT_SECONDS > int(effective_total_duration):
            warnings.append(
                f"requested shot_count={int(shot_count)} with total_duration_seconds="
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
        if per_shot_overrides:
            probe_plan = {
                "shots": [
                    {
                        "id": e["id"],
                        "length": e["length"],
                        "seed": str(e["seed"]),
                        "steps": int(e.get("steps", 20)),
                    }
                    for e in entries
                ]
            }
            probe_plan = _apply_per_shot_overrides_local(probe_plan, per_shot_overrides)
            by_id = {str(s.get("id")): s for s in (probe_plan.get("shots") or [])}
            for e in entries:
                patched = by_id.get(e["id"], {})
                e["length"] = int(patched.get("length", e["length"]))
                e["seed"] = str(patched.get("seed", e["seed"]))
                e["steps"] = int(patched.get("steps", e.get("steps", 20)))

        # ---- Stage 1: prefix + CAST ----------------------------------- #
        if prompt_prefix_input:
            prefix_lines = split_prefix_paragraphs(prompt_prefix_input)
            cast: dict[str, str] = {}
            prefix_text = "\n".join(prefix_lines)
        else:
            prefix_lines, cast = self._synth_prefix(
                idea,
                category,
                output_language,
                shots,
                seed=seed,
                manifest=manifest,
            )
            prefix_text = "\n".join(prefix_lines)

        # ---- Stage 2: per-clip or single_call ------------------------- #
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
            )
            for entry in entries:
                entry["prompt"] = all_shot_prompts.get(entry["id"], [])
        else:
            # per_shot: one LLM call per entry. Baseline shape:
            # _generate_shot_prompt accepts prev_lines / prev_id /
            # prev_subject_definitions and builds the continuation
            # block + reference_directive internally.
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
                )
                entry["prompt"] = shot_prompt

        # ---- Stage 3: assemble plan_json ------------------------------ #
        shots_for_plan: list[dict] = []
        for entry in entries:
            shot_dict: dict[str, Any] = {
                "id": entry["id"],
                "prompt": entry.get("prompt", []),
                "length": entry["length"],
                "seed": str(entry["seed"]),
                "steps": int(entry.get("steps", 20)),
            }
            if entry["source"].get("subject_definitions"):
                shot_dict["subject_definitions"] = entry["source"]["subject_definitions"]
            shots_for_plan.append(shot_dict)
        plan = {
            "defaults": {"steps": 20},
            "shots": shots_for_plan,
            "prompt_prefix": list(prefix_lines),
        }
        # Validate using the strict upstream contract shape while keeping
        # node-level extension fields (defaults / steps) in emitted plan_json.
        plan_for_validate = {
            "shots": [
                {
                    "id": s["id"],
                    "prompt": s["prompt"],
                    "length": s["length"],
                    "seed": s["seed"],
                }
                for s in shots_for_plan
            ],
            "prompt_prefix": list(prefix_lines),
        }
        plan_errors = validate_plan(plan_for_validate, schema=schema)
        if plan_errors:
            raise RuntimeError("plan validation failed: " + "; ".join(plan_errors))
        label_errors = validate_label_policy(
            plan_for_validate,
            ref_code,
            manifest or [],
        )
        if label_errors:
            raise RuntimeError(
                "label policy failed: " + "; ".join(label_errors)
            )
        shots_json = plan_to_json_string(plan["shots"])
        plan_json = plan_to_json_string(plan)
        preflight = build_preflight_report(
            plan, warnings, mode=ref_code, manifest=manifest or []
        )
        plan_preview = build_plan_preview(plan)
        return {
            "plan_json": plan_json,
            "shots_json": shots_json,
            "shot_prompts": shots_json,
            "prompt_prefix_out": json.dumps(prefix_lines, ensure_ascii=False),
            "preflight_report": preflight,
            "plan_preview": plan_preview,
        }



class MiniMaxH3LoopPromptGenerator:
    """ComfyUI node: free-form user_input -> Production Plan ``plan_json``.

    Paste a concept paragraph or one beat per line into ``user_input``;
    the node splits it into ``shot_count`` scenes whose durations sum
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
                            "paragraph, one beat per line, or something in "
                            "between. The LLM always re-splits it into shots "
                            "against the total_duration_seconds budget. "
                            "Blank = default concept."
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
                "shot_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "tooltip": (
                            "Number of scenes. 0 = auto: the LLM decides scene "
                            "count from your material and total_duration_seconds "
                            "while keeping each scene in the 4-14s band and "
                            "avoiding meaningless oversplitting. >0 = exactly "
                            "that many (explicit count is always honoured)."
                        ),
                    },
                ),
                "total_duration_seconds": (
                    "INT",
                    {
                        "default": DEFAULT_TOTAL_DURATION_SECONDS,
                        "min": 5,
                        "max": 1800,
                        "tooltip": (
                            "Whole-board duration budget. The storyboard LLM "
                            "distributes it across scenes; each scene rounds up "
                            "onto the 17k+5 frame grid, so the delivered total "
                            "lands close to this (±20% tolerated silently).\n\n"
                            "MiniMax-H3 single-generation window: 4-15 s "
                            "(MiniMax-H3-Max 5-15 s) per upstream "
                            "MiniMax-AI/MiniMax-H3 README and "
                            "platform.minimaxi.com API. Context-Loop "
                            "accepts up to 149.667 s per Scene, but the H3 "
                            "model will reject or degrade entries longer "
                            "than 15 s. This node hard-caps every per-shot "
                            "duration_seconds at 14 s and emits a preflight "
                            "warning when clamping occurred."
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
                    {"default": LOOP_CATEGORIES[0]},
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
                            "same_across_scenes: every scene shares one seed "
                            "(better identity/style hold, recommended). "
                            "per_scene_increment: seed_base+index for each "
                            "scene (more variety)."
                        ),
                    },
                ),
                "split_bias": (
                    list(SPLIT_BIASES),
                    {
                        "default": SPLIT_BIASES[0],
                        "tooltip": (
                            "How aggressively auto-splitting cuts scenes when "
                            "shot_count=0. conservative: fewer/longer scenes; "
                            "balanced: default; aggressive: more/shorter scenes "
                            "(all still constrained to 4-14s per scene)."
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "plan_json",
        "shot_prompts",
        "prompt_prefix_out",
        "preflight_report",
        "plan_preview",
    )
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        concept: str = "",
        shots_text: str = "",
        *,
        user_input: str = "",
        seed=None,
        shot_count=0,
        duration_seconds: int = 0,
        total_duration_seconds=DEFAULT_TOTAL_DURATION_SECONDS,
        split_bias=SPLIT_BIASES[0],
        generation_mode=GENERATION_MODES[0],
        category="",
        width: int = 544,
        height: int = 960,
        output_language="en",
        prompt_prefix_input: str = "",
        per_shot_overrides: str = "",
        unified_seed=True,
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
    ):
        enhancer = H3LoopPromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            concept=concept,
            shots_text=shots_text,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            prompt_prefix_input=prompt_prefix_input,
            per_shot_overrides=per_shot_overrides,
            shot_count=shot_count,
            unified_seed=unified_seed,
            seed_mode=seed_mode,
            user_input=user_input,
            total_duration_seconds=total_duration_seconds,
            split_bias=split_bias,
            generation_mode=generation_mode,
            category=category,
            output_language=output_language,
            seed=seed,
            reference_mode=reference_mode,
            references_text=references_text,
            images=images,
            caption_mode=caption_mode,
            force_recaption=force_recaption,
            caption_cache_scope=caption_cache_scope,
        )
        return (
            out["plan_json"],
            out["shot_prompts"],
            out["prompt_prefix_out"],
            out["preflight_report"],
            out["plan_preview"],
        )

    def is_changed(
        self,
        llm_service_connector,
        concept: str = "",
        shots_text: str = "",
        *,
        user_input: str = "",
        seed=None,
        shot_count=0,
        duration_seconds: int = 0,
        total_duration_seconds=DEFAULT_TOTAL_DURATION_SECONDS,
        split_bias=SPLIT_BIASES[0],
        generation_mode=GENERATION_MODES[0],
        category="",
        width: int = 544,
        height: int = 960,
        output_language="en",
        prompt_prefix_input: str = "",
        per_shot_overrides: str = "",
        unified_seed=True,
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
    ):
        h = hashlib.md5()
        # Use both concept and user_input so we hash whatever the
        # ComfyUI widget passes plus the historical alias.
        effective_concept = concept or user_input
        for part in (
            effective_concept,
            shots_text,
            str(seed),
            str(shot_count),
            str(duration_seconds),
            str(bool(unified_seed)),
            parse_seed_mode(seed_mode),
            str(total_duration_seconds),
            parse_split_bias(split_bias),
            generation_mode,
            category,
            str(width),
            str(height),
            output_language,
            prompt_prefix_input,
            per_shot_overrides,
            str(temperature),
            str(max_tokens),
            str(timeout),
            parse_reference_mode(reference_mode),
            parse_caption_mode(caption_mode),
            str(bool(force_recaption)),
            str(caption_cache_scope or "memory_disk"),
        ):
            h.update((part or "").encode("utf-8"))
        # images: hash tensor shape (content is data, not signal; the
        # LLM captions the pixel bytes downstream, not the hash).
        if images is None:
            h.update(b"none")
        else:
            try:
                shape = tuple(images.shape)
            except AttributeError:
                shape = ()
            h.update(repr(shape).encode("utf-8"))
            h.update(str(getattr(images, "dtype", "")).encode("utf-8"))
            # Include a small content fingerprint so same-shape image
            # updates still invalidate the node.
            sample_bytes = b""
            try:
                if hasattr(images, "detach") and hasattr(images, "cpu"):
                    flat = images.detach().cpu().reshape(-1)
                    sample = flat[:1024]
                    sample_bytes = bytes(sample.numpy().tobytes())
                elif hasattr(images, "reshape") and hasattr(images, "tobytes"):
                    flat = images.reshape(-1)
                    sample = flat[:1024]
                    sample_bytes = bytes(sample.tobytes())
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
