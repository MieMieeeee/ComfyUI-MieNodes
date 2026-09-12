"""MiniMax H3 Loop plan-generator ComfyUI node.

Turns a concept + a storyboard (e.g. ``MiniMaxH3StoryboardGenerator``
``shots_json``) into a ``plan_json`` STRING that plugs straight into
ethanfel/ComfyUI-MiniMaxH3-Context-Loop's ``MiniMaxH3ChainPlanModern``
(Production Plan) ``plan_json_input`` socket — a non-empty upstream
string overrides the editor's stored plan.

Pipeline:
  * Stage 0 (deterministic) — parse ``shots_text`` (storyboard JSON or
    one-line-per-shot), convert durations onto the H3 17k+5 frame grid,
    derive per-shot seeds, apply ``per_shot_overrides``.
  * Stage 1 (LLM, optional) — derive the shared ``prompt_prefix`` when
    the user did not supply one.
  * Stage 2 (LLM) — write each clip's prompt in the three-section H3
    form; clips 2+ get an explicit continuation directive referencing
    the previous clip's ending (unbroken motion chain, ambience bed
    carries across the boundary). ``per_shot`` mode = one LLM call per
    clip (best continuity); ``single_call`` = one call for the whole
    board (cheaper/faster).
  * Stage 3 (deterministic) — assemble + validate the strict plan shape
    (``defaults``/``shots``/``prompt_prefix`` only) and emit the
    preflight report + markdown preview.

Failures raise ``RuntimeError`` — a partial plan must never reach the
Production Plan node.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:
        from core.utils import mie_log

try:
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        DEFAULT_DURATION_SECONDS,
        DEFAULT_MUSIC_LINE,
        DEFAULT_STEPS,
        PREFIX_SYNTH_SYSTEM,
        REFERENCE_MODES,
        REFERENCE_MODE_CODES,
        SCHEMA_SIX,
        apply_overrides,
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
        derive_seed,
        derive_seed_base,
        description_body,
        ensure_keyframe_idiom,
        find_alias_tokens,
        find_native_labels,
        find_semantic_anchors,
        length_to_seconds,
        log_pipeline,
        manifest_normalize_json,
        parse_per_shot_overrides,
        parse_references_text,
        parse_shots_text,
        plan_to_json_string,
        SCHEMA_THREE,
        schema_for_mode,
        parse_reference_mode,
        seconds_to_length,
        shot_system_prompt,
        shot_system_prompt_ref2v,
        sound_body,
        split_prefix_paragraphs,
        split_six_sections,
        split_three_sections,
        validate_label_policy,
        validate_manifest,
        validate_plan,
        SIX_SECTION_FIELDS,
        _manifest_digest,
        _mode_note_for_prefix,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts import (
        SYSTEM_STORYBOARD_PROMPT,
        PARSE_RETRY_CORRECTION,
        build_user_text as build_storyboard_user_text,
        extract_json_array,
        normalize_shots,
    )
    from _mienodes_internal.nodes.llm.h3_prompts import CATEGORIES
except ImportError:
    from .minimax_h3_loop_prompts import (
        DEFAULT_DURATION_SECONDS,
        DEFAULT_MUSIC_LINE,
        DEFAULT_STEPS,
        PREFIX_SYNTH_SYSTEM,
        REFERENCE_MODES,
        REFERENCE_MODE_CODES,
        SCHEMA_SIX,
        apply_overrides,
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
        derive_seed,
        derive_seed_base,
        description_body,
        ensure_keyframe_idiom,
        find_alias_tokens,
        find_native_labels,
        find_semantic_anchors,
        length_to_seconds,
        log_pipeline,
        manifest_normalize_json,
        parse_per_shot_overrides,
        parse_references_text,
        parse_shots_text,
        plan_to_json_string,
        SCHEMA_THREE,
        schema_for_mode,
        parse_reference_mode,
        seconds_to_length,
        shot_system_prompt,
        shot_system_prompt_ref2v,
        sound_body,
        split_prefix_paragraphs,
        split_six_sections,
        split_three_sections,
        validate_label_policy,
        validate_manifest,
        validate_plan,
        SIX_SECTION_FIELDS,
        _manifest_digest,
        _mode_note_for_prefix,
    )
    from .minimax_h3_storyboard_prompts import (
        SYSTEM_STORYBOARD_PROMPT,
        PARSE_RETRY_CORRECTION,
        build_user_text as build_storyboard_user_text,
        extract_json_array,
        normalize_shots,
    )
    from .h3_prompts import CATEGORIES


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Structured output: 0.4 keeps the three-section contract stable (the
# h3 sibling uses the same value for stage-2 enhancement).
_DEFAULT_TEMPERATURE = 0.4
_DEFAULT_MAX_TOKENS = 8192  # reasoning-model budget, see LTX-2.5 sibling note
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768
_DEFAULT_TIMEOUT = 120

GENERATION_MODES = (
    "per_shot - 逐场生成(推荐)",
    "single_call - 单次调用(快/省)",
)
GENERATION_MODE_CODES = ("per_shot", "single_call")

# Fixed storyboard style for the inline auto-storyboard (standalone mode).
# "single_continuous" is the only board shape that matches this node's
# chain contract (unbroken motion, invisible_cut handoffs, carried
# ambience) — cut-grammar styles like parallel_montage would fight the
# stage-2 continuation rules.
AUTO_STORYBOARD_STYLE = "single_continuous"

_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_PARSE_RETRIES = 1


def parse_generation_mode(mode: str) -> str:
    code = (mode or "").split(" - ", 1)[0].strip()
    return code if code in GENERATION_MODE_CODES else GENERATION_MODE_CODES[0]


def postprocess_reply(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = _THINK_BLOCK_RE.sub("", text, count=1).strip()
    return text


def _default_concept() -> str:
    return (
        "A quiet, visually striking short film with one clear protagonist, "
        "one hero prop, and a simple emotional arc told in a continuous "
        "chain of moments."
    )


class H3LoopPromptEnhancer:
    """Loop-plan generator that talks to the project's
    LLMServiceConnector (shares the family logging + timeout pattern)."""

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

    def _invoke(
        self,
        messages: list[dict],
        *,
        temperature: float,
        seed: Optional[int],
        stage: str,
    ) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=temperature,
                max_tokens=self.max_tokens,
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
    # Stage 1: prompt_prefix
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
    ) -> list[str]:
        # The storyboard digest grounds the prefix in the actual clips —
        # with concept-only input the LLM tends to invent protagonists and
        # default art styles that contradict the shots.
        messages = self._messages(
            PREFIX_SYNTH_SYSTEM,
            build_prefix_user_text(
                concept,
                category,
                language_name,
                shots_digest=build_shots_digest(shots),
                mode_note=_mode_note_for_prefix(mode),
                manifest_digest=_manifest_digest(manifest or []),
            ),
        )
        from_llm = None
        last_error: Optional[Exception] = None
        for attempt in range(1 + _PARSE_RETRIES):
            raw = postprocess_reply(
                self._invoke(
                    messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"prefix[attempt {attempt + 1}]",
                )
            )
            paragraphs = split_prefix_paragraphs(raw)
            if paragraphs:
                from_llm = paragraphs
                break
            last_error = ValueError("empty prompt_prefix reply")
        if from_llm is None:
            raise RuntimeError(f"prompt_prefix synthesis failed: {last_error}")
        return from_llm

    # ------------------------------------------------------------------ #
    # Stage 0.5: auto-storyboard (shots_text empty + shot_count > 0)
    # ------------------------------------------------------------------ #
    def _auto_storyboard(
        self,
        concept: str,
        shot_count: int,
        category: str,
        language: str,
        *,
        seed: Optional[int],
    ) -> tuple[list[dict], list[str]]:
        """Run the storyboard pipeline inline (one LLM call) so the node
        works standalone: concept + shot_count -> plan_json."""
        user_text = build_storyboard_user_text(
            (concept or "").strip(),
            int(shot_count),
            AUTO_STORYBOARD_STYLE,
            category,
            language,
        )
        messages = self._messages(SYSTEM_STORYBOARD_PROMPT, user_text)
        last_error: Optional[Exception] = None
        last_head = "<no reply>"
        for attempt in range(1 + _PARSE_RETRIES):
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
                    stage=f"storyboard[attempt {attempt + 1}]",
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
        width: int,
        height: int,
        duration_seconds: int,
        language_name: str,
        seed: Optional[int],
        mode: str = "t2va",
        manifest: Optional[list[dict]] = None,
        prev_soundscape_ref2v: Optional[list[str]] = None,
    ) -> list[str]:
        code = parse_reference_mode(mode)
        schema = schema_for_mode(mode)
        # Reference directive (D6): first-sentence idiom / subject binding.
        reference_directive = build_reference_directive(
            mode, manifest or [], clip_index, duration_seconds
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
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            language_name=language_name,
            reference_directive=reference_directive,
            manifest_digest=_manifest_digest(manifest or []),
        )
        # System prompt dispatch: ref2va uses the six-section addendum.
        system = (
            shot_system_prompt_ref2v() if schema == SCHEMA_SIX else shot_system_prompt()
        )
        messages = self._messages(system, user_text)
        last_error: Optional[Exception] = None
        for attempt in range(1 + _PARSE_RETRIES):
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
                        "Your previous reply could not be parsed. Try again. "
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
                    return split_six_sections(raw)
                lines_out = split_three_sections(raw)
                if code in ("i2va", "fl2va"):
                    # Deterministic idiom enforcement: live E2E showed the
                    # LLM drops angle brackets / skips the end-target line.
                    lines_out = ensure_keyframe_idiom(
                        lines_out,
                        mode=code,
                        manifest=manifest or [],
                        clip_index=clip_index,
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
        width: int,
        height: int,
        duration_seconds: int,
        language_name: str,
        seed: Optional[int],
    ) -> dict[str, list[str]]:
        user_text = build_single_call_user_text(
            concept=concept,
            prefix_text=prefix_text,
            category=category,
            shots=shots,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            language_name=language_name,
        )
        messages = self._messages(shot_system_prompt(), user_text)
        last_error: Optional[Exception] = None
        last_head = "<no reply>"
        for attempt in range(1 + _PARSE_RETRIES):
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
        concept: str,
        shots_text: str,
        *,
        duration_seconds: int = DEFAULT_DURATION_SECONDS,
        generation_mode: str = "per_shot",
        category: str = "",
        width: int = 544,
        height: int = 960,
        output_language: str = "en",
        seed: Optional[int] = None,
        prompt_prefix_input: str = "",
        per_shot_overrides: str = "",
        shot_count: int = 0,
        unified_seed: bool = True,
        reference_mode: str = REFERENCE_MODES[0],
        references_text: str = "",
    ) -> dict:
        warnings: list[str] = []
        idea = (concept or "").strip() or _default_concept()
        ref_code = parse_reference_mode(reference_mode)
        schema = schema_for_mode(ref_code)

        # ---- Stage 0: deterministic preprocessing ---------------------- #
        if int(width) % 32 or int(height) % 32:
            raise RuntimeError(
                f"width/height must be multiples of 32 (got {width}x{height})"
            )

        # ---- Stage 0a: parse + validate references manifest ------------ #
        try:
            manifest = parse_references_text(references_text)
        except ValueError as exc:
            raise RuntimeError(f"references_text: {exc}") from exc
        manifest_errors = validate_manifest(manifest, ref_code)
        if manifest_errors:
            raise RuntimeError(
                "references_text failed reference_mode=" + ref_code + " validation: "
                + "; ".join(manifest_errors)
            )

        # single_call mode refuses keyframe / ref2va modes (D8).
        gen_code = parse_generation_mode(generation_mode)
        if gen_code == "single_call" and ref_code != "t2va":
            raise RuntimeError(
                f"single_call generation mode is not supported with reference_mode={ref_code}; "
                "use per_shot for i2va / fl2va / ref2va (the per-clip keyframe idiom "
                "and label policy must be applied per scene, not merged into one reply)."
            )

        # User-supplied prompt_prefix in non-t2va modes must NOT contain
        # native labels or @/# tokens (D3): identity / labels belong in
        # the per-clip keyframe idiom / subject_definitions, NOT in the
        # shared prefix.
        if ref_code != "t2va" and (prompt_prefix_input or "").strip():
            joined = "\n".join(split_prefix_paragraphs(prompt_prefix_input))
            bad_labels = find_native_labels(joined)
            bad_aliases = find_alias_tokens(joined)
            bad_anchors = find_semantic_anchors(joined)
            if bad_labels or bad_aliases or bad_anchors:
                pieces = []
                if bad_labels:
                    pieces.append(
                        "native labels "
                        + ", ".join(f"<{k} {n}>" for k, n in bad_labels[:3])
                    )
                if bad_aliases:
                    pieces.append(
                        "@alias tokens "
                        + ", ".join("@" + a for a in bad_aliases[:3])
                    )
                if bad_anchors:
                    pieces.append(
                        "#tag tokens "
                        + ", ".join("#" + n for n, _ in bad_anchors[:3])
                    )
                raise RuntimeError(
                    "prompt_prefix_input must not carry "
                    + " / ".join(pieces)
                    + f" in reference_mode={ref_code}; strip them — identity words "
                    "belong in the per-clip keyframe idiom / subject_definitions, "
                    "not in the shared prefix."
                )

        count = int(shot_count or 0)
        if not (shots_text or "").strip() and count > 0:
            # Standalone mode: no storyboard connected — build one inline.
            shots, sb_warnings = self._auto_storyboard(
                idea,
                count,
                category,
                (output_language or "en").strip().lower(),
                seed=seed,
            )
            warnings.extend(f"auto-storyboard: {w}" for w in sb_warnings)
        else:
            try:
                shots = parse_shots_text(shots_text)
            except ValueError as exc:
                # Bad user input fails the node with one consistent error type.
                raise RuntimeError(str(exc)) from exc
            if count > 0 and len(shots) > count:
                dropped = ", ".join(s["id"] for s in shots[count:])
                shots = shots[:count]
                warnings.append(
                    f"shot_count={count}: trimmed incoming storyboard to first {count} (dropped: {dropped})"
                )
        try:
            overrides = parse_per_shot_overrides(per_shot_overrides)
            shots = apply_overrides(shots, overrides)
        except ValueError as exc:
            # Bad user input fails the node with one consistent error type.
            raise RuntimeError(str(exc)) from exc
        seed_base = derive_seed_base(seed)

        entries: list[dict] = []
        for i, shot in enumerate(shots, start=1):
            if "length" in shot:
                length = int(shot["length"])
            else:
                dur = shot.get("duration_seconds") or duration_seconds
                length = seconds_to_length(dur)
            # "seed" only ever arrives as a digit string from overrides.
            overridden_seed = shot.get("seed")
            shot_seed = (
                overridden_seed
                if isinstance(overridden_seed, str) and overridden_seed.isdigit()
                else derive_seed(seed_base, i, unified=bool(unified_seed))
            )
            entries.append(
                {
                    "id": shot["id"],
                    "source": shot,
                    "length": length,
                    "seed": shot_seed,
                    "steps": int(shot.get("steps") or DEFAULT_STEPS),
                }
            )
        log_pipeline(
            f"stage0 done: {len(entries)} clips, mode={parse_generation_mode(generation_mode)}"
        )

        # ---- Stage 1: shared prompt_prefix ------------------------------ #
        prefix_lines = split_prefix_paragraphs(prompt_prefix_input or "")
        if not prefix_lines:
            prefix_lines = self._synth_prefix(
                idea,
                category,
                output_language,
                shots,
                seed=seed,
                mode=ref_code,
                manifest=manifest,
            )
            log_pipeline(f"stage1 derived prompt_prefix ({len(prefix_lines)} paragraph(s))")
        else:
            log_pipeline(f"stage1 using user prompt_prefix ({len(prefix_lines)} paragraph(s))")
        prefix_text = "\n\n".join(prefix_lines)

        # ---- Stage 2: clip prompts -------------------------------------- #
        language_name = "Chinese" if (output_language or "en").strip().lower() == "zh" else "English"
        mode = parse_generation_mode(generation_mode)
        prompts: dict[str, list[str]] = {}
        if mode == "single_call":
            prompts = self._generate_all_shots_single_call(
                concept=idea,
                prefix_text=prefix_text,
                category=category,
                shots=[e["source"] for e in entries],
                width=width,
                height=height,
                duration_seconds=int(duration_seconds),
                language_name=language_name,
                seed=seed,
            )
        else:
            prev_lines: Optional[list[str]] = None
            prev_id = ""
            prev_subject_definitions: list[str] = []
            prev_soundscape: list[str] = []
            for i, entry in enumerate(entries, start=1):
                lines = self._generate_shot_prompt(
                    concept=idea,
                    prefix_text=prefix_text,
                    category=category,
                    shot=entry["source"],
                    clip_index=i,
                    clip_count=len(entries),
                    prev_lines=prev_lines,
                    prev_id=prev_id,
                    prev_subject_definitions=prev_subject_definitions,
                    width=width,
                    height=height,
                    # Pace against the ACTUAL grid-rounded length (10s ->
                    # 243 frames -> 10.1s), not the requested seconds.
                    duration_seconds=length_to_seconds(entry["length"]),
                    language_name=language_name,
                    seed=seed,
                    mode=ref_code,
                    manifest=manifest,
                )
                prompts[entry["id"]] = lines
                prev_lines, prev_id = lines, entry["id"]
                if schema == SCHEMA_SIX:
                    # Carry the last scene's subject_definitions body into
                    # the next scene's continuation block — body ONLY: end
                    # at the summary header so summary/retention_analysis
                    # text is not smuggled under the bindings heading.
                    try:
                        subj_start = lines.index(f"{SIX_SECTION_FIELDS[0]}:") + 1
                        subj_end = lines.index(f"{SIX_SECTION_FIELDS[1]}:")
                    except ValueError:
                        subj_start, subj_end = 0, 0
                    prev_subject_definitions = (
                        lines[subj_start:subj_end] if subj_end > subj_start else []
                    )
                log_pipeline(f"stage2 clip {i}/{len(entries)} ({entry['id']}) done")

        # ---- Stage 3: assemble + validate -------------------------------- #
        shot_entries = [
            {
                "id": e["id"],
                "prompt": prompts[e["id"]],
                "length": e["length"],
                "seed": e["seed"],
                "steps": e["steps"],
            }
            for e in entries
        ]
        plan = build_plan(shot_entries, prefix_lines, default_steps=DEFAULT_STEPS)
        errors = validate_plan(plan, schema=schema)
        if errors:
            raise RuntimeError("plan validation failed: " + "; ".join(errors))
        label_errors = validate_label_policy(plan, ref_code, manifest)
        if label_errors:
            raise RuntimeError(
                f"label policy failed for reference_mode={ref_code}: "
                + "; ".join(label_errors)
            )
        log_pipeline(
            f"stage3 plan assembled: {len(plan['shots'])} shots, "
            f"schema={schema}, validation clean"
        )

        return {
            "plan_json": plan_to_json_string(plan),
            "shot_prompts": plan_to_json_string([s["prompt"] for s in plan["shots"]]),
            "prompt_prefix_out": plan_to_json_string(prefix_lines),
            "preflight_report": build_preflight_report(
                plan, warnings, mode=ref_code, manifest=manifest
            ),
            "plan_preview": build_plan_preview(plan),
        }


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class MiniMaxH3LoopPromptGenerator:
    """ComfyUI node: concept + storyboard -> Production Plan ``plan_json``.

    Connect ``MiniMaxH3StoryboardGenerator.shots_json`` to ``shots_text``
    (or paste one shot per line), then ``plan_json`` into
    ``MiniMaxH3ChainPlanModern.plan_json_input``. The generated JSON only
    carries ``defaults`` / ``shots`` / ``prompt_prefix`` — canvas size,
    continuation mode, run name etc. stay on the Plan node's widgets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "concept": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Overall creative concept (same one fed to the "
                            "storyboard generator). Blank = default concept."
                        ),
                    },
                ),
                "shots_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Storyboard input: connect "
                            "MiniMaxH3StoryboardGenerator.shots_json here, or "
                            "paste one shot per line of natural language."
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
                            "Seed base for the per-shot seed chain "
                            "(shot N gets base+N as a string). 0 = derive "
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
                        "max": 20,
                        "tooltip": (
                            "0 = follow shots_text (must be non-empty). "
                            ">0 with EMPTY shots_text = auto-generate the "
                            "storyboard from concept (one extra LLM call), "
                            "making this node standalone. >0 with non-empty "
                            "shots_text = use only the first N shots."
                        ),
                    },
                ),
                "duration_seconds": (
                    "INT",
                    {
                        "default": DEFAULT_DURATION_SECONDS,
                        "min": 1,
                        "max": 149,
                        "tooltip": (
                            "Default clip duration; converted up onto the H3 "
                            "17k+5 frame grid (10s -> 243 frames). Shots with "
                            "their own duration_seconds use it instead."
                        ),
                    },
                ),
                "generation_mode": (
                    list(GENERATION_MODES),
                    {
                        "default": GENERATION_MODES[0],
                        "tooltip": (
                            "per_shot: one LLM call per clip, best continuity "
                            "(recommended). single_call: one LLM call for the "
                            "whole board (cheaper/faster).",
                        ),
                    },
                ),
                "category": (
                    list(CATEGORIES),
                    {"default": CATEGORIES[0]},
                ),
                "width": (
                    "INT",
                    {
                        "default": 544,
                        "min": 32,
                        "max": 4096,
                        "step": 32,
                        "tooltip": (
                            "Canvas size hint for prompt writing. Validated "
                            "as a multiple of 32; NOT written into the plan "
                            "JSON (set it on the Plan node too)."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {"default": 960, "min": 32, "max": 4096, "step": 32},
                ),
                "output_language": (["en", "zh"], {"default": "en"}),
                "prompt_prefix_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Shared prefix pinned before every clip prompt "
                            "(identity/wardrobe/style). Blank = derived by "
                            "one light LLM call."
                        ),
                    },
                ),
                "per_shot_overrides": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "One per line: 'id:length=N' (H3 grid), "
                            "'id:seed=NNNN' (uint64), 'id:steps=N'."
                        ),
                    },
                ),
                "unified_seed": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True = every clip shares the same seed "
                            "(same noise init -> better cross-clip "
                            "identity/style hold, recommended for seamless "
                            "chains). False = per-clip seeds (seed + index) "
                            "for diversity. 'id:seed=N' overrides always "
                            "win."
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
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _DEFAULT_MAX_TOKENS,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                    },
                ),
                "timeout": ([30, 60, 120, 300], {"default": _DEFAULT_TIMEOUT}),
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
                "references_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Reference manifest. Either a JSON array of "
                            "{slot, about, role} objects or one "
                            "'Picture N: <about>' line per slot. "
                            "Empty for t2va; 1 for i2va; "
                            "2 for fl2va; 1..9 for ref2va. role=identity "
                            "binds that slot to <Subject N> in ref2va."
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
        concept,
        shots_text,
        seed=None,
        shot_count=0,
        unified_seed=True,
        duration_seconds=DEFAULT_DURATION_SECONDS,
        generation_mode=GENERATION_MODES[0],
        category="",
        width=544,
        height=960,
        output_language="en",
        prompt_prefix_input="",
        per_shot_overrides="",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        reference_mode=REFERENCE_MODES[0],
        references_text="",
    ):
        enhancer = H3LoopPromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            concept,
            shots_text,
            duration_seconds=duration_seconds,
            generation_mode=generation_mode,
            category=category,
            width=width,
            height=height,
            output_language=output_language,
            seed=seed,
            prompt_prefix_input=prompt_prefix_input,
            per_shot_overrides=per_shot_overrides,
            shot_count=shot_count,
            unified_seed=unified_seed,
            reference_mode=reference_mode,
            references_text=references_text,
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
        concept,
        shots_text,
        seed=None,
        shot_count=0,
        unified_seed=True,
        duration_seconds=DEFAULT_DURATION_SECONDS,
        generation_mode=GENERATION_MODES[0],
        category="",
        width=544,
        height=960,
        output_language="en",
        prompt_prefix_input="",
        per_shot_overrides="",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
                timeout=_DEFAULT_TIMEOUT,
        reference_mode=REFERENCE_MODES[0],
        references_text="",
    ):
        h = hashlib.md5()
        for part in (
            concept,
            shots_text,
            str(seed),
            str(shot_count),
            str(bool(unified_seed)),
            str(duration_seconds),
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
            manifest_normalize_json(parse_references_text(references_text)),
        ):
            h.update((part or "").encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
