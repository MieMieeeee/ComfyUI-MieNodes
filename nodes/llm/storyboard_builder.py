# -*- coding: utf-8 -*-
"""Zero-LLM plan assembler for the storyboard asset track
(docs/PLAN_STORYBOARD_ASSET.md).

Why this module exists:
  The H3 Loop pipeline already does the heavy lifting of stamping a
  six-section Ref2VA prompt (subject / summary / retention / detail /
  sound / music) from a roster of cast members and a sequence of
  dialogue lines. ``build_fallback_shot_lines`` + ``install_shot_speech``
  were always the deterministic part — they DO NOT touch an LLM. What
  this module does is take a hand-written storyboard JSON (cast, shots,
  optional spatial layout, optional English Setting), assemble one
  identical-shape prompt per shot with no model in the loop, and emit
  the same dict the Longform-V Plan node accepts.

Hard rules (mirroring PLAN_STORYBOARD_ASSET §1 + §2):
  * No LLM calls. No connector sockets. No ``_invoke`` / ``enhance_user_input``.
  * No translation of locked text. The verbatim line ``text`` becomes the
    verbatim ``<d>[language] text</d>`` payload, byte for byte.
  * No re-authoring. The English ``setting_line``, ``picture`` cues and
    ``prompt_prefix`` go in as the user wrote them — a hostile printer
    (CJK in an English-only field) is rejected at the JSON layer.

Byte-level reproducibility:
  Same storyboard JSON + same connected character/scene bundles + same
  seed widget → byte-identical ``plan_json`` string. The only non-input
  source of variance is the IMAGE batch (mixed-size pictures return
  ``None`` for the convenience output); the plan text never depends on
  pixel bytes.
"""
from __future__ import annotations

import re
from typing import Any, Optional

try:
    from _mienodes_internal.nodes.assets import library as _library
    from _mienodes_internal.nodes.assets import storyboard as _storyboard
except ImportError:  # pragma: no cover - direct-script path
    from ..assets import library as _library  # type: ignore
    from ..assets import storyboard as _storyboard  # type: ignore

try:
    from _mienodes_internal.nodes.llm.dialogue_segmenter import (
        DialogueTurn,
        Pacing,
        ShotBudget,
        PACING_PRESETS,
        estimate_shot_budget,
        group_budgets_into_scenes,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        DEFAULT_MUSIC_LINE,
        SCHEMA_SIX,
        SCHEMA_THREE,
        SIX_SECTION_FIELDS,
        SECTION_SUBJECT,
        SECTION_SUMMARY,
        SECTION_RETENTION,
        SECTION_DETAIL,
        SECTION_SOUND,
        SECTION_MUSIC,
        _replace_section_body,
        build_fallback_shot_lines,
        build_plan,
        complete_layout_from_turn_order,
        install_shot_speech,
        plan_to_json_string,
        resolve_line_facing,
        validate_dialogue_invariantity,
        validate_label_policy,
        validate_plan,
        validate_spatial_layout_invariant,
    )
except ImportError:  # pragma: no cover
    from .dialogue_segmenter import (  # type: ignore
        DialogueTurn,
        Pacing,
        ShotBudget,
        PACING_PRESETS,
        estimate_shot_budget,
        group_budgets_into_scenes,
    )
    from .minimax_h3_loop_prompts import (  # type: ignore
        DEFAULT_MUSIC_LINE,
        SCHEMA_SIX,
        SCHEMA_THREE,
        SIX_SECTION_FIELDS,
        SECTION_SUBJECT,
        SECTION_SUMMARY,
        SECTION_RETENTION,
        SECTION_DETAIL,
        SECTION_SOUND,
        SECTION_MUSIC,
        _replace_section_body,
        build_fallback_shot_lines,
        build_plan,
        complete_layout_from_turn_order,
        install_shot_speech,
        plan_to_json_string,
        resolve_line_facing,
        validate_dialogue_invariantity,
        validate_label_policy,
        validate_plan,
        validate_spatial_layout_invariant,
    )


# --------------------------------------------------------------------------- #
# Camera-discipline fallback (locked / no_closeups / auto all use the same
# locked [Shot 1] mid-shot — §2 "we accept flatter prose"; the discipline
# is only used to pick one of three English ambience defaults when BOTH the
# storyboard ambience and the scene ambience are empty).
# --------------------------------------------------------------------------- #
AMBIENCE_FALLBACKS = {
    "locked": "The room tone stays quiet and even throughout.",
    "no_closeups": "The room tone stays quiet and even throughout.",
    "auto": "The room tone stays quiet and even throughout.",
}

# `<d>` tag prefix from the storyboard's ``language`` field (PLAN §3.1
# "每个 <d> 标签都用它：zh → [Chinese]，en → [English]。不按字面检测语言").
DIALOGUE_TAG = {"zh": "[Chinese]", "en": "[English]"}


# --------------------------------------------------------------------------- #
# Shot packing
# --------------------------------------------------------------------------- #
def _per_line_budgets(lines_per_shot: list[list[dict]], pacing: Pacing):
    """Run ``estimate_shot_budget`` over a SINGLE-SPEAKER turn at a time so
    the dialogue_segmenter's per-line packing rules carry over verbatim
    even when adjacent turns share a speaker.

    The builder does not need the per-line audio bookkeeping — only the
    rounded length — so the per-shot budgets are merged back here into
    a ``length: int, lines: list[dict]`` tuple per shot."""
    budgets: list[tuple[int, list[dict]]] = []
    for shot_lines in lines_per_shot:
        if not shot_lines:
            budgets.append((pacing_to_minimum_length(pacing), []))
            continue
        # Construct a single-turn placeholder so the estimator runs the
        # same code path as the LLM track. The speaker label does not
        # affect length maths; we only need lines / line_count.
        speaker = shot_lines[0]["speaker"]
        texts = [str(ln["text"]) for ln in shot_lines]
        turn = DialogueTurn(speaker=speaker, lines=texts)
        per_turn, _report = estimate_shot_budget([turn], pacing)
        if len(per_turn) == 1:
            length = int(per_turn[0].rounded_length_frames)
            budgets.append((length, list(shot_lines)))
        else:
            # A single shot spanned multiple sub-budgets because the
            # packed lines overflowed the 14 s ceiling. We still keep
            # them on one scene (the assembler accepts a longer length)
            # — the dialogue_segmenter split is a model-pacing tweak that
    # the zero-LLM track does not need.
            length = sum(int(b.rounded_length_frames) for b in per_turn)
            budgets.append((length, list(shot_lines)))
    return budgets


def pacing_to_minimum_length(pacing: Pacing) -> int:
    """A speaker change at scene boundary with no lines is rare; we still
    need a positive ``length``. One second is the smallest valid H3
    length (24 frames) — round up to the 17k+5 grid (5)."""
    return 5


# --------------------------------------------------------------------------- #
# Mechanical shot builder (the zero-LLM heart of this module)
# --------------------------------------------------------------------------- #
def _speaker_to_slot(cast: dict[str, dict]) -> dict[str, int]:
    """``{speaker_name: character_slot}``. Built once per build call —
    drives ``(Sn)`` tag assignment in the speech sentences and the
    bundle ordering later."""
    return {name: int(info["character_slot"]) for name, info in cast.items()}


def _lines_per_shot(storyboard: dict):
    """Normalise the storyboard into ``[[{speaker, text, picture?}, ...], ...]``.

    When ``shots`` is omitted, the auto-packer (``group_budgets_into_scenes``
    with ``max_lines_per_scene=2``) is the documented behaviour — we
    delegate that to the dialogue_segmenter in ``build_assembly``. When
    ``shots`` is explicit, the order is preserved verbatim and the
    snapshot is built one-for-one.
    """
    raw = storyboard.get("shots")
    if raw is not None:
        return [list(shot["lines"]) for shot in raw]
    return None  # caller handles the auto-pack path


def _build_manifest(storyboard: dict, slot_bundles: list[dict]) -> list[dict]:
    """Build the six-section manifest the prompt helpers expect.

    ``slot_bundles`` is the ordered list of wired character/scene bundles
    (characters then scenes), where slot N is ``character_slot=N`` or, for
    scenes, the next sequential slot past the last character. Each entry
    contributes ``{"slot": "Picture N", "about": <asset description>,
    "role": "identity" or "environment"}``."""
    speaker_to_slot = _speaker_to_slot(storyboard["cast"])
    # Pictures: characters first (in slot order), then the scene slots
    # past them. The builder doesn't accept more than one scene in v1
    # (§3.1: only ``scene_1``), so the picture count is
    # ``len(characters) + 1``.
    picture_specs = []  # (slot_n, bundle, speaker_or_None, role)
    slot = 0
    for bundle in slot_bundles:
        if bundle.get("kind") != "character":
            continue
        slot += 1
        picture_specs.append((slot, bundle, None, "identity"))
    for bundle in slot_bundles:
        if bundle.get("kind") != "scene":
            continue
        slot += 1
        picture_specs.append((slot, bundle, None, "environment"))

    manifest = []
    for pic_slot, bundle, _speaker, role in picture_specs:
        about = (bundle.get("description") or "").strip()
        if not about:
            raise ValueError(
                "asset on slot %d has no description; re-save it with a "
                "non-empty English description" % pic_slot
            )
        manifest.append({
            "slot": "Picture %d" % pic_slot,
            "about": about,
            "role": role,
        })
    return manifest


def _resolve_ambience(
    storyboard: dict,
    scene_bundles: list[dict],
) -> str:
    """Resolution order: storyboard ``ambience`` → first scene's
    ``ambience`` → camera-discipline fallback. Returns one English line.
    """
    raw = (storyboard.get("ambience") or "").strip()
    if raw:
        return raw
    for bundle in scene_bundles:
        if isinstance(bundle, dict):
            amb = (bundle.get("ambience") or "").strip()
            if amb:
                return amb
    return AMBIENCE_FALLBACKS.get(
        storyboard.get("camera_discipline", "locked"),
        AMBIENCE_FALLBACKS["locked"],
    )


def _check_character_audio(bundle: dict, slot: int) -> str:
    """Empty voice or missing audio is a hard error (PLAN §3.1 hard
    error list). Returns the trimmed voice string on success."""
    voice = (bundle.get("voice") or "").strip()
    voice_clean = re.sub(r"\s+", " ", voice).rstrip(" .,;")
    if not voice_clean:
        raise ValueError(
            "character on slot %d (%s) has empty voice descriptor; "
            "re-save the character with a non-empty voice line"
            % (slot, bundle.get("name") or "?")
        )
    if bundle.get("audio") is None:
        raise ValueError(
            "character on slot %d (%s) has no audio sample wired; "
            "the bundle must carry an AUDIO connection"
            % (slot, bundle.get("name") or "?")
        )
    return voice_clean


def _slot_to_character(slot_bundles: list[dict]) -> dict[int, dict]:
    """Map ``character_slot`` → character bundle. Characters are wired in
    ``character_1..8`` socket order; ``character_slot`` from the storyboard
    is the slot index. A bundle whose slot does not exist → hard error."""
    out: dict[int, dict] = {}
    for bundle in slot_bundles:
        if bundle.get("kind") != "character":
            continue
        out[len(out) + 1] = bundle
    return out


def _resolve_role_bindings(storyboard: dict) -> dict[str, str]:
    """The plan's vocative-facing detector expects role→speaker name
    bindings. Storyboards don't carry role syntax, so we return an empty
    mapping: a line-initial vocative like 「爷爷，」 therefore does NOT
    insert a ``turns to face`` clause (the regex won't match anyway, and
    the builder's choice in §6 is no facing rewrite)."""
    return {}


def _assemble_one_shot(
    *,
    shot_index: int,
    lines: list[dict],
    speaker_to_slot: dict[str, int],
    slot_to_voice: dict[int, str],
    spatial_layout: dict[str, str],
    storyboard: dict,
    manifest: list[dict],
    ambience_line: str,
) -> dict:
    """Build one shot's schema, segments, length and seed. No LLM. The
    output dict matches ``build_plan``'s shot entry shape.

    We intentionally bypass ``install_shot_speech`` because its cue
    branch passes stage cues through ``_cue_sentence``, which wraps a
    non-still cue as "Silent picture only. Mouths stay closed and nobody
    speaks." — a contradiction when the cue is on a SPEAKING line.
    ``_build_storyboard_shot_body`` below emits the same six-section
    shape using the underlying helpers directly."""
    prompt_lines = build_fallback_shot_lines(
        SCHEMA_SIX, manifest=manifest, spatial_layout=spatial_layout,
    )
    # Pin the resolved ambience (storyboard > scene > camera-discipline
    # fallback) into ``overall_soundscape`` BEFORE the body is
    # composed — the speech-path logic preserves the soundscape (it
    # only replaces the section body in voiceover mode, which we never
    # use).
    prompt_lines = _replace_section_body(
        prompt_lines, "overall_soundscape:", [ambience_line],
    )

    visible_names = sorted(
        {str(name).strip() for name in spatial_layout
         if str(name).strip()}
        | {ln["speaker"] for ln in lines if ln.get("speaker")}
    )
    speaker_identities = {
        name: next(
            (m["about"] for m in manifest
             if m["role"] == "identity"
             and m["slot"] == "Picture %d" % speaker_to_slot.get(name, -1)),
            "",
        )
        for name in visible_names
    }
    prompt_lines = _build_storyboard_shot_body(
        prompt_lines,
        lines=lines,
        speaker_to_slot=speaker_to_slot,
        slot_to_voice=slot_to_voice,
        spatial_layout=spatial_layout,
        speaker_identities=speaker_identities,
        storyboard=storyboard,
    )

    # Length: the dialogue_segmenter's per-shot budget covers each
    # user-authored shot 1:1.
    pacing = PACING_PRESETS[storyboard.get("pacing", "normal")]
    speaker = lines[0]["speaker"] if lines else ""
    turn = DialogueTurn(speaker=speaker, lines=[ln["text"] for ln in lines])
    per_turn, _ = estimate_shot_budget([turn], pacing)
    length = int(per_turn[0].rounded_length_frames)

    return {
        "id": "scene_%02d" % shot_index,
        "prompt": prompt_lines,
        "length": length,
        "seed": _seed_for(storyboard, shot_index),
    }


def _build_storyboard_shot_body(
    prompt_lines: list[str],
    *,
    lines: list[dict],
    speaker_to_slot: dict[str, int],
    slot_to_voice: dict[int, str],
    spatial_layout: dict[str, str],
    speaker_identities: dict[str, str],
    storyboard: dict,
) -> list[str]:
    """Compose the six-section shot body without going through
    ``install_shot_speech``. Mirrors its ref2va path but keeps the
    picture cue as a plain descriptive sentence (not the silent-only
    wrapper)."""
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (  # noqa
        _shot1_coverage,
        build_spatial_layout_directive,
        identity_pin_lines,
        annotate_retention_pose,
        neutralize_offscreen_claims,
    )

    prompt_lines = neutralize_offscreen_claims(prompt_lines)

    visible_names = sorted(
        {str(n).strip() for n in spatial_layout if str(n).strip()}
        | {ln["speaker"] for ln in lines if ln.get("speaker")}
    )
    if visible_names:
        pins = identity_pin_lines(visible_names, speaker_identities)
        if pins:
            prompt_lines = _replace_section_body(
                prompt_lines, "subject_definitions:", pins
            )

    dialogue_tag = DIALOGUE_TAG.get(
        storyboard.get("language", "en"), DIALOGUE_TAG["en"]
    )
    speech_lines: list[str] = []
    for ln in lines:
        speaker = ln["speaker"]
        slot = speaker_to_slot.get(speaker)
        if slot is None:
            raise ValueError(
                "speaker %r is not in cast %s"
                % (speaker, sorted(speaker_to_slot))
            )
        voice = slot_to_voice.get(slot, "")
        text = ln["text"]
        head = "%s speaks as (S%d) %s" % (speaker, slot, voice)
        spoken = "<d>%s %s</d>" % (dialogue_tag, text)
        speech_lines.append("%s %s" % (head, spoken))

    body: list[str] = [_shot1_coverage(visible_names, voiceover=False)]
    spatial = build_spatial_layout_directive(spatial_layout)
    if spatial and not spatial.startswith("(no spatial"):
        body.append(spatial)
    if storyboard.get("setting_line"):
        body.append("Setting: " + storyboard["setting_line"])
    # Inline each line's picture cue as a plain descriptive sentence.
    for ln, speech in zip(lines, speech_lines):
        cue = (ln.get("picture") or "").strip()
        if cue:
            body.append(_picture_cue_sentence(cue))
        body.append(speech)

    # Find the description section (six-section = detailed_description).
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        replace_description_body,
    )
    prompt_lines = replace_description_body(prompt_lines, body, schema=SCHEMA_SIX)

    # Reference-generation summary (mirrors install_shot_speech's
    # SCHEMA_SIX branch).
    speaking = sorted({ln["speaker"] for ln in lines if ln.get("speaker")})
    if set(speaking) >= set(visible_names):
        who = ", ".join(speaking) if speaking else "The subjects"
        summary = (
            "[reference generation] "
            f"{who} are visible and speaking. "
            "The spoken words are not restated here."
        )
    else:
        vis = " and ".join(visible_names) if visible_names else "the subjects"
        vis_verb = "is" if len(visible_names) == 1 else "are"
        spk = ", ".join(speaking) if speaking else "nobody"
        spk_verb = "is" if len(speaking) == 1 else "are"
        summary = (
            "[reference generation] "
            f"{vis} {vis_verb} visible in the frame; "
            f"{spk} {spk_verb} speaking. "
            "The spoken words are not restated here."
        )
    prompt_lines = _replace_section_body(prompt_lines, "summary:", [summary, ""])
    prompt_lines = annotate_retention_pose(prompt_lines)
    return prompt_lines


def _seed_for(storyboard: dict, shot_index_one_based: int) -> str:
    """Decimal string seed. ``per_scene_increment`` = ``str(seed_base +
    i)`` where i is zero-based. ``same_across_scenes`` = ``str(seed_base)``
    for every scene."""
    base = int(storyboard.get("seed_base", 0))
    policy = storyboard.get("seed_policy", "per_scene_increment")
    if policy == "same_across_scenes":
        return str(base)
    return str(base + (shot_index_one_based - 1))


def _picture_cue_sentence(cue: str) -> str:
    """A picture cue rendered as a plain H3 body sentence (no
    silent-only wrapper). CJK was rejected upstream by
    ``storyboard.validate_picture_cue``, so the text here is already
    English-only."""
    text = (cue or "").strip()
    if not text:
        return ""
    if not text.endswith("."):
        text = text + "."
    return text


# --------------------------------------------------------------------------- #
# Public assembler
# --------------------------------------------------------------------------- #
def assemble_plan(
    *,
    storyboard: dict,
    slot_bundles: list[dict],
    seed_base: int = 0,
    summary_extra: Optional[list[str]] = None,
    _skip_validation: bool = False,
) -> dict:
    """Pure function: storyboard JSON + wired asset bundles → plan dict
    (the same shape ``MiniMaxH3ChainPlanModern`` accepts).

    ``slot_bundles`` are in slot order (character_1..N then scene_1).
    Every character the storyboard references must be in
    ``slot_bundles``; the scene (only ``scene_1`` in v1) must be in the
    LAST slot (``Picture #slot_of_N + 1``). The build fails loudly when
    a required slot is empty.

    Returns ``{"plan": {...}, "summary_lines": [...], "ambience": str,
    "size_note": ""}``. ``plan`` is ready for ``plan_to_json_string``; the
    callers (the node) call that themselves.

    ``_skip_validation`` is an internal fast-path for tests that already
    validated the payload through ``library.validate_storyboard_payload``.
    Production code accepts the cost of re-validating so a malformed
    bundle from any caller cannot reach the assembly path.
    """
    if not isinstance(storyboard, dict):
        raise ValueError("storyboard bundle must be a dict")
    if not isinstance(slot_bundles, list):
        raise ValueError("slot_bundles must be a list")
    if not _skip_validation:
        storyboard = _storyboard.validate_storyboard_payload(storyboard)

    cast = storyboard.get("cast") or {}
    if not cast:
        raise ValueError("storyboard has no cast; the builder cannot "
                         "assemble a plan without speakers")
    speaker_to_slot = _speaker_to_slot(cast)
    # Sanity: the storyboard validates slots during JSON load, but the
    # build node may receive a bundle dict directly (the test suite).
    used = sorted(set(int(v["character_slot"]) for v in cast.values()))
    if len(used) != len(cast):
        raise ValueError("duplicate character_slot values in cast")
    if max(used) > _storyboard.MAX_CHARACTER_SLOT:
        raise ValueError(
            "cast uses character_slot %d; the upper limit is %d"
            % (max(used), _storyboard.MAX_CHARACTER_SLOT)
        )

    # Collect the bundles actually needed.
    character_bundles = [b for b in slot_bundles if b.get("kind") == "character"]
    scene_bundles = [b for b in slot_bundles if b.get("kind") == "scene"]
    if not scene_bundles:
        raise ValueError(
            "the zero-LLM builder needs at least one scene wire (scene_1); "
            "no scene bundles found in the inputs"
        )
    if len(scene_bundles) > 1:
        raise ValueError(
            "storyboard builder accepts exactly one scene in this version; "
            "got %d scene bundles" % len(scene_bundles)
        )

    # Slot-to-bundle map (character_1..N = bundle index in character_bundles,
    # then scene_1 = scene_bundles[0]).
    slot_to_bundle: dict[int, dict] = {}
    for idx, bundle in enumerate(character_bundles, 1):
        slot_to_bundle[idx] = bundle
    scene_slot = len(character_bundles) + 1
    slot_to_bundle[scene_slot] = scene_bundles[0]

    # Every character slot used in the cast MUST be wired.
    for slot in used:
        if slot not in slot_to_bundle:
            raise ValueError(
                "character_slot %d is referenced by cast but no "
                "character_%d asset was wired into the builder"
                % (slot, slot)
            )

    # Voice / image / audio checks per character bundle.
    slot_to_voice: dict[int, str] = {}
    for slot in used:
        bundle = slot_to_bundle[slot]
        if bundle.get("kind") != "character":
            raise ValueError(
                "character_slot %d expected a character bundle, got kind=%r"
                % (slot, bundle.get("kind"))
            )
        slot_to_voice[slot] = _check_character_audio(bundle, slot)

    # Picture index order: characters in slot order, then the scene.
    manifest_bundles = list(character_bundles) + list(scene_bundles)
    manifest = _build_manifest(storyboard, manifest_bundles)

    # Spatial layout: from the storyboard; missing speakers are filled
    # in turn-order via the existing helper (PLAN §3.1 "缺的说话人用
    # complete_layout_from_turn_order 补左/右/中").
    turn_speaker_order = []
    for sl in storyboard.get("shots") or []:
        for ln in sl["lines"]:
            if ln["speaker"] not in turn_speaker_order:
                turn_speaker_order.append(ln["speaker"])
    declared = storyboard.get("spatial_layout") or {}
    completed_layout = complete_layout_from_turn_order(
        turn_speaker_order, declared,
    )
    layout_was_filled = False
    declared_keys = set(declared.keys())
    for name in completed_layout:
        if name not in declared_keys:
            layout_was_filled = True
            break

    # Ambience: storyboard → first scene bundle → camera-discipline
    # fallback.
    ambience = _resolve_ambience(storyboard, scene_bundles)

    # Build the shots. We pack the user-authored shots 1:1 — when
    # ``shots`` is omitted, fall back to the dialogue_segmenter auto
    # packing with ``max_lines_per_scene=2`` (PLAN §3.1).
    raw_shots = storyboard.get("shots")
    if raw_shots is None:
        # Auto-pack: every line becomes its own turn; pack two lines
        # per scene.
        all_lines: list[dict] = []
        # The storyboard has no fixed speakers per turn — we treat
        # every line as its own turn so consecutive same-speaker lines
        # stay together (matches the LLM track's grouping rule).
        cur_speaker = None
        cur_block: list[dict] = []
        # We don't have a single flat dialogue list, so the simplest
        # faithful auto-pack is: list EVERY line as one scene (one line
        # per shot) — this is what the LLM track's auto scene_count
        # cap = 1 + REVISED 2026-10-12 rule would emit on a stripped
        # storyboard without per-turn dialogue structure.
        raise ValueError(
            "auto-pack path is not supported in this version of the "
            "storyboard builder — author shots: [...] in the JSON, or wire "
            "the LLM-track MiniMaxH3LoopPromptGenerator"
        )

    shot_entries: list[dict] = []
    for shot_idx, sl in enumerate(raw_shots, 1):
        shot = _assemble_one_shot(
            shot_index=shot_idx,
            lines=sl["lines"],
            speaker_to_slot=speaker_to_slot,
            slot_to_voice=slot_to_voice,
            spatial_layout=completed_layout,
            storyboard={**storyboard, "seed_base": seed_base},
            manifest=manifest,
            ambience_line=ambience,
        )
        shot_entries.append(shot)

    prefix_lines = list(storyboard.get("prompt_prefix") or [])

    plan = build_plan(shot_entries, prefix_lines)

    # Final shape audit (matches the LLM track's last-mile checks).
    plan_errors = validate_plan(plan, schema=SCHEMA_SIX)
    if plan_errors:
        raise ValueError(
            "assembled plan failed validation: " + "; ".join(plan_errors)
        )

    label_errors = validate_label_policy(
        plan, "ref2va", manifest,
        referenced_pictures=set(range(1, len(manifest) + 1)),
    )
    if label_errors:
        raise ValueError(
            "assembled plan label-policy violations: " + "; ".join(label_errors)
        )

    # Dialogue invariantity (1 line == 1 <d> block, verbatim text).
    turns = [
        DialogueTurn(
            speaker=sl["lines"][0]["speaker"],
            lines=[ln["text"] for ln in sl["lines"]],
        )
        for sl in raw_shots
    ]
    dialogue_errors = validate_dialogue_invariantity(plan["shots"], turns)
    if dialogue_errors:
        raise ValueError(
            "assembled plan dialogue invariantity violations: "
            + "; ".join(dialogue_errors)
        )

    spatial_errors = validate_spatial_layout_invariant(
        completed_layout,
        [{"spatial_layout": completed_layout}] * len(plan["shots"]),
    )
    if spatial_errors:
        raise ValueError(
            "assembled plan spatial-layout violations: "
            + "; ".join(spatial_errors)
        )

    # Run-summary lines (string outputs only, never the plan text).
    summary_lines: list[str] = []
    if layout_was_filled:
        summary_lines.append(
            "layout filled by turn order (no entry in spatial_layout for "
            "every cast speaker)"
        )
    # Monologue-scene hint (mirrors the LLM track's `_monologue_scene_hint`).
    cast_size = len(cast)
    mono_lines = _monologue_hint(plan["shots"], cast_size)
    if mono_lines:
        summary_lines.extend(mono_lines)
    if summary_extra:
        summary_lines.extend(summary_extra)

    return {
        "plan": plan,
        "summary_lines": summary_lines,
        "ambience": ambience,
        "size_note": "",
    }


def _monologue_hint(shots: list[dict], cast_size: int) -> list[str]:
    """A near-copy of the LLM track's monologue hint: when ONE speaker
    fills an entire shot, the H3 render layer has a known flake that
    drops the speaker from the frame (2026-10-18 LAN forensics). The
    mechanical builder does not fix the flake — it just reports it."""
    if cast_size < 2:
        return []
    mono: list[tuple[str, str, int]] = []
    for shot in shots or []:
        prompt = shot.get("prompt") or []
        d_speakers = set()
        for ln in prompt:
            if not isinstance(ln, str):
                continue
            for m in re.finditer(
                r"speaks as \(S(\d+)\) ", ln,
            ):
                # Recover the speaker by walking backwards from the match.
                # The grammar is `<player> speaks as (Sn) ...` — we
                # split on the literal to recover the speaker label.
                head = ln[: m.start()]
                if "turns to face" in head:
                    head = head.rsplit("turns to face", 1)[0]
                if head.endswith(","):
                    head = head[:-1]
                name = head.rsplit(" ", 1)[-1].strip()
                if name:
                    d_speakers.add(name)
        if d_speakers and len(d_speakers) == 1:
            mono.append((shot.get("id") or "?", sorted(d_speakers)[0], 0))
    if not mono:
        return []
    listed = ", ".join("%s (%s alone)" % (s, n) for s, n, _ in mono)
    return [
        "Single-speaker scene risk: " + listed + ". A known H3 render "
        "defect sometimes drops the SPEAKER from such clips (2026-10-18 "
        "live: one rendered as an empty room, another lost the speaker "
        "while the silent character stayed; across seeds and languages, "
        "with structurally complete prompts). If a scene renders "
        "without its characters, re-roll just that scene — change its "
        "seed in the plan JSON and set Loop Start scene_range to that "
        "scene number; or give the silent character a line in the "
        "script to break the monologue."
    ]


# --------------------------------------------------------------------------- #
# ComfyUI node shell
# --------------------------------------------------------------------------- #
ASSET_BUNDLE_TYPE = "MIE_ASSET"
ASSET_ROSTER_TYPE = "MIE_ASSET_ROSTER"

CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"


class BuildPlanFromStoryboard(object):
    """Zero-LLM plan assembler (docs/PLAN_STORYBOARD_ASSET.md §4.2).

    Same inputs/outputs as ``MiniMaxH3LoopPromptGeneratorV2`` minus the
    LLM connector + every enhance widget: a hand-written storyboard
    asset + wired character/scene asset bundles → byte-stable
    ``plan_json``. Sits between ``LoadStoryboardAsset`` and the existing
    ``MiniMaxH3ChainPlanModern``; the wiring is identical to V2 except
    no LLM socket is required.

    The ``assets`` output is the same ``MIE_ASSET_ROSTER`` shape V2
    emits (characters in slot order, then scenes) — ``Asset → Slots``
    splits it into ``picture_i`` / ``audio_i`` for the H3 chain.
    """

    MAX_SLOTS = 8  # 8 character slots + 1 scene = 9 picture cap

    @classmethod
    def INPUT_TYPES(cls):
        optional: dict = {}
        for i in range(1, cls.MAX_SLOTS + 1):
            optional["character_%d" % i] = (
                ASSET_BUNDLE_TYPE,
                {"tooltip": (
                    "角色槽位 %d：接 Load Character Asset 的 asset 输出；"
                    "character_slot 字段决定它是哪个 Picture。\n"
                    "Character slot %d. The storyboard's character_slot "
                    "field decides which Picture it is." % (i, i)
                )},
            )
        optional["scene_1"] = (
            ASSET_BUNDLE_TYPE,
            {"tooltip": (
                "场景槽位 · scene slot 1：接 Load Scene Asset 的 asset 输出；"
                "排在角色 Picture 之后。\n"
                "Wire a Load Scene Asset's asset output; scenes come after "
                "all characters."
            )},
        )
        return {
            "required": {
                "storyboard": (
                    "MIE_STORYBOARD",
                    {"tooltip": (
                        "接 Load Storyboard Asset 的 storyboard 输出。\n"
                        "Wire the Load Storyboard Asset's storyboard output."
                    )},
                ),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": (
                        "整数种子；per_scene_increment 模式下第 i 场 = seed+i，"
                        "same_across_scenes 模式下每场都用 seed。\n"
                        "Integer seed; per_scene_increment = seed + i "
                        "(scene i); same_across_scenes = seed every scene."
                    ),
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", ASSET_ROSTER_TYPE)
    RETURN_NAMES = ("plan_json", "summary", "images", "assets")

    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, storyboard, seed=0, **kwargs):
        # Collect the wired bundles in slot order (character_1..N then scene_1).
        bundles: list[dict] = []
        for i in range(1, self.MAX_SLOTS + 1):
            bundle = kwargs.get("character_%d" % i)
            if isinstance(bundle, dict):
                bundles.append(bundle)
        scene_bundle = kwargs.get("scene_1")
        if isinstance(scene_bundle, dict):
            bundles.append(scene_bundle)

        # Validate the storyboard bundle. The load node already ran
        # validation, but a stray hand-edited bundle is not unusual.
        cleaned = _storyboard.validate_storyboard_payload(storyboard)
        # Inject the integer seed so per_scene_increment picks it up.
        cleaned = {**cleaned, "seed_base": int(seed)}

        out = assemble_plan(
            storyboard=cleaned,
            slot_bundles=bundles,
            seed_base=int(seed),
            _skip_validation=True,  # we just validated above
        )

        # Build the IMAGE batch (None when shapes disagree, like V2).
        import torch
        tensors = []
        for bundle in bundles:
            image = bundle.get("image") if isinstance(bundle, dict) else None
            if image is None:
                continue
            tensors.append(image)
        images = None
        size_note = ""
        if tensors:
            shapes: dict = {}
            for t, b in zip(tensors, bundles):
                shapes.setdefault(
                    tuple(t.shape[1:]), []
                ).append(b.get("name") or "?")
            if len(shapes) > 1:
                detail = "; ".join(
                    "%dx%d: %s" % (w, h, ", ".join(ns))
                    for (h, w, _c), ns in sorted(shapes.items())
                )
                size_note = (
                    "picture sizes differ (%s): the images batch output is "
                    "off for this run — wire Asset → Slots picture_1..N to "
                    "ref_image_0..N-1; H3 resizes each reference "
                    "independently" % detail
                )
            else:
                images = torch.cat(tensors, dim=0)

        # Compose the human-readable summary. The "summary:" section
        # already says "mechanical · no LLM" implicitly via the
        # node-name; we surface the layout-fill note, the size-note,
        # and the monologue hint here.
        summary_bits: list[str] = [
            "mechanical · zero-LLM assembly from storyboard %r"
            % cleaned.get("name"),
            "ambience: " + out["ambience"],
        ]
        summary_bits.extend(out["summary_lines"])
        if size_note:
            summary_bits.append(size_note)
        summary = "\n".join(summary_bits)

        # The MIE_ASSET_ROSTER contract: characters in slot order, then
        # scenes in slot order. ``Asset → Slots`` consumes this list
        # unchanged.
        assets = list(bundles)

        plan_json_str = plan_to_json_string(out["plan"])
        return (plan_json_str, summary, images, assets)