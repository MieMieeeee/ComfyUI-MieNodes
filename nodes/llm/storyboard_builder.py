# -*- coding: utf-8 -*-
"""Zero-LLM plan assembler for the storyboard asset track
(docs/PLAN_STORYBOARD_ASSET.md).

Why this module exists:
  The H3 Loop pipeline already does the heavy lifting of stamping a
  six-section Ref2VA prompt (subject / summary / retention / detail /
  sound / music) from a roster of cast members and a sequence of
  dialogue lines. ``build_fallback_shot_lines`` + the speech-installer
  helpers were always the deterministic part — they DO NOT touch an LLM.
  What this module does is take a hand-written storyboard JSON (cast,
  shots, optional spatial layout, optional English Setting), assemble
  one identical-shape prompt per shot with no model in the loop, and
  emit the same dict the Longform-V Plan node accepts.

Hard rules (mirroring PLAN_STORYBOARD_ASSET §1 + §2):
  * No LLM calls. No connector sockets. No ``_invoke`` / ``enhance_user_input``.
  * No translation of locked text. The verbatim line ``text`` becomes the
    verbatim ``<d>[language] text</d>`` payload, byte for byte.
  * No re-authoring. The English ``setting_line``, ``picture`` cues and
    ``prompt_prefix`` go in as the user wrote them — a hostile printer
    (CJK in an English-only field) is rejected at the JSON layer.

Slot semantics (PLAN §4.2 — explicitly NOT compacting empty sockets):
  ``character_slot`` is an ABSOLUTE picture number 1..8 (or 1..N for the
  characters the storyboard actually uses). When the cast references
  slots {1, 3} but the user wires only ``character_1`` and ``character_3``
  with ``character_2`` empty, the empty slot is a HARD ERROR — silently
  binding slot 2 to whatever is wired at socket 3 would produce a
  correct-looking plan with the WRONG character bound to slot 2 (a silent
  mis-binding; reviewed 2026-10-19). The build node collects every
  socket (including None) and ``assemble_plan`` raises with the exact
  empty slot number before any prompt text is generated.

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
        PACING_PRESETS,
        estimate_shot_budget,
        group_budgets_into_scenes,
    )
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompts import (
        DEFAULT_MUSIC_LINE,
        SCHEMA_SIX,
        SIX_SECTION_FIELDS,
        SECTION_SUBJECT,
        SECTION_SUMMARY,
        SECTION_RETENTION,
        SECTION_DETAIL,
        SECTION_SOUND,
        SECTION_MUSIC,
        _replace_section_body,
        _shot1_coverage,
        annotate_retention_pose,
        build_fallback_shot_lines,
        build_plan,
        build_spatial_layout_directive,
        complete_layout_from_turn_order,
        identity_pin_lines,
        neutralize_offscreen_claims,
        plan_to_json_string,
        replace_description_body,
        validate_dialogue_invariantity,
        validate_label_policy,
        validate_plan,
        validate_spatial_layout_invariant,
    )
except ImportError:  # pragma: no cover
    from .dialogue_segmenter import (  # type: ignore
        DialogueTurn,
        Pacing,
        PACING_PRESETS,
        estimate_shot_budget,
        group_budgets_into_scenes,
    )
    from .minimax_h3_loop_prompts import (  # type: ignore
        DEFAULT_MUSIC_LINE,
        SCHEMA_SIX,
        SIX_SECTION_FIELDS,
        SECTION_SUBJECT,
        SECTION_SUMMARY,
        SECTION_RETENTION,
        SECTION_DETAIL,
        SECTION_SOUND,
        SECTION_MUSIC,
        _replace_section_body,
        _shot1_coverage,
        annotate_retention_pose,
        build_fallback_shot_lines,
        build_plan,
        build_spatial_layout_directive,
        complete_layout_from_turn_order,
        identity_pin_lines,
        neutralize_offscreen_claims,
        plan_to_json_string,
        replace_description_body,
        validate_dialogue_invariantity,
        validate_label_policy,
        validate_plan,
        validate_spatial_layout_invariant,
    )


# --------------------------------------------------------------------------- #
# Camera-discipline fallback — three distinct English ambience lines the
# builder uses when BOTH the storyboard ``ambience`` AND the scene
# ``ambience`` are empty. PLAN §3.1 says the discipline picks one of three
# constants; v1 keeps all three visibly different so the choice is not
# silently a no-op.
# --------------------------------------------------------------------------- #
AMBIENCE_FALLBACKS = {
    "locked": (
        "The room tone stays quiet and even throughout, with no "
        "non-diegetic sound beyond the spoken lines."
    ),
    "no_closeups": (
        "The ambient bed of the room carries through every shot; no "
        "extra foley enters when the framing changes."
    ),
    "auto": (
        "The room tone follows the natural ambience of the setting; "
        "soft handling and fabric sounds ride underneath the dialogue."
    ),
}

# `<d>` tag prefix from the storyboard's ``language`` field (PLAN §3.1
# "每个 <d> 标签都用它：zh → [Chinese]，en → [English]。不按字面检测语言").
DIALOGUE_TAG = {"zh": "[Chinese]", "en": "[English]"}


# --------------------------------------------------------------------------- #
# Shot packing
# --------------------------------------------------------------------------- #
def _lines_per_shot(storyboard: dict):
    """Normalise the storyboard into ``[[{speaker, text, picture?}, ...], ...]``.

    Three input shapes are accepted (mutually exclusive — exactly one
    must produce a non-None result):

    1. ``shots`` present (preferred): each shot's ``lines`` is preserved
       verbatim — the author owns the cut.
    2. ``lines`` present (top-level flat list of ``{speaker, text,
       picture?}``): packed via the dialogue_segmenter's minimal-cut
       rule (``group_budgets_into_scenes`` with
       ``max_lines_per_scene=2``) — PLAN §3.1.
    3. both absent: hard error.
    """
    raw = storyboard.get("shots")
    if raw is not None:
        return [list(shot["lines"]) for shot in raw]
    flat = storyboard.get("lines")
    if flat is None:
        raise ValueError(
            "storyboard must declare either 'shots: [...]' (authored) or "
            "'lines: [...]' (auto-packed by the minimal-cut rule)"
        )
    return _auto_pack_flat_lines(flat, storyboard)


def _auto_pack_flat_lines(
    flat_lines: list[dict],
    storyboard: dict,
) -> list[list[dict]]:
    """Auto-pack a flat ``lines`` list into scenes (PLAN §3.1).

    The dialogue_segmenter's ``estimate_shot_budget`` + ``group_budgets_into_scenes``
    chain expects ``DialogueTurn`` objects. We construct ONE turn per
    line (the storyboard has no fixed turn grouping), let the estimator
    measure each line's runtime, then pack 2 lines per scene.
    """
    if not isinstance(flat_lines, list) or not flat_lines:
        raise ValueError(
            "storyboard.lines must be a non-empty list of "
            "{speaker, text, picture?} entries"
        )
    pacing_key = storyboard.get("pacing", "normal")
    if pacing_key not in PACING_PRESETS:
        raise ValueError(
            "unknown pacing %r (storyboard validator should have rejected "
            "this earlier)" % pacing_key
        )
    pacing = PACING_PRESETS[pacing_key]

    turns: list[DialogueTurn] = []
    for entry in flat_lines:
        if not isinstance(entry, dict):
            raise ValueError(
                "storyboard.lines entries must be objects, got %s"
                % type(entry).__name__
            )
        speaker = entry.get("speaker")
        text = entry.get("text")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(
                "storyboard.lines entry missing 'speaker': %r" % entry
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "storyboard.lines entry missing 'text': %r" % entry
            )
        turns.append(DialogueTurn(speaker=speaker.strip(),
                                  lines=[text]))

    budgets, _report = estimate_shot_budget(turns, pacing)
    scenes = group_budgets_into_scenes(
        budgets, pacing, max_lines_per_scene=2,
    )
    out: list[list[dict]] = []
    for scene in scenes:
        lines_for_scene: list[dict] = []
        for budget in scene:
            # Budget.lines is the per-line TEXT list (we constructed one
            # turn per flat entry); recover the original flat entry so
            # ``picture`` cues ride along.
            for line_text in budget.lines:
                for entry in flat_lines:
                    if entry.get("text") == line_text:
                        lines_for_scene.append(dict(entry))
                        break
        if lines_for_scene:
            out.append(lines_for_scene)
    return out


# --------------------------------------------------------------------------- #
# Mechanical shot builder (the zero-LLM heart of this module)
# --------------------------------------------------------------------------- #
def _speaker_to_slot(cast: dict[str, dict]) -> dict[str, int]:
    """``{speaker_name: character_slot}``. Built once per build call —
    drives ``(Sn)`` tag assignment in the speech sentences and the
    bundle ordering later."""
    return {name: int(info["character_slot"]) for name, info in cast.items()}


def _build_manifest(
    storyboard: dict,
    used_slots: list[int],
    slot_to_bundle: dict[int, dict | None],
) -> list[dict]:
    """Build the six-section manifest the prompt helpers expect.

    ``used_slots`` is the sorted list of picture numbers actually
    occupied by the wired bundles (1..N for characters + scene). Each
    contributes ``{"slot": "Picture N", "about": <asset description>,
    "role": "identity" or "environment"}``.

    The cast's declared ``character_slot`` values are NOT trusted here
    — by the time ``_build_manifest`` runs, ``slot_to_bundle`` has
    already been validated to contain every slot the cast needs.
    """
    picture_specs: list[tuple[int, dict, str]] = []
    for slot in sorted(used_slots):
        bundle = slot_to_bundle.get(slot)
        if bundle is None:
            continue  # already reported by the earlier empty-slot check
        role = (
            "identity" if bundle.get("kind") == "character" else "environment"
        )
        picture_specs.append((slot, bundle, role))

    manifest: list[dict] = []
    for pic_slot, bundle, role in picture_specs:
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


def _shot_length(lines: list[dict], pacing: Pacing) -> int:
    """Length of one user-authored shot, measured by the dialogue
    segmenter's per-shot budget.

    ``estimate_shot_budget`` may split a single turn into sub-budgets
    when the packed runtime overflows the 14 s window. The zero-LLM
    track keeps the lines on a single shot (the assembler accepts a
    longer length), so the returned length is the SUM of all
    sub-budget frames — picking ``[0].rounded_length_frames`` would drop
    the tail."""
    if not lines:
        return 5  # smallest valid H3 length (24 fps; 17k+5 grid → 5)
    speaker = lines[0]["speaker"]
    turn = DialogueTurn(
        speaker=speaker, lines=[ln["text"] for ln in lines],
    )
    per_turn, _report = estimate_shot_budget([turn], pacing)
    return sum(int(b.rounded_length_frames) for b in per_turn)


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
    """Build one shot's schema, segments, length and seed. No LLM.

    The body is composed by ``_build_storyboard_shot_body`` (this
    function is a thin wrapper). We deliberately bypass
    ``install_shot_speech`` because its cue branch wraps a non-still
    cue as "Silent picture only. Mouths stay closed and nobody
    speaks." — a contradiction when the cue is on a SPEAKING line.
    See PLAN_STORYBOARD_ASSET §5 (deviation note) for the rationale.
    """
    prompt_lines = build_fallback_shot_lines(
        SCHEMA_SIX, manifest=manifest, spatial_layout=spatial_layout,
    )
    # Pin the resolved ambience (storyboard > scene > camera-discipline
    # fallback) into ``overall_soundscape`` BEFORE the body is composed —
    # the speech-path logic preserves the soundscape (it only replaces
    # the section body in voiceover mode, which we never use).
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

    pacing = PACING_PRESETS[storyboard.get("pacing", "normal")]
    length = _shot_length(lines, pacing)

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
    """Compose the six-section shot body. The plan §5 deviation note:
    ``install_shot_speech`` wraps a non-still picture cue as "Silent
    picture only. Mouths stay closed and nobody speaks.", which
    contradicts the upcoming speaking line. This helper mirrors
    ``install_shot_speech``'s ref2va path but emits the picture cue
    as a plain descriptive sentence."""
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

    prompt_lines = replace_description_body(
        prompt_lines, body, schema=SCHEMA_SIX,
    )

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
# Picture-numbering convention: characters occupy CONTIGUOUS slots
# 1..N (validated by ``validate_cast`` + re-checked in ``assemble_plan``);
# the scene sits at ``max(cast_slots) + 1``. The contiguity rule is
# enforced at three layers — JSON save, JSON load + assemble, and
# build-time — so a non-contiguous cast fails fast rather than silently
# producing wrong-character references (review 2026-10-19 R2).
SCENE_SLOT_OFFSET = 0  # scene sits at ``max(cast_slots) + 1``.


def _slot_layout(
    cast: dict[str, dict],
    max_character_slot: int,
) -> tuple[list[int], int]:
    """The full set of occupied slots: every character slot used by
    cast, then ``scene`` at ``max(cast_slots) + 1`` (or 1 if no
    characters). Returns ``(used_slots, scene_slot)``."""
    cast_slots = sorted({int(v["character_slot"]) for v in cast.values()})
    scene_slot = (max(cast_slots) if cast_slots else 0) + 1
    used = list(cast_slots) + [scene_slot]
    return used, scene_slot


def assemble_plan(
    *,
    storyboard: dict,
    slot_bundles: dict[int, dict | None],
    seed_base: int = 0,
    summary_extra: Optional[list[str]] = None,
    _skip_validation: bool = False,
) -> dict:
    """Pure function: storyboard JSON + slot→bundle map → plan dict.

    ``slot_bundles`` is keyed by picture number:

    * ``1..MAX_CHARACTER_SLOT`` → character bundle or ``None``;
    * the scene bundle lives at the FIRST unused slot past the cast's
      max (or ``1`` when there are no characters). The build node
      surfaces the scene bundle at index ``MAX_CHARACTER_SLOT + 1``
      (i.e., socket ``scene_1``) and ``assemble_plan`` relocates it
      here so the picture-numbering contract (``scene = max(cast) +
      1``) is enforced uniformly.

    Every character slot the cast declares must be filled. Every
    wired slot must be either a character (1..MAX) or the scene at
    ``max(cast) + 1``. Any mismatch is a loud error.

    Returns ``{"plan": {...}, "summary_lines": [...], "ambience": str,
    "size_note": ""}``.
    """
    if not isinstance(storyboard, dict):
        raise ValueError("storyboard bundle must be a dict")
    if not isinstance(slot_bundles, dict):
        raise ValueError("slot_bundles must be a dict keyed by picture number")
    if not _skip_validation:
        storyboard = _storyboard.validate_storyboard_payload(storyboard)

    cast = storyboard.get("cast") or {}
    if not cast:
        raise ValueError(
            "storyboard has no cast; the builder cannot assemble a "
            "plan without speakers"
        )
    speaker_to_slot = _speaker_to_slot(cast)
    cast_slots = sorted({int(v["character_slot"]) for v in cast.values()})
    if len(cast_slots) != len(cast):
        raise ValueError("duplicate character_slot values in cast")
    # Required contiguous 1..N — see review 2026-10-19 R2 (gap-cast
    # silent mis-binding): a non-contiguous cast makes the plan's
    # absolute ``<Picture N>`` / ``(Sn)`` numbering diverge from
    # ``Asset → Slots``' compressed picture_i / audio_i indexing, so
    # identity text rides on the wrong reference image and voice on
    # the wrong audio. Refuse at the source.
    expected_slots = list(range(1, len(cast_slots) + 1))
    if cast_slots != expected_slots:
        raise ValueError(
            "cast character_slots must be contiguous 1..N (got %s); "
            "non-contiguous numbering makes <Picture N> in the plan "
            "point to a different asset than the Asset → Slots wiring, "
            "so the speaker's identity text would ride on the wrong "
            "reference image and the voice on the wrong audio. Re-number "
            "the cast or fill in the missing slot"
            % cast_slots
        )
    if max(cast_slots) > _storyboard.MAX_CHARACTER_SLOT:
        raise ValueError(
            "cast uses character_slot %d; the upper limit is %d"
            % (max(cast_slots), _storyboard.MAX_CHARACTER_SLOT)
        )

    # Resolve the picture-numbering layout (where the scene sits).
    used_slots, scene_slot = _slot_layout(cast, _storyboard.MAX_CHARACTER_SLOT)

    # Strip a scene bundle out of slot_bundles if the build node
    # placed it at MAX+1 — relocate it to the actual scene slot so
    # the picture numbering stays coherent.
    scene_bundle_raw = slot_bundles.get(_storyboard.MAX_CHARACTER_SLOT + 1)
    if scene_bundle_raw is not None and scene_bundle_raw.get("kind") == "scene":
        slot_bundles = dict(slot_bundles)
        slot_bundles.pop(_storyboard.MAX_CHARACTER_SLOT + 1, None)
        slot_bundles[scene_slot] = scene_bundle_raw

    # Every cast slot must be wired (HARD ERROR — no compaction).
    for slot in cast_slots:
        bundle = slot_bundles.get(slot)
        if bundle is None:
            raise ValueError(
                "character_slot %d is declared by cast (%s) but no asset "
                "is wired into socket 'character_%d' on the build node; "
                "either wire the asset, or remove the slot from the cast. "
                "Empty slots in the middle of a wiring are a hard error "
                "(PLAN_STORYBOARD_ASSET §4.2 — silent re-binding would "
                "produce wrong-character <Picture N> tags)"
                % (slot, sorted(cast.keys()), slot)
            )

    # Voice / audio checks per character bundle.
    slot_to_voice: dict[int, str] = {}
    for slot in cast_slots:
        bundle = slot_bundles[slot]
        if bundle.get("kind") != "character":
            raise ValueError(
                "character_slot %d expected a character bundle, got kind=%r"
                % (slot, bundle.get("kind"))
            )
        slot_to_voice[slot] = _check_character_audio(bundle, slot)

    # Scene check: exactly one scene bundle at the scene slot.
    scene_bundle = slot_bundles.get(scene_slot)
    if scene_bundle is None or scene_bundle.get("kind") != "scene":
        raise ValueError(
            "the zero-LLM builder needs exactly one scene wire at "
            "Picture %d; got %r"
            % (scene_slot, scene_bundle.get("kind") if scene_bundle else None)
        )
    scene_bundles = [scene_bundle]

    manifest = _build_manifest(storyboard, used_slots, slot_bundles)

    # Spatial layout: from the storyboard; missing speakers are filled
    # in turn-order via the existing helper (PLAN §3.1 "缺的说话人用
    # complete_layout_from_turn_order 补左/右/中").
    turn_speaker_order: list[str] = []
    for sl in storyboard.get("shots") or storyboard.get("lines") or []:
        lines_in = sl.get("lines") if "lines" in sl else [sl]
        for ln in lines_in:
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

    ambience = _resolve_ambience(storyboard, scene_bundles)

    # Build the shots. Shots are explicit, or auto-packed from a top-level
    # ``lines`` field (PLAN §3.1).
    raw_shots_in = storyboard.get("shots")
    flat_lines_in = storyboard.get("lines")
    if raw_shots_in is None and flat_lines_in is None:
        raise ValueError(
            "storyboard must declare either 'shots: [...]' (authored) or "
            "'lines: [...]' (auto-packed by the minimal-cut rule)"
        )
    if raw_shots_in is not None and flat_lines_in is not None:
        raise ValueError(
            "storyboard carries both 'shots' and 'lines'; provide exactly "
            "one — 'shots' for an authored cut, 'lines' for auto-packing"
        )
    if raw_shots_in is not None:
        shots_per_scene = [list(shot["lines"]) for shot in raw_shots_in]
        raw_shots = raw_shots_in
    else:
        # Auto-pack path. We still need a raw_shots view for the dialogue
        # invariantity check downstream — synthesise it from the packed
        # scenes.
        shots_per_scene = _auto_pack_flat_lines(flat_lines_in, storyboard)
        raw_shots = [{"lines": s} for s in shots_per_scene]

    shot_entries: list[dict] = []
    for shot_idx, sl_lines in enumerate(shots_per_scene, 1):
        shot = _assemble_one_shot(
            shot_index=shot_idx,
            lines=sl_lines,
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
            speaker=sl_lines[0]["speaker"],
            lines=[ln["text"] for ln in sl_lines],
        )
        for sl_lines in shots_per_scene
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

    summary_lines: list[str] = []
    if layout_was_filled:
        summary_lines.append(
            "layout filled by turn order (no entry in spatial_layout for "
            "every cast speaker)"
        )
    # Monologue-scene hint (mirrors the LLM track's `_monologue_scene_hint`).
    cast_size = len(cast)
    mono_lines = _monologue_hint_from_shots(raw_shots, cast_size)
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


def _monologue_hint_from_shots(
    raw_shots: list[dict],
    cast_size: int,
) -> list[str]:
    """Run-summary advice for single-speaker scenes. Mirrors the LLM
    track's ``_monologue_scene_hint`` (2026-10-18 LAN forensics:
    continuation clips with ONE speaker sometimes drop the speaker
    from the frame). Uses the storyboard's authored structure
    directly — no regex reverse-engineering of finished prompt text,
    which breaks on multi-word speaker names.

    The plan §5 deviation note (vocative turning) is unrelated: the
    hint inspects the storyboard's pre-assembly shape, so it works
    even when the assembled speech sentence rewrites the speaker's
    head."""
    if cast_size < 2:
        return []
    mono: list[tuple[str, str, int]] = []
    for shot_idx, sl in enumerate(raw_shots or [], 1):
        lines = sl.get("lines") if isinstance(sl, dict) and "lines" in sl else [sl]
        speakers = {
            str(ln.get("speaker") or "").strip()
            for ln in lines
            if ln.get("speaker")
        }
        speakers = {s for s in speakers if s}
        if lines and len(speakers) == 1:
            mono.append((
                "scene_%02d" % shot_idx,
                sorted(speakers)[0],
                len(lines),
            ))
    if not mono:
        return []
    listed = ", ".join(
        "%s (%s speaks all %d line(s))" % (sid, spk, n)
        for sid, spk, n in mono
    )
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

# The scene bundle is parked at MAX + 1 inside the socket dict. The
# ``assemble_plan`` caller relocates it to ``max(cast_slots) + 1``.
SCENE_SOCKET_INDEX = 0  # placeholder; actual value set on the class.


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
        # Collect the wired bundles into a slot-keyed dict. Empty
        # sockets are PRESERVED as None — ``assemble_plan`` raises when
        # a cast slot is empty (PLAN §4.2 "不把空洞压紧"); non-empty
        # sockets keep their character_slot identity.
        slot_bundles: dict[int, dict | None] = {}
        for i in range(1, self.MAX_SLOTS + 1):
            slot_bundles[i] = kwargs.get("character_%d" % i)
        # Park the scene bundle at MAX + 1; ``assemble_plan`` relocates
        # it to ``max(cast_slots) + 1`` once it knows the cast.
        slot_bundles[self.MAX_SLOTS + 1] = kwargs.get("scene_1")

        # Validate the storyboard bundle. The load node already ran
        # validation, but a stray hand-edited bundle is not unusual.
        cleaned = _storyboard.validate_storyboard_payload(storyboard)
        # Inject the integer seed so per_scene_increment picks it up.
        cleaned = {**cleaned, "seed_base": int(seed)}

        out = assemble_plan(
            storyboard=cleaned,
            slot_bundles=slot_bundles,
            seed_base=int(seed),
            _skip_validation=True,  # we just validated above
        )

        # The MIE_ASSET_ROSTER contract: characters in slot order, then
        # the scene. We walk the resolved slot dict (so empty sockets
        # stay out of the roster) and emit one entry per occupied slot.
        roster: list[dict] = []
        bundles_for_images: list[dict] = []
        cast_slots = sorted({int(v["character_slot"])
                             for v in cleaned["cast"].values()})
        max_cast_slot = max(cast_slots) if cast_slots else 0
        scene_slot = max_cast_slot + 1
        for slot in cast_slots:
            bundle = slot_bundles.get(slot)
            if isinstance(bundle, dict):
                roster.append(bundle)
                bundles_for_images.append(bundle)
        scene_bundle = slot_bundles.get(scene_slot)
        if isinstance(scene_bundle, dict):
            roster.append(scene_bundle)
            bundles_for_images.append(scene_bundle)

        # Build the IMAGE batch (None when shapes disagree, like V2).
        import torch
        tensors = []
        for bundle in bundles_for_images:
            image = bundle.get("image") if isinstance(bundle, dict) else None
            if image is None:
                continue
            tensors.append(image)
        images = None
        size_note = ""
        if tensors:
            shapes: dict = {}
            for t, b in zip(tensors, bundles_for_images):
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

        plan_json_str = plan_to_json_string(out["plan"])
        return (plan_json_str, summary, images, roster)