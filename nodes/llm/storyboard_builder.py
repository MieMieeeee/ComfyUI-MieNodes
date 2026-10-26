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
        _insert_section_suffix,
        _replace_section_body,
        _shot1_coverage,
        annotate_retention_pose,
        absent_declarations,
        build_fallback_shot_lines,
        build_plan,
        build_spatial_layout_directive,
        complete_layout_from_turn_order,
        identity_pin_lines,
        neutralize_offscreen_claims,
        offscreen_speech_line,
        install_presence_speech,
        on_screen_speech_line,
        plan_to_json_string,
        pov_shot1_coverage,
        presence_summary,
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
        _insert_section_suffix,
        _replace_section_body,
        _shot1_coverage,
        annotate_retention_pose,
        absent_declarations,
        build_fallback_shot_lines,
        build_plan,
        build_spatial_layout_directive,
        complete_layout_from_turn_order,
        identity_pin_lines,
        neutralize_offscreen_claims,
        offscreen_speech_line,
        install_presence_speech,
        on_screen_speech_line,
        plan_to_json_string,
        pov_shot1_coverage,
        presence_summary,
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

    ``used_slots`` is the sorted list of PICTURE numbers actually
    occupied (on-screen characters, then the scene). Each contributes
    ``{"slot": "Picture N", "about": <asset description>, "role":
    "identity" or "environment"}``.

    Off-screen speakers are NOT in the manifest: they are never
    rendered, so they take no picture and no Subject. Their voice — if
    an asset is wired — is bound through the ``(S<n>)`` / audio channel
    instead (2026-10-20 ruling: the off-screen asset is optional and
    contributes its VOICE, not a picture).
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
    absent: Optional[list[str]] = None,
    offscreen_names: Optional[set] = None,
    speaker_ids: Optional[dict[str, str]] = None,
    speaker_descriptions: Optional[dict[str, str]] = None,
) -> dict:
    """Build one shot's schema, segments, length and seed. No LLM.

    The body is composed by ``_build_storyboard_shot_body`` (this
    function is a thin wrapper). Presence boards use the shared
    ``install_presence_speech``. The legacy installer's cue branch wraps a non-still
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

    offscreen = set(offscreen_names or ())
    all_names = sorted(speaker_to_slot)
    # Identity text comes from the ASSET description, not from the
    # manifest: an off-screen speaker has no picture (so no manifest
    # entry) but still carries a description — the place where the
    # "never render me" instruction lives.
    descs = dict(speaker_descriptions or {})
    speaker_identities = {
        name: descs.get(name, "") for name in all_names
    }
    prompt_lines = _build_storyboard_shot_body(
        prompt_lines,
        lines=lines,
        speaker_to_slot=speaker_to_slot,
        slot_to_voice=slot_to_voice,
        spatial_layout=spatial_layout,
        speaker_identities=speaker_identities,
        storyboard=storyboard,
        absent=absent,
        offscreen_names=offscreen,
        speaker_ids=speaker_ids,
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
    absent: Optional[list[str]] = None,
    offscreen_names: Optional[set] = None,
    speaker_ids: Optional[dict[str, str]] = None,
) -> list[str]:
    """Compose the six-section shot body. The plan §5 deviation note:
    ``install_shot_speech`` wraps a non-still picture cue as "Silent
    picture only. Mouths stay closed and nobody speaks.", which
    contradicts the upcoming speaking line. This helper mirrors
    ``install_shot_speech``'s ref2va path but emits the picture cue
    as a plain descriptive sentence.

    Presence (PLAN_POV_OFFSCREEN §4): ``offscreen_names`` / ``absent``
    default to empty, and with both empty every branch below collapses to
    the pre-POV path, so a non-POV board assembles to identical bytes.

    ``speaker_ids`` maps a speaker to its ``(S<n>)`` tag. The tag is the
    AUDIO position (the n-th wired character), not the character_slot:
    an off-screen speaker's asset is optional (2026-10-20 ruling), and
    leaving it unwired would otherwise leave a hole in the audio list
    that H3 renumbers around — which would break the (S<n>) ↔
    ref_audio_N binding. A speaker with no entry here (an unwired
    off-screen voice) gets NO tag and NO voice descriptor.
    """
    offscreen = set(offscreen_names or ())
    absent_list = list(absent or ())
    pov_board = bool(offscreen)
    ids = dict(speaker_ids or {})

    # Presence is STRUCTURAL, so the anti-POV scrubber must not assert
    # "all speaking characters are visible" on a board that declared an
    # off-screen voice (§4.5).
    prompt_lines = neutralize_offscreen_claims(
        prompt_lines, declared_offscreen=pov_board,
    )

    # Per-shot presence set = cast members neither absent nor off-screen.
    all_names = sorted(speaker_to_slot)
    present_names = [
        n for n in all_names
        if n not in set(absent_list) and n not in offscreen
    ]
    visible_names = present_names
    # Identity pins / bare-name aliases stay FULL CAST on purpose
    # (§4.3): omitting an identity anchor is the historical defect
    # source, and the whole reference chain is wired anyway. An
    # off-screen speaker WITH an asset keeps his alias — the asset
    # description is where the "never render me" instruction lives.
    pin_names = [
        n for n in all_names
        if n not in offscreen or speaker_identities.get(n)
    ]
    if pin_names and not (offscreen or absent_list):
        pins = identity_pin_lines(pin_names, speaker_identities)
        if pins:
            # APPEND the pins, never REPLACE the section body. The body's
            # manifest lines are the identity anchors —
            # ``<Subject K> -> <Picture M>: <description>`` and
            # ``<Picture N>: <scene description>`` — and they are what
            # tells H3 which reference image is which character and what
            # the room is. Replacing the body dropped both, so H3 had
            # no room description at all (it invented one) and no
            # subject→picture binding (it followed the prose instead of
            # the reference image). This is the same append the LLM
            # track does in ``install_shot_speech``.
            prompt_lines = _insert_section_suffix(
                prompt_lines, "subject_definitions:", pins
            )

    board_lang = storyboard.get("language", "en")
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
        sid = ids.get(speaker, "")
        text = ln["text"]
        # Per-line language override (PLAN_POV_OFFSCREEN §3.3) — the
        # board value is the default, so a board with no ``lang`` keys
        # produces exactly the tag it always did.
        tag = DIALOGUE_TAG.get(
            ln.get("lang") or board_lang, DIALOGUE_TAG[board_lang]
        )
        if speaker in offscreen:
            # An unwired off-screen voice declares NO voice at all and
            # lets H3 choose one.
            speech_lines.append(offscreen_speech_line(
                sid, voice if sid else "", text, tag,
                listener=(present_names[0] if present_names else ""),
                reaction=(ln.get("picture") or "").strip(),
                listeners=present_names,
            ))
        else:
            speech_lines.append(on_screen_speech_line(
                speaker, sid, voice, text, tag,
                pov_board=pov_board,
            ))

    if offscreen or absent_list:
        return install_presence_speech(
            prompt_lines, speech_lines, cast_names=all_names,
            line_speakers=[ln["speaker"] for ln in lines],
            offscreen_names=offscreen, absent=absent_list,
            spatial_layout=spatial_layout, speaker_identities=speaker_identities,
            camera_perspective=storyboard.get("camera_perspective", "observer"),
            picture_cues=[(ln.get("picture") or "").strip() for ln in lines],
            opening_setting=[storyboard["setting_line"]] if storyboard.get("setting_line") else [],
        )
    body = [_shot1_coverage(visible_names, voiceover=False)]
    # The spatial pin only names people who are actually in frame.
    shot_layout = {
        n: pos for n, pos in (spatial_layout or {}).items()
        if n in set(present_names)
    }
    spatial = build_spatial_layout_directive(shot_layout)
    if spatial and not spatial.startswith("(no spatial"):
        body.append(spatial)
    if storyboard.get("setting_line"):
        body.append("Setting: " + storyboard["setting_line"])
    for ln, speech in zip(lines, speech_lines):
        cue = (ln.get("picture") or "").strip()
        if cue:
            body.append(_picture_cue_sentence(cue))
        body.append(speech)

    prompt_lines = replace_description_body(
        prompt_lines, body, schema=SCHEMA_SIX,
    )

    summary = presence_summary(
        on_screen_names=present_names,
        speaking_names=sorted(
            {ln["speaker"] for ln in lines
             if ln.get("speaker") and ln["speaker"] not in offscreen}
        ),
        offscreen_names=sorted(
            {ln["speaker"] for ln in lines
             if ln.get("speaker") and ln["speaker"] in offscreen}
        ),
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


def _missing_slot_message(
    *,
    slot: int,
    cast: dict,
    cast_slots: list[int],
    slot_bundles: dict,
    scene_slot: int,
) -> str:
    """The error for a cast slot with no asset wired (PLAN §4.2).

    It must show the SLOT ORDER, never an alphabetical name list: the
    whole point of the message is "put this asset in this socket", and a
    sorted list of names reads as a slot order that is not the one the
    storyboard declares. It also lists what IS currently wired, because
    the usual cause is an asset sitting in the wrong socket (the author
    re-ordered the cast and moved one wire but not the other).
    """
    owner = next(
        (name for name, info in cast.items()
         if int(info["character_slot"]) == slot),
        "?",
    )
    table = []
    for s in sorted(cast_slots):
        who = next(
            (n for n, i in cast.items() if int(i["character_slot"]) == s),
            "?",
        )
        offscreen = next(
            (bool(i.get("offscreen")) for n, i in cast.items()
             if int(i["character_slot"]) == s),
            False,
        )
        mark = "  <-- MISSING" if s == slot else ""
        table.append("character_%d = %s%s%s" % (
            s, who, "（画外音 off-screen）" if offscreen else "", mark,
        ))
    table.append("character_%d = (scene)" % scene_slot)
    wired = []
    for s in sorted(slot_bundles):
        bundle = slot_bundles[s]
        if not isinstance(bundle, dict):
            continue
        if s == scene_slot:
            wired.append("character_%d = %s (scene)" % (
                s, bundle.get("name") or "?",
            ))
        else:
            wired.append("character_%d = %s" % (s, bundle.get("name") or "?"))
    return (
        "character_slot %d (%s) is declared by the storyboard's cast, but "
        "no asset is wired into socket 'character_%d' on the build node. "
        "The storyboard's cast, in SLOT ORDER, is: %s. Currently wired: %s. "
        "Wire the matching asset into 'character_%d' (the socket number IS "
        "the storyboard's character_slot — an asset in the wrong socket is "
        "a different character), or drop that slot from the cast. "
        "Only the sockets listed above are used; every other character "
        "socket may stay empty. "
        "Empty slots are a hard error on purpose (PLAN_STORYBOARD_ASSET "
        "§4.2 — silently re-binding would put a character's identity "
        "text and voice on the wrong <Picture N>)."
        % (
            slot, owner, slot,
            "; ".join(table),
            ("; ".join(wired) if wired else "(nothing wired)"),
            slot,
        )
    )


def _offscreen_slots(cast: dict) -> set[int]:
    """The character_slots whose cast member is declared off-screen."""
    return {
        int(info["character_slot"])
        for info in cast.values()
        if isinstance(info, dict) and info.get("offscreen")
    }


def _assemble_authored_plan(storyboard: dict, slot_bundles: dict) -> dict:
    """Keep an authored H3 prompt intact; references are an independent choice."""
    cast = storyboard["cast"]
    if not cast:
        raise ValueError("authored storyboard requires a cast")
    ordered = sorted(cast.items(), key=lambda item: int(item[1]["character_slot"]))
    roster_slots, picture_slots = [], []
    generated = storyboard["audio_mode"] == "generated"
    for name, info in ordered:
        slot = int(info["character_slot"])
        bundle = slot_bundles.get(slot)
        if bundle is None and info.get("offscreen") and (generated or not storyboard.get("_structured_events")):
            continue
        if not isinstance(bundle, dict) or bundle.get("kind") != "character":
            raise ValueError("character_%d (%s) requires a character asset" % (slot, name))
        if not info.get("offscreen") and bundle.get("image") is None:
            raise ValueError("character_%d (%s) requires an image" % (slot, name))
        if not generated:
            _check_character_audio(bundle, slot)
        roster_slots.append(slot)
        if not info.get("offscreen"):
            picture_slots.append(slot)
    scene = slot_bundles.get(_storyboard.MAX_CHARACTER_SLOT + 1)
    scene_slot = None
    if scene is not None:
        if not isinstance(scene, dict) or scene.get("kind") != "scene" or scene.get("image") is None:
            raise ValueError("scene_1 requires a scene asset with an image")
        scene_slot = _storyboard.MAX_CHARACTER_SLOT + 1
        roster_slots.append(scene_slot)
        picture_slots.append(scene_slot)
    reference_context = []
    summary_lines = ["authored prompt preserved; audio mode: " + storyboard["audio_mode"]]
    if not generated and storyboard.get("_structured_events"):
        reference_context.extend(
            "Voice casting: (S%d) uses the voice identity from <Audio %d> only; the reference words are not dialogue."
            % (index, index) for index in range(1, len(ordered) + 1)
        )
    if scene_slot is not None:
        scene_picture = len(picture_slots)
        character_pictures = ", ".join(
            "<Picture %d>" % index
            for index, slot in enumerate(picture_slots, 1) if slot != scene_slot
        )
        reference_context += [
            "Use %s for the characters' identities and appearance only; "
            "do not use their backgrounds." % character_pictures,
        ]
        for name, info in ordered:
            slot = int(info["character_slot"])
            if info.get("offscreen"):
                continue
            picture = picture_slots.index(slot) + 1
            reference_context.append(
                "%s is the on-screen character shown in <Picture %d>; "
                "use <Picture %d> for %s's identity and appearance throughout."
                % (name, picture, picture, name)
            )
        # A source-image caption may describe a wide establishing view. Keep
        # its camera/framing instructions out of the authored shot; H3 gets
        # the environment from the scene image itself.
        reference_context.append(
            "Scene reference: <Picture %d> provides only the environment, "
            "materials and lighting. It does not define camera position, "
            "shot size or character blocking." % scene_picture
        )
        if not storyboard.get("_structured_events"):
            reference_context.append(
                "Start immediately with the on-screen characters from %s already inside "
                "the environment from <Picture %d>, "
                "in the authored framing. Keep them continuously visible from the first frame "
                "through the end of the shot."
                % (character_pictures, scene_picture)
            )
        summary_lines.append(
            "optional scene %r -> <Picture %d>; character portrait backgrounds excluded"
            % (scene.get("name") or "?", scene_picture)
        )
    shots = []
    for index, shot in enumerate(storyboard["shots"], 1):
        # H3's temporal grid is 17*k+5 at 24 fps; nearest representable duration.
        frames = max(5, 17 * round((shot["duration_seconds"] * 24 - 5) / 17) + 5)
        shots.append({"id": "scene_%02d" % index, "prompt": shot["prompt"].split("\n"),
                      "length": frames, "seed": _seed_for(storyboard, index)})
    return {
        "plan": {"shots": shots, "prompt_prefix": reference_context},
        "summary_lines": summary_lines,
        "ambience": storyboard.get("ambience", ""), "size_note": "",
        "roster_slots": roster_slots, "picture_slots": picture_slots,
        "picture_of_slot": {slot: i for i, slot in enumerate(picture_slots, 1)},
        "scene_slot": scene_slot,
        "unwired_offscreen": [name for name, info in ordered
                               if info.get("offscreen") and slot_bundles.get(int(info["character_slot"])) is None],
    }


def assemble_plan(
    *,
    storyboard: dict,
    slot_bundles: dict[int, dict | None],
    seed_base: int = 0,
    summary_extra: Optional[list[str]] = None,
    _skip_validation: bool = False,
) -> dict:
    """Pure function: storyboard JSON + slot→bundle map → plan dict.

    ``slot_bundles`` is keyed by the storyboard's ``character_slot``:

    * ``1..MAX_CHARACTER_SLOT`` → character bundle or ``None``;
    * the scene bundle lives at the FIRST unused slot past the cast's
      max (or ``1`` when there are no characters). The build node
      surfaces the scene bundle at index ``MAX_CHARACTER_SLOT + 1``
      (i.e., socket ``scene_1``) and ``assemble_plan`` relocates it
      here so the picture-numbering contract stays coherent.

    TWO NUMBERING SPACES, and they are not the same thing any more
    (2026-10-20 ruling: the off-screen speaker's asset is optional and
    contributes its VOICE, not a picture):

    * **picture space** — ``<Picture N>`` / ``<Subject K>``: the
      ON-SCREEN characters in cast-slot order, then the scene. An
      off-screen speaker never appears here: he is never rendered, and
      handing H3 a picture of him is exactly the risk the placeholder
      card was invented to avoid. So a POV board really is one
      character plus one scene.
    * **voice space** — ``(S<n>)`` / ``ref_audio_N``: the WIRED
      characters in cast-slot order. ``(S<n>)`` is the AUDIO position,
      not the character_slot: leaving an off-screen asset unwired would
      otherwise punch a hole in the audio list, and H3 renumbers around
      a hole (so the plan's ``(S2)`` would land on the wrong clip).
      A board with every cast member wired — every non-POV board —
      has the two spaces identical, so its bytes do not move.

    Returns ``{"plan": {...}, "summary_lines": [...], "ambience": str,
    "size_note": ""}``.
    """
    if not isinstance(storyboard, dict):
        raise ValueError("storyboard bundle must be a dict")
    if not isinstance(slot_bundles, dict):
        raise ValueError("slot_bundles must be a dict keyed by picture number")
    if not _skip_validation:
        storyboard = _validate_storyboard_bundle(storyboard)
    if storyboard.get("version") == 2:
        from ..assets.storyboard_contract import to_runtime
        storyboard = to_runtime(storyboard)
    if storyboard.get("assembly_mode") == "authored":
        return _assemble_authored_plan(storyboard, slot_bundles)

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

    # Board-level off-screen declaration (PLAN_POV_OFFSCREEN §3.1).
    board_offscreen = {
        name for name, info in cast.items() if info.get("offscreen")
    }
    onscreen_slots = [s for s in cast_slots if s not in _offscreen_slots(cast)]
    offscreen_slots = _offscreen_slots(cast)

    # The scene sits at the first slot past the cast's max — it is a
    # SOCKET position, not a picture number.
    scene_slot = (max(cast_slots) if cast_slots else 0) + 1

    # Strip a scene bundle out of slot_bundles if the build node
    # placed it at MAX+1 — relocate it to the actual scene slot so
    # the socket positions stay coherent.
    scene_bundle_raw = slot_bundles.get(_storyboard.MAX_CHARACTER_SLOT + 1)
    if scene_bundle_raw is not None and scene_bundle_raw.get("kind") == "scene":
        slot_bundles = dict(slot_bundles)
        slot_bundles.pop(_storyboard.MAX_CHARACTER_SLOT + 1, None)
        slot_bundles[scene_slot] = scene_bundle_raw

    # Every ON-SCREEN cast slot must be wired (HARD ERROR — no
    # compaction). An OFF-SCREEN slot is optional: wired means "use
    # this voice", unwired means "declare no voice and let H3 choose".
    for slot in onscreen_slots:
        bundle = slot_bundles.get(slot)
        if bundle is None:
            raise ValueError(
                _missing_slot_message(
                    slot=slot,
                    cast=cast,
                    cast_slots=cast_slots,
                    slot_bundles=slot_bundles,
                    scene_slot=scene_slot,
                )
            )

    # Voice / audio checks per WIRED character bundle. An off-screen
    # slot with nothing wired is legal and simply contributes no voice.
    slot_to_voice: dict[int, str] = {}
    wired_character_slots: list[int] = []
    for slot in cast_slots:
        bundle = slot_bundles.get(slot)
        if bundle is None:
            if slot in offscreen_slots:
                continue  # optional: no voice declared, H3 chooses
            raise ValueError(
                "character_slot %d expected a character bundle, got nothing"
                % slot
            )
        if bundle.get("kind") != "character":
            raise ValueError(
                "character_slot %d expected a character bundle, got kind=%r"
                % (slot, bundle.get("kind"))
            )
        slot_to_voice[slot] = _check_character_audio(bundle, slot)
        wired_character_slots.append(slot)

    # VOICE SPACE: ``(S<n>)`` is the AUDIO position — the n-th WIRED
    # character in cast-slot order. Leaving an off-screen asset unwired
    # must not punch a hole in the audio list (H3 renumbers around a
    # hole, and the plan's ``(S2)`` would land on the wrong clip).
    speaker_ids: dict[str, str] = {}
    for i, slot in enumerate(wired_character_slots, 1):
        for name, info in cast.items():
            if int(info["character_slot"]) == slot:
                speaker_ids[name] = "S%d" % i

    # PICTURE SPACE: the ON-SCREEN characters in cast-slot order, then
    # the scene. An off-screen speaker never takes a picture.
    picture_slots = list(onscreen_slots)
    picture_of_slot: dict[int, int] = {
        slot: i for i, slot in enumerate(picture_slots, 1)
    }
    picture_of_slot[scene_slot] = len(picture_slots) + 1
    # The manifest is indexed by PICTURE number (1..N), so build it
    # against a picture-numbered view of the bundles.
    picture_bundles = {
        picture_of_slot[slot]: slot_bundles[slot] for slot in picture_slots
    }
    picture_bundles[picture_of_slot[scene_slot]] = slot_bundles.get(scene_slot)
    used_slots = list(range(1, len(picture_slots) + 2))

    # Scene check: exactly one scene bundle at the scene slot.
    scene_bundle = slot_bundles.get(scene_slot)
    if scene_bundle is None or scene_bundle.get("kind") != "scene":
        raise ValueError(
            "the zero-LLM builder needs exactly one scene wire at "
            "Picture %d; got %r"
            % (picture_of_slot[scene_slot],
               scene_bundle.get("kind") if scene_bundle else None)
        )
    scene_bundles = [scene_bundle]

    manifest = _build_manifest(storyboard, used_slots, picture_bundles)

    # Spatial layout: from the storyboard; missing speakers are filled
    # in turn-order via the existing helper (PLAN §3.1 "缺的说话人用
    # complete_layout_from_turn_order 补左/右/中").
    #
    # Off-screen speakers are EXCLUDED from completion (PLAN_POV_OFFSCREEN
    # §3.1): a frame pin is the instruction "this character is visible at
    # <position>", which is exactly what an off-screen voice contradicts.
    # The validator already refuses a hand-written pin for one; skipping
    # completion closes the same hole on the automatic path.
    board_offscreen = {
        name for name, info in cast.items() if info.get("offscreen")
    }
    turn_speaker_order: list[str] = []
    for sl in storyboard.get("shots") or storyboard.get("lines") or []:
        lines_in = sl.get("lines") if "lines" in sl else [sl]
        for ln in lines_in:
            if (ln["speaker"] not in turn_speaker_order
                    and ln["speaker"] not in board_offscreen):
                turn_speaker_order.append(ln["speaker"])
    # A silent cast member is still present unless explicitly absent.
    for name in cast:
        if name not in board_offscreen and name not in turn_speaker_order:
            turn_speaker_order.append(name)
    declared = {
        n: pos for n, pos in (storyboard.get("spatial_layout") or {}).items()
        if n not in board_offscreen
    }
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
        absent_per_scene = [
            list(shot.get("absent") or []) for shot in raw_shots_in
        ]
        raw_shots = raw_shots_in
    else:
        # Auto-pack path. We still need a raw_shots view for the dialogue
        # invariantity check downstream — synthesise it from the packed
        # scenes. Per-shot ``absent`` is an authored field with no
        # auto-pack equivalent, so every packed scene is fully present.
        shots_per_scene = _auto_pack_flat_lines(flat_lines_in, storyboard)
        raw_shots = [{"lines": s} for s in shots_per_scene]
        absent_per_scene = [[] for _ in shots_per_scene]

    # Board-level off-screen declaration (PLAN_POV_OFFSCREEN §3.1).
    offscreen_names = {
        name for name, info in cast.items() if info.get("offscreen")
    }

    # Identity text per speaker, from the wired bundles. An off-screen
    # speaker with an asset keeps his description (that is where the
    # "never render me" instruction lives); an unwired one has none.
    speaker_descriptions: dict[str, str] = {}
    for name, info in cast.items():
        slot = int(info["character_slot"])
        bundle = slot_bundles.get(slot)
        if isinstance(bundle, dict):
            speaker_descriptions[name] = (bundle.get("description") or "").strip()

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
            absent=absent_per_scene[shot_idx - 1],
            offscreen_names=offscreen_names,
            speaker_ids=speaker_ids,
            speaker_descriptions=speaker_descriptions,
        )
        shot_entries.append(shot)

    # Local trials showed misattributed lines. This is a diagnostic risk,
    # not proof of a language-based speaker assignment rule: audio inputs,
    # model sampling and ASR have not been isolated. Never demand a
    # bilingual rewrite or change the authored words as a workaround.
    language_warnings: list[str] = []
    if offscreen_names:
        board_lang = storyboard.get("language", "en")
        on_screen_names = {
            name for name in cast if name not in offscreen_names
        }
        for shot_idx, sl_lines in enumerate(shots_per_scene, 1):
            off_langs, on_langs = set(), set()
            for ln in sl_lines:
                lang = ln.get("lang") or board_lang
                (off_langs if ln["speaker"] in offscreen_names
                 else on_langs).add(lang)
            shared = sorted(off_langs & on_langs)
            if shared:
                names = sorted(
                    {ln["speaker"] for ln in sl_lines
                     if ln["speaker"] in offscreen_names}
                )
                language_warnings.append(
                    "scene_%02d: the off-screen line(s) of %s use the same "
                    "language (%s) as the on-screen character's line(s). H3 "
                    "has misattributed same-language lines in local trials; "
                    "this is an observed risk, not a language restriction. "
                    "Keep the authored dialogue and verify every line's "
                    "words, voice attribution and visible mouth movement."
                    % (shot_idx, ", ".join(names), ", ".join(shared))
                )
            # A NAME-VOCATIVE for the visible character at the head of an
            # off-screen line: H3 has the visible character in the scene,
            # so a leading name can be resolved as the SPEAKER. Live
            # 2026-10-20: with the vocative the line degraded to
            # invented English ("Claude, let me ask you about the origin
            # of this payment"); without it H3 produced the right
            # content in the voice's language. Rewording helps, but the
            # real requirement is a reference sample in the line's
            # language.
            for ln in sl_lines:
                if ln["speaker"] not in offscreen_names:
                    continue
                first = str(ln.get("text") or "").strip()
                for name in on_screen_names:
                    if first.startswith(name) and first[len(name):len(name) + 1] in (
                        ",", "，", " ", ":", "：", "!", "？", "?"
                    ):
                        language_warnings.append(
                            "scene_%02d: the off-screen line of %s starts "
                            "with the visible character's name %r. H3 may "
                            "resolve that name as the SPEAKER and then "
                            "invent the rest of the sentence (live "
                            "2026-10-20: \"Claude, help me resolve this "
                            "document.\" came back as \"Claude, let me ask "
                            "you about the origin of this payment\", while "
                            "the same line without the name came back "
                            "correct in the voice's language). Reword so the "
                            "line does not lead with her name."
                            % (shot_idx, ln["speaker"], name)
                        )
                        break

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
    summary_lines.extend(language_warnings)
    if offscreen_names:
        summary_lines.append(
            "Reference audio language and voice attribution are not verified "
            "by the asset schema. Check every rendered line against the "
            "authored text and its intended speaker."
        )
    # An unwired off-screen speaker declares no voice at all — that is a
    # legal state, but it is worth saying out loud, because the voice
    # then drifts between renders.
    if out_unwired := [
        name for name, info in sorted(
            cast.items(), key=lambda kv: int(kv[1]["character_slot"]),
        )
        if info.get("offscreen")
        and not isinstance(
            slot_bundles.get(int(info["character_slot"])), dict
        )
    ]:
        summary_lines.append(
            "Off-screen voice without an asset: %s. No voice descriptor "
            "and no (S<n>) tag were written for %s, so H3 picks the timbre "
            "itself and it may differ between renders. Wire an asset into "
            "that character's socket to pin the voice."
            % (", ".join(out_unwired), "it" if len(out_unwired) == 1 else "them")
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
        # The SOCKET positions that make up the roster: the wired
        # characters in cast-slot order, then the scene. This is the
        # VOICE space — the off-screen speaker is in it (his audio is
        # needed) even though he has no picture.
        "roster_slots": list(wired_character_slots) + [scene_slot],
        # The PICTURE space: the on-screen characters, then the scene.
        "picture_slots": list(picture_slots) + [scene_slot],
        # Socket position -> picture number.
        "picture_of_slot": picture_of_slot,
        "scene_slot": scene_slot,
        # The off-screen speakers that were left unwired (their voice
        # was left to H3) — the run summary says so.
        "unwired_offscreen": [
            name for name, info in sorted(
                cast.items(), key=lambda kv: int(kv[1]["character_slot"]),
            )
            if info.get("offscreen")
            and not isinstance(slot_bundles.get(int(info["character_slot"])), dict)
        ],
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

MAX_CHARACTER_SLOTS = 8  # 8 characters + 1 scene = the H3 9-picture cap
# Board kinds the zero-LLM track emits. Storyboard v1 is dialogue-only
# (PLAN_STORYBOARD_FIRST_ASSET_V3 §4.1); the literal exists so a
# downstream consumer sees a stable string rather than a missing key.
BOARD_KIND_DIALOGUE = "dialogue"


def _storyboard_optional_slots(max_slots: int) -> dict:
    """The ``character_1..N`` + ``scene_1`` socket block, shared verbatim
    by ``BuildPlanFromStoryboard`` and ``MiniMaxH3LoopPromptGeneratorV3``
    so the two nodes are wire-compatible."""
    optional: dict = {}
    for i in range(1, max_slots + 1):
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
            "all characters. Optional for assembly_mode=authored."
        )},
    )
    return optional


def _storyboard_required_inputs() -> dict:
    """``storyboard`` + ``seed``. Shared by both node classes."""
    return {
        "storyboard": (
            "MIE_STORYBOARD",
            {"tooltip": (
                "接 Load Storyboard Asset 的 storyboard 输出。\n"
                "Wire the Load Storyboard Asset's storyboard output. "
                "assembly_mode=authored preserves the complete shot prompt; "
                "audio_mode=generated sends images without reference audio."
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
    }


def collect_slot_bundles(kwargs: dict, max_slots: int) -> dict[int, dict | None]:
    """Build the slot-keyed bundle dict from the node's socket kwargs.

    Empty sockets are PRESERVED as ``None`` — ``assemble_plan`` raises
    when a cast-declared slot is unwired (PLAN §4.2 「不把空洞压紧」),
    so a gap in the middle is a loud error, never a silent re-bind. The
    scene bundle is parked at ``max_slots + 1``; ``assemble_plan``
    relocates it to ``max(cast_slots) + 1`` once it knows the cast.
    """
    slot_bundles: dict[int, dict | None] = {}
    for i in range(1, max_slots + 1):
        slot_bundles[i] = kwargs.get("character_%d" % i)
    slot_bundles[max_slots + 1] = kwargs.get("scene_1")
    return slot_bundles


def _validate_storyboard_bundle(bundle):
    # Load/Save nodes attach a local path to their socket value. It is
    # transport metadata, never part of the portable JSON contract.
    if isinstance(bundle, dict) and "_path" in bundle:
        bundle = {key: value for key, value in bundle.items() if key != "_path"}
    return _storyboard.validate_storyboard_payload(bundle)


def assemble_from_sockets(
    *,
    storyboard: dict,
    kwargs: dict,
    seed: int,
    max_slots: int,
) -> dict:
    """The shared body of both zero-LLM nodes (PLAN §4.2 / V3 §4.2).

    Wraps ``assemble_plan`` and the deterministic tail — roster in slot
    order, the IMAGE batch (with the mixed-size caveat), and the run
    summary. Returns::

        {"plan_json": str, "summary": str, "images": Tensor|None,
         "assets": list, "storyboard": cleaned_dict}

    Both ``BuildPlanFromStoryboard`` and ``MiniMaxH3LoopPromptGeneratorV3``
    call this, which is what makes their ``plan_json`` byte-identical for
    the same inputs (V3 §6.2).
    """
    slot_bundles = collect_slot_bundles(kwargs, max_slots)

    # Validate the storyboard bundle. The load node already ran
    # validation, but a stray hand-edited bundle is not unusual.
    cleaned = _validate_storyboard_bundle(storyboard)
    if cleaned.get("version") == 2:
        from ..assets.storyboard_contract import to_runtime
        cleaned = to_runtime(cleaned)
    # Inject the integer seed so per_scene_increment picks it up.
    cleaned = {**cleaned, "seed_base": int(seed)}

    out = assemble_plan(
        storyboard=cleaned,
        slot_bundles=slot_bundles,
        seed_base=int(seed),
        _skip_validation=True,  # we just validated above
    )

    # The MIE_ASSET_ROSTER contract: the WIRED characters in cast-slot
    # order, then the scene. The resolved slots come back from
    # ``assemble_plan`` (it is the one place that knows which slots
    # exist), so empty sockets stay out of the roster.
    #
    # An off-screen character IS in the roster — his audio is needed —
    # but his PICTURE is dropped: he is never rendered, and handing H3
    # a picture of him is the exact risk the placeholder card existed to
    # avoid. ``Asset → Slots`` skips a picture-less roster entry when
    # it numbers picture_i, so the picture list stays "on-screen
    # characters + scene" while the audio list stays "every voiced
    # character".
    roster: list[dict] = []
    bundles_for_images: list[dict] = []
    offscreen_slots = {
        int(info["character_slot"])
        for info in cleaned["cast"].values() if info.get("offscreen")
    }
    for slot in out["roster_slots"]:
        bundle = slot_bundles.get(slot)
        if not isinstance(bundle, dict):
            # The scene lives at ``max(cast)+1`` on assemble_plan's
            # internal copy; on OUR copy it is still parked at
            # ``max_slots + 1``. Resolve it by kind.
            bundle = next(
                (
                    b for b in slot_bundles.values()
                    if isinstance(b, dict) and b.get("kind") == "scene"
                ),
                None,
            )
        if not isinstance(bundle, dict):
            continue
        if cleaned.get("audio_mode") == "generated":
            bundle = dict(bundle, audio=None)
        if slot in offscreen_slots:
            # Voice only: keep the bundle (and its audio) but drop the
            # picture.
            entry = dict(bundle)
            entry["image"] = None
            roster.append(entry)
            continue
        roster.append(bundle)
        bundles_for_images.append(bundle)

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

    # Compose the human-readable summary. `mechanical` only ever appears
    # in this STRING, never in the plan text (PLAN §2).
    summary_bits: list[str] = [
        "mechanical · zero-LLM assembly from storyboard %r"
        % cleaned.get("name"),
        "ambience: " + out["ambience"],
    ]
    summary_bits.extend(out["summary_lines"])
    if size_note:
        summary_bits.append(size_note)
    summary = "\n".join(summary_bits)

    return {
        "plan_json": plan_to_json_string(out["plan"]),
        "summary": summary,
        "images": images,
        "assets": roster,
        "storyboard": cleaned,
    }


def storyboard_cache_key(storyboard: dict) -> str:
    """Content hash of a storyboard BUNDLE for ``IS_CHANGED``.

    The bundle is a plain dict, so the canonical JSON form is stable
    (sorted keys, no whitespace) — editing a widget on the Load node
    without touching the file does not bust the cache, but any change to
    the storyboard's actual content does.
    """
    import hashlib
    import json as _json
    try:
        payload = _json.dumps(
            storyboard, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        )
    except Exception:
        return _library.MISSING_CACHE_MARKER
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    MAX_SLOTS = MAX_CHARACTER_SLOTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _storyboard_required_inputs(),
            "optional": _storyboard_optional_slots(cls.MAX_SLOTS),
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", ASSET_ROSTER_TYPE)
    RETURN_NAMES = ("plan_json", "summary", "images", "assets")

    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, storyboard, seed=0, **kwargs):
        out = assemble_from_sockets(
            storyboard=storyboard,
            kwargs=kwargs,
            seed=int(seed),
            max_slots=self.MAX_SLOTS,
        )
        return (
            out["plan_json"],
            out["summary"],
            out["images"],
            out["assets"],
        )


class MiniMaxH3LoopPromptGeneratorV3(object):
    """V3 generator facade over the zero-LLM storyboard track
    (docs/PLAN_STORYBOARD_FIRST_ASSET_V3.md §4).

    Shape-compatible with ``MiniMaxH3LoopPromptGeneratorV2``: same five
    outputs in the same order and types (``plan_json``, ``summary``,
    ``board_kind``, ``images``, ``assets``), same CATEGORY, same
    self-growing ``character_N`` sockets. The difference is that V3 has
    NO LLM connector and no prose — the storyboard asset IS the script,
    so swapping a V2 node for a V3 node (rewiring only the ``storyboard``
    line) moves a workflow onto the storyboard track without touching
    anything downstream of ``plan_json``.

    ``board_kind`` is the stable ``"dialogue"`` literal (storyboard v1 is
    dialogue-only). It exists so downstream wiring written against V2's
    five-output shape keeps working unchanged.
    """

    MAX_SLOTS = MAX_CHARACTER_SLOTS
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _storyboard_required_inputs(),
            "optional": _storyboard_optional_slots(cls.MAX_SLOTS),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", ASSET_ROSTER_TYPE)
    RETURN_NAMES = ("plan_json", "summary", "board_kind", "images", "assets")

    FUNCTION = "generate_v3"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Storyboard content hash + every wired asset bundle's content.

        A Load-storyboard node already hashes its file, so re-queueing
        after an edit re-runs that side; this side additionally covers a
        user re-picking a DIFFERENT asset into an already-wired slot
        (the asset's own Load node hashes its files too, but the roster
        it feeds is a bundle, so hashing the bundle here keeps the
        generator honest when someone rewires from the canvas).
        """
        storyboard = kwargs.get("storyboard")
        parts = [storyboard_cache_key(storyboard)]
        for i in range(1, cls.MAX_SLOTS + 1):
            parts.append(_bundle_fingerprint(kwargs.get("character_%d" % i)))
        parts.append(_bundle_fingerprint(kwargs.get("scene_1")))
        return "|".join(parts)

    def generate_v3(self, storyboard, seed=0, **kwargs):
        out = assemble_from_sockets(
            storyboard=storyboard,
            kwargs=kwargs,
            seed=int(seed),
            max_slots=self.MAX_SLOTS,
        )
        return (
            out["plan_json"],
            out["summary"],
            BOARD_KIND_DIALOGUE,
            out["images"],
            out["assets"],
        )


def _bundle_fingerprint(bundle) -> str:
    """Stable identity of one wired asset bundle for IS_CHANGED.

    The text fields (name / description / voice / ambience) decide what
    lands in the plan; the tensors are deliberately excluded — a re-render
    of the same portrait changes pixels without changing a single word of
    ``plan_json``, and busting the cache on pixels would re-run the
    assembler for nothing.
    """
    if bundle is None:
        return "-"
    if not isinstance(bundle, dict):
        return _library.MISSING_CACHE_MARKER
    return "\x1f".join([
        str(bundle.get("kind") or ""),
        str(bundle.get("name") or ""),
        str(bundle.get("description") or ""),
        str(bundle.get("voice") or ""),
        str(bundle.get("ambience") or ""),
        "audio" if bundle.get("audio") is not None else "-",
    ])
