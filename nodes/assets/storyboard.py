# -*- coding: utf-8 -*-
"""On-disk storyboard asset (docs/PLAN_STORYBOARD_ASSET.md).

A storyboard is the AUTHORED INPUT to the zero-LLM plan assembler
(``nodes/llm/storyboard_builder.py``): a hand-written cast of speakers, a
sequence of dialogue shots, an optional English Setting sentence, and a
multi-line English prompt prefix. The schema lives in this module —
``LoadStoryboardAsset`` / ``SaveStoryboardAsset`` and the builder both read
and write through the helpers below.

This module is filesystem-only: no ComfyUI import, no torch, no
``nodes.llm``. Validation here mirrors the rules in PLAN_STORYBOARD_ASSET
§3.1 (length limits, CJK bans on English-only fields, slot uniqueness,
shot count caps) so the builder can rely on the JSON being well-formed.

Directory layout (a third kind alongside characters / scenes):

    <library_dir>/storyboards/<name>/storyboard.json

A storyboard has NO images / audio (it is the SCRIPT, not the cast), so
``library.py``'s ``kind must be character or scene`` branch is untouched
and the storyboard kind lives in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

try:
    from . import library as _library
except ImportError:  # pragma: no cover - direct-script path
    import library as _library


STORYBOARDS_DIRNAME = "storyboards"
STORYBOARD_JSON_NAME = "storyboard.json"

STORYBOARD_VERSION = 1

# Field length caps. The Setting / picture cue fields are English-only by
# contract (the assembly node inserts them verbatim into the H3 prompt);
# names (cast / speaker labels) follow the asset whitelist so they can
# never collide with directory traversal.
NAME_MAX_CHARS = _library.NAME_MAX_CHARS
PROMPT_PREFIX_MAX_LINES = 32
PROMPT_PREFIX_LINE_MAX_CHARS = 400
SETTING_LINE_MAX_CHARS = 400
PICTURE_CUE_MAX_CHARS = 400
AMBIENCE_MAX_CHARS = _library.AMBIENCE_MAX_CHARS

# Hard caps for the cast / shots. H3 caps at 9 pictures per scene; one
# of those is the scene slot, so a single scene can hold 0..8 characters.
MAX_CHARACTER_SLOT = 8
MAX_SHOT_LINES = 2  # Per §3.1 "shots can hold 1..2 lines".

# Pacing presets map directly to dialogue_segmenter.PACING_PRESETS keys.
PACING_CHOICES = ("fast", "normal", "slow")

CAMERA_DISCIPLINE_CHOICES = ("locked", "no_closeups", "auto")
SEED_POLICY_CHOICES = ("per_scene_increment", "same_across_scenes")

# Language codes that drive the <d> tag prefix.
LANGUAGE_CHOICES = ("zh", "en")
DIALOGUE_TAG = {"zh": "[Chinese]", "en": "[English]"}


# --------------------------------------------------------------------------- #
# Field validators (each returns the cleaned value or raises ValueError)
# --------------------------------------------------------------------------- #
_CJK_RE = re.compile(r"[\u3400-\u9FFF]")


def _require_clean_english(value, field, max_chars):
    """Validate a single-line English field. Empty is legal unless the
    field is required (caller checks ``not value`` separately). CJK is
    rejected because the value goes into an H3 prompt verbatim.

    Accepts the cleaned value as-is (a non-empty string passes through)
    — this keeps save-storyboard roundtrips idempotent."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(
            "%s must be a string, got %s" % (field, type(value).__name__)
        )
    stripped = value.strip()
    if len(stripped) > max_chars:
        raise ValueError(
            "%s longer than %d characters (%d given)"
            % (field, max_chars, len(stripped))
        )
    if _CJK_RE.search(stripped):
        raise ValueError(
            "%s must be English (no CJK characters); got %r" % (field, stripped)
        )
    return stripped


def validate_prompt_prefix(text):
    """``prompt_prefix`` is a multi-line English string. Returns the list of
    non-empty lines (in order). Empty / whitespace-only input → empty list.
    A line over 400 chars or with CJK chars is rejected — the assembly
    node does not retouch this text, only joins it into prompt_prefix[].

    Accepts EITHER the raw author string (multi-line, embedded ``\n``)
    or an already-cleaned list of non-empty strings. The dual-input
    shape lets the save node roundtrip through ``validate_storyboard_payload``
    without choking on a cleaned payload."""
    if isinstance(text, list):
        out: list[str] = []
        for i, raw in enumerate(text):
            line = str(raw).strip() if raw is not None else ""
            if not line:
                continue
            if _CJK_RE.search(line):
                raise ValueError(
                    "prompt_prefix line %d must be English; got %r"
                    % (i + 1, line)
                )
            if len(line) > PROMPT_PREFIX_LINE_MAX_CHARS:
                raise ValueError(
                    "prompt_prefix line %d longer than %d characters"
                    % (i + 1, PROMPT_PREFIX_LINE_MAX_CHARS)
                )
            out.append(line)
            if len(out) > PROMPT_PREFIX_MAX_LINES:
                raise ValueError(
                    "prompt_prefix has more than %d non-empty lines"
                    % PROMPT_PREFIX_MAX_LINES
                )
        return out
    if not isinstance(text, str):
        raise ValueError(
            "prompt_prefix must be a string, got %s" % type(text).__name__
        )
    out = []
    for i, raw in enumerate(text.splitlines() or [text]):
        line = raw.strip()
        if not line:
            continue
        if _CJK_RE.search(line):
            raise ValueError(
                "prompt_prefix line %d must be English; got %r" % (i + 1, line)
            )
        if len(line) > PROMPT_PREFIX_LINE_MAX_CHARS:
            raise ValueError(
                "prompt_prefix line %d longer than %d characters"
                % (i + 1, PROMPT_PREFIX_LINE_MAX_CHARS)
            )
        out.append(line)
        if len(out) > PROMPT_PREFIX_MAX_LINES:
            raise ValueError(
                "prompt_prefix has more than %d non-empty lines"
                % PROMPT_PREFIX_MAX_LINES
            )
    return out


def validate_setting_line(value):
    """``setting_line``: optional English one-liner. Empty string is the
    "no Setting: line in plan" marker."""
    if value is None:
        return ""
    return _require_clean_english(
        value, "setting_line", SETTING_LINE_MAX_CHARS,
    )


def validate_ambience_line(value):
    """Optional one-line English ambience. Same constraints as the
    Setting line but kept as a separate helper so the test surface stays
    symmetric with ``library.validate_ambience``."""
    if value is None:
        return ""
    return _require_clean_english(
        value, "ambience", AMBIENCE_MAX_CHARS,
    )


def validate_picture_cue(value):
    """Optional English action sentence on a single line. Empty / None
    means the line carries no <cue>."""
    if value is None:
        return ""
    return _require_clean_english(
        value, "lines[].picture", PICTURE_CUE_MAX_CHARS,
    )


def validate_language(value):
    if not isinstance(value, str):
        raise ValueError(
            "language must be a string, got %s" % type(value).__name__
        )
    lang = value.strip().lower()
    if lang not in LANGUAGE_CHOICES:
        raise ValueError(
            "language must be one of %s, got %r"
            % (LANGUAGE_CHOICES, value)
        )
    return lang


def validate_camera_discipline(value):
    if not isinstance(value, str):
        raise ValueError(
            "camera_discipline must be a string, got %s"
            % type(value).__name__
        )
    cd = value.strip().lower()
    if cd not in CAMERA_DISCIPLINE_CHOICES:
        raise ValueError(
            "camera_discipline must be one of %s, got %r"
            % (CAMERA_DISCIPLINE_CHOICES, value)
        )
    return cd


def validate_pacing(value):
    if not isinstance(value, str):
        raise ValueError(
            "pacing must be a string, got %s" % type(value).__name__
        )
    p = value.strip().lower()
    if p not in PACING_CHOICES:
        raise ValueError(
            "pacing must be one of %s, got %r" % (PACING_CHOICES, value)
        )
    return p


def validate_seed_policy(value):
    if not isinstance(value, str):
        raise ValueError(
            "seed_policy must be a string, got %s"
            % type(value).__name__
        )
    p = value.strip().lower()
    if p not in SEED_POLICY_CHOICES:
        raise ValueError(
            "seed_policy must be one of %s, got %r"
            % (SEED_POLICY_CHOICES, value)
        )
    return p


def validate_line_lang(value, *, where):
    """``lines[].lang`` — the per-line dialogue-language override
    (PLAN_POV_OFFSCREEN §3.3).

    Absent / ``None`` means "inherit the board's ``language``"; the
    cleaned payload then omits the key entirely so a board that never
    used a per-line override round-trips byte-identically. An explicit
    value must be one of ``LANGUAGE_CHOICES`` — the same two codes the
    board-level field accepts, because the ``<d>`` tag is derived from
    exactly this pair."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "%s.lang must be a string, got %s" % (where, type(value).__name__)
        )
    lang = value.strip().lower()
    if lang not in LANGUAGE_CHOICES:
        raise ValueError(
            "%s.lang must be one of %s, got %r" % (where, LANGUAGE_CHOICES, value)
        )
    return lang


def validate_absent(raw_absent, cast, offscreen_names):
    """``shots[].absent`` — who is NOT in the frame for this shot
    (PLAN_POV_OFFSCREEN §3.2).

    Every name must be a cast member, and an ``offscreen`` speaker may
    never be listed (an off-screen voice is not "absent" — it is heard;
    the two are different declarations and conflating them would drop the
    speaker's own speech line). Returns a sorted list; ``[]`` when the
    key is absent or empty (the common case, which keeps the cleaned
    payload byte-identical to a non-POV board)."""
    if raw_absent is None:
        return []
    if not isinstance(raw_absent, list):
        raise ValueError(
            "shots[].absent must be a list of cast names, got %s"
            % type(raw_absent).__name__
        )
    out: list[str] = []
    for entry in raw_absent:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                "shots[].absent entries must be non-empty strings, got %r" % (entry,)
            )
        name = _library.validate_name(entry)
        if name not in cast:
            raise ValueError(
                "shots[].absent contains %r, which is not in cast %s"
                % (name, sorted(cast.keys()))
            )
        if name in offscreen_names:
            raise ValueError(
                "shots[].absent contains %r, which is an offscreen speaker; "
                "an off-screen voice is HEARD, not absent — drop it from "
                "'absent' (the cast's 'offscreen': true already says it "
                "is not seen)" % (name,)
            )
        if name not in out:
            out.append(name)
    return out


def validate_cast(cast):
    """``{speaker_name: {"character_slot": N, "offscreen": bool?}}``.

    ``character_slot``: 1..MAX_CHARACTER_SLOT, unique across speakers,
    CONTIGUOUS 1..N, every speaker name valid by the asset whitelist (no
    path separators, no dots, max NAME_MAX_CHARS).

    ``offscreen`` (PLAN_POV_OFFSCREEN §3.1, default False): the speaker is
    heard but never seen — first-person POV dialogue. It is a DECLARATION,
    never inferred: presence is structural data, and the semantic-facts
    extractor / the prose enhancer never guess it.

    Contiguity is enforced here (not just at assemble time) because a
    non-contiguous cast silently makes the plan's ``<Picture N>`` /
    ``(Sn)`` numbering diverge from ``Asset → Slots``' compressed
    picture_i / audio_i indexing — review 2026-10-19 R2 caught this
    silently producing wrong-character references.
    """
    if not isinstance(cast, dict) or not cast:
        raise ValueError(
            "cast must be a non-empty dict of speaker → {character_slot: N}"
        )
    used_slots: set[int] = set()
    cleaned: dict[str, dict] = {}
    for raw_name, raw_info in cast.items():
        name = _library.validate_name(raw_name)
        if not isinstance(raw_info, dict):
            raise ValueError(
                "cast[%r] must be an object with 'character_slot', got %s"
                % (raw_name, type(raw_info).__name__)
            )
        slot = raw_info.get("character_slot")
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise ValueError(
                "cast[%r].character_slot must be an int, got %r"
                % (raw_name, slot)
            )
        if slot < 1 or slot > MAX_CHARACTER_SLOT:
            raise ValueError(
                "cast[%r].character_slot %d out of range 1..%d"
                % (raw_name, slot, MAX_CHARACTER_SLOT)
            )
        if slot in used_slots:
            raise ValueError(
                "cast[%r].character_slot %d is used by more than one speaker; "
                "every slot must be unique" % (raw_name, slot)
            )
        used_slots.add(slot)
        offscreen = raw_info.get("offscreen", False)
        if not isinstance(offscreen, bool):
            raise ValueError(
                "cast[%r].offscreen must be a bool, got %r" % (raw_name, offscreen)
            )
        entry = {"character_slot": slot}
        if offscreen:
            # Stored only when true so a cleaned payload is byte-identical
            # to a board that never used POV (byte-stability red line,
            # PLAN_POV_OFFSCREEN §2).
            entry["offscreen"] = True
        cleaned[name] = entry
    sorted_slots = sorted(used_slots)
    expected = list(range(1, len(sorted_slots) + 1))
    if sorted_slots != expected:
        raise ValueError(
            "cast character_slots must be contiguous 1..N (got %s); "
            "non-contiguous numbering makes <Picture N> in the plan "
            "point to a different asset than the Asset → Slots wiring, "
            "so the speaker's identity text would ride on the wrong "
            "reference image and the voice on the wrong audio. Re-number "
            "the cast or fill in the missing slot"
            % sorted_slots
        )
    return cleaned


def validate_shots(shots, cast=None, offscreen_names=frozenset()):
    """Each shot is a dict with ``lines`` (1..MAX_SHOT_LINES entries).
    Every speaker must be in ``cast``; every line carries text (verbatim,
    no CJK ban — dialogue language is the storyboard's ``language`` and
    characters can speak CJK in a ``zh`` board). Each line MAY carry
    ``picture`` (English action sentence) and ``lang`` (per-line
    dialogue-language override, PLAN_POV_OFFSCREEN §3.3).

    Each shot MAY carry ``absent`` (who is not in frame this shot,
    PLAN_POV_OFFSCREEN §3.2). The key is dropped from the cleaned output
    when it is absent/empty so a non-POV board round-trips byte-identically.

    ``cast`` / ``offscreen_names`` are optional so the pure shape check
    still works standalone; when a cast IS supplied, every shot must
    leave at least one on-screen character in frame (an all-empty shot is
    the known empty-room render defect, so it is refused at authoring
    time rather than discovered in the video).
    """
    if not isinstance(shots, list):
        raise ValueError(
            "shots must be a list, got %s" % type(shots).__name__
        )
    if not shots:
        raise ValueError(
            "shots: [] is empty; omit 'shots' to auto-pack instead of "
            "passing an empty array"
        )
    cast = cast or {}
    offscreen_names = offscreen_names or frozenset()
    cleaned: list[dict] = []
    for shot_idx, raw_shot in enumerate(shots, 1):
        if not isinstance(raw_shot, dict):
            raise ValueError(
                "shot %d must be an object, got %s"
                % (shot_idx, type(raw_shot).__name__)
            )
        raw_lines = raw_shot.get("lines")
        if not isinstance(raw_lines, list) or not raw_lines:
            raise ValueError(
                "shot %d.lines must be a non-empty list" % shot_idx
            )
        if len(raw_lines) > MAX_SHOT_LINES:
            raise ValueError(
                "shot %d has %d lines; per-shot cap is %d"
                % (shot_idx, len(raw_lines), MAX_SHOT_LINES)
            )
        cleaned_lines = []
        for line_idx, raw_line in enumerate(raw_lines, 1):
            if not isinstance(raw_line, dict):
                raise ValueError(
                    "shot %d line %d must be an object, got %s"
                    % (shot_idx, line_idx, type(raw_line).__name__)
                )
            speaker = raw_line.get("speaker")
            text = raw_line.get("text")
            if not isinstance(speaker, str) or not speaker.strip():
                raise ValueError(
                    "shot %d line %d: 'speaker' must be a non-empty string"
                    % (shot_idx, line_idx)
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "shot %d line %d: 'text' must be a non-empty string"
                    % (shot_idx, line_idx)
                )
            line = {"speaker": speaker.strip(), "text": text}
            if "picture" in raw_line and raw_line["picture"] not in (None, ""):
                line["picture"] = validate_picture_cue(raw_line["picture"])
            line_lang = validate_line_lang(
                raw_line.get("lang"),
                where="shot %d line %d" % (shot_idx, line_idx),
            )
            if line_lang:
                line["lang"] = line_lang
            cleaned_lines.append(line)
        shot = {"lines": cleaned_lines}
        absent = validate_absent(raw_shot.get("absent"), cast, offscreen_names)
        if absent:
            shot["absent"] = absent
            # Every shot keeps at least one on-screen character. H3
            # renders an all-off-screen shot as an empty room plate
            # (2026-10-18 LAN forensics), so we refuse the board here.
            on_screen = [
                n for n in cast
                if n not in absent and n not in offscreen_names
            ]
            if cast and not on_screen:
                raise ValueError(
                    "shot %d marks %s absent, leaving no on-screen "
                    "character; a shot with nobody in frame renders as an "
                    "empty room plate — keep at least one cast member "
                    "visible" % (shot_idx, absent)
                )
        cleaned.append(shot)
    return cleaned


def validate_spatial_layout(layout, cast):
    """Optional ``{speaker: "left of frame" / "right of frame" / ...}``.
    Every key must be a cast member; values must be non-empty English."""
    if layout in (None, {}):
        return None
    if not isinstance(layout, dict):
        raise ValueError(
            "spatial_layout must be an object, got %s"
            % type(layout).__name__
        )
    cleaned: dict[str, str] = {}
    for raw_name, raw_pos in layout.items():
        name = _library.validate_name(raw_name)
        if name not in cast:
            raise ValueError(
                "spatial_layout[%r]: speaker is not in cast %s"
                % (raw_name, sorted(cast.keys()))
            )
        if not isinstance(raw_pos, str) or not raw_pos.strip():
            raise ValueError(
                "spatial_layout[%r] must be a non-empty English phrase"
                % (raw_name,)
            )
        cleaned[name] = raw_pos.strip()
    return cleaned


def validate_lines(raw_lines):
    """``lines``: a flat list of ``{speaker, text, picture?}`` entries
    (PLAN §3.1 auto-pack field). Speaker names must be in ``cast``;
    text is verbatim; picture cue is the same English-only rule as
    ``shots[].lines[].picture``."""
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(
            "lines must be a non-empty list of {speaker, text, picture?}"
        )
    cleaned: list[dict] = []
    for idx, raw_entry in enumerate(raw_lines, 1):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                "lines[%d] must be an object, got %s"
                % (idx, type(raw_entry).__name__)
            )
        speaker = raw_entry.get("speaker")
        text = raw_entry.get("text")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(
                "lines[%d]: 'speaker' must be a non-empty string" % idx
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "lines[%d]: 'text' must be a non-empty string" % idx
            )
        entry = {"speaker": speaker.strip(), "text": text}
        if "picture" in raw_entry and raw_entry["picture"] not in (None, ""):
            entry["picture"] = validate_picture_cue(raw_entry["picture"])
        entry_lang = validate_line_lang(
            raw_entry.get("lang"), where="lines[%d]" % idx,
        )
        if entry_lang:
            entry["lang"] = entry_lang
        cleaned.append(entry)
    return cleaned


def validate_storyboard_payload(payload, *, require_shots=False):
    """Shape-check the JSON dict. Returns the cleaned dict.

    Pass ``require_shots=True`` from a path that already knows shots are
    present (the save node — the user typed them). When ``shots`` is
    absent the assembler later uses the auto-pack default; ``require_shots``
    is therefore False on the load node (the file may legitimately omit it).
    """
    if not isinstance(payload, dict):
        raise ValueError(
            "storyboard.json must contain a JSON object, got %s"
            % type(payload).__name__
        )
    version = payload.get("version")
    if version != STORYBOARD_VERSION:
        raise ValueError(
            "storyboard.json version must be %d, got %r; this node does "
            "not migrate older formats — re-author against the current "
            "schema" % (STORYBOARD_VERSION, version)
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("storyboard.json 'name' must be a non-empty string")
    name = _library.validate_name(name)

    language = validate_language(payload.get("language", "zh"))
    camera_discipline = validate_camera_discipline(
        payload.get("camera_discipline", "locked")
    )
    pacing = validate_pacing(payload.get("pacing", "normal"))
    seed_policy = validate_seed_policy(
        payload.get("seed_policy", "per_scene_increment")
    )
    prompt_prefix = validate_prompt_prefix(
        payload.get("prompt_prefix", "")
    )
    if not prompt_prefix:
        raise ValueError(
            "prompt_prefix is required and must contain at least one "
            "non-empty English line (the global visual style sentence)"
        )

    cast = validate_cast(payload.get("cast", {}))
    offscreen_names = {
        name for name, info in cast.items() if info.get("offscreen")
    }
    setting_line = validate_setting_line(payload.get("setting_line", ""))
    ambience = validate_ambience_line(payload.get("ambience", ""))

    raw_shots = payload.get("shots")
    raw_lines = payload.get("lines")
    if raw_shots is not None and raw_lines is not None:
        raise ValueError(
            "storyboard carries both 'shots' and 'lines'; provide exactly "
            "one — 'shots' for an authored cut, 'lines' for auto-packing"
        )
    if raw_shots is None:
        if require_shots:
            raise ValueError(
                "shots is required when saving a storyboard (omit 'shots' "
                "in the saved JSON only when auto-packing on the build node)"
            )
        shots = None
    else:
        shots = validate_shots(raw_shots, cast, offscreen_names)
    if raw_lines is None:
        lines = None
    else:
        lines = validate_lines(raw_lines)

    spatial_layout = validate_spatial_layout(
        payload.get("spatial_layout"), cast,
    )
    # An off-screen speaker has no place in the frame, so a spatial pin
    # for one is a contradiction — the pin is exactly the instruction
    # "this character is visible at <position>". (Loop var deliberately
    # NOT named ``name``: that holds the storyboard's own name.)
    for pinned_name in (spatial_layout or {}):
        if pinned_name in offscreen_names:
            raise ValueError(
                "spatial_layout pins %r, which is an offscreen speaker; an "
                "off-screen voice is not in frame, so it cannot have a "
                "frame position — drop it from spatial_layout"
                % (pinned_name,)
            )

    # Cross-check: every speaker in shots or lines must be in cast.
    if shots:
        for shot_idx, shot in enumerate(shots, 1):
            for line_idx, line in enumerate(shot["lines"], 1):
                if line["speaker"] not in cast:
                    raise ValueError(
                        "shot %d line %d: speaker %r is not in cast %s"
                        % (shot_idx, line_idx, line["speaker"],
                           sorted(cast.keys()))
                    )
    if lines:
        for idx, line in enumerate(lines, 1):
            if line["speaker"] not in cast:
                raise ValueError(
                    "lines[%d]: speaker %r is not in cast %s"
                    % (idx, line["speaker"], sorted(cast.keys()))
                )

    # A POV board needs at least one on-screen character SOMEWHERE on the
    # board. With every cast member offscreen there is nobody to face the
    # lens in any shot, which is the empty-room defect again.
    if cast and offscreen_names and len(offscreen_names) == len(cast):
        raise ValueError(
            "every cast member is offscreen (%s); a first-person POV board "
            "needs at least one character in frame to face the camera — "
            "un-mark at least one cast entry" % sorted(offscreen_names)
        )

    cleaned = {
        "name": name,
        "version": STORYBOARD_VERSION,
        "language": language,
        "camera_discipline": camera_discipline,
        "pacing": pacing,
        "seed_policy": seed_policy,
        "prompt_prefix": prompt_prefix,
        "cast": cast,
        "setting_line": setting_line,
        "shots": shots,
        "spatial_layout": spatial_layout,
    }
    if lines is not None:
        cleaned["lines"] = lines
    if ambience:
        cleaned["ambience"] = ambience
    return cleaned


# --------------------------------------------------------------------------- #
# Disk operations (pure: bytes in, dict out, no torch, no ComfyUI)
# --------------------------------------------------------------------------- #
def _storyboards_dir(library_dir):
    """The ``storyboards/`` subdir under the resolved asset root."""
    root = _library.validate_library_dir(library_dir)
    return root / STORYBOARDS_DIRNAME


def _storyboard_path(library_dir, name):
    """Absolute path to ``<root>/storyboards/<name>/storyboard.json``."""
    n = _library.validate_name(name)
    return _storyboards_dir(library_dir) / n / STORYBOARD_JSON_NAME


def save_storyboard(library_dir, payload):
    """Validate the payload, then atomically write
    ``<root>/storyboards/<name>/storyboard.json``. Returns the cleaned
    payload, the absolute path, and the status (``created`` /
    ``overwritten``).

    The directory is created LAZY — only after the cleaned payload
    passes validation, so a malformed input never leaves an empty
    ``storyboards/<name>/`` directory behind. A pre-existing same-name
    directory is overwritten in place (casefolded comparison).
    """
    cleaned = validate_storyboard_payload(payload, require_shots=True)
    name = cleaned["name"]
    sb_dir = _storyboards_dir(library_dir)
    sb_dir.mkdir(parents=True, exist_ok=True)
    existing = _library._find_casefold_sibling(sb_dir, name)
    if existing is not None and not existing.is_dir():
        raise ValueError(
            "%s exists but is not a directory; cannot store storyboard %r "
            "there" % (existing, name)
        )
    if existing is None:
        target_dir = sb_dir / name
        status = "created"
    else:
        target_dir = existing
        status = "overwritten"
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / STORYBOARD_JSON_NAME
    payload_bytes = (json.dumps(cleaned, ensure_ascii=False) + "\n").encode("utf-8")
    tmp_fd, tmp_name = __import__("tempfile").mkstemp(
        prefix=".mie_storyboard_", dir=str(target_dir)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(payload_bytes)
        os.replace(tmp_path, json_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return cleaned, str(json_path), status


def load_storyboard(library_dir, name):
    """Read + shape-check one storyboard JSON. Returns the cleaned payload
    plus the absolute path. Raises with the path on every failure."""
    json_path = _storyboard_path(library_dir, name)
    if not json_path.is_file():
        raise FileNotFoundError(
            "storyboard %r not found at %s" % (name, json_path)
        )
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError("%s is not valid JSON: %s" % (json_path, e)) from e
    if not isinstance(payload, dict):
        raise ValueError("%s must contain a JSON object" % json_path)
    cleaned = validate_storyboard_payload(payload)
    # The on-disk name field must equal the directory name (the same
    # consistency rule scenes and characters enforce).
    if cleaned["name"] != json_path.parent.name:
        raise ValueError(
            "%s declares name %r but the directory is %r; rename the "
            "directory or fix the JSON so both match"
            % (json_path, cleaned["name"], json_path.parent.name)
        )
    return cleaned, str(json_path)


def list_storyboards(library_dir):
    """Readable storyboard names (one per line, sorted). Skip report in
    ``status`` mirrors ``list_characters`` — bad directories are skipped,
    not raised, so one broken file can't split the dropdown."""
    sb_dir = _storyboards_dir(library_dir)
    names: list[str] = []
    skipped: list[str] = []
    if not sb_dir.is_dir():
        return {"names": "", "status": "ok"}
    for entry in sb_dir.iterdir():
        if not entry.is_dir():
            continue
        reason = _storyboard_entry_problem(entry)
        if reason is None:
            names.append(entry.name)
        else:
            skipped.append("skipped %s: %s" % (entry.name, reason))
    return {
        "names": "\n".join(sorted(names)),
        "status": "ok" if not skipped else "\n".join(skipped),
    }


def _storyboard_entry_problem(entry):
    try:
        _library.validate_name(entry.name)
    except ValueError as e:
        return str(e)
    json_path = entry / STORYBOARD_JSON_NAME
    if not json_path.is_file():
        return "missing %s" % STORYBOARD_JSON_NAME
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return "not valid JSON: %s" % e
    if not isinstance(payload, dict):
        return "storyboard.json must contain a JSON object"
    try:
        cleaned = validate_storyboard_payload(payload)
    except ValueError as e:
        return str(e)
    if cleaned["name"] != entry.name:
        return (
            "JSON name %r does not match directory name %r"
            % (cleaned["name"], entry.name)
        )
    return None


def storyboard_cache_key(library_dir, name):
    """Content hash of one storyboard JSON for IS_CHANGED. Returns the
    MISSING marker when the file is gone (so a failed read is not
    cached as a success — mirrors ``library.character_cache_key``)."""
    try:
        _library.validate_library_dir(library_dir)
        n = _library.validate_name(name)
        sb_dir = _library._find_casefold_sibling(
            _storyboards_dir(library_dir), n
        )
        if sb_dir is None or not sb_dir.is_dir():
            return _library.MISSING_CACHE_MARKER
        json_path = sb_dir / STORYBOARD_JSON_NAME
        if not json_path.is_file():
            return _library.MISSING_CACHE_MARKER
        digest = hashlib.sha256()
        digest.update(json_path.name.encode("utf-8"))
        digest.update(json_path.read_bytes())
        return digest.hexdigest()
    except Exception:
        return _library.MISSING_CACHE_MARKER


def storyboard_requirements(cleaned: dict) -> list[dict]:
    """What the build node needs wired, in SOCKET ORDER.

    The storyboard is a script; the build node's sockets are a fixed
    1..8 + scene_1 grid, so the author has to know which of those
    sockets this particular board cares about. Answering that question
    at LOAD time (the Load node's preview) is the difference between
    "wire two characters" and "guess and read the error".

    Each entry is::

        {"socket": "character_1", "kind": "character"|"scene",
         "name": "<cast member>", "offscreen": bool, "required": bool}

    ``required`` is False for a socket the storyboard does not use —
    the build node ignores unused higher sockets (PLAN_STORYBOARD_ASSET
    §4.2), so the list distinguishes "must fill this" from "leave it
    alone".
    """
    cast = cleaned.get("cast") or {}
    shots = cleaned.get("shots") or []
    lines = cleaned.get("lines") or []

    used: set[str] = set()
    if shots:
        for shot in shots:
            for line in shot.get("lines") or []:
                used.add(str(line.get("speaker") or ""))
    elif lines:
        for line in lines:
            used.add(str(line.get("speaker") or ""))

    out: list[dict] = []
    for name, info in sorted(
        cast.items(), key=lambda kv: int(kv[1]["character_slot"]),
    ):
        slot = int(info["character_slot"])
        offscreen = bool(info.get("offscreen"))
        out.append({
            "socket": "character_%d" % slot,
            "kind": "character",
            "name": name,
            "offscreen": offscreen,
            # An off-screen speaker's asset is OPTIONAL (2026-10-20
            # ruling): wired means "use this voice", unwired means
            # "declare no voice and let H3 choose". An on-screen
            # character is always required — he is in the frame.
            "required": not offscreen,
            "note": (
                "画外音：接上则用他的音色，不接则由 H3 自选 · "
                "off-screen: wire it to use its voice, or leave it empty "
                "and H3 chooses"
                if offscreen else ""
            ),
        })
    out.append({
        "socket": "scene_1",
        "kind": "scene",
        "name": None,
        "offscreen": False,
        "required": True,
        "note": "",
    })
    return out


def read_storyboard_meta(library_dir, name):
    """Frontend preview meta for one storyboard: a text summary + zero
    images (the preview panel renders text-only when image_count == 0).
    Raises ValueError on unknown names, FileNotFoundError when the
    storyboards directory is absent — same contract as
    ``library.read_asset_meta``."""
    cleaned, path = load_storyboard(library_dir, name)
    lines = sum(len(shot.get("lines") or []) for shot in cleaned.get("shots") or [])
    offscreen = sorted(
        n for n, v in (cleaned.get("cast") or {}).items()
        if isinstance(v, dict) and v.get("offscreen")
    )
    requirements = storyboard_requirements(cleaned)
    description = (
        "Storyboard %s: %d shot(s), %d line(s), language=%s%s." % (
            cleaned.get("name", "?"),
            len(cleaned.get("shots") or []),
            lines,
            cleaned.get("language", "?"),
            (", off-screen: " + ", ".join(offscreen)) if offscreen else "",
        )
    )
    return {
        "name": cleaned.get("name", name),
        "kind": "storyboard",
        "description": description,
        # The wiring recipe: which build-node socket needs which cast
        # member, so the author does not have to open the JSON to find
        # out how many characters and how many scenes this board wants.
        "requirements": requirements,
        "image_count": 0,
        "audio_count": None,
        "picture_size": None,
        "picture_mtime_ns": None,
    }


def augment_assets_payload(payload, library_dir):
    """Add the ``storyboards`` key to a ``library.list_assets_payload``
    result (the server joins the two modules; library.py must not import
    this module — storyboard.py imports library validators, the reverse
    would be circular)."""
    listing = list_storyboards(library_dir)
    names = [n for n in listing.get("names", "").splitlines() if n]
    payload["storyboards"] = [{"name": n} for n in names]
    return payload
