"""Deterministic plan-building layer for the ``MiniMaxH3LoopPromptGenerator``.

Owns everything that must NOT depend on LLM whims:

* H3 length grid math — valid raw lengths satisfy ``length % 17 == 5``
  within 5..3592 at 24 fps (``seconds_to_length`` / ``is_valid_length``);
* ``split_three_sections`` — turn one clip's LLM reply into the exact
  prompt line-array shape used by the Production Plan workflow
  (``["integrated_multimodal_description:", ..., "", "overall_soundscape:",
  ..., "", "non_diegetic_music:", ...]``);
* ``parse_shots_text`` — accept the storyboard generator's ``shots_json``
  (JSON array / object with ``shots``) or one-line-per-shot natural language;
* ``parse_per_shot_overrides`` / ``apply_overrides`` — ``id:length=N`` /
  ``id:seed=NNNN`` / ``id:steps=N`` overrides;
* ``build_plan`` + ``validate_plan`` — assemble and assert the strict
  plan shape (top-level keys exactly ``defaults`` / ``shots`` /
  ``prompt_prefix``; ``prompt`` and ``prompt_prefix`` are arrays; seeds
  are digit strings; no continuation_mode / context_length / width /
  height ever enters the JSON — those live on the Plan node's widgets);
* preflight report + markdown preview rendering;
* prompt-template builders for the three LLM stages (prefix synthesis,
  per-shot generation, single-call generation).
* reference / keyframe mode wiring (t2va / i2va / fl2va / ref2va).

Schema source: H3_CHAIN_FORMAT_GUIDE.md + the real Production Plan
(``MiniMaxH3ChainPlanModern``) JSON from the user's workflow — see
``prompts/h3_loop/UPSTREAM.md``.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text

try:
    from _mienodes_internal.nodes.llm.h3_prompts import (
        aspect_ratio_string,
        category_advice,
        parse_category,
        system_t2v_prompt,
        system_reference_prompt,
    )
except ImportError:
    from .h3_prompts import (
        aspect_ratio_string,
        category_advice,
        parse_category,
        system_t2v_prompt,
        system_reference_prompt,
    )

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:
        from core.utils import mie_log


# --------------------------------------------------------------------------- #
# H3 timing grid (24 fps, valid raw lengths 17k+5 within 5..3592)
# --------------------------------------------------------------------------- #
FPS = 24
MIN_LENGTH_FRAMES = 5
MAX_LENGTH_FRAMES = 3592  # 17*211+5; ~149.667 s
GRID_STEP = 17

UINT64_MAX = 0xFFFFFFFFFFFFFFFF
MIN_STEPS = 1
MAX_STEPS = 10000
MIN_SHOTS = 1
MAX_SHOTS = 128
DEFAULT_STEPS = 20

# Prompted per-clip duration default (10 s -> 243 frames, the grid value
# used by the reference workflow).
DEFAULT_DURATION_SECONDS = 10


def seconds_to_length(duration_seconds: Any) -> int:
    """Round a duration request up onto the H3 ``17k+5`` frame grid.

    ``raw = ceil(seconds * 24)``; result is the smallest grid frame count
    ``>= raw`` (5, 22, 39, ... 3592). Raises ``ValueError`` for
    non-positive durations.
    """
    try:
        seconds = float(duration_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid duration_seconds: {duration_seconds!r}") from exc
    if seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds!r}")
    raw = math.ceil(seconds * FPS - 1e-9)
    k = max(0, math.ceil((raw - MIN_LENGTH_FRAMES) / GRID_STEP - 1e-9))
    length = MIN_LENGTH_FRAMES + k * GRID_STEP
    if length > MAX_LENGTH_FRAMES:
        length = MAX_LENGTH_FRAMES
    return length


def is_valid_length(length: Any) -> bool:
    try:
        n = int(length)
    except (TypeError, ValueError):
        return False
    return (
        MIN_LENGTH_FRAMES <= n <= MAX_LENGTH_FRAMES
        and (n - MIN_LENGTH_FRAMES) % GRID_STEP == 0
    )


def length_to_seconds(length: int) -> float:
    return round(int(length) / FPS, 2)


# --------------------------------------------------------------------------- #
# Section fields + three-section split
# --------------------------------------------------------------------------- #
SECTION_DESCRIPTION = "integrated_multimodal_description"
SECTION_SOUND = "overall_soundscape"
SECTION_MUSIC = "non_diegetic_music"
SECTION_FIELDS = (SECTION_DESCRIPTION, SECTION_SOUND, SECTION_MUSIC)

DEFAULT_MUSIC_LINE = "No non-diegetic music."


# --------------------------------------------------------------------------- #
# Reference / keyframe modes + schema constants
# --------------------------------------------------------------------------- #
# Display strings follow the project convention "code - 中文" so
# ``parse_reference_mode`` can split them back to the short code (mirrors
# ``parse_generation_mode`` in the node wrapper).
REFERENCE_MODES = (
    "t2va - 文生视频链(默认)",
    "i2va - 首帧关键帧(文生视频+首帧图)",
    "fl2va - 首尾帧关键帧(首帧+逐场尾帧)",
    "ref2va - 参考图(N张/全场景)",
)
REFERENCE_MODE_CODES = ("t2va", "i2va", "fl2va", "ref2va")

SCHEMA_THREE = "three_section"
SCHEMA_SIX = "six_section"
SCHEMA_CHOICES = (SCHEMA_THREE, SCHEMA_SIX)

# Ref2VA's six bare headers in exact order (mirrors the official Ref2V Basic
# workflow: subject_definitions, summary, retention_analysis,
# detailed_description, overall_soundscape, non_diegetic_music).
SECTION_SUBJECT = "subject_definitions"
SECTION_SUMMARY = "summary"
SECTION_RETENTION = "retention_analysis"
SECTION_DETAIL = "detailed_description"
SIX_SECTION_FIELDS = (
    SECTION_SUBJECT,
    SECTION_SUMMARY,
    SECTION_RETENTION,
    SECTION_DETAIL,
    SECTION_SOUND,
    SECTION_MUSIC,
)

# Stock Ref2VA picture cap (H3_CHAIN_FORMAT_GUIDE: 9 pictures per scene).
MAX_MANIFEST_PICTURES = 9
MAX_MANIFEST_VIDEOS = 3  # stock Ref2VA: 9 pictures, 3 videos, 3 standalone audios


def parse_reference_mode(mode: str) -> str:
    """Split the bilingual dropdown label back to the short mode code;
    unknown values fall back to ``t2va`` (the chain default)."""
    code = (mode or "").split(" - ", 1)[0].strip()
    return code if code in REFERENCE_MODE_CODES else REFERENCE_MODE_CODES[0]


def schema_for_mode(mode: str) -> str:
    """The schema a reference mode emits: only ref2va uses six sections;
    t2va / i2va / fl2va keep the three-section form."""
    return SCHEMA_SIX if parse_reference_mode(mode) == "ref2va" else SCHEMA_THREE


# --------------------------------------------------------------------------- #
# Label scanners (native H3 labels vs @alias / #tag tokens)
# --------------------------------------------------------------------------- #
_NATIVE_LABEL_RE = re.compile(
    r"<(Picture|Video|Audio|Subject)\s+(\d{1,2})>"
)
# @alias tokens: ASCII identifier, 1..64 chars, NOT preceded by an
# identifier char so we do not accidentally match email addresses.
_REFERENCE_ALIAS_RE = re.compile(
    r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]{0,63})"
)
# #tag tokens with optional trailing "[<seconds>s]" window anchor (the
# P0 Scene Prompt Editor's semantic anchor grammar). The closing-tag
# lookahead is optional so a bare #tag is still recognized.
_SEMANTIC_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_])#([A-Za-z][A-Za-z0-9_-]{0,63})"
    r"(?:\[([0-9]+(?:\.[0-9]+)?)s?\]|(?!\[))"
    r"(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)


def find_native_labels(text: str) -> list[tuple[str, int]]:
    """All native H3 labels in ``text`` as ``(kind, n)`` tuples (sorted
    in document order). ``kind`` is one of Picture / Video / Audio /
    Subject."""
    return [(m.group(1), int(m.group(2))) for m in _NATIVE_LABEL_RE.finditer(text or "")]


def find_alias_tokens(text: str) -> list[str]:
    """All @alias tokens in ``text`` (in document order, dedup not
    applied — the validator reports every occurrence)."""
    return [m.group(1) for m in _REFERENCE_ALIAS_RE.finditer(text or "")]


def find_semantic_anchors(text: str) -> list[tuple[str, str]]:
    """All #tag tokens in ``text`` as (name, window) tuples; ``window``
    is the trailing ``[Ns]`` suffix if present else empty string."""
    return [
        (m.group(1), m.group(2) or "")
        for m in _SEMANTIC_ANCHOR_RE.finditer(text or "")
    ]


def manifest_normalize_json(manifest: list[dict]) -> str:
    """Deterministic JSON form for the manifest, used by ``is_changed``
    so whitespace-only edits do not bust the cache."""
    cleaned = [
        {k: v for k, v in (m or {}).items() if v is not None}
        for m in (manifest or [])
    ]
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Manifest parsing + per-mode validation
# --------------------------------------------------------------------------- #
_MANIFEST_ROLES = ("identity", "destination", "environment")
_SLOT_RE = re.compile(
    r"^\s*<?\s*(Picture|Video|Audio)\s+(\d{1,2})\s*>?\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _parse_manifest_slot_label(raw: Any) -> str:
    """``"Picture 3"`` / ``"<Picture 3>"`` -> ``"Picture 3"``; reject
    anything else (Video / Audio / unknown)."""
    if raw is None:
        raise ValueError("manifest slot label missing")
    s = str(raw).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    norm = re.sub(r"\s+", " ", s)
    if not re.match(r"^(Picture|Video|Audio)\s+\d{1,2}$", norm, re.IGNORECASE):
        raise ValueError(
            f"manifest slot must look like 'Picture 3' or '<Picture 3>', got {raw!r}"
        )
    kind, num = norm.split(" ", 1)
    return f"{kind.capitalize()} {int(num)}"


def parse_references_text(text: str) -> list[dict]:
    """Parse ``references_text`` into a canonical manifest list.

    Each entry is ``{"slot": "Picture 1", "about": "...", "role":
    "identity" | "destination" | "environment"}``. Accepts:

    - A strict JSON array of objects (keys: ``slot`` / ``about`` /
      ``role``; ``picture`` / ``label`` are aliases for ``slot``;
      ``description`` / ``text`` are aliases for ``about``).
    - One natural line per picture: ``Picture 1: courier face, ...``.
      The ``<Picture 1>`` bracket form is also accepted.

    Role defaults: slot 1 -> ``identity``; other slots -> ``destination``.
    Invalid lines / Video / Audio slots / gaps in numbering raise
    ``ValueError`` with the offending line number.

    fl2va slot mapping (verified against the upstream gate wiring):
    Picture 1 = OPENING frame; Picture 2 = scene 1 end target;
    Picture k (k>=2) = scene k end target (FrameIndexSwitch.frame_k).
    Scenes 2+ all expose the per-scene image under the single
    ``<Picture 1>`` label, so the about text for scene N>=2 is
    manifest[N] when present, otherwise alternating manifest[0] /
    manifest[1] so legacy 2-image workflows keep their A->B->A rhythm.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    cleaned: list[dict] | None = None
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("pictures"), list):
            items = data["pictures"]
        else:
            items = None
        if items is not None:
            cleaned = []
            for i, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"manifest line {i}: expected an object")
                slot_raw = item.get("slot") or item.get("picture") or item.get("label")
                about = (
                    item.get("about")
                    or item.get("description")
                    or item.get("text")
                )
                role = (item.get("role") or "").strip().lower() or None
                cleaned.append(
                    {
                        "slot": _parse_manifest_slot_label(slot_raw),
                        "about": str(about or "").strip(),
                        "role": role,
                    }
                )
    if cleaned is None:
        cleaned = []
        for i, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _SLOT_RE.match(line)
            if not m:
                raise ValueError(
                    f"manifest line {i}: expected 'Picture N: <about>' or '<Picture N>: <about>', got {line!r}"
                )
            kind = m.group(1).capitalize()
            num = int(m.group(2))
            slot = f"{kind} {num}"
            about = m.group(3).strip()
            if not about:
                raise ValueError(f"manifest line {i}: empty about text for {slot}")
            cleaned.append({"slot": slot, "about": about, "role": None})

    final: list[dict] = []
    seen_video = False
    videos_seen = 0
    for i, entry in enumerate(cleaned, start=1):
        slot = entry["slot"]
        kind = slot.split(" ", 1)[0]
        if kind == "Audio":
            raise ValueError(
                f"manifest line {i}: {slot!r} is an Audio slot; standalone "
                "audio references are not supported"
            )
        if kind == "Video":
            # Guide presentation order: pictures first, then videos
            # (ordinals are independent per kind, but the manifest lists
            # entries in presentation order).
            seen_video = True
            videos_seen += 1
            num = int(slot.split(" ", 1)[1])
            if num != videos_seen:
                raise ValueError(
                    f"manifest line {i}: video slots must be contiguous "
                    f"Video 1..M, got {slot!r}"
                )
            role = entry.get("role") or "destination"
            role = str(role).strip().lower()
            if role not in _MANIFEST_ROLES:
                raise ValueError(
                    f"manifest line {i}: role must be one of {_MANIFEST_ROLES}, got {role!r}"
                )
            final.append({"slot": slot, "about": str(entry["about"]).strip(), "role": role})
            continue
        if seen_video:
            raise ValueError(
                f"manifest line {i}: {slot!r} appears after a Video slot; "
                "list all Picture slots before Video slots"
            )
        num = int(slot.split(" ", 1)[1])
        if num != i:
            raise ValueError(
                f"manifest line {i}: numbering must start at Picture 1 and stay contiguous, got {slot!r}"
            )
        role = entry.get("role") or ("identity" if i == 1 else "destination")
        role = str(role).strip().lower()
        if role not in _MANIFEST_ROLES:
            raise ValueError(
                f"manifest line {i}: role must be one of {_MANIFEST_ROLES}, got {role!r}"
            )
        final.append({"slot": slot, "about": str(entry["about"]).strip(), "role": role})
    return final


def validate_manifest(manifest: list[dict], mode: str) -> list[str]:
    """Per-mode manifest shape contract. Returns a list of error strings
    (empty list = valid)."""
    code = parse_reference_mode(mode)
    n = len(manifest or [])
    errors: list[str] = []
    if code == "t2va":
        if n != 0:
            errors.append(
                f"t2va must have an empty references_text (got {n} entries); "
                "switch reference_mode to i2va / fl2va / ref2va to use images."
            )
        return errors
    if n == 0:
        errors.append(
            f"{code} requires references_text with at least one picture; "
            "supply either JSON or 'Picture 1: <about>' lines."
        )
        return errors
    if code == "i2va":
        if n != 1:
            errors.append(f"i2va needs exactly 1 picture, got {n}")
        elif manifest[0]["slot"] != "Picture 1":
            errors.append(
                f"i2va needs 'Picture 1', got {manifest[0]['slot']!r}"
            )
        return errors
    if code == "fl2va":
        # fl2va manifest: 2..9 contiguous pictures. Picture 1 is the
        # OPENING frame; Picture 2 is scene 1 end target; Picture
        # k>=2 is scene k end target (matches FrameIndexSwitch.frame_k).
        if n < 2:
            errors.append(
                f"fl2va needs at least 2 pictures (Picture 1 = opening, "
                f"Picture 2 = scene 1 end target), got {n}"
            )
        elif n > MAX_MANIFEST_PICTURES:
            errors.append(
                f"fl2va supports up to {MAX_MANIFEST_PICTURES} pictures, got {n}"
            )
        elif [m["slot"] for m in manifest] != [
            f"Picture {i + 1}" for i in range(n)
        ]:
            errors.append(
                f"fl2va needs contiguous Picture 1..{n}, got {[m['slot'] for m in manifest]}"
            )
        return errors
    if code == "ref2va":
        pics = [m for m in manifest if (m["slot"] or "").startswith("Picture ")]
        vids = [m for m in manifest if (m["slot"] or "").startswith("Video ")]
        if len(pics) > MAX_MANIFEST_PICTURES:
            errors.append(
                f"ref2va supports up to {MAX_MANIFEST_PICTURES} pictures, got {len(pics)}"
            )
        if len(vids) > MAX_MANIFEST_VIDEOS:
            errors.append(
                f"ref2va supports up to {MAX_MANIFEST_VIDEOS} video "
                f"references, got {len(vids)}"
            )
        if not pics and not vids:
            errors.append("ref2va manifest is empty after slot parsing")
        expected_pics = [f"Picture {i+1}" for i in range(len(pics))]
        if [m["slot"] for m in pics] != expected_pics:
            errors.append(
                f"ref2va needs contiguous Picture 1..{len(pics)} (pictures "
                f"before videos), got {[m['slot'] for m in pics]}"
            )
        expected_vids = [f"Video {i+1}" for i in range(len(vids))]
        if [m["slot"] for m in vids] != expected_vids:
            errors.append(
                f"ref2va needs contiguous Video 1..{len(vids)}, got "
                f"{[m['slot'] for m in vids]}"
            )
        return errors
    errors.append(f"unknown reference mode: {mode!r}")
    return errors


# --------------------------------------------------------------------------- #
# Label policy per reference mode (validator: errors, not warnings)
# --------------------------------------------------------------------------- #
def _shot_texts(plan: dict) -> list[str]:
    """Concatenate every text field we need to scan: each shot's full
    prompt array plus the prompt_prefix lines."""
    chunks: list[str] = []
    for shot in plan.get("shots") or []:
        for line in shot.get("prompt") or []:
            if isinstance(line, str):
                chunks.append(line)
    for line in plan.get("prompt_prefix") or []:
        if isinstance(line, str):
            chunks.append(line)
    return chunks


def _identity_subject_slots(manifest: list[dict]) -> dict[int, int]:
    """Map ``Subject N`` (1-indexed) to its ``Picture N`` slot for the
    identity-role entries of the manifest (D5: deterministic binding,
    not LLM-invented)."""
    out: dict[int, int] = {}
    sub_idx = 0
    for entry in manifest or []:
        if entry.get("role") != "identity":
            continue
        sub_idx += 1
        slot = entry.get("slot") or ""
        num = int(slot.split(" ", 1)[1]) if slot.startswith("Picture ") else 0
        out[sub_idx] = num
    return out


def validate_label_policy(plan: dict, mode: str, manifest: list[dict]) -> list[str]:
    """Per-mode native-label + @alias / #tag contract. Returns errors.
    Empty list = the plan's text matches the mode's label contract.

    Universal: any ``@alias`` or ``#tag`` token anywhere in the plan
    text is an error (D9). P0 does not support Scheduled Ref2VA / the
    Scene Prompt Editor's dialogue markup — only native labels.
    """
    code = parse_reference_mode(mode)
    chunks = _shot_texts(plan)
    errors: list[str] = []

    # D9: @alias / #tag tokens are forbidden across all modes.
    for i, text in enumerate(chunks, start=1):
        aliases = find_alias_tokens(text)
        if aliases:
            errors.append(
                f"shot/prefix {i}: @alias tokens are not supported in this version "
                f"({', '.join('@' + a for a in aliases[:3])}); "
                "use the native <Picture N> / <Subject N> labels only."
            )
        anchors = find_semantic_anchors(text)
        if anchors:
            errors.append(
                f"shot/prefix {i}: #tag tokens are not supported in this version "
                f"({', '.join('#' + n for n, _ in anchors[:3])}); "
                "remove the Semantic Anchor / Dialogue markup."
            )

    # Per-mode native-label policy.
    shots = plan.get("shots") or []
    if code == "t2va":
        # Any native label in t2va is an error: t2va has no wired images.
        for i, text in enumerate(chunks, start=1):
            for kind, num in find_native_labels(text):
                errors.append(
                    f"shot/prefix {i}: <{kind} {num}> is not valid in t2va mode; "
                    "switch reference_mode to i2va / fl2va / ref2va to use labels."
                )
        return errors

    if code == "i2va":
        # Scene 1 may reference <Picture 1>; scenes 2+ must have zero
        # native labels (official I2V: First-Scene Image Gate hides
        # Picture 1 from continuation scenes).
        for s_idx, shot in enumerate(shots, start=1):
            for ln in shot.get("prompt") or []:
                if not isinstance(ln, str):
                    continue
                for kind, num in find_native_labels(ln):
                    if kind in ("Video", "Audio"):
                        errors.append(
                            f"shot {s_idx}: <{kind} {num}> not allowed in i2va; pictures only"
                        )
                        continue
                    if kind == "Subject":
                        errors.append(
                            f"shot {s_idx}: <Subject {num}> is a Ref2VA construct; i2va uses <Picture 1> only"
                        )
                        continue
                    if s_idx == 1 and num == 1:
                        continue
                    if s_idx == 1:
                        errors.append(
                            f"shot {s_idx}: i2va allows <Picture 1> only in scene 1, got <Picture {num}>"
                        )
                    else:
                        errors.append(
                            f"shot {s_idx}: i2va hides Picture 1 from continuation scenes; got <Picture {num}>"
                        )
        return errors

    if code == "fl2va":
        # Scene 1 may reference <Picture 1> / <Picture 2>; scenes 2+
        # alternate targets <Picture (N % 2) + 1>.
        for s_idx, shot in enumerate(shots, start=1):
            for ln in shot.get("prompt") or []:
                if not isinstance(ln, str):
                    continue
                for kind, num in find_native_labels(ln):
                    if kind in ("Video", "Audio"):
                        errors.append(
                            f"shot {s_idx}: <{kind} {num}> not allowed in fl2va; pictures only"
                        )
                        continue
                    if kind == "Subject":
                        errors.append(
                            f"shot {s_idx}: <Subject {num}> is a Ref2VA construct; fl2va uses <Picture N> only"
                        )
                        continue
                    if s_idx == 1:
                        if num in (1, 2):
                            continue
                        errors.append(
                            f"shot 1: fl2va allows <Picture 1> / <Picture 2> only, got <Picture {num}>"
                        )
                    else:
                        # Per-scene end target is always <Picture 1>:
                        # the FL2VA gate exposes a single per-scene
                        # image (the FrameIndexSwitch.frame_N output)
                        # under the <Picture 1> label; there is no
                        # second picture to reference in scenes 2+.
                        expected = 1
                        if num == expected:
                            continue
                        errors.append(
                            f"shot {s_idx}: fl2va end target is <Picture 1>, got <Picture {num}>"
                        )
        return errors

    if code == "ref2va":
        # Every label must be a manifest slot or a Subject bound to an
        # identity slot (D5: deterministic binding). Numbering must be
        # contiguous.
        manifest_pic_nums = set()
        manifest_vid_nums = set()
        for entry in manifest or []:
            slot = entry.get("slot") or ""
            if slot.startswith("Picture "):
                manifest_pic_nums.add(int(slot.split(" ", 1)[1]))
            elif slot.startswith("Video "):
                manifest_vid_nums.add(int(slot.split(" ", 1)[1]))
        subj_to_pic = _identity_subject_slots(manifest)
        used_subjects: set[int] = set()
        for s_idx, shot in enumerate(shots, start=1):
            shot_labels = []
            for ln in shot.get("prompt") or []:
                if isinstance(ln, str):
                    shot_labels.extend(find_native_labels(ln))
            shot_pic_nums = {n for k, n in shot_labels if k == "Picture"}
            for kind, num in shot_labels:
                if kind == "Audio":
                    errors.append(
                        f"shot {s_idx}: <Audio {num}> not allowed in ref2va; "
                        "pictures and videos only"
                    )
                    continue
                if kind == "Video":
                    if num in manifest_vid_nums:
                        continue
                    errors.append(
                        f"shot {s_idx}: <Video {num}> is not in the manifest; "
                        f"manifest videos: {sorted(manifest_vid_nums) or 'none'}"
                    )
                    continue
                if kind == "Picture":
                    if num in manifest_pic_nums:
                        continue
                    errors.append(
                        f"shot {s_idx}: <Picture {num}> is not in the manifest; "
                        f"manifest pictures: {sorted(manifest_pic_nums) or 'none'}"
                    )
                    continue
                # Subject N
                if num not in subj_to_pic:
                    errors.append(
                        f"shot {s_idx}: <Subject {num}> has no matching manifest "
                        "identity entry; bind it via role='identity' in the manifest."
                    )
                    continue
                used_subjects.add(num)
                pic_num = subj_to_pic[num]
                # The paired picture label must appear in the same shot
                # (otherwise the compiler cannot tell which <Picture>
                # <Subject> refers to).
                if pic_num not in shot_pic_nums:
                    errors.append(
                        f"shot {s_idx}: <Subject {num}> needs its paired <Picture {pic_num}> in the same scene"
                    )
        # Subject numbering contiguity: every identity slot must be
        # bound somewhere in the plan.
        missing = [k for k in sorted(subj_to_pic) if k not in used_subjects]
        if missing:
            errors.append(
                f"ref2va: identity Subject {missing} never appears in the plan; "
                "reference it from each shot that needs that identity."
            )
        return errors

    errors.append(f"unknown reference mode: {mode!r}")
    return errors


def _strip_outer_blanks(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _find_marker(lines: list[str], field: str, from_index: int) -> int:
    """Index of the line that is exactly ``<field>:`` (or bare ``<field>``),
    searched from ``from_index``; -1 when absent."""
    for i in range(from_index, len(lines)):
        stripped = lines[i].strip().rstrip(":").strip().lower()
        if stripped == field:
            return i
    return -1


def split_three_sections(
    text: str,
    *,
    music_default: str = DEFAULT_MUSIC_LINE,
) -> list[str]:
    """Split one clip's reply into the Production Plan prompt line-array.

    Expected reply shape (bare headers, one blank line between sections):

        integrated_multimodal_description:
        [Shot 1] ...

        overall_soundscape:
        ...

        non_diegetic_music:
        ...

    Tolerates a preamble before the first header, ``<think>`` blocks
    already removed by the caller, and a missing music section (filled
    with ``music_default``). Missing/empty description or soundscape
    raises ``ValueError`` (the caller's retry trigger).
    """
    if not text or not text.strip():
        raise ValueError("empty clip reply")
    lines = text.replace("\r\n", "\n").split("\n")

    desc_i = _find_marker(lines, SECTION_DESCRIPTION, 0)
    if desc_i < 0:
        raise ValueError(f"missing {SECTION_DESCRIPTION}: header")
    sound_i = _find_marker(lines, SECTION_SOUND, desc_i + 1)
    if sound_i < 0:
        raise ValueError(f"missing {SECTION_SOUND}: header")
    music_i = _find_marker(lines, SECTION_MUSIC, sound_i + 1)

    desc_body = _strip_outer_blanks(lines[desc_i + 1 : sound_i])
    if not desc_body:
        raise ValueError(f"empty {SECTION_DESCRIPTION} body")
    sound_body = _strip_outer_blanks(
        lines[sound_i + 1 : music_i if music_i >= 0 else len(lines)]
    )
    if not sound_body:
        raise ValueError(f"empty {SECTION_SOUND} body")
    if music_i >= 0:
        music_body = _strip_outer_blanks(lines[music_i + 1 :])
        if not music_body:
            music_body = [music_default]
    else:
        music_body = [music_default]

    return (
        [f"{SECTION_DESCRIPTION}:"]
        + desc_body
        + ["", f"{SECTION_SOUND}:"]
        + sound_body
        + ["", f"{SECTION_MUSIC}:"]
        + music_body
    )


def split_six_sections(
    text: str,
    *,
    music_default: str = DEFAULT_MUSIC_LINE,
) -> list[str]:
    """Split one Ref2VA clip's reply into the six-section Production
    Plan prompt line-array. Same tolerant pattern as ``split_three_sections``
    (preamble tolerated, headers found in order, outer blanks stripped,
    missing music filled with the default). Missing/empty
    subject_definitions / summary / retention_analysis /
    detailed_description / overall_soundscape raise ``ValueError`` —
    the contract is six mandatory sections, no join-everything fallback."""
    if not text or not text.strip():
        raise ValueError("empty clip reply")
    lines = text.replace("\r\n", "\n").split("\n")

    indices: list[int] = []
    cursor = 0
    music_idx = -1
    for field in SIX_SECTION_FIELDS:
        idx = _find_marker(lines, field, cursor)
        if idx < 0:
            if field == SECTION_MUSIC:
                # Music is OPTIONAL with default fill (mirrors the
                # three-section split's tolerant behaviour).
                cursor = len(lines)
                continue
            raise ValueError(f"missing {field}: header")
        if field == SECTION_MUSIC:
            music_idx = idx
        indices.append(idx)
        cursor = idx + 1

    bodies: list[list[str]] = []
    for i, field in enumerate(SIX_SECTION_FIELDS):
        if field == SECTION_MUSIC and music_idx < 0:
            bodies.append([music_default])
            continue
        start = indices[i] + 1
        # End is the next known section's index, or end-of-text.
        if i + 1 < len(indices):
            end = indices[i + 1]
        else:
            end = len(lines)
        body = _strip_outer_blanks(lines[start:end])
        if not body:
            if field == SECTION_MUSIC:
                body = [music_default]
            else:
                raise ValueError(f"empty {field} body")
        bodies.append(body)

    out: list[str] = [f"{SIX_SECTION_FIELDS[0]}:"]
    out += bodies[0]
    for i in range(1, len(SIX_SECTION_FIELDS)):
        out += ["", f"{SIX_SECTION_FIELDS[i]}:"]
        out += bodies[i]
    return out


def description_body(prompt_lines: list[str], *, schema: str = SCHEMA_THREE) -> str:
    """Body text of a split prompt's main description block — used to
    hand the FULL previous clip description to the next clip so
    mid-paragraph identity/prop anchors are not truncated away.

    Schema-aware: the three-section reply's description block is
    ``integrated_multimodal_description:``; the six-section reply's
    is ``detailed_description:``. For ``six_section`` a missing header
    raises ``ValueError`` (no join-everything fallback — that would
    silently smuggle subject_definitions / retention_analysis into the
    continuation handoff)."""
    if schema == SCHEMA_THREE:
        try:
            start = prompt_lines.index(f"{SECTION_DESCRIPTION}:") + 1
            end = prompt_lines.index(f"{SECTION_SOUND}:")
        except ValueError:
            return "\n".join(prompt_lines)
        return "\n".join(_strip_outer_blanks(prompt_lines[start:end]))
    if schema == SCHEMA_SIX:
        try:
            start = prompt_lines.index(f"{SECTION_DETAIL}:") + 1
            end = prompt_lines.index(f"{SECTION_SOUND}:")
        except ValueError as exc:
            raise ValueError(
                "six-section reply missing detailed_description: header"
            ) from exc
        return "\n".join(_strip_outer_blanks(prompt_lines[start:end]))
    raise ValueError(f"unknown schema: {schema!r}")


def sound_body(prompt_lines: list[str]) -> str:
    """The overall_soundscape body of a split prompt (the exact bed the
    next clip must carry across the boundary)."""
    try:
        start = prompt_lines.index(f"{SECTION_SOUND}:") + 1
        end = prompt_lines.index(f"{SECTION_MUSIC}:")
    except ValueError:
        return ""
    return "\n".join(_strip_outer_blanks(prompt_lines[start:end]))


def previous_tail(prompt_lines: list[str], max_chars: int = 480) -> str:
    """Tail of a clip's description (last non-empty lines, capped)."""
    body = description_body(prompt_lines)
    non_empty = [ln for ln in body.split("\n") if ln.strip()]
    tail = "\n".join(non_empty[-2:]) if non_empty else body
    if len(tail) > max_chars:
        tail = "..." + tail[-max_chars:]
    return tail


# --------------------------------------------------------------------------- #
# Pacing: beat density must scale with the clip's real duration
# --------------------------------------------------------------------------- #
def pacing_directive(seconds: Any, *, continued: bool = False) -> str:
    """Beat-density guidance scaled to the clip's actual raw duration.

    A thin prompt makes H3 dilate time — one small gesture stretched over
    10 seconds reads as slow motion. Naming an explicit beat budget keeps
    long clips densely choreographed and short clips uncluttered.
    ``continued=True`` (clips 2+) notes that the beat budget must fit the
    new action that follows the carried overlap, not the raw length."""
    s = float(seconds)
    if s <= 3:
        beats = "one single clear gesture"
        extra = "hold one camera state; do not add events"
    elif s <= 7:
        beats = "1-2 action beats"
        extra = "one camera development (the start or end of one move)"
    elif s <= 12:
        beats = "2-3 distinct action beats"
        extra = "a camera move that develops across the clip, plus one change of light or blocking"
    elif s <= 20.1:  # 20.1 covers the 20s request's grid value (481 = 20.04s)
        beats = "3-4 distinct action beats"
        extra = "evolving camera and blocking with at least one clear energy shift"
    else:
        beats = "4-6 distinct action beats"
        extra = "vary distance and energy across the clip; include one in-clip setup or location change if the story allows"
    text = (
        f"Pacing (this clip generates ~{s:.1f}s): plan {beats}. {extra}. "
        "Each beat is a short burst with a clear impact instant, not a "
        "stretched continuum. Write action at real-time speed — never "
        "stretch one small gesture across the whole duration (that renders "
        "as slow motion); if the action finishes early, start the next beat "
        "or develop the camera instead of slowing down. Match camera speed "
        "to content energy: energetic clips never take slow camera "
        "adjectives. Every 2-3 seconds must bring visible change: new "
        "action, camera motion, or light."
    )
    if continued:
        text += (
            " The opening overlap carried from the previous clip does not "
            "count toward this budget: fit the beats into the new action "
            "that follows it."
        )
    return text


# --------------------------------------------------------------------------- #
# Shots input parsing
# --------------------------------------------------------------------------- #
_KNOWN_SHOT_KEYS = (
    "id",
    "description",
    "prompt",
    "shot_type",
    "camera_movement",
    "transition_in",
    "duration_seconds",
    "narrative_beat",
    "characters",
    "props",
    "notes",
)
_ID_SAFE_RE = re.compile(r"[^0-9a-zA-Z_]+")


def slugify_id(raw: Any, index: int) -> str:
    """Production-Plan-safe shot id (unsafe filename chars -> ``_``,
    <= 96 chars); falls back to ``clip_NNNN`` (the plan node's default)."""
    text = _ID_SAFE_RE.sub("_", str(raw or "").strip().lower()).strip("_")
    text = re.sub(r"_+", "_", text)[:96]
    return text or f"clip_{index:04d}"


def parse_shots_text(text: str) -> list[dict]:
    """Parse the ``shots_text`` input into canonical shot dicts.

    Accepts (in order of attempt): a JSON array, a JSON object with a
    ``shots`` array, or plain text with one non-empty line per shot.
    Raises ``ValueError`` on empty input or an empty shot list.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("shots_text is empty — connect a storyboard or list one shot per line")

    items: list[Any] | None = None
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("shots"), list):
            items = data["shots"]
    if items is None:
        items = [ln for ln in raw.split("\n") if ln.strip()]

    shots: list[dict] = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, dict):
            raise ValueError(f"shot {i}: expected an object, got {type(item).__name__}")
        desc = str(item.get("description") or item.get("prompt") or "").strip()
        if not desc:
            raise ValueError(f"shot {i}: empty description")
        shot = {k: item[k] for k in _KNOWN_SHOT_KEYS if k in item}
        shot.pop("prompt", None)  # alias already folded into description
        shot["description"] = desc
        shot["id"] = slugify_id(item.get("id"), i)
        shots.append(shot)

    if not shots:
        raise ValueError("shots_text parsed to zero shots")
    if len(shots) > MAX_SHOTS:
        raise ValueError(f"{len(shots)} shots exceeds the plan maximum of {MAX_SHOTS}")
    return shots


# --------------------------------------------------------------------------- #
# Per-shot overrides: "id:length=N" / "id:seed=NNNN" / "id:steps=N"
# --------------------------------------------------------------------------- #
_OVERRIDE_KEYS = ("length", "seed", "steps")


def parse_per_shot_overrides(text: str) -> dict[str, dict[str, Any]]:
    """Parse override lines into ``{id_lower: {key: value}}``.

    ``length`` is validated against the grid, ``seed`` against uint64
    (stored as string), ``steps`` against 1..10000. Malformed lines,
    unknown keys, or bad values raise ``ValueError``.
    """
    overrides: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate((text or "").split("\n"), start=1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"per_shot_overrides line {line_no}: expected 'id:key=value', got {entry!r}")
        left, _, value = entry.partition("=")
        shot_id, sep, key = left.strip().partition(":")
        if not shot_id or not sep:
            raise ValueError(f"per_shot_overrides line {line_no}: expected 'id:key=value', got {entry!r}")
        key = key.strip().lower()
        value = value.strip()
        if key not in _OVERRIDE_KEYS:
            raise ValueError(
                f"per_shot_overrides line {line_no}: unknown key {key!r} (allowed: {', '.join(_OVERRIDE_KEYS)})"
            )
        if key == "length":
            n = int(value) if value.lstrip("-").isdigit() else -1
            if not is_valid_length(n):
                raise ValueError(
                    f"per_shot_overrides line {line_no}: length {value!r} is off the H3 grid (length % 17 == 5, 5..3592)"
                )
            overrides.setdefault(shot_id.lower(), {})["length"] = n
        elif key == "seed":
            if not value.isdigit() or int(value) > UINT64_MAX:
                raise ValueError(
                    f"per_shot_overrides line {line_no}: seed {value!r} is not a uint64"
                )
            overrides.setdefault(shot_id.lower(), {})["seed"] = value
        else:  # steps
            n = int(value) if value.lstrip("-").isdigit() else -1
            if not (MIN_STEPS <= n <= MAX_STEPS):
                raise ValueError(
                    f"per_shot_overrides line {line_no}: steps {value!r} outside 1..10000"
                )
            overrides.setdefault(shot_id.lower(), {})["steps"] = n
    return overrides


def apply_overrides(
    shots: list[dict],
    overrides: dict[str, dict[str, Any]],
) -> list[dict]:
    """Return shot copies with overrides applied. Unknown override ids
    raise ``ValueError`` (a typo must fail loudly, not silently pass)."""
    remaining = dict(overrides)
    out: list[dict] = []
    for shot in shots:
        shot = dict(shot)
        ov = remaining.pop(shot["id"].lower(), None)
        if ov:
            shot.update(ov)
        out.append(shot)
    if remaining:
        unknown = ", ".join(sorted(remaining))
        raise ValueError(f"per_shot_overrides reference unknown shot id(s): {unknown}")
    return out


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def derive_seed_base(seed_widget: Any) -> int:
    """Seed widget 0 means 'derive from wall clock' (each run differs);
    any other value is used verbatim as the chain's seed base."""
    seed = int(seed_widget or 0)
    if seed > 0:
        return seed % 1_000_000_000_000
    return int(time.time() * 1000) % 1_000_000_000_000


def derive_seed(seed_base: int, index: int, unified: bool = False) -> str:
    """Per-shot seed as a digit string (string keeps values above
    JavaScript's exact range intact, per the format guide).

    ``unified=True``: every clip shares ``seed_base`` — with the same
    noise initialization, cross-clip identity/style holds better.
    ``unified=False``: ``seed_base + index`` for per-clip diversity."""
    if unified:
        return str(int(seed_base))
    return str(int(seed_base) + int(index))


# --------------------------------------------------------------------------- #
# Plan assembly + validation
# --------------------------------------------------------------------------- #
def build_plan(
    shot_entries: list[dict],
    prefix_lines: list[str],
    default_steps: int = DEFAULT_STEPS,
) -> dict:
    """Assemble the strict plan dict.

    ``shot_entries``: ``{"id", "prompt": [lines], "length", "seed",
    "steps" (optional)}`` — field order matches the reference workflow's
    Production Plan JSON (id, prompt, steps, length, seed).
    """
    shots_out: list[dict] = []
    for entry in shot_entries:
        shot = {
            "id": entry["id"],
            "prompt": list(entry["prompt"]),
            "steps": int(entry.get("steps") or default_steps),
            "length": int(entry["length"]),
            "seed": str(entry["seed"]),
        }
        shots_out.append(shot)
    return {
        "defaults": {"steps": int(default_steps)},
        "shots": shots_out,
        "prompt_prefix": [str(ln) for ln in prefix_lines],
    }


def validate_plan(plan: dict, *, schema: str = SCHEMA_THREE) -> list[str]:
    """Return a list of contract violations (empty list = valid).

    Schema-aware:
    - Top-level keys: ``defaults`` + ``shots`` always required;
      ``prompt_prefix`` is OPTIONAL (D3: keyframe modes may emit just
      defaults + shots; ref2va always emits one). Any other top-level
      key is an error.
    - Three-section: each shot's prompt array must start with
      ``integrated_multimodal_description:`` and end with
      ``non_diegetic_music:``.
    - Six-section: each shot's prompt array must contain all six bare
      Ref2V headers in exact order.
    """
    errors: list[str] = []
    top = set(plan.keys())
    # ``prompt_prefix`` is OPTIONAL (D3); only unknown extras or a
    # missing ``defaults`` / ``shots`` are errors.
    if top - {"defaults", "shots", "prompt_prefix"}:
        errors.append(
            f"top-level keys must be a subset of {{'defaults', 'shots', 'prompt_prefix'}}, got {sorted(top)}"
        )
    if "defaults" not in top:
        errors.append("top-level key 'defaults' is required")
    if "shots" not in top:
        errors.append("top-level key 'shots' is required")
    shots = plan.get("shots")
    if not isinstance(shots, list) or not (MIN_SHOTS <= len(shots) <= MAX_SHOTS):
        errors.append(f"shots must be a list of {MIN_SHOTS}..{MAX_SHOTS} entries")
        return errors
    # schema-specific prompt contract
    if schema == SCHEMA_SIX:
        first_header = SECTION_SUBJECT
        required_headers = list(SIX_SECTION_FIELDS)
    else:
        first_header = SECTION_DESCRIPTION
        required_headers = [SECTION_DESCRIPTION, SECTION_SOUND, SECTION_MUSIC]
    seen_ids: set[str] = set()
    for i, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            errors.append(f"shot {i}: not an object")
            continue
        sid = shot.get("id")
        if not isinstance(sid, str) or not sid.strip():
            errors.append(f"shot {i}: missing id")
        elif sid in seen_ids:
            errors.append(f"shot {i}: duplicate id {sid!r}")
        else:
            seen_ids.add(sid)
        prompt = shot.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not all(isinstance(ln, str) for ln in prompt)
        ):
            errors.append(f"shot {i}: prompt must be a non-empty array of strings")
        elif schema == SCHEMA_THREE:
            if prompt[0].strip().lower().rstrip(":") != SECTION_DESCRIPTION:
                errors.append(
                    f"shot {i}: prompt must start with '{SECTION_DESCRIPTION}:'"
                )
        elif schema == SCHEMA_SIX:
            stripped = [
                (ln.strip().rstrip(":").strip().lower() if isinstance(ln, str) else "")
                for ln in prompt
            ]
            if stripped[0] != first_header:
                errors.append(
                    f"shot {i}: prompt must start with '{first_header}:'"
                )
            for field in required_headers:
                if field not in stripped:
                    errors.append(
                        f"shot {i}: prompt is missing required header '{field}:'"
                    )
            # Headers must be in the official order.
            positions = [
                (stripped.index(f), f)
                for f in required_headers
                if f in stripped
            ]
            if [f for _, f in sorted(positions)] != required_headers:
                errors.append(
                    f"shot {i}: prompt headers must be in order {required_headers}"
                )
        else:
            errors.append(f"shot {i}: unknown schema {schema!r}")
        if not is_valid_length(shot.get("length")):
            errors.append(f"shot {i}: length {shot.get('length')!r} off the 17k+5 grid")
        seed = shot.get("seed")
        if not isinstance(seed, str) or not seed.isdigit() or int(seed) > UINT64_MAX:
            errors.append(f"shot {i}: seed must be a uint64 digit string")
        steps = shot.get("steps")
        if not isinstance(steps, int) or not (MIN_STEPS <= steps <= MAX_STEPS):
            errors.append(f"shot {i}: steps {steps!r} outside 1..10000")
        extra = set(shot.keys()) - {"id", "prompt", "steps", "length", "seed"}
        if extra:
            errors.append(f"shot {i}: unexpected keys {sorted(extra)}")
    prefix = plan.get("prompt_prefix")
    if prefix is not None:
        # D3: prompt_prefix may be absent (defaults/shots only) OR an
        # empty list (ref2va style-only prefix); when present and
        # non-empty, every line must be a non-empty string.
        if not isinstance(prefix, list) or not all(
            isinstance(ln, str) and ln.strip() for ln in prefix
        ):
            errors.append(
                "prompt_prefix, when present, must be an array of non-empty strings"
            )
    defaults = plan.get("defaults")
    if not isinstance(defaults, dict) or set(defaults.keys()) != {"steps"}:
        errors.append("defaults must contain exactly a 'steps' key")
    return errors


def plan_to_json_string(plan: dict) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def build_preflight_report(
    plan: dict,
    warnings: list[str] | None = None,
    *,
    mode: str = "t2va",
    manifest: list[dict] | None = None,
) -> str:
    """Preflight report covering the deterministic plan shape AND the
    reference-mode wiring (mode, manifest summary, per-mode label
    usage, and D10 graph wiring hints — those hints are informational
    and NEVER written into plan JSON)."""
    shots = plan.get("shots") or []
    total_frames = sum(int(s.get("length") or 0) for s in shots)
    code = parse_reference_mode(mode)
    lines = [
        "H3 Loop Plan preflight",
        f"Reference mode: {code}",
        f"Shots: {len(shots)}",
        f"Total raw length: {total_frames} frames ({length_to_seconds(total_frames) if total_frames else 0:.2f}s at {FPS}fps)",
        f"prompt_prefix: {len(plan.get('prompt_prefix') or [])} line(s)",
        "",
    ]
    if manifest is not None:
        lines.append(f"Manifest entries: {len(manifest)}")
        for entry in manifest:
            slot = entry.get("slot")
            about = (entry.get("about") or "").strip()
            role = entry.get("role")
            about_brief = about if len(about) <= 80 else about[:77] + "..."
            lines.append(f"  - {slot} [{role}]: {about_brief}")
        lines.append("")

    # Per-mode native-label usage summary (counts only — policy errors
    # are surfaced by validate_label_policy, not here).
    counts: dict[str, int] = {"Picture": 0, "Video": 0, "Audio": 0, "Subject": 0}
    alias_total = 0
    anchor_total = 0
    for shot in shots:
        for ln in shot.get("prompt") or []:
            if not isinstance(ln, str):
                continue
            for kind, _num in find_native_labels(ln):
                counts[kind] = counts.get(kind, 0) + 1
            alias_total += len(find_alias_tokens(ln))
            anchor_total += len(find_semantic_anchors(ln))
    lines.append(
        "Label usage: "
        f"Pictures={counts['Picture']}, "
        f"Videos={counts['Video']}, "
        f"Audios={counts['Audio']}, "
        f"Subjects={counts['Subject']}, "
        f"@aliases={alias_total}, "
        f"#tags={anchor_total}"
    )
    lines.append("")

    for i, s in enumerate(shots, start=1):
        lines.append(
            f"  {i}. {s.get('id')}  length={s.get('length')} "
            f"({length_to_seconds(s.get('length') or 0):.2f}s)  steps={s.get('steps')}  "
            f"seed={s.get('seed')}  prompt_lines={len(s.get('prompt') or [])}"
        )

    # D10 graph wiring hints — informational, never written into plan JSON.
    if code == "i2va":
        lines.append("")
        lines.append(
            "graph wiring: LoadImage -> MiniMax H3 First-Scene Image Gate -> "
            "MiniMax H3 Image to Video first_frame (Plan default "
            "context_length=22 works; no override needed)"
        )
        lines.append(
            "note: the Scene Prompt Editor may classify these prompts as T2VA "
            "(the official I2V/FL2V workflows trip the same editor check); "
            "generation is unaffected."
        )
    elif code == "fl2va":
        lines.append("")
        lines.append(
            "graph wiring: 2x LoadImage -> MiniMax H3 Chain Frame Index Switch -> "
            "First-Scene Image Gate last_frame; Gate image = opening frame -> "
            "Image to Video first_frame + last_frame"
        )
        lines.append(
            "note: the Scene Prompt Editor may classify these prompts as T2VA "
            "(the official I2V/FL2V workflows trip the same editor check); "
            "generation is unaffected."
        )
    elif code == "ref2va":
        lines.append("")
        lines.append(
            "graph wiring: N x LoadImage -> MiniMax H3 Reference to Video images "
            "(same pictures active for every scene)"
        )

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in warnings)
    else:
        lines.append("")
        lines.append("Warnings: none")
    return "\n".join(lines)


def build_plan_preview(plan: dict) -> str:
    rows = ["| # | id | length | ~s | seed | first prompt line |", "|---|----|--------|----|------|-------------------|"]
    for i, s in enumerate(plan.get("shots") or [], start=1):
        first = next(
            (ln for ln in (s.get("prompt") or []) if ln.strip()),
            "",
        )
        first = first.replace("|", "\\|")
        if len(first) > 60:
            first = first[:57] + "..."
        rows.append(
            f"| {i} | `{s.get('id')}` | {s.get('length')} | "
            f"{length_to_seconds(s.get('length') or 0):.1f} | {s.get('seed')} | {first} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
SHOT_SYSTEM_ADDENDUM = load_prompt_text("h3_loop/shot_system_addendum")
_CONTINUATION_TEMPLATE = load_prompt_text("h3_loop/shot_continuation")
_CONTINUATION_REF2V_TEMPLATE = load_prompt_text("h3_loop/shot_continuation_ref2v")
_SHOT_USER_TEMPLATE = load_prompt_text("h3_loop/shot_user_template")
PREFIX_SYNTH_SYSTEM = load_prompt_text("h3_loop/prefix_synth_system")
_PREFIX_SYNTH_USER_TEMPLATE = load_prompt_text("h3_loop/prefix_synth_user")
SINGLE_CALL_FORMAT = load_prompt_text("h3_loop/single_call_format")
REF2V_ADDENDUM = load_prompt_text("h3_loop/ref2v_addendum")


def shot_system_prompt() -> str:
    """Stage-2 system prompt: the shared official H3 t2v guide plus the
    loop addendum (single continuous clip, bare headers, chaining)."""
    return f"{system_t2v_prompt().rstrip()}\n\n{SHOT_SYSTEM_ADDENDUM.strip()}\n"


def shot_system_prompt_ref2v() -> str:
    """Stage-2 system prompt for ref2va: the shared H3 reference guide
    plus the six-section Ref2V loop addendum (subject_definitions
    binding, summary `[reference generation]`, retention markers,
    detailed_description with `[Shot 1]`, deterministic subject binding).
    """
    return f"{system_reference_prompt().rstrip()}\n\n{REF2V_ADDENDUM.strip()}\n"


def _genre_advice_block(category: str) -> str:
    code = parse_category(category)
    advice = category_advice(code).strip()
    if advice:
        return f"Genre guidance ({code}): {advice}"
    return "Genre guidance: none; follow the concept as written."


def build_prefix_user_text(
    concept: str,
    category: str,
    language_name: str,
    shots_digest: str = "",
    *,
    mode_note: str = "",
    manifest_digest: str = "",
) -> str:
    return _PREFIX_SYNTH_USER_TEMPLATE.format(
        concept=(concept or "").strip(),
        storyboard_digest=(shots_digest or "(no storyboard provided)").strip(),
        genre_advice=_genre_advice_block(category),
        language_name=language_name,
        mode_note=(mode_note or "").strip(),
        manifest_digest=(manifest_digest or "no reference images").strip(),
    )


def build_shots_digest(shots: list[dict], max_desc_chars: int = 160) -> str:
    """One line per shot (``- id: description``) grounding the prefix
    synthesis in the actual storyboard — without this, a thin concept
    lets the prefix drift (e.g. inventing human protagonists)."""
    lines: list[str] = []
    for shot in shots:
        desc = str(shot.get("description") or "").strip().replace("\n", " ")
        if len(desc) > max_desc_chars:
            desc = desc[: max_desc_chars - 3] + "..."
        lines.append(f"- {shot.get('id')}: {desc}")
    return "\n".join(lines)


def build_continuation_block(
    previous_id: str,
    previous_description: str,
    previous_soundscape: str,
) -> str:
    """Continuation block for clips 2+: the FULL previous description
    plus its exact overall_soundscape bed, so identity/prop anchors
    (wherever they sit in the paragraph) survive the handoff and the
    carried ambience is real, not reinvented."""
    return _CONTINUATION_TEMPLATE.format(
        previous_id=previous_id,
        previous_description=(previous_description or "").strip(),
        previous_soundscape=(previous_soundscape or "").strip()
        or "(the previous clip established no explicit bed; keep silence-adjacent continuity)",
    )


def build_continuation_block_ref2v(
    previous_id: str,
    previous_subject_definitions,
    previous_description: str,
    previous_soundscape: str,
) -> str:
    """Ref2VA continuation block (D5/D4): six-section carry-over rules.
    Re-asserts subject_definitions in this scene own voice, preserves
    the [reference generation] summary tag, carries the exact
    overall_soundscape bed, and ends mid-action unless final.

    previous_subject_definitions may be a string (already joined) or a
    list of body lines (as carried from the previous shot prompt)."""
    if isinstance(previous_subject_definitions, list):
        joined_subject = "\n".join(
            ln for ln in previous_subject_definitions if isinstance(ln, str)
        ).strip()
    else:
        joined_subject = (previous_subject_definitions or "").strip()
    return _CONTINUATION_REF2V_TEMPLATE.format(
        previous_id=previous_id,
        previous_subject_definitions=joined_subject,
        previous_description=(previous_description or "").strip(),
        previous_soundscape=(previous_soundscape or "").strip()
        or "(the previous clip established no explicit bed; keep silence-adjacent continuity)",
    )


def build_reference_directive(
    mode: str,
    manifest: list[dict],
    clip_index: int,
    seconds: Any,
) -> str:
    """Per-mode reference directive woven into the first sentence of
    the description block (D6). Returns "" for t2va (no labels).

    - i2va scene 1: official opening idiom with Picture 1'''s about text.
    - fl2va scene 1: D6 scene-1 idiom naming both pictures.
    - fl2va scene N>=2: D6 L2VA end-target sentence converging on
      ``<Picture (N % 2) + 1>`` (alternates between Picture 1 / 2).
    - ref2va: deterministic subject_definitions + summary +
      retention_analysis construction guide for the LLM (D5)."""
    code = parse_reference_mode(mode)
    if code == "t2va":
        return ""
    if code == "i2va":
        if clip_index != 1 or not manifest:
            return ""
        about = manifest[0].get("about", "")
        return (
            f"At 0.00 seconds, <Picture 1> is fully referenced as the opening frame. "
            f"Animate the exact {about} shown in <Picture 1>."
        )
    if code == "fl2va":
        if clip_index == 1:
            if len(manifest) < 2:
                return ""
            a1 = manifest[0].get("about", "")
            a2 = manifest[1].get("about", "")
            return (
                f"<Picture 1> aligns with 0.00 seconds and <Picture 2> aligns "
                f"with the final target frame. Begin from the exact {a1} shown in "
                f"<Picture 1>. Progressively match the {a2} in <Picture 2>. "
                f"Reach <Picture 2> only on the final frame; do not freeze early or cut."
            )
        # Scenes 2+: the L2VA gate exposes ONE image under <Picture 1> -
        # the per-scene end target. Pick that image's manifest entry when
        # the user supplied one (5-picture layout: Picture 1 = opening,
        # Picture k>=2 = scene k's end target). Fall back to alternating
        # manifest[0] / manifest[1] for legacy 2-image layouts so old
        # workflows keep their A->B->A rhythm.
        if clip_index < len(manifest):
            about = manifest[clip_index].get("about", "")
        elif clip_index % 2 == 0:
            about = manifest[0].get("about", "")
        else:
            about = manifest[1].get("about", "") if len(manifest) > 1 else ""
        return (
            f"During the final seconds, progressively align the visible scene with "
            f"{about} shown in <Picture 1>, reaching that picture only on the "
            f"final frame without a cut or early hold."
        )
    if code == "ref2va":
        if not manifest:
            return ""
        identity_slots: list[str] = []
        for i, entry in enumerate(manifest, start=1):
            if entry.get("role") == "identity":
                identity_slots.append(f"Picture {i}")
        subj_list = ", ".join(
            f"<Subject {n + 1}>" for n in range(len(identity_slots))
        )
        lines: list[str] = [
            "[Deterministic subject binding — DO NOT INVENT NEW SUBJECTS]",
            f"Manifest slots (in order): {', '.join(f'<Picture {i + 1}>' for i in range(len(manifest)))}.",
        ]
        if identity_slots:
            lines.append(
                f"Identity slots: {', '.join(identity_slots)} bind to "
                f"{subj_list} (numbered in slot order, 1-indexed)."
            )
            for i, slot in enumerate(identity_slots, start=1):
                pic_num = int(slot.split(" ", 1)[1])
                about = next(
                    (m.get("about", "") for m in manifest if m.get("slot") == slot),
                    "",
                )
                lines.append(
                    f"  <Subject {i}> -> <Picture {pic_num}>: {about}"
                )
        else:
            lines.append(
                "No identity-role slots in the manifest; describe every picture "
                "by its <about> in subject_definitions (no <Subject N> binding)."
            )
        non_identity = [m for m in manifest if m.get("role") != "identity"]
        if non_identity:
            lines.append(
                "Non-identity slots stay plain pictures — describe them by their "
                "<about> (destination / environment), no <Subject N> binding."
            )
        lines.append(
            "summary MUST start with '[reference generation]' (Ref2VA task type; "
            "do NOT use '[video continuation + ...]' here)."
        )
        lines.append(
            "retention_analysis: identity slots -> 'fully_preserved - ...'; "
            "non-identity slots -> 'reference - ...'. Every Picture / Subject "
            "label used in this scene MUST appear in retention_analysis."
        )
        if any((m.get("slot") or "").startswith("Video ") for m in manifest):
            video_slots = ", ".join(
                f"<{m['slot']}>" for m in manifest
                if (m.get("slot") or "").startswith("Video ")
            )
            lines.append(
                f"VIDEO MOTION REFERENCE: {video_slots} is a performance / "
                "motion reference. The on-screen performer must reproduce "
                "that reference's choreography and performance timing "
                "exactly, for the FULL duration of this clip. Its setting, "
                "wardrobe, and camera framing are NOT part of the reference "
                "unless this scene's description says otherwise; describe "
                "this scene's own environment and camera in "
                "detailed_description. retention_analysis must list the "
                "video slot as the motion source ('reference - choreography "
                "and timing')."
            )
        return "\n".join(lines)
    return ""


# --------------------------------------------------------------------------- #
# Deterministic keyframe-idiom enforcement (live E2E finding: the LLM
# reliably mangles the official opening idiom — drops the angle brackets,
# merges sentences, or skips the scene-2 end target — and the stock
# I2V/FL2V tokenizer needs the literal <Picture N> tokens).
# --------------------------------------------------------------------------- #
_I2VA_IDIOM_RE = re.compile(
    r"At\s+0\.00\s+seconds,?\s*<?\s*Picture\s*1\s*>?\s*"
    r"is\s+fully\s+referenced\s+as\s+the\s+opening\s+frame\.?",
    re.IGNORECASE,
)
_FL2VA_IDIOM_RE = re.compile(
    r"<?\s*Picture\s*1\s*>?\s*aligns\s+with\s+0\.00\s+seconds\s+and\s+"
    r"<?\s*Picture\s*2\s*>?\s*aligns\s+with\s+the\s+final\s+target\s+frame\.?",
    re.IGNORECASE,
)
_FL2VA_REACH_RE = re.compile(
    r"<\s*Picture\s*2\s*>[^.\n]{0,80}only\s+on\s+the\s+final\s+frame"
    r"|only\s+on\s+the\s+final\s+frame[^.\n]{0,80}<\s*Picture\s*2\s*>",
    re.IGNORECASE,
)
_FL2VA_END_TARGET_RE_TMPL = (
    r"<\s*Picture\s*{target}\s*>[^.\n]{{0,160}}final\s+frame"
    r"|final\s+frame[^.\n]{{0,160}}<\s*Picture\s*{target}\s*>"
)


def ensure_keyframe_idiom(
    lines: list[str],
    *,
    mode: str,
    manifest: list[dict],
    clip_index: int,
) -> list[str]:
    """Enforce the official keyframe idiom on a split THREE-section prompt.

    i2va scene 1 / fl2va scene 1: the canonical bracketed opening sentence
    replaces whatever idiom-shaped segment the LLM wrote on the first
    description body line (bracket-less variants match); when no idiom is
    present the canonical sentence is prepended. fl2va additionally
    guarantees the scene-1 reach sentence and the scene-N>=2 end-target
    sentence (appended at the end of the description body when absent).
    Returns a NEW list; t2va/ref2va return ``lines`` unchanged.
    """
    code = parse_reference_mode(mode)
    if code not in ("i2va", "fl2va"):
        return lines
    out = list(lines)
    try:
        desc_header = out.index(f"{SECTION_DESCRIPTION}:")
        sound_header = out.index(f"{SECTION_SOUND}:")
    except ValueError:
        return out
    body_start = desc_header + 1
    while body_start < sound_header and not out[body_start].strip():
        body_start += 1
    if body_start >= sound_header:
        return out

    if code == "i2va":
        if clip_index != 1:
            return out  # scenes 2+ must stay label-free
        canonical = (
            "At 0.00 seconds, <Picture 1> is fully referenced as the opening frame."
        )
        _replace_or_prepend_idiom(out, body_start, _I2VA_IDIOM_RE, canonical)
        return out

    # fl2va
    if clip_index == 1:
        if len(manifest) < 2:
            return out
        canonical = (
            "<Picture 1> aligns with 0.00 seconds and <Picture 2> aligns "
            "with the final target frame."
        )
        _replace_or_prepend_idiom(out, body_start, _FL2VA_IDIOM_RE, canonical)
        # Guarantee the reach sentence closes the description body.
        body_text = "\n".join(out[body_start:sound_header])
        if not _FL2VA_REACH_RE.search(body_text):
            insert_at = sound_header
            while insert_at > body_start and not out[insert_at - 1].strip():
                insert_at -= 1
            out.insert(
                insert_at,
                "Reach <Picture 2> only on the final frame; do not freeze early or cut.",
            )
        return out

    # fl2va scene N>=2: L2VA end-target sentence. The check requires
    # the literal <Picture 1> token near a final-frame phrase — live E2E
    # showed the LLM writes token-free paraphrases ("aligning with the
    # warrior's stance on the final frame") that must NOT count as
    # present. The label is always <Picture 1> (verified against the
    # upstream gate wiring — FrameIndexSwitch exposes the per-scene
    # image under Picture 1, not Picture 2).
    end_re = re.compile(
        _FL2VA_END_TARGET_RE_TMPL.format(target=1), re.IGNORECASE
    )
    body_text = "\n".join(out[body_start:sound_header])
    if end_re.search(body_text):
        return out
    if clip_index < len(manifest):
        about = manifest[clip_index].get("about", "")
    elif clip_index % 2 == 0:
        about = manifest[0].get("about", "")
    else:
        about = manifest[1].get("about", "") if len(manifest) > 1 else ""
    insert_at = sound_header
    while insert_at > body_start and not out[insert_at - 1].strip():
        insert_at -= 1
    out.insert(
        insert_at,
        f"During the final seconds, progressively align the visible scene with "
        f"{about} shown in <Picture 1>, reaching that picture only on the "
        f"final frame without a cut or early hold.",
    )
    return out


def _replace_or_prepend_idiom(
    out: list[str], body_start: int, pattern: "re.Pattern[str]", canonical: str
) -> None:
    """Replace the idiom-shaped segment on the first body line with the
    canonical bracketed sentence; prepend as its own line when absent."""
    first = out[body_start]
    if pattern.search(first):
        out[body_start] = pattern.sub(canonical, first, count=1)
        # Collapse an accidental double space where the segment ended mid-line.
        out[body_start] = out[body_start].replace("  ", " ", 1)
        return
    out.insert(body_start, canonical)


def _manifest_digest(manifest: list[dict]) -> str:
    """One line per manifest slot for the user template: ``- <Picture
    N> [<role>]: <about>``. Empty manifest -> ``no reference images``."""
    if not manifest:
        return "no reference images"
    out: list[str] = []
    for i, entry in enumerate(manifest, start=1):
        slot = entry.get("slot") or f"Picture {i}"
        about = (entry.get("about") or "").strip()
        role = entry.get("role") or "destination"
        about_brief = about if len(about) <= 120 else about[:117] + "..."
        out.append(f"- {slot} [{role}]: {about_brief}")
    return "\n".join(out)


def _mode_note_for_prefix(mode: str) -> str:
    """Per-mode policy note for the prefix synthesis user template
    (D3): identity words vs. style-only vs. style/setting stub."""
    code = parse_reference_mode(mode)
    if code == "t2va":
        return (
            "Prefix policy: include the protagonist(s)''' exact identity, "
            "wardrobe, recurring props, setting, lighting, palette, tempo, "
            "art style, and global exclusions."
        )
    if code == "i2va" or code == "fl2va":
        return (
            "Prefix policy (keyframe mode): NO subject appearance words, "
            "identities, or recurring props. The opening keyframe image "
            "supplies identity; the per-clip continuation wording re-asserts "
            "identity when needed. Cover ONLY style, lighting, palette, "
            "tempo, art style, and the global exclusions (no text, no logo, "
            "no extra people, no non-diegetic music)."
        )
    if code == "ref2va":
        return (
            "Prefix policy (ref2va): SHORTER style + setting stub. Identity "
            "for every recurring subject is established in each scene'''s "
            "subject_definitions (deterministic binding to <Subject N>), NOT "
            "in the shared prefix. Cover ONLY the overall art style, light "
            "direction, palette, tempo, and global exclusions."
        )
    return ""


def build_shot_user_text(
    *,
    concept: str,
    prefix_text: str,
    category: str,
    continuation_block: str,
    shot: dict,
    clip_index: int,
    clip_count: int,
    width: int,
    height: int,
    duration_seconds: Any,
    language_name: str,
    reference_directive: str = "",
    manifest_digest: str = "",
) -> str:
    # ``duration_seconds`` should be the clip's ACTUAL grid-rounded length
    # (``length_to_seconds(length)``) so the pacing budget matches what H3
    # will really generate, not the requested seconds.
    seconds = float(duration_seconds)
    return _SHOT_USER_TEMPLATE.format(
        concept=(concept or "").strip(),
        prompt_prefix=prefix_text.strip(),
        genre_advice=_genre_advice_block(category),
        continuation_block=continuation_block.strip(),
        clip_index=int(clip_index),
        clip_count=int(clip_count),
        shot_json=json.dumps(shot, ensure_ascii=False, indent=2),
        width=int(width),
        height=int(height),
        aspect=aspect_ratio_string(int(width), int(height)),
        duration_seconds=seconds,
        pacing_directive=pacing_directive(seconds, continued=clip_index > 1),
        language_name=language_name,
        reference_directive=(reference_directive or "").strip(),
        manifest_digest=(manifest_digest or "no reference images").strip(),
    )


def build_single_call_user_text(
    *,
    concept: str,
    prefix_text: str,
    category: str,
    shots: list[dict],
    width: int,
    height: int,
    duration_seconds: int,
    language_name: str,
) -> str:
    board = json.dumps(shots, ensure_ascii=False, indent=2)
    return (
        f"Concept (whole production):\n{(concept or '').strip()}\n\n"
        f"Shared identity / style prefix (binding for every clip):\n{prefix_text.strip()}\n\n"
        f"{_genre_advice_block(category)}\n\n"
        f"Storyboard entries (ALL {len(shots)} clips, in order):\n{board}\n\n"
        f"Canvas: {int(width)}x{int(height)} ({aspect_ratio_string(int(width), int(height))}). "
        f"Clip duration: {int(duration_seconds)} seconds. Output language: {language_name}.\n\n"
        f"{SINGLE_CALL_FORMAT.format(clip_count=len(shots)).strip()}"
    )


def split_prefix_paragraphs(text: str) -> list[str]:
    """Split a prefix synthesis reply into paragraphs (blank-line
    separated, outer whitespace stripped, empties dropped)."""
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()
    ]
    return paragraphs


def log_pipeline(message: str) -> None:
    mie_log(f"H3LOOP: {message}")
