"""Deterministic plan-building layer for the ``MiniMaxH3LoopPromptGenerator``.

Owns everything that must NOT depend on LLM whims:

* H3 length grid math — valid raw lengths satisfy ``length % 17 == 5``
  within 5..3592 at 24 fps (``seconds_to_length`` / ``is_valid_length``);
* ``split_three_sections`` — turn one clip's LLM reply into the exact
  prompt line-array shape used by the Production Plan workflow
  (``["integrated_multimodal_description:", ..., "", "overall_soundscape:",
  ..., "", "non_diegetic_music:", ...]``);
* ``build_plan`` + ``validate_plan`` — assemble and assert the strict
  plan shape (top-level keys ``shots`` / ``prompt_prefix`` only;
  ``prompt`` and ``prompt_prefix`` are arrays; seeds are digit strings;
  no steps / duration_seconds / continuation_mode / context_length /
  width / height ever enters the JSON — generation parameters live on
  the Plan node's widgets);
* preflight report + markdown preview rendering;
* prompt-template builders for the three LLM stages (storyboard split
  with a whole-board duration budget, prefix synthesis, per-shot
  generation, single-call generation);
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
from typing import Any, Optional

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text

try:
    from _mienodes_internal.nodes.llm.h3_prompts import (
        category_advice,
        parse_category,
        system_t2v_prompt,
        system_reference_prompt,
    )
except ImportError:
    from .h3_prompts import (
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
MIN_SHOTS = 1
MAX_SHOTS = 128

# Whole-board duration budget default (seconds). The storyboard LLM
# distributes it across shots; each shot's length then rounds UP onto
# the 17k+5 grid, so the delivered total lands close to this.
DEFAULT_TOTAL_DURATION_SECONDS = 15


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


def parse_references_text(
    text: str,
    *,
    reference_mode: Optional[str] = None,
) -> list[dict]:
    """Parse ``references_text`` into a canonical manifest list.

    Each entry is ``{"slot": "Picture 1", "about": "...", "role":
    "identity" | "destination" | "environment"}``. Accepts:

    - A strict JSON array of objects (keys: ``slot`` / ``about`` /
      ``role``; ``picture`` / ``label`` are aliases for ``slot``;
      ``description`` / ``text`` are aliases for ``about``).
    - One natural line per picture: ``Picture 1: courier face, ...``.
      The ``<Picture 1>`` bracket form is also accepted.

    Role defaults (applied in DESCENDING PRIORITY — higher wins):
      1. Explicit ``role`` key in the parsed line wins always.
      2. When ``reference_mode`` is explicitly ``ref2va``: EVERY Picture
         slot defaults to ``identity`` (Ref2VA = every reference image
         is an identity anchor; destinations are explicitly opted-in).
      3. Otherwise: slot 1 -> ``identity``; other slots -> ``destination``.

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
    ref_code = parse_reference_mode(reference_mode) if reference_mode else None
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
        # Role defaults, applied in DESCENDING PRIORITY:
        #   explicit role (user wrote it) > ref2va-mode broad default >
        #   legacy i==1 default.
        explicit_role = entry.get("role")
        if explicit_role:
            default_role = str(explicit_role).strip().lower()
        elif ref_code == "ref2va" or i == 1:
            default_role = "identity"
        else:
            default_role = "destination"
        role = default_role
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


def validate_label_policy(
    plan: dict,
    mode: str,
    manifest: list[dict],
    referenced_pictures: Optional[set] = None,
) -> list[str]:
    """Per-mode native-label + @alias / #tag contract. Returns errors.
    Empty list = the plan's text matches the mode's label contract.

    Universal: any ``@alias`` or ``#tag`` token anywhere in the plan
    text is an error (D9). P0 does not support Scheduled Ref2VA / the
    Scene Prompt Editor's dialogue markup — only native labels.

    ``referenced_pictures`` (ref2va only): the picture slot numbers the
    CONCEPT text actually names (图1 / Picture 2 ...). Identity Subjects
    bound to pictures outside that set are NOT required to appear in
    the plan (live failure 2026-09-21: 5 wired pictures, concept named
    only 图1-3, the model correctly referenced Subjects 1-3 and the
    whole 6-minute run died at final validation demanding Subjects 4/5).
    ``None`` keeps the strict all-slots contract (tests, direct callers);
    an empty set falls back to strict too — a concept that names no
    pictures gets the conservative reading."""
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
        # Subjects bound to pictures the CONCEPT never names: optional
        # (their absence is fine — see required_subjects below) and,
        # when the model mentions one anyway, the paired-<Picture> rule
        # is waived too — the mention is harmless noise about an unused
        # reference; failing the whole post-spend run on it destroyed a
        # 226s live board (2026-09-21 20:58).
        optional_subject_pics: set = set()
        if referenced_pictures:
            optional_subject_pics = {
                pic for pic in manifest_pic_nums if pic not in referenced_pictures
            }
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
                # <Subject> refers to) — EXCEPT for Subjects bound to
                # pictures the concept never uses (optional mentions).
                if (
                    pic_num not in shot_pic_nums
                    and pic_num not in optional_subject_pics
                ):
                    errors.append(
                        f"shot {s_idx}: <Subject {num}> needs its paired <Picture {pic_num}> in the same scene"
                    )
        # Subject numbering contiguity: every identity slot the CONCEPT
        # actually uses must be bound somewhere in the plan. Pictures the
        # concept never names are optional (their Subjects may appear —
        # the model often defines them — but are never REQUIRED; the
        # preflight already warns about the unused wiring).
        required_subjects = set(subj_to_pic)
        if referenced_pictures:
            required_subjects = {
                k for k, pic in subj_to_pic.items() if pic in referenced_pictures
            }
        missing = [
            k for k in sorted(required_subjects) if k not in used_subjects
        ]
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
def pacing_directive(seconds: Any, *, continued: bool = False, reference_mode: str = "t2va") -> str:
    """Beat-density guidance scaled to the clip's actual raw duration.

    A thin prompt makes H3 dilate time — one small gesture stretched over
    10 seconds reads as slow motion. Naming an explicit beat budget keeps
    long clips densely choreographed and short clips uncluttered.
    ``continued=True`` (clips 2+) notes that the beat budget must fit the
    new action that follows the carried overlap, not the raw length.

    Keyframe modes (fl2va / i2va) have a heavier load because every
    action beat also has to anchor against a wired reference image —
    too many beats crowds the frame and dilutes the anchor. We halve
    the budget for these modes plus add a "carry-over is a state delta,
    not a recap" rule so the LLM doesn't restate the previous beat
    inside the current scene's body."""
    s = float(seconds)
    is_keyframe = reference_mode in ("fl2va", "i2va")
    if s <= 3:
        beats = "one single clear gesture"
        extra = "hold one camera state; do not add events"
    elif s <= 7:
        beats = "1-2 action beats" if not is_keyframe else "1 short action beat"
        extra = (
            "one camera development (the start or end of one move)"
            if not is_keyframe
            else "keep the beat simple so the reference image stays the visual anchor"
        )
    elif s <= 12:
        beats = "2-3 distinct action beats" if not is_keyframe else "1-2 distinct action beats"
        extra = (
            "a camera move that develops across the clip, plus one change of light or blocking"
            if not is_keyframe
            else "one camera development; keep the beat simple so the reference image stays the visual anchor"
        )
    elif s <= 20.1:  # 20.1 covers the 20s request's grid value (481 = 20.04s)
        beats = "3-4 distinct action beats" if not is_keyframe else "2-3 distinct action beats"
        extra = (
            "evolving camera and blocking with at least one clear energy shift"
            if not is_keyframe
            else "evolving camera and blocking; do not pad"
        )
    else:
        beats = "4-6 distinct action beats" if not is_keyframe else "3-4 distinct action beats"
        extra = (
            "vary distance and energy across the clip; include one in-clip setup or location change if the story allows"
            if not is_keyframe
            else "vary distance and energy across the clip"
        )
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
            "that follows it. Also, do NOT recap the previous scene's body "
            "inside this scene — the carried overlap is a one-sentence state "
            "delta (where the previous clip ended mid-action), NOT a recap. "
            "Write the new action beginning where the previous one ended; "
            "do not re-narrate the previous beat."
        )
    return text


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
def build_plan(shot_entries: list[dict], prefix_lines: list[str]) -> dict:
    """Assemble the strict plan dict.

    ``shot_entries``: ``{"id", "prompt": [lines], "length", "seed"}`` —
    field order (id, prompt, length, seed). Sampler steps and every
    other generation parameter stay on the Plan node's widgets; the
    JSON carries only prompt / timing / seed content.
    """
    shots_out: list[dict] = []
    for entry in shot_entries:
        shots_out.append(
            {
                "id": entry["id"],
                "prompt": list(entry["prompt"]),
                "length": int(entry["length"]),
                "seed": str(entry["seed"]),
            }
        )
    return {
        "shots": shots_out,
        "prompt_prefix": [str(ln) for ln in prefix_lines],
    }


def validate_plan(plan: dict, *, schema: str = SCHEMA_THREE) -> list[str]:
    """Return a list of contract violations (empty list = valid).

    Schema-aware:
    - Top-level keys: ``shots`` always required; ``prompt_prefix`` is
      OPTIONAL (keyframe modes may emit shots only; ref2va always emits
      one). Any other top-level key is an error.
    - Three-section: each shot's prompt array must start with
      ``integrated_multimodal_description:`` and end with
      ``non_diegetic_music:``.
    - Six-section: each shot's prompt array must contain all six bare
      Ref2V headers in exact order.
    """
    errors: list[str] = []
    top = set(plan.keys())
    # ``prompt_prefix`` is OPTIONAL; only unknown extras or a missing
    # ``shots`` are errors.
    if top - {"shots", "prompt_prefix"}:
        errors.append(
            f"top-level keys must be a subset of {{'shots', 'prompt_prefix'}}, got {sorted(top)}"
        )
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
        extra = set(shot.keys()) - {"id", "prompt", "length", "seed"}
        if extra:
            errors.append(f"shot {i}: unexpected keys {sorted(extra)}")
    prefix = plan.get("prompt_prefix")
    if prefix is not None:
        # prompt_prefix may be absent (shots only) OR an empty list
        # (ref2va style-only prefix); when present, every line must be a
        # non-empty string.
        if not isinstance(prefix, list) or not all(
            isinstance(ln, str) and ln.strip() for ln in prefix
        ):
            errors.append(
                "prompt_prefix, when present, must be an array of non-empty strings"
            )
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
        f"Scenes: {len(shots)}",
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
            f"({length_to_seconds(s.get('length') or 0):.2f}s)  "
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


def build_spatial_layout_directive(spatial_layout: Optional[dict] = None) -> str:
    """Render a spatial-layout dict (subject → on-screen position) as the
    binding sentence injected into every per-shot user template.

    The format is deterministic so the LLM cannot paraphrase the position
    ("at left of frame" must not become "on the left side of the picture
    frame" — paraphrases defeat the cross-clip continuity check).

    Relative positions harvested from the semantic-facts extractor
    ("beside 小猫", "left of the kitten") render without the "at"
    ("黑猫 stays beside 小猫") — they anchor one subject to another,
    not to the frame.

    A empty / None spatial_layout returns a single neutral line so the
    template still has something in the slot. The per-shot LLM is then
    instructed to mirror the prefix's spatial layout sentence verbatim.
    """
    if not spatial_layout:
        return (
            "(no spatial_layout declared for this board; mirror any "
            "spatial layout sentence in the shared prefix above, and "
            "preserve positions across consecutive clips)"
        )
    entries: list[str] = []
    absolute = {
        "left of frame", "right of frame", "center of frame",
        "left slot", "right slot", "center slot",
        "screen-left", "screen-right", "camera-left", "camera-right",
    }
    for name, pos in spatial_layout.items():
        pos_s = str(pos)
        # Relative anchors ("beside 小猫") render without "at"; the
        # canonical absolutes ("left of frame") keep it — the prefix
        # check must not catch the absolutes that start with the same
        # words.
        if (
            pos_s not in absolute
            and pos_s.startswith(
                ("beside ", "next to ", "left of ", "right of ", "behind ", "in front of ")
            )
        ):
            entries.append(f"{name} stays {pos_s}")
        else:
            entries.append(f"{name} stays at {pos_s}")
    if not entries:
        return (
            "(no spatial_layout declared for this board; mirror any "
            "spatial layout sentence in the shared prefix above)"
        )
    return "; ".join(entries) + "."


# ---------------------------------------------------------------------------
# Semantic facts (layout / picture bindings) — LLM judges, code verifies.
#
# The 2026-09 design review conclusion: spatial layout and picture
# bindings are SEMANTIC judgements (who is where; which picture is which
# character) and the regex pile that owned them accumulated a fix per
# phrasing (的小猫 fragments, 黑 above the [一-龥] ceiling, 黑猫是爸爸
# name swallowing, enumerated delimiters) while remaining blind to
# relative positions ("黑猫在小猫旁边" — the user's own input!). The
# span-anchored extractor already reads the concept with full intent;
# these helpers harvest + MECHANICALLY VERIFY its layout/bindings facts
# the same way turn spans are verified: a fact's name must literally
# appear in the source text, positions must normalise to the canonical
# vocabulary. Unverifiable entries are dropped, and the regex extractors
# remain as fallback when the LLM returns nothing.
# ---------------------------------------------------------------------------
_FACT_POSITION_ZH = {
    "画面左侧": "left of frame", "左边": "left of frame",
    "画面左": "left of frame", "左侧": "left of frame",
    "画左": "left of frame", "左方": "left of frame",
    "画面右侧": "right of frame", "右边": "right of frame",
    "画面右": "right of frame", "右侧": "right of frame",
    "画右": "right of frame", "右方": "right of frame",
    "画面中央": "center of frame", "中间": "center of frame",
    "中央": "center of frame", "画面中间": "center of frame",
    "居中": "center of frame",
}
_FACT_POSITION_EN = {
    "left of frame": "left of frame",
    "left side of frame": "left of frame",
    "left of the frame": "left of frame",
    "left side": "left of frame",
    "left": "left of frame",
    "right of frame": "right of frame",
    "right side of frame": "right of frame",
    "right of the frame": "right of frame",
    "right side": "right of frame",
    "right": "right of frame",
    "center of frame": "center of frame",
    "centre of frame": "center of frame",
    "center of the frame": "center of frame",
    "center-frame": "center of frame",
    "center": "center of frame",
    "centered": "center of frame",
    "middle": "center of frame",
}
_FACT_RELATIVE_RE = re.compile(
    r"^(beside|next to|left of|right of|behind|in front of)\s+(.+)$",
    re.IGNORECASE,
)


def normalize_fact_position(raw: str) -> Optional[str]:
    """Normalise one LLM-returned position phrase to the canonical
    vocabulary: ``left of frame`` / ``right of frame`` / ``center of
    frame`` / ``beside <name>``-style relative anchors. Returns None
    when nothing recognisable — the caller drops the entry."""
    s = str(raw or "").strip()
    if not s:
        return None
    for zh, canon in _FACT_POSITION_ZH.items():
        if zh in s:
            return canon
    low = s.lower().strip().rstrip(".")
    if low in _FACT_POSITION_EN:
        return _FACT_POSITION_EN[low]
    m = _FACT_RELATIVE_RE.match(low)
    if m and m.group(2).strip():
        rel = m.group(1).lower()
        anchor = m.group(2).strip()
        if rel == "next to":
            rel = "beside"
        return f"{rel} {anchor}"
    return None


def harvest_semantic_facts(parsed: dict, concept: str) -> dict:
    """Mechanically verify + shape the LLM's layout / bindings facts.

    ``parsed`` is the extractor reply's JSON object. Verification: every
    returned NAME must literally appear in the source concept (the same
    trust-but-verify contract as turn spans — the LLM points, the code
    confirms); positions must normalise; picture slots must be 1..9.
    Unverifiable entries are dropped silently (regex fallback covers).

    Returns ``{"layout": {name: canonical_pos}, "bindings":
    [(name, slot), ...]}`` — the exact shapes ``extract_spatial_layout``
    and ``_concept_picture_bindings`` produce, so consumers are
    source-agnostic.
    """
    out: dict = {"layout": {}, "bindings": []}
    if not isinstance(parsed, dict) or not concept:
        return out
    for entry in parsed.get("layout") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        pos = normalize_fact_position(str(entry.get("position") or ""))
        if not name or not pos:
            continue
        if name not in concept:
            continue  # hallucinated name — drop
        if name not in out["layout"]:
            out["layout"][name] = pos
    seen_names: set = set()
    for entry in parsed.get("bindings") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        try:
            slot = int(entry.get("picture"))
        except (TypeError, ValueError):
            continue
        if not name or not (1 <= slot <= 9) or name not in concept:
            continue
        if name in seen_names:
            continue  # first binding wins per name
        seen_names.add(name)
        out["bindings"].append((name, slot))
    return out


# Binding tempo directives — one per pacing preset. Injected into every
# per-shot user template (and the single-call text) so the WRITING tempo
# matches the board's pacing: fast boards chain beats tightly, slow
# boards let actions breathe. The per-shot system addendum's slow-word
# ban exempts MEASURED-preset calm content from the camera-speed rule.
_TEMPO_DIRECTIVES = {
    "fast": (
        "Tempo: BRISK (fast pacing preset). This board cuts fast: chain "
        "action beats tightly one after another, keep visible motion "
        "continuous, no lingering holds, no rest frames until the final "
        "clip. Favor quick, complete actions over long setups."
    ),
    "normal": (
        "Tempo: NATURAL (normal pacing preset). Real-time conversational "
        "flow: let each beat land clearly, then move on; steady forward "
        "motion without rushing."
    ),
    "slow": (
        "Tempo: MEASURED (slow pacing preset). Let each action breathe: "
        "longer holds, unhurried continuous motion, calm camera. Motion "
        "stays at real-time speed — never slow motion."
    ),
}


def build_tempo_directive(pacing_key: str) -> str:
    """Render the binding tempo sentence for a pacing preset key
    (unknown keys fall back to the natural-tempo directive)."""
    return _TEMPO_DIRECTIVES.get(
        str(pacing_key or "").strip().lower(),
        _TEMPO_DIRECTIVES["normal"],
    )


def _genre_advice_block(category: str) -> str:
    code = parse_category(category)
    advice = category_advice(code).strip()
    if advice:
        return f"Genre guidance ({code}): {advice}"
    return "Genre guidance: none; follow the concept as written."


# Canonical position buckets. Every recognised token — English or
# Chinese — normalises to ONE of three canonical strings so the
# cross-clip invariant compares semantics, not spelling: "centre of
# frame" / "camera-left" / "画面左" / "left slot" all collapse to
# "left of frame". A phrase that mixes CONFLICTING buckets ("enters
# from camera-left and stops at center-frame") is movement prose, not
# a binding declaration — it normalises to None.
_POSITION_CANON_EN = {
    "left of frame": "left of frame",
    "screen-left": "left of frame",
    "camera-left": "left of frame",
    "left slot": "left of frame",
    "right of frame": "right of frame",
    "screen-right": "right of frame",
    "camera-right": "right of frame",
    "right slot": "right of frame",
    "center of frame": "center of frame",
    "centre of frame": "center of frame",
    "center-frame": "center of frame",
    "centre-frame": "center of frame",
    "center slot": "center of frame",
}
_POSITION_CANON_ZH = {
    "画面左侧": "left of frame",
    "画面左边": "left of frame",
    "画面左": "left of frame",
    "画左": "left of frame",
    "左侧": "left of frame",
    "左边": "left of frame",
    "画面右侧": "right of frame",
    "画面右边": "right of frame",
    "画面右": "right of frame",
    "画右": "right of frame",
    "右侧": "right of frame",
    "右边": "right of frame",
    "画面中央": "center of frame",
    "画面中间": "center of frame",
    "中央": "center of frame",
    "中间": "center of frame",
}


def _normalise_position(raw: str) -> Optional[str]:
    """Map a raw position phrase to one of the three canonical buckets
    (``left of frame`` / ``right of frame`` / ``center of frame``).
    Returns ``None`` when the phrase is too generic to anchor
    cross-clip continuity, or when it mixes conflicting buckets.

    Chinese and English tokens collapse to the SAME canonical bucket,
    so a board declared in Chinese ("画面左侧") stays comparable against
    per-shot output prose written in English ("left of frame") — that
    language-neutrality is what the cross-clip drift check keys on."""
    s = str(raw or "").strip()
    if not s:
        return None
    s_lower = s.lower()
    buckets = set()
    for tok, canon in _POSITION_CANON_EN.items():
        if tok in s_lower:
            buckets.add(canon)
    for tok, canon in _POSITION_CANON_ZH.items():
        if tok in s:
            buckets.add(canon)
    if len(buckets) == 1:
        return buckets.pop()
    return None


# Patterns matched against the rewritten user_input (which the enhancer
# has already normalised). The left side captures the subject name; the
# right side captures the position phrase. Order matters — the first
# match wins so the most specific pattern (Picture N → Subject N (left
# slot)) should be tried before the generic "Subject sits at left of
# frame".
_SPATIAL_PATTERNS: tuple[tuple[re.Pattern, int], ...] = (
    # Picture N -> Subject M (left slot / right slot)
    (re.compile(
        r"(?:Picture|图)\s*(\d+)\s*[-–—→]+\s*(?:Subject|主体|角色)?\s*(\d+)?"
        r"\s*[:：]?\s*"
        r"(?P<pos>[^。\n,，;；]+?\s*(?:slot|位|侧|frame))",
        re.IGNORECASE,
    ), 1),
    # <Subject N> (right slot) / Subject 2 (left of frame)
    (re.compile(
        r"(?:<Subject\s+(\d+)>|Subject\s+(\d+)|主体\s*(\d+)|角色\s*(\d+))"
        r"\s*[:：]?\s*[\(（]?\s*"
        r"(?P<pos>[^。\n)）]+?\s*(?:slot|位|侧|frame))",
        re.IGNORECASE,
    ), 2),
    # Chinese name explicitly named with parentheses: "哈利猫（坐画面左侧）"
    # or "莎莉猫 (at left of frame)" — the parenthetical form is the
    # canonical one the enhancer rule emits. ``(?!的)`` blocks picture-
    # prefix fragments ("图3的小猫 (center of frame)" must yield 小猫,
    # never 的小猫 — live 2026-09-21: the garbage key rode into the
    # prefix spatial pin).
    (re.compile(
        r"(?P<name>(?!的)[A-Za-z][A-Za-z0-9_-]{0,31}|(?!的)[一-鿿]{2,6})"
        r"\s*[\(（]\s*"
        r"(?:坐|站|seated|sits?|stands?|stays?)?"
        r"\s*(?:at\s+)?(?P<pos>[^)）]+?\s*(?:左侧|右侧|中央|左边|右边|中间|left\s+of\s+frame|right\s+of\s+frame|center\s+of\s+frame|left\s+slot|right\s+slot))",
        re.IGNORECASE,
    ), 3),
    # English named character: "Sahli sits at left of frame" /
    # "the cream cat stays at right of frame throughout"
    (re.compile(
        r"(?P<name>[A-Z][a-z]+(?:[ _-][A-Z][a-z]+)*)"
        r"\s+(?:sits?|seated|stays?|stands?|enters?)"
        r"\s+(?:at\s+)?(?P<pos>[^。\n,，;；]+?\s*(?:left\s+of\s+frame|right\s+of\s+frame|center\s+of\s+frame|left\s+slot|right\s+slot))",
    ), 4),
    # 中文 "<角色名>坐画面左侧" (no parentheses — fallback when the
    # enhancer didn't wrap the position in parens).
    (re.compile(
        r"(?P<name>(?!的)[一-鿿]{2,6}(?:猫|狗|人|男孩|女孩|男人|女人|角色|主体))"
        r"\s*坐?\s*"
        r"(?P<pos>[^。\n,，;；]+?\s*(?:画面?(?:左侧|右侧|中央|左边|右边|中间)|画左|画右))",
    ), 5),
)

# Leading particles that mark a captured "name" as a sentence fragment
# ("对着图3的**的小猫**"), not a character. Stripped before the
# subject-name check; real names never start with these.
_NAME_LEADING_PARTICLES = "的着在和与或对向从把被让"


# Chinese particles that almost certainly mean the preceding 2-3 char
# token is a noun-phrase fragment, NOT a subject name. Used to filter
# false positives like "金色窗光 从 左侧" -> reject "金色窗光"; allow
# "金色窗光旁的猫 坐 画面左侧" -> reject "金色窗光旁的猫" because the
# true subject is the noun after the possessive 的.
_CHINESE_STOPWORDS = (
    "光", "色", "窗", "光从", "光色", "色调", "灯", "门", "墙",
    "桌子", "椅子", "地面", "房间", "屋内", "窗光", "窗光色",
    "影", "光照", "影调",
)


def _looks_like_subject_name(raw: str) -> bool:
    """Reject Chinese tokens that are clearly environmental nouns
    (lighting, surfaces, furniture) being mistaken for subject names.
    ASCII tokens are always accepted (English name 'Sahli', 'Harry')."""
    if not raw:
        return False
    # ASCII identifier — assume OK; English rules apply.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", raw):
        return True
    # Chinese 2-6 char token — must end with a subject noun marker or
    # be inside the canonical name list (cat/dog/person/role etc.).
    if re.fullmatch(r"[一-鿿]{2,6}", raw):
        if raw in _CHINESE_STOPWORDS:
            return False
        if raw.endswith(("猫", "狗", "人", "鸟", "兽", "角色", "主体",
                         "男孩", "女孩", "男人", "女人", "公", "母")):
            return True
        # Bare Chinese 2-6 char that doesn't end with a subject marker
        # is almost always something else (a room feature, a colour,
        # an abstract noun). Reject — too risky to extract.
        return False
    return False


def extract_spatial_layout(concept: str) -> dict[str, str]:
    """Pull a stable per-subject on-screen position out of the rewritten
    user_input. The function is rule-based and deterministic — it does
    NOT call an LLM. The extracted dict is what gets carried into the
    prefix sentence and into every per-shot user template via
    ``build_spatial_layout_directive``.

    Returns ``{}`` when the concept has no recognisable position tokens.
    Callers MUST treat an empty dict as "no binding spatial layout" and
    rely on the prefix-only path; the board-level preflight check
    (``validate_spatial_layout_invariant`` over the declared layout +
    every generated shot's re-extracted layout) flags any disagreement
    as a warning.

    Subject naming follows the cast roster convention: "<Subject N>"
    tokens, plain "Subject N" / "主体 N" / "角色 N", or a Chinese name
    that ends with a subject marker (猫/狗/人/角色/主体/...). Generic
    Chinese nouns (lighting, furniture, surfaces) are filtered out
    because they almost always produce false positives — a sentence
    like "金色窗光从左侧洒入" is lighting info, not a spatial layout
    declaration.
    """
    out: dict[str, str] = {}
    text = (concept or "").strip()
    if not text:
        return out
    for pattern, group_idx in _SPATIAL_PATTERNS:
        for match in pattern.finditer(text):
            pos_raw = match.group("pos")
            pos = _normalise_position(pos_raw)
            if not pos:
                continue
            # Find the subject key in this match — depends on which
            # group captured it.
            name: Optional[str] = None
            if group_idx == 1:
                pic = match.group(1)
                subj = match.group(2) or pic
                name = f"Subject {subj}" if subj else f"Picture {pic}"
            elif group_idx == 2:
                # groups (1..4) for <Subject N> / Subject N / 主体 N / 角色 N
                for g in (1, 2, 3, 4):
                    v = match.group(g)
                    if v:
                        name = f"Subject {v}"
                        break
            elif group_idx in (3, 4, 5):
                raw_name = match.group("name")
                if raw_name:
                    # Strip leading particles that mark a sentence
                    # fragment ("对着图3的**的**小猫") — real names never
                    # start with 的/着/在/&c.
                    cleaned = raw_name.strip().lstrip(_NAME_LEADING_PARTICLES)
                    if cleaned and _looks_like_subject_name(cleaned):
                        name = cleaned
            if not name:
                continue
            # First-match-wins per name. If the same subject shows up
            # twice with different positions, that's a preflight
            # warning (validated in scene-to-scene continuity check).
            if name not in out:
                out[name] = pos
    return out


def validate_spatial_layout_invariant(
    spatial_layout: dict[str, str],
    scenes: list[dict],
) -> list[str]:
    """Cross-clip spatial-layout invariant. When ``scenes`` each carry a
    ``spatial_layout`` field (extracted from their shot description by
    the per-shot LLM or supplied upstream), every subject whose position
    is recorded must keep the same position across every scene it
    appears in.

    Returns an empty list when the board is consistent (or when no
    scenes carry a layout to compare)."""
    if not spatial_layout or not scenes:
        return []
    errors: list[str] = []
    seen: dict[str, tuple[str, int]] = {}
    for i, scene in enumerate(scenes, start=1):
        layout = scene.get("spatial_layout") or {}
        for name, pos in layout.items():
            if name in seen:
                prev_pos, prev_idx = seen[name]
                if pos != prev_pos:
                    errors.append(
                        f"scene {prev_idx} declared {name} at {prev_pos!r}, "
                        f"scene {i} declared {name} at {pos!r}; spatial "
                        f"layout must stay constant across the board"
                    )
            else:
                seen[name] = (pos, i)
    return errors





def build_prefix_user_text(
    concept: str,
    category: str,
    language_name: str,
    shots_digest: str = "",
    *,
    mode_note: str = "",
    manifest_digest: str = "",
    cast_roster: str = "(none named)",
) -> str:
    return _PREFIX_SYNTH_USER_TEMPLATE.format(
        concept=(concept or "").strip(),
        storyboard_digest=(shots_digest or "(no storyboard provided)").strip(),
        genre_advice=_genre_advice_block(category),
        language_name=language_name,
        mode_note=(mode_note or "").strip(),
        manifest_digest=(manifest_digest or "no reference images").strip(),
        cast_roster=(cast_roster or "(none named)").strip(),
    )


# Heuristic roster fallback: capitalized proper-noun-like tokens and
# Chinese 2-4 char noun phrases from a shot's description text. Only
# used when the structured ``characters`` array is empty.
EN_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:[ _-][A-Z][a-z]+)*\b")
ZH_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")

# Stopwords filtered from the heuristic fallback: capitalized
# sentence-initial words the EN proper-noun regex would otherwise sweep
# up as cast members ("The", "Then", "During", ...).
_ROSTER_STOPWORDS = frozenset(
    {
        "the", "a", "an", "then", "when", "during", "after", "before",
        "his", "her", "its", "their", "this", "that", "camera", "shot",
        "she", "he", "they", "it", "as", "in", "on", "at",
    }
)


def _shot_roster_tokens(shot: dict) -> set[str]:
    """Per-shot on-screen subject tokens. The structured ``characters``
    array is the authoritative roster (the storyboard LLM's own cast
    assignment) and is used verbatim when present; the capitalized-word
    + Chinese-phrase heuristic on the description text is only a
    fallback for boards that left ``characters`` empty."""
    roster = shot.get("characters") or []
    named = {str(c).strip().lower() for c in roster if str(c).strip()}
    if named:
        return named
    desc = str(shot.get("description") or "")
    tokens: set[str] = set()
    for match in EN_PROPER_RE.findall(desc):
        # A multi-word match may open with a capitalized stopword
        # ("The Cat" matches as one compound) — drop stopword words
        # instead of discarding the whole compound.
        kept = [
            w for w in match.split() if w.lower() not in _ROSTER_STOPWORDS
        ]
        if kept:
            tokens.add(" ".join(kept).lower())
    tokens.update(ZH_PHRASE_RE.findall(desc))
    return tokens


def build_shots_digest(shots: list[dict], max_desc_chars: int = 160) -> str:
    """One line per shot with presence hints grounding the prefix
    synthesis in the actual storyboard — without this, a thin concept
    lets the prefix drift (e.g. inventing human protagonists), AND
    sequences with a late-arriving character (e.g. "two fighters, then
    a third drops from the sky") collapse every scene into a single
    over-packed frame because the prefix declares ALL characters as
    recurring. Each line tags the shot with which named subjects appear
    here (first_seen) vs. which appear across the whole board (all_shots).
    The prefix synth uses this to keep one-shot characters out of the
    shared prefix."""
    lines: list[str] = []
    for shot in shots:
        desc = str(shot.get("description") or "").strip().replace("\n", " ")
        if len(desc) > max_desc_chars:
            desc = desc[: max_desc_chars - 3] + "..."
        lines.append(f"- {shot.get('id')}: {desc}")
    if not lines:
        return ""
    per_shot = [_shot_roster_tokens(shot) for shot in shots]
    all_set = set.intersection(*per_shot) if per_shot else set()
    decorated: list[str] = []
    for shot, tokens in zip(shots, per_shot):
        only_here = sorted(tokens - all_set)
        recurring = sorted(tokens & all_set)
        flags: list[str] = []
        if recurring:
            flags.append("all_shots=" + ",".join(recurring))
        if only_here:
            flags.append("first_seen=" + ",".join(only_here))
        # Recompute desc per-shot — do NOT reuse the outer loop's binding.
        desc = str(shot.get("description") or "").strip().replace("\n", " ")
        if len(desc) > max_desc_chars:
            desc = desc[: max_desc_chars - 3] + "..."
        suffix = ("  # " + "; ".join(flags)) if flags else ""
        decorated.append(f"- {shot.get('id')}: {desc}{suffix}")
    return "\n".join(decorated)


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
    *,
    category: Optional[str] = None,
    referenced_pictures: Optional[set] = None,
) -> str:
    """Per-mode reference directive woven into the first sentence of
    the description block (D6). Returns "" for t2va (no labels).

    - i2va scene 1: official opening idiom with Picture 1's about text.
    - fl2va scene 1: scene-1 idiom naming both pictures.
    - fl2va scene N>=2: end-target sentence converging on
      ``<Picture (N % 2) + 1>`` (alternates between Picture 1 / 2).
    - ref2va: deterministic subject_definitions + summary +
      retention_analysis construction guide for the LLM (D5), plus
      optional SPOKEN SCENE CONTRACT / GENRE CONTRACT blocks when
      ``category`` is ``dialogue`` (set by the user via the category
      widget).

    ``category`` is the parsed code from the ``category`` widget — e.g.
    ``"dialogue"``. When ``category == "dialogue"``, the ref2va branch
    appends a one-rule SPOKEN SCENE CONTRACT (shot boundaries land on
    completed utterances) and a two-rule GENRE CONTRACT (setup → beat →
    punchline + verbatim <d>...</d> language tags).
    """
    code = parse_reference_mode(mode)
    if code == "t2va":
        return ""
    if code == "i2va":
        if clip_index != 1 or not manifest:
            return ""
        about = manifest[0].get("about", "")
        return (
            f"At 0.00 seconds, <Picture 1> is fully referenced as the opening frame. "
            f"This scene animates <Picture 1> (exact: {about}) from its frozen pose "
            f"to the next beat of the concept. Describe only the smooth visual path "
            f"between them."
        )
    if code == "fl2va":
        if clip_index == 1:
            if len(manifest) < 2:
                return ""
            a1 = manifest[0].get("about", "")
            return (
                f"<Picture 1> is the opening frame; <Picture 2> is the closing frame. "
                f"Describe the smooth interpolation between them. "
                f"<Picture 1> (exact: {a1}) is the opening pose; "
                f"<Picture 2> is the closing pose the next scene arrives at."
            )
        # Scenes 2+: the L2VA gate exposes ONE image under <Picture 1> -
        # the per-scene end target. Pick that image's manifest entry when
        # the user supplied one (5-picture layout: Picture 1 = opening,
        # Picture k>=2 = scene k end target). Fall back to alternating
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
        # Pictures the concept never names (图N / Picture N): wired and
        # captioned, but the story does not use them. Teaching their
        # bindings only invites the model to mention Subjects it does
        # not need (live failure 2026-09-21 20:58: scene_02 referenced
        # <Subject 4>/<Subject 5> unpaired); instead the directive
        # names ONLY the used slots and forbids the rest explicitly.
        all_pic_nums = list(range(1, len(manifest) + 1))
        unused_pics = (
            [n for n in all_pic_nums if n not in referenced_pictures]
            if referenced_pictures
            else []
        )
        used_pic_nums = [n for n in all_pic_nums if n not in unused_pics]
        bound_pairs: list[tuple[int, str]] = []
        subj_seq = 0
        for i, entry in enumerate(manifest, start=1):
            if entry.get("role") != "identity":
                continue
            subj_seq += 1  # Subject numbering follows MANIFEST identity
            # order (matches _identity_subject_slots), not the used
            # subset — skipping unused slots keeps stable numbering.
            if i in unused_pics:
                continue
            bound_pairs.append((subj_seq, f"Picture {i}"))
        subj_list = ", ".join(f"<Subject {n}>" for n, _ in bound_pairs)
        slots_line = ", ".join(f"<Picture {n}>" for n in used_pic_nums)
        if unused_pics:
            slots_line += (
                f". Pictures {unused_pics} are wired but NOT used by this "
                "concept — do NOT reference them or their <Subject N> "
                "anywhere in the reply"
            )
        lines: list[str] = [
            "[Deterministic subject binding — DO NOT INVENT NEW SUBJECTS]",
            f"Manifest slots (in order): {slots_line}.",
        ]
        if bound_pairs:
            lines.append(
                f"Identity slots: {', '.join(slot for _, slot in bound_pairs)} "
                f"bind to {subj_list} (numbered in manifest identity order)."
            )
            for subj_n, slot in bound_pairs:
                pic_num = int(slot.split(" ", 1)[1])
                about = next(
                    (m.get("about", "") for m in manifest if m.get("slot") == slot),
                    "",
                )
                lines.append(
                    f"  <Subject {subj_n}> -> <Picture {pic_num}>: {about}"
                )
        else:
            lines.append(
                "No identity-role slots among the used pictures; describe every "
                "picture by its <about> in subject_definitions (no <Subject N> "
                "binding)."
            )
        non_identity = [
            entry for i, entry in enumerate(manifest, start=1)
            if entry.get("role") != "identity" and i not in unused_pics
        ]
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
        # Category-driven contracts. category is a single code string;
        # the user picks it via the category widget. Only dialogue
        # unlocks the spoken/genre contract blocks today.
        category_code = parse_category(category) if category else ""
        if category_code == "dialogue":
            lines.append(
                "---\n"
                "[SPOKEN SCENE CONTRACT = dialogue "
                "(selected via the category widget). Shot boundaries MUST "
                "land after a completed utterance or a visible reaction "
                "beat, never in the middle of a spoken sentence.\n"
                "---"
            )
            lines.append(
                "---\n"
                "[GENRE CONTRACT = dialogue "
                "(selected via the category widget).]\n"
                "  (1) Dialogue on this clip is LOCKED as data: the node "
                "appends the verbatim <d>[Language]...</d> speech blocks "
                "itself. You write NO <d> blocks and NO spoken words, "
                "quoted or unquoted — any copy is stripped automatically. "
                "ONE input dialogue line = ONE appended block; never "
                "split, paraphrase, or reorder the lines in your "
                "choreography.\n"
                "  (2) The appended blocks keep the speaker's original "
                "language untranslated with full-name tags "
                "(`[English]` / `[Chinese]`). Refer to speech by speaker "
                "and beat, never by restating the words.\n"
                "---"
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
# Matches the "Reach <Picture 2> only on the final frame" closing line
# used by both fl2va scene 1 and scene 2+ enforcer paths.
_FL2VA_REACH_RE = re.compile(
    r"<\s*Picture\s*[12]\s*>[^.\n]{0,80}only\s+on\s+the\s+final\s+frame"
    r"|only\s+on\s+the\s+final\s+frame[^.\n]{0,80}<\s*Picture\s*[12]\s*>",
    re.IGNORECASE,
)


def ensure_keyframe_idiom(
    lines: list[str],
    *,
    mode: str,
    manifest: list[dict],
    clip_index: int,
) -> list[str]:
    """Enforce the official keyframe idiom on a split THREE-section prompt.

    i2va scene 1: replace any idiom-shaped segment with the canonical
    opening sentence; prepend when absent.
    fl2va scene 1: replace any idiom-shaped segment with the canonical
    opening sentence; prepend when absent; ensure the scene closes with
    "Reach <Picture 2> only on the final frame; do not freeze early or
    cut." when no reach sentence exists.
    fl2va scene N>=2: ensure "Reach <Picture 1> only on the final frame;
    do not freeze early or cut." closes the description body when no
    reach sentence exists.
    i2va scenes 2+ / t2va / ref2va: pass through unchanged.

    Returns a NEW list.
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
        if clip_index != 1 or not manifest:
            return out
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

    # fl2va scene N>=2: append the single-line reach sentence if missing.
    body_text = "\n".join(out[body_start:sound_header])
    if not _FL2VA_REACH_RE.search(body_text):
        insert_at = sound_header
        while insert_at > body_start and not out[insert_at - 1].strip():
            insert_at -= 1
        out.insert(
            insert_at,
            "Reach <Picture 1> only on the final frame; do not freeze early or cut.",
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
    """Policy note for the prefix synthesis user template. Unified across
    all reference modes: the prefix NEVER carries characters — identity
    anchoring travels per-clip through the deterministic cast blocks."""
    return (
        "Prefix policy (all modes): whole-video invariants only — art "
        "style, setting, lighting/palette (non-exclusive), tempo, and "
        "the global exclusions. NEVER any character, identity, or prop "
        "in the prefix; identity travels per-clip through the cast "
        "blocks built from your CAST sheet."
    )


def _detect_dialogue_language(line: str) -> str:
    """Pick the canonical H3 dialogue language tag for one line.

    Heuristic: count CJK characters; if any CJK is present, the line
    is treated as Chinese. Otherwise English. Empty / pure-punctuation
    lines fall back to ``[Chinese]`` to preserve the policy default
    the rest of the addendum enforces.
    """
    if any("\u4e00" <= c <= "\u9fff" for c in line):
        return "[Chinese]"
    if any(c.isascii() and c.isalpha() for c in line):
        return "[English]"
    return "[Chinese]"


def build_dialogue_language_policy(dialogue_lines: Optional[list[str]]) -> str:
    """Derive the spoken-language policy sentence from the ACTUAL
    extracted lines (2026-09-22 live failure: an all-English dialogue
    board was generated under a hardcoded "dialogue is Chinese by
    default" policy sentence, pulling against the verbatim <d> blocks).

    The board's own lines are the source of truth — English dialogue
    means an English-dialogue board; there is no "default" language.
    """
    langs = sorted(
        {
            _detect_dialogue_language(str(ln)).strip("[]")
            for ln in (dialogue_lines or [])
            if str(ln).strip()
        }
    )
    if not langs:
        return (
            "No spoken dialogue in this production; write no spoken "
            "words anywhere."
        )
    if len(langs) == 1:
        return (
            f"All spoken dialogue is {langs[0]}: every <d> block the node "
            f"appends already carries its [{langs[0]}] tag and the exact "
            "original wording — reproduce it verbatim; NEVER translate "
            "spoken lines in either direction."
        )
    return (
        "Spoken dialogue is mixed-language ("
        + "/".join(langs)
        + "): each <d> block the node appends carries its own language "
        "tag and exact original wording — reproduce each verbatim; NEVER "
        "translate spoken lines in either direction."
    )


_THREE_SECTION_HEADERS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
_SIX_SECTION_HEADERS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


# Kinship the concept actually declares. ``黑猫是爸爸`` and
# ``黑猫（猫爸爸`` both bind the ROLE to the character name. Vocatives
# (Mommy / 妈妈) look the role up; the per-shot model is not asked to
# guess who a line is spoken to.
_ROLE_CANON = {
    "爸爸": "爸爸", "父亲": "爸爸",
    "妈妈": "妈妈", "母亲": "妈妈",
    "女儿": "女儿", "儿子": "儿子",
}
_ROLE_MENTION_RE = re.compile(
    r"([^\s，,。；;：:（）()的\d]{1,12})"
    r"(?:是|为|[（(][^）)]{0,8}?)"
    r"(爸爸|父亲|妈妈|母亲|女儿|儿子)"
)
_ROLE_NAME_STOP = {"不", "没", "就", "也", "这", "那", "是", "为", "有", "在"}
# Line-initial only. ``Mommy, daddy is...`` addresses 妈妈; the later
# "daddy" is not a second addressee. ``mom`` does not match ``mommy``
# because of the word boundary.
_LINE_VOCATIVE: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^(?:mommy|mum|mama|mother|mom)\b", re.IGNORECASE), "妈妈"),
    (re.compile(r"^(?:daddy|papa|father|dad)\b", re.IGNORECASE), "爸爸"),
    (re.compile(r"^妈妈"), "妈妈"),
    (re.compile(r"^爸爸"), "爸爸"),
)


def extract_role_bindings(concept: str) -> dict[str, str]:
    """Map a kinship role to the character the concept assigns.

    ``图1的黑猫是爸爸`` and ``黑猫（猫爸爸，画面右边）`` both yield
    ``{"爸爸": "黑猫"}``. First assignment wins. Roles the text does
    not state are absent — callers must not guess.
    """
    out: dict[str, str] = {}
    for match in _ROLE_MENTION_RE.finditer(concept or ""):
        name = match.group(1).strip()
        role = _ROLE_CANON.get(match.group(2), "")
        # ``不是爸爸`` / ``他不是爸爸`` is a negation, not a binding.
        if (
            not name
            or name in _ROLE_NAME_STOP
            or name.endswith("不")
            or not role
            or role in out
        ):
            continue
        out[role] = name
    return out


def line_initial_role(line: str) -> str:
    """The kinship role a line opens by addressing, or ``""``.

    Strips one layer of wrapping quotes. A vocative later in the line
    is ignored — ``Mommy, daddy is so ugly`` addresses 妈妈.
    """
    text = (line or "").strip()
    text = text.lstrip("\"'“”「『")
    for pattern, role in _LINE_VOCATIVE:
        if pattern.match(text):
            return role
    return ""


def frame_side_label(position: str) -> str:
    """``LEFT`` / ``RIGHT`` / ``CENTER``, or ``""`` when the layout
    entry is not a side (``beside 小猫`` stays unnamed)."""
    pos = str(position or "").strip().lower()
    if pos.startswith("left"):
        return "LEFT"
    if pos.startswith("right"):
        return "RIGHT"
    if pos.startswith(("cent", "middle")):
        return "CENTER"
    return ""


def resolve_line_facing(
    line: str,
    speaker: str,
    role_bindings: Optional[dict],
    spatial_layout: Optional[dict],
    known_speakers: Optional[set] = None,
) -> Optional[tuple[str, str]]:
    """``(addressee name, side label)`` for a line-initial vocative.

    The side label is ``""`` when the layout has no left/right/center
    for that addressee. Returns None when the line has no vocative,
    the role is not bound to a character, or the speaker would be
    told to face themselves.
    """
    role = line_initial_role(line)
    if not role:
        return None
    bindings = dict(role_bindings or {})
    addressee = str(bindings.get(role) or "").strip()
    if not addressee and role in set(known_speakers or ()):
        addressee = role
    speaker_name = (speaker or "").strip()
    if not addressee or addressee == speaker_name:
        return None
    side = frame_side_label(str((spatial_layout or {}).get(addressee) or ""))
    return addressee, side


def _clean_voice_descriptor(speaker: str, voices: dict) -> str:
    voice = str(voices.get(speaker) or "").strip()
    voice = re.sub(r"\s+", " ", voice)
    voice = voice.replace("<", "").replace(">", "")
    # A trailing period splits the descriptor from the <d> tag. The
    # 12:49 sheet came back as "soft rounded timbre." and the speech
    # sentence became "timbre. <d>".
    voice = voice.rstrip(" .,;")
    return voice or default_voice_for(speaker)


def _speech_act_sentence(
    speaker: str,
    sid: str,
    voice: str,
    line: str,
    facing: Optional[tuple[str, str]] = None,
) -> str:
    """One node-owned sentence: voice descriptor, then the ``<d>`` tag.

    The grammar that survived live checks is ``as (Sn) <descriptor> <d>``
    with nothing between the descriptor and the tag, and no colon
    between ``(Sn)`` and ``<d>``. A descriptor in a ``voice:`` slot on
    the attribution line was ignored (C2/C3). A descriptor between
    ``(Sn)`` and the colon was lip-synced to the wrong character (C4).
    The same descriptor written paragraphs above the appended block
    was ignored (C5).
    """
    spoken = f"<d>{_detect_dialogue_language(line)} {line}</d>"
    if facing:
        who, side = facing
        side_bit = f" on the {side}" if side else ""
        if sid:
            head = (
                f"{speaker} turns to face {who}{side_bit}, "
                f"mouth opening as ({sid}) {voice}"
            )
        else:
            head = (
                f"{speaker} turns to face {who}{side_bit}, "
                f"mouth opening, {voice}"
            )
    elif sid:
        head = f"{speaker} speaks as ({sid}) {voice}"
    else:
        head = f"{speaker} speaks, {voice}"
    return f"{head} {spoken}"


def assemble_dialogue_line_blocks(
    dialogue_lines: list[str],
    *,
    line_speakers: Optional[list[str]] = None,
    turn_speaker: str = "",
    speaker_id_map: Optional[dict] = None,
    speaker_identities: Optional[dict] = None,
    first_appearance_speakers: Optional[set] = None,
    speaker_voices: Optional[dict] = None,
    concept: str = "",
    spatial_layout: Optional[dict] = None,
    role_bindings: Optional[dict] = None,
) -> list[str]:
    """Assemble one speech sentence per line. No identity line.

    The spoken words are copied. The voice descriptor sits immediately
    before the ``<d>`` tag. CAST / picture captions are not included —
    ``install_shot_speech`` pins those under ``subject_definitions``.
    A line-initial vocative turns the speaker toward the bound
    character. ``speaker_identities`` and ``first_appearance_speakers``
    are accepted so existing callers keep working; identity placement
    is no longer this function's job.
    """
    del speaker_identities, first_appearance_speakers
    sid_map = dict(speaker_id_map or {})
    voices = dict(speaker_voices or {})
    spk_per_line = list(line_speakers or [])
    if len(spk_per_line) != len(dialogue_lines):
        spk_per_line = [(turn_speaker or "").strip()] * len(dialogue_lines)
    bound_roles = extract_role_bindings(concept)
    # Caller-supplied bindings fill roles the concept text dropped
    # (a rewrite that deletes 「是爸爸」) and override on conflict.
    for role, name in dict(role_bindings or {}).items():
        if role and name:
            bound_roles[str(role)] = str(name)
    blocks: list[str] = []
    for i, line in enumerate(dialogue_lines):
        speaker = (spk_per_line[i] if i < len(spk_per_line) else "").strip()
        speaker = speaker or (turn_speaker or "(speaker)").strip()
        sid = str(sid_map.get(speaker) or "")
        voice = _clean_voice_descriptor(speaker, voices)
        facing = resolve_line_facing(
            line, speaker, bound_roles, spatial_layout, set(sid_map),
        )
        blocks.append(_speech_act_sentence(speaker, sid, voice, line, facing))
    return blocks


def identity_pin_lines(
    speakers: list[str],
    speaker_identities: Optional[dict],
    *,
    only: Optional[set] = None,
    max_chars: int = 0,
) -> list[str]:
    """One ``name: identity`` line per speaker, for a section that is
    not the speech sentence.

    Picture captions are hundreds of characters. Pasted on the line
    before a ``<d>`` tag they buried the mouth cue (live render
    2026-09-22 11:57). ``max_chars`` drops an identity that would
    dominate a three-section description; six-section boards pin the
    full caption under ``subject_definitions`` instead.
    """
    identities = dict(speaker_identities or {})
    limit = set(only) if only is not None else None
    pins: list[str] = []
    seen: set = set()
    for name in speakers:
        speaker = (name or "").strip()
        if not speaker or speaker in seen:
            continue
        if limit is not None and speaker not in limit:
            continue
        seen.add(speaker)
        raw = str(
            identities.get(speaker) or identities.get(speaker.lower()) or ""
        ).strip()
        if not raw:
            continue
        if max_chars and len(raw) > max_chars:
            continue
        pins.append(f"{speaker}: {raw}")
    return pins


_HIDE_SPEAKER_RE = re.compile(
    r"off-screen|off screen|voice-?over|\bPOV\b|point of view|eyeline|"
    r"first-person|first person|behind the camera|not in frame|"
    r"out of frame|as the .{0,40}camera|"
    r"第一人称|画外|不在画面|镜头外",
    re.IGNORECASE,
)
_VISIBLE_CAST = (
    "All speaking characters are visible in the frame. "
    "None of them is the camera."
)


def neutralize_offscreen_claims(prompt_lines: list[str]) -> list[str]:
    """Drop lines that hide a speaker or make them the camera.

    The 11:57 render made 白猫 the lens, so her lines had no mouth.
    Replacing the word ``POV`` with ``in frame`` left
    "First-person in frame" and "serve as the camera", which the
    12:49 plan still used. The whole line goes. Speech lines stay.
    """
    out: list[str] = []
    noted = False
    header_set = {h for h in _SIX_SECTION_HEADERS} | {h for h in _THREE_SECTION_HEADERS}
    for line in prompt_lines:
        key = line.strip().lower()
        if "<d>" in line or key in header_set:
            out.append(line)
            continue
        if _HIDE_SPEAKER_RE.search(line):
            if not noted:
                out.append(_VISIBLE_CAST)
                noted = True
            continue
        out.append(line)
    return out


def replace_description_body(
    prompt_lines: list[str],
    body_lines: list[str],
    *,
    schema: str = SCHEMA_THREE,
) -> list[str]:
    """Replace the description section body. Other sections stay.

    The shot model's beat prose is not kept. On the 11:57 render that
    prose was a second screenplay (mother off-screen, kitten facing
    the lens) and the speech sentences parked under it lost.
    """
    if not body_lines:
        return list(prompt_lines)
    headers = _SIX_SECTION_HEADERS if schema == SCHEMA_SIX else _THREE_SECTION_HEADERS
    target = (
        "detailed_description:" if schema == SCHEMA_SIX
        else "integrated_multimodal_description:"
    )
    header_set = {h for h in headers}
    start = None
    end = None
    for i, ln in enumerate(prompt_lines):
        key = ln.strip().lower()
        if key not in header_set:
            continue
        if key == target and start is None:
            start = i
            continue
        if start is not None:
            end = i
            break
    if start is None:
        return list(prompt_lines) + list(body_lines)
    if end is None:
        end = len(prompt_lines)
    out = list(prompt_lines)
    out[start + 1:end] = list(body_lines)
    return out


def install_shot_speech(
    prompt_lines: list[str],
    speech_lines: list[str],
    *,
    schema: str = SCHEMA_THREE,
    spatial_layout: Optional[dict] = None,
    speaker_identities: Optional[dict] = None,
    identity_speakers: Optional[list[str]] = None,
    first_appearance_speakers: Optional[set] = None,
) -> list[str]:
    """Put speech sentences in the description, identity somewhere else.

    Six-section (ref2va): the picture caption is pinned under
    ``subject_definitions``. Three-section: a short identity line is
    allowed at the top of the description, and only on the speaker's
    first clip. The description body is then the spatial sentence plus
    one speech sentence per line.
    """
    lines = neutralize_offscreen_claims(prompt_lines)
    speakers = [s for s in (identity_speakers or []) if s]
    if schema == SCHEMA_SIX and speakers:
        pins = identity_pin_lines(speakers, speaker_identities)
        if pins:
            lines = _insert_section_suffix(lines, "subject_definitions:", pins)
    body: list[str] = ["[Shot 1]"]
    spatial = build_spatial_layout_directive(spatial_layout)
    if spatial and not spatial.startswith("(no spatial"):
        body.append(spatial)
    if schema != SCHEMA_SIX:
        body.extend(identity_pin_lines(
            speakers,
            speaker_identities,
            only=set(first_appearance_speakers or ()),
            max_chars=180,
        ))
    body.extend(speech_lines)
    return replace_description_body(lines, body, schema=schema)


def _insert_section_suffix(
    prompt_lines: list[str],
    header: str,
    extra: list[str],
) -> list[str]:
    """Append ``extra`` at the end of the section that starts at ``header``."""
    if not extra:
        return list(prompt_lines)
    header_set = {h for h in _SIX_SECTION_HEADERS} | {h for h in _THREE_SECTION_HEADERS}
    start = None
    for i, ln in enumerate(prompt_lines):
        if ln.strip().lower() == header:
            start = i
            break
    if start is None:
        return list(prompt_lines)
    end = len(prompt_lines)
    for i in range(start + 1, len(prompt_lines)):
        if prompt_lines[i].strip().lower() in header_set:
            end = i
            break
    # Sit above the blank line that separates sections, when there is one.
    insert_at = end
    while insert_at > start + 1 and not prompt_lines[insert_at - 1].strip():
        insert_at -= 1
    out = list(prompt_lines)
    out[insert_at:insert_at] = list(extra)
    return out


def scrub_dialogue_from_prompt_text(
    text: str,
    dialogue_lines: list[str],
) -> tuple[str, list[str]]:
    """Strip any dialogue the stage-2 model wrote despite the lock.

    The stage-2 contract for a dialogue shot is "write visuals only";
    the node appends the verbatim blocks itself. If the model leaked
    anyway, this removes (a) every ``<d>...</d>`` block it emitted —
    otherwise the 1 line = 1 block invariant would double-count — and
    (b) bare verbatim copies of the locked lines, replaced by an
    ellipsis so surrounding prose still reads. Returns
    ``(scrubbed_text, notes)``; notes name what was removed.
    """
    notes: list[str] = []
    out = text
    model_blocks = re.findall(r"<d>\[[^\]]*\].*?</d>", out, re.DOTALL)
    for block in model_blocks:
        out = out.replace(block, "…", 1)
        notes.append(f"removed model-written {block[:40]!r}")
    for line in dialogue_lines:
        if line and line in out:
            out = out.replace(line, "…")
            notes.append(f"replaced leaked line {line[:40]!r}")
    return out, notes


def append_dialogue_blocks_to_sections(
    prompt_lines: list[str],
    blocks: list[str],
    *,
    schema: str = SCHEMA_THREE,
) -> list[str]:
    """Append the assembled dialogue blocks inside the right section.

    Three-section schema: end of ``integrated_multimodal_description``.
    Six-section schema: end of ``detailed_description``. Returns a new
    list; the input is not mutated.
    """
    if not blocks:
        return list(prompt_lines)
    headers = _SIX_SECTION_HEADERS if schema == SCHEMA_SIX else _THREE_SECTION_HEADERS
    target = "detailed_description:" if schema == SCHEMA_SIX else "integrated_multimodal_description:"
    header_set = {h for h in headers}
    header_idx = None
    for i, ln in enumerate(prompt_lines):
        if ln.strip().lower() in header_set:
            if ln.strip().lower() == target:
                header_idx = i
            elif header_idx is not None:
                break
    if header_idx is None:
        # Unknown shape: append at the end rather than lose the lines.
        return list(prompt_lines) + list(blocks)
    # Walk to the last non-empty line of the section body so the
    # blocks land after the prose, before the blank separator line.
    insert_at = header_idx + 1
    for i in range(header_idx + 1, len(prompt_lines)):
        if prompt_lines[i].strip().lower() in header_set:
            break
        if prompt_lines[i].strip():
            insert_at = i + 1
    out = list(prompt_lines)
    out[insert_at:insert_at] = blocks
    return out


def build_voice_performance_directive(
    dialogue_lines: Optional[list[str]],
    line_speakers: Optional[list[str]],
    turn_speaker: str,
    speaker_id_map: Optional[dict],
    speaker_voices: Optional[dict],
) -> str:
    """Tell the shot model the voice phrases the node will write.

    The node appends one speech sentence per line, with the descriptor
    glued to the ``<d>`` tag. The model must not write those phrases
    or any ``(S<n>)`` tag itself — a second copy paragraphs away is
    what C5 showed the TTS ignores, and a hand-written gaze can
    contradict the vocative facing on the same sentence.
    """
    if not dialogue_lines:
        return ""
    sid_map = dict(speaker_id_map or {})
    voices = dict(speaker_voices or {})
    spk = list(line_speakers or [])
    if len(spk) != len(dialogue_lines):
        spk = [turn_speaker or ""] * len(dialogue_lines)
    seen: set = set()
    phrases: list[str] = []
    for name in spk:
        n = (name or "").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        sid = sid_map.get(n)
        v = str(voices.get(n) or "").strip() or default_voice_for(n)
        phrases.append(
            f"{n} as ({sid}) {v}" if sid else f"{n} ({v})"
        )
    if not phrases:
        return ""
    example_voice = phrases[0].split(" as ")[-1]
    return (
        "Voice performance binding (the NODE writes this next to each "
        "<d> tag; you do not): every appended speech sentence glues "
        "the exact descriptor to that line in the form "
        f'"... as {example_voice} <d>...". '
        "Write NO (S<n>) tags and NO voice descriptors in your prose; "
        "do not restate these phrases:\n"
        + "\n".join(f"  - {p}" for p in phrases)
        + "\nWhen a locked line is marked with who it is spoken to, "
        "the same node sentence also turns the speaker toward them. "
        "Do not turn that speaker toward anyone else on that line.\n"
        "NEVER quote the spoken words themselves in prose (the node "
        "appends them); a character who does not speak in this clip "
        "gets no tag."
    )


def build_shot_user_text(
    *,
    concept: str,
    prefix_text: str,
    category: str,
    continuation_block: str,
    shot: dict,
    clip_index: int,
    clip_count: int,
    duration_seconds: Any,
    language_name: str,
    reference_directive: str = "",
    manifest_digest: str = "",
    cast_block: str = "",
    reference_mode: str = "t2va",
    dialogue_lines: Optional[list[str]] = None,
    turn_index: Optional[int] = None,
    turn_speaker: Optional[str] = None,
    speaker_id_map: Optional[dict] = None,
    line_speakers: Optional[list[str]] = None,
    first_appearance_speakers: Optional[set] = None,
    spatial_layout: Optional[dict] = None,
    tempo_directive: str = "",
    speaker_voices: Optional[dict] = None,
    role_bindings: Optional[dict] = None,
) -> str:
    # ``duration_seconds`` should be the clip's ACTUAL grid-rounded length
    # (``length_to_seconds(length)``) so the pacing budget matches what H3
    # will really generate, not the requested seconds.
    seconds = float(duration_seconds)
    sid_map = dict(speaker_id_map or {})
    spk_per_line = list(line_speakers or [])
    if dialogue_lines and len(spk_per_line) != len(dialogue_lines):
        spk_per_line = [(turn_speaker or "")] * len(dialogue_lines)
    if dialogue_lines:
        bound_roles = extract_role_bindings(concept)
        for role, name in dict(role_bindings or {}).items():
            if role and name:
                bound_roles[str(role)] = str(name)
        dlg_lines = []
        for i, line in enumerate(dialogue_lines):
            prefix = ""
            speaker = spk_per_line[i] if i < len(spk_per_line) else ""
            sid = sid_map.get(speaker) if speaker else None
            facing = resolve_line_facing(
                line, speaker, bound_roles, spatial_layout, set(sid_map),
            )
            aim = ""
            if facing:
                who, side = facing
                aim = (
                    f" -> {who} ({side} of frame)" if side else f" -> {who}"
                )
            if speaker and sid:
                prefix = f"{speaker} ({sid}){aim}: "
            elif speaker:
                prefix = f"{speaker}{aim}: "
            dlg_lines.append(f"  {i + 1}. {prefix}{line}")
        dlg_block = (
            f"Dialogue is LOCKED for this shot (turn {int(turn_index or 0)}, "
            f"speaker {(turn_speaker or '(unknown)').strip()}). The node "
            "REPLACES the description body with one sentence per line "
            "after you reply. You write NO dialogue: no <d> blocks, no "
            "quoted or unquoted spoken words (any copy is stripped). "
            "Write the room, the light, and the camera in the other "
            "sections. Do not describe mouths, turns, or who looks at "
            "whom — the node writes that sentence, with the voice "
            "descriptor glued to the verbatim <d> tag. Every character "
            "who has a line in this clip is IN FRAME and speaks with "
            "their own mouth; do not put them off-screen, behind the "
            "camera, or at the lens. A line marked `-> name (SIDE of "
            "frame)` is spoken TO that character:\n"
            + "\n".join(dlg_lines)
        )
        t_idx = int(turn_index or 0)
        t_spk = (turn_speaker or "(unknown)").strip()
    else:
        dlg_block = "(no dialogue lines assigned to this turn)"
        t_idx = int(turn_index or 0)
        t_spk = (turn_speaker or "(narrator)").strip()
    # Fixed speaker-ID directive: the map is derived deterministically
    # from the storyboard turn order, so every per-shot call sees the
    # exact same (S<n>) assignment and cannot renumber voices. On a
    # dialogue shot the tags ride on the node-appended speech sentences,
    # so the model is told to write NO tags in prose.
    if dialogue_lines:
        speaker_id_directive = (
            "Speaker tags are owned by the node: each appended speech "
            "sentence carries its speaker's fixed (S<n>) tag glued to "
            "the <d> block, and the speaker's first spoken clip gets a "
            "separate CAST identity line with no <d> tag "
            f"({format_speaker_id_map_text(sid_map) or 'no map'}). "
            "Write NO (S<n>) tags in your prose; never renumber, never "
            "invent tags; non-vocal on-screen characters get NO tag."
        )
    else:
        speaker_id_directive = ""
    return _SHOT_USER_TEMPLATE.format(
        concept=(concept or "").strip(),
        prompt_prefix=prefix_text.strip(),
        spatial_layout_directive=build_spatial_layout_directive(spatial_layout),
        genre_advice=_genre_advice_block(category),
        continuation_block=continuation_block.strip(),
        clip_index=int(clip_index),
        clip_count=int(clip_count),
        shot_json=json.dumps(
            {k: v for k, v in shot.items() if not str(k).startswith("_")},
            ensure_ascii=False,
            indent=2,
        ),
        duration_seconds=seconds,
        pacing_directive=pacing_directive(
            seconds,
            continued=clip_index > 1,
            reference_mode=reference_mode,
        ),
        language_name=language_name,
        dialogue_language_policy=build_dialogue_language_policy(dialogue_lines),
        reference_directive=(reference_directive or "").strip(),
        manifest_digest=(manifest_digest or "no reference images").strip(),
        cast_block=(cast_block or "(none named)").strip(),
        turn_index=t_idx,
        turn_speaker=t_spk,
        dialogue_lines_block=dlg_block,
        speaker_id_directive=(
            speaker_id_directive
            + "\n\n"
            + build_voice_performance_directive(
                dialogue_lines,
                spk_per_line,
                t_spk,
                speaker_id_map,
                speaker_voices,
            )
        ).strip(),
        tempo_directive=(tempo_directive or "").strip(),
    )


def build_single_call_user_text(
    *,
    concept: str,
    prefix_text: str,
    category: str,
    shots: list[dict],
    duration_seconds: int,
    language_name: str,
    cast_sheet: str = "",
    speaker_id_map: Optional[dict] = None,
    spatial_layout: Optional[dict] = None,
    tempo_directive: str = "",
) -> str:
    board = json.dumps(shots, ensure_ascii=False, indent=2)
    map_text = format_speaker_id_map_text(speaker_id_map or {})
    speaker_map_block = (
        "Speaker ID map (FIXED for the whole production): "
        f"{map_text}\n"
        "The node attaches each speaker's (S<n>) tag — and, at their "
        "first spoken clip, their CAST identity — to the appended speech "
        "blocks. Write NO (S<n>) tag in your prose; non-vocal on-screen "
        "characters get NO tag.\n\n"
        if map_text
        else ""
    )
    spatial_block = (
        "Spatial layout for every clip (binding — must match the prefix "
        "above and must be preserved exactly from any prior clip that "
        f"named these positions):\n{build_spatial_layout_directive(spatial_layout)}\n\n"
    )
    tempo_block = (
        f"{(tempo_directive or '').strip()}\n\n"
        if (tempo_directive or "").strip()
        else ""
    )
    has_dialogue = any(shot.get("_dialogue_lines") for shot in shots)
    all_dialogue_lines = [
        str(ln)
        for shot in shots
        for ln in (shot.get("_dialogue_lines") or [])
        if str(ln).strip()
    ]
    dialogue_lock_block = (
        "Dialogue is LOCKED as data: the node REPLACES each clip's "
        "description body with one sentence per _dialogue_lines entry. "
        "You write NO dialogue anywhere: no <d> blocks, no quoted or "
        "unquoted spoken words (any copy is stripped automatically). "
        "The node glues each speaker's voice descriptor onto that "
        "line's <d> tag. Every speaking character stays in frame; do "
        "not write them off-screen or as the camera. Describe the "
        "room, the light, and the camera only.\n\n"
        if has_dialogue
        else ""
    )
    return (
        f"Concept (whole production):\n{(concept or '').strip()}\n\n"
        f"Shared style/setting prefix (binding for every clip):\n{prefix_text.strip()}\n\n"
        f"{spatial_block}"
        f"{tempo_block}"
        f"Cast sheet (pick each clip's on-screen members by exact name):\n"
        f"{cast_sheet or '(none named)'}\n\n"
        f"{_genre_advice_block(category)}\n\n"
        f"{speaker_map_block}"
        f"{dialogue_lock_block}"
        f"Storyboard entries (ALL {len(shots)} clips, in order):\n{board}\n\n"
        f"Average clip duration: {int(duration_seconds)} seconds (each entry's own "
        f"duration_seconds in the board above is binding). Output language: {language_name}.\n\n"
        f"Language policy: narrative prose follows {language_name}. "
        f"{build_dialogue_language_policy(all_dialogue_lines)}\n\n"
        f"{SINGLE_CALL_FORMAT.format(clip_count=len(shots)).strip()}"
    )


def split_prefix_paragraphs(text: str) -> list[str]:
    """Split a prefix synthesis reply into paragraphs (blank-line
    separated, outer whitespace stripped, empties dropped)."""
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()
    ]
    return paragraphs


def split_prefix_and_cast(raw: str) -> tuple[list[str], dict[str, str]]:
    """Split a stage-1 reply into ``(prefix paragraphs, cast sheet)``.

    Expected shape: one or more prefix paragraphs, then a line starting
    with ``CAST:`` followed by one ``name: identity`` line per named
    character. A missing CAST marker yields an empty dict (the caller's
    retry trigger when a roster was requested). Cast keys are normalized
    to lowercase for roster matching; malformed lines are skipped."""
    lines, _, cast, _ = split_prefix_sections(raw)
    return lines, cast


def split_prefix_sections(raw: str) -> tuple:
    """Full stage-1 reply parse: ``(prefix paragraphs, end_idx, cast,
    voices)``. Sections are ``CAST:`` then optional ``VOICE:`` — each a
    header line followed by ``name: value`` lines. The voice sheet is
    the per-speaker TTS descriptor (gender + pitch + timbre); the
    upstream guide requires it beside the (S<n>) tag, and scenes
    generate independently so it must be re-attachable per scene."""
    text = (raw or "").strip()
    if not text:
        return [], -1, {}, {}
    lines = text.split("\n")

    def _section_header_idx(marker: str, start: int) -> int:
        return next(
            (
                i
                for i in range(start, len(lines))
                if lines[i].strip().lower().startswith(marker)
            ),
            -1,
        )

    cast_idx = _section_header_idx("cast:", 0)
    if cast_idx < 0:
        return split_prefix_paragraphs(text), -1, {}, {}
    voice_idx = _section_header_idx("voice:", cast_idx + 1)
    cast_end = voice_idx if voice_idx >= 0 else len(lines)
    prefix_part = "\n".join(lines[:cast_idx])

    def _parse_pairs(seg: list[str]) -> dict:
        out: dict = {}
        for ln in seg:
            entry = ln.strip()
            if not entry or ":" not in entry:
                continue
            name, _, value = entry.partition(":")
            name = name.strip()
            value = value.strip()
            if name and value:
                out[name.lower()] = value
        return out

    cast = _parse_pairs(lines[cast_idx + 1 : cast_end])
    voices = (
        _parse_pairs(lines[voice_idx + 1 :]) if voice_idx >= 0 else {}
    )
    return split_prefix_paragraphs(prefix_part), cast_idx, cast, voices


# Heuristic voice fallback for boards that never ran the prefix LLM
# (local-prefix path) or whose VOICE sheet is missing entries. Gender
# markers cover the common 爸爸/妈妈/女儿-style role names and the
# English equivalents; unknown names get a neutral adult descriptor —
# a stable neutral voice still pins cross-scene consistency, which is
# what the TTS actually needs.
_VOICE_FEMALE_MARKERS = (
    "妈", "母", "女", "娘", "姐", "妹", "婆", "妻", "婶", "嫂", "姑",
    "mother", "mom", "mama", "mum", "daughter", "girl", "sister", "wife",
    "she", "her", "female", "queen", "princess", "aunt",
)
_VOICE_MALE_MARKERS = (
    "爸", "父", "公", "男", "哥", "弟", "爷", "夫", "叔", "伯", "兄",
    "father", "dad", "papa", "son", "boy", "brother", "husband", "he",
    "him", "male", "king", "prince", "uncle",
)


def default_voice_for(name: str) -> str:
    """Stable per-name TTS descriptor (gender + pitch + timbre) for the
    voice sheet fallback. Same name in -> same descriptor out (the
    stability IS the feature: cross-scene voice consistency)."""
    low = (name or "").lower()
    if any(m in low for m in _VOICE_FEMALE_MARKERS):
        return "adult female, warm mid-range pitch, soft rounded timbre"
    if any(m in low for m in _VOICE_MALE_MARKERS):
        return "adult male, low-mid pitch, steady timbre"
    return "adult, mid-range pitch, natural timbre"


def build_cast_block(cast: dict[str, str], roster: list[str]) -> str:
    """Identity lines for THIS clip's on-screen roster, drawn from the
    cast sheet. Names match case-insensitively; an empty roster (boards
    without ``characters`` arrays) falls back to the whole sheet so
    identity anchoring still happens. Empty sheet -> ``""``."""
    if not cast:
        return ""
    if roster:
        lines = [
            f"{name}: {cast[name.lower()]}"
            for name in roster
            if name.lower() in cast
        ]
    else:
        lines = [f"{name}: {identity}" for name, identity in cast.items()]
    return "\n".join(lines)


def build_cast_sheet_text(cast: dict[str, str]) -> str:
    """The whole cast sheet as text (single-call mode writes every clip
    in one reply and picks from the sheet itself)."""
    return "\n".join(f"{name}: {identity}" for name, identity in cast.items())


def log_pipeline(message: str) -> None:
    mie_log(f"H3LOOP: {message}")


# --------------------------------------------------------------------------- #
# Dialogue invariantity validator
# --------------------------------------------------------------------------- #
# Matches a single ``<d>[Chinese] ...</d>`` or ``<d>[English] ...</d>``
# block. Used by ``validate_dialogue_invariantity`` to enforce:
#   (1) every dialogue line in the concept lands as exactly one block
#   (2) the text inside the block is verbatim (whitespace-stripped)
#       equal to the input line.
_D_TAG_RE = re.compile(
    r"<d>\[(?:Chinese|English)\](?P<text>.*?)</d>",
    re.DOTALL | re.IGNORECASE,
)


def extract_d_blocks(text: str) -> list[str]:
    """Return the inner text of every ``<d>...</d>`` block (stripped)."""
    return [m.group("text").strip() for m in _D_TAG_RE.finditer(text or "")]


def count_d_blocks(text: str) -> int:
    """Count well-formed ``<d>...</d>`` blocks (paired tags only).

    Implemented on top of ``extract_d_blocks`` so the count and the
    extracted list can never disagree about what counts as a block —
    production paths only ever call ``extract_d_blocks`` so the old
    head-only count was a footgun."""
    return len(extract_d_blocks(text))


def validate_dialogue_invariantity(
    plan_shots: list[dict],
    turns: list,  # list[DialogueTurn]; kept untyped to avoid circular import
) -> list[str]:
    """Verify each plan shot preserves the dialogue invariantity contract.

    Errors returned (empty list = pass):
      - shot count != turn count
      - per-shot <d> block count != turn.line_count
      - any <d> block inner text != turn.lines[i] (verbatim, stripped)

    The function is intentionally strict; the generator calls it after
    each per-shot LLM reply and retries on failure (max 2 attempts).
    """
    errors: list[str] = []
    if len(plan_shots) != len(turns):
        errors.append(
            f"shot count {len(plan_shots)} != turn count {len(turns)}"
        )
    for idx, (shot, turn) in enumerate(zip(plan_shots, turns)):
        prompt_field = shot.get("prompt") or shot.get("description") or ""
        if isinstance(prompt_field, list):
            prompt_text = "\n".join(prompt_field)
        else:
            prompt_text = str(prompt_field)
        d_blocks = extract_d_blocks(prompt_text)
        if len(d_blocks) != len(turn.lines):
            errors.append(
                f"shot {idx}: expected {len(turn.lines)} <d> blocks "
                f"(turn '{turn.speaker}' has {len(turn.lines)} lines), "
                f"got {len(d_blocks)}"
            )
            continue
        for i, (block_text, expected) in enumerate(zip(d_blocks, turn.lines)):
            if block_text != expected.strip():
                errors.append(
                    f"shot {idx} line {i}: verbatim mismatch "
                    f"(speaker '{turn.speaker}')\n"
                    f"  expected: {expected!r}\n"
                    f"  got:      {block_text!r}"
                )
    return errors


# --------------------------------------------------------------------- #
# Speaker-ID contract (official H3 rule: "A speaker keeps the same ID
# across shots; non-vocal characters get no ID"). Per-shot LLM calls
# cannot keep IDs stable on their own — each call only sees one shot —
# so the map is derived deterministically from the storyboard's turn
# order and mechanically enforced on every reply.
# --------------------------------------------------------------------- #
_SPEAKER_ID_RE = re.compile(r"\(S(\d+)\)")
_GENDER_WORD_RE = re.compile(
    r"\b(?:female|male|woman|man|women|men|girl|boy|gentleman|lady)\b|[男女]",
    re.IGNORECASE,
)


def build_speaker_id_map(shots: list) -> dict:
    """Deterministic ``speaker -> "S<n>"`` map ordered by each speaker's
    first spoken line across the storyboard (``_line_speakers`` when the
    shot is a packed multi-turn scene, ``_turn_speaker`` otherwise).

    Empty map for narration boards (no speaking lines anywhere) —
    the legacy no-map behaviour applies there."""
    order: list[str] = []
    for shot in shots or []:
        per_line = (shot or {}).get("_line_speakers") or []
        candidates = (
            [str(s).strip() for s in per_line]
            if per_line
            else [str((shot or {}).get("_turn_speaker") or "").strip()]
        )
        for speaker in candidates:
            if speaker and speaker not in order:
                order.append(speaker)
    return {name: f"S{i + 1}" for i, name in enumerate(order)}


def format_speaker_id_map_text(id_map: dict) -> str:
    """Render a map as ``莎莉猫=(S1), 哈利猫=(S2)`` for prompt injection."""
    return ", ".join(
        f"{name}=({sid})" for name, sid in (id_map or {}).items()
    )


def repair_speaker_ids(
    lines: list,
    speaker: str,
    sid: str,
    *,
    first_appearance: bool = False,
) -> tuple:
    """Mechanically enforce the speaker-ID contract on one shot reply.

    ``lines`` is the split section body of a single shot. Every line
    carrying a ``<d>`` block is a speaking paragraph; any ``(S<n>)``
    tag inside it is rewritten to the speaker's mapped ``sid`` — the
    model cannot renumber voices even when it tries. Non-speaking
    lines are left untouched.

    Returns ``(repaired_lines, problems)``. Problems are contract
    violations a retry should fix (empty list = clean):
      - the speaking paragraph carries no ``(S<n>)`` tag at all;
      - the speaker's FIRST spoken clip lacks a gender word beside the
        tag (the upstream guide requires identity — gender / pitch /
        timbre — at first appearance).
    """
    speaking_idx = [i for i, ln in enumerate(lines) if "<d" in ln]
    out_lines: list = []
    for i, ln in enumerate(lines):
        if i in speaking_idx:
            ln = _SPEAKER_ID_RE.sub(f"({sid})", ln)
        out_lines.append(ln)
    problems: list[str] = []
    if speaking_idx:
        speaking_text = "\n".join(lines[i] for i in speaking_idx)
        if not _SPEAKER_ID_RE.search(speaking_text):
            problems.append(
                f"speaking paragraph carries no (S<n>) tag; attach ({sid}) "
                f"to speaker '{speaker}' OUTSIDE the <d> block, next to "
                "their voice identity"
            )
        elif first_appearance and not _GENDER_WORD_RE.search(speaking_text):
            problems.append(
                f"first spoken appearance of '{speaker}' must state the "
                f"voice identity (gender + pitch + timbre) beside the "
                f"({sid}) tag"
            )
    return out_lines, problems


_D_BLOCK_SPLIT_RE = re.compile(r"(<d>\[[^\]]*\].*?</d>)")


def repair_speaker_ids_for_lines(
    lines: list,
    line_speakers: list,
    sid_map: dict,
    *,
    first_appearance_speakers: Optional[set] = None,
) -> tuple:
    """Multi-speaker variant of ``repair_speaker_ids`` for packed scenes.

    ``line_speakers[k]`` names the speaker of the k-th ``<d>`` block (in
    order). Delivery prose for line k sits between the previous block's
    ``</d>`` and block k's ``<d>`` — every ``(S<n>)`` tag in that segment
    is rewritten to line k's mapped ID. Text after the LAST block on a
    line is left untouched (closing reaction; ownership ambiguous).

    Contract checks (retry-worthy problems):
      - block-count vs speaker-list mismatch (cannot map reliably);
      - a speaker's FIRST block in this shot carries no tag;
      - a speaker's first spoken clip of the video (in
        ``first_appearance_speakers``) lacks a gender word at their first
        block in this shot.
    Returns ``(repaired_lines, problems)``.
    """
    firsts = set(first_appearance_speakers or ())
    out_lines: list = []
    problems: list[str] = []
    block_cursor = 0
    seen_in_shot: set = set()
    # Walk text lines that carry blocks; others pass through untouched.
    for ln in lines:
        if "<d>" not in ln:
            out_lines.append(ln)
            continue
        parts = _D_BLOCK_SPLIT_RE.split(ln)
        # parts alternate: text, block, text, block, ..., text
        # Segment before block j introduces line (base + j); text after
        # the last block is a tail (untouched for rewriting, but counts
        # toward the LAST line's gender region).
        rebuilt: list[str] = []
        block_count = sum(1 for p in parts if p.startswith("<d>"))
        first_base = block_cursor
        block_cursor += block_count
        tail_text = parts[-1] if len(parts) % 2 == 1 else ""
        block_no = 0
        for p in parts:
            if p.startswith("<d>"):
                rebuilt.append(p)
                block_no += 1
                continue
            # Text segment: it introduces the NEXT block if any remains
            # in this line; else it is the tail.
            next_in_line = block_no < block_count
            if not p:
                rebuilt.append(p)
                continue
            if next_in_line:
                line_idx = first_base + block_no
                if line_idx >= len(line_speakers):
                    rebuilt.append(p)
                    continue
                speaker = line_speakers[line_idx]
                sid = (sid_map or {}).get(speaker)
                new_p = (
                    _SPEAKER_ID_RE.sub(f"({sid})", p)
                    if sid
                    else p
                )
                rebuilt.append(new_p)
                seg = new_p
                # Gender region for the last block also includes the tail.
                extra = tail_text if line_idx == first_base + block_count - 1 else ""
                _check_segment(
                    seg + " " + extra,
                    speaker,
                    sid,
                    line_idx,
                    line_speakers,
                    seen_in_shot,
                    firsts,
                    problems,
                )
            else:
                rebuilt.append(p)
        out_lines.append("".join(rebuilt))
    if block_cursor != len(line_speakers):
        problems.insert(
            0,
            f"speaker-ID map mismatch: {block_cursor} <d> block(s) but "
            f"{len(line_speakers)} speaker entr(y|ies) provided",
        )
    return out_lines, problems


def _check_segment(
    region: str,
    speaker: str,
    sid,
    line_idx: int,
    line_speakers: list,
    seen_in_shot: set,
    firsts: set,
    problems: list,
) -> None:
    """Shared per-line contract checks for the multi-speaker repair."""
    if sid is None:
        return
    if speaker not in seen_in_shot:
        seen_in_shot.add(speaker)
        if not _SPEAKER_ID_RE.search(region):
            problems.append(
                f"line {line_idx + 1}: speaker '{speaker}' carries no "
                f"(S<n>) tag; attach ({sid}) OUTSIDE the <d> block, next "
                "to their voice identity"
            )
        elif speaker in firsts and not _GENDER_WORD_RE.search(region):
            problems.append(
                f"line {line_idx + 1}: first spoken appearance of "
                f"'{speaker}' must state the voice identity (gender + "
                f"pitch + timbre) beside the ({sid}) tag"
            )

