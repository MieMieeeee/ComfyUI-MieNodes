"""Storyboard templates, whitelists, and parsing for the
``MiniMaxH3StoryboardGenerator`` node.

The node's methodology (beats before shots, hook, shot-size variety,
character anchor, transition logic, pacing, continuity bible) is encoded
in the bundled system prompt ``prompts/h3_storyboard/system_storyboard.txt``
(sources listed in ``prompts/h3_storyboard/UPSTREAM.md``). This module
owns everything deterministic around it:

* dropdown tuples + code parsing (styles / output formats / languages),
* ``STYLE_ADVICE`` — per-style concrete guidance injected into the user
  turn (same pattern as ``h3_prompts.CATEGORY_ADVICE``),
* shot_type / transition whitelists used both by the prompt contract and
  by ``normalize_shots`` post-validation,
* ``extract_json_array`` — tolerant JSON-array extraction (bare array,
  ```json fenced, prose-wrapped),
* ``normalize_shots`` — coerce + validate the LLM output into the
  canonical shot dicts (unique ids, whitelist fallbacks, duration
  clamping, count reconciliation),
* ``render_storyboard_markdown`` — the human-readable ``storyboard_text``
  output in three formats (table / detailed / minimal).

Genre dropdown reuses ``h3_prompts.CATEGORIES`` / ``CATEGORY_ADVICE`` so
the storyboard and the H3 prompt generator share one taxonomy.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:
        from core.utils import mie_log

try:
    from _mienodes_internal.nodes.llm.h3_prompts import (
        CATEGORIES,
        category_advice,
        parse_category,
    )
except ImportError:
    from .h3_prompts import (
        CATEGORIES,
        category_advice,
        parse_category,
    )

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_text


# --------------------------------------------------------------------------- #
# Dropdowns. Display strings follow the project-wide "<code> - <label>"
# contract (literal ASCII " - " separator) so ``parse_*`` can split the
# code back out; Chinese sub-labels use "/" internally, never " - ".
# --------------------------------------------------------------------------- #
STYLES = (
    "narrative_arc - 叙事弧光/三幕",
    "parallel_montage - 平行蒙太奇",
    "rhythmic_cuts - 节奏剪辑",
    "single_continuous - 一镜到底",
    "character_study - 人物弧光",
)

STYLE_CODES = (
    "narrative_arc",
    "parallel_montage",
    "rhythmic_cuts",
    "single_continuous",
    "character_study",
)

DEFAULT_STYLE = STYLE_CODES[0]

OUTPUT_FORMATS = (
    "table - Markdown表格",
    "detailed - 每场多字段",
    "minimal - 仅id+描述",
)

OUTPUT_FORMAT_CODES = (
    "table",
    "detailed",
    "minimal",
)

DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_CODES[0]

LANGUAGES = ("en", "zh")
DEFAULT_LANGUAGE = "en"

_LANGUAGE_NAMES = {"en": "English", "zh": "Chinese"}


def language_name(language: str) -> str:
    """Display name for an output-language code (fallback: the raw code)."""
    return _LANGUAGE_NAMES.get((language or "").strip().lower(), (language or "en"))


def parse_style(style: str) -> str:
    """Extract the style code from a display string / bare code / None."""
    if not style:
        return style
    return style.split(" - ", 1)[0].strip()


def parse_output_format(output_format: str) -> str:
    """Extract the output-format code from a display string / bare code."""
    if not output_format:
        return output_format
    return output_format.split(" - ", 1)[0].strip()


# --------------------------------------------------------------------------- #
# Style advice (English keyword guidance injected into the user turn).
# --------------------------------------------------------------------------- #
STYLE_ADVICE = {
    "narrative_arc": (
        "Three-act shape scaled to the shot count (setup -> complication -> "
        "resolution). Escalate shot intimacy as emotion rises; the final "
        "beat lands the theme on a concrete image."
    ),
    "parallel_montage": (
        "Two or more interwoven lines of action alternate; each cut switches "
        "lines but keeps a shared rhythm. Mirror shot sizes between lines and "
        "converge the lines in the final shot."
    ),
    "rhythmic_cuts": (
        "Cut on a beat; progressively shorter durations build to a peak, then "
        "one longer shot releases the energy. Use matched motion across cuts "
        "(whip pans, matched movement direction)."
    ),
    "single_continuous": (
        "One unbroken take segmented for generation: every entry is a stretch "
        "of the SAME continuous camera move; transition_in is invisible_cut "
        "for every entry after the first; camera movement must chain smoothly "
        "(a move ending leftward begins the next still moving leftward)."
    ),
    "character_study": (
        "One protagonist in frame throughout; the board tracks an emotional "
        "arc through changing distance (far -> near), light, and micro-"
        "actions; the final shot reads the feeling on the face or hands."
    ),
}


def style_advice(style: str) -> str:
    code = parse_style(style)
    if code in STYLE_ADVICE:
        return STYLE_ADVICE[code]
    return STYLE_ADVICE[DEFAULT_STYLE]


# --------------------------------------------------------------------------- #
# Whitelists (the prompt contract and normalize_shots share them).
# --------------------------------------------------------------------------- #
SHOT_TYPES = (
    "extreme_wide_shot",
    "wide_shot",
    "medium_wide_shot",
    "medium_shot",
    "medium_close_up",
    "close_up",
    "extreme_close_up",
    "over_the_shoulder",
    "point_of_view",
    "aerial_shot",
    "two_shot",
    "insert_shot",
)

TRANSITIONS = (
    "hard_cut",
    "fade_from_black",
    "fade_to_black",
    "cross_dissolve",
    "match_cut",
    "smash_cut",
    "wipe",
    "whip_pan",
    "invisible_cut",
    "cut_in",
    "cut_away",
)

DEFAULT_SHOT_TYPE = "medium_shot"
# First shot opens the board; a bad/unknown value falls back to this.
DEFAULT_FIRST_TRANSITION = "fade_from_black"
DEFAULT_TRANSITION = "hard_cut"

DEFAULT_DURATION_SECONDS = 8
MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 60

# Prompted hard cap; ``normalize_shots`` never returns more than this.
MAX_SHOTS = 128


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
SYSTEM_STORYBOARD_PROMPT = load_prompt_text("h3_storyboard/system_storyboard")
_USER_TEMPLATE = load_prompt_text("h3_storyboard/user_storyboard_template")


def build_user_text(
    concept: str,
    shot_count: int,
    style: str,
    genre: str,
    language: str,
    *,
    total_duration_seconds: Any = None,
    reference_digest: str = "",
    genre_tags: Optional[list[str]] = None,
    split_bias: str = "balanced",
) -> str:
    """Build the storyboard user turn from the bundled template.

    ``shot_count <= 0`` asks the LLM to decide the count itself (used by
    the loop node's auto-storyboard). ``total_duration_seconds`` (loop
    node only) injects a whole-board duration budget the per-shot
    ``duration_seconds`` values must sum close to; None keeps the
    standalone storyboard behaviour unchanged. ``reference_digest``
    (loop node only, image modes) carries one digest line per reference
    caption; the standalone node never passes it (the empty default
    keeps existing callers byte-identical)."""
    n = int(shot_count or 0)
    bias_code = str(split_bias or "balanced").split(" - ", 1)[0].strip().lower()
    if bias_code not in {"balanced", "conservative", "aggressive"}:
        bias_code = "balanced"
    split_bias_directive = {
        "conservative": (
            "- split bias: conservative. Prefer fewer, longer shots near "
            "the upper half of the 4..14s band; only split when the beat "
            "clearly changes audience understanding."
        ),
        "aggressive": (
            "- split bias: aggressive. Prefer more, shorter shots near "
            "the lower half of the 4..14s band, but still avoid empty "
            "micro-splits with no narrative change."
        ),
        "balanced": (
            "- split bias: balanced. Choose a natural middle cadence in "
            "the 4..14s band and split only on meaningful beat changes."
        ),
    }[bias_code]
    if n > 0:
        count_directive = f"- shot_count: exactly {n} shots."
        count_directive += "\n" + split_bias_directive
        closing = f"exactly {n}"
    else:
        if total_duration_seconds and int(total_duration_seconds) > 0:
            total = int(total_duration_seconds)
            min_count = max(1, int(math.ceil(total / 14.0)))
            max_count = max(min_count, int(total // 4))
            count_directive = (
                "- shot_count: you decide. Keep each shot duration_seconds "
                "within 4..14 seconds, and split only when a new beat "
                "changes viewer understanding (no meaningless micro-splits). "
                f"For this ~{total}s budget, a natural range is roughly "
                f"{min_count}..{max_count} shots; if the material truly "
                "needs outside this range, explain it in notes."
            )
            count_directive += "\n" + split_bias_directive
        else:
            count_directive = (
                "- shot_count: you decide. Keep each shot duration_seconds "
                "within 4..14 seconds, and split only on meaningful beats "
                "(no meaningless micro-splits)."
            )
            count_directive += "\n" + split_bias_directive
        closing = "the count you chose"
    if total_duration_seconds:
        budget_directive = (
            f"- total duration budget: ~{int(total_duration_seconds)} seconds "
            "across ALL shots COMBINED. Choose each shot's duration_seconds so "
            "the sum lands close to that budget (each clip later rounds up "
            "onto the 17n+5 raw frame grid, so being a few seconds off is "
            "fine). Spend longer on key beats, shorter on connective beats."
        )
    else:
        budget_directive = ""
    digest = (reference_digest or "").strip()
    if digest:
        reference_block = (
            "- reference keyframes (BINDING): the cast, wardrobe, hero props and "
            "key settings MUST come from the reference captions below. Use the "
            "exact same character names in each shot's `characters` array; "
            "never invent, rename, merge, or contradict them. Sequence the "
            "beats so the story tours the keyframes in slot order.\n"
            "- reference captions (slot order = story order):\n"
            + "\n".join("  " + line for line in digest.splitlines())
        )
    else:
        reference_block = ""
    style_code = parse_style(style)
    style_label = next(
        (s for s in STYLES if parse_style(s) == style_code), STYLES[0]
    )
    genre_code = parse_category(genre)
    advice = category_advice(genre_code).strip()
    if not advice:
        advice = "no genre-specific guidance; follow the concept as written"
    norm_genre_tags: set[str] = set()
    if isinstance(genre_tags, list):
        for item in genre_tags:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key:
                norm_genre_tags.add(key)
    spoken_hit = bool(
        norm_genre_tags
        & {"dialogue", "jokes", "banter", "monologue", "argue", "voiceover", "musical"}
    )
    spoken_guidance = ""
    if spoken_hit:
        if n > 0:
            shot_shape_line = (
                f"- spoken-scene guidance: keep the requested {n} shots, but cut by "
                "complete speaking beats rather than partial lines."
            )
        elif total_duration_seconds and int(total_duration_seconds) <= 20:
            shot_shape_line = (
                "- spoken-scene guidance: prefer 2 to 3 shots for this duration budget; "
                "only exceed that if the material truly needs more distinct speaking beats."
            )
        else:
            shot_shape_line = (
                "- spoken-scene guidance: prefer fewer, longer shots for speaking beats; "
                "do not oversplit a simple talk scene into many tiny clips."
            )
        spoken_guidance = (
            shot_shape_line + "\n"
            "- spoken-scene guidance: align shot boundaries to a completed utterance or "
            "a clear reaction beat. Do NOT cut in the middle of a spoken sentence.\n"
            "- spoken-scene guidance: if dialogue continues across shots, finish the "
            "audible sentence first, then hand off on the pause / breath / reaction / "
            "camera continuation."
        )
    lang = (language or DEFAULT_LANGUAGE).strip().lower()
    if lang not in LANGUAGES:
        lang = DEFAULT_LANGUAGE
    return _USER_TEMPLATE.format(
        concept=(concept or "").strip(),
        count_directive=count_directive,
        style_name=style_label,
        style_advice=style_advice(style_code),
        genre_advice=advice,
        spoken_guidance=spoken_guidance,
        language=lang,
        duration_budget=budget_directive,
        reference_block=reference_block,
        closing_directive=closing,
    )


# --------------------------------------------------------------------------- #
# JSON extraction (tolerant: bare array, ```json fence, prose-wrapped,
# {"shots": [...]}-wrapped, trailing commas, fullwidth quotes)
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
# LLMs writing Chinese-adjacent content sometimes emit fullwidth quotes /
# colons inside JSON; normalized before a second parse attempt.
_FULLWIDTH_MAP = str.maketrans(
    {
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\uff1a": ":",  # ：
    }
)

# Appended as a corrective user turn when a storyboard reply fails to parse.
PARSE_RETRY_CORRECTION = (
    "Your previous reply could not be parsed as a JSON array — it was cut "
    "off before the closing ]. Reply again with ONLY the complete JSON "
    "array of storyboard entries: no prose, no markdown fences, no code "
    "block, compact JSON without extra whitespace, SHORT strings (each "
    "description at most 2 sentences, each notes at most 1 short "
    "sentence). Start directly with [ and END with ]."
)


def _repaired(text: str) -> str:
    """Best-effort repair of common LLM JSON quirks (trailing commas,
    fullwidth quotes). Returns the input unchanged when already clean."""
    normalized = text.translate(_FULLWIDTH_MAP)
    return _TRAILING_COMMA_RE.sub(r"\1", normalized)


def _salvage_truncated_array(text: str) -> list[Any] | None:
    """Salvage a JSON array that was cut off before its closing ``]``
    (a max_tokens truncation — the live failure mode for long boards).

    Scans with a string-aware state machine, notes the end offset of
    every COMPLETE top-level element, then re-parses the text truncated
    after the last complete element plus a closing bracket. Returns the
    parsed list, or None when not even one complete element exists."""
    normalized = text.translate(_FULLWIDTH_MAP).strip()
    if not normalized.startswith("["):
        return None
    depth = 0
    in_string = False
    escaped = False
    last_complete = -1  # offset just past the last complete element's '}'
    for i, ch in enumerate(normalized):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 1 and ch == "}" and normalized.startswith("["):
                # A top-level array element just closed.
                last_complete = i + 1
            elif depth == 0 and ch == "]":
                return None  # the array closes — not truncated; let
                # the normal candidates handle it
    if last_complete < 0:
        return None
    salvaged = normalized[:last_complete] + "]"
    try:
        data = json.loads(salvaged)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    mie_log(
        f"H3SB salvage: storyboard reply was truncated; recovered "
        f"{len(data)} complete entries"
    )
    return data


def _salvage_elementwise(text: str) -> list[Any] | None:
    """Element-wise salvage for a board whose JSON is corrupted INSIDE one
    element (live failure: an unescaped quote in a description string
    breaks json.loads for the whole array). Scans with a string-aware
    state machine for every top-level ``{...}`` span, parses each element
    on its own, and keeps the ones that load. Returns the surviving
    dicts (>=2 — a single survivor is more likely a false-positive span
    than a board), or None."""
    normalized = text.translate(_FULLWIDTH_MAP)
    elements: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    obj_start = -1
    for i, ch in enumerate(normalized):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                elements.append(normalized[obj_start : i + 1])
                obj_start = -1
            if depth < 0:
                depth = 0
    out: list[dict] = []
    for elem in elements:
        for cand in (elem, _repaired(elem)):
            try:
                data = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("description", data.get("id")):
                out.append(data)
                break
    if len(out) < 2:
        return None
    mie_log(
        f"H3SB salvage: storyboard array corrupted mid-element; recovered "
        f"{len(out)} of {len(elements)} entries element-wise"
    )
    return out


def extract_json_array(text: str) -> list[Any]:
    """Extract the storyboard shots from an LLM reply.

    Tries, in order: the stripped text, each ```-fenced block, then the
    outermost ``[...]`` span — each both raw and quirk-repaired. A plain
    array wins; a parsed ``{"shots": [...]}`` object is unwrapped as a
    fallback. Raises ``ValueError`` when nothing yields shots.
    """
    if not text or not text.strip():
        raise ValueError("empty storyboard reply")
    bases: list[str] = [text.strip()]
    bases.extend(m.strip() for m in _FENCE_RE.findall(text))
    depth = 0
    in_string = False
    escaped = False
    start = -1
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    bases.append(text[start : i + 1])
    candidates: list[str] = []
    for base in bases:
        candidates.append(base)
        repaired = _repaired(base)
        if repaired != base:
            candidates.append(repaired)
    unwrapped_shots: list[Any] | None = None
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
        if (
            isinstance(data, dict)
            and isinstance(data.get("shots"), list)
            and unwrapped_shots is None
        ):
            unwrapped_shots = data["shots"]
    # Last resorts, in order: a reply cut off before the closing ] (max
    # token / mid-stream cutoff), then a board corrupted inside one
    # element (unescaped quote). Both recover the complete entries.
    for base in bases:
        salvaged = _salvage_truncated_array(base)
        if salvaged is not None:
            return salvaged
    for base in bases:
        salvaged = _salvage_elementwise(base)
        if salvaged is not None:
            return salvaged
    if unwrapped_shots is not None:
        return unwrapped_shots
    raise ValueError("no JSON array found in storyboard reply")


# --------------------------------------------------------------------------- #
# Normalization / validation
# --------------------------------------------------------------------------- #
_ID_SAFE_RE = re.compile(r"[^0-9a-zA-Z_]+")
_KNOWN_KEYS = (
    "id",
    "description",
    "shot_type",
    "camera_movement",
    "transition_in",
    "duration_seconds",
    "narrative_beat",
    "characters",
    "props",
    "notes",
)


def _slugify_id(raw: Any, index: int) -> str:
    text = _ID_SAFE_RE.sub("_", str(raw or "").strip().lower()).strip("_")
    text = re.sub(r"_+", "_", text)[:96]
    return text or f"scene_{index:02d}"


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _norm_duration(value: Any) -> tuple[int, str | None]:
    try:
        dur = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT_DURATION_SECONDS, "missing/invalid duration_seconds; used default"
    if dur < MIN_DURATION_SECONDS or dur > MAX_DURATION_SECONDS:
        return max(MIN_DURATION_SECONDS, min(MAX_DURATION_SECONDS, dur)), (
            f"duration_seconds {value} clamped to {max(MIN_DURATION_SECONDS, min(MAX_DURATION_SECONDS, dur))}"
        )
    return dur, None


def normalize_shots(
    raw_shots: list[Any],
    expected_count: int,
) -> tuple[list[dict], list[str]]:
    """Coerce raw LLM JSON elements into canonical shot dicts.

    Returns ``(shots, warnings)``. Shot count reconciliation: more than
    ``expected_count`` -> trimmed (the board was asked for N beats);
    fewer -> kept as-is. Both cases warn. Raises ``ValueError`` when a
    shot has no usable description at all (caller's retry trigger) or
    the list is empty.
    """
    warnings: list[str] = []
    shots: list[dict] = []

    for i, raw in enumerate(raw_shots, start=1):
        if isinstance(raw, str):
            raw = {"description": raw}
        if not isinstance(raw, dict):
            warnings.append(f"shot {i}: not an object; skipped")
            continue
        # Accept "prompt" as an alias for description.
        description = str(raw.get("description") or raw.get("prompt") or "").strip()
        if not description:
            raise ValueError(f"shot {i}: empty description")
        shot_type = str(raw.get("shot_type") or "").strip()
        if shot_type not in SHOT_TYPES:
            fallback = DEFAULT_SHOT_TYPE
            warnings.append(
                f"shot {i}: unknown shot_type {shot_type!r}; used {fallback}"
            )
            shot_type = fallback
        transition = str(raw.get("transition_in") or "").strip()
        if transition not in TRANSITIONS:
            fallback = (
                DEFAULT_FIRST_TRANSITION if len(shots) == 0 else DEFAULT_TRANSITION
            )
            warnings.append(
                f"shot {i}: unknown transition_in {transition!r}; used {fallback}"
            )
            transition = fallback
        duration, warn = _norm_duration(raw.get("duration_seconds"))
        if warn:
            warnings.append(f"shot {i}: {warn}")
        camera = _ID_SAFE_RE.sub("_", str(raw.get("camera_movement") or "").strip().lower()).strip("_")
        shots.append(
            {
                "id": _slugify_id(raw.get("id"), len(shots) + 1),
                "description": description,
                "shot_type": shot_type,
                "camera_movement": camera or "static_lockoff",
                "transition_in": transition,
                "duration_seconds": duration,
                "narrative_beat": str(raw.get("narrative_beat") or "").strip(),
                "characters": _coerce_str_list(raw.get("characters")),
                "props": _coerce_str_list(raw.get("props")),
                "notes": str(raw.get("notes") or "").strip(),
            }
        )

    if not shots:
        raise ValueError("storyboard reply contained no usable shots")

    # Unique ids: suffix duplicates with _2, _3, ...
    seen: dict[str, int] = {}
    for shot in shots:
        base = shot["id"]
        if base in seen:
            seen[base] += 1
            shot["id"] = f"{base}_{seen[base]}"
            warnings.append(f"duplicate id {base!r} renamed to {shot['id']!r}")
        else:
            seen[base] = 1

    expected = int(expected_count)
    if expected > 0:
        if len(shots) > expected:
            warnings.append(
                f"LLM returned {len(shots)} shots; trimmed to requested {expected}"
            )
            shots = shots[:expected]
        elif len(shots) < expected:
            warnings.append(
                f"LLM returned {len(shots)} shots; requested {expected} — kept actual count"
            )
    for w in warnings:
        mie_log(f"H3SB normalize: {w}")
    return shots, warnings


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_storyboard_markdown(
    shots: list[dict],
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    warnings: list[str] | None = None,
) -> str:
    """Render the human-readable ``storyboard_text`` overview."""
    code = parse_output_format(output_format)
    if code not in OUTPUT_FORMAT_CODES:
        code = DEFAULT_OUTPUT_FORMAT
    lines: list[str] = [f"# Storyboard — {len(shots)} shots"]
    if code == "table":
        lines.append("")
        lines.append("| # | ID | Beat | Shot | Camera | In | Dur(s) | Description |")
        lines.append("|---|----|------|------|--------|----|--------|-------------|")
        for i, s in enumerate(shots, start=1):
            lines.append(
                "| {n} | `{id}` | {beat} | {shot} | {cam} | {trans} | {dur} | {desc} |".format(
                    n=i,
                    id=_md_escape(s["id"]),
                    beat=_md_escape(s.get("narrative_beat") or "-"),
                    shot=_md_escape(s["shot_type"]),
                    cam=_md_escape(s.get("camera_movement") or "-"),
                    trans=_md_escape(s["transition_in"]),
                    dur=s["duration_seconds"],
                    desc=_md_escape(s["description"]),
                )
            )
    elif code == "detailed":
        for i, s in enumerate(shots, start=1):
            lines.append("")
            lines.append(
                "### {n}. `{id}` — {beat} ({dur}s)".format(
                    n=i,
                    id=s["id"],
                    beat=s.get("narrative_beat") or "beat",
                    dur=s["duration_seconds"],
                )
            )
            lines.append(f"**Description:** {s['description']}")
            meta = (
                f"- Shot: {s['shot_type']} | Camera: "
                f"{s.get('camera_movement') or '-'} | In: {s['transition_in']}"
            )
            lines.append(meta)
            if s.get("characters"):
                lines.append(f"- Characters: {', '.join(s['characters'])}")
            if s.get("props"):
                lines.append(f"- Props: {', '.join(s['props'])}")
            if s.get("notes"):
                lines.append(f"- Notes: {s['notes']}")
    else:  # minimal
        lines.append("")
        for i, s in enumerate(shots, start=1):
            lines.append(f"{i}. `{s['id']}` — {s['description']}")
    if warnings:
        lines.append("")
        lines.append("**Warnings:**")
        for w in warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


def shots_to_json_string(shots: list[dict]) -> str:
    """Serialize normalized shots for the ``shots_json`` output (stays
    JSON-parseable by ``MiniMaxH3LoopPromptGenerator.parse_shots_text``)."""
    return json.dumps(shots, ensure_ascii=False, indent=2)
