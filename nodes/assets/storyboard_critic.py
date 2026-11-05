"""Cross-shot continuity critic for portable storyboard assets.

The asset contract validates ONE asset in isolation: field shapes, event
fidelity, the visibility replay, the duration budget. It cannot see that
Character 1 wears a red coat in scene_01 and a blue coat in scene_03,
that the single apple ends up in two pairs of hands, or that a night
light preset and a daylight direction describe the same film.

This module answers those cross-shot questions mechanically — no model,
no cost, no network — and reports them as issues. The model half of the
critic lives in ``nodes/llm/storyboard_critic.py``.

Issue shape::

    {"path": str, "kind": str, "severity": "hard"|"soft",
     "excerpt": str, "reason": str}

``hard`` means the copy contradicts ITSELF: two halves of the same
instruction cannot both be true on screen, so the renderer has to pick
one and the other half becomes a defect. ``soft`` means a drift worth a
human glance (an appearance that changed between shots, a prop whose
holder changed without a handover).

The execution copy is English-only (the contract machine-checks that),
so the lexicons below are English-only on purpose: a Chinese direction
never reaches the compiler, and matching it here would only add noise.
"""
from __future__ import annotations

import re

HARD = "hard"
SOFT = "soft"

# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------
_COLORS = (
    ("red", r"\b(?:red|crimson|scarlet|maroon)\b"),
    ("blue", r"\b(?:blue|navy|azure|indigo)\b"),
    ("green", r"\b(?:green|olive|emerald)\b"),
    ("black", r"\bblack\b"),
    ("white", r"\bwhite\b"),
    ("yellow", r"\b(?:yellow|amber|golden)\b"),
    ("orange", r"\borange\b"),
    ("purple", r"\b(?:purple|violet|lavender|magenta)\b"),
    ("pink", r"\bpink\b"),
    ("gray", r"\b(?:grey|gray|silver)\b"),
    ("brown", r"\b(?:brown|tan|beige)\b"),
)

_GARMENTS = (
    ("coat", r"\b(?:coat|jacket|overcoat|trench coat)\b"),
    ("shirt", r"\b(?:shirt|blouse|t-shirt|tee)\b"),
    ("dress", r"\bdress\b"),
    ("hat", r"\b(?:hat|cap|beanie|helmet)\b"),
    ("scarf", r"\bscarf\b"),
    ("skirt", r"\bskirt\b"),
    ("trousers", r"\b(?:trousers|pants|jeans)\b"),
    ("uniform", r"\b(?:uniform|apron|robe)\b"),
    ("sweater", r"\b(?:sweater|jumper|cardigan)\b"),
    ("shoes", r"\b(?:shoes|boots|sneakers)\b"),
)

# Words that can follow a possession verb but never ARE the prop.
_NOT_A_PROP = {
    "hands", "hand", "air", "lap", "pocket", "place", "position", "reach",
    "mouth", "gaze", "face", "arms", "arm", "chest", "side", "front",
    "back", "one", "single", "other", "same", "own", "own", "her", "his",
    "their", "its", "the", "a", "an", "both", "two", "free", "empty",
}

# "the single apple", "the dark sedan" -> head noun + single marker.
# The phrase stops at a clause word, so "holds the single apple and turns"
# captures "single apple" and not "single apple and turns".
_PROP_STOP_WORDS = ("and|while|as|at|in|with|to|from|then|before|after|that|"
                     "which|when|because|but|so|or|still|yet")
_PROP_WORD = r"(?:(?!(?:%s)\b)[a-z][a-z\-]*)" % _PROP_STOP_WORDS
_PROP_PHRASE = r"((?:%s)(?:\s+(?:%s)){0,2})" % (_PROP_WORD, _PROP_WORD)

_POSSESSION_PATTERNS = (
    # holds / carries / grips / clutches / keeps + determiner + noun phrase
    re.compile(r"\b(?:holds?|holding|carries|carrying|grips|gripping|clutches|"
               r"clutching|keeps|keeping|retains|retaining)\s+"
               r"(?:the|a|an|one|single|his|her|their)\s+"
               + _PROP_PHRASE, re.I),
    # the X in his/her hands
    re.compile(r"\b(?:the|a|an)\s+" + _PROP_PHRASE + r"\s+in\s+(?:his|her|their)\s+hands\b", re.I),
    # his/her hands hold the X
    re.compile(r"\b(?:his|her|their)\s+hands\s+(?:hold|carry|keep|clutch|hold\s+on\s+to)\s+"
               r"(?:the|a|an|one|single)\s+" + _PROP_PHRASE, re.I),
)

# A COMPLETED handover. "offers"/"extends" are NOT transfers: the asset
# contract keeps an offered object with the offering actor until an
# explicit accepted handover, so an offer must not move the prop.
_TRANSFER_PATTERNS = (
    re.compile(r"\b(?:hands?|gives?|passes?)\s+(?:the|a|an|one|single|it|them)?\s*"
               + _PROP_PHRASE + r"\s+to\s+Character\s+(\d+)", re.I),
    re.compile(r"\b(?:hands?|gives?|passes?)\s+(?:it|them)\s+to\s+Character\s+(\d+)", re.I),
)

_RELEASE_PATTERNS = (
    re.compile(r"\b(?:puts?|places?|sets?|lays?|drops?|releases?|returns?)\s+"
               r"(?:the|a|an|one|single)\s+" + _PROP_PHRASE + r"\s+"
               r"(?:down|aside|back|on\s+the\s+\w+)", re.I),
    re.compile(r"\blets?\s+go\s+of\s+(?:the|a|an)\s+" + _PROP_PHRASE, re.I),
)

_EMPTY_HANDS = re.compile(
    r"\b(?:empty\s+(?:open\s+)?(?:hands?|palms?)|hands?\s+(?:are\s+|remain\s+|stay\s+)?"
    r"(?:visibly\s+)?empty|palms?\s+(?:are\s+)?empty|empty-handed)\b", re.I)

_CHARACTER = re.compile(r"\bCharacter\s+(\d+)\b")

_SINGLE_MARKER = re.compile(r"\b(?:single|only\s+one|the\s+only|sole|one)\b", re.I)

_DAYLIGHT_WORDS = re.compile(
    r"\b(?:daylight|sunlight|sunlit|bright\s+sun|midday|noon|morning\s+sun|"
    r"golden\s+sunlight)\b", re.I)
_NIGHT_WORDS = re.compile(
    r"\b(?:night|nighttime|lamplight|lamp\s+light|candlelight|moonlight|"
    r"darkness|after\s+dark)\b", re.I)

_NIGHT_PRESETS = ("night_lamp", "night_screen", "late_night")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _head_noun(phrase):
    """Last meaningful word of a captured noun phrase."""
    words = [w for w in re.split(r"\s+", (phrase or "").strip().lower()) if w]
    while words and words[-1] in _NOT_A_PROP:
        words.pop()
    if not words:
        return None
    head = words[-1]
    if head in _NOT_A_PROP or len(head) < 3:
        return None
    return head


def _mentions_character(text, number):
    return bool(re.search(r"\bCharacter\s+%d\b" % number, text or "", re.I))


def _color_garments(text):
    """Every (garment, colour, position) stated in one piece of copy."""
    text = text or ""
    pairs = []
    for garment, g_pattern in _GARMENTS:
        for match in re.finditer(g_pattern, text, re.I):
            window = text[max(0, match.start() - 60):match.end() + 24]
            best = None
            for color, c_pattern in _COLORS:
                found = None
                for cm in re.finditer(c_pattern, window, re.I):
                    found = cm
                if found is None:
                    continue
                if best is None or found.start() > best[1]:
                    best = (color, found.start())
            if best:
                pairs.append((garment, best[0], match.start()))
    return pairs


def _nearest_character_before(text, index):
    """The Character N mentioned closest BEFORE ``index`` (None if none)."""
    last = None
    for match in _CHARACTER.finditer(text or ""):
        if match.start() <= index:
            last = int(match.group(1))
        else:
            break
    return last


def _presence_by_shot(asset):
    """[(visible_at_start, visible_at_end)] per shot."""
    roles = {r["id"]: r for r in asset.get("roles", ())}
    visible = {int(r["id"].rsplit("_", 1)[1])
               for r in roles.values() if r.get("initially_visible")}
    out = []
    for shot in asset.get("shots", ()):
        start = set(visible)
        for event in shot.get("events", ()):
            number = int(str(event.get("character", "character_0")).rsplit("_", 1)[1])
            kind = event.get("type")
            if kind == "enter":
                visible.add(number)
            elif kind == "exit":
                visible.discard(number)
        out.append((start, set(visible)))
    return out


def _character_segments(text):
    """[(number, segment)] - the copy that belongs to each character.

    "Character 1 keeps both hands visibly empty ... Character 2 retains
    the single apple" is TWO statements, not one: the empty hands belong
    to Character 1 and the apple to Character 2. Every prop check reads
    the segment, so the contract's own copy (the refusing actor's empty
    hands while the other actor holds the object) is never a conflict.
    """
    text = text or ""
    mentions = [(m.start(), int(m.group(1))) for m in _CHARACTER.finditer(text)]
    out = []
    for index, (start, number) in enumerate(mentions):
        end = mentions[index + 1][0] if index + 1 < len(mentions) else len(text)
        out.append((number, text[start:end]))
    return out


def _held_props(segment):
    """{prop: phrase} - the props one character's copy says they hold."""
    found = {}
    for pattern in _POSSESSION_PATTERNS:
        for match in pattern.finditer(segment or ""):
            head = _head_noun(match.group(1))
            if head:
                found.setdefault(head, match.group(0).strip())
    return found


def _transfers_in(text):
    """[(prop_head, to_character)] for COMPLETED handovers."""
    out = []
    for pattern in _TRANSFER_PATTERNS:
        for match in pattern.finditer(text or ""):
            head = _head_noun(match.group(1)) if match.lastindex and match.lastindex >= 1 else None
            out.append((head, int(match.group(match.lastindex))))
    return out


def _releases_in(text):
    out = set()
    for pattern in _RELEASE_PATTERNS:
        for match in pattern.finditer(text or ""):
            head = _head_noun(match.group(1))
            if head:
                out.add(head)
    return out


def _issue(path, kind, severity, excerpt, reason):
    return {"path": path, "kind": kind, "severity": severity,
            "excerpt": (excerpt or "")[:160], "reason": reason}


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------
def _check_appearance(asset):
    """One garment must not change colour between shots."""
    execution = asset.get("execution") or {}
    seen = {}
    designs = execution.get("character_designs") or {}
    for role_id, description in designs.items():
        number = int(str(role_id).rsplit("_", 1)[1])
        for garment, color, _position in _color_garments(description):
            seen.setdefault((number, garment), {}).setdefault(color, []).append(
                "execution.character_designs." + role_id)
    for shot in asset.get("shots", ()):
        for event in shot.get("events", ()):
            direction = (execution.get("directions") or {}).get(event["id"])
            if not direction:
                continue
            # A direction describes its ACTOR (the contract requires it to
            # start with that Character N), so the garments it mentions
            # belong to the actor - not to whoever was named last.
            number = int(str(event.get("character", "character_0")).rsplit("_", 1)[1])
            for garment, color, _position in _color_garments(direction):
                seen.setdefault((number, garment), {}).setdefault(color, []).append(
                    "execution.directions." + event["id"])
    issues = []
    for (number, garment), colors in sorted(seen.items()):
        if len(colors) < 2:
            continue
        listed = sorted(colors)
        issues.append(_issue(
            "execution.directions", "appearance_conflict", SOFT,
            "Character %d's %s: %s" % (number, garment, ", ".join(listed)),
            "Character %d wears the %s in %s across the asset — one garment keeps one colour "
            "for the whole story, or the render drifts." % (number, garment, " and ".join(listed))))
    return issues


def _check_props(asset):
    """Props: one holder at a time, empty hands stay empty."""
    execution = asset.get("execution") or {}
    directions = execution.get("directions") or {}
    issues = []
    holder_state = {}
    for shot in asset.get("shots", ()):
        shot_props = {}
        for event in shot.get("events", ()):
            direction = directions.get(event["id"])
            if not direction:
                continue
            path = "execution.directions." + event["id"]
            for number, segment in _character_segments(direction):
                held = _held_props(segment)
                for prop, phrase in held.items():
                    shot_props.setdefault(prop, {})[number] = phrase
                if held and _EMPTY_HANDS.search(segment):
                    issues.append(_issue(
                        path, "prop_holds_and_empty_hands", HARD, direction,
                        "Character %d holds %s and is also described with empty hands in the "
                        "same beat — one of the two instructions cannot be drawn."
                        % (number, phrase)))
                for prop in held:
                    if prop not in holder_state:
                        holder_state[prop] = number
                    elif holder_state[prop] != number:
                        issues.append(_issue(
                            path, "prop_holder_conflict", SOFT, direction,
                            "Character %d holds %s, but the previous state of that prop was "
                            "Character %d and no handover is written."
                            % (number, prop, holder_state[prop])))
                        holder_state[prop] = number
            for prop, to_number in _transfers_in(direction):
                holder_state[prop] = to_number
            for prop in _releases_in(direction):
                holder_state.pop(prop, None)
        for prop, owners in shot_props.items():
            if len(owners) < 2:
                continue
            numbers = sorted(owners)
            single = any(_SINGLE_MARKER.search(phrase) for phrase in owners.values())
            issues.append(_issue(
                "execution.directions", "prop_double_holder",
                HARD if single else SOFT,
                "%s held by Character %s" % (prop, ", ".join(str(n) for n in numbers)),
                "The %s is in two pairs of hands in the same segment%s." %
                (prop, " and the copy calls it the single one" if single else "")))
    return issues


def _check_presence(asset):
    """A direction must not put an absent character on stage."""
    execution = asset.get("execution") or {}
    directions = execution.get("directions") or {}
    presence = _presence_by_shot(asset)
    issues = []
    for shot, (start, end) in zip(asset.get("shots", ()), presence):
        on_stage = start | end
        for event in shot.get("events", ()):
            direction = directions.get(event["id"])
            if not direction:
                continue
            for number in {int(m.group(1)) for m in _CHARACTER.finditer(direction)}:
                if number in on_stage:
                    continue
                issues.append(_issue(
                    "execution.directions." + event["id"], "character_absent", SOFT, direction,
                    "Character %d is not on stage in %s, but the direction puts them in the shot."
                    % (number, shot["id"])))
    return issues


def _check_light(asset):
    """The light preset and the copy must describe the same hour."""
    execution = asset.get("execution") or {}
    light = asset.get("light", "follow_image")
    if light == "follow_image":
        return []
    copy = " ".join([execution.get("visible_state", "") or ""]
                     + list((execution.get("directions") or {}).values()))
    if not copy.strip():
        return []
    issues = []
    if light in _NIGHT_PRESETS and _DAYLIGHT_WORDS.search(copy):
        issues.append(_issue(
            "execution.directions", "light_conflict", SOFT, copy[:120],
            "The light preset is %s but the performance copy describes daylight." % light))
    elif light in ("daylight", "morning") and _NIGHT_WORDS.search(copy):
        issues.append(_issue(
            "execution.directions", "light_conflict", SOFT, copy[:120],
            "The light preset is %s but the performance copy describes night." % light))
    return issues


def _check_beats(asset):
    """One segment carries one beat; more actions need an internal cut."""
    issues = []
    for shot in asset.get("shots", ()):
        actions = [e for e in shot.get("events", ()) if e.get("type") == "action"]
        if len(actions) >= 3 and not shot.get("camera_cuts"):
            issues.append(_issue(
                shot["id"], "multi_action_segment", SOFT, shot["id"],
                "%s stacks %d actions in one segment without an internal cut — one segment "
                "renders one beat." % (shot["id"], len(actions))))
    return issues


def _check_opening_state(asset):
    """visible_state describes the first frame: nobody who is not there yet."""
    execution = asset.get("execution") or {}
    visible_state = execution.get("visible_state") or ""
    if not visible_state:
        return []
    presence = _presence_by_shot(asset)
    if not presence:
        return []
    start, end = presence[0]
    on_stage = start | end
    issues = []
    for number in {int(m.group(1)) for m in _CHARACTER.finditer(visible_state)}:
        if number in on_stage:
            continue
        issues.append(_issue(
            "execution.visible_state", "opening_state_absent", SOFT, visible_state,
            "The opening state names Character %d, who is not on stage in the first segment."
            % number))
    return issues


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def review_asset_critique(asset):
    """Cross-shot issues for a validated v2 asset.

    Never edits anything; returns a list of issue dicts (empty when the
    asset is self-consistent).
    """
    if not isinstance(asset, dict) or asset.get("version") != 2:
        return []
    issues = []
    for check in (_check_appearance, _check_props, _check_presence,
                  _check_light, _check_beats, _check_opening_state):
        try:
            issues.extend(check(asset))
        except Exception:  # a critic must never break a generation
            continue
    order = {HARD: 0, SOFT: 1}
    issues.sort(key=lambda i: (order.get(i["severity"], 2), i["path"], i["kind"]))
    return issues


def review_asset_critique_hard(asset):
    """Only the self-contradicting issues — the ones worth a re-roll."""
    return [i for i in review_asset_critique(asset) if i["severity"] == HARD]


def review_shot_list_critique(shots):
    """Cross-shot issues for the legacy shot list (V1/V2 auto-storyboard).

    The legacy board has no execution copy, so the checks are limited to
    what the board itself states: a recurring prop or character that
    changes cast without a transition note.
    """
    issues = []
    if not isinstance(shots, list):
        return issues
    prop_owners = {}
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        characters = {str(c).strip() for c in (shot.get("characters") or []) if str(c).strip()}
        for prop in (shot.get("props") or []):
            key = str(prop).strip().lower()
            if not key:
                continue
            previous = prop_owners.get(key)
            if previous is not None and previous != characters:
                issues.append(_issue(
                    shot.get("id") or "shot_%d" % (index + 1), "prop_cast_conflict", SOFT, key,
                    "The prop %r sits with a different cast here (%s) than earlier (%s)."
                    % (key, ", ".join(sorted(characters)) or "-", ", ".join(sorted(previous)) or "-")))
            prop_owners.setdefault(key, characters)
    return issues


def critique_warnings(issues):
    """Render issues as the human-readable warning lines the nodes print."""
    out = []
    for issue in issues:
        mark = "⛔" if issue.get("severity") == HARD else "⚠"
        out.append("%s %s：%s" % (mark, issue.get("reason", ""), issue.get("excerpt", "")))
    return out
