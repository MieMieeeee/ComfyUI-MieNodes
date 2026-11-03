"""Portable storyboard v2: locked role events and deterministic H3 compilation."""
from __future__ import annotations

import copy
import math
import re

from .library import validate_name

ROLE_MAX = 8
FRAMINGS = ("two_shot", "medium_shot", "medium_close_up", "close_up", "wide_shot")
MOVEMENTS = ("locked", "gentle_follow")
PERSPECTIVES = ("observer", "first_person")
EVENT_TYPES = ("dialogue", "action", "enter", "exit")
LANGUAGES = ("zh", "en")

# Time-of-day light anchors (2026-10-29 ruling): light is its OWN selector
# field, not prose inside ``setting``. A preset compiles to this fixed
# English sentence in every shot prompt — no LLM in the loop. ``follow_image``
# (the default) writes nothing: the wired scene image's own lighting stands,
# and an UNWIRED story at night must NOT stay on follow_image (H3 renders
# unspecified time as daylight). ``custom`` requires ``light_text`` (user
# language), which the generation LLM translates to ``execution.light``.
LIGHT_CHOICES = ("follow_image", "daylight", "golden_hour", "night_lamp",
                 "night_screen", "late_night", "morning", "custom")
LIGHT_PROMPTS = {
    "daylight": "Soft natural daylight fills the scene; the time is broad day.",
    "golden_hour": "Warm low golden-hour sunlight rakes across the scene, long soft shadows.",
    "night_lamp": "Night: the window shows a pitch-dark sky and the scene is lit by a warm table lamp.",
    "night_screen": "Night: the window shows a pitch-dark sky and the scene is lit by the cool glow of a monitor screen.",
    "late_night": "Pitch-dark late night: only faint scattered light remains and the window shows a black sky.",
    "morning": "Soft early-morning light, low and gently cool, enters the scene.",
}

# The soundscape line ``to_runtime`` stamps into every compiled shot when
# nothing better exists. ``storyboard_builder`` swaps it for the WIRED
# scene asset's ``ambience`` (the v2 asset itself has no ambience field —
# sound belongs to the environment asset, not the directing contract).
RUNTIME_SOUNDSCAPE = "Natural room tone, breathing and synchronized movement sounds."
# Role voice keeps the SAME cap as the character asset's voice line
# (library.VOICE_MAX_CHARS): the two fields describe one timbre, and the
# system prompt tells the author to write them to match each other.
ROLE_VOICE_MAX_CHARS = 200


def _bare_phrase(value):
    """Drop trailing sentence punctuation from a spliced phrase.

    ``voice`` is spliced into the H3 prompt as ``<voice>: <d>…``; a
    trailing period therefore produces a broken line ("…pace.: <d>…").
    Normalizing here keeps the fix mechanical instead of asking every
    author to remember the rule.
    """
    return value.rstrip(" .。、,，;；!！?？").strip()


def _text(value, label, *, empty=False, maximum=12000):
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value) > maximum:
        raise ValueError("%s must be %sa string of at most %d characters" %
                         (label, "" if empty else "a non-empty ", maximum))
    if "<d>" in value or "</d>" in value:
        raise ValueError("%s must not contain H3 dialogue tags" % label)
    return value.strip()


def parse_script(script, roles, language="zh"):
    """Read explicit role-labelled events. The LLM never owns these records.

    Humans, not agents, type these lines: common speech verbs before the
    colon (说/问/喊/回答…) are accepted as dialogue labels, and the error
    message TEACHES (echoes the line, shows the format, points at the
    concept mode) instead of just refusing.
    """
    script = _text(script, "source_script", maximum=30000)
    aliases = {}
    for i, role in enumerate(roles, 1):
        for alias in (role["id"], "角色%d" % i, "Character %d" % i, role["name"]):
            alias = alias.casefold()
            if alias in aliases and aliases[alias] != role["id"]:
                raise ValueError("role names/aliases must be unique: %s" % alias)
            aliases[alias] = role["id"]
    names = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    speech_verbs = ("说道", "问道", "喊道", "答道", "回答", "笑道", "怒道",
                    "轻声说", "低声说", "说", "问", "喊", "叫", "道", "答",
                    "嘀咕", "吐槽", "开口")
    types = "动作|对白|入场|离场|action|dialogue|enter|exit|" + "|".join(speech_verbs)
    pattern = re.compile(r"^\s*(" + names + r")\s*(" + types + r")?"
                         r"\s*(?:\[(zh|en)\])?\s*[：:]\s*(.+)$", re.IGNORECASE)
    events = []
    kind_of = {"动作": "action", "对白": "dialogue", "入场": "enter", "离场": "exit",
               **{verb: "dialogue" for verb in speech_verbs}}
    for number, line in enumerate(script.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(
                "source_script 第 %d 行无法识别 · line %d does not match: 「%s」\n"
                "每行必须以角色开头 · every event line starts with a role label:\n"
                "  台词 角色1：你好    动作 角色1动作：他站起来    入场 角色1入场：…    离场 角色1离场：…\n"
                "旁白：先在 character_settings 加一行「角色1：旁白 | 画外 | 无」再以 角色1： 开头写；"
                "场景描述放 scene，导演要求放 constraints。只想给个点子？把输入模式 mode 切到 concept 概念创作，"
                "写什么都行 · just an idea? switch mode to concept and write freely"
                % (number, number, line.strip()[:60]))
        alias, kind, lang, text = match.groups()
        kind = kind_of.get((kind or "对白"), (kind or "对白").lower())
        if kind not in EVENT_TYPES:
            kind = "dialogue"
        events.append({"id": "event_%03d" % (len(events) + 1),
                       "type": kind, "character": aliases[alias.casefold()],
                       "text": _text(text, "source event text", maximum=2000),
                       "language": lang or language})
    if not events or len(events) > 200:
        raise ValueError("source_script needs 1..200 explicit role events")
    return events


def parse_roles(count, settings=""):
    if type(count) is not int or not 1 <= count <= ROLE_MAX:
        raise ValueError("character_count must be an integer in 1..8")
    roles = [{"id": "character_%d" % i, "name": "角色%d" % i,
              "presence": "on_screen", "initially_visible": True,
              "position": "left" if count == 2 and i == 1 else
                          "right" if count == 2 else "center", "voice": ""}
             for i in range(1, count + 1)]
    if not settings.strip():
        return roles
    seen = set()
    for number, line in enumerate(settings.splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(r"^\s*(?:角色|character_?\s*)(\d+)\s*[：:]\s*(.+)$", line, re.I)
        if not match:
            raise ValueError(
                "character_settings 第 %d 行无法识别 · line %d does not match: 「%s」\n"
                "每角色一行（分隔符用 | 或逗号都可以）· one line per role, | or commas both work:\n"
                "  角色1：小明 | 画面内 | 左侧 | 低沉男声\n"
                "状态：画面内/画外/稍后入场；位置：左侧/右侧/中间/无；概念创作模式可整栏留空由 AI 定"
                % (number, number, line.strip()[:60]))
        index = int(match[1])
        # Commas are accepted as separators for the first three columns
        # (humans forget the pipes); maxsplit keeps the VOICE column whole —
        # voice descriptors legitimately contain commas ("adult male, low
        # mid-range pitch").
        fields = [x.strip() for x in re.split(r"\s*[|，,、]\s*", match[2], maxsplit=3) if x.strip()]
        if index not in range(1, count + 1):
            raise ValueError(
                "character_settings 第 %d 行的角色编号 角色%d 超出 character_count=%d · "
                "line %d: 角色%d is outside character_count=%d\n"
                "把 character_count 改成 %d，或删掉这一行：「%s」"
                % (number, index, count, number, index, count,
                   max(count, index), line.strip()[:60]))
        if index in seen or not 1 <= len(fields) <= 4:
            raise ValueError(
                "character_settings 第 %d 行有重复编号或字段数超限（1..4 个）· line %d has a "
                "duplicate role or too many fields: 「%s」" % (number, number, line.strip()[:60]))
        seen.add(index)
        role = roles[index - 1]
        role["name"] = _text(fields[0], "role name", maximum=40)
        if len(fields) > 1:
            status = fields[1]
            if status not in ("画面内", "画外", "稍后入场", "on_screen", "off_screen", "enter_later"):
                raise ValueError(
                    "character_settings 第 %d 行的角色状态无法识别 · line %d unknown presence "
                    "「%s」：可用 画面内 / 画外 / 稍后入场 · use 画面内 / 画外 / 稍后入场"
                    % (number, number, status))
            role["presence"] = "off_screen" if status in ("画外", "off_screen") else "on_screen"
            role["initially_visible"] = status in ("画面内", "on_screen")
            if role["presence"] == "off_screen":
                role["position"] = "none"
        if len(fields) > 2:
            role["position"] = {"左侧": "left", "右侧": "right", "中间": "center",
                                "无": "none"}.get(fields[2], fields[2])
        if len(fields) > 3:
            role["voice"] = fields[3]
    if len(seen) != count:
        defined = sorted(seen)
        raise ValueError(
            "character_settings 定义了 %d 个角色（%s），但 character_count=%d · "
            "settings define %d role(s) (%s) while character_count=%d\n"
            "两边要一致：把 character_count 改成 %d，或补齐/删掉角色行 · make them agree"
            % (len(seen), "、".join("角色%d" % i for i in defined), count,
               len(seen), ", ".join(str(i) for i in defined), count, len(seen)))
    return roles


def validate_v2(payload, *, _check_timing=True, _check_execution=True):
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 2:
        raise ValueError("storyboard version must be 2")
    allowed = {"version", "name", "language", "roles", "camera", "look", "setting", "scene",
               "light", "light_text", "constraints", "target_duration_seconds",
               "continuation_overlap_frames", "source_script", "shots", "execution", "audio_mode",
               "character_designs"}
    if set(payload) - allowed:
        raise ValueError("unknown storyboard fields: %s" % sorted(set(payload) - allowed))
    out = copy.deepcopy(payload)
    if out.get("audio_mode", "generated") not in ("generated", "reference"):
        raise ValueError("audio_mode must be generated or reference")
    out["name"] = validate_name(out.get("name", ""))
    if out.get("language") not in LANGUAGES:
        raise ValueError("language must be zh or en")
    # Scene / light split (2026-10-29 ruling): ``scene`` is the optional
    # environment description the builder DROPS whenever a scene asset is
    # wired (the asset overrides the text mechanically); ``setting`` narrows
    # to performance staging + story-required spatial facts. Defaults never
    # belong in the unconditional staging/visible-state/direction copy.
    scene = _text(payload.get("scene") or "", "scene", empty=True, maximum=3000)
    if scene:
        out["scene"] = scene
    elif "scene" in out:
        del out["scene"]
    light = out.get("light", "follow_image")
    if light not in LIGHT_CHOICES:
        raise ValueError("light must be one of %s" % (LIGHT_CHOICES,))
    if light == "custom":
        out["light_text"] = _text(out.get("light_text", ""), "light_text", maximum=600)
    elif str(out.get("light_text") or "").strip():
        raise ValueError("light_text is only valid together with light=custom")
    if light != "follow_image" or "light" in payload:
        out["light"] = light
    elif "light" in out:
        del out["light"]
    roles = out.get("roles")
    if not isinstance(roles, list) or not 1 <= len(roles) <= ROLE_MAX:
        raise ValueError("roles must contain 1..8 characters")
    if out.get("audio_mode") == "reference" and len(roles) > 3:
        raise ValueError("reference audio supports at most 3 character roles")
    for i, role in enumerate(roles, 1):
        if not isinstance(role, dict) or role.get("id") != "character_%d" % i:
            raise ValueError("role character IDs must be contiguous character_1..N")
        if set(role) - {"id", "name", "presence", "initially_visible", "position", "voice"}:
            raise ValueError("unknown role fields")
        role["name"] = _text(role.get("name"), "role name", maximum=40)
        if role.get("presence") not in ("on_screen", "off_screen"):
            raise ValueError("role presence must be on_screen or off_screen")
        role.setdefault("initially_visible", role["presence"] == "on_screen")
        if type(role["initially_visible"]) is not bool:
            raise ValueError("role initially_visible must be boolean")
        if role["presence"] == "off_screen" and role["initially_visible"]:
            raise ValueError("off_screen role cannot be initially visible")
        role.setdefault("position", "none")
        if role["position"] not in ("left", "right", "center", "none"):
            raise ValueError("role position must be left/right/center/none")
        if role["presence"] == "off_screen" and role["position"] != "none":
            raise ValueError("off_screen role position must be none")
        role["voice"] = _bare_phrase(_text(role.get("voice", ""), "role voice", empty=True,
                                           maximum=ROLE_VOICE_MAX_CHARS))
    if not any(r["presence"] == "on_screen" for r in roles):
        raise ValueError("at least one on_screen role is required")
    if "character_designs" in out:
        designs = out["character_designs"]
        visible_ids = {r["id"] for r in roles if r["presence"] == "on_screen"}
        if not isinstance(designs, dict) or not set(designs) <= visible_ids:
            raise ValueError("character_designs must map declared on_screen role IDs to default appearances")
        for who, description in designs.items():
            designs[who] = _text(description, "character_designs." + who, maximum=3000)
    camera = out.get("camera")
    if not isinstance(camera, dict) or set(camera) != {"framing", "movement", "perspective"}:
        raise ValueError("camera requires framing, movement and perspective")
    if (camera["framing"] not in FRAMINGS or camera["movement"] not in MOVEMENTS
            or camera["perspective"] not in PERSPECTIVES):
        raise ValueError("invalid camera framing/movement/perspective")
    for key in ("look", "setting", "constraints"):
        out[key] = _text(out.get(key, ""), key, empty=True, maximum=3000)
    duration = out.get("target_duration_seconds")
    if type(duration) not in (float, int) or not math.isfinite(duration) or not 1 <= duration <= 600:
        raise ValueError("target_duration_seconds must be finite and in 1..600")
    overlap = out.get("continuation_overlap_frames", 22)
    if type(overlap) is not int or not 0 <= overlap <= 120:
        raise ValueError("continuation_overlap_frames must be an integer in 0..120")
    out["continuation_overlap_frames"] = overlap
    out["source_script"] = _text(out.get("source_script"), "source_script", maximum=30000)
    expected = parse_script(out["source_script"], roles, out["language"])
    shots = out.get("shots")
    if not isinstance(shots, list) or not 1 <= len(shots) <= min(20, len(expected)):
        raise ValueError("shots must contain 1..20 nonempty event groups")
    actual = []
    for i, shot in enumerate(shots, 1):
        if not isinstance(shot, dict) or set(shot) - {"id", "duration_seconds", "events", "camera_cuts"}:
            raise ValueError("shot only accepts id, duration_seconds, events and optional camera_cuts")
        if shot.get("id") != "scene_%02d" % i:
            raise ValueError("shot IDs must be contiguous scene_01..N")
        seconds = shot.get("duration_seconds")
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or not 0 < seconds <= 30:
            raise ValueError("shot duration_seconds must be finite and in (0,30]")
        events = shot.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("each shot requires nonempty events")
        for event in events:
            if (not isinstance(event, dict) or set(event) != {"id", "type", "character", "text", "language"}
                    or event.get("type") not in EVENT_TYPES or event.get("language") not in LANGUAGES):
                raise ValueError("invalid source event fields/type/language")
        cuts = shot.get("camera_cuts", [])
        if not isinstance(cuts, list) or len(cuts) > 19:
            raise ValueError("camera_cuts must be a list of at most 19 internal cuts")
        event_order = {event["id"]: index for index, event in enumerate(events)}
        previous_cut = 0
        for cut in cuts:
            if (not isinstance(cut, dict) or set(cut) != {"before_event", "framing"}
                    or cut.get("framing") not in FRAMINGS
                    or event_order.get(cut.get("before_event"), -1) <= previous_cut):
                raise ValueError("camera_cuts require valid framing and unique ordered events after the first event")
            previous_cut = event_order[cut["before_event"]]
        actual.extend(events)
    if actual != expected:
        # Name the FIRST mismatch so a retrying model (or a human) can fix
        # the exact spot instead of guessing (2026-10-30: Flash failed three
        # consecutive retries on the generic message).
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                raise ValueError(
                    "source events differ at #%d: source_script says (%s %s %r) but shots "
                    "carry (%s %s %r); text, actor, type, language and order must remain "
                    "verbatim" % (i + 1, e["type"], e["character"], e["text"][:40],
                                  a["type"], a["character"], a["text"][:40]))
        raise ValueError("source events differ: shots carry %d events, source_script has %d; "
                         "text, actor, type, language and order must remain verbatim"
                         % (len(actual), len(expected)))
    if _check_execution:
        _validate_execution(out, expected)
    _replay_visibility(out)
    if _check_timing:
        for shot, frames in zip(out["shots"], allocate_frames(out)):
            shot["duration_seconds"] = frames / 24
    return out


def _english(value, label, *, empty=False):
    value = _text(value, label, empty=empty, maximum=6000)
    if re.search(r"[\u3400-\u9fff]", value):
        raise ValueError(label + " must be English; Chinese belongs only in source records or dialogue")
    return value


def _positive_visual(value, label, *, empty=False):
    value = _english(value, label, empty=empty)
    if re.search(r"\b(?:no|not|never|without|cannot|can't|don't|doesn't|isn't|aren't|avoid|exclude|forbid)\b", value, re.I):
        raise ValueError(label + " must describe a positive physical state; keep exclusions in source constraints")
    return value


def _validate_execution(asset, events):
    execution = asset.get("execution")
    required = {"look", "setting", "visible_state", "voices", "directions"}
    allowed = required | {"scene", "light", "character_designs"}
    if not isinstance(execution, dict) or not required <= set(execution) <= allowed:
        raise ValueError("execution requires English look, setting, visible_state, voices and directions"
                         " (optional scene, light and character_designs)")
    for key in ("look", "setting"):
        execution[key] = _english(execution[key], "execution." + key, empty=not bool(asset[key]))
        if not asset[key] and execution[key]:
            raise ValueError("execution." + key + " cannot invent an absent source instruction")
    # execution.scene mirrors the source scene exactly like look/setting;
    # execution.light exists ONLY for light=custom (preset light is compiled
    # from the fixed English table, never stored). Empty values leave NO key
    # so an asset that never used the split round-trips byte-identically.
    raw_scene = execution.get("scene", "")
    raw_light = execution.get("light", "")
    execution["scene"] = _english(raw_scene, "execution.scene",
                                  empty=not bool(asset.get("scene", "")))
    if not asset.get("scene", "") and execution["scene"]:
        raise ValueError("execution.scene cannot invent an absent source scene")
    custom_light = asset.get("light", "follow_image") == "custom"
    execution["light"] = _english(raw_light, "execution.light", empty=not custom_light)
    if not custom_light and execution["light"]:
        raise ValueError("execution.light is only valid together with light=custom")
    if not execution["scene"]:
        execution.pop("scene", None)
    if not execution["light"]:
        execution.pop("light", None)
    designs = asset.get("character_designs", {})
    translations = execution.get("character_designs", {})
    if not isinstance(translations, dict) or set(translations) != set(designs):
        raise ValueError("execution.character_designs must match source character_designs exactly")
    for who, description in translations.items():
        translations[who] = _english(description, "execution.character_designs." + who)
    execution["visible_state"] = _positive_visual(execution["visible_state"], "execution.visible_state")
    voices = execution["voices"]
    if not isinstance(voices, dict) or set(voices) != {r["id"] for r in asset["roles"]}:
        raise ValueError("execution.voices must map every numbered role exactly once")
    for role in asset["roles"]:
        voices[role["id"]] = _bare_phrase(
            _english(voices[role["id"]], "execution voice", empty=not bool(role["voice"])))
    directions = execution["directions"]
    required = {event["id"] for event in events if event["type"] != "dialogue"}
    if not isinstance(directions, dict) or set(directions) != required:
        raise ValueError("execution.directions must map every action/enter/exit event ID exactly once")
    for event_id in directions:
        directions[event_id] = _positive_visual(directions[event_id], "execution direction " + event_id)
        event = next(event for event in events if event["id"] == event_id)
        actor = "Character " + event["character"].rsplit("_", 1)[1]
        if not re.match(re.escape(actor) + r"\b", directions[event_id]):
            raise ValueError("execution direction %s must start with its source actor %s" % (event_id, actor))


def _replay_visibility(asset):
    roles = {r["id"]: r for r in asset["roles"]}
    visible = {r["id"] for r in roles.values() if r["initially_visible"]}
    openings = []
    for shot in asset["shots"]:
        openings.append(set(visible))
        for event in shot["events"]:
            who, kind = event["character"], event["type"]
            if kind == "enter":
                if who in visible or roles[who]["presence"] == "off_screen":
                    raise ValueError("source enter event requires an absent on_screen role")
                visible.add(who)
            elif kind == "exit":
                if who not in visible:
                    raise ValueError("source exit event requires a visible role")
                visible.remove(who)
            elif kind == "action" and who not in visible:
                raise ValueError("source action requires a visible actor; use 入场 before acting")
    return openings


def estimate_shot_seconds(shot, rate=8.0):
    """Content seconds the duration estimator demands for one shot.

    ``rate`` is CJK characters per second. The default (8) is the
    feasibility floor the allocator uses; live renders measured H3
    delivering Chinese at 4.5-6.3 characters per second, so the
    conservative rate (5.5) is what a review must budget against.
    """
    seconds = 0.0
    for event in shot["events"]:
        if event["type"] == "dialogue":
            seconds += len(re.findall(r"[\u3400-\u9fff]", event["text"])) / rate
            seconds += len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", event["text"])) / 3 + 0.2
        else:
            seconds += 0.5
    return seconds


def allocate_frames(asset):
    """Budget delivered time once, then apportion exact H3-grid raw clips."""
    shots, overlap = asset["shots"], asset["continuation_overlap_frames"]
    count = len(shots)
    total = round((asset["target_duration_seconds"] * 24 + overlap * (count - 1) - 5 * count) / 17)
    minimum = []
    for i, shot in enumerate(shots):
        raw = max(1.0, estimate_shot_seconds(shot)) * 24 + (overlap if i else 0)
        minimum.append(max(0, math.ceil((raw - 5) / 17)))
    if any(k > 42 for k in minimum) or total > 42 * count:
        raise ValueError("duration/shot_count cannot fit H3's 30-second per-shot limit")
    if total < sum(minimum):
        raise ValueError("target duration is too short for the supplied source events; increase it")
    caps = [min(42, minimum[i] + 2)
            if len(shot["events"]) == 1 and shot["events"][0]["type"] == "dialogue"
            else 42
            for i, shot in enumerate(shots)]
    if total > sum(caps):
        raise ValueError("target duration is too long for the event density; reduce it, add visible actions, or merge short dialogue shots")
    weights = [max(0.01, (s["duration_seconds"] * 24 - 5) / 17) for s in shots]
    ideal = [total * w / sum(weights) for w in weights]
    result = list(minimum)
    for _ in range(total - sum(result)):
        index = max((i for i in range(count) if result[i] < caps[i]),
                    key=lambda i: ideal[i] - result[i])
        result[index] += 1
    return [17 * k + 5 for k in result]


def storyboard_warnings(asset):
    """Mechanical render-risk diagnostics for a validated v2 asset.

    Loud notes only; they never change validation or compilation. Mirrors
    the two live-render defect classes recorded in 2026-10: budgets pinned
    at the feasibility floor clip the final beat, and a continuation clip
    whose dialogue is owned by one character can drop that speaker.
    """
    shots, overlap = asset["shots"], asset["continuation_overlap_frames"]
    count = len(shots)
    total = round((asset["target_duration_seconds"] * 24 + overlap * (count - 1) - 5 * count) / 17)
    minimum = sum(max(0, math.ceil((max(1.0, estimate_shot_seconds(shot)) * 24
                                    + (overlap if i else 0) - 5) / 17))
                 for i, shot in enumerate(shots))
    out = []
    if total <= minimum:
        out.append("target_duration_seconds 恰好压在各片段最小可行预算上（零缓冲）：实际语速稍慢就会截掉收尾节拍，"
                   "建议提高约 1 秒或接受重摇 · duration pinned at the feasibility floor")
    frames = allocate_frames(asset)
    for i, (shot, length) in enumerate(zip(shots, frames)):
        need = estimate_shot_seconds(shot)
        effective = (length - (overlap if i else 0)) / 24
        if effective - need < 0.2:
            out.append("%s 预算贴地：扣除续接重叠后 %.2f 秒要装约 %.2f 秒台词与动作，收尾节拍最可能被截"
                       " · shot budget at the floor" % (shot["id"], effective, need))
        # A clip whose speech fills it loses its SHORTEST line: H3 drops a
        # clause rather than overrun the budget. Budget against the measured
        # delivery rate (5-6 characters per second), not the 8 the
        # feasibility floor assumes.
        slow_need = estimate_shot_seconds(shot, rate=5.5)
        if effective - slow_need < 0.5:
            out.append("%s 台词按慢速语速需要约 %.2f 秒，本镜只有 %.2f 秒：塞满的片段会被丢掉最短的一句"
                       " · speech budget too tight for a slow delivery"
                       % (shot["id"], slow_need, effective))
    # 2026-11-03: the vanish failure needs a VISIBLE sole speaker; an
    # off-screen voice cannot vanish from a frame it was never in, so a
    # voiceover-only continuation clip is not the trigger shape.
    presence = {r["id"]: r.get("presence", "on_screen")
                for r in asset.get("roles", ())}
    for shot in shots[1:]:
        speakers = {event["character"] for event in shot["events"] if event["type"] == "dialogue"}
        if len(speakers) == 1 and presence.get(next(iter(speakers))) != "off_screen":
            out.append("%s 是续生成片段且全部台词由单一角色独占：已知 H3 缺陷触发形态（说话者可能从画面消失），"
                       "建议合并片段或准备重摇该镜 · single-speaker continuation clip" % shot["id"])
    # Shouting delivery (2026-10-30 live): a furious scolding rendered at
    # ~2.4 chars/s — less than HALF the 5.5 budget rate — and ate the
    # ending beat. Mechanical proxy: the fraction of dialogue lines that
    # end with an exclamation mark.
    for shot in shots:
        lines = [e for e in shot["events"] if e["type"] == "dialogue"]
        shouted = [e for e in lines if e["text"].rstrip().endswith(("！", "!"))]
        if lines and len(shouted) * 2 >= len(lines):
            out.append("%s 台词 %d/%d 句为感叹式喊叫：实测怒斥语速约 2.5 字/秒（预算按 5.5 会截尾），"
                       "建议上调目标时长并给收尾拍留 1 秒以上 · shouting delivery measured "
                       "~2.5 chars/s" % (shot["id"], len(shouted), len(lines)))
    # Same-speaker burst (2026-10-30 live): after 5 consecutive lines by one
    # speaker the final line's lip-sync drifted to the listener's mouth
    # (the voice stayed correct) — prompt-level "lips closed" lines did not
    # hold. Mechanical proxy: longest run of consecutive dialogue events by
    # the same character inside one shot.
    for shot in shots:
        best = run = 0
        prev = None
        for event in shot["events"]:
            if event["type"] == "dialogue":
                run = run + 1 if event["character"] == prev else 1
                prev = event["character"]
                best = max(best, run)
            else:
                run, prev = 0, None
        if best >= 5:
            out.append("%s 有同一角色连续 %d 句台词的连珠炮：末句口型可能漂到对面角色嘴上"
                       "（2026-10-30 实测，声音不变嘴动漂移）；建议拆散到 ≤4 句，或把收尾反应"
                       "拆成无台词的第二片段 · long same-speaker burst, final-line lip-drift risk"
                       % (shot["id"], best))
    return out


def to_runtime(asset, *, wired_character_slots=(), wired_scene=False):
    """Compile events with picture IDs allocated only to wired on-screen roles.

    Assemblers pass the actual slot set; an empty set means text-only.
    Role IDs and speaker tags always remain stable regardless of pictures.

    ``wired_scene`` (2026-11-01 third-party review, P1): when a scene asset
    is wired, the replaceable DEFAULT fields — look, scene prose and the
    light line — are simply NOT EMITTED (field-level conditional output).
    The previous post-hoc line deletion could never match multi-line field
    values, leaving residue while the summary claimed an override.
    """
    asset = validate_v2(asset)
    roles = asset["roles"]
    execution = asset["execution"]
    indices = {r["id"]: i for i, r in enumerate(roles, 1)}
    names = {r["id"]: "Character %d" % i for i, r in enumerate(roles, 1)}
    picture_ids = [r["id"] for r in roles if r["presence"] == "on_screen"
                   and indices[r["id"]] in wired_character_slots]
    picture_of = {who: i for i, who in enumerate(picture_ids, 1)}
    openings, frames = _replay_visibility(asset), allocate_frames(asset)
    shots = []
    movement = "static locked-off camera maintaining the same framing" if asset["camera"]["movement"] == "locked" else "gentle camera follow motivated by the action"
    perspective = "first-person POV toward the camera" if asset["camera"]["perspective"] == "first_person" else "an observer viewpoint, separate from the cast"
    for shot, opening, length in zip(asset["shots"], openings, frames):
        framing = asset["camera"]["framing"].replace("_", " ")
        opening_subjects = [("%s from <Picture %d>" % (names[r["id"]], picture_of[r["id"]])
                             if r["id"] in picture_of else names[r["id"]])
                            for r in roles if r["id"] in opening]
        if opening_subjects:
            framing += " of %s, already visible at the first frame" % ", ".join(opening_subjects)
        lines = ["integrated_multimodal_description: [Shot 1] %s, %s, %s." %
                 (framing, movement, perspective)]
        if execution["look"] and not wired_scene:
            # Replaceable DEFAULT (2026-11-01): a wired scene asset owns the
            # visual world's style; scene-less renders keep the look line.
            lines.append(execution["look"])
        for role in roles:
            who, name = role["id"], names[role["id"]]
            if who in picture_of:
                lines.append("%s is the figure in <Picture %d>; use this reference for identity and appearance only."
                             % (name, picture_of[who]))
            elif role["presence"] == "off_screen":
                lines.append("%s stays exclusively off-screen as a voice." % name)
            else:
                design = execution.get("character_designs", {}).get(who, "")
                lines.append("%s appearance: %s" % (name, design or
                             "Generate a distinct appearance appropriate to the story and keep it consistent throughout."))
        lines.append("Opening cast: %s. Begin directly in this framing."
                     % (", ".join(names[r["id"]] for r in roles if r["id"] in opening) or "no visible cast before the explicit entrance"))
        absent = [names[r["id"]] for r in roles if r["id"] not in opening]
        if absent:
            lines.append("Not visible at the opening: %s. Only explicit entrance events may add them." % ", ".join(absent))
        for role in roles:
            if shot is asset["shots"][0] and role["id"] in opening and role["position"] != "none":
                lines.append("%s begins at %s of frame." % (names[role["id"]], role["position"]))
        if execution["setting"]:
            lines.append("Opening staging: " + execution["setting"])
        if execution.get("scene") and not wired_scene:
            # The optional environment line — a replaceable DEFAULT, not
            # emitted at all when a scene asset is wired (2026-10-29 ruling;
            # 2026-11-01 field-level conditional output replaces the old
            # post-hoc line strip, which never matched multi-line prose).
            lines.append("Scene context: " + execution["scene"])
        light = asset.get("light", "follow_image")
        light_line = (execution.get("light") or "") if light == "custom" else LIGHT_PROMPTS.get(light, "")
        if light_line and not wired_scene:
            # The light line is a replaceable DEFAULT too — dropped entirely
            # when a scene asset is wired (its own lighting stands).
            lines.append("Light: " + light_line)
        if shot is not asset["shots"][0]:
            lines.append("Continue the current blocking, completed action state and prop ownership from the preceding clip.")
        lines += ["Only the explicitly tagged dialogue is spoken. Do not speak stage directions or role labels.",
                  "Perform the following events once, in the exact written order; one speaker at a time, without overlapping or repeating lines.",
                  "Each specified prop exists in its stated quantity and stays with its stated holder. An offered object stays with the offering actor until an explicit accepted handover; refusal leaves the other actor empty-handed.",
                  "During dialogue-only events, keep incidental reference props with their initial holder and limit movement to natural speaking reactions.",
                  "The image contains exactly the declared visible cast in the present setting. Spoken stories remain audio within this present conversation."]
        lines.append("Visible state: " + execution["visible_state"])
        visible, speech = set(opening), []
        cuts = {cut["before_event"]: cut["framing"] for cut in shot.get("camera_cuts", [])}
        internal_shot = 1
        for event in shot["events"]:
            if event["id"] in cuts:
                internal_shot += 1
                lines.append("[Shot %d] %s, %s, %s. Cut directly to this framing within the same scene."
                             % (internal_shot, cuts[event["id"]].replace("_", " "), movement, perspective))
            who, kind = event["character"], event["type"]
            name = names[who]
            if kind == "dialogue":
                voice = execution["voices"][who] or "a natural distinct voice"
                lang = "Chinese" if event["language"] == "zh" else "English"
                tag = "<d>[%s] %s</d>" % (lang, event["text"])
                listeners = [names[r["id"]] for r in roles if r["id"] in visible and r["id"] != who]
                if who in visible:
                    lines.append("%s speaks as (S%d), %s: %s %s's lips move naturally with this line."
                                 % (name, indices[who], voice, tag, name))
                else:
                    listener_state = " and ".join(n + " remains on-screen with lips completely closed" for n in listeners)
                    lead = listener_state + " while an" if listener_state else "An"
                    lines.append(lead + " off-screen voice (S%d), %s, from %s says in voiceover: %s"
                                 % (indices[who], voice, name, tag))
                if listeners:
                    lines.append("While this line is spoken, %s." % " and ".join(n + "'s lips remain completely closed" for n in listeners))
                speech.append({"speaker": name, "lang": event["language"], "text": event["text"]})
            else:
                lines.append("%s %s: %s" % (name, {"action": "performs this action", "enter": "enters the frame", "exit": "leaves the frame"}[kind], execution["directions"][event["id"]]))
                if kind == "enter":
                    visible.add(who)
                elif kind == "exit":
                    visible.remove(who)
        lines += ["overall_soundscape: " + RUNTIME_SOUNDSCAPE,
                  "non_diegetic_music: No non-diegetic music."]
        shots.append({"lines": speech, "prompt": "\n".join(lines), "duration_seconds": length / 24})
    return {"name": asset["name"], "version": 1, "assembly_mode": "authored", "audio_mode": asset.get("audio_mode", "generated"),
            "language": asset["language"], "camera_discipline": "locked", "camera_perspective": asset["camera"]["perspective"],
            # light rides along for the builder's wired-scene conflict note
            # (the compiled "Light: ..." line already lives in each shot's
            # prompt; the raw key is transport metadata, not v1 contract).
            "light": asset.get("light", "follow_image"),
            # look text rides along so the builder can DROP the exact line
            # when a wired scene asset owns the visual world (2026-11-01
            # ruling: the script look is a DEFAULT; a wired scene image's
            # own style wins, like its lighting and environment).
            "look_text": (asset.get("execution") or {}).get("look", ""),
            "pacing": "normal", "seed_policy": "per_scene_increment", "prompt_prefix": [],
            "cast": {names[r["id"]]: {"character_slot": indices[r["id"]], "offscreen": r["presence"] == "off_screen"} for r in roles},
            "shots": shots, "ambience": "Natural location ambience.", "setting_line": "", "spatial_layout": {},
            "_structured_events": True}
