"""One checked LLM grouping pass for portable storyboard assets."""
from __future__ import annotations

import json
import re
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompts" / "h3_storyboard" / "system_asset.txt"


class AssetStoryboardEnhancer:
    def __init__(self, llm_service_connector, *, temperature=0.2, max_tokens=8192, timeout=120):
        self.llm = llm_service_connector
        self.temperature, self.max_tokens, self.timeout = float(temperature), int(max_tokens), timeout

    def __call__(self, concept, shot_count, *, mode="script", character_count=2, character_settings="",
                 name="新剧本", target_duration_seconds=20, camera_framing="medium_shot",
                 camera_movement="locked", camera_perspective="observer", language="zh",
                 look="", setting="", constraints="", continuation_overlap_frames=22,
                 seed=0, audio_mode="generated", scene="", light="follow_image",
                 light_text="", **_unused):
        from ..assets.storyboard_contract import parse_roles, parse_script, validate_v2
        if mode == "concept":
            # Free-creation mode (2026-10-29 ruling): the user pastes an idea,
            # a story paragraph or loose dialogue lines — the LLM authors the
            # COMPLETE asset. No format requirements on the input at all.
            # Wiring facts stay caller-owned; every creative choice is a
            # non-binding preference the model may adapt to the concept.
            if not str(concept or "").strip():
                raise ValueError("concept 是空的：写一个点子、一段故事或几句台词草稿都行 · "
                                 "concept is empty; type an idea, a story beat or draft lines")
            preferences = {"character_count_hint": character_count,
                           "target_duration_seconds": target_duration_seconds,
                           "camera": {"framing": camera_framing, "movement": camera_movement,
                                      "perspective": camera_perspective},
                           "language": language}
            if shot_count:
                preferences["shot_count"] = shot_count
            if str(character_settings).strip():
                preferences["character_settings"] = character_settings
            for key, value in (("look", look), ("setting", setting), ("scene", scene),
                               ("constraints", constraints), ("light", light),
                               ("light_text", light_text)):
                if str(value).strip():
                    preferences[key] = value
            payload = {"mode": "concept", "concept": str(concept).strip(),
                       "caller_owned": {"name": name, "audio_mode": audio_mode,
                                        "continuation_overlap_frames": continuation_overlap_frames},
                       "preferences": preferences}
            header = None
            expected_shots = 0
        else:
            roles = parse_roles(character_count, character_settings)
            events = parse_script(concept, roles, language)
            if type(shot_count) is not int or not 0 <= shot_count <= min(20, len(events)):
                raise ValueError("shot_count must be 0..20 and cannot exceed the number of source events")
            header = {"version": 2, "name": name, "language": language, "roles": roles,
                      "camera": {"framing": camera_framing, "movement": camera_movement,
                                 "perspective": camera_perspective},
                      "look": look, "setting": setting, "constraints": constraints,
                      "target_duration_seconds": target_duration_seconds,
                      "continuation_overlap_frames": continuation_overlap_frames,
                      "source_script": concept.strip()}
            if audio_mode != "generated":
                header["audio_mode"] = audio_mode
            if str(scene).strip():
                header["scene"] = str(scene).strip()
            if light != "follow_image":
                header["light"] = light
            if light == "custom":
                header["light_text"] = str(light_text).strip()
            # Check caller fields and source visibility before a provider request.
            # An arbitrary probe partition cannot prove timing infeasibility: the
            # model may need unequal groups to fit a long utterance into one clip.
            probe_count = shot_count or min(len(events), 20)
            groups = [events[i * len(events) // probe_count:(i + 1) * len(events) // probe_count]
                      for i in range(probe_count)]
            probe = dict(header, shots=[{"id": "scene_%02d" % (i + 1),
                                        "duration_seconds": min(30, target_duration_seconds / probe_count + 1),
                                        "events": group} for i, group in enumerate(groups)])
            validate_v2(probe, _check_timing=False, _check_execution=False)
            payload = {"mode": "script", "asset_header": header, "locked_events": events}
            if shot_count:
                payload["required_shot_count"] = shot_count
            expected_shots = shot_count
        system = PROMPT_PATH.read_text(encoding="utf-8")
        user = json.dumps(payload, ensure_ascii=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        error = "no response"
        for attempt in range(3):
            previous_timeout = getattr(self.llm, "timeout", None)
            try:
                if self.timeout:
                    self.llm.timeout = int(self.timeout)
                raw = self.llm.invoke(messages, seed=seed, temperature=self.temperature,
                                      max_tokens=self.max_tokens)
            finally:
                if previous_timeout is not None:
                    self.llm.timeout = previous_timeout
            raw = re.sub(r"^\s*<think>.*?</think>\s*", "", raw or "", flags=re.S).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
            try:
                candidate = json.loads(raw)
                if not isinstance(candidate, dict):
                    raise ValueError("expected one version-2 asset JSON object")
                # Caller-owned checks: wiring facts both modes enforce.
                if str(candidate.get("name", "")).strip() != str(name).strip():
                    raise ValueError("validation: caller-owned name changed")
                if candidate.get("audio_mode", "generated") != audio_mode:
                    raise ValueError("validation: caller-owned audio_mode changed")
                if mode != "concept" and header is not None:
                    for key, value in header.items():
                        if candidate.get(key) != value:
                            raise ValueError("validation: caller-owned %s changed" % key)
                    if expected_shots and (not isinstance(candidate.get("shots"), list)
                                           or len(candidate["shots"]) != expected_shots):
                        raise ValueError("validation: requested shot_count was not honored")
                elif candidate.get("continuation_overlap_frames") != continuation_overlap_frames:
                    raise ValueError("validation: caller-owned continuation_overlap_frames changed")
                return validate_v2(candidate)
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
                if attempt < 2:
                    messages = messages + [{"role": "assistant", "content": raw},
                                           {"role": "user", "content":
                                            "Asset validation failed: %s. Correct this specific error. "
                                            "Keep asset_header and locked_events unchanged; return only the complete JSON object."
                                            % error[:800]}]
        raise RuntimeError("storyboard asset validation failed after 3 attempts: " + error)


def preview_asset(asset):
    from ..assets.storyboard_contract import allocate_frames, storyboard_warnings
    names = {r["id"]: r["name"] for r in asset["roles"]}
    frames = allocate_frames(asset)
    delivered = (sum(frames) - asset["continuation_overlap_frames"] * (len(frames) - 1)) / 24
    lines = ["# 剧本资产 · " + asset["name"], "",
             "%d 个角色 · %d 个生成片段 · %d 个镜头 · 预计交付 %.3f 秒 · 续接重叠 %d 帧" %
             (len(names), len(frames), sum(1 + len(s.get("camera_cuts", [])) for s in asset["shots"]),
              delivered, asset["continuation_overlap_frames"]), ""]
    for warning in storyboard_warnings(asset):
        lines += ["⚠ " + warning, ""]
    for role in asset["roles"]:
        lines.append("- %s → %s · %s · %s" %
                     (role["id"], role["name"], role["presence"], role["position"]))
    scene = (asset.get("scene") or "").strip()
    if scene:
        lines += ["", "场景 · scene（接线场景资产时被覆盖 · overridden by a wired scene asset）：",
                  "  " + scene]
    light = asset.get("light", "follow_image")
    if light != "follow_image":
        detail = ("：%s" % asset.get("light_text", "")) if light == "custom" else ""
        lines += ["", "光锚 · light：%s%s" % (light, detail)]
    for shot, length in zip(asset["shots"], frames):
        lines += ["", "## %s · 原始 %.3f 秒" % (shot["id"], length / 24)]
        cuts = {cut["before_event"]: cut["framing"] for cut in shot.get("camera_cuts", [])}
        for event in shot["events"]:
            if event["id"] in cuts:
                lines.append("- 切镜 → " + cuts[event["id"]])
            lines.append("- [%s] %s · %s：%s" %
                         (event["id"], names[event["character"]], event["type"], event["text"]))
    return "\n".join(lines)
