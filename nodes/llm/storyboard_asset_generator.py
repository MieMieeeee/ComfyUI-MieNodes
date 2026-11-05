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
        # Critic notes for the LAST call (the node prints them).
        self.warnings = []

    def __call__(self, concept, shot_count, *, mode="script", base_asset=None,
                 character_count=2, character_settings="",
                 name="新剧本", target_duration_seconds=20, camera_framing="medium_shot",
                 camera_movement="locked", camera_perspective="observer", language="zh",
                 look="", setting="", constraints="", continuation_overlap_frames=22,
                 seed=0, audio_mode="generated", scene="", light="follow_image",
                 light_text="", character_designs="", critic="on", **_unused):
        from ..assets.storyboard_contract import parse_roles, parse_script, validate_v2
        if mode != "revise":
            if isinstance(character_designs, str):
                try:
                    character_designs = json.loads(character_designs.strip() or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError("character_designs must be a JSON object mapping role IDs to default appearances") from exc
            if not isinstance(character_designs, dict) or any(
                    not isinstance(key, str) or not re.fullmatch(r"character_[1-8]", key)
                    or not isinstance(value, str) or not value.strip()
                    for key, value in character_designs.items()):
                raise ValueError("character_designs must map character_1..8 to nonempty default appearances")
        if mode == "revise":
            # Revision mode (2026-10-29): the asset is the saved source of
            # truth users iterate on. Wire the previous version's storyboard
            # bundle, write the requested change in the concept box, and the
            # LLM produces the NEXT revision — untouched parts stay put.
            if not isinstance(base_asset, dict) or not base_asset:
                raise ValueError(
                    "修订模式需要把「加载剧本资产」的 storyboard 输出接到本节点的 storyboard "
                    "输入（先加载要微调的那一版）· revise mode needs the previous version "
                    "wired into the storyboard input (Load Script Asset → storyboard)")
            if not str(concept or "").strip():
                raise ValueError(
                    "concept（修订要求）是空的：写这一版要改什么，例如「结尾改成她笑而不语，"
                    "其余不动」· write what THIS revision should change; everything else stays")
            base = {k: v for k, v in base_asset.items() if k != "_path"}
            base = validate_v2(base)
            payload = {"mode": "revise",
                       "base_asset": base,
                       "revision": str(concept).strip(),
                       "caller_owned": {"name": name}}
            inherited_audio = base.get("audio_mode", "generated")
            inherited_overlap = base.get("continuation_overlap_frames", 22)
            # Wiring facts are INHERITED from the base revision and stay
            # caller-owned: the model may not flip audio_mode or the loop
            # overlap while rewriting dialogue.
            audio_mode = inherited_audio
            continuation_overlap_frames = inherited_overlap
            header = None
            expected_shots = 0
        elif mode == "concept":
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
            if character_designs:
                preferences["character_designs"] = character_designs
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
            if character_designs:
                header["character_designs"] = character_designs
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
        # The critic stage (2026-11-05) sits AFTER the contract: the asset
        # is schema-valid first, then a second reader looks across the
        # shots. A contract failure is fatal after three attempts (the
        # asset is the contract); a critic failure is not - the last
        # contract-valid asset is returned with the notes attached.
        contract_valid = None
        notes = []
        critic_on = str(critic or "on").lower() != "off"
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
                from ..assets.storyboard_review import review_asset_independence
                validated = validate_v2(candidate)
                issues = review_asset_independence(validated)
                if issues:
                    issue = issues[0]
                    raise ValueError("%s: %s Offending text: %s" % (
                        issue["path"], issue["reason"], issue["excerpt"]))
                contract_valid = validated
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
                if attempt < 2:
                    messages = messages + [{"role": "assistant", "content": raw},
                                           {"role": "user", "content":
                                            "Asset validation failed: %s. Correct this specific error. "
                                            "Keep asset_header and locked_events unchanged; return only the complete JSON object."
                                            % error[:800]}]
                    continue
                break
            # --- contract satisfied; now the critic reads across shots ---
            feedback = ""
            notes = []
            if critic_on:
                from ..assets.storyboard_critic import (HARD, critique_warnings,
                                                        review_asset_critique)
                issues = review_asset_critique(validated)
                notes = critique_warnings(issues)
                hard = [i for i in issues if i["severity"] == HARD]
                if hard:
                    feedback = ("The execution copy contradicts itself:\n" +
                                "\n".join("- %s：%s" % (i["path"], i["reason"])
                                          for i in hard[:5]))
                else:
                    from .storyboard_critic import (critical_findings, findings_feedback,
                                                    findings_warnings, run_asset_critic)
                    result = run_asset_critic(self.llm, validated, temperature=0.0,
                                              max_tokens=2048, timeout=self.timeout, seed=seed)
                    notes = notes + findings_warnings(result)
                    critical = critical_findings(result)
                    if critical:
                        feedback = findings_feedback(critical)
            if not feedback:
                self.warnings = notes
                return validated
            error = feedback
            if attempt < 2:
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content":
                                        "Storyboard critic: %s\nFix ONLY these problems. Keep every "
                                        "event (id, type, character, text, language) and every "
                                        "caller-owned field unchanged; return only the complete JSON object."
                                        % feedback[:1200]}]
                continue
            break
        if contract_valid is not None:
            # The contract holds, the critic did not get its way: ship the
            # asset with the notes instead of failing the whole node.
            self.warnings = notes + ["⚠ critic：连续性复核未在 3 次内通过，已交付通过契约校验的版本 · "
                                     "critic could not be satisfied; the contract-valid asset was kept"]
            return contract_valid
        raise RuntimeError("storyboard asset validation failed after 3 attempts: " + error)


def asset_diff(old, new, limit=5):
    """Mechanical old→new revision report (revise mode's review surface).

    Pure diff, no judgments: which events survived verbatim, which were
    reworded/added/removed (pairing by actor+type), which structural
    fields moved, and how the delivered duration changed. The user reads
    this to decide whether the revision is the one they wanted.
    """
    def flat(a):
        return [e for s in a["shots"] for e in s["events"]]
    old_events, new_events = flat(old), flat(new)
    kept = [e for e in old_events if e in new_events]
    removed_pool = [e for e in old_events if e not in new_events]
    added_pool = [e for e in new_events if e not in old_events]
    by_key = {}
    for e in removed_pool:
        by_key.setdefault((e["type"], e["character"]), []).append(e)
    reworded, pure_added = [], []
    for e in added_pool:
        bucket = by_key.get((e["type"], e["character"]))
        if bucket:
            reworded.append((bucket.pop(0), e))
        else:
            pure_added.append(e)
    pure_removed = [e for bucket in by_key.values() for e in bucket]
    lines = ["修订对比 · revision diff（旧 %s → 新 %s）" % (old.get("name", "?"), new.get("name", "?")),
             "事件：%d 条逐字未动 · %d 条改写 · 新增 %d · 删除 %d（%d → %d 条）"
             % (len(kept), len(reworded), len(pure_added), len(pure_removed),
                len(old_events), len(new_events))]
    for o, n in reworded[:limit]:
        # Dialogue rewrites carry ⚠ (2026-10-30 live: M3 sneaked a filler
        # character into an untouched line; the diff caught it) — the whole
        # point of the revision review surface.
        tag = ("⚠ 台词改写 · dialogue reworded" if o["type"] == "dialogue"
               else "改写 · reworded")
        lines.append("  %s · %s「%s」→「%s」"
                     % (tag, o["character"], o["text"][:28], n["text"][:28]))
    for e in pure_added[:limit]:
        lines.append("  新增 · %s「%s」" % (e["character"], e["text"][:32]))
    for e in pure_removed[:limit]:
        lines.append("  删除 · %s「%s」" % (e["character"], e["text"][:32]))
    for field in ("look", "setting", "scene", "constraints", "character_designs"):
        if (old.get(field) or "") != (new.get(field) or ""):
            lines.append("字段 · %s 已更新 · %s updated" % (field, field))
    old_light, new_light = old.get("light", "follow_image"), new.get("light", "follow_image")
    if old_light != new_light:
        lines.append("光锚 · light：%s → %s" % (old_light, new_light))
    if old.get("camera") != new.get("camera"):
        lines.append("相机 · camera：%s → %s" % (old.get("camera"), new.get("camera")))
    old_roles = {r["id"]: r for r in old["roles"]}
    for r in new["roles"]:
        o = old_roles.get(r["id"])
        if o is None:
            lines.append("角色 · %s 新增：%s" % (r["id"], r["name"]))
        elif any(o.get(k) != r.get(k) for k in ("name", "presence", "position", "voice")):
            lines.append("角色 · %s 调整：%s → %s（画面 %s/%s · 位置 %s/%s）"
                         % (r["id"], o.get("name"), r.get("name"),
                            o.get("presence"), r.get("presence"),
                            o.get("position"), r.get("position")))
    from ..assets.storyboard_contract import allocate_frames
    for label, a in (("旧", old), ("新", new)):
        frames = allocate_frames(a)
        overlap = a.get("continuation_overlap_frames", 22)
        lines.append("时长 · %s：%d 段 · 交付 %.2f 秒"
                     % (label, len(frames),
                        (sum(frames) - overlap * (len(frames) - 1)) / 24))
    return lines


def preview_asset(asset, diff_lines=None, warnings=None):
    from ..assets.storyboard_contract import allocate_frames, storyboard_warnings
    names = {r["id"]: r["name"] for r in asset["roles"]}
    frames = allocate_frames(asset)
    delivered = (sum(frames) - asset["continuation_overlap_frames"] * (len(frames) - 1)) / 24
    lines = ["# 剧本资产 · " + asset["name"], "",
             "%d 个角色 · %d 个生成片段 · %d 个镜头 · 预计交付 %.3f 秒" %
             (len(names), len(frames), sum(1 + len(s.get("camera_cuts", [])) for s in asset["shots"]),
              delivered), ""]
    for warning in storyboard_warnings(asset):
        lines += ["⚠ " + warning, ""]
    for warning in (warnings or []):
        lines += [str(warning), ""]
    if diff_lines:
        lines += ["", "## " + diff_lines[0]] + diff_lines[1:]
    for role in asset["roles"]:
        lines.append("- %s → %s · %s · %s" %
                     (role["id"], role["name"], role["presence"], role["position"]))
    for who, design in asset.get("character_designs", {}).items():
        lines.append("- 默认角色设计 · %s（接入该角色资产时覆盖）：%s" % (who, design))
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
