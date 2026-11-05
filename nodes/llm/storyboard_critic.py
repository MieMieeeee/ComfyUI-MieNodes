"""The model half of the storyboard critic.

The deterministic half lives in ``nodes/assets/storyboard_critic.py`` and
catches what two instructions cannot both be true about. This half asks
a plain text LLM the questions only a reader can answer: does every
source line survive, does a character keep the same coat, does a prop
change hands without a written handover, does the board still tell the
story in order.

No vision model is involved: the critic reads the asset JSON, not a
render. It therefore costs one extra provider call per generation and
never touches the render loop - the assets stay "generated once, rendered
without a model".

Protocol rules (borrowed from the orchestrator's critic, which is the
only public implementation of this idea I could read):
  * a finding WITHOUT a location is discarded - a criticism nobody can
    act on is noise;
  * an unreadable reply is "unsure", and unsure never blocks: the asset
    is returned with the problem noted, not rejected;
  * only "critical" findings justify a re-roll; "minor" ones are notes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts" / "h3_storyboard"
ASSET_CRITIC_PROMPT = PROMPT_DIR / "critic_asset.txt"
SHOT_CRITIC_PROMPT = PROMPT_DIR / "critic_shots.txt"

MAX_FINDINGS = 8
_RETRY_HINT = ("Reply with ONLY the JSON object {\"findings\": [...]} and nothing else. "
               "Every finding needs a path.")


def _strip(raw):
    raw = re.sub(r"^\s*<think>.*?</think>\s*", "", raw or "", flags=re.S).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    return raw


def _parse_findings(raw):
    """(findings, unsure) - a reply we cannot read is unsure, not a failure."""
    try:
        data = json.loads(_strip(raw))
    except (ValueError, TypeError):
        return [], True
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        return [], True
    findings = []
    for item in data["findings"]:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            # A finding nobody can locate is discarded.
            continue
        severity = str(item.get("severity") or "").strip().lower()
        findings.append({
            "severity": "critical" if severity == "critical" else "minor",
            "path": path,
            "issue": str(item.get("issue") or "").strip(),
            "expected": str(item.get("expected") or "").strip(),
            "observed": str(item.get("observed") or "").strip(),
            "fix": str(item.get("fix") or "").strip(),
        })
    return findings[:MAX_FINDINGS], False


def _call(llm, messages, *, temperature, max_tokens, timeout, seed):
    previous_timeout = getattr(llm, "timeout", None)
    try:
        if timeout:
            llm.timeout = int(timeout)
        return llm.invoke(messages, seed=seed, temperature=temperature,
                          max_tokens=max_tokens)
    finally:
        if previous_timeout is not None:
            llm.timeout = previous_timeout


def _run(llm, system, payload, *, temperature, max_tokens, timeout, seed):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    try:
        raw = _call(llm, messages, temperature=temperature, max_tokens=max_tokens,
                    timeout=timeout, seed=seed)
    except Exception:
        # A critic that cannot reach the provider must not break a
        # generation: report unsure and let the asset through.
        return {"findings": [], "unsure": True}
    findings, unsure = _parse_findings(raw)
    if not unsure:
        return {"findings": findings, "unsure": False}
    # One strict retry, then the reply counts as unsure.
    retry = messages + [{"role": "assistant", "content": _strip(raw) or ""},
                         {"role": "user", "content": _RETRY_HINT}]
    try:
        raw = _call(llm, retry, temperature=temperature, max_tokens=max_tokens,
                    timeout=timeout, seed=seed)
    except Exception:
        return {"findings": [], "unsure": True}
    findings, unsure = _parse_findings(raw)
    return {"findings": findings, "unsure": unsure}


def _asset_view(asset):
    """The projection the critic reads - the story, not the bookkeeping."""
    execution = asset.get("execution") or {}
    directions = execution.get("directions") or {}
    shots = []
    for shot in asset.get("shots", ()):
        shots.append({
            "id": shot.get("id"),
            "duration_seconds": shot.get("duration_seconds"),
            "camera_cuts": shot.get("camera_cuts") or [],
            "events": [{"id": e.get("id"), "type": e.get("type"),
                        "character": e.get("character"), "text": e.get("text")}
                       for e in shot.get("events", ())],
            "directions": {e["id"]: directions[e["id"]] for e in shot.get("events", ())
                           if e.get("id") in directions},
        })
    return {
        "source_script": asset.get("source_script", ""),
        "language": asset.get("language"),
        "camera": asset.get("camera"),
        "light": asset.get("light", "follow_image"),
        "light_text": asset.get("light_text", ""),
        "scene": asset.get("scene", ""),
        "look": asset.get("look", ""),
        "setting": asset.get("setting", ""),
        "constraints": asset.get("constraints", ""),
        "target_duration_seconds": asset.get("target_duration_seconds"),
        "roles": [{"id": r.get("id"), "name": r.get("name"),
                   "presence": r.get("presence"),
                   "initially_visible": r.get("initially_visible"),
                   "position": r.get("position")} for r in asset.get("roles", ())],
        "character_designs": asset.get("character_designs") or {},
        "shots": shots,
        "execution": {
            "visible_state": execution.get("visible_state", ""),
            "setting": execution.get("setting", ""),
            "directions": directions,
            "character_designs": execution.get("character_designs") or {},
        },
    }


def run_asset_critic(llm, asset, *, temperature=0.0, max_tokens=2048, timeout=120, seed=None):
    """Critique one asset. Returns {"findings": [...], "unsure": bool}."""
    system = ASSET_CRITIC_PROMPT.read_text(encoding="utf-8")
    return _run(llm, system, _asset_view(asset), temperature=temperature,
                max_tokens=max_tokens, timeout=timeout, seed=seed)


def run_shot_list_critic(llm, shots, *, concept="", temperature=0.0, max_tokens=2048,
                         timeout=120, seed=None):
    """Critique a legacy shot list. Returns {"findings": [...], "unsure": bool}."""
    system = SHOT_CRITIC_PROMPT.read_text(encoding="utf-8")
    payload = {"concept": concept, "shots": [
        {key: shot.get(key) for key in
         ("id", "description", "shot_type", "camera_movement", "transition_in",
          "duration_seconds", "narrative_beat", "characters", "props", "notes")}
        for shot in (shots or []) if isinstance(shot, dict)]}
    return _run(llm, system, payload, temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, seed=seed)


def critical_findings(result):
    """Only the findings that justify a re-roll."""
    return [f for f in (result or {}).get("findings", []) if f.get("severity") == "critical"]


def findings_feedback(findings, *, headline="The storyboard critic found problems a viewer would notice:"):
    """The corrective turn the repair loop sends back to the model."""
    if not findings:
        return ""
    lines = [headline]
    for finding in findings:
        detail = finding.get("fix") or finding.get("issue") or ""
        lines.append("- [%s] %s" % (finding.get("path", "?"), detail))
    return "\n".join(lines)


def findings_warnings(result):
    """Human-readable notes for the node preview."""
    out = []
    for finding in (result or {}).get("findings", []):
        mark = "⛔" if finding.get("severity") == "critical" else "⚠"
        out.append("%s critic · %s：%s" % (mark, finding.get("path", "?"),
                                           finding.get("issue") or finding.get("fix") or ""))
    if (result or {}).get("unsure"):
        out.append("⚠ critic：这次的审片回复无法解析，本次未做连续性复核 · critic reply was unreadable")
    return out
