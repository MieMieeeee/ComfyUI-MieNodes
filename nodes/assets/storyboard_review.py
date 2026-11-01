"""Conservative checks for visual defaults leaking into portable performance copy."""
import re

# These are high-confidence warnings, not a complete semantic validator.
_RULES = [
    ("light", r"\b(?:morning|afternoon|evening)\s+(?:daylight|sunlight)\b[^.;\n]*"),
    ("light", r"\b(?:in the lamplight|face lit by the screen glow)\b"),
    ("scene", r"\b(?:against|with|in front of)\s+(?:a|an|the)\s+[^.;\n]{0,90}\bbackdrop\b"),
    ("light", r"\b(?:illuminated|lit|bathed)\s+(?:by|in|with)\s+[^.;\n]{1,70}"),
    ("light", r"\b(?:neon|warm|cool|soft|bright|dim)\s+lighting\s+(?:fills|illuminates|bathes)\b[^.;\n]*"),
    ("character_designs", r"\bCharacter\s+\d+\s+has\s+[^.;\n]{0,60}\b(?:eyes|hair|fur|skin)\b"),
]

def review_asset_independence(asset):
    """Return relocation feedback; never edit dialogue, events, or input assets."""
    execution = asset.get("execution") or {}
    fields = {"execution." + key: execution.get(key, "")
              for key in ("setting", "visible_state")}
    fields.update({"execution.directions." + key: value
                   for key, value in (execution.get("directions") or {}).items()})
    defaults = [(key, execution.get(key, "")) for key in ("scene", "light", "look")]
    defaults.extend(("character_designs", value) for value in
                    (execution.get("character_designs") or {}).values())
    issues = []
    for path, value in fields.items():
        if not isinstance(value, str):
            continue
        found = set()
        for target, pattern in _RULES:
            match = re.search(pattern, value, re.I)
            if match:
                found.add((target, match.group(0)))
        for target, default in defaults:
            if isinstance(default, str) and len(default.strip()) >= 20:
                fragment = default.strip().rstrip(".。")
                if fragment.casefold() in value.casefold():
                    found.add((target, fragment))
        for target, excerpt in sorted(found):
            issues.append({"path": path, "target": target, "excerpt": excerpt,
                           "reason": "Move default visual design to %s; keep only portable performance here." % target})
    return issues
