"""Bake ideogram4/_system_*.txt from magic_prompt_v1.txt [SYSTEM] + mode suffixes.

Run after editing magic_prompt_v1.txt [SYSTEM] or the COMPOSITION MODE blocks
at the bottom of each _system_*.txt file.

    python scripts/build_ideogram4_system_prompts.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "nodes" / "llm" / "prompts" / "ideogram4"
MAGIC = PROMPTS / "magic_prompt_v1.txt"

MODE_SUFFIXES = {
    "simple": """

COMPOSITION MODE (simple): Default for scenes, interiors, collages, portraits, landscapes, and magazine covers with a hero subject.
- Omit bbox on elements unless the user explicitly requests precise pixel placement.
- Use positional language in desc instead (e.g. top center, lower left quadrant, upper-right corner).
- For horizontal magazine/poster titles: describe wide horizontal placement in desc; do NOT use tall narrow vertical text-column bboxes.
- For cutout/sticker requests in ComfyUI: use a solid chroma-blue (#0000FF) or chroma-green (#00FF00) background shell — do NOT use transparent background.
""",
    "complex": """

COMPOSITION MODE (complex): For typography-dense posters and multi-zone layouts (Flow / T-Rex style).
- Every element must have a bbox [y_min, x_min, y_max, x_max] on the 0–1000 grid.
- type:"text" requires both a text field and a bbox; use wide horizontal bboxes for horizontal lines.
- Avoid tall narrow vertical text-column bboxes unless the user asks for vertical / tategaki text.
""",
    "movable": """

COMPOSITION MODE (movable): Editable layout — bbox is the SOLE position authority.
- Every element MUST have a bbox [y1, x1, y2, x2].
- Do NOT describe placement in desc or high_level_description; the renderer reads position only from bbox.
- This lets you move any element by editing its bbox with zero textual conflict.

## MOVABLE / BBOX-ONLY POSITIONING (OVERRIDES all earlier placement rules)

In this mode the renderer reads each element's position ONLY from its bbox. Spatial placement language is FORBIDDEN in desc and high_level_description. The bbox is the sole source of truth; editing a bbox must move the element with zero textual conflict.

### desc fields (obj AND text) — NO placement language
- Describe identity, material, color, form, intrinsic parts, and pose ONLY.
- Write each object as if it floats in a void — no frame, no neighbours, no scene context referenced.
- FORBIDDEN placement words/phrases: upper, lower, left, right, corner, foreground, background (as position), center, centre, quadrant, top, bottom, edge, near, beside, next to, adjacent, flanking, above, below, behind, in front of, surrounded by, against the [sky/wall/ground], in the [sky/field/air], overhead, underneath.
- IGNORE the earlier "Anchor placements to named references" rule entirely. Do NOT anchor to spatial landmarks of the scene.
- INTRINSIC anchors are fine and encouraged — references to the object's OWN body or parts are NOT placement and stay allowed: "a scar on the left cheek", "a ring on the right hand", "logo on the chest". Here left/right name the object's own anatomy, not its slot in the frame.
- For text elements: desc covers font family, weight, size, color, case, style and decorative effects ONLY — never location. This overrides the earlier "desc covers … location" clause.

### high_level_description — NO placement language
- Describe subjects, medium, mood and palette ONLY.
- Do NOT state where anything sits in the frame (no "in the foreground", "to the upper-left", "in the corner").

### bbox is MANDATORY on every element
- Every obj and text element MUST carry a bbox [y1, x1, y2, x2].
- The "OMIT bboxes" guidance from earlier does NOT apply in this mode.

### Unchanged
- All non-placement rules still apply in full: specificity / commit-to-one-value, no shadows in descs, no hedge words, background as scene shell, single subject per element, exhaustive text fidelity, pop-culture / brand naming, CJK and non-ASCII preservation, single-line minified JSON output contract.
- `background` keeps describing the scene shell (sky, ground, distant context) as before — it is not a movable element, so its natural spatial structure is fine.
""",
}


def parse_system_block(raw: str) -> str:
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[1:-1].strip().lower()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    if "system" not in sections:
        raise ValueError(f"{MAGIC.name} has no [SYSTEM] section")
    return sections["system"]


def main() -> int:
    base = parse_system_block(MAGIC.read_text(encoding="utf-8"))
    for mode, suffix in MODE_SUFFIXES.items():
        out = PROMPTS / f"_system_{mode}.txt"
        content = base + suffix
        out.write_text(content, encoding="utf-8")
        print(f"wrote {out.name} ({len(content)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
