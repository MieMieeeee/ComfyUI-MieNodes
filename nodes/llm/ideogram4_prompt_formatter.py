"""Parse, validate, and normalize Ideogram 4 JSON captions (used by Ideogram4PromptGenerator)."""

from __future__ import annotations

import json
import re

try:
    from _mienodes_internal.nodes.llm.ideogram4_caption_verifier import CaptionVerifier
except ImportError:
    from .ideogram4_caption_verifier import CaptionVerifier

# ComfyUI full-schema workflow: drop stray aspect_ratio (size is external), keep bboxes.
_STRIP_ASPECT_RATIO = True
_STRIP_BBOXES = False

_TOP_LEVEL_KEEP = frozenset(
    {
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    }
)


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def repair_json_text(text: str) -> str:
    i, j = text.find("{"), text.rfind("}")
    t = text[i : j + 1] if i != -1 and j > i else text
    return re.sub(
        r'("(?:[^"\\]|\\.)*")|,(\s*[}\]])',
        lambda m: m.group(1) or m.group(2),
        t,
    )


def parse_caption_dict(raw_text: str) -> tuple[dict | None, list[str]]:
    """Try to parse caption JSON with progressive repairs. Returns (dict, fix_log)."""
    fixes: list[str] = []
    text = (raw_text or "").strip()
    if not text:
        return None, fixes

    fenced = strip_code_fences(text)
    if fenced != text:
        fixes.append("removed markdown code fences")
        text = fenced

    candidates: list[str] = [text]
    repaired = repair_json_text(text)
    if repaired != text:
        candidates.append(repaired)
        fixes.append("repaired trailing commas / extracted JSON object")

    m = re.search(r"\{[\s\S]*\}", text)
    if m and m.group(0) not in candidates:
        candidates.append(m.group(0))
        rep = repair_json_text(m.group(0))
        if rep not in candidates:
            candidates.append(rep)
        fixes.append("extracted outermost JSON object from surrounding text")

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, fixes
        except json.JSONDecodeError:
            continue
    return None, fixes


def _normalize_hex(color: str) -> str | None:
    if not isinstance(color, str):
        return None
    c = color.strip()
    if not c.startswith("#"):
        c = "#" + c
    if len(c) == 4:
        c = "#" + "".join(ch * 2 for ch in c[1:])
    if len(c) != 7:
        return None
    return c.upper()


def _normalize_palette(palette, max_colors: int) -> tuple[list[str], bool]:
    if not isinstance(palette, list):
        return [], False
    out: list[str] = []
    changed = False
    for color in palette[:max_colors]:
        norm = _normalize_hex(color)
        if norm is None:
            return out, changed
        if norm != color:
            changed = True
        out.append(norm)
    return out, changed


def _normalize_bbox(bbox) -> tuple[list[int] | None, bool]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, False
    vals: list[int] = []
    for v in bbox:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, False
        vals.append(int(round(v)))
    ymin, xmin, ymax, xmax = vals
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    def clamp(v: int) -> int:
        return max(0, min(1000, v))

    normed = [clamp(ymin), clamp(xmin), clamp(ymax), clamp(xmax)]
    return normed, normed != vals


def _ordered_dict(d: dict, order: tuple[str, ...]) -> dict:
    known = [k for k in order if k in d]
    extra = [k for k in d if k not in order]
    return {k: d[k] for k in (*known, *extra)}


def reorder_caption_keys(caption: dict) -> tuple[dict, bool]:
    """Reorder caption keys to canonical schema order."""
    verifier = CaptionVerifier()
    changed = False
    out: dict = {}

    for key in ("high_level_description", "style_description", "compositional_deconstruction"):
        if key in caption:
            out[key] = caption[key]

    extra_top = [k for k in caption if k not in _TOP_LEVEL_KEEP]
    if extra_top:
        changed = True
    for k in extra_top:
        out.pop(k, None)

    sd = out.get("style_description")
    if isinstance(sd, dict):
        try:
            order = verifier._style_description_key_order(sd)
            new_sd = _ordered_dict(sd, order)
            if list(new_sd.keys()) != list(sd.keys()):
                changed = True
            out["style_description"] = new_sd
        except ValueError:
            pass

    cd = out.get("compositional_deconstruction")
    if isinstance(cd, dict):
        new_cd = _ordered_dict(cd, verifier.compositional_deconstruction_key_order)
        if list(new_cd.keys()) != list(cd.keys()):
            changed = True
        elements = new_cd.get("elements")
        if isinstance(elements, list):
            reordered_elements = []
            for elem in elements:
                if not isinstance(elem, dict):
                    reordered_elements.append(elem)
                    continue
                try:
                    order = verifier._element_key_order(elem)
                    new_elem = _ordered_dict(elem, order)
                    if list(new_elem.keys()) != list(elem.keys()):
                        changed = True
                    reordered_elements.append(new_elem)
                except ValueError:
                    reordered_elements.append(elem)
            new_cd["elements"] = reordered_elements
        out["compositional_deconstruction"] = new_cd

    if out != caption:
        changed = True
    return out, changed


def _repair_missing_text_field(elem: dict) -> bool:
    """If LLM put literal copy only in ``desc``, try to extract a quoted ``text`` value."""
    if elem.get("type") != "text" or "text" in elem:
        return False
    desc = elem.get("desc") or ""
    if not isinstance(desc, str):
        return False
    patterns = (
        r"text\s+['\"]([^'\"]+)['\"]",
        r"masthead\s+['\"]([^'\"]+)['\"]",
        r"headline\s+['\"]([^'\"]+)['\"]",
        r"title\s+['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]{1,120})['\"]",
    )
    for pat in patterns:
        m = re.search(pat, desc, flags=re.IGNORECASE)
        if m:
            elem["text"] = m.group(1)
            return True
    return False


def normalize_caption(caption: dict) -> tuple[dict, list[str]]:
    """Apply auto-fixes that do not change semantic meaning."""
    fixes: list[str] = []
    data = json.loads(json.dumps(caption))  # deep copy

    if _STRIP_ASPECT_RATIO and "aspect_ratio" in data:
        data.pop("aspect_ratio", None)
        fixes.append("removed top-level aspect_ratio")

    for key in list(data.keys()):
        if key not in _TOP_LEVEL_KEEP:
            data.pop(key, None)
            fixes.append(f"removed unknown top-level key '{key}'")

    sd = data.get("style_description")
    if isinstance(sd, dict):
        pal, pal_changed = _normalize_palette(
            sd.get("color_palette"), CaptionVerifier.style_description_palette_max
        )
        if pal_changed:
            sd["color_palette"] = pal
            fixes.append("normalized style_description.color_palette to uppercase #RRGGBB")

    cd = data.get("compositional_deconstruction")
    if isinstance(cd, dict):
        elements = cd.get("elements")
        if isinstance(elements, list):
            for i, elem in enumerate(elements):
                if not isinstance(elem, dict):
                    continue
                if _repair_missing_text_field(elem):
                    fixes.append(f"inferred elements[{i}].text from quoted text in desc")
                if _STRIP_BBOXES and "bbox" in elem:
                    elem.pop("bbox", None)
                    fixes.append(f"removed elements[{i}].bbox")
                elif "bbox" in elem:
                    normed, bbox_changed = _normalize_bbox(elem["bbox"])
                    if normed is not None and bbox_changed:
                        elem["bbox"] = normed
                        fixes.append(f"normalized elements[{i}].bbox coordinates")
                if "color_palette" in elem:
                    pal, pal_changed = _normalize_palette(
                        elem.get("color_palette"), CaptionVerifier.element_palette_max
                    )
                    if pal_changed:
                        elem["color_palette"] = pal
                        fixes.append(f"normalized elements[{i}].color_palette")

    data, reordered = reorder_caption_keys(data)
    if reordered:
        fixes.append("reordered JSON keys to canonical schema order")

    return data, fixes


def compact_caption(caption: dict) -> str:
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def format_ideogram4_caption(raw_text: str) -> tuple[str, str]:
    """Parse, auto-fix, validate, and return (compact_json, format_log).

    Raises ``ValueError`` when the caption has fatal schema errors.
    """
    caption, parse_fixes = parse_caption_dict(raw_text)
    if caption is None:
        raise ValueError(
            "Ideogram 4 caption: cannot parse JSON from LLM output "
            "(tried fence removal, comma repair, and object extraction)"
        )

    caption, norm_fixes = normalize_caption(caption)
    all_fixes = parse_fixes + norm_fixes

    raw_compact = compact_caption(caption)
    warnings = CaptionVerifier().verify(caption)
    fatal, minor = CaptionVerifier.split_warnings(warnings)

    if fatal:
        detail = "\n".join(f"  - {w}" for w in fatal)
        raise ValueError(
            "Ideogram 4 caption: failed schema validation:\n" + detail
        )

    log_lines = []
    if all_fixes:
        log_lines.append("Auto-fixes:")
        log_lines.extend(f"  - {f}" for f in all_fixes)
    if minor:
        log_lines.append("Warnings (non-fatal):")
        log_lines.extend(f"  - {w}" for w in minor)
    if not log_lines:
        log_lines.append("OK: no fixes needed")

    return raw_compact, "\n".join(log_lines)

