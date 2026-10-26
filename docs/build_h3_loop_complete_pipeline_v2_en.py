"""Build the English counterpart of h3_loop_complete_pipeline_v2_zh.excalidraw.

Takes the FIXED zh file as the geometry/layout source of truth (same boxes,
colors, arrows, ids) and swaps every text for its English translation.
Geometry is recomputed with the rules extracted from the Excalidraw sources
(same as _fix_centering3.py / _fix_free_text_dims.py / _fix_label_widths.py):

- bound label height = lines * fontSize * lineHeight, y centered in the box;
- English detail lines are word-wrapped to the box interior width
  (box.width - 24) so text stays inside the boxes; identifier tokens are
  never broken (they contain no spaces);
- label width = max(interior, longest line + 8) so the editor's raster cache
  (width + fontSize padding) can't clip; x pins the visual center;
- free texts (title/subtitle/footnote) get real width/height and are
  centered over the canvas (they were 10x10 placeholders in the stale
  English v2 file, i.e. invisible in the editor).

Output: h3_loop_complete_pipeline_v2_en.excalidraw (new file; the stale
unsuffixed English v2 is left untouched — it is not tracked in git).
"""
import json
import os
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "h3_loop_complete_pipeline_v2_zh.excalidraw")
OUT = os.path.join(HERE, "h3_loop_complete_pipeline_v2_en.excalidraw")

# element id -> english lines (logical lines; wrapped to box width later)
EN = {
    "label_001": ["MiniMax H3 Loop Pipeline v2 (aligned with latest code)"],
    "label_002": ["Focus: semantic fact extraction + dialogue-as-data + voice consistency"],
    "label_003": [
        "v2 key changes: facts-gated extraction, local prefix shortcut, "
        "VOICE sheet, dialogue-as-data replacement, board_kind output"
    ],
    "txt_001": [
        "Entry / Widgets",
        "MiniMaxH3LoopPromptGenerator.generate",
        "user_input / images / mode / pacing / seed",
        "Returns: plan_json, summary, board_kind",
    ],
    "txt_002": [
        "Optional Enhancement",
        "enhance_user_input=on → run_enhancer",
        "Extracts category + notes + rewritten text",
        "Missing BEGIN/END block → hard failure",
    ],
    "txt_003": [
        "Reference Image Manifest",
        "images → _caption_images (memory/disk cache)",
        "or parse references_text",
        "Mode gating: i2va/fl2va/ref2va need references",
    ],
    "txt_004": [
        "Consistency / Fact Candidates",
        "_referenced_picture_numbers (concept text)",
        "_manifest_consistency_warnings",
        "Character bindings = original + rewritten text",
    ],
    "txt_005": [
        "Stage 0.5a: Dialogue Fast Path",
        "parse_structured_dialogue_turns",
        "All lines speaker:line → deterministic spans",
        "Skips LLM dialogue extraction when possible",
    ],
    "txt_006": [
        "Stage 0.5b: extract_board_facts",
        "One LLM call: turns + layout + bindings",
        "Mechanical span validation + relocate_extracted_lines",
        "Drops meta pseudo-speakers (scene-setting/camera/…)",
    ],
    "txt_007": [
        "Stage 0.5c: Scene Strategy",
        "Dialogue + scene_count=0 → minimal-cut packing",
        "Dialogue + explicit count → even line distribution",
        "Too many scenes → silent reaction shots inserted",
    ],
    "txt_008": [
        "Stage 0.5d: Storyboard",
        "Dialogue → deterministic shot building",
        "Narration → storyboard LLM",
        "category none → dialogue (if any)",
    ],
    "txt_009": [
        "Stage 0: Deterministic Guardrails",
        "Duration clamped to 4-14 s",
        "seconds_to_length → 17k+5 frames",
        "Derives per-scene seed (or one seed for all)",
    ],
    "txt_010": [
        "Stage 1: Prefix + CAST + VOICE",
        "Short dialogue/no refs → local prefix shortcut",
        "Else _synth_prefix (LLM) + split_prefix_sections",
        "VOICE sheet parsed per speaker",
    ],
    "txt_011": [
        "Stages 1.55 / 1.6 / 1.5",
        "Speaker voice sheet + default_voice_for fallback",
        "Caption-based CAST override (identity anchoring)",
        "Spatial layout: semantic facts first, regex fallback",
    ],
    "txt_012": [
        "Prefix Safety Scrub",
        "scrub_prefix_dialogue_language",
        "Removes forced dialogue-language claims",
        "Language ownership belongs to <d> lines only",
    ],
    "txt_013": [
        "Stage 2: Prompt Generation",
        "per_shot (default) or single_call (t2va only)",
        "build_reference_directive + continuation directives",
        "Injects tempo/layout/bindings into user template",
    ],
    "txt_014": [
        "Dialogue-as-Data Injection",
        "scrub_dialogue_from_prompt_text",
        "assemble_dialogue_line_blocks (voice+S<n>+facing)",
        "install_shot_speech → node-owned description body",
    ],
    "txt_015": [
        "Stage 2.5 Check",
        "validate_spatial_layout_invariant",
        "Spatial drift → warning only",
        "Does not block generation",
    ],
    "txt_016": [
        "Stages 3 + 4",
        "validate_plan",
        "validate_label_policy (referenced pictures)",
        "Outputs plan_json + summary + board_kind",
    ],
}


def char_em(ch: str) -> float:
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 1.0
    if ch == " ":
        return 0.30
    if ch.isupper() or ch.isdigit():
        return 0.62
    if ch.islower():
        return 0.52
    return 0.45


def est_line_width(line: str, font_size: float) -> float:
    return sum(char_em(c) for c in line) * font_size


def word_wrap(line: str, font_size: float, max_width: float) -> list:
    """Greedy word wrap; single tokens wider than max_width stay unbroken."""
    words = line.split(" ")
    out, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if est_line_width(trial, font_size) <= max_width or not cur:
            cur = trial
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

boxes = {el["id"]: el for el in data["elements"] if el["type"] == "rectangle"}
canvas_center_x = (
    min(b["x"] for b in boxes.values())
    + max(b["x"] + b["width"] for b in boxes.values())
) / 2

for el in data["elements"]:
    if el["type"] != "text" or el["id"] not in EN:
        continue
    fs, lh = el["fontSize"], el["lineHeight"]
    lines = EN[el["id"]]
    if el.get("containerId") in boxes:
        b = boxes[el["containerId"]]
        interior = b["width"] - 24
        wrapped = [wl for logical in lines for wl in word_wrap(logical, fs, interior)]
        text = "\n".join(wrapped)
        el["height"] = len(wrapped) * fs * lh
        el["y"] = b["y"] + (b["height"] - el["height"]) / 2
        need = max(est_line_width(wl, fs) for wl in wrapped) + 8
        el["width"] = max(interior, need)
        el["x"] = b["x"] + (b["width"] - el["width"]) / 2
        print(
            f'{el["id"]}: {len(wrapped)} lines h={el["height"]:.1f} '
            f'w={el["width"]:.1f} (box interior {interior})'
        )
    else:
        text = "\n".join(lines)
        el["height"] = len(lines) * fs * lh
        need = max(est_line_width(ln, fs) for ln in lines) * 1.15
        el["width"] = need
        el["x"] = canvas_center_x - need / 2
        print(f'{el["id"]}: w={need:.1f} h={el["height"]:.1f} center={canvas_center_x}')
    el["text"] = text
    el["originalText"] = text

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nwrote {os.path.basename(OUT)}")
