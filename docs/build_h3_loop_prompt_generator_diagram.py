"""Build a single-image Excalidraw diagram of the
MiniMaxH3LoopPromptGenerator node — parameters + call flow on one page.

Layout (1600 wide x ~1200 tall):

    title  ─────────────────────────────────────────────────
    ┌──required──┐  ┌── central node ──┐  ┌──outputs──┐
    │  4 boxes  │  │ MiniMaxH3...     │  │ 3 boxes   │
    └───────────┘  │                 │  └───────────┘
    ┌─ optional ─┐  │                 │  ┌─ summary ─┐
    │ 12 boxes   │  └─────────────────┘  │ config    │
    └────────────┘                        │ recipes   │
                                          └───────────┘
    ─────────── 6-stage call flow strip ───────────
    enhance → caption → facts → stage0 → prefix → shot
"""
import json
import random
from pathlib import Path

OUT = Path(
    r"C:/Users/administered/PycharmProjects/ComfyUI-MieNodes/docs/"
    "h3_loop_prompt_generator_diagram.excalidraw"
)

random.seed(7)


def rng():
    return random.randint(1, 2_000_000_000)


def base(eid, x, y, width, height, **overrides):
    e = {
        "id": eid,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": rng(),
        "version": 1,
        "versionNonce": rng(),
        "isDeleted": False,
        "boundElements": None,
        "updated": 1700000000000,
        "link": None,
        "locked": False,
    }
    e.update(overrides)
    return e


def stage_box(
    box_id, x, y, w, h, label_lines, *,
    bg="#a5d8ff", stroke="#1e1e1e", stroke_width=2,
    font_size=14, title_size=18,
):
    text_id = f"{box_id}__label"
    rect = base(
        box_id, x, y, w, h,
        type="rectangle",
        roundness={"type": 3},
        strokeColor=stroke,
        backgroundColor=bg,
        fillStyle="solid",
        strokeWidth=stroke_width,
        strokeStyle="solid",
        boundElements=[{"id": text_id, "type": "text"}],
    )
    title = label_lines[0]
    body = label_lines[1:]
    body_text = "\n" + "\n".join(body) if body else ""
    full = f"{title}{body_text}"
    txt = base(
        text_id,
        x, y, w, h,
        type="text",
        roundness=None,
        strokeColor="#1e1e1e",
        backgroundColor="transparent",
        fillStyle="solid",
        strokeWidth=1,
        boundElements=None,
        text=full,
        fontSize=font_size,
        fontFamily=1,
        textAlign="center",
        verticalAlign="middle",
        baseline=font_size,
        originalText=full,
        lineHeight=1.25,
        autoResize=True,
        containerId=box_id,
    )
    return [rect, txt]


def text_only(eid, x, y, w, h, text, *, font_size=16, font_family=1,
              text_align="left", vertical_align="middle", color="#1e1e1e"):
    return base(
        eid, x, y, w, h,
        type="text",
        roundness=None,
        strokeColor=color,
        backgroundColor="transparent",
        fillStyle="solid",
        strokeWidth=1,
        boundElements=None,
        text=text,
        fontSize=font_size,
        fontFamily=font_family,
        textAlign=text_align,
        verticalAlign=vertical_align,
        baseline=font_size,
        originalText=text,
        lineHeight=1.25,
        autoResize=True,
        containerId=None,
    )


def arrow(eid, x1, y1, x2, y2, *, stroke="#1e1e1e", width=2, dash=False):
    return base(
        eid, x1, y1,
        abs(x2 - x1), abs(y2 - y1),
        type="arrow",
        roundness={"type": 2},
        strokeColor=stroke,
        backgroundColor="transparent",
        fillStyle="solid",
        strokeWidth=width,
        strokeStyle="dashed" if dash else "solid",
        points=[[0, 0], [x2 - x1, y2 - y1]],
        startBinding=None,
        endBinding=None,
        lastCommittedPoint=None,
        startArrowhead=None,
        endArrowhead="arrow",
    )


elements = []

# ────── Header ──────
elements.append(text_only(
    "title", 100, 20, 1500, 44,
    "MiniMaxH3LoopPromptGenerator  —  参数 + 调用流一张图",
    font_size=28, text_align="center", vertical_align="middle",
))
elements.append(text_only(
    "subtitle", 100, 70, 1500, 24,
    "INPUT_TYPES  →  节点处理链  →  plan_json (H3 Chain 消费)",
    font_size=14, color="#495057", text_align="center", vertical_align="middle",
))

# ────── Required inputs (top-left, narrow column) ──────
RX = 60  # left column x
RY = 120
elements.append(text_only(
    "req_header", RX, RY, 380, 24,
    "Required  (4)",
    font_size=16, color="#c92a2a", text_align="left", vertical_align="middle",
))

# 4 required boxes — short stack
req_widgets = [
    ("req_connector", "llm_service_connector",
     ["LLMServiceConnector (link)",
      "SetMiniMaxTokenPlanLLMServiceConnector"]),
    ("req_userinput", "user_input",
     ["STRING multiline",
      "对话行 = locked verbatim"]),
    ("req_enhance", "enhance_user_input",
     ["off (default) / on",
      "on = +1 LLM 润色 (canonical 别用)"]),
    ("req_seed", "seed",
     ["INT default=0  |  0=时钟派生"]),
]
y = RY + 30
for box_id, name, lines in req_widgets:
    elements.extend(stage_box(
        box_id, RX, y, 380, 70,
        [name] + lines,
        bg="#ffe3e3", stroke="#c92a2a", font_size=11, title_size=14,
    ))
    y += 78

# ────── Optional inputs (left column, lower) — two-column grid ──────
OY = y + 20
elements.append(text_only(
    "opt_header", RX, OY, 800, 24,
    "Optional  (13) — 全部 0/默认值即 auto",
    font_size=16, color="#1971c2", text_align="left", vertical_align="middle",
))

opt_widgets = [
    ("opt_scene", "scene_count",
     ["INT default=0  |  0=auto 最小切分"]),
    ("opt_duration", "total_duration_seconds",
     ["INT default=0  |  每场 clamp 4-14s"]),
    ("opt_pacing", "pacing",
     ["fast/normal(def)/slow  |  语速, 不影响切数"]),
    ("opt_genmode", "generation_mode",
     ["per_shot (def) / single_call  |  逐场连续性好"]),
    ("opt_category", "category",
     ["none/dialogue/action  |  抽到对话 → auto dialogue"]),
    ("opt_lang", "output_language",
     ["en (def) / zh  |  不影响 locked 对白"]),
    ("opt_seedmode", "seed_mode",
     ["per_scene_increment (def)  |  全场不同 seed"]),
    ("opt_temp", "temperature",
     ["FLOAT default=0.4  |  0.0-2.0  step 0.05"]),
    ("opt_tokens", "max_tokens",
     ["INT default=16384  |  64-32768  思维链 16k+"]),
    ("opt_timeout", "timeout",
     ["60/120/300(def)/600 s  |  Ollama 用 120"]),
    ("opt_refmode", "reference_mode",
     ["t2va(def)/i2va/fl2va/ref2va  |  → images 接线"]),
    ("opt_images", "images",
     ["IMAGE batch 1-9  |  i2va/fl2va/ref2va 必接"]),
    ("opt_caption", "caption_mode",
     ["memory_disk(def)/memory_only/no_cache/force_recap"]),
]
# All 13 widgets in one tight stack on the left strip.
opt_col_w = 380
opt_box_h = 58
opt_y_step = 64
y_start = OY + 30
for i, (box_id, name, lines) in enumerate(opt_widgets):
    cy = y_start + i * opt_y_step
    elements.extend(stage_box(
        box_id, RX, cy, opt_col_w, opt_box_h,
        [name] + lines,
        bg="#d0ebff", stroke="#1971c2", font_size=9, title_size=12,
    ))

# ────── Center node (huge) ──────
CX = 500
CY = 200
elements.extend(stage_box(
    "node_center", CX, CY, 540, 880,
    [
        "MiniMaxH3LoopPromptGenerator | Mie",
        "",
        "FUNCTION = generate",
        "CATEGORY = 🐑 MieNodes/🐑 Prompt Generator",
        "",
        "user_input",
        "  →  enhance (optional)",
        "  →  caption pipeline (5 图片)",
        "  →  semantic facts (layout / bindings)",
        "  →  stage0: scene_count + duration",
        "  →  prefix (3-section contract)",
        "  →  per-shot shot: 6-section / 3-section",
        "  →  speaker voice sheet + identity pin",
        "  →  assemble dialogue_line_blocks",
        "  →  <d>[Lang] ...</d>  (locked verbatim)",
        "",
        "RETURN_TYPES = (STRING, STRING, STRING)",
        "RETURN_NAMES = plan_json, summary, board_kind",
    ],
    bg="#fff3bf", stroke="#1e1e1e", font_size=14, title_size=20,
    stroke_width=3,
))

# ────── Outputs (right column) ──────
OX = 1100
OY = 200
elements.append(text_only(
    "out_header", OX, OY, 440, 24,
    "Outputs  (3)",
    font_size=16, color="#2b8a3e", text_align="left", vertical_align="middle",
))

outputs = [
    ("out_plan_json", "plan_json  (STRING)",
     ["H3 生产链消费的 JSON",
      "{ shots: [...], prompt_prefix: [...] }",
      "→ MiniMaxH3ChainPlanModern.plan_json_input"]),
    ("out_summary", "summary  (STRING)",
     ["人类可读总览",
      "enhance rewrite / 切片决策",
      "引用图 manifest / seed 链"]),
    ("out_kind", "board_kind  (STRING)",
     ["dialogue / narration / reference",
      "供下游节点选合同",
      "switch-on 用"]),
]
y = OY + 30
for box_id, name, lines in outputs:
    elements.extend(stage_box(
        box_id, OX, y, 440, 90,
        [name] + lines,
        bg="#d3f9d8", stroke="#2b8a3e", font_size=12, title_size=15,
    ))
    y += 105

# ────── Quick recipes (right column, below outputs) ──────
RX2 = 1100
RY2 = y + 30
elements.append(text_only(
    "recipes_header", RX2, RY2, 440, 24,
    "Quick recipes",
    font_size=16, color="#862e9c", text_align="left", vertical_align="middle",
))
y = RY2 + 30
recipes = [
    ("rec_first", "首次跑通",
     ["enhance=on, pacing=normal",
      "scene_count=0, duration=0",
      "per_shot + t2va"]),
    ("rec_canon", "Canonical 文本",
     ["enhance=off (防双重润色)",
      "其余默认即可",
      "user_input 已结构化时"]),
    ("rec_continue", "稳定性 / 续接",
     ["seed_mode=per_scene_increment",
      "seed 给固定值",
      "checkpoint recovery"]),
    ("rec_local", "本地 Ollama",
     ["timeout=120 (冷启动)",
      "max_tokens=4096 (小模型)",
      "temperature 默认"]),
]
for box_id, name, lines in recipes:
    elements.extend(stage_box(
        box_id, RX2, y, 440, 75,
        [name] + lines,
        bg="#eebefa", stroke="#862e9c", font_size=11, title_size=13,
    ))
    y += 85

# ────── Bottom: 6-stage call-flow strip ──────
FY = 1430
elements.append(text_only(
    "flow_header", 60, FY, 1480, 24,
    "  ↳ 一次完整调用的 6 阶段 LLM 流（per_shot 时 stage6 按场景数 ×N）",
    font_size=14, color="#1e1e1e", text_align="left", vertical_align="middle",
))
FY += 30
flow = [
    ("flow1", "1. enhance",
     ["可选 +1 LLM",
      "MiniMax-M3 改写",
      "user_input 规范化"]),
    ("flow2", "2. caption",
     ["图片 caption",
      "memory+disk cache",
      "5 张并行调"]),
    ("flow3", "3. semantic facts",
     ["layout / bindings",
      "LLM span extract",
      "deterministic + fallback"]),
    ("flow4", "4. stage0 storyboard",
     ["scene_count 切分",
      "duration 分配",
      "自动 pacing"]),
    ("flow5", "5. prefix",
     ["3-section 合同",
      "voice sheet",
      "identity pin"]),
    ("flow6", "6. shot × N",
     ["6-section / 3-section",
      "dialogue_line_blocks",
      "<d>[Lang] locked"]),
]
fx = 60
fw = 245
gap = 5
for box_id, name, lines in flow:
    elements.extend(stage_box(
        box_id, fx, FY, fw, 100,
        [name] + lines,
        bg="#c5f6fa", stroke="#0c8599", font_size=11, title_size=14,
    ))
    fx += fw + gap
    if box_id != "flow6":
        # arrow to next stage
        elements.append(arrow(
            f"{box_id}_arrow", fx - gap, FY + 50,
            fx, FY + 50, stroke="#0c8599", width=2,
        ))


# ────── Wire everything to JSON ─ ──────
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

OUT.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT}")
print(f"  {len(elements)} elements")