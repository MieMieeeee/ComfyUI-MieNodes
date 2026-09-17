"""Build a clean, minimal Excalidraw flow diagram of the H3 Loop
prompt-enhancement harness — based on the simple narrative description,
not on the full code surface.

Layout:
  Row 1: title + 2 stages (草稿 → 规范化)
  Row 2: 6 stages of the main loop (0.6 → 0.5 → 0 → 1 → 2 → 3)
"""
import json
import random
from pathlib import Path

OUT = Path(
    r"C:/Users/administered/PycharmProjects/ComfyUI-MieNodes/docs/"
    "h3_loop_flow.excalidraw"
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
    font_size=18, title_size=20,
):
    """One stage box: a rectangle paired with a bound text label.
    label_lines is a list of strings — first line is the stage title
    (bold-ish, larger); the rest are short description lines."""
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
        index=f"a{abs(hash(box_id)) % 99999:05d}",
    )
    # Build the label: title line first, blank, then body
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
        lineHeight=1.3,
        autoResize=True,
        containerId=box_id,
        index=f"b{abs(hash(box_id)) % 99999:05d}",
    )
    return [rect, txt]


def text_only(eid, x, y, w, h, text, *, font_size=18, font_family=1,
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


def arrow(eid, x, y, points, *, stroke="#1e1e1e", width=2, dash=False):
    return base(
        eid, x, y,
        abs(points[-1][0] - points[0][0]) if len(points) > 1 else 0,
        abs(points[-1][1] - points[0][1]) if len(points) > 1 else 0,
        type="arrow",
        roundness={"type": 2},
        strokeColor=stroke,
        backgroundColor="transparent",
        fillStyle="solid",
        strokeWidth=width,
        strokeStyle="dashed" if dash else "solid",
        points=points,
        startBinding=None,
        endBinding=None,
        lastCommittedPoint=None,
        startArrowhead=None,
        endArrowhead="arrow",
    )


elements = []

# ───────── Header ─────────
elements.append(text_only(
    "title", 200, 30, 1500, 44,
    "MiniMax H3 Loop — 提示词增强流程",
    font_size=32, text_align="center", vertical_align="middle"))

elements.append(text_only(
    "subtitle", 200, 80, 1500, 28,
    "rough draft → Production Plan JSON",
    font_size=16, color="#495057", text_align="center", vertical_align="middle"))

# ───────── Section A: 草稿规范化 ─────────
elements.append(text_only(
    "section_a", 60, 140, 600, 28,
    "第一段  ·  草稿规范化",
    font_size=18, color="#1971c2", text_align="left", vertical_align="middle"))

elements.extend(stage_box(
    "draft", 60, 180, 260, 130,
    [
        "草稿",
        "用户随手写的文字",
        "+ 类别 / 参考图模式",
        "+ 总时长 / 节奏等参数",
    ],
    bg="#fff3bf", font_size=15, title_size=20,
))

elements.extend(stage_box(
    "enhancer", 400, 180, 360, 130,
    [
        "UserInputEnhancer",
        "1 次 LLM",
        "四类分流：对白 / 动作 / 旁白 / 参考图",
        "改写 + 风险标注",
    ],
    bg="#a5d8ff", font_size=15, title_size=20,
))

elements.extend(stage_box(
    "user_input", 840, 180, 300, 130,
    [
        "user_input (规范文本)",
        "角色名：台词  对齐风格",
        "一拍一行           单段叙事",
        "图1→图2            镜头提示",
    ],
    bg="#d0ebff", font_size=14, title_size=20,
))

elements.append(arrow("a1", 320, 245, [[0, 0], [80, 0]]))
elements.append(arrow("a2", 760, 245, [[0, 0], [80, 0]]))

# ───────── Section B: 主循环 ─────────
elements.append(text_only(
    "section_b", 60, 350, 1000, 28,
    "第二段  ·  主节点 H3LoopPromptEnhancer",
    font_size=18, color="#2f9e44", text_align="left", vertical_align="middle"))

# Six stage boxes in a horizontal flow
W = 230
H = 200
GAP = 20
START_X = 60
Y = 400

stages = [
    (
        "stage_06",
        "Step 0.6 · 图片打标",
        ["按模式路由 (首帧/首尾/参考)",
         "逐张 LLM 写主体描述",
         "缓存到内存+磁盘",
         "→ 覆盖图说清单"],
        "#d0ebff", "#1971c2",
    ),
    (
        "stage_05",
        "Step 0.5 · 自动分场",
        ["对白稿: 抽取+节奏估算+打包",
         "叙事稿: LLM 自由分场",
         "产出 N 个 4-14 秒的场",
         "(场数=对白轮次 / LLM 决定)"],
        "#d3f9d8", "#2f9e44",
    ),
    (
        "stage_0",
        "Step 0 · 时长护栏",
        ["强制每场 4-14 秒",
         "对齐 17k+5 帧网格",
         "派生每场种子",
         "(纯确定性, 不调 LLM)"],
        "#e5dbff", "#5f3dc4",
    ),
    (
        "stage_1",
        "Step 1 · 共享前缀 + 角色档案",
        ["1 次 LLM",
         "全片不变项 (画风/调色/排除)",
         "+ 角色档案 (每人一行身份)",
         "(前缀里绝不出现角色)"],
        "#ffe3e3", "#c92a2a",
    ),
    (
        "stage_2",
        "Step 2 · 每场提示词",
        ["默认逐场独立 LLM",
         "注入前缀 + 上一场描述+音床",
         "参考图模式: 六段式结构",
         "(纯文生可选一次性调用)"],
        "#fff4e6", "#e67700",
    ),
    (
        "stage_3",
        "Step 3 · 装配 + 校验",
        ["组装 plan_json",
         "校验 schema / 标签 / 对话不变式",
         "输出 5 个 socket:",
         "  plan_json / shots / 前缀 /",
         "  preflight / preview"],
        "#c5f6fa", "#0c8599",
    ),
]

for i, (sid, title, body, bg, stroke) in enumerate(stages):
    x = START_X + i * (W + GAP)
    elements.extend(stage_box(
        sid, x, Y, W, H,
        [title] + body,
        bg=bg, stroke=stroke, stroke_width=2,
        font_size=13, title_size=15,
    ))

# arrows between the six stages
for i in range(len(stages) - 1):
    x_from = START_X + i * (W + GAP) + W
    x_to = START_X + (i + 1) * (W + GAP)
    aid = f"loop_{i}_{i+1}"
    elements.append(arrow(aid, x_from, Y + H // 2,
                         [[0, 0], [GAP, 0]], stroke="#495057"))

# ───────── Footer ─────────
elements.append(text_only(
    "footer", 60, 640, 1500, 28,
    "确定性：时长网格 / 节奏数学 / schema 校验 / 标签策略 / 对话不变式     //     LLM：草稿改写 / 分场 / 打标 / 前缀 / 每场",
    font_size=12, color="#868e96", text_align="center", vertical_align="middle"))

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "gridSize": 20,
    },
    "files": {},
}

OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
print(f"  rectangles: {sum(1 for e in elements if e['type']=='rectangle')}")
print(f"  texts:      {sum(1 for e in elements if e['type']=='text')}")
print(f"  arrows:     {sum(1 for e in elements if e['type']=='arrow')}")
