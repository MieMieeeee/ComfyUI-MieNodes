# -*- coding: utf-8 -*-
"""User-facing H3 Loop intro diagram (no code). Opens in Excalidraw.

Text is pre-positioned at the visual center of each box (horizontal +
vertical). Arrows are orthogonal, uncurved, and routed through gaps
so they do not cross other nodes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).with_name("h3_loop_user_intro.excalidraw")
random.seed(21)

LINE_HEIGHT = 1.3
PAD = 12


def rng() -> int:
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
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
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
        "index": f"a{abs(hash(eid)) % 99999:05d}",
    }
    e.update(overrides)
    return e


def measure(lines, font_size, line_height=LINE_HEIGHT):
    def line_w(s: str) -> float:
        w = 0.0
        for ch in s:
            w += font_size if ord(ch) > 127 else font_size * 0.55
        return w

    width = max((line_w(s) for s in lines), default=font_size)
    height = max(1, len(lines)) * font_size * line_height
    return width, height


def labeled_shape(shape, box_id, x, y, w, h, lines, *, bg, font_size=18):
    """Box + text pre-positioned at the visual center.

    Text is grouped with the box but NOT bound via ``containerId``, so
    opening the file in Excalidraw cannot reflow it to the top.
    """
    text_id = f"{box_id}__t"
    gid = f"g_{box_id}"
    roundness = {"type": 3} if shape != "diamond" else None
    inner_w, inner_h = w - 2 * PAD, h - 2 * PAD
    tw, th = measure(lines, font_size)
    tw = min(tw, inner_w)
    th = min(th, inner_h)
    tx = x + (w - tw) / 2
    ty = y + (h - th) / 2
    rect = base(
        box_id, x, y, w, h,
        type=shape,
        roundness=roundness,
        backgroundColor=bg,
        groupIds=[gid],
        boundElements=None,
    )
    full = "\n".join(lines)
    txt = base(
        text_id, tx, ty, tw, th,
        type="text",
        strokeWidth=1,
        groupIds=[gid],
        boundElements=None,
        text=full,
        originalText=full,
        fontSize=font_size,
        fontFamily=5,
        textAlign="center",
        verticalAlign="middle",
        baseline=font_size,
        lineHeight=LINE_HEIGHT,
        autoResize=False,
        containerId=None,
    )
    return [rect, txt]


def text_only(eid, x, y, w, h, text, *, font_size=20, color="#1e1e1e",
              align="left"):
    return base(
        eid, x, y, w, h,
        type="text",
        strokeWidth=1,
        strokeColor=color,
        text=text,
        originalText=text,
        fontSize=font_size,
        fontFamily=5,
        textAlign=align,
        verticalAlign="middle",
        baseline=font_size,
        lineHeight=1.25,
        autoResize=True,
        containerId=None,
    )


def arrow(eid, points_abs, *, dash=False, color="#1e1e1e"):
    """Orthogonal arrow. ``points_abs`` are canvas coordinates.
    ``roundness`` is None so elbows stay square and do not bow into boxes.
    """
    x0, y0 = points_abs[0]
    rel = [[px - x0, py - y0] for px, py in points_abs]
    xs = [p[0] for p in rel]
    ys = [p[1] for p in rel]
    return base(
        eid, x0, y0, max(xs) - min(xs), max(ys) - min(ys),
        type="arrow",
        roundness=None,
        strokeColor=color,
        strokeWidth=2,
        strokeStyle="dashed" if dash else "solid",
        points=rel,
        startBinding=None,
        endBinding=None,
        lastCommittedPoint=None,
        startArrowhead=None,
        endArrowhead="arrow",
    )


# ---------------------------------------------------------------------------
# Layout — three horizontal bands with explicit corridors between them.
#
#   y 170–280   inputs (no arrows between them)
#   y 280–360   corridor (image dashed line runs LEFT in the margin)
#   y 360–620   extract → diamond → pack/story
#   y 620–720   corridor (pack/story merge runs here, then down into style)
#   y 720–850   style → write → plan → out
#   y 880       legend
# ---------------------------------------------------------------------------

elements = []

elements.append(text_only(
    "title", 80, 20, 1480, 44,
    "从一段想法，到一条可连续生成的片子",
    font_size=36, align="center",
))
elements.append(text_only(
    "subtitle", 80, 68, 1480, 28,
    "MiniMax H3 Loop  ·  给用户看的流程，不涉及实现细节",
    font_size=18, color="#495057", align="center",
))

# ── Band 1: inputs ──
elements.append(text_only(
    "sec1", 80, 112, 400, 28,
    "1. 你准备什么",
    font_size=22, color="#1971c2",
))
# x: 80 / 460 / 840 / 1220   w: 340 / 340 / 340 / 300   gap 40
elements.extend(labeled_shape(
    "rectangle", "in_concept", 80, 148, 340, 120,
    ["概念与对白", "谁在画面里、坐哪、说什么", "一句话一行最稳"],
    bg="#fff3bf", font_size=18,
))
elements.extend(labeled_shape(
    "rectangle", "in_img", 460, 148, 340, 120,
    ["可选：参考图", "锁外貌和服装，没有图就纯文生", "全片人物和每一场都会用"],
    bg="#fff3bf", font_size=17,
))
elements.extend(labeled_shape(
    "rectangle", "in_knob", 840, 148, 340, 120,
    ["节奏 / 时长 / 场数", "节奏只管语速，不增减场数", "场数 0 = 少切；填了就是你要的"],
    bg="#fff3bf", font_size=17,
))
elements.extend(labeled_shape(
    "rectangle", "in_polish", 1220, 148, 300, 120,
    ["可选：先润色", "抽词前把草稿写成规范概念", "已是「角色：台词」可关掉"],
    bg="#fff3bf", font_size=16,
))

# ── Band 2: listen ──
elements.append(text_only(
    "sec2", 80, 308, 560, 28,
    "2. 先听清有没有人在说话",
    font_size=22, color="#1971c2",
))
# extract 80–400,  diamond 480–740,  pack/story 840–1220
elements.extend(labeled_shape(
    "rectangle", "extract", 80, 360, 320, 140,
    ["找出每一句台词", "「角色：台词」至少两句则直接切开", "自由散文才去听；听不清改走旁白"],
    bg="#a5d8ff", font_size=16,
))
elements.extend(labeled_shape(
    "diamond", "has_dlg", 480, 350, 260, 160,
    ["有对白吗？"],
    bg="#ffd43b", font_size=22,
))
elements.extend(labeled_shape(
    "rectangle", "pack", 840, 340, 380, 120,
    ["有 → 按说话打包成场", "未填场数：14 秒内少切，换人不强切", "填了场数：均分；多的是静默反应"],
    bg="#b2f2bb", font_size=16,
))
elements.extend(labeled_shape(
    "rectangle", "story", 840, 500, 380, 120,
    ["没有 → 按故事拆成场", "纯旁白、空镜、动作戏", "由模型决定切几刀"],
    bg="#d0ebff", font_size=17,
))

# extract right 400 → diamond left 480, at extract/diamond mid y=430
elements.append(arrow("e1", [(400, 430), (480, 430)]))
# diamond right 740, upper → pack left 840, pack mid y=400
elements.append(arrow("e2", [(740, 400), (840, 400)]))
# diamond right 740, lower → story left 840, story mid y=560
# diamond bottom is 510, so drop to y=560 in the 740–840 gap first
elements.append(arrow("e3", [(740, 480), (740, 560), (840, 560)]))
# "有/没有" sit in the 740–800 gap, left of the x=800 merge corridor.
elements.append(text_only("yes", 748, 368, 48, 22, "有", font_size=16, color="#2f9e44"))
elements.append(text_only("no", 742, 532, 52, 22, "没有", font_size=16, color="#1971c2"))

# ── Band 3: plan ──
# Title sits above write/plan (x>=420) so the drop into style at x=230
# does not run through the heading.
elements.append(text_only(
    "sec3", 420, 656, 500, 28,
    "3. 写成一条连续的分场计划",
    font_size=22, color="#1971c2",
))
# y=696–826. gaps of 40: 80–380, 420–760, 800–1120, 1160–1480
elements.extend(labeled_shape(
    "rectangle", "style", 80, 696, 300, 130,
    ["定全片风格与人物", "画风、光线、谁是谁", "短对白无图：从场景设定直接裁"],
    bg="#b2f2bb", font_size=16,
))
elements.extend(labeled_shape(
    "rectangle", "write", 420, 696, 340, 130,
    ["逐场写画面与声音", "模型只填镜头、表情、声音", "台词用原文拼上，抄的会拿掉"],
    bg="#b2f2bb", font_size=16,
))
elements.extend(labeled_shape(
    "rectangle", "plan", 800, 696, 320, 130,
    ["得到分场计划", "每场几秒、说什么、怎么接", "交给 H3 循环逐段生成"],
    bg="#96f2d7", font_size=16,
))
elements.extend(labeled_shape(
    "ellipse", "out", 1160, 696, 320, 130,
    ["成片", "多场接成一条连续视频", "本局是对白还是旁白一目了然"],
    bg="#ffd43b", font_size=16,
))

elements.append(arrow("m3", [(380, 761), (420, 761)]))
elements.append(arrow("m4", [(760, 761), (800, 761)]))
elements.append(arrow("m5", [(1120, 761), (1160, 761)]))

# Merge pack + story into style via the LEFT of pack/story (x=800),
# then the y=640 corridor (between story bottom 620 and section title 656),
# then down into style's top. x=800 is the 40px gap between diamond/extract
# band and pack (pack left=840). story/pack occupy x>=840, so x=800 is empty.
# y=640 is below story (620) and above the section-3 title (656).
elements.append(arrow("m_pack", [
    (840, 400),   # pack left, mid
    (800, 400),   # into the 800 corridor
    (800, 640),   # down, left of pack AND story
    (230, 640),   # left along the corridor to above style center
    (230, 696),   # down into style top
]))
elements.append(arrow("m_story", [
    (840, 560),
    (800, 560),
    (800, 640),   # joins the same corridor; stop at the join
]))

# Images lock identity: leave in_img at the BOTTOM, run LEFT in the
# page margin (x=40), then DOWN past extract (extract left=80), then
# RIGHT into style. Never enters diamond / pack / write.
elements.append(arrow("img_to_style", [
    (630, 268),   # in_img bottom center (460+340/2)
    (630, 292),   # into the 280–308 corridor under inputs
    (40, 292),    # left margin
    (40, 761),    # down the margin, left of extract/style
    (80, 761),    # into style left
], dash=True, color="#868e96"))
# Below the y=292 dashed run, above extract (y=360), in the left margin.
elements.append(text_only(
    "img_note", 48, 318, 220, 22,
    "有图则锁进全片和每一场",
    font_size=14, color="#868e96",
))

# legend
elements.extend(labeled_shape(
    "rectangle", "lg1", 80, 860, 160, 48, ["你给的"], bg="#fff3bf", font_size=16,
))
elements.extend(labeled_shape(
    "rectangle", "lg2", 260, 860, 160, 48, ["机器判断"], bg="#a5d8ff", font_size=16,
))
elements.extend(labeled_shape(
    "rectangle", "lg3", 440, 860, 160, 48, ["写出来的计划"], bg="#b2f2bb", font_size=16,
))
elements.append(text_only(
    "lg4", 640, 868, 840, 36,
    "对白路径保证：不漏词、不拆词、不改词。旁白路径没有这条保证。",
    font_size=16, color="#495057",
))

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
print(f"wrote {OUT}  elements={len(elements)}")


def _verify(doc):
    shapes = [
        e for e in doc["elements"]
        if e["type"] in ("rectangle", "diamond", "ellipse")
    ]
    free_text = [
        e for e in doc["elements"]
        if e["type"] == "text"
        and not str(e["id"]).endswith("__t")
        and e["id"] not in ("title", "subtitle")
    ]
    arrows = [e for e in doc["elements"] if e["type"] == "arrow"]
    hits = []

    def segs(a):
        x, y = a["x"], a["y"]
        pts = [[x + px, y + py] for px, py in a["points"]]
        return list(zip(pts, pts[1:]))

    def aabb(e):
        return e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]

    def inside(x, y, box, eps=1.0):
        l, t, r, b = box
        return l + eps < x < r - eps and t + eps < y < b - eps

    obstacles = shapes + free_text
    for a in arrows:
        for p1, p2 in segs(a):
            for i in range(1, 24):
                k = i / 24
                x = p1[0] + (p2[0] - p1[0]) * k
                y = p1[1] + (p2[1] - p1[1]) * k
                for s in obstacles:
                    if inside(x, y, aabb(s)):
                        hits.append((a["id"], s["id"]))
                        break
    unique = sorted(set(hits))
    if unique:
        print("VERIFY HITS", unique)
    else:
        print("VERIFY ok: no arrow through shapes or free text")


_verify(doc)
