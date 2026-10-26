# 白披风骑士猫的独白 · 实跑验收（2026-10-26）

用分镜资产 v2 契约 + `白披风骑士猫` 角色资产跑通的第一条**用户自有角色**的独白视频。

## 结果

- 成片：`白披风骑士猫的独白.mp4`（8.0 秒，960×544，H3 ref2va 链路）
- 台词：**ASR 逐字命中**，三句顺序正确、无增删、无漏字——
  「拿着剑我就没法拥抱你，放下剑我就没法保护你，所以你为什么不主动抱过来呢?」
- 帧检（见同目录 PNG）：
  - 开场：猫骑士身份完整（灰虎斑、粉蝴蝶结、银甲、奶白披风），剑**唯一**竖持于一只爪
  - 中段：剑已放低、剑尖朝下贴腿边；**全场无第二把剑**
  - 结尾：另一只爪前伸、掌心朝上，姿态与台词呼应

## 接线

```
LoadStoryboardAsset(白披风骑士猫的独白)
  → MiniMaxH3LoopPromptGeneratorV3    ← 零 LLM
  → AssetToSlots
  → MiniMaxH3ReferenceToVideo
  → MiniMaxH3ChainPlanModern → LoopStart → … → Assemble
```

| 槽位 | 资产 |
|---|---|
| character_1 | 白披风骑士猫（image 01.png + 参考音 01.wav） |
| scene_1 | 毛绒客厅（写实渲染、暖窗光，与角色的写实毛发质感匹配） |

驱动脚本 `_live_knight_cat.py`，基准图 `C:\Users\administered\Downloads\c1.json`。
run_name `sb_202610262210_knightcat`（ASCII only），seed 202610262210。

## 分镜资产里用到的规则（2026-10-26 实跑回流那批）

- **单段**：8 秒内容全部装进一个生成片段，规避「续生成片段单人独白」触发形态。
- **observer 视角**：独白不用 first_person（会画出观众的手）。
- **setting 只留光锚**：「午后柔和的自然光从一侧斜照 + 暖色环境补光」，不写房型/墙面/家具——房型与陈设交给接入的场景资产，否则 setting 文案会盖过场景参考图。实测场景资产**赢**，绿沙发/落地灯/书架完整保留。
- **道具唯一性**：剑在 visible_state 与每条 direction 里都写成 "the only sword in the scene"；实测**无复制品**。
- **单手原则**：握剑的爪全程握剑，另一只爪做唯一的精细动作（摊掌），两者不同时发力。
- **不追求手指细节**：按用户裁定，没有让 H3 做逐根手指的动作。
- **台词数字口语化**、**voice 与接线角色一致且无尾标点**：voice 写的是角色资产自己的声线规格
  （young female, soft sweet voice…），且不带结尾标点，避免拼出 `…: <d>` 的破行。

## 复现

```text
python scripts/validate_storyboard_asset.py docs/storyboard-assets/白披风骑士猫的独白.json
python scripts/import_storyboard_assets.py docs/storyboard-assets/白披风骑士猫的独白.json --overwrite
python _live_knight_cat.py
```
