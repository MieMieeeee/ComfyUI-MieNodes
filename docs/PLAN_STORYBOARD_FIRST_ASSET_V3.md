# 第一张真实分镜资产 + V3 零 LLM 生成器 实施计划（待第三方评审）

> 版本 2026-10-19。前置：分镜轨道已合并（`283148f`→`2f30396`→`5a08048`，
> Load/Save/Build 三节点 + 连号校验 + 金样等价测试，1675 测试绿）。
> 本计划补齐两块缺口：库里有真实可用的分镜资产；生成器家族出现 V3 门面。

## 0. 一句话

从 prompt(15) 金样反推出**第一张真实分镜资产**（中英双份）入库，并新增
**`MiniMaxH3LoopPromptGeneratorV3`**——V2 同形插座、五输出一致、内部走
`assemble_plan` 的零 LLM 门面，让现有 V2 工作流**换一个节点**即可切到分镜轨道。

## 1. 现状与差距

- `storyboards/` 目录在本地与 LAN 两个部署库中**都不存在**：轨道从未端到端跑过，
  Load 节点下拉是空的；
- 组装入口是独立的 `BuildPlanFromStoryboard`：插座是**固定 character_1..8 +
  scene_1**（无自增长）、输出四根（无 `board_kind`）——与 V2 生成器的使用习惯
  （socket 自增长、五输出、同 CATEGORY）不一致，切换轨道要重接线；
  V2 工作流的肌肉记忆在 V3 上应当原样成立。

## 2. 不做什么

- **不动 V1/V2 生成器**与 Build 节点的行为（Build 保留，见 §9）；
- **不给 V3 加任何 LLM 控件**（无 connector / enhance / temperature / max_tokens）；
- **不做分镜编辑器**（分镜仍是 JSON 文件 + Save 节点粘贴）；
- **不在 V3 上放 pacing / scene_count / language / camera 控件**——这些真相
  只住在分镜资产里，节点加覆写控件等于制造第二个真值来源（见 §9 决策）；
- **不做多场景/多分镜合并**（v1 仍是一个分镜 + 一个场景）。

## 3. 第一张真实分镜资产（板栗对话，中英双份）

### 3.1 来源与内容

从 `tests/fixtures/plan_english_kitten_grandpa.json`（prompt(15) 金样，4 幕 7 句，
小猫妹妹 + 猫爷爷 + 客厅）反推，用现有 `scripts/harvest_storyboard_from_plan.py`
抽取 shots/speakers，人工补齐以下字段后经 `SaveStoryboardAsset` 入库：

| 字段 | 值 |
|---|---|
| `cast` | 小猫妹妹→slot 1，猫爷爷→slot 2（连号 1..2） |
| `language` | en 版 = `en`（台词逐字来自金样）；zh 版 = `zh`（台词用板栗中文原句，7 句照抄） |
| `prompt_prefix` | 金样 prefix 原文（英文，两版共用） |
| `setting_line` | 金样 Setting 的英文对应句（手写一句，如 "Grandpa and the kitten sit on the living-room sofa, talking."；金样里的中文原句**不进**此字段） |
| `spatial_layout` | 小猫妹妹=left of frame，猫爷爷=right of frame |
| `ambience` | 一句英文（手写，如金样 soundscape 的首句） |
| 其余 | `version:1, camera_discipline:locked, pacing:normal, seed_policy:per_scene_increment` |

- **en 版**是等价性证明载体：台词/结构与金样同源，跑 §6.1 断言。
- **zh 版**是真实渲染载体：中文台词 + `language: zh`，给真机验收用。

### 3.2 存放（三个位置，各有职责）

1. `tests/fixtures/storyboard_banli_en.json` + `storyboard_banli_zh.json`——
   进仓库，测试直接加载，不依赖部署库；
2. 本地部署库 `asset_library/storyboards/板栗对话-EN|ZH/storyboard.json`——
   用 Save 节点（粘贴 JSON）写入，供画布下拉真实使用；
3. LAN 部署库同步骤手写一份（库是 per-machine 的，同步脚本不搬运
   asset_library——这是资产库 V2 的既定设计，不做改变）。

## 4. V3 节点：`MiniMaxH3LoopPromptGeneratorV3`

### 4.1 形态（对齐 V2 的使用习惯）

- **required**：`storyboard`（MIE_STORYBOARD）、`seed`（INT）；
- **optional**：`character_1..9`（MIE_ASSET，**JS 自增长**，同 V2：接线后长出下一
  号插座）、`scene_1`——插座语义与 Build 一致：不压实、cast 槽连号 1..N、
  缺线/跳号由 `assemble_plan` 响亮报错；
- **输出**：`(plan_json, summary, board_kind, images, assets)`——五根，名字、
  类型、顺序与 V2 完全一致；`board_kind` 恒为 `"dialogue"`（分镜 v1 只有对话轨）；
- `summary` 首行含 `mechanical · zero-LLM` 与分镜名，续接布局补全提示、
  独白镜重摇建议、尺寸不一致提示（复用 assemble_plan 返回的 summary_lines）；
- **UI**：`零 LLM 分镜生成器 V3 · Loop Plan Generator V3 (Zero LLM)`，
  CATEGORY 与 V2 相同；
- **IS_CHANGED**：分镜内容哈希 + 各槽位资产内容哈希（与 LoadAssetV2 的
  内容失效机制一致——改了 storyboard.json 或换绑资产必须重跑）。

### 4.2 实现

- 落点：`nodes/llm/storyboard_builder.py` 内新增类（零 LLM 面板集中一文件），
  `build_v3()` 是 `assemble_plan` 的薄包装 + roster/images 组装（逻辑照抄
  `BuildPlanFromStoryboard.build` 的尾部，抽成共享私有函数，两节点零重复）；
- `__init__.py` 注册 + 双语显示名；
- **JS 自增长**：找到为 `MiniMaxH3LoopPromptGeneratorV2` 实现 character_N
  插座自增长的 JS（`js/` 目录，按节点名匹配），把 V3 的节点名加进同一匹配表——
  这是本计划最容易被漏的集成点，漏了则 V3 的插座不会自动长出。

### 4.3 与 Build 节点的关系

两者共存：Build 是通用组装入口（分镜轨道的"裸"形态），V3 是生成器家族门面
（V2 工作流的置换件）。内部共享同一 `assemble_plan`，行为逐字节一致（§6.2 锁定）。

## 5. 代码落点

| 内容 | 位置 |
|---|---|
| V3 类 + 共享包装 | `nodes/llm/storyboard_builder.py` |
| 注册/显示名 | `__init__.py`（纯增量） |
| JS 插座自增长 | `js/` 中 V2 自增长实现处加 V3 节点名 |
| 两份分镜资产 fixture | `tests/fixtures/storyboard_banli_{en,zh}.json` |
| 部署库入库 | SaveStoryboardAsset 节点（本地 + LAN 各手工一次） |

不改动：V1/V2、`nodes/assets/`、渲染链、`assemble_plan` 本体。

## 6. 测试

1. **en 分镜 == 金样结构**：用 `storyboard_banli_en.json` 走 V3 构建，断言沿用
   `tests/test_storyboard_golden_equivalence.py` 的全部 13 条（台词逐字/绑定一致/
   六段头序/Setting==手写英文/retention==固定句/标签==language/(Sn)==slot/
   字节可复现/无违禁标记）；
2. **V3 == Build 逐字节**：同一 storyboard + 同槽位 + 同 seed，两节点 `plan_json`
   完全相等；
3. **zh 分镜可渲染结构**：7 句中文台词逐字、`[Chinese]` 标签、其余结构断言同上；
4. **错误透传**：gap-cast / cast 槽空线 / 空 voice / 双字段（shots+lines）经 V3
   报错文案与 Build 一致；
5. **注册与 UI**：V3 在 NODE_CLASS_MAPPINGS；显示名双语（断言含 `·` 与两侧文字）；
6. **IS_CHANGED**：改 storyboard.json 字节后哈希变化（纯函数级测试）。
   JS 无测试基建 → 真机验证项（§8）。

## 7. 实施顺序

1. 两份分镜资产：harvest + 人工字段 + fixture 落盘 + 测试 §6.1/§6.3；
2. V3 类 + 共享包装 + 注册 + 显示名，测试 §6.2/§6.4/§6.5/§6.6；
3. JS 自增长接入 V3 节点名；
4. 全量离线测试绿 → 提交；
5. 双机同步 + 本地真重启（三验：新 banner / netstat pid / 探活）；
6. 真机验证：画布上 V3 出现在生成器分类、插座自增长、下拉选到「板栗对话-ZH」、
   V3 → PlanModern → LoopStart 渲一遍 4 幕（渲染质量只记录，不回头改组装）；
   LAN 同步后同样点一遍。

## 8. 验收

- 两个部署库的 Load 下拉出现两张板栗分镜；en 版经 V3 构建通过全部金样断言；
- V3 与 Build 输出逐字节一致；zh 版结构断言全绿；
- 画布实测：V2 工作流把 V2 节点换成 V3（改接 storyboard 一根线）即可出 plan，
  插座自增长与 V2 手感一致；
- 全量离线测试通过；旧工作流零影响（纯增量注册）。

## 9. 已定决策与给评审人的问题

已定（沿袭既有裁决）：
- 零 LLM 红线、台词逐字、CJK 禁令、连号 1..N、双语文案——全部继承
  PLAN_STORYBOARD_ASSET §9 与三轮 review 基线；
- V3 不带 pacing/scene_count/language/camera 控件：真相只在分镜资产里。

评审问题：
1. Build 与 V3 长期共存，还是 V3 稳定后把 Build 转为 hidden（保留加载旧工作流
   能力）？当前方案：共存可见；
2. `board_kind` 恒为 `"dialogue"` 是否够用（V2 下游有没有真消费该值的节点？
   若无消费者，随便一个稳定字面量都行）；
3. en 版分镜的 `setting_line` 允许与金样 Setting 意译（金样是中文原句），等价性
   断言只锁「== 手写字段值」而非「== 金样 Setting」——确认接受；
4. 部署库入库是手工步骤（Save 节点粘贴），要不要在仓库加一个
   `scripts/install_storyboards_to_library.py` 一键脚本？当前方案：不加，两份
   资产手工入两次即可，避免 per-machine 库出现第二条写入路径。
