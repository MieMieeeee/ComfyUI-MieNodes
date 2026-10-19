# 分镜资产 + 机械 Plan 构建 实施计划（待第三方评审）

> 版本 2026-10-19。前置依赖：资产库 V2（`docs/PLAN_ASSET_LIBRARY.md`，已落地）、
> H3 Loop 生成器确定性化系列修复（`9a5d1e6`/`0cc25d9`/`cfa1487`，已落地）。
> 本文档描述的是**新增第二条 plan 生产轨道**，不改动现有 LLM 生成轨道。

## 0. 一句话

把验证过的"分镜"固化成第三类资产；角色/场景资产嵌入分镜槽位后，由**纯代码节点**
（无 LLM connector）直接拼装出与现有生成器同 schema 的 `plan_json`，喂给现有的
`MiniMaxH3ChainPlanModern`。同分镜 + 同资产 ⇒ **字节级相同的 plan**。

## 1. 要解决的事

现状（2026-10-18 组合矩阵实测）：资产 roster 模式下，LLM 生成器做的事本质是
"带噪音的模板填充"——9 个组合的终稿结构完全同构，差异只有名字/描述/台词；
而噪音侧产出过整类缺陷（scene_01 漏 `<Picture 3>` 定义、`Setting:` 中文原句
残留、fur 措辞、镜头指令自相矛盾、facts 空回复导致空间钉全灭）。同时：

- 一次生成 6–9 个串行 LLM 调用，~75% 耗时是模型思考（延迟画像实测）；
- Token Plan 配额耗尽（HTTP 402）会整条管线停摆（10-18 两次实测）；
- 同输入重跑 plan 不确定性（LLM 温度/重试），生产复现只能靠"存下 plan 手改"。

分镜资产轨道把**重复生产**（同剧本换角色/场景、批量重摇、多语言版本）从
LLM 管线里拿出来：零 token、秒级、字节可复现、结构缺陷类别整类消灭。

## 2. 不做什么

- **不动 LLM 生成轨道**：新剧本/散文输入/自动润色仍走 V1/V2 生成器，双轨并存，
  喂同一个 PlanModern；
- **不做"从剧本自动生成分镜资产"的 LLM 工具**（第一版只做手工编写 + 从金样
  plan 一次性反推的脚本工具，见 §8）；
- **不做分镜的图形化编辑器**（分镜就是 JSON 文件 + Load 节点下拉）；
- **不改渲染链**（LoopStart/Preflight/参考图接线/音频接线全部原样）；
- **不给渲染层的 H3 缺陷兜底**（独白镜丢说话人等仍按既有"重摇配方"处理）。

## 3. 磁盘格式

### 3.1 分镜资产（新 kind：`storyboard`）

目录：`<library>/storyboards/<名字>/storyboard.json`（沿用每资产一目录的布局；
无 images/audio 子目录，纯 JSON）：

```jsonc
{
  "name": "客厅板栗对话",
  "kind": "storyboard",
  "version": 1,                     // schema 版本，构建节点拒绝不认识的 major
  "language": "zh",                 // 台词语言（写进 <d>[zh/en/...] 标签）
  "camera_discipline": "locked",    // locked | no_closeups | auto
  "pacing": "normal",
  "seed_policy": "per_scene_increment",
  "prompt_prefix": "A cozy cottage living room ... (英文风格块，整段模板常量)",
  "cast": {                          // speaker 名 → 角色槽位 的声明式绑定
    "孙女": {"character_slot": 1},   // 说话名可以 ≠ 资产名（见 §6.3）
    "爷爷": {"character_slot": 2}
  },
  "scene_slots": {"客厅": 1},         // 可选：剧本内场景名 → scene 槽位
  "setting_line": "Grandpa and the girl sit on the sofa, talking.",  // 英文 Setting 行（可空=不写）
  "shots": [                          // 预切分镜；缺省则由台词行数走 minimal-cut packing
    {"lines": [{"speaker": "孙女", "text": "爷爷，我一直有个心理阴影。"},
                {"speaker": "爷爷", "text": "是什么呢。"}]},
    {"lines": [{"speaker": "孙女", "text": "小时候我还没上小学。"},
                {"speaker": "孙女", "text": "咱家院子里有几只小刺猬。"}]}
    // ... 每镜 ≤2 句（H3 窗口约束，构建时机械校验）
  ],
  "spatial_layout": {"孙女": "left of frame", "爷爷": "right of frame"},
  "ambience": {"scene_slot:1": "Quiet room tone with ..."}  // 可选：按场景槽位覆盖氛围句
}
```

约束（构建时校验，违规即报错，不猜测）：
- `shots` 展开后台词顺序 = 一个线性剧本；同一镜内 ≤2 句；
- `cast` 必须覆盖全部 speaker；未绑定的 speaker 名报错（对齐"空输入即响亮报错"先例）；
- 台词文本视为**锁定数据**：构建节点逐字写入 `<d>` 块，永不翻译/改写。

### 3.2 场景资产追加可选字段 `ambience`

`character.json`/`scene.json` 之外，scene 资产增加可选 `"ambience": "<英文氛围句>"`。
分镜 `ambience` 缺省时回退到场景资产的该字段；两者都空时用节点内置的按
`camera_discipline` + 场景描述生成的固定句（不是 LLM，是三选一常量表）。
旧场景资产无此字段 ⇒ 照常加载（向后兼容，不迁移）。

## 4. 节点

### 4.1 `LoadStoryboardAsset`（读分镜）

- 输入：`library_dir`（STRING，空=默认根）、`name`（combo/STRING，同 LoadAssetV2 形态）；
- 输出：`MIE_STORYBOARD`（新类型）+ `status`（STRING）；
- 校验：schema version、JSON 可解析；加载失败响亮报错（先例：mp3+wav 歧义）。

### 4.2 `BuildPlanFromStoryboard`（机械构建，核心节点）

- 输入：
  - `storyboard`（MIE_STORYBOARD，必需）
  - `character_1..3`、`scene_1`（MIE_ASSET，与 V2 生成器同款 socket 形态；槽位顺序 = `<Picture N>`）
  - `seed`（INT）、`enhance`/润色等 LLM 相关控件**一律没有**（本轨道无 LLM）
- 输出：`plan_json`（STRING，与生成器同 schema）、`summary`（STRING，含
  独白镜风险建议——直接调用现有 `_monologue_scene_hint`）、`images`（IMAGE 批，
  与 V2 的 assets→Slots 接线兼容）
- `FUNCTION` 内部**纯同步代码**，无 connector、无网络、无文件写入（除库读取）。

### 4.3 保存路径

`SaveStoryboardAsset`：输入 JSON 文本 + 名字，写 `storyboards/<名字>/storyboard.json`
（带 version/字段校验；同名覆盖走与现有 Save 资产相同的确认语义）。
第一版可以只提供"粘贴 JSON 保存"，不做逐字段表单。

## 5. 代码落点与复用清单

关键原则：**每一段 plan 内容都复用现有确定性函数**，不新写拼装逻辑。

| plan 区段 | 复用的现有函数（nodes/llm/minimax_h3_loop_prompts.py） |
|---|---|
| subject_definitions 绑定行 | `repair_label_pairings` 的规范形 `<Subject k> -> <Picture m>: about` |
| 裸名别名 | `identity_pin_lines`（说话名→资产描述） |
| summary | `install_shot_speech`（含可见名单双格式） |
| [Shot 1] 散文 | `_shot1_coverage` |
| 空间钉 | `build_spatial_layout_directive` + 分镜自带 layout（不再需要 facts） |
| 台词块 + 声纹 | `assemble_dialogue_line_blocks` + 资产 `voice` 字段 + `build_speaker_id_map` |
| retention | 资产描述特征句 + `_POSE_LOCK_NOTE`（"fur or hair as the subject actually has"） |
| packing（分镜未预切时） | `distribute_lines_to_scenes`（minimal-cut） |
| 校验 | `validate_dialogue_invariantity`、`validate_label_policy`、`validate_spatial_layout_invariant` 原样跑终稿 |

**落点争议点（请评审重点看）**：`nodes/assets/` 的红线是"不 import `nodes/llm/`"
（资产库计划书 §边界）。但上表函数全在 `nodes/llm/`。两个选项：
- A. 新节点放 `nodes/llm/storyboard_builder.py`（LLM 模块里放一个不调 LLM 的节点）；
- B. 把纯拼装函数下沉到中立模块（如 `core/plan_assembly.py`），`nodes/llm` 与
  新节点共同 import——动 28 处调用点，风险高于收益。
- **推荐 A**（零迁移；模块名是历史包袱，早已名不副实，见 H3LoopPromptEnhancer 的 docstring 先例）。

## 6. 关键语义决策

### 6.1 双轨关系
LLM 生成器负责**创作**（新剧本、非规范输入、润色）；分镜资产负责**生产**（复用
剧本换皮、批量重摇、多语言版本）。两轨输出同 schema，下游零感知。

### 6.2 字节可复现
同分镜 + 同资产 + 同 seed ⇒ 逐字节相同 plan（无 LLM、无时间戳、无随机序）。
差异只允许来自：用户改分镜/资产/seed。

### 6.3 说话名 ≠ 资产名（原生支持）
分镜 `cast` 声明式绑定（`孙女 → character_slot 1`），plan 内以**说话名**面向 H3
（对齐 2026-10-15 矩阵实测结论：终稿用对话名，资产名不必出现）。

### 6.4 Setting 行
分镜模板显式携带英文 `setting_line`（可空）。**机械轨道不产生中文原句残留**——
这是对 LLM 轨道唯一未根除警告项的正面规避。

### 6.5 独白镜
构建节点在 summary 中同样调用 `_monologue_scene_hint` 提示重摇配方；plan 结构
本身不因独白而改（渲染层缺陷，两轨道同等暴露）。

## 7. 测试

1. **金样等价（核心验收）**：`tests/fixtures/plan_english_kitten_grandpa.json`
   （prompt(15) 金样）反推出第一张分镜资产；离线单测断言 `BuildPlanFromStoryboard`
   的输出与金样在**结构上逐项相等**（台词/绑定/别名/summary/钉/retention 后缀/
   违禁标记为零）。允许 prose 措辞差异的区段（retention 枚举句、soundscape）按
   "锚点集合相等"断言（列出必须命中的实体词），不逐字比较。
2. 槽位校验：speaker 未绑定、>2 句/镜、缺场景槽、schema version 不符 ⇒ 各自响亮报错。
3. 说话名≠资产名（孙女/爷爷 vs 小猫妹妹/猫爷爷）：终稿用说话名，槽位锚 `<Picture i>` 齐全。
4. 资产兼容：无 `ambience` 字段的旧场景资产照常；character.json 缺 voice ⇒ 报错
  （对齐"空输入即响亮报错"）。
5. 复用矩阵校验器：`tests/test_h3_loop_live_combo_matrix.py` 的 `verify_plan`
  提为可离线 import 的校验函数，对机械 plan 跑同一套硬检查（台词逐字/归属/
  在场/钉/资产/违禁）。
6. 真机 E2E（可选，渲染验收）：机械 plan 走 PlanModern→LoopStart 渲一遍 4 幕短剧。

## 8. 实施顺序

1. 场景资产 `ambience` 可选字段（向后兼容 + 3 条测试）；
2. 分镜 schema + `LoadStoryboardAsset` + `SaveStoryboardAsset`；
3. `BuildPlanFromStoryboard`（§5 复用表逐行接）+ 金样反推脚本
   `scripts/harvest_storyboard_from_plan.py`（从一份 plan JSON 抽模板，人工确认后入库）；
4. 测试 §7.1–§7.5 全绿；提交；
5. （评审通过后）真机渲染验收 + 双机同步。

## 9. 已定决策（沿袭既有裁决，不再开放）

- 锁定台词永不翻译/改写（2026-09-22 契约）；
- 节点代码零关键词分支/零硬编码补句（2026-09-27 裁决）——本轨道天然满足；
- UI 文案 中文 · English 顺序（2026-10-11 裁决）；
- `Setting:` 中文残留规避为显式英文字段（本文 §6.4）；
- 双轨并存，LLM 轨道不动。

## 10. 给评审人的问题清单

1. §5 落点争议 A vs B（新节点进 `nodes/llm/` vs 下沉中立模块）；
2. 分镜 schema 是否需要 `version` 之外的向后兼容策略（读旧版本分镜时：拒绝
   还是迁移？当前方案：拒绝 + 报错说明字段变更）；
3. `shots` 预切分 vs 全缺省走 minimal-cut packing 的边界（当前方案：两者皆可，
   预切分优先；是否应锁死为"必须预切"以最大化可复现性？）；
4. 分镜资产是否需要进入 `AssetsToRoster`/`AssetToSlots` 生态（当前方案：**不进**
   ——分镜不产生参考图/音频槽，走独立的 Load→Build 接线；评审确认无隐藏耦合）；
5. `summary` 输出是否需要与 LLM 轨道逐字同构（当前方案：结构同构、来源标注
   "mechanical"，其余文案一致）。

## 11. 验收

- `BuildPlanFromStoryboard` 对金样分镜产出零错误 plan（§7.1 断言全绿）；
- 矩阵校验器 9 项硬检查全过、零警告（无 Setting 残留、无 fur、无违禁标记）；
- 全量离线测试通过；双机同步后老工作流零影响（新节点纯增量注册）。
