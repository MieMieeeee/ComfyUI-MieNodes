# H3 全参考（full-reference）契约对齐计划（v2，已吸收第三方评审）

> 版本 v2，2026-11-09。**v1 经第三方评审；本版逐条吸收评审结论。**
> 上游依据文件：`C:\Users\administered\Downloads\minimax-moody.txt`
> （《Full-Reference Mode Rewrite Output Format Guide》，374 行）。
> 计划改动面横跨四个仓库，本文件在 `ComfyUI-MieNodes`（本仓），代码真源在
> `F:/ComfyUI-Mie-Prompt` / `F:/ComfyUI-Mie-Craft` / `F:/ComfyUI-Mie-H3-Engine`。
>
> **v2 状态声明**：评审裁定的三件优先项（F7+F6、F5、F4）**已实施并全套件通过**
> （`mie_craft` `a07fba8`、`mie_prompt` `b7fc1ea`、本仓同日 commit；
> `python -m pytest tests/ -q --ignore=docs/qa` = **2055 passed, 10 skipped**，
> 基线 2051 + 新增 4 测试）。F3 按评审结论**降级为非破坏性**，先改本计划再排期。
> F2 仍等用户 Q9 裁定。真机渲染矩阵（§5.2-3）**尚未跑**，是本计划剩余的主要验收。

### v1→v2 评审改判一览

| v1 的说法 | 评审结论 | v2 处置 |
|---|---|---|
| 六段名是 tokenizer 注册 token（引 `h3_prompt_schema_core.mjs:24-28`） | **v1 引证错误**：`:24-28` 注册的是 `<d>`/`<|cutoff|>`/lyrics/caption 等特殊 token；段名只是 `:1-14` 的普通字段清单，`detectH3Mode` 是**编辑器启发式**，不是模型侧门 | §3-F3 改写：破坏性论证不成立 |
| F3 是破坏性变更，须升 v3 或新增节点 | 不破坏：无参考图时字节不变即可 | F3 降级为非破坏性增强，覆盖 v2 现节点 |
| F1 建议：环境槽改 `weak_reference` | **驳回**：官方 `Ref2V Basic/Studio` 样例对目的地 Picture 写的就是 `reference - ... as the destination` | 环境槽走路线乙保持 `reference`；仅视频槽走路线甲改 `attribute_transfer` |
| F5 例句 "Character 2 remains seated with the parcel…" | **驳回**：这是发明姿势；`execution.visible_state` 是**开场状态**（`contract.py:657` 事件循环前写一次），不得复用为切点末状态 | 固定承接句：只用资产已知状态 + 类目级不重置规则 |
| 时间戳禁令归入"已一致" | **归错类**：上游 `:240` 要求 full-reference 后续镜写 `At MM:SS.mmm`；我们的禁令是 loop 产品决定 | 移入 §7 产品决定 |
| "仅仅存在视频/音频不产生任务类型 = 已一致" | **口径过宽**：`:2797` 强制 `[reference generation]` 并禁 `[video continuation + ...]`，这是 F9 的能力边界 | §2 该行改为指向 F9 |
| 开篇 "2 处破坏性变更" | 与 §4 表格自相矛盾（只有 F3 标了破坏性） | v2：**零破坏性变更** |

---

## 0. 一句话

上游那份 full-reference 输出格式指南，和我们的 **H3 提示词生成节点（loop 轨道）**
与 **剧本资产生成节点（v2 编译轨）**逐条比对后，找出 **7 处口径偏差 + 3 处待实测项**；
评审后裁定：**零破坏性变更**——三件优先项（视频槽 `attribute_transfer` + 切点承接句 +
summary 参考关系句）已实施，F3 降级为"仅多图接入时切六段式"的非破坏性增强，
F2 等音频路由裁决。

---

## 1. 背景与判定口径

### 1.1 这份上游文件是什么

它不是新模型能力，是 **MiniMax H3 full-reference 模式的重写输出格式规范**——
和我们 `mie_prompt/prompts/h3/system_reference.txt` 的上游同源（那份派生自
`VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`，见 `mie_prompt/prompts/h3/UPSTREAM.md`）。
所以它的价值**不在模型知识**，而在**它把几件事写死了，而我们的模板有比它松的地方**。

它定义/收紧的东西（与本次评审相关的部分）：

| 上游位置 | 条款 |
|---|---|
| `:7` | **Video Continuity Rule**：默认单条连续时间线；不主动引入切点/跳接/视角切换 |
| `:36-45` | 四类标签：`<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` |
| `:51-92` | `<Subject N>` 是"可复用的可见内容单元"，可由多个参考资产定义；`<Picture N>` 是"具体帧锚/构图锚" |
| `:110-126` | `<Audio N>` 是一等参考：可复制（copy）也可只引用（reference），且要绑定目标说话人 |
| `:141-165` | `summary` 必须是**任务类型前缀**开头，六种类型可 `+` 组合；"仅仅存在视频/音频不自动产生对应任务类型" |
| `:178-188` | **视觉保留关系只有四档**：`fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference` |
| `:190` | **连续性不是保留关系**：姿态/位置/动作/构图随时间自然变化 ≠ `partially_preserved` |
| `:210-219` | **音频保留关系四档**：`fully_copy` / `partially_copy` / `reference` / `weak_reference` |
| `:240` | full-reference 模式下，后续 `[Shot N]` 要求带 `At MM:SS.mmm` 时间戳（**我们不吸纳**，见 §7） |
| `:251-259` | full-reference 模式差异表：主字段换成 `detailed_description`；风格前置 1-2 句写在 `[Shot 1]` **之前**；参考标签要写进正文；音频关系要写进对应段 |
| `:340-374` | 完整样例（六段齐全，含 `<Audio 1>` 定义与音频保留行） |

### 1.2 我们的两条轨道

| 轨道 | 入口 | 代码真源 | 说明 |
|---|---|---|---|
| **A. H3 提示词生成节点**（loop） | `nodes/llm/minimax_h3_loop_prompt_generator.py` | `mie_craft/h3_loop_prompts.py` + `mie_prompt/prompts/h3_loop/*.txt` | 本仓只是薄壳；ref2va 模式出六段式 |
| **B. 剧本资产生成节点** | `nodes/llm/minimax_h3_storyboard_generator.py` | `mie_craft/script/contract.py`（v2 编译）+ `mie_prompt/prompts/h3_storyboard/system_asset.txt`（资产契约） | 零 LLM 编译成 H3 plan |

**轨道 B 的编译产物形态**：`contract.py:611` 硬编码
`integrated_multimodal_description: [Shot 1] ...` + `overall_soundscape:` +
`non_diegetic_music:`，即**三段式**；但正文里有 `<Picture N>` 标签
（`contract.py:620`、`:635`），而接入侧确实把多张图喂进 H3 的 ref 位
（`builder.py:1610-1671` roster → `Asset→Slots` → `picture_i` → `ref_image_0..N-1`）。
**轨道 A 的 ref2va 才是六段式。** 两条轨道对同一份参考素材给出两种形状，
仍是本次最大的结构性问题（处置见 F3，v2 已降级为非破坏性）。

### 1.3 判定口径（评审已认可，v2 沿用）

- **上游条款 = 硬约束**：凡是上游明文规定、且我们产出与之冲突的，算偏差。
- **上游条款 = 参考**：上游没写但我们自己定了的，只要自洽且有实测支撑，**不算偏差**。
- **引擎现状 ≠ 上游**：MiniMax 自家发布的样例工作流也可能与指南不一致（见 F1），
  这类冲突**按样例裁决**（评审裁定：环境槽随样例）。
- **我们自己的产品决定优先于上游**：如"正文由节点确定性生成、模型写的不保留"，
  这是既定架构（"确定性归一化是基线""LLM 判断、代码保证"），不因上游写法不同而推翻。

---

## 2. 已核对、**确认不是问题**的条款

| 上游条款 | 我们的现状 | 结论 |
|---|---|---|
| 六段头序、`<Subject N>`↔`<Picture M>` 同句绑定 | `ref2v_addendum.txt:6-11, :22-34` | 已一致 |
| identity 槽不另写 `<Picture N>` 保留行 | `ref2v_addendum.txt:41-58` | 已一致（与上游 `:96-104` 同款） |
| 角色说话写 `<Subject N> (Sx)` | 我们写"名字 + (S1)"，绑定关系放 `subject_definitions` | 不是硬偏差（见 F10） |
| 说话人 ID 全局固定、不重编号 | `ref2v_addendum.txt:85-103` + 节点侧 speaker map | 已一致 |
| "仅仅存在视频/音频不自动产生对应任务类型" | loop 轨恒用 `[reference generation]` 且禁 `[video continuation + ...]`（`h3_loop_prompts.py:2797`） | **这是 F9 的能力边界**（loop 没有源视频编辑/续写能力），不是对该条款的实现——上游这句话本身我们不违反 |
| 引擎已支持调度参考的 `<Audio N>` 编号 | `chain_nodes.py:3775, :3789` | 已一致（引擎侧） |
| `(S<n>)` 与音频位的对应 | 逐路径核对：`audio_mode=reference` 时每角色必须接资产（`builder.py:797-799` 抛错），`generated` 时无音频映射 | 无错位，**不是缺陷** |

**v2 勘误**（v1 在此表的两处归错类，评审指出）：

1. **时间戳禁令不在本表**。上游 `:240` 对 full-reference 后续镜**要求**时间戳；
   我们的禁令（`shot_system_addendum.txt:59`）是 loop 链式产品的**自主决定**
   （链路靠 carried overlap 续接，时间戳没有意义），已移入 §7。
   轨道 B 的 `camera_cuts` 内部切点是第三件事（一个生成片段内的换景别），
   loop 禁令**不得**外推到它上面。
2. 引擎 web 侧 `h3_prompt_schema_core.mjs:24-28` 注册的精确 token 是
   `<d>` / `</d>` / `<|cutoff|>` / `<|lyrics_start|>` / `<|lyrics_end|>` /
   `<|caption_start|>` / `<|caption_end|>`——**不含段名**。段名只是 `:1-14`
   的普通字段清单；`detectH3Mode`（`:117-125`）是编辑器启发式。v1 曾据此
   论证"段名是模型词表"，**错误**，F3 的破坏性论证随之失效。

---

## 3. 偏差清单（v2：标注评审处置与实施状态）

### F1〔P1→已实施〕保留关系词表跨类使用 —— 拆成两半分别裁决

**上游**：视觉标签（`<Subject N>`/`<Picture N>`/`<Video N>`）只能用
`fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference`（`:178-188`）；
`reference` 属**音频档**（`:210-219`）。

**评审裁定（Q1）**：F1 拆开——

- **环境/目的地 Picture 槽 → 路线乙**：保持 `reference - use as the destination
  environment; do not copy a final pose.` 不动。依据：官方
  `Ref2V Basic / Studio - MiniMax H3 0.6.json` 的 retention 块对目的地 Picture
  写的正是 `<Picture 2>: reference - use its potting table, ... as the destination;
  do not copy the final pose before the action reaches it.`——上游生态内部不自洽，
  样例即惯例。v1 建议的 `weak_reference`（语义更弱，可能放松环境约束）**驳回**。
- **视频动作槽 → 路线甲**：`reference - choreography and timing` 是把音频档词
  用在视觉标签上，且引擎编辑器按标签类型选补全词表（`h3_prompt_completion_core.mjs:145-163`）
  补不出这组词。改 `attribute_transfer`（与 F7 合并实施）。

**实施状态**：✅ 已落地（`mie_craft a07fba8` + `mie_prompt b7fc1ea`）——
指令（`h3_loop_prompts.py` VIDEO MOTION REFERENCE 块）、回退句
（`build_fallback_shot_lines` 的 Video 分支）、addendum（`ref2v_addendum.txt`
Video motion slots 规则）三处同改；环境槽两处原样保留。
回归测试：`test_video_motion_directive_and_fallback_use_attribute_transfer`。

---

### F2〔P0，阻塞中〕`<Audio N>` 三处口径不一致

**上游**：`<Audio N>` 是一等参考标签（`:110-126`），要么复制要么引用，
且"当 `<Audio N>` 明确对应某个目标说话人时，复用该说话人的全局 ID"（`:120`），
并须在 `subject_definitions` 有定义、在 `retention_analysis` 有**音频档**保留行。

**我们的三种口径**：

| 位置 | 现状 |
|---|---|
| `mie_craft/h3_loop_prompts.py:1155-1160` | **禁止**任何 `<Audio N>` 出现在 ref2va 正文（硬报错） |
| `mie_craft/h3_loop_prompts.py:356-358` | 清单里出现 Audio 槽**直接抛异常**："standalone audio references are not supported" |
| `mie_craft/script/builder.py:830` | **却在写** `<Audio N>`：`Voice casting: (S%d) uses the voice identity from <Audio %d> only; the reference words are not dialogue.` |

第三条没有对应的 `subject_definitions` 定义行，也没有音频档保留词
（`fully_copy`/`partially_copy`/`reference`/`weak_reference`），
只有一句自然语言散文。机制原因：轨道 A 的 ref2va 走 `validate_label_policy`，
轨道 B 的 v2 编译**不走**任何 plan 级校验（builder 提前返回，见 F3 附注）。

**评审处置**：维持 v1 判断——**只统一词表与标签口径，不碰路由**；等 Q9。
若 Q9 裁定走外部 TTS（`lip_sync_source`），`builder.py:830` 那行**删除**而非搬家
（外部 TTS 下 H3 不再吃参考音频，写 `<Audio N>` 反而误导）；
若裁定走 native ref2va，则按上游形态补定义行 + 音频档保留行。
**实施状态**：⏸ 等用户 Q9。

---

### F3〔P0→降级为非破坏性，先改计划再排期〕剧本资产编译产物形状

**上游**：`:255` 明确 full-reference 模式主字段换成 `detailed_description`，
`integrated_multimodal_description` 被**替换**而非并存；`:257` 要求参考标签写进正文。

**我们的现状**：轨道 B 编译出来是三段式（`contract.py:611`），正文带 `<Picture N>`
（`contract.py:620`、`:635`），`builder.py:833-873` 还往 `prompt_prefix` 塞了一堆
`<Picture N>` 引用句。

**v2 改判（评审结论）**：**不是破坏性变更，不升 v3、不新增节点**。理由：
段名不是模型词表（§2 勘误-2），形状差异是**提示词形状**问题；
三段式 + 参考图在真机上一直能跑（2026-10-26 起多部成片实证）。

**v2 的实施形状**（排期前须再过一次用户确认）：

1. **无参考图接入：三段式字节不变**——`to_runtime` 现产物一字不动，
   已发布 v2 资产的回放不受影响。
2. **接了参考图（roster 有图）：编译六段式**——数据全部来自现有
   `execution` 字段（framing/movement/perspective/visible_state/directions），
   只是**换壳**：`integrated_multimodal_description:` → `subject_definitions:` +
   `summary:` + `retention_analysis:` + `detailed_description:`，
   逐镜正文行原样搬进 `detailed_description`。
   retention 行按 F1 裁定：identity → `fully_preserved`，场景 → `reference`。
3. **补 plan 级校验的洞（评审新发现，独立于 1/2 也要做）**：
   `builder.py:1010-1011` 的
   `if storyboard.get("assembly_mode") == "authored": return _assemble_authored_plan(...)`
   提前返回，绕过 `validate_plan`（`:1310`）、`validate_label_policy`（`:1316`）、
   `validate_dialogue_invariantity`。v2 编译产物**按实际产出的形状**过校验
   （三段式过三段 schema，六段式过六段 schema），不是强制六段。

**实施状态**：📋 本节即"先改计划"；排期待用户点头。

---

### F4〔P1→已实施〕`summary` 不描述参考关系

**上游**：`:143-151` —— summary 要"概述任务类型、目标视频、以及**主要参考关系**"，
并使用已定义的标签；`:348` 样例的 summary 明确写了 `<Subject 3>`、`<Subject 1>`、
`<Audio 1>` 各扮演什么。

**评审裁定（Q4）**：用**标签**；**追加**一句，不重写现有句；口播内容不进 summary。
`presence_summary` 和 `install_shot_speech` **都要改**（v1 只列了前者，
漏了非 POV 主路径）。

**实施**：✅ `reference_relation_sentence(manifest, referenced_pictures)` 新增于
`h3_loop_prompts.py`：

- identity 对 `<Subject K> -> <Picture M> bind identity and appearance`；
  场景图 `set the destination environment`；视频槽 `supply choreography and timing only`。
- Subject 编号沿用清单 identity 顺序（与 fallback/repair 同约定）；
  **`referenced_pictures` 过滤**：概念没引用的 Picture 不绑定——实施中真机测试
  当场抓住了这条（Subjects 4/5 从复制品摆件图漏进 shot 3 的 summary），
  与评审"未绑定标签不进 summary"的口径一致。
- 两条安装路径（`install_presence_speech` / `install_shot_speech`）+ 节点
  `_apply_locked_speech` 全部穿参；三段式路径不受影响（六段式才追加）；
  无 manifest / 清单无槽位 → 追加空串，历史字节不动。

回归测试：`test_reference_relation_sentence_built_from_manifest_labels`、
`test_install_shot_speech_appends_reference_relation_to_summary`。

---

### F5〔P1→已实施〕内部切点不承接上一状态

**上游**：`:11-17` Video Continuity Rule——新 `[Shot]` 的第一帧必须**自然延续**
上一 `[Shot]` 的最终状态（主体姿态、位置、动作、道具、环境、光照）。

**我们的现状**：`contract.py` 的内部切点行只交代新景别：
`[Shot %d] %s, %s, %s. Cut directly to this framing within the same scene.`

**评审裁定（Q5）**：编译器机械拼**固定承接句**，只使用**资产里已有的状态**；
**不得发明姿势/道具**（v1 例句 "remains seated with the parcel still in his right
hand" 驳回）；`execution.visible_state` 是**开场状态**（`contract.py:657` 在事件
循环前写一次），**不得**复用为切点末状态。

**实施**：✅ 切点行后追加
`Cut continuity: <此刻在画的角色> in the frame continue exactly as the action has
reached them; positions, held props, and the scene carry over unchanged, and only
the framing changes.`
（无人时用 "the scene continues exactly as the action has reached it; …"）。
"在画角色"来自已处理的 enter/exit 事件集合；"positions / held props / scene"
是**类目级不重置规则**，不指名任何具体姿势。字节确定性：按 roles 顺序遍历。
回归测试：`test_internal_camera_cut_gets_fixed_state_carry_over`。

---

### F6〔P2→已实施〕缺少"连续性不是保留关系"的防误用条款

**上游 `:190`**："连续性本身不是参考保留关系。不要因为主体的姿态、位置、动作、
镜头构图随时间自然变化就把它标成 partially preserved。"

**评审裁定**：随 F7 一起写——"**动作随时间变化不是 partially preserved**"。

**实施**：✅ 三处同落地（指令句 "A motion or performance that plays out over time
is never 'partially_preserved' - the transfer is whole or absent."、回退句、
addendum Video 规则末句）。

---

### F7〔P1→已实施〕视频动作参考应该标 `attribute_transfer`

**上游 `:187`**：`attribute_transfer` = "参考特征被转移到另一个可识别的目标主体上"。
引擎自己的运动语义锚（`chain_nodes.py:4007-4035`）写的正是
"Transfer only that visible performance without importing the source identity,
wardrobe, setting, lighting, or composition."——教科书式的 attribute transfer，
但我们给它的保留行用的是音频档 `reference`。

**评审裁定**：与 F6 合并为第一优先项。**实施**：✅ 见 F1/F6。
另核实：引擎 web 词表 `H3_VISUAL_RETENTION_MARKERS`
（`h3_prompt_schema_core.mjs:69-74`）**本就含** `attribute_transfer`，
引擎侧无需改动。

---

### F8〔P2→留待实测，本轮不做〕full-reference 的"风格前置句"

上游 `:256` 要求风格 1-2 句写在 `[Shot 1]` 之前；我们（和官方 `Ref2V Studio`
样例一样）直接从 `[Shot 1]` 开始，风格在 `prompt_prefix`。引擎分析器对缺前置句
只给 warning（`h3_prompt_schema_core.mjs:390-394`）。

**评审处置（Q6）**：不做。与 `prompt_prefix` 风格句重复表达反而稀释；
留一个待实测项：若真机出现"风格漂移随片段衰减"，再评估把风格句复制到
`detailed_description` 段首是否有效。**实施状态**：⏸ 记录不动。

---

### F9〔P2→判定：不是偏差〕loop 轨的任务类型前缀只有一种

loop 恒用 `[reference generation]` 并禁 `[video continuation + ...]`
（`h3_loop_prompts.py:2797`、`ref2v_addendum.txt:36-37`）。

**评审处置（Q7）**：**不构成偏差**——loop 没有"编辑一条源视频 / 从既有视频续写"
的能力，上游六类任务是能力边界的词表，不是输出义务。评审另指出 v1 低估的一点：
**单发节点**（`mie_prompt/h3_prompts.py:330-333` 的 `structure_guidance`）
已写全六类与组合规则，**loop 与单发的词表口径是分裂的**——将来 loop 若引入
源视频编辑能力，词表现成，但要先统一两处口径再扩。**实施状态**：📋 记录。

---

### F10〔P2→判定：不是硬偏差；改修死模板〕`<Subject N>` 与说话句

上游 `:295-301` 要求说话时"视觉参考标签与说话人 ID 同时保留"，
形如 `<Subject 2> (S1) turns toward ...`。我们两者都有（名字+(S1) 在说话句、
`<Subject N>` 绑定在 `subject_definitions`），只是不在同一句——**不是硬偏差**。

**评审处置（Q8）**：**不把 `<Subject N>` 写进说话句**（与
`subject_definitions` 重复稀释）。但要修一个 v1 没发现的**死模板**：
`ref2v_addendum.txt:85-103` 的说话人 ID 规则示例写的是
`<Subject 2> (S1), ... says ...` 形态，而 `install_shot_speech` 用**名字句**
整段覆写正文——模型被教了一个**永远不会存活到最终产物**的句式。
修法：把 addendum 示例改成与节点实际产出一致的"名字 + (S1)"形态。
**实施状态**：📋 小改，随下一批 prompt 文本变更一起（避免单独动 SYNC 面）。

---

### 评审新发现（v1 没列的）

| 编号 | 发现 | 处置 |
|---|---|---|
| N1 | **authored 提前返回绕过全部 plan 级校验**（`builder.py:1010-1011` → `validate_plan`/`validate_label_policy`/`validate_dialogue_invariantity` 全不跑） | 并入 F3-3：按实际产出形状过校验，独立于六段式排期 |
| N2 | **轨道 B 参考关系写三遍**：`builder.py:833-873`（prefix 总述 + 逐角色 identity 句 + Scene reference 句）+ `contract.py` 逐镜 `<Picture N>` 句 + voice casting 句——同一条参考关系三处重复表达 | F3 换六段壳时一并收敛：prefix 保留总述，逐镜关系进 `subject_definitions`/`retention_analysis`，消除三写 |
| N3 | F9 的 loop/单发词表分裂被 v1 低估（见 F9） | 记录，扩能力前先统一 |

---

## 4. 改动面与仓库归属（v2 实况）

| 计划项 | 改哪个仓库 / 哪个文件 | 破坏性 | 状态 |
|---|---|---|---|
| F7+F6 | `mie_craft/h3_loop_prompts.py`（指令 VIDEO MOTION REFERENCE 块 + `build_fallback_shot_lines` Video 分支）；`mie_prompt/.../ref2v_addendum.txt`（Video 规则） | 否（提示词文本） | ✅ `a07fba8`/`b7fc1ea` |
| F4 | `mie_craft/h3_loop_prompts.py`（`reference_relation_sentence` + `presence_summary` + 两 installer）；本仓节点 `_apply_locked_speech` 穿参 | 否 | ✅ 同上 |
| F5 | `mie_craft/script/contract.py`（切点承接句） | 否 | ✅ `a07fba8` |
| F10-死模板 | `mie_prompt/.../ref2v_addendum.txt:85-103` 示例句式 | 否 | 📋 待做 |
| F2 | 同上 + `mie_craft/script/builder.py:830` | 否，但**依赖 Q9** | ⏸ |
| F3 | `mie_craft/script/contract.py` + `builder.py`（N1 校验洞独立可先行） | **否**（v2 改判） | 📋 排期中 |
| F8 / F9 | 不动 / 记录 | — | ⏸ / 📋 |

**跨仓同步纪律**（既有规则，本次适用）：
1. **SYNC PAIR**：若改动 `mie_prompt/prompts/h3_storyboard/system_asset.txt`，
   必须同 commit 改 `docs/script_asset_system_prompt.txt`（本仓），
   否则 `tests/test_script_asset_prompt_sync.py` 会红。本轮未动该对。
2. **三包锁步发布**：`scripts/release_authoring.py` 管 `mie_llm` / `mie_prompt` /
   `mie_craft` 的同版本发布；本轮只本地 commit + 本地安装（0.1.5 源码态），
   **未发版**。
3. **本机 + 局域网双部署**：改完同步本机（`C:/PP/V9`）与局域网
   （`\\192.168.31.150`）两个部署库；**不推送 Registry、不发版**。
4. **H3 链路固定**：按 2026-10-29 裁定，本次只改资产层与提示词层，
   `F:/ComfyUI-Mie-H3-Engine` **不动**（本轮核实其词表本就含
   `attribute_transfer`，无需动）。

---

## 5. 实施顺序与验收（v2）

### 5.1 批次

- **Batch 1（提示词变更，跟真机矩阵）**：F7+F6 + F4 + F5 —— **代码已落地**；
  剩真机验收（v1 误标"低风险"：三件都改提示词文本，必须过真机）。
- **Batch 2（小改）**：F10 死模板修正（addendum 一处示例句式）。
- **Batch 3（非破坏性增强，排期待用户点头）**：F3（六段式换壳 + N1 校验洞 +
  N2 三写收敛；无参考图字节不变）。
- **Batch 4（依赖外部裁决）**：F2 —— 等 Q9。
- **Batch 5（记录/待实测）**：F8 / F9。

### 5.2 已完成的验收

1. **字节级**：`python -m pytest tests/ -q --ignore=docs/qa` =
   **2055 passed, 10 skipped**（基线 2051 + 4 新测试；
   `mie_craft`/`mie_prompt` 源码仓无独立测试目录，测试全在本仓）。
2. **回归覆盖**：视频槽新档（directive+fallback）、summary 追加句
   （含 `referenced_pictures` 过滤——实施中被真机测试抓住的泄漏）、
   切点承接句（位置 + 类目级不重置 + 不复用 visible_state）各有专测。
3. **历史字节不动**：无 manifest / 三段式 / 无切点路径的既有断言全部原样通过
   （未改一行旧断言，除新增测试外零 diff）。

### 5.3 剩余验收（Batch 1 收口条件）

1. **提示词级**：对同一输入，plan 的 `prompt` 数组 diff 逐行人工过一遍，
   确认只有预期行变化（`Reference relations:` 句、`Cut continuity:` 句、
   Video 槽 retention 行）。
2. **零 LLM 渲染级（release 门槛）**：正确流程产出的分镜，经零 LLM `BuildPlan`
   能正确渲染——`tests/test_h3_loop_live_combo_matrix.py`（`MIE_LIVE_E2E=1`）
   真机矩阵 + ASR 验收；重点盯：环境槽保持 `reference` 后与旧成片无回归、
   summary 多一句标签后主体一致性不掉、切点承接句不与模型生成漂移打架。
3. **回流**：真机若跑出问题，按既定纪律回流为规则/测试，再重跑。

---

## 6. 问题清单（v2：Q1-Q8 已按评审结论闭环，Q9 待用户）

| 编号 | 问题 | 评审结论（v2 采纳） |
|---|---|---|
| Q1 | 视觉保留关系收口到上游四档吗？环境槽、视频槽各选哪一档？ | **拆开**：环境槽随官方样例保持 `reference`（路线乙）；视频槽 `attribute_transfer`（路线甲）。✅ 已实施 |
| Q2 | `<Audio N>` 定义行 + 保留行放 `subject_definitions`+`retention_analysis` 还是留 `prompt_prefix`？ | 等 Q9：native ref2va → 按上游形态补；外部 TTS → 删 `builder.py:830` 行 |
| Q3 | F3 走"资产契约升 v3"还是"新增节点"？ | **都不走**：非破坏性增强——无参考图字节不变，有参考图换六段壳，节点与 v2 契约不动 |
| Q4 | summary 参考关系用标签还是自然语言？追加还是重写？ | 标签；追加一句；两处安装路径都改；未绑定标签（`referenced_pictures` 之外）不进。✅ 已实施 |
| Q5 | 切点状态承接：编译器机械拼，还是资产契约让 LLM 写？ | **编译器机械拼固定句**，只用资产已知状态，不发明姿势，不复用开场 visible_state。✅ 已实施 |
| Q6 | full-reference 风格前置句做不做？ | 不做（与 prefix 重复稀释）；留待实测项 |
| Q7 | loop 轨要不要扩展任务类型？ | 不扩；六类词表是能力边界不是输出义务；扩前先统一 loop/单发口径（N3） |
| Q8 | 说话句要不要带 `<Subject N>`？ | 不带；改修 addendum 的死模板（F10） |
| Q9 | （**用户裁决**）H3 音频路由：native ref2va 音色克隆 vs 外部 TTS + lip_sync_source？ | ⏸ 待用户；阻塞 F2 |

---

## 7. 明确**不做**的事（v2 增补一条）

- **不推翻"正文由节点确定性生成"**（"LLM 判断、代码保证"）。上游 `:339-345`
  那套"模型写 detailed_description"的写法不引入。
- **不引入 350-500 词正文预算**（上游 `:269`）：那是整片量级；我们是逐 clip
  预算（`shot_system_addendum.txt` 15-20 词/秒）。
- **不强制 prompt 正文一律英文**（上游 `:5`）：我们有 `output_language`（含中文），
  且资产契约要求 `execution` 无中文，两处已各自成立，不冲突。
- **【v2 增补】不吸纳 full-reference 时间戳**（上游 `:240`）：loop 轨的
  `At MM:SS.mmm` 禁令是**链式产品决定**——一个 clip 是一条连续镜头、无内部时钟，
  链路靠 carried overlap 续接；上游时间戳服务于"一条提示词内多镜定位"，
  与我们的分 clip 形态不同构。该禁令**也不外推**到轨道 B 的 `camera_cuts`
  内部切点（那是同一次生成内的换景别，`[Shot N]` 序号已足够）。
- **不动 H3 引擎**（`F:/ComfyUI-Mie-H3-Engine`），只引用其代码作为证据
  （本轮核实其视觉词表本就含 `attribute_transfer`，无需改）。
- **不改 `<Picture N>` 的四种帧锚角色**（上游 `:72-92`）：我们的 i2va/fl2va
  已有独立的帧位机制（`ensure_keyframe_idiom`），不在本期范围。
- **不引入"一个 Subject 由多个资产定义"**（上游 `:51-70`）：我们的槽位语义是一对一，
  多资产合成 Subject 是引擎运动锚的能力，不是资产层的。
- **不发版、不推送 Registry**（既定纪律：只同步本机与局域网）。
