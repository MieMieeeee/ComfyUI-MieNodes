# 分镜资产 + 零 LLM 组装 实施计划

> 修订 2026-10-19。前置依赖：资产库 V2（`docs/PLAN_ASSET_LIBRARY.md`，已落地）、
> H3 Loop 生成器确定性化系列修复（`9a5d1e6`/`0cc25d9`/`cfa1487`，已落地）。
> 本文档描述的是**新增第二条 plan 生产轨道**，不改动现有 LLM 生成轨道。
>
> 目标已经定死：角色、场景、分镜三份资产都在时，组装阶段**完全不调用 LLM**。
> 可以接受比金样更平的散文。不允许为了把散文写好看而在本节点加模型、
> 翻译、润色或从描述里抽句子。

## 0. 一句话

把验证过的分镜固化成一份 JSON 资产。角色资产和场景资产按槽位接上之后，
纯代码节点拼出与现有生成器同 schema 的 `plan_json`，喂给现有的
`MiniMaxH3ChainPlanModern`。这条节点没有 LLM connector。
同分镜 + 同资产 + 同 seed ⇒ 字节级相同的 plan。

## 1. 要解决的事

现状（2026-10-18 组合矩阵实测）：资产 roster 模式下，LLM 生成器做的事本质是
带噪音的模板填充。9 个组合的终稿段落相同，差异是名字、描述和台词。
模型侧产出过整类缺陷：scene_01 漏 `<Picture 3>`、`Setting:` 留下中文原句、
镜头指令自相矛盾、facts 空回复导致空间钉全灭。同时：

- 一次生成 6–9 个串行 LLM 调用，约 75% 的耗时是模型思考；
- Token Plan 配额耗尽（HTTP 402）会整条管线停摆（10-18 两次实测）；
- 同输入重跑 plan 不确定，生产复现只能靠存下 plan 再手改。

生成器后半段已经是代码在盖章：台词逐字、`<Picture N>` 绑定、空间钉、
`[Shot 1]`、声纹句、pose 约束，都发生在模型交卷之后。资产描述和音色也是
库里现成的英文。组装阶段要做的是把这层盖章变成唯一的作者，模型不再出场。

会变平、并且接受变平的部分：

- retention 不再每场枚举一遍服装和毛色，每场同一句固定模板；
- soundscape 不再每场加一句不同的环境色，全片一句氛围；
- `[Shot 1]` 固定是锁机位的中景双人句，`camera_discipline=auto` 也不另写特写；
- 多语言是另一份已经翻译好的分镜，本节点不翻译。

结构缺陷那一类（漏 Picture、中文 Setting、镜头自相矛盾、facts 空回复）
随模型一起从这条轨道消失。

## 2. 不做什么

- **不动 LLM 生成轨道**。新散文、不规范输入、自动润色仍走 V1/V2 生成器。
  两轨喂同一个 PlanModern。
- **组装节点不调用模型**。没有 connector 插座，没有 enhance / temperature /
  max_tokens / timeout，没有视觉打标，没有 facts 抽取，没有 prefix 合成。
- **不在组装时翻译或改写**。台词、Setting、风格块、动作句、氛围句都是资产里
  已经写好的文本。缺了就报错，或者按下面写明的常量顶上。不临时生成。
- **不做「从散文自动生成分镜」**。第一版是手写 JSON，外加一个从金样 plan
  抽模板的脚本（§8）。抽出来的中文 Setting 不写入 `setting_line`。
- **不做分镜图形编辑器**。分镜是 JSON 文件 + Load 节点下拉。
- **不做多场景**。第一版只有 `scene_1`。`shots` 里不写场景名。
- **不改渲染链**。LoopStart / Preflight / 参考图 / 音频接线保持原样。
- **不给渲染层的 H3 缺陷兜底**。独白镜丢说话人仍只在 summary 里提示重摇配方。
  真机若觉得模板散文太平，记录为渲染侧限制，不把模型加回本节点。
- **不把 `mechanical` 写进 H3 prompt**。这个词只出现在节点的 STRING summary。

## 3. 磁盘格式

### 3.1 分镜资产

目录：`<library>/storyboards/<名字>/storyboard.json`。
无 images/audio 子目录。它不是 character/scene 的第三种 `kind`，
`library.py` 里 `kind must be character or scene` 的分支不动。
读写放在新的 `nodes/assets/storyboard.py`。

```jsonc
{
  "name": "客厅板栗对话",
  "kind": "storyboard",
  "version": 1,
  "language": "zh",
  "camera_discipline": "locked",
  "pacing": "normal",
  "seed_policy": "per_scene_increment",
  "prompt_prefix": "A cozy cottage living room, polished 3D, warm light.\nThe camera stays locked off.",
  "cast": {
    "孙女": {"character_slot": 1},
    "爷爷": {"character_slot": 2}
  },
  "setting_line": "Grandpa and the girl sit on the sofa, talking.",
  "shots": [
    {"lines": [
      {"speaker": "孙女", "text": "爷爷，我一直有个心理阴影。"},
      {"speaker": "爷爷", "text": "是什么呢。"}
    ]},
    {"lines": [
      {"speaker": "孙女", "text": "小时候我还没上小学。",
       "picture": "She keeps both paws in her lap."},
      {"speaker": "孙女", "text": "咱家院子里有几只小刺猬。"}
    ]}
  ],
  "spatial_layout": {"孙女": "left of frame", "爷爷": "right of frame"},
  "ambience": "Quiet room tone with soft indoor air. No music."
}
```

字段：

| 字段 | 规则 |
|---|---|
| `version` | 整数。只认 `1`。其他值报错并写出文件路径和期望版本。不迁移。 |
| `language` | `zh` 或 `en`。每个 `<d>` 标签都用它：`zh` → `[Chinese]`，`en` → `[English]`。不按字面检测语言。 |
| `camera_discipline` | `locked` / `no_closeups` / `auto`。三种都输出同一句锁机位 `[Shot 1]`。`auto` 不表示「自己发明景别」。该字段只在两处氛围都空时，用来选三句常量里的哪一句。 |
| `pacing` | `fast` / `normal` / `slow`，对应 `dialogue_segmenter.PACING_PRESETS`。决定每场 `length`，不决定景别。 |
| `seed_policy` | `per_scene_increment` 或 `same_across_scenes`。 |
| `prompt_prefix` | 必填、非空、英文。按换行拆成 `plan_json` 的字符串数组，丢掉空行。含汉字则报错。内容原样进入 plan，节点不检查里面有没有角色名。 |
| `cast` | 说话名 → `character_slot`（1..8，因为还要留一个场景图，H3 图片上限 9）。槽位不可重复。说话名可以不等于资产名。 |
| `setting_line` | 可空。空 = plan 里不写 `Setting:`。非空必须是英文，含汉字则报错。 |
| `shots` | 见下方切分规则。 |
| `spatial_layout` | 可空。有则键必须是 `cast` 里的说话名，值是位置短语（如 `left of frame`）。缺的说话人用 `complete_layout_from_turn_order` 补左/右/中，并在 STRING summary 里写明「layout filled by turn order」。这是固定顺序，不是模型。 |
| `ambience` | 可空的一句英文。全片每场共用。含汉字则报错。 |
| `lines[].picture` | 可空的英文动作句，写在该句台词之前。含汉字则报错。没有这个字段就不写动作。 |

切分：

- 省略 `shots`：把全部台词按出场顺序交给 `group_budgets_into_scenes`，`max_lines_per_scene=2`。这是现有的 minimal-cut。
  - 顶部增加 `lines` 字段作为输入：`lines: [{speaker, text, picture?}, ...]`
    是按出现顺序排列的扁平台词列表，与 `shots` 互斥。校验层和组装节点
    都拒绝同时携带两个字段；任一字段存在即触发对应的组装路径。
- `shots` 为数组：作者预切。每镜 1..2 句，说话人必须在 `cast` 里，镜与镜拼回来的台词顺序就是整份剧本。
- `shots: []` 或 `lines: []`：报错，不当成省略。

构建时还有这些硬错误，不猜测、不填默认音色：

- 展开后出现 `cast` 里没有的说话人；
- 某个被引用的 `character_slot` 或 `scene_1` 没有接线；
- 角色资产的 `voice` 去空白后为空；
- 被引用的角色没有音频（Load 的单音频约束已经会先失败；构建节点再查一次 bundle）；
- 同一资产接进两个槽。

台词文本是锁定数据，逐字写入 `<d>`，不翻译、不改写。

### 3.2 场景资产的可选 `ambience`

`scene.json` 增加可选 `"ambience": "<英文一句>"`。

选用顺序：分镜 `ambience`，否则场景资产的 `ambience`，否则按 `camera_discipline`
取节点内的三句英文常量。不对场景描述做关键词抽取。

旧场景文件没有该字段时照常加载。`load_scene` 交给 `MIE_ASSET` bundle 时要带上
这个字段（没有则为空字符串）。`save_scene` 今天重写 JSON 时只留 `name` 和
`description`，实现时必须把已有的 `ambience` 原样写回，避免用户以后覆盖保存时
把它删掉。

## 4. 节点

### 4.1 `LoadStoryboardAsset`

- `library_dir`：STRING，默认空，widget 隐藏，空 = 本机库。形态对齐 LoadAssetV2。
- `name`：下拉，只列 `storyboards/`。
- 输出：`MIE_STORYBOARD`、`status`。
- `IS_CHANGED` 返回文件内容哈希。改完 `storyboard.json` 再排队必须重跑，不能吃缓存。
- JSON 坏、版本不认、必填字段缺：报错里带绝对路径。

### 4.2 `BuildPlanFromStoryboard`

输入：

- `storyboard`（`MIE_STORYBOARD`，必填）
- `character_1..9`、`scene_1`（`MIE_ASSET`，插座随接线增长，上限与 V2 相同）
- `seed`（INT）

没有 LLM 相关控件。

槽位规则和 V2 的 `_slot_bundles` 不同：**不把空洞压紧**。
`character_slot: 2` 就是第二号插座，也是 `<Picture 2>` 和 `(S2)`。
中间缺线就报错。没被分镜引用的更高号插座忽略。
图片顺序是被引用的角色槽从 1 数到 N，然后 `scene_1` 占下一个 Picture。
第一版只有一个场景，所以角色槽最多 8。

> **已报备实现细节（2026-10-19）**：节点内部把 `character_1..8`
> + `scene_1` 收集成一个 **`{slot: bundle_or_None}` 字典**，空槽位
> 显式留 `None`（不 skip）。`assemble_plan` 按 `character_slot` 显式
> 取 bundle，缺失即报「character_slot N declared by cast ([...]) but
> no asset is wired into socket 'character_N'…」，避免静默错绑（评审
> P0 实测复现：cast 槽 {1,2} 但 character_2 空线 → 旧实现会把
> character_3 上的资产错绑到 (S2)，新版直接 raise）。`scene_1` 插座
> 在节点内部先放在 `MAX_SLOTS+1`，`assemble_plan` 收到 cast 后
> 把场景搬到 `max(cast_slots) + 1` 的 picture 位（编号合同「场景
> = 角色最大槽 + 1」由 `assemble_plan` 统一保证）。

输出，对齐 V2 生成器里下游真正用到的那几根：

| 输出 | 内容 |
|---|---|
| `plan_json` | `plan_to_json_string` 的结果。顶层只有 `shots` 和 `prompt_prefix`。 |
| `summary` | 运行说明。含独白镜重摇建议、`mechanical` 来源、补过 layout 时的那一句。尺寸不一致时也写在这里。 |
| `images` | 与 V2 相同的批。各图尺寸不一致时为 None，并在 summary 里说明改接 Asset → Slots。 |
| `assets` | `MIE_ASSET_ROSTER`：被引用的角色 bundle（槽位序）再加 `scene_1`。这就是 `AssetToSlots` 的唯一入口，图片和音频都从这里拆。 |

分镜文件本身不进 roster，它没有参考图，也没有音频。
`assets` 必须输出，是因为只输出 `images` 会丢掉音频，用户会再拉一根 Load，
两套顺序会对不上。

节点函数是同步纯代码。不读网络，不写文件。不接收 connector。

`(Sn)` 按 `character_slot` 编号，不按谁先说话。
爷爷在槽位 2、又是第一句台词时，这句是 `(S2)`，音频仍在 `audio_2`。
不用 `build_speaker_id_map`，那个函数按首句出场编号，会和 `AssetToSlots` 的
`audio_n` = 第 n 个角色错位。

声纹用资产的 `voice`，去掉首尾空白和末尾句号。空则报错。
不调用 `default_voice_for`。

### 4.3 `SaveStoryboardAsset`

粘贴 JSON + 名字，校验后写入 `storyboards/<名字>/storyboard.json`。
没有逐字段表单。覆盖方式对齐 Save Asset：选中已有名字就是覆盖，
status 写 `overwritten` / `created`。不做 Delete 那种二次 `confirm_token`。

## 5. 代码落点

已定，不再二选一。

- `nodes/assets/storyboard.py`：分镜的列目录、读、写、内容哈希。不 import `nodes/llm/`。
- `nodes/llm/storyboard_builder.py`：构建节点。可以 import `minimax_h3_loop_prompts` 和 `dialogue_segmenter`。
- 不把拼装函数下沉到新的中立模块。那些函数是「在已有骨架上改段落」，不是一套独立组装库，挪调用点没有收益。
- `__init__.py` 只追加这三个节点的注册。旧节点的 `INPUT_TYPES` 不动。

组装分两步，不新写一套六段式：

1. `build_fallback_shot_lines(SCHEMA_SIX, manifest, spatial_layout)` 造出骨架。
   manifest 由接线资产拼成：角色 `role=identity`、`about` = 该资产 `description`，
   槽位号用分镜里的 `character_slot`；场景 `role=environment`、`about` = 场景描述，
   排在所有角色 Picture 之后。这一步产出
   `<Subject k> -> <Picture m>: {description}` 和 `<Picture s>: {scene description}`。
   retention 用这个函数自带的固定句（已经写了不保留参考头角）。
   不从描述里再摘特征，也不追加 `_POSE_LOCK_NOTE`：追加条件要求行内还没有
   `head angle`，而这句固定文本里已经有了。
2. 把选定的氛围句先写进骨架的 `overall_soundscape`。然后调用 `install_shot_speech`。
   对白路径不会改写 soundscape（它只在 voiceover 时替换该段）。
   `install_shot_speech` 负责：裸名别名、`summary:` 段、`[Shot 1]`、空间钉、
   把台词句放进 `detailed_description`。调用时传入：
   - `speaker_identities`：说话名 → 该槽角色的 `description`（键是说话名，不是资产名）
   - `line_speakers`、`speech_lines`
   - `spatial_layout`
   - `opening_setting`：`setting_line` 非空时放进这个列表
   - `schema=SCHEMA_SIX`

台词句的语法与 `_speech_act_sentence` 相同，标签改由分镜 `language` 决定：

```text
{说话名} speaks as (S{character_slot}) {voice} <d>[Chinese] {台词}</d>
```

`en` 时标签为 `[English]`。声纹和 `<d>` 之间不放句号。
不调用 `assemble_dialogue_line_blocks`，因为它内部会走 `_detect_dialogue_language`。

> **已报备偏差（实现端，2026-10-19）**：**不**调用 `install_shot_speech`，
> 改用同模块内 `_build_storyboard_shot_body` 自己拼装 description body、
> summary 与 retention 的标记。原因：`install_shot_speech` 的 stage-cue
> 分支会调用 `_cue_sentence`，对非「still」型动作句输出
> `Silent picture only. Mouths stay closed and nobody speaks. The
> picture shows: ...` 的封装；当该 cue 出现在**说话行**之前时，下文
> 紧跟 `<d>` 是互相矛盾的。分镜轨道要求画面句旁就是说话句，所以动作
> 句改为直出（cue 文本去尾句号 + 一次句号），不再走 `_cue_sentence`。
> 六段 header 顺序、`subject_definitions` 绑定、`summary` 的 reference-
> generation 措辞、`retention_analysis` 的 pose-annotation 与上游 LLM 轨
> 道保持一致。

行首呼语的转身被丢弃。理由：分镜不携带 role 语法，`resolve_line_facing`
需要 role→speaker 绑定才能构造转身句，构造空集合等于无效调用。LLM 轨
道在「爷爷，」半角逗号才匹配；分镜若作者写了呼语，`turns to face` 子句
也不会上**本体**——shot LLM 还是自己猜眼神。

`lines[].picture` 在校验层把汉字挡掉后，作为一句直白的英文动作句放在对
应台词之前。**不**走 `_cue_sentence`（见上面偏差说明）。

其他区段：

| 区段 | 做法 |
|---|---|
| 省略 `shots` 时的切分 | `estimate_shot_budget` + `group_budgets_into_scenes(..., max_lines_per_scene=2)`。不用 `distribute_lines_to_scenes`，那个函数是「给定场数后按时长均分」。 |
| 预切时的 `length` | 同一套 `estimate_shot_budget` / `seconds_to_length`，用该镜里的句子算帧数。不把金样的 192/243/158 写死。 |
| `seed` | 十进制字符串。`per_scene_increment` 时第 i 场（从 0）为 `str(seed + i)`。 |
| `id` | `scene_01` 起，两位补零。 |
| 音乐段 | 骨架里的 `DEFAULT_MUSIC_LINE`（`No non-diegetic music.`），不另写。 |
| 序列化 | `plan_to_json_string`。 |
| 终稿校验 | `validate_plan(..., schema=SCHEMA_SIX)`、`validate_dialogue_invariantity`、`validate_label_policy`、`validate_spatial_layout_invariant`。任一失败则节点报错，不输出凑合的 plan。 |
| 独白提示 | `_monologue_scene_hint` 只写入 STRING `summary`。不改 plan 正文。 |

`repair_label_pairings` 不负责产出绑定。骨架已经是规范形。

## 6. 语义

### 6.1 双轨

LLM 生成器继续负责创作：新散文、不规范输入、润色。
分镜轨道负责生产：同一分镜换角色或换场景、换 seed、批量重摇。
多语言是另一份分镜，台词已经是目标语言。本节点看到什么字就写入什么字。

### 6.2 字节可复现

同分镜文件字节、同角色/场景的 description 与 voice、同槽位接线、同 seed
⇒ `plan_json` 逐字节相同。没有时间戳，没有随机序，没有模型重试。
图的像素不进 plan 文本；图只影响 `images` / `assets`。

### 6.3 说话名和资产名

plan 里面向 H3 的人名是说话名。资产名可以不出现。
`<Picture N>` 和 `(Sn)` 跟 `character_slot`，跟谁先开口无关。

### 6.4 Setting

只可能来自分镜的英文 `setting_line`。空就没有 `Setting:` 行。
不会出现中文原句残留。

### 6.5 独白镜

summary 里给出和 LLM 轨道相同的重摇提示。plan 结构不因为独白而改。

### 6.6 故意放弃的效果

下列差异不算组装失败：

- retention 是固定模板，不是金样里那种逐场特征长句；
- soundscape 是一句氛围，不是金样里那种逐场加料；
- 全片同一个锁机位 `[Shot 1]`；
- 没有模型写的镜头散文。

这些差异是目标的一部分。真机渲染用来确认 H3 还能把人放在画面里，
不作为把 LLM 加回组装节点的入口。

## 7. 测试

1. **金样结构，不对散文逐字。** 用 `tests/fixtures/plan_english_kitten_grandpa.json`
   做素材，人工确认后得到第一张分镜。反推脚本不得把金样里的中文
   `Setting: 客厅里，猫爷爷和小猫妹妹坐在沙发上聊天。` 写进 `setting_line`。
   离线测试锁定：
   - 台词逐字、说话人正确；
   - 每场都有每个角色的 `<Picture i>` 和场景的 Picture；
   - 每场主体行都是 `<Subject k> -> <Picture m>:`，格式全片一致
     （金样自己在 scene_01 和 scene_02 上格式不一致，机械结果不复制这种不一致）；
   - 空间钉是 `名字 stays ...`；
   - 六段 header 顺序齐全；
   - 没有 `[Shot 2]`、`At 00:`、`@alias`；
   - 若写了 Setting，文本等于分镜的英文 `setting_line`；
   - retention 等于 `build_fallback_shot_lines` 的固定句；
   - `<d>` 标签等于分镜 `language`；
   - `(Sn)` 等于 `character_slot`。
2. 报错：说话人未绑定、一镜超过 2 句、引用槽未接线、`version` 不是 1、
   `voice` 为空、`setting_line` / `prompt_prefix` / `picture` 含汉字、
   `shots` 为 `[]`、槽位重复。
3. 说话名不等于资产名，并且槽位 2 先说话。终稿用说话名，第一句是 `(S2)`，
   `<Picture 2>` 仍在，roster 里音频顺序仍是角色槽位序。
4. 无 `ambience` 的旧场景能加载；保存往返不丢 `ambience`；空 voice 报错，
   且输出里不出现 `default_voice_for` 的那句默认音色。
5. 从 `verify_plan` 拆出离线函数，入参是台词列表和槽位数量，不再读
   `combo["user_input"]`。硬检查保留台词逐字、归属、在场、空间钉、Picture、
   违禁标记。不把「plan 里不许出现 fur」列为本轨道验收：猫的资产描述里本来就有 fur。
   中文 `Setting:` 残留是硬错误。
6. 真机渲染可选。机械 plan 走 PlanModern → LoopStart，渲一遍 4 幕。
   目的是看模板散文下人物是否还在画面里。渲染变差只记录，不改回 LLM 组装。

## 8. 实施顺序

1. 场景 `ambience`：读取、bundle 字段、`save_scene` 往返保留、旧文件兼容。测试 §7.4 的资产半边。
2. `nodes/assets/storyboard.py` + Load / Save + `IS_CHANGED`。不改 character/scene 的 kind 判断。
3. `BuildPlanFromStoryboard` 按 §5 接上，并输出 `assets`。
   `scripts/harvest_storyboard_from_plan.py` 只抽台词、说话人和 Picture 描述供人核对；
   中文 Setting 丢弃，`setting_line` 留给人工填写。
4. §7.1–§7.5 全绿后提交。
5. 可选真机渲染。双机同步时新节点是纯增量注册，旧工作流不改接线。

## 9. 已定决策

- 资产齐全则组装零 LLM。散文可以比金样平。不为散文质量把模型加回本节点。
- 锁定台词永不翻译、永不改写（2026-09-22）。Setting、prefix、动作、氛围同样只抄资产。
- 新代码不对用户文本做关键词分支（2026-09-27）。氛围三句只按 `camera_discipline` 枚举选择。
- UI 文案保持中文 · English 的顺序（2026-10-11）。
- 双轨并存。LLM 轨道的节点和 prompt 文件不动。
- 构建节点放在 `nodes/llm/storyboard_builder.py`。分镜读写放在 `nodes/assets/storyboard.py`。不下沉中立模块。
- 不认识的 `version` 拒绝，不迁移。
- 省略 `shots` 可以 packing；金样使用预切。空数组报错。可复现性来自纯函数，不来自「必须预切」。
- 分镜不进 roster。构建节点必须输出 `assets`，供 Asset → Slots 拆图和音频。槽位不压紧。
- prompt 里的 `summary:` 用 `install_shot_speech` 的现成句子。`mechanical` 只写在 STRING summary。
- `(Sn)` 等于 `character_slot`，不等于首个说话人。

## 10. 验收

- 构建节点的源码和它调用的组装路径上没有 LLM connector，也没有网络调用。
- 金样分镜的 §7.1 断言全绿，`validate_plan` 零错误。
- §7.5 的硬检查通过。中文 Setting 残留为零。不要求「全文无 fur」。
- 全量离线测试通过。
- 新节点注册之后，未使用这些节点的旧工作流行为不变。
