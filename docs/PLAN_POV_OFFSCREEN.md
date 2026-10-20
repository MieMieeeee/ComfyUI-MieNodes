# 第一人称画外音对白 + 每镜在场控制 实施计划（待第三方评审）

> 版本 2026-10-20。前置：分镜轨道与 V3 已落地（`283148f`→`6074729`），
> 板栗双资产可用；画外音 freeze 分支（全员旁白）已存在且被测试逐字锁定。
> 本计划新增**混合形态**：画外有声说话人 + 画面内说话人同镜对话（第一人称
> POV），以及**每镜在场集合**（角色某一幕不在画面）。

## 0. 一句话

给说话人加 `offscreen` 申报、给每镜加 `absent` 名单，行模板按三形分化
（画面内 / 画外音 / 全员旁白 freeze 不动），让 H3 不再把画外台词绑到画面
人物的嘴上，也让我们能明确表达「这一镜没有他」。

## 1. 要解决的事（社区痛点 + 我们的现状）

社区验证的 POV 对话写法三要素：固定句式 `says in an off-screen voiceover`、
闭嘴指令内嵌 `while … lips remain completely closed`、**具体**无声反应动作
（写 "listening" 常失效）。我们的现状：

| 要素 | 现状 |
|---|---|
| 固定句式 | **已有**，但只在全员旁白（`_speakers_are_voiceover` 是 `all()` 语义）时启用 |
| 闭嘴指令 | 已有独立尾句形态；缺社区更稳的 while 内嵌从句 |
| 具体反应动作 | 无（freeze 分支只写反向禁令 "No gesture"） |
| **混合场景** | **不支持且被清洗**：只要画面内角色也说话，整镜走普通路径；`neutralize_offscreen_claims` 会丢画外表述并断言 "All speaking characters are visible in the frame." |
| 说话人申报 | `_VOICEOVER_NAMES` 只认「旁白/narrator」；`小D（画外）：` 不命中 |
| 每镜缺席 | 无语法——可见名单 = 布局∪说话人，全员每幕在场钉死 |
| 混语言台词 | 社区案例正是 EN/ZH 逐句交替；storyboard 的 `language` 是全片单值 |

## 2. 不做什么

- **不动全员旁白 freeze 分支**（模板被测试逐字锁定，行为正确）；
- **不动非 POV 板的任何字节**——现有金样（板栗 EN/ZH、prompt(15)）改动前后
  `plan_json` 必须逐字节不变（新模板只在存在 `offscreen` 申报时激活）；
- **不改资产 schema**：资产本身照旧（肖像/voice/audio），但**画外角色
  不再占参考图**——见 §3.1 的「两个编号空间」；
- **不做镜头拆分自动化**（社区次选方案）：拆镜是用户编剧决策，不在组装层做；
- **不碰渲染链**。

> **2026-10-20 修订（用户裁决）**：画外音说话人的资产**可选**。
> 接上 → 用它的音色（`(Sn)` + 声纹描述照常写）；不接 → **不声明音色**，
> 句子里没有 `(Sn)` 也没有声纹描述，由 H3 自己选音色（运行摘要会提示
> 这一点）。同时画外音**永不占 `<Picture>` / `<Subject>`**——他永远不被
> 渲染，把他的参考图交给 H3 正是当初要发明「占位卡」去规避的风险。
> 于是「画外音板」在画面上真的是单角色板。

## 3. 数据模型

### 3.1 说话人 presence（板级申报）

- **storyboard/V3**：`cast` 条目加 `"offscreen": true`（默认 false）。
  validator：`offscreen` 角色不得出现在任何 `spatial_layout` / `absent` 里；
  **每镜至少一名 on_screen 角色在场**（全空镜 = 已知空镜缺陷形态，报错）。
  **画外角色的资产可选**（2026-10-20）：`character_N` 空着不接不是错误，
  只是不声明音色。
- **两个编号空间**（2026-10-20 修订的关键）：
  - **画面空间** `<Picture N>` / `<Subject K>`：**只数在画面内的角色**
    （按 `character_slot` 序），然后是场景。画外角色**永不占图**。
  - **声音空间** `(S<n>)` / `ref_audio_N`：**只数有声音的角色**
    （按 `character_slot` 序）。`(S<n>)` 是**音频位次**，不是
    `character_slot`——画外角色不接线时若留个洞，H3 会绕洞重编号，
    plan 里的 `(S2)` 就会落到错误的音轨上。
  - 全员接线时两个空间重合，所以非 POV 板的字节不变。
- **V2 输入语法**：说话人名带后缀 `（画外）` / `(off-screen)` / `(offscreen)`
  / `（画外音）`（全/半角括号均可）→ 解析层剥离后缀、产出 presence 标注。
  这是**输入格式语法**（同 `名：行` 本身），不是关键词分支；台词与说话人
  前缀是锁定数据，润色不触碰（既有契约）。
- 语义 facts / 增强层不猜 presence——只认显式申报。

### 3.2 每镜在场集合（storyboard 先行）

- `shots[]` 加 `"absent": ["猫爷爷"]`（默认 `[]`）。validator：`absent` ⊆
  cast 且不得含 `offscreen` 角色。
- **V2 的每镜缺席本版不做**（其可见名单来自布局∪说话人，改动面在 facts 层，
  排二期；见 §10-3）。

### 3.3 逐句语言覆盖（混语言台词）

- `lines[]` 加可选 `"lang": "zh" | "en"`，缺省用板级 `language`。
  `<d>` 标签逐句取值。V2 轨道维持现状（`_detect_dialogue_language` 逐句）。

## 4. 组装层（共享咽喉，V1/V2/V3 同时生效）

### 4.1 行模板三形（只在板内存在 offscreen 申报时启用新形）

```
画面内（POV 板追加正向口型句）:
  {名} speaks as (S{n}) {voice} <d>[lang] {台词}</d> {名}'s lips move
  naturally with the line.

画外音（社区第 1 优先级写法逐字采用；资产接了）:
  An off-screen voice (S{n}), {voice}, says in an off-screen voiceover:
  <d>[lang] {台词}</d> while {画面角色}'s lips remain completely closed;
  {反应动作}.

画外音（资产没接——不声明音色，H3 自选）:
  An off-screen voice says in an off-screen voiceover:
  <d>[lang] {台词}</d> while {画面角色}'s lips remain completely closed;
  {反应动作}.

全员旁白 freeze: 现状模板原样。
```

- `{voice}` 每句都带（机械、字节可复现，不做首现省略）；**资产没接时
  整段 `(S{n}), {voice},` 都省略**（2026-10-20）。
- `{反应动作}`：优先用该句已有的动作槽（storyboard `lines[].picture` /
  V2 stage cue）；缺省回退 `"she only listens with a slight nod"` 的
  **去性别化固定句**：`{名} only listens with a slight nod`。不做代词推断。
- 非画外行的正向口型句只在 POV 板追加（§2 字节稳定红线的直接推论）。

> **2026-10-20 真机实测：H3 侧的三条硬事实**（渲染 5 次逐帧取证）
>
> 1. **H3 自己出声，不存在独立 TTS。** 链路的 `audio_source: plan`
>    走的是「按 plan 的时间轴拼 H3 自己生成的声音」。所以 `<d>` 台词
>    能不能被听懂，取决于 **H3 用参考音频的音色去念这句话**。
> 2. **H3 用参考音频的语言念台词。** 小D 的参考样本是中文，于是他的
>    英文台词被念成中文：「Please help me resolve this document.」→
>    「请帮我帮忙把这些文件解决」。**每个说话人需要一段与自己台词
>    同语言的参考音频**——这是画外音板能不能出声的前提，不是润色。
>    库里的样本目前**全是中文**（无一英文），所以英文画外音必须先补
>    一段英文样本。
> 3. **画面内角色的动作槽不要带「发声」含义。** 动作槽排版在画外音
>    台词**之后**、画面内台词**之前**，这个位置上的「with a teasing
>    tone」会被 H3 当成「有人在发声」而把**画面内角色**的嘴对上。
>    三次渲染的一致对照：
>    - `Claude smiles slightly.` → 画外音段嘴闭（0.0056）
>    - `Claude leans in a little.` → 嘴闭（0.0107）
>    - `Claude smirks with a teasing tone…` → **嘴开（0.0634）**
>    动作槽应当是**无声的身体动作**（偏头、抬手、眼神），不要写
>    tone / voice / murmur 之类暗示发声的词。这是作者的内容约定，
>    组装层不做关键词判断。
>
> 另：**画外音台词不要以画面内角色的名字开头**。带呼语时 H3 可能把
> 那个名字当成说话人，然后把整句编掉（实测「Claude, help me resolve
> this document.」→「Claude, let me ask you about the origin of this
> payment」；去掉呼语后内容正确，只是语言仍随音色走）。

### 4.2 summary 分支（POV/缺席板）

```
混合镜:   [reference generation] {画面内在场} is/are visible in the frame
          and speaking; {画外}'s voice is off-screen.
纯听镜:   [reference generation] {画面内在场} is visible in the frame,
          listening; {画外}'s voice is off-screen.
缺席镜:   [reference generation] {present} alone is/are visible in the frame…
```
可见名单 = per-shot 在场集合（cast − absent − offscreen 的画面内成员），
不再恒等于全员。

### 4.3 缺席声明（明确优于省略）

缺席镜 `detailed_description` 里，紧跟 `[Shot 1]` 行后逐人写：

```
{名} is not in the frame in this shot.
```

该镜的散文/空间钉/summary 只写在场者；**裸名别名、`<Picture>` 绑定、
retention 保持全员**（身份锚省略是历史缺陷源；参考图整链都在）。
独白镜消失缺陷的教训：省略 = 让 H3 猜。

> **2026-10-20 修订**：上面这条「保持全员」只适用于**缺席**角色
> （他在别的镜头里确实在画面里，身份锚要留着）。**画外角色不同**：
> 他永远不在画面里，所以没有 `<Picture>`、没有 `<Subject>`、没有
> retention 行——把一张他的参考图交给 H3，正是当初要发明「占位卡」
> 去规避的风险。他保留的只有**裸名别名**（`小D: <资产描述>`），因为
> 资产描述里往往写着「他绝不出现在画面里」这句关键指令。

### 4.4 `[Shot 1]` POV 模板（板级存在 offscreen 时替换双人模板）

```
[Shot 1] First-person point of view: the lens sits where the off-screen
speaker is. {画面角色} faces the camera at a stable conversational
distance, eyes on the lens, speaking to the unseen speaker. The camera
holds still.
```

历史注记：旧反 POV 清洗（12:49）针对的是**未申报**的画外/镜头声称；
本模板是申报后的结构化 POV，两者不冲突（见 4.5）。

### 4.5 `neutralize_offscreen_claims` 条件化

签名加 `declared_offscreen: bool = False`。为 True 时：**仍丢**散文里的
画外/POV 声称（presence 是结构数据，散文声称仍是缺陷类），但**不再插入**
"All speaking characters are visible in the frame." 断言（它与申报矛盾）。
False 时行为逐字节不变。

### 4.6 addendum POV 契约（V2 轨道）

固定句式 / while 闭嘴从句 / 具体反应动作 / `(Sn)` 跨镜稳定 / 画外与画面
台词之间留一拍（时间错开）。时间错开同时在 cut-budget advice 通道加一条
重摇建议（渲染串口型时单镜重摇），不动 packing 算法。

## 5. 代码落点

| 改动 | 位置 |
|---|---|
| presence/absent/lang 校验 | `nodes/assets/storyboard.py`（validator + 常量） |
| 三形行模板 + summary 分支 + POV `[Shot 1]` + 缺席声明 | `nodes/llm/minimax_h3_loop_prompts.py`（新函数，`install_shot_speech` 分流调用）；storyboard body 同一分流 |
| `neutralize_offscreen_claims` 条件化 | 同上（默认参数保旧字节） |
| V2 `（画外）` 后缀解析 | `nodes/llm/dialogue_segmenter.py`（结构解析）+ 提取器提示词一句 |
| addendum POV 契约 + 时间错开建议 | `nodes/llm/prompts/h3_loop/ref2v_addendum.txt` + `_cut_budget_advice` 通道 |
| V3/storyboard 组装接入 | `nodes/llm/storyboard_builder.py`（presence/absent/lang 透传） |

## 6. 测试

1. **三形模板逐字锁定**（含 while 从句、反应缺省句、正向口型句）；
2. **金样字节稳定**：板栗 EN/ZH + prompt(15) 夹具改动前后 `plan_json`
   逐字节相等（红线回归）；
3. **POV 金样**：新夹具 `storyboard_pov_mixed.json`（画外 + 画面内 + 逐句
   lang 混排 + 一镜 absent），断言：画外句形态、闭嘴从句、反应动作来自
   `picture` 槽、summary 纯听镜措辞、缺席声明行、别名/绑定/retention 全员；
4. **validator**：offscreen 进 layout/absent 报错；全空镜报错；absent 含
   offscreen 报错；absent 未在 cast 报错；lang 非法值报错；
5. **neutralize 条件化**：declared=True 不出断言行、False 逐字节旧行为；
6. **V2 后缀解析**：`小D（画外）：` → 说话人「小D」+ offscreen；半角/全角
   括号；无后缀名字不受影响（含名字本身带括号限定语如「（嫌弃）」不误判——
   只认画外白名单词）；
7. **(Sn)/audio 槽位**：画外角色编号=槽位、`audio_n` 对齐不因 offscreen 变。

## 7. 实施顺序

1. validator + storyboard schema（presence/absent/lang）；
2. 组装层三形模板 + summary 分支 + POV `[Shot 1]` + 缺席声明 +
   `neutralize` 条件化（先跑 §6.2 字节稳定红线）；
3. V2 后缀解析 + addendum 契约 + advice；
4. POV 金样夹具 + 全部测试；
5. 双机同步 + 本地真重启（三验）；
6. 真机验收：POV 分镜渲一遍，帧检查画面角色在画外台词段**不动嘴**
   （社区痛点即验收项）；串口型则按重摇建议处理。

## 8. 已定决策

- 非 POV 板字节零变化是红线（模板激活条件 = 板内存在 offscreen 申报）；
- presence 只认显式申报（语法/字段），facts 与增强层不猜；
- 画外角色资产 schema 不动，槽位语义不变；
- 缺席用明确声明行，别名/绑定/retention 保持全员；
- 反应动作缺省句去性别化（重复名字，不推代词）；
- 全员旁白 freeze 分支与锁定测试原样。

## 9. 给评审人的问题

1. 画面内行的正向口型句（"lips move naturally with the line"）只加在 POV
   板——确认非 POV 板不加（否则金样字节全变）；
2. 逐句 `lang` 覆盖是否也开放给非 POV 板（当前方案：开放，正交能力）；
3. V2 的每镜缺席排二期是否可接受（当前方案：可以，V2 可见名单机制在
   facts 层，动它波及布局补全链）；
4. `{voice}` 每句重复 vs 首现全量后续省略（当前方案：每句重复，换字节
   稳定与机械性）；
5. POV `[Shot 1]` 模板中 "eyes on the lens" 与历史反 POV 断言
   （"None of them is the camera"）的并存措辞是否需要再润（当前方案：
   declared_offscreen 时不插入该断言，语义已自洽）。

## 10. 验收

- §6 全绿，其中 §6.2 金样字节稳定为**一票否决项**；
- POV 板真机渲染：画外台词段画面角色嘴部静止（帧取证），画面内台词段
  正常动嘴；缺席镜无幽灵角色、回归镜无残留；
- 全量离线测试通过；旧工作流（含板栗资产直出）行为零变化。
