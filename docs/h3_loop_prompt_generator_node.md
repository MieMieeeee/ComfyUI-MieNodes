# MiniMaxH3LoopPromptGenerator 参数说明

ComfyUI 节点 `MiniMaxH3LoopPromptGenerator|Mie`，类路径 `nodes.llm.minimax_h3_loop_prompt_generator.MiniMaxH3LoopPromptGenerator`。把任意形式的 `user_input`（概念段、对话行、混排）变成 MiniMax H3 生产链消费的 `plan_json`，再串入 `MiniMaxH3ChainPlanModern.plan_json_input`。

**CATEGORY:** `🐑 MieNodes/🐑 Prompt Generator`
**FUNCTION:** `generate`
**OUTPUT:** `plan_json` (STRING), `summary` (STRING), `board_kind` (STRING)
**DEFAULTS 来源：** 模块常量 `_DEFAULT_TEMPERATURE` / `_DEFAULT_MAX_TOKENS` / `_DEFAULT_TIMEOUT`，可被 widget 覆盖。

---

## 1. 必需参数（required）

| 参数 | 类型 | 默认 | 取值 / 范围 | 说明 |
|---|---|---|---|---|
| `llm_service_connector` | LLMServiceConnector (link) | — | 任意已注册的 connector 节点（`SetMiniMaxTokenPlanLLMServiceConnector` 等） | 决定调哪个模型走 MiniMax-M3 / DeepSeek-R1 / GLM 等。**connector 的 api_token / model_select 决定实际请求**，节点本身不读 keys。 |
| `user_input` | STRING (multiline) | `""` | 任意文本 | 概念段、对话行、混排都行。**对话行是 locked contract：verbatim 进 `<d>[...]` 块，不会被翻译**（见 `enhance_user_input` 说明）。空字符串 = 用节点内置默认概念。 |
| `enhance_user_input` | 选择 | `"off - 不润色(默认)"` | `off - 不润色(默认)` / `on - 自动润色后再规划` | `on` 时先用 User Input Enhancer 改写再规划。下游解析器认得出的对白板，或整份草稿就是一行台词时，会缓存并逐字保留，改写不合格就钉回原句，不会为了改词再打模型。散文里的冒号行（`要求：…`）走创作，不会被钉成说话人。`off`（默认）= 原样使用 user_input。上游已经接了增强器时保持 off，避免再润色一次。改 `scene_count` 或总时长会重跑规划，但同一草稿、同一 `reference_mode`、同一连接器的润色走缓存。 |
| `seed` | INT | `0` | `0` … `2^64-1` | 场景 seed 链的基底（场景 N → base+N 转字符串）。`0` = 每次运行从时钟派生。`control_after_generate=True` 在 ComfyUI 端可以"运行后随机化"重投。 |

---

## 2. 可选参数（optional）

### 2.1 切片 / 时长

| 参数 | 类型 | 默认 | 取值 / 范围 | 说明 |
|---|---|---|---|---|
| `scene_count` | INT | `0` | `0` … `128` | `0` = auto。对话板：连续 turn 最小切分装进 14s H3 窗口（speaker 切换不强制切）；pacing 不影响切数，只影响语速 + BRISK/MEASURED 提示指令。旁白板：让 storyboard LLM 用 pacing 当切分 bias（fast=aggressive / slow=conservative）。`>0` 对话板：每场均匀分台词；超出 1 行/场 的部分变成 reaction 切（镜头给听话者）；**一句话永不被拦腰切**。`>0` 旁白板：恰好这么多场。 |
| `total_duration_seconds` | INT | `0` | `0` … `1800` | 整板时长预算。`0` = auto：按对话行数 + pacing 推 TTS 速率 + 每句停顿估出。无对话时回退 15s。`>0` 会 clamp 每场到 4–14s；超出 14s 会 preflight 警告（单生成窗口上限）。每场向上对齐到 17k+5 帧格，最终 ±20% 静默允许。 |
| `pacing` | 选择 | `"normal - 正常（语速·推荐）"` | `fast - 快（语速）` / `normal - 正常（语速·推荐）` / `slow - 慢（语速）` | **整板语速控制，不是切数控制。**对话板只影响 BRISK/natural/MEASURED 提示词 + auto 长度的 TTS 速率与停顿（CN fast 4.5 字/s + 1.0s 停顿，normal 3.5 + 1.5s，slow 2.8 + 2.0s）。旁白板同步影响 storyboard 切分 bias。**切数控制请用 `scene_count`，时长用 `total_duration_seconds`。** |

### 2.2 生成模式 / 内容类别

| 参数 | 类型 | 默认 | 取值 / 范围 | 说明 |
|---|---|---|---|---|
| `generation_mode` | 选择 | `"per_shot - 逐场生成(推荐)"` | `per_shot - 逐场生成(推荐)` / `single_call - 单次调用(快/省)` | `per_shot`：每场一次 LLM 调用，连续性最好（推荐）。`single_call`：整板一次调用（更便宜更快，但场间一致性下降）。 |
| `category` | 选择 | `"none - 不指定"` | `none - 不指定` / `dialogue - 对白/对话/相声` / `action - 动作戏/打斗/飙车` | `none` 默认不附加电影摄影合同；但只要抽到对话行就**自动**走 spoken-scene contract（中近景、过肩镜头、eyeline 匹配、无字幕、无水印）。`dialogue` 显式锁定该合同。`action` 加 motion-blur / camera-shake / 低角度 / quick-cut / 4-6s 战斗 beat。**对话 verbatim 合同在抽到行的任何 category 都生效**。 |

### 2.3 语言 / / 种子

| 参数 | 类型 | 默认 | 取值 / 范围 | 说明 |
|---|---|---|---|---|
| `output_language` | 选择 | `"en"` | `en` / `zh` | 旁白文案语言。`en` 默认推荐（H3 稳定）。**不影响 locked 对白行语种**——对白按行内字符自动判定（`_detect_dialogue_language`）。 |
| `seed_mode` | 选择 | `"per_scene_increment - 每场seed递增(推荐)"` | `per_scene_increment - 每场seed递增(推荐)` / `same_across_scenes - 全场同seed` | `per_scene_increment`：base+index 派 seed——deterministic 但每场不同；与上游 ComfyUI-MiniMaxH3-Context-Loop 默认行为一致（`sha256(f"{base}:{index}:{shot_id}")`），也支持 checkpoint recovery。`same_across_scenes`：全场同 seed——刻意要"重复动作节律"时才用。 |

### 2.4 LLM 调参（高级）

| 参数 | 类型 | 默认 | 取值 / 范围 | 说明 |
|---|---|---|---|---|
| `temperature` | FLOAT | `0.4` | `0.0` … `2.0`，step `0.05` | LLM 创造性 / 随机性。保持默认。`0.4` 是 H3 兄弟节点 stage-2 增强用的同一值，三段式合同稳定性已校验。 |
| `max_tokens` | INT | `16384` | `64` … `32768` | 单次 LLM 调用上限。思维链模型（MiniMax-M3 / DeepSeek-R1）需要 16k+ 头部防止 reasoning 吃光 answer；低于 8k 容易返空。 |
| `timeout` | 选择 | `300` | `60` / `120` / `300` / `600` | 单次调用超时（秒）。`60`/`120` 适合本地 Ollama 冷启动；公网 MiniMax-M3 / DeepSeek 推荐 300。 |

### 2.5 参考图 / 视觉链

| 参数 | 类型 | 默认 | 取值 / 范围 | 说明 |
|---|---|---|---|---|
| `reference_mode` | 选择 | `"t2va - 文生视频链(默认)"` | `t2va - 文生视频链(默认)` / `i2va - 首帧关键帧(文生视频+首帧图)` / `fl2va - 首尾帧关键帧(首帧+逐场尾帧)` / `ref2va - 参考图(N张/全场景)` | 选哪种参考图接线方式。t2va 忽略 images。i2va 只用 `images[0]`（其余被警告丢弃）。fl2va 用 `images[0]` 作首帧，`images[1..]` 作逐场尾帧（Chain Frame Index Switch frame_1..frame_{N-1}）。ref2va 全部 images 每场都激活（六段式 prompt 合同）。 |
| `images` | IMAGE (batch) | — | 1–9 张 | 对应 `reference_mode` 的接线位置（见上）。`t2va` 不需要图；`i2va` / `fl2va` / `ref2va` 必须接。场景过渡意图（A→B→A）写进 `user_input`。 |
| `caption_mode` | 选择 | `"cache_memory_disk - 缓存:内存+磁盘(推荐)"` | `cache_memory_disk - 缓存:内存+磁盘(推荐)` / `cache_memory_only - 缓存:仅内存` / `no_cache - 禁用缓存` / `force_recaption_once - 本次强制重打标` | 图片 caption 缓存策略。`cache_memory_disk` 推荐（跨 run 复用）。`force_recaption_once` 仅本次重新打标（之后回到 memory_disk 缓存）。 |

---

## 3. 输出

| 输出名 | 含义 |
|---|---|
| `plan_json` (STRING) | 完整的 MiniMax H3 生产计划。结构 `{ `shots`: [...], `prompt_prefix`: [...] }`。每个 shot 携带 `id`、`prompt`（数组，ref2va 是六段式，否则三段式）、`length`（帧数）、`seed`（按 seed_mode 派生）。`plan_json_input` 直接对接 `MiniMaxH3ChainPlanModern.plan_json_input`（link 自动接管）。 |
| `summary` (STRING) | 人类可读的总览：enhance rewrite（如开启）、stage0 切片决策、引用图 manifest 摘要、生成模式、每场 seed 链。 |
| `board_kind` (STRING) | 板的种类标记，供下游节点判断合同：`"dialogue"` / `"narration"` / `"reference"` 等。 |

---

## 4. 何时用哪个值

| 场景 | 建议配置 |
|---|---|
| **首次跑通** | `enhance_user_input=on`（粗稿也能规划）+ `pacing=normal` + `scene_count=0` + `total_duration_seconds=0` + `generation_mode=per_shot` + `reference_mode=t2va`（无图）/ `ref2va`（有 1–9 张参考图）。 |
| **canonical 文本**（你已写好结构） | `enhance_user_input=off`（防双重润色）+ 同样的 pacing / scene / duration 默认。 |
| **要更稳的连续性** | `seed_mode=per_scene_increment` + `seed` 给一个固定值（不依赖时钟）。 |
| **多图过渡**（A→B→A） | `reference_mode=fl2va` + 至少 2 张图 + 在 `user_input` 里写过渡意图。 |
| **全场参考**（同一组人物反复出现） | `reference_mode=ref2va` + 1–9 张图 + 六段式 contract（`subject_definitions` 配 `<Subject N> binds to <Picture M>`）。 |
| **本地 Ollama** | `timeout=120`（默认 60 不够 7B+ 模型冷启动）。`max_tokens=4096`（本地小模型思维链短）。 |
| **debug payload 压力** | 已自动：`POST` 起始行 + Timeout/5xx 重试行会带 `request_chars=... last_user_chars=...`（见 `services/llm.py::_payload_chars`）。 |

---

## 5. V2 变体：`MiniMaxH3LoopPromptGeneratorV2`

类路径同上，`MiniMaxH3LoopPromptGeneratorV2`。**参数与 V1 完全一致**（`INPUT_TYPES` 是从 V1 派生的，不是抄的，所以 widget 集合永远不会漂移），只多两个可选插座：

| 参数 | 类型 | 说明 |
|---|---|---|
| `asset_roster` | STRING (multiline) | 资产库交来的花名册 JSON（角色 + 场景）。接 `Load Asset Batch` 的 `roster_json`，或手写 |

`FUNCTION` = `generate_v2`。**输出和 V1 逐字节同格式**（`plan_json` / `summary` / `board_kind`），`validate_plan` 与标签策略一行未改，下游 `MiniMaxH3ChainPlanModern.plan_json_input` 的接法不变。

### roster 的形状

```json
[{"name": "夏栀", "appearance": "A young East Asian woman ...",
  "voice": "adult female, clear and slightly warm ...",
  "slot": 1}]
```

条目带 `kind`（`character` / `scene`）：角色携带 `appearance` + `voice`，场景携带 `description`（不塞 null）。`slot` 是该资产在 `images` 批次里的位置（1-based，最多 9）。形状问题（JSON 坏了、缺 `kind`、重名、撞 slot、slot 非整数、超范围）**在任何一次 LLM 调用之前**抛 `ValueError`。

### 接上之后行为怎么变

`user_input` 从此只写剧情——谁在哪、发生什么、说什么（「在哪里」由场景资产回答）。下面五件事从"模型猜"变成"数据说"：

| 事项 | V1（猜） | V2（有 roster 时） |
|---|---|---|
| 角色绑第几张图 | 手写 `图1的X`，靠 LLM 抽取、失败退正则（正则只认中文 `图N`） | roster 的 `slot`，优先级高于一切抽取结果 |
| 角色长相 | 多模态 LLM 看图打标（一次视觉调用 + 磁盘缓存） | `appearance` 逐字注入，**视觉调用被完全跳过** |
| CAST 身份行 | prefix LLM 按名字猜，与 caption 打架 | 钉身份机制自动接管（roster 绑定进了 `merged_bindings`，它本来就在遍历这个列表） |
| 说话人音色 | prefix LLM 的 VOICE sheet，猜不出按名字性别词兜底 | `voice` 逐字贴在 `<d>` 标签前，优先级高于模型 |
| 故事发生在哪里 | 概念稿散文里写，每条工作流抄一遍 | 场景资产占一个 `<Picture N>` 槽位（role=`environment`），`description` 逐字注入 |

字幕摘要里会逐条说明走的是哪条路：`asset roster: Picture N uses the authored description ...`、`identity pin: ... replaced with the character asset's authored description`、`asset voice: 'X' uses the authored voice ... instead of the prefix LLM's sheet`。

### 场景条目的特殊性

- 场景槽位的 manifest role 是 `environment`，**不占 `<Subject N>` 编号**——模型把它当「动作发生的地方」，不在环境上钉角色身份。
- 场景不进 CAST 覆盖（`roster_bindings` 只返回角色），也不进音色表（场景不发声）。
- 提示词规则分两段：角色段「逐字照抄、不得新增衣着颜色」；场景段「环境参考、按描述写空间与光照」。
- **「2 个角色 + 1 个场景」的板子视觉调用为 0**——场景描述同样是手写的。

### 混合批次与非 roster 槽位

- roster 只覆盖部分槽位时（比如 2 个角色 + 1 张临时环境参考图），**只有 roster 外的槽位走 captioner**。两条路径同一段代码，没有分支判断。
- `reference_mode=t2va` 没有参考槽位，roster 会被忽略并写一条 warning，不会静默消失。
- roster 非空但 `images` 空，或 slot 超出实际图片数 → 报错并给出接线办法。

### 不接 roster 时

不接 roster 时 V2 与 V1 走完全相同的路径。模板层面：镜头用户模板在空 roster 下渲染与 b1df1f6 **逐字节相同**（`test_empty_roster_template_is_byte_identical_to_pre_roster` 冻结了旧模板全文做比对）。要如实说明的一点：`prefix_synth_system.txt` 与 `ref2v_addendum.txt` 两份系统提示词为支持花名册改了措辞，它们由 V1/V2 共享——无 roster 的板子语义等价，但发给 LLM 的文本与旧版不完全相同，故不承诺旧工作流的 LLM 输出逐位复现。V1 节点类本身一行未改。

### 本期不做

`voice_sample` 音频不进 plan：ref2va 下 `<Audio N>` 被 `validate_label_policy` 显式禁止（有过把旁白绑到 Audio 1 之后音色崩塌的历史），且 native ref2va 与外部 TTS 两条音频路线尚未裁定。库里照存照取，等定了再作为独立开关加。

---

## 6. 与节点相关的文档

- `docs/ASSET_LIBRARY_GUIDE.md` —— 资产库（角色/场景）与 V2 生成器的接线
- `docs/h3_loop_live_e2e_runbook.md` —— sync / restart / API 提交 / 真实 ComfyUI 验证的完整跑通手册
- `docs/h3_loop_open_issues_voice_attribution.md` —— 历史未结案的口型归属 / 声纹一致问题清单
- 节点内嵌代码注释：上述每个 widget 的 `tooltip` 都是节点自带提示
- 全套测试：`tests/test_h3_loop_prompt_generator.py` + `tests/test_h3_loop_prompts.py` + `tests/test_h3_loop_english_dialogue_and_meta.py` + `tests/test_h3_loop_asset_roster.py`（1479 pass）