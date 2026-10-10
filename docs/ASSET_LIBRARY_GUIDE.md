# 数字资产库使用说明

MieNodes 的资产库把「跨镜头要重复用的角色和场景」存成磁盘上一个普通文件夹，用 7 个节点读写。它不做检索界面、不做删除——它负责把一份角色（名字 + 人物描述 + 音色描述 + 一张定妆图 + 一段音色样本）和一份场景（名字 + 一张场景图 + 一句描述）原样存取，并把它们作为**数据**交给 H3 提示词生成器：角色是「谁在演」，场景是「故事发生在哪里」。

菜单位置：`🐑 MieNodes/🐑 Assets`

设计原始计划见 [`PLAN_ASSET_LIBRARY.md`](PLAN_ASSET_LIBRARY.md)（含评审记录和已定决策表；**第四次修订（2026-10-09）**记录了本文描述的三项决策：必填 `appearance`、场景带图、花名册接 H3——与本文一致，与该文前两节的原始边界不一致处以第四次修订为准）。本文讲**代码是怎么实现的、节点有哪些、实际怎么用、怎么接 H3**。

---

## 1. 磁盘格式

```text
<library_dir>/
  characters/<名字>/character.json      名字 + 人物描述 + 音色描述
  characters/<名字>/portrait.png        定妆图
  characters/<名字>/voice_sample.wav    音色样本（节点写入的格式）
  characters/<名字>/voice_sample.mp3    音色样本（手工放进去的格式，二选一）
  scenes/<名字>/scene.json              名字 + 场景描述
  scenes/<名字>/scene.png               场景图
```

`character.json`：

```json
{"name": "夏栀",
 "appearance": "A young East Asian woman in her early twenties with long dark brown hair ...",
 "voice": "adult female, clear and slightly warm timbre, low-mid register, unhurried pace"}
```

`scenes/<名字>/scene.json`：

```json
{"name": "便利店", "description": "A convenience store at midnight, cold fluorescent tubes, an old ice cooler at the end of the aisles."}
```

场景和角色是同一个模子：**一张图 + 一段手写英文描述**，角色多一个音色字段和一段音色样本，场景没有（它不发声）。`description` 必填、上限 800 字符、原样注入提示词，和 `appearance` 完全同一条规矩。写入也是同一套两文件提交点——先 `scene.png` 再 `scene.json`，JSON 是提交点。

### 两个 text 字段写英文，原样注入

`appearance` 和 `voice` 都会被**逐字**插进 H3 提示词，不经过任何翻译。管线里原来的多模态 captioner 也是固定要求输出英文的，所以这个约定跟原来一致。要跟 `output_language=en`（默认值）对齐。

- `appearance`：**必填**，非空，上限 800 字符。写 60–100 个英文词就够——这个长度是沿用 captioner 原来的要求。它替掉的正是那次视觉调用：角色长什么样由你写，不让模型看图猜。
- `voice`：非空，上限 200 字符。只能是音色描述，**不要写外貌**——它会紧贴在台词的 `<d>` 标签前面，写衣服进去就是噪声。

**老资产的迁移**：缺 `appearance` 的 `character.json` 会**大声报错**，不会静默退回 caption。理由是：一份 plan 里同时存在「手写真相」和「机器猜的真相」，就会烧出两个互相矛盾的身份（这是 H3 管线里"钉身份"机制当初要治的 bug）。要么全手写，要么全 caption。报错文本会告诉你怎么补，补法是直接往 JSON 里加一行，或者用 `Save Character Asset` 重存一次。

**库目录每台机器固定、不用手填**（第六次修订）：默认 `$MIE_ASSET_LIBRARY` 环境变量，未设则 `~/mie_asset_library`。所有节点上 `library_dir` 已隐藏，点「📁 库目录」按钮可查看/临时编辑（老工作流里已填的路径继续生效）。**库必须放在插件目录外面。** `scripts/sync_to_comfyui.py` 是镜像同步——目标端多出来的文件会被删掉。库存写在 `custom_nodes/ComfyUI-MieNodes/` 里，下一次部署同步就没了，不是被覆盖而是直接消失。代码用 `library.py` 自己的 `__file__` 往上三层定位插件根，开发仓库和部署副本共用同一条判断，不写死某台机器的路径。

---

## 2. 代码是怎么分层的

| 文件 | 职责 | 依赖 |
|---|---|---|
| [`nodes/assets/library.py`](../nodes/assets/library.py) | 校验、原子写、读、列目录、内容哈希 | 只用标准库 + `pathlib`。不 import ComfyUI、torch、`nodes.llm` |
| [`nodes/assets/nodes.py`](../nodes/assets/nodes.py) | 7 个节点，张量 ↔ PNG / WAV 的转换 | numpy、torch、Pillow、`wave`（`mp3` 读取时懒加载 `soundfile`） |
| [`nodes/assets/__init__.py`](../nodes/assets/__init__.py) | 导出 7 个节点类 | — |
| 根 [`__init__.py`](../__init__.py) | 注册到 `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` | 与仓库其他节点同一套 `add_suffix()` / `add_emoji()` |
| [`tests/test_asset_library.py`](../tests/test_asset_library.py) | 40 个测试，覆盖第 3、4、5 节讲的所有行为 | pytest `tmp_path`，不需要起 ComfyUI |
| [`tests/test_h3_loop_asset_roster.py`](../tests/test_h3_loop_asset_roster.py) | 33 个测试：花名册解析、零视觉调用、场景槽位、模板回归 | 同上 |

分层的意义在测试：`library.py` 是纯文件系统的，所以整套磁盘行为（原子性、大小写合并、损坏目录、失败窗口）都能在几毫秒内跑完，不必启动 ComfyUI。图片和音频在磁盘层只以**字节**进出，张量换算留在节点层。

### 五个值得记住的实现细节

**1. 三文件提交点。** 保存角色不是直接写文件，而是先把 3 个内容都写成同目录下的临时文件（`tempfile.mkstemp(dir=...)`，保证同一文件系统），再按固定顺序 `os.replace`：先 `portrait.png`，再 `voice_sample.wav`（顺手删掉被取代的残留 `.mp3`），最后 `character.json`。`character.json` 是**提交点**——它被替换的那一刻整份资产才算新。每一步的失败窗口都在错误里写明当时磁盘上是什么状态：

| 失败发生在 | 磁盘上留下什么 |
|---|---|
| 替换 PNG | 三个文件全是旧内容 |
| 替换音色样本 | 新 PNG + 旧样本 + 旧 JSON |
| 替换 JSON | 新 PNG + 新样本 + **旧音色描述** |

三种情况都抛错，`status` 不会返回成功。

**2. 大小写合并靠列目录，不靠 `exists()`。** 保存 `Tiger` 之后再存 `tiger`，`_find_casefold_sibling()` 会遍历 `characters/` 找到 `Tiger` 并覆盖它，目录名保持第一次的拼写。这条在 Linux（区分大小写）上尤其重要：用 `exists()` 的实现会新建第二个目录，把一个角色变成两个。

**3. 缓存策略分两类。** 保存和列出的节点 `IS_CHANGED` 返回 `float("nan")`——库随时可能在 ComfyUI 之外被改（资源管理器、记事本、别的机器），这两个操作每次排队都必须真跑。读取节点反过来用**内容哈希**：`character_cache_key()` 哈希 JSON + PNG + 样本三个文件的 sha256，任一文件在磁盘上被改动哈希就变，ComfyUI 就重跑；内容没动就吃缓存，不会因为反复排队而白算。哈希函数**从不抛错**，任何异常（缺失、读不了、非法）都返回固定的 `MISSING_CACHE_MARKER`，避免把一次失败缓存成成功。

**4. AUDIO 约定。** `nodes.py` 里的音频约定是 `{"waveform": [1, C, N] float in [-1, 1], "sample_rate": int}`——和 `QwenTTSNode` 产出的、`WavConcatNode` 消费的是同一个格式。保存节点用标准库 `wave` 写 16-bit PCM WAV（不引入任何编码器依赖），读回时误差不超过 `1/32767`。`.mp3` 只出现在**读取**路径，用 `soundfile` 懒加载解码，所以只用 WAV 的环境根本不会 import 它。图像方向同理：float 0–1 → `*255` → uint8 → PNG，误差不超过 `1/255`；保存节点返回的 IMAGE 是**从落盘 PNG 读回的**，也就是说它就是之后 Load 会给你的那张量化图，两边严格一致。

**5. 批量加载不跳过、不凑合。** `Load Asset Batch` 一次读多个资产，缺任何一个就抛错并带上名字。跳过一个会让后面所有 slot 编号整体错位，把 A 的图绑到 B 身上——这比直接失败危险得多。图片尺寸不一致也会报错（`torch.cat` 自己的错误只说「tensor」不说人名），错误里会写明每个尺寸对应哪些资产。

---

## 3. 七个节点

### Save Character Asset（`SaveCharacterAsset`）

| 输入 | 类型 | 说明 |
|---|---|---|
| `library_dir` | STRING | 库的绝对路径，必须在插件目录外 |
| `name` | STRING | 1–40 字，只能是中日韩汉字、英文字母、数字、`-`、`_` |
| `appearance` | STRING (multiline) | 英文人物描述，1–800 字。**必填** |
| `voice` | STRING (multiline) | 英文音色描述，1–200 字 |
| `image` | IMAGE | 定妆图，**必须恰好 1 帧**（批次会报错，先挑一帧） |
| `voice_sample` | AUDIO | 音色样本，形状必须是 `[1, C, N]` |

输出：`name` (STRING)、`voice` (STRING)、`image` (IMAGE)、`status` (STRING = `created` / `overwritten`)、`appearance` (STRING)。
`OUTPUT_NODE = True`，`IS_CHANGED` 为 `nan`——每次运行都真写。

### Load Character Asset（`LoadCharacterAsset`）

输入 `library_dir`、`name`。输出 `name` (STRING)、`voice` (STRING)、`image` (IMAGE)、`voice_sample` (AUDIO)、`appearance` (STRING)。
`IS_CHANGED` = 内容哈希。

### Load Asset Batch（`LoadAssetBatch`）

接 H3 用的是这个，不是单个 Load。一次把角色和场景都装进来。

| 输入 | 类型 | 说明 |
|---|---|---|
| `library_dir` | STRING | 同上 |
| `names` | STRING (multiline) | 一行一个资产名，角色和场景混着写。可以从 `List Character Assets` / `List Scene Assets` 的 `names` 输出连线过来 |

| 输出 | 类型 | 说明 |
|---|---|---|
| `images` | IMAGE | 按 `names` 顺序拼成的批次 |
| `names` | STRING (multiline) | 逐行对齐，人看的 |
| `roster_json` | STRING | 喂给 V2 生成器的结构化数据 |
| `status` | STRING | 成功写明加载了几个角色几个场景 |

`IS_CHANGED` = 所有资产文件哈希的组合，**任一文件改动或任一名字换位**都会让缓存失效。

`roster_json`：

```json
[{"name": "夏栀", "kind": "character", "slot": 1,
  "appearance": "A young East Asian woman ...",
  "voice": "adult female, clear and slightly warm ..."},
 {"name": "便利店", "kind": "scene", "slot": 2,
  "description": "A convenience store at midnight, cold fluorescent tubes ..."}]
```

`slot` 就是该资产在 `images` 批次里的位置（1-based）。**这是资产库和 H3 管线之间唯一的契约**，两条输出必须一起接。每个条目带 `kind`（`character` / `scene`），只携带自己类型的字段。**一个名字同时存在于 `characters/` 和 `scenes/` 下会报错**——不猜它到底是哪个。

### List Character Assets（`ListCharacterAssets`）

输入 `library_dir`。输出 `names` (STRING，每个名字一行，已排序)、`status` (STRING)。
`status` 全是 `ok` 时写 `ok`；有跳过项时逐条写明目录名和原因（例如 `skipped 便利店: missing portrait.png`）。没有 `characters/` 目录时 `names` 为空串、`status` 为 `ok`。`IS_CHANGED` 为 `nan`。

### Save Scene Asset（`SaveSceneAsset`）

输入 `library_dir`、`name`、`description`（英文，1–800 字）、`image`（恰好 1 帧）。输出 `name`、`description`、`image`、`status`。`OUTPUT_NODE = True`，`IS_CHANGED` 为 `nan`。

### Load Scene Asset（`LoadSceneAsset`）

输入 `library_dir`、`name`。输出 `name`、`description`、`image`。`IS_CHANGED` = 内容哈希（scene.json + scene.png 两个文件）。

### List Scene Assets（`ListSceneAssets`）

输入 `library_dir`。输出 `names`、`status`。语义与角色 List 相同，列的是 `scenes/` 下的目录。

---

## 4. 怎么用

**所有资产节点都是选择器**（前端辅助，第五/七次修订）：

- **下拉 + 常驻库名**：`name` 是下拉（只列本类型资产）；节点上的「📁 库…」按钮常驻显示解析后的库名（tooltip 是完整路径），点按展开可编辑的路径框。🔄 随时重扫；保存执行成功后列表和缩略图自动作废刷新。
- **保存 = 先看见再覆盖**：选已有名字时，节点**自动把库里的正文回填**到输入框，并逐字段标记「与库一致 / **将覆盖**」——你改了什么、将要覆盖什么，在队列前就看得见。选 `（✏️ 新名字…）` 新建。**图片/音频插座可不接**：覆盖时不接 = 磁盘上的图和音色样本逐字节保留（改一句描述不用重接图）；新建时仍必填。
- **读取 = 带预览**：选中名字即显示缩略图 + 正文摘要（Load Character 还显示音色），预览走 HTTP 不进队列。
- **批量 = 图片选择器 + 槽位条**：「🖼 选择资产…」打开卡片网格（缩略图 + 类型徽章；同名角色/场景的卡片标「同名冲突」不可选，不用等到执行才报错）。点卡片按顺序填槽。节点上是一条**缩略图槽位条**：序号、名字、类型、✕ 移除，**拖动排序**（顺序就是 `<Picture N>`）。`names` 文本默认收起，「✎ 名单」展开手改——它仍是唯一序列化真相，老工作流的手打名单照常加载并渲染成槽位条。


### 场景 A：在工作流里建一个角色

```text
LoadImage ──► Save Character Asset
QwenTTSNode / 任意 TTS ──► voice_sample

Save Character Asset ──► status（看 created / overwritten）
```

先给 `library_dir` 一个空的绝对路径，例如 `C:\Users\你\mie_asset_library`（`characters/` 和 `scenes/` 会自动建）。

音色样本不一定要用 TTS 生成。手头有一段满意的音频，用文件管理器拖进 `characters/<名字>/` 命名成 `voice_sample.mp3`，再手写一个 `character.json` 就完成了入库——这条路完全绕过节点，保真度更高（不做 PCM 量化）。前提是**目录名和 JSON 里的 `name` 逐字相同**，否则读取会报错（这是有意的：改名要在文件管理器里同时改两处，代码不猜）。

反过来，节点保存会写 `voice_sample.wav` 并删掉同目录下的 `.mp3`，不会留下让下次读取歧义的两个样本。

### 场景 B：不用 ComfyUI 建库（脚本方式）

`library.py` 是零依赖的纯文件系统模块，可以直接当脚本 API 用，建库、补批量导入都不必开 ComfyUI：

```python
from nodes.assets import library

library.save_character(
    r"C:\Users\你\mie_asset_library",
    "夏栀",
    "A young East Asian woman with long dark brown hair ...",
    "adult female, clear and slightly warm timbre, unhurried pace",
    png_bytes,          # 你磁盘上的图片读成 bytes
    wav_bytes,          # 16-bit PCM WAV 的 bytes
)
print(library.list_characters(r"C:\Users\你\mie_asset_library"))
print(library.load_character_batch(r"C:\Users\你\mie_asset_library", "夏栀\n便利店主"))
```

（从仓库里这样 import 时会先执行 `nodes/assets/__init__.py`，连带把 torch / Pillow 也拉进来——想只用到纯文件系统的部分，直接把 `library.py` 拷走或按文件路径加载即可，它本身只依赖标准库。）

返回的 dict：`save_*` 给 `name` / `appearance` / `voice`|`setting` / `status`；`load_character` 给 `name` / `appearance` / `voice` / `png_bytes` / `voice_sample_bytes` / `voice_sample_ext`；`load_character_batch` 给 `characters`（每个带一个 `slot`）；`list_*` 给 `names` / `status`。

### 场景 C：接到 H3 提示词链（V2 生成器）

这是资产库和 H3 的正式接口。接线：

```text
List Character Assets ──names──┐
List Scene Assets ───names─────┤（拼进同一个 names 文本）
                               ▼
LoadAssetBatch ──images────────► MiniMaxH3LoopPromptGeneratorV2.images
              └─roster_json───► V2.asset_roster

V2 ────user_input（只写剧情：谁在哪、发生什么、说什么）
V2.plan_json ─────────────────► MiniMaxH3ChainPlanModern.plan_json_input
```

`reference_mode` 选 **ref2va**。`user_input` 从此只写剧情——「在哪里」由场景资产回答。

**接上之后管线不再猜这五件事：**

| 以前 | 现在 |
|---|---|
| 定妆图按批次顺序编成 `<Picture N>`，不知道 N 是谁 | `roster_json` 的 `slot` 直接对应批次位置，绑定是数据不是推断 |
| 角色长相靠多模态 LLM 看图打标（一次视觉调用 + 磁盘缓存） | `appearance` 逐字进入提示词，**这一条视觉调用被完全跳过** |
| 长相描述进 CAST 时会和「按名字猜的长相」打架（live 2026-09-21 爸爸和妈妈各裂成两只猫） | 手写的 `appearance` 就是唯一真相，现成的「钉身份」机制自动接管 |
| 说话人音色靠 prefix LLM 猜，猜不出就按名字里的性别词兜底 | `voice` 逐字贴在台词 `<d>` 标签前，优先级高于模型的一切猜测 |
| 「故事发生在哪里」散落在概念稿散文里，每条工作流抄一遍 | 场景资产占一个 `<Picture N>` 槽位，`description` 逐字注入，role 是 `environment` |

**场景槽位的特殊性**（这是设计，不是限制）：

- 场景槽位的 manifest role 是 `environment`，**不占 `<Subject N>` 编号**（`<Subject>` 只数 identity 槽）。模型把它当「动作发生的地方」用，不会试图在环境上钉角色身份。
- 场景不进 CAST 覆盖列表（把环境钉到角色身份行上是错的），也不进音色表（场景不发声）。
- 场景的 `description` 同样跳过视觉打标。**「2 个角色 + 1 个场景」的板子跑下来，视觉调用是 0 次。**
- 提示词里的规则分两段写：角色段讲「逐字照抄、不得新增衣着颜色」；场景段讲「这是环境参考、按描述写空间与光照」。两套规则不能混——「不得新增衣着」对场景是废话。

摘要（`summary` 输出）会写明每一条走的是哪条路：`asset roster: Picture 1 uses the authored description from the character asset (vision caption skipped)`、`identity pin: ... replaced with the character asset's authored description`、`asset voice: 'X' uses the authored voice ... instead of the prefix LLM's sheet`。

**接线错了会响，不会猜：**

- roster 有条目但 `images` 空 → 报错并告诉你接 `Load Asset Batch` 的 `images`。
- roster 的 slot 超出实际图片数（最常见的错：批量节点接了但 images 忘了接）→ 报错写明"只有 N 个槽位"。
- roster JSON 坏了 / 重名 / 重 slot / slot 不是整数 → 解析阶段报错，**在任何一次 LLM 调用之前**。
- 混合批次是允许的：roster 覆盖的槽位走手写描述，其他槽位（比如一张环境参考图）照旧走 captioner，两种情况同一条代码路径。

**`plan_json` 的格式一个字都没变**，`validate_plan` 和标签策略一行没改，下游 `MiniMaxH3ChainPlanModern` 的接法不变。V1 对不接 roster 的板子：镜头用户模板渲染与改动前**逐字节相同**（`test_empty_roster_template_is_byte_identical_to_pre_roster` 锁定——空 roster 块就是原来那行空行）；但两份共享系统提示词（`prefix_synth_system.txt` / `ref2v_addendum.txt`）的措辞变了（「caption 是真相」改成「手写花名册行或 caption 都是真相」），无 roster 时语义等价、文本有变，所以**不承诺** V1 的 LLM 输出逐字节复现旧版本。

**音频这一期没接。** `voice_sample` 还在库里存着，但 V2 不输出 AUDIO，plan 里也不写 `<Audio N>`：ref2va 下这个标签被代码显式禁止（有过把旁白绑到 Audio 1 之后音色崩塌的历史），而且 native ref2va 与外部 TTS 两条音频路线还没定。等路线定了再作为独立开关加。

### 场景 D：接到 H3 参考视频节点（手工）

不用 V2 生成器、自己搭 H3 链的话，字段对应是这样（`MiniMaxH3ReferenceToVideo`，ComfyUI 核心 `comfy_extras/nodes_minimax_h3.py`）：

```text
Load Character Asset ── image ─────────► ref_images.ref_image_N   （Autogrow，max 9）
                     ├─ voice_sample ──► ref_audios.ref_audio_N   （Autogrow，max 3）
                     ├─ name   ──┐
                     ├─ voice  ──┼──► 手工拼进概念稿
                     └─ appearance┘
```

- **H3 自己克隆音色。** 音色样本走 `ref_audios`，模型 VAE 编码整段参考音频（`waveform[:1]` 容忍批次维，任意采样率自动重采样）。所以 QwenTTS 在这套里的角色是**样本工厂**（合成一段你喜欢的音色存进库），不是生成时配音的 TTS。
- **多个角色的定妆图要组批**用 `LoadAssetBatch` 的 `images` 输出，或者 ComfyUI 核心的 **Batch Images**（`BatchImagesNode`）。仓库里的 `LoadImageBatch` / `SaveImageBatch` 读写的是 tensor_store 的 `.pt`，不是拼批次。
- 这条路全靠手工接线，绑定关系没有机器保证——能用 V2 就用 V2。

### 场景 E：列出并接入 name 下拉

`List Character Assets` / `List Scene Assets` 的 `names` 输出是多行 STRING，可以直接连进 `LoadAssetBatch` 的 `names`。要喂给别的节点当选择列表，接一个文本拆分 / 行转列表类节点即可。库本身**故意没有**下拉 UI——那需要前端 JS 扩展，越过了「不做检索界面和前端扩展」这条边界。

### 场景 F：场景资产

场景资产 = 一张场景图 + 一句英文描述。接 H3 时不用单独接 `Load Scene Asset`——把场景名字混进 `Load Asset Batch` 的 `names`（见场景 C），它就占了 roster 里一个 `kind: "scene"` 的槽位，成为「故事发生的地方」。`Load Scene Asset` 单独用的时候是取出 `description` 接文本节点、`image` 接图片节点。本会话早先草稿里的 V2 `scene_setting` 插座（从未提交、从未发布）已删除：「在哪里」只有 roster 和 user_input 散文两个入口，不再有第三个。V1 从来没有过这个插座。

---

## 5. 排查表

| 报错 / 现象 | 原因 | 处理 |
|---|---|---|
| `library_dir is inside the plugin folder ...` | 库建在 `custom_nodes/ComfyUI-MieNodes/` 里面 | 换个插件目录外的绝对路径 |
| `library_dir must be an absolute path` | 填了相对路径或空串 | 填完整绝对路径 |
| `invalid asset name ...` | 名字含空格、点、斜杠、标点，或超过 40 字，或撞 Windows 设备名（`CON`、`NUL`、`COM1`…） | 只用汉字/字母/数字/`-`/`_` |
| `has no 'appearance' field` | 老资产，加 `appearance` 字段之前建的 | 往 JSON 里补一段英文人物描述，或用 `Save Character Asset` 重存 |
| `the portrait must be exactly one frame` | `image` 传进来一个批次 | 先挑出单帧 |
| `cannot batch pictures of different sizes` | 一批资产里图片分辨率不一致 | 统一导出到同一分辨率；H3 本来就把参考图缩到短边 2048，统一尺寸不损失什么 |
| `exists both as a character and as a scene` | 同一个名字在 `characters/` 和 `scenes/` 下都有 | 改掉一个名字，每个名字只能指一种资产 |
| `missing scene picture` | 场景目录缺 `scene.png` | 补图或重存 |
| `asset roster puts a character in slot N, but only M reference slot(s) exist` | roster 里的 slot 超出实际接进来的图片数 | 把 `LoadAssetBatch` 的 `images` 接上，或裁剪 roster |
| `asset_roster was provided but the images socket is empty` | 接了 `roster_json` 但忘了接 `images` | 两条输出一起接 |
| `needs 'kind' to be 'character' or 'scene'` | roster 条目缺 `kind` 或写错 | 重新跑 `Load Asset Batch`，或按报错补 |
| `asset_roster is not valid JSON` / `listed twice` / `slot N to both` | roster 格式坏了、角色重名、slot 撞号 | 按报错修；最省事是重新跑一次 `LoadAssetBatch` |
| `ambiguous voice sample: both ... exist` | 目录里同时有 `.wav` 和 `.mp3` | 留一个（节点保存会自动删掉旧的 `.mp3`） |
| `declares name 'x' but the directory is 'y'` | 手工改名后目录名和 JSON 里的 `name` 不一致 | 两处改成一样 |
| `... failed; the library now holds the NEW portrait and NEW voice sample with the OLD voice line` | 三文件提交的最后一步失败 | 重新成功保存一次即全部重写 |
| List 的 `status` 里出现 `skipped ...` | 该目录缺 JSON / 缺图 / 缺样本 / 缺 `appearance` / JSON 损坏 / 名字不合法 | 按提示补文件；补完再 List 就会出来 |

List 节点的设计是**跳过 + 说明**，不是整次失败：一个坏目录不会让整次列出失败。读取节点相反，任何问题都直接抛错，错误里带绝对路径。

V2 生成器的报错同理：roster 的所有形状问题都在**第一次 LLM 调用之前**就抛出来，不会烧掉 token 才发现 JSON 写错了。

---

## 6. v1 已知限制

这些是评审时明确接受的取舍，不是 bug（「name 是手打字符串」一条已在第五次修订后移除——现在是下拉）：

- **没有删除节点、没有检索界面、没有前端扩展。** 删除和改名用文件管理器；改名必须同时改目录名和 JSON 里的 `name`。
- **改人物描述 / 音色 / 场景描述要走 Load → 改 → Save 的回环**（或者直接编辑 JSON），因为 Save 需要全部输入齐全（角色还要音色样本，场景还要图）。
- **`appearance` 和 `voice` 必须是英文，且不翻译。** 这是这一版刻意选的：手写即最终形态，确定性最好、改一个字不用重跑。代价是中文用户要自己写英文——写不好就先让旧管线的 captioner 出一版，再改。
- **`voice_sample` 这一期没接 H3。** 库照存照取，V2 不输出 AUDIO、不写 `<Audio N>`，等音频路线定了再加。
- **库不在版本控制里时没有备份方案。** 它就是一个普通文件夹，请自己按需备份或纳入你自己的 Git 仓库。
- **没有资产之间的关系**（谁属于哪出戏、角色和场景的配对），也没有多套服装 / 道具 / 表演视频。这些都不在这一版的范围里。
