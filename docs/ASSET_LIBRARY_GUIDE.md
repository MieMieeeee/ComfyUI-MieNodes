# 数字资产库使用说明（V2）

MieNodes 的资产库把「跨镜头要重复用的角色和场景」存成磁盘上一个普通文件夹，7 个节点读写，并作为**数据**交给 H3 提示词生成器。规格以 [`PLAN_ASSET_LIBRARY_V2.md`](PLAN_ASSET_LIBRARY_V2.md) 为准（本文是它的使用说明）；上一版计划 `PLAN_ASSET_LIBRARY.md` 留作历史。

菜单位置：`🐑 MieNodes/🐑 Assets`

---

## 1. 库在哪：每台机器一个，不可选

- **库就在插件目录里的 `asset_library/` 子文件夹**（2026-10-10 裁定）。环境变量 `MIE_ASSET_LIBRARY` 非空时优先用它（开发机把库指到部署副本之外时用）。
- 旧的 `~/mie_asset_library` 会在**第一次使用时自动整体搬进**新位置（搬移不是拷贝，只发生一次）。
- 同步脚本对这个目录**既不复制也不镜像删除**（和 `mie_llm_keys.json` 同类的机器本地数据）；`.gitignore` 同样忽略它——每台机器、每份插件副本各有一份自己的库。
- **节点上没有路径可填**。`library_dir` widget 还在输入表里但永远隐藏（老工作流的下标不错位），执行、扫描、缓存键全部忽略它的值——工作流里存过的旧路径不再生效。
- 唯一的目录动作是「📂 打开文件夹」按钮：打开解析出的那一个目录（不存在则先创建），在文件管理器里做删除/改名。
- 节点上的预览/选择器头部显示解析后的完整路径，只供核对。

## 2. 磁盘格式

```text
<库>/characters/<名字>/character.json     {"name": ..., "description": ..., "voice": ...}
<库>/characters/<名字>/images/01.png      1–8 张 PNG，同宽同高，自然排序（2.png 在 10.png 前）
<库>/characters/<名字>/images/02.png
<库>/characters/<名字>/audio/01.wav       0–8 条；节点写 16-bit PCM WAV；
<库>/characters/<名字>/audio/02.mp3       手放的 .mp3 照读（同序号 .wav+.mp3 并存报错）
<库>/scenes/<名字>/scene.json             {"name": ..., "description": ...}
<库>/scenes/<名字>/images/01.png          1–8 张，同宽同高
```

- `description`（≤800 字）与 `voice`（≤200 字）是手写英文、原样进提示词。
- **同一资产内的图必须同宽同高**——一个 `IMAGE` 插座装不下两种分辨率；不满足时保存和读取都报错，错误里点名每张文件名和尺寸。
- 角色至少一张图；音频可以是零条。场景没有音频。

### 旧库自动兼容

2026-10-09 之前的库（`portrait.png` / `voice_sample.wav|mp3` / `scene.png`、JSON 字段 `appearance`）**直接可读**：按 1 图 + 1 音频对待，`appearance` 当作 `description`。对旧资产做任何一次成功保存都会写成新布局，并把旧文件**晋升**（原字节搬进 `images/01.png` / `audio/01.<原扩展名>`）后清理——只读永远不删。

## 3. 节点（四个，简化模型）

| 模型 | 内容 |
|---|---|
| 角色 | **一张图**（可以是三视图/角色设定图）+ 一句图片描述 + **一段音频** + 一句声音描述 |
| 场景 | **一张图** + 一句图片描述 |

（2026-10-10 深夜裁定：V1 多图/多音频节点已**整体移除**，只保留简化模型；旧工作流里的 V1 节点会显示为缺失。）

- `Save/Load Character Asset`：单图单音频；每个输出之外还有一根 **`asset`** 线（MIE_CHARACTER_ASSET 资产包，一根线带全 name/description/voice/image/audio）。
- `Save/Load Scene Asset`：单图；同样有 `asset` 线（MIE_SCENE_ASSET）。
- 保存节点：下拉只列本类型；选中已有名字会预览库里的内容（缩略图 + 描述 + 音色，**只展示、不做一致性判断**）；「✏️ 新名字…」新建。
- **图和音频不用接线**：节点上有两个醒目的媒体卡片（🖼 定妆图、🎵 音色样本），每张卡片 = 一个文件下拉 + 一个渐变色**上传按钮**。支持：点按钮选文件、把文件**拖到卡片上**、聚焦卡片后 **Ctrl+V 粘贴**；上传成功自动选中新文件（不用刷新）。下拉列出 ComfyUI input 目录的已有文件（音频在 input/audio/）。**接了 `image`/`audio` 插座时插座优先**；两者都空且是覆盖 = 保留磁盘上的文件。

**保存怎么生效**：Save 节点是**输出节点**——改完 widget（名字、描述、上传选择）后需要**执行一次**（点 ComfyUI 的 Queue/执行按钮）才真正写入磁盘。节点底部有一个绿色大按钮「**💾 应用保存 · Save (Queue)**」（排在媒体卡片和预览面板之后），点了直接把当前图排入队列（等价 Queue），执行完弹「资产已保存」提示，同时**画布上所有资产节点的下拉和预览缩略图自动刷新**——保存后 Load 节点上看到旧图的情况不再出现（同一队列里 Load 若已用旧数据跑过，再执行一次即拿到新内容：Load 的缓存键是库文件的内容哈希，文件一变缓存必失效）。

**描述不想手写？—— Save Character/Scene Asset (Auto) 节点**：它就是对应的 Save 节点（同样的名字下拉、媒体卡片、应用保存按钮、覆盖语义），只多接一个 `llm_service_connector`。**留空的 description/voice 由 LLM 在同一次执行里起草并直接保存**；你填了的字段逐字保存，LLM 绝不重写。描述用 H3 生成器同一套 60–100 词英文身份描述 captioner；音色从名字+外貌推断——**模型听不到音频**，wav/mp3 对它毫无区别；status 输出会标注哪些字段是 LLM 起草的（`LLM drafted description and voice`），听过样本后改字段再保存一次即可。
两个可选提示词让 LLM 起草得更准：**图像描述提示词**（image_prompt，给描述调用的作者先验——气质/着装细节、必须写进的事实，任意语言）和**音色提示词**（voice_prompt，音色 LLM 听不到音频，真实音色只能在这里告诉它，如「老年男声，沙哑，语速慢」）。提示词原样附在各自调用的 user 消息里作为 Author notes，系统提示词的风格契约（60–100 词英文、一行式音色描述）不变；留空则行为与不带提示词完全一致。场景版只有 image_prompt（场景没有音色字段）。
注意：起草需要图。description 留空且既没接线也没选上传图时，节点会明确报错（接图或自己填描述二选一）。

**镜头纪律**：生成器上的 `camera_discipline` 选 `locked - 固定机位` 可保证「镜头固定、构图全场不变」——分镜 LLM 默认有一条景别交替规则会主动切特写，概念里的中文相机语言只是软提示；locked 由代码强制执行（详见 `h3_loop_prompt_generator_node.md`）。

**删除资产 —— Delete Character/Scene Asset 节点**：名字下拉只列该类型的现有资产，预览面板显示即将删除的内容；**两步确认**——选好名字后把 `confirm 确认删除` 开关翻成 True 再排队，开关为 False 时排队只会报错提醒、绝不删除。执行后输出被删名字和 status（删了几个文件），画布上所有资产节点的下拉和预览自动刷新；引用该资产的 Load 节点下次执行会明确报「不存在」。删除同样只作用于本机固定库。
此外，**Load 节点顶部和 Delete 节点底部有「🗑 删除 · Delete」按钮**（保存节点是创作入口，不放破坏性按钮；Delete 节点先经预览看清再看到底部的红色删除）：删的是该节点 name 下拉当前选中的资产，两击确认（第一击按钮变红进入待确认态、5 秒自动解除，第二击才真正删除），即时生效不进队列；删除后画布上所有资产节点的下拉和预览自动刷新。

**Prompt Generator V2 的接法**：`character_1..9` / `scene_1..9` 是**插座**——把 Load 节点的 `asset` 输出一根线接过来；接上 `character_1` 才露出 `character_2`。节点上没有任何选择控件；生成器从线上的资产包组装 roster 与 manifest（零视觉调用），槽位顺序（角色在前、场景在后）就是 `<Picture N>`。第 4 个输出 `images` 是按槽位拼好的图片批次，直接接视频链。

```text
Load Character Asset（孙女）──asset──► V2.character_1
Load Character Asset（爷爷）──asset──► V2.character_2
Load Scene Asset（客厅）──────asset──► V2.scene_1

V2.plan_json ──► MiniMaxH3ChainPlanModern.plan_json_input
V2.assets ──► Asset → Slots ├──picture_1──► ref2v.ref_image_0   （孙女 = Picture 1）
                            ├──picture_2──► ref2v.ref_image_1   （爷爷 = Picture 2）
                            ├──picture_3──► ref2v.ref_image_2   （客厅 = Picture 3，环境）
                            ├──audio_1───► ref2v.ref_audio_0   （孙女 = Audio 1）
                            └──audio_2───► ref2v.ref_audio_1   （爷爷 = Audio 2）
```

**顺序保障（Asset → Slots 节点，从生成器接出）**：V2 的第 5 个输出 `assets` 是接进来的全部资产包按槽位序（角色1..N → 场景1..M = `<Picture N>`）的汇总。Asset → Slots 吃这一个输入，拆出**生成提示词所用的同一批图片和音频**：

- `picture_1..9`（IMAGE）：第 i 张恒等于第 i 个资产——接 `ref_image_{i-1}`，图片对齐由构造保证。**各资产分辨率可以不同**（竖幅三视图 + 横幅三视图混用没问题）：每个输出保持该资产的原生尺寸，H3 对每路参考独立缩放（match/max）；只有生成器自己的 `images` 批次输出在尺寸不一时会置空并在 summary 里说明——用 Slots 节点的 picture 输出即可；
- `audio_1..3`（AUDIO）：角色的音色样本按角色序（场景无音频；H3 独立音频上限 3）——接 `ref_audio_{i-1}`，`<Audio N>` 与 plan 的角色顺序一致；
- `count` / `audio_count`（INT）：各有几张、几条。

输出列表是**静态的 14 个**（count + picture_1..9 + audio_count + audio_1..3）：用几根接几根，空槽输出 None，H3 跳过且不占编号。刻意不做输出裁剪——裁剪会平移连线槽位下标、把已存的音频线错挂到图片输出上（真机踩过：ref_audio 收到 IMAGE）。**数量显示在生成器节点上**：接角色/场景时实时刷新「📊 N 图 · M 音（含场景 M）」标签；Slots 节点的 `count` / `audio_count` INT 输出在执行后给出确切数值。注意：`MiniMaxH3TaggedReferenceScene` 按 @tag 链重编号、只激活 prompt 引用的 tag——V2 的 plan 用原生 `<Picture N>` 无 @别名，直连 ref_image_N 才是对口路径。

## 4. 接到 H3

```text
LoadAssetBatch ──images───────► MiniMaxH3LoopPromptGeneratorV2.images
              └─roster_json──► V2.asset_roster
LoadCharacterAsset ──images──► 需要 reference 图的链（H3 会把长边缩到 2048）
V2.plan_json ────────────────► MiniMaxH3ChainPlanModern.plan_json_input
```

roster 走 V2 生成器：槽位跳过视觉打标、场景槽位 role=`environment`、音色逐字贴 `<d>`——这些行为见 `h3_loop_prompt_generator_node.md`，本计划不改 H3。

## 5. 排查表（常用）

| 报错 | 处理 |
|---|---|
| `images of one asset must share one resolution` | 同一资产导出成同宽同高；错误里已点名每张 |
| `audio index "01" has both 01.wav and 01.mp3` | 每个序号只留一个扩展名 |
| `holds N images; an asset supports at most 8` | 拆分或删减 |
| `audio_3 does not exist … has 2 audio clip(s)` | 该角色只有 2 条音频；接 audio_1/2，或往 `audio/` 补文件 |
| `exists both as a character and as a scene` | 文件管理器里给其中一个改名 |
| 旧工作流里的 `library_dir` | 被忽略是**设计**：库固定在插件目录的 `asset_library/`，或设 `MIE_ASSET_LIBRARY` |

## 6. 边界

- 不做侧栏、搜索页、删除/改名界面（所以有「打开文件夹」）。
- 不缩放图片、不拼接音频。
- 上限 8 图 / 8 音频是**节点插座**上限；H3 的 9 图 / 3 音频限制在链路那一侧。
- H3 提示词与生成器不在本计划内（`roster_json` 契约保持兼容）。
