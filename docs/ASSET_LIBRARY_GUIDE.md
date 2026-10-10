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

## 3. 七个节点

### Load Character Asset

| 输出 | 类型 | 内容 |
|---|---|---|
| `name` / `description` / `voice` | STRING | 目录名 / 描述 / 音色描述 |
| `images` | IMAGE | `[N,H,W,3]`，N 为 1–8，顺序即文件名顺序 |
| `audio_1` … `audio_8` | AUDIO | 每条音频一个插座。角色只有 K 条时，`audio_{K+1}..8` 是 `ExecutionBlocker`，消息写明只有几条；接了它的下游会失败并显示该消息，没人接则无影响 |

### Load Scene Asset

输出 `name` / `description`（STRING）+ `images`（`[N,H,W,3]`）。场景没有音频。

### Save Character / Save Scene

- 名字下拉只列本类型；首项 `（✏️ 新名字…）` 选后才出现 `new_name`。
- 选中已有名字：**自动回填库里的 description / voice** 并逐字段标「与库一致 / **将覆盖**」，缩略图预览。
- `images` 插座接一个 `[1..8,H,W,3]` 批次 = **整组替换**；不接 = 保留磁盘上的图（改一句描述不用重接任何东西）。音频同理：`audio_1..8` 一条都不接 = 保留；接任意几条 = 用这几条整组替换。插座逐个生长：接上 `audio_1` 才露出 `audio_2`，上限 8。新建必须至少一张图。
- 保存成功后画布上所有资产节点的下拉/缩略图自动作废刷新。

### Load Asset Batch

- 「🖼 选择资产…」弹窗：卡片**3:4 竖幅完整显示**第一张图（`object-fit: contain`，不裁剪），角标写「N 张」「N 条音频」；同名角色/场景的卡片不可选并写明原因。
- 节点上是**可拖动排序的缩略图槽位条**（序号 = `<Picture N>`），✕ 移除。
- **没有名单编辑入口**：`names` 文本永久隐藏，只作工作流里的顺序记录；老工作流加载后照常还原槽位条。
- 输出 `images` 只取每个资产的**第一张**（多图会打乱 `<Picture N>`）；完整图组走 Load 节点。
- `roster_json` 条目：`name/kind/slot/description/image_count`（+角色的 `voice/audio_count/appearance`，`appearance` 与 `description` 同值，兼容已上线的花名册解析）。

### List Character / List Scene Assets

不变：名字一行一个 + status（下拉时代之后基本只剩兼容用途）。

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
