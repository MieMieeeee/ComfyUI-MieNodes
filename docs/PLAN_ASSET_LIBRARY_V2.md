# 资产库计划（第二版）

> 日期：2026-10-09
> 这份计划替换 `docs/PLAN_ASSET_LIBRARY.md` 里和下面五条冲突的部分。旧文件留下，作为已经评审过的上一版，不再当作实现依据。
> 本版只改资产的存储、读取和节点界面。不改 H3 提示词，不改 `MiniMaxH3LoopPromptGenerator`。
>
> **2026-10-10 第二次修订（用户裁定，简化）：**原有 7 个资产节点显示名加 **(V1)** 标志、行为不变。新增 4 个无标志的简化节点（Save/Load × Character/Scene），模型收窄为：**角色 = 一张图（可以是三视图）+ 一句图片描述 + 一段音频 + 一句声音描述；场景 = 一张图 + 一句图片描述**——不满足（多图/多音频）的资产在简化节点上明确报错并指向 (V1) 节点。`MiniMaxH3LoopPromptGeneratorV2` 新增 `character_1..9` / `scene_1..9` 下拉槽（角色槽自动增加；顺序 = 角色1..N 然后 场景1..M，即 `<Picture N>`），执行时自行从固定库读图读文，**不再需要 asset_roster 排线与 images 批次**；新增第 4 个输出 `images`（按槽位拼好的首图批次，直供视频链）。旧 asset_roster + LoadAssetBatch 路线继续可用（与下拉槽互斥，同时填会报错）。
> **2026-10-10 修订（用户裁定）：**第 3 节的库位置改为**插件目录内的 `asset_library/` 子文件夹**——撤销旧计划「必须在插件包根之外」的规则（同步脚本对该目录既不复制也不镜像删除，`.gitignore` 忽略它）。`$MIE_ASSET_LIBRARY` 仍为最高优先；旧的 `~/mie_asset_library` 首次使用时自动整体搬入。第 6 节的 `roster_json` 补充：`image_count` / `audio_count` / 角色 `description` 由生成器端的花名册解析原样保留，供下游直接消费。

## 1. 要做成的五件事

1. 资产目录不由用户选择。节点上与目录相关的唯一动作是打开这个文件夹。
2. 批量加载节点上不提供名单编辑。「✎ 名单」和多行名字框都去掉。
3. 选择资产的弹窗里，肖像完整可见，不被横框裁掉。
4. 读取一个角色，得到：名称、描述、图片（一张或多张）、音频（零条或多条）、音色描述。
5. 读取一个场景，得到：名称、描述、图片（一张或多张）。第 5 条原文写了「角色」，字段里没有音频和音色，这里按场景实施。

## 2. 不做什么

- 不在节点上编辑、粘贴或切换库路径。工作流里已经存下的 `library_dir` 不再生效。
- 不做侧栏、搜索页、删除和改名。删除和改名仍用文件管理器，所以才要「打开文件夹」。
- 不把多张图缩放成同一尺寸。分辨率不同就报错，并写出每一张的文件名和尺寸。
- 不把多条音频接成一条波形。时长和采样率可以不同，每条单独一个 `AUDIO`。
- 不改 H3 的提示词和生成器。批量节点的 `roster_json` 仍给现有接线用，见第 6 节。
- 单份资产最多 8 张图、角色最多 8 条音频。这是节点插座数量的上限，不是 H3 的 9 张图 / 3 条音频。

## 3. 库在哪

每台机器一个目录。解析顺序与现在相同，但调用时不再接受节点上的路径：

1. 环境变量 `MIE_ASSET_LIBRARY`（非空时）
2. 否则 `~/mie_asset_library`

`validate_library_dir` 的其余规则保留：必须是绝对路径，必须在插件包根之外。

界面只有按钮「打开文件夹」。它请求 `POST /mie_nodes/assets/open`。这个接口不读取客户端传来的路径，只打开上面解析出的那一个目录（不存在则先创建）。Windows 用 `os.startfile`。排队执行不调用它。

`library_dir` 这个 widget 可以留在输入列表的原位置并一直隐藏，避免已保存工作流的 widget 下标错位。执行、扫描、缓存键都忽略它的值。节点上不再有「📁 库…」，也不再把路径框翻出来编辑。常驻一行字显示解析后的路径，只供核对，不能改。

## 4. 磁盘

```text
<库>/characters/<名字>/character.json
<库>/characters/<名字>/images/01.png
<库>/characters/<名字>/images/02.png
<库>/characters/<名字>/audio/01.wav
<库>/characters/<名字>/audio/02.mp3
<库>/scenes/<名字>/scene.json
<库>/scenes/<名字>/images/01.png
```

`character.json`：

```json
{"name": "夏栀", "description": "……", "voice": "adult female, clear, low-mid register"}
```

`scene.json`：

```json
{"name": "便利店", "description": "……"}
```

- `description` 和 `voice` 仍是去空白后的非空字符串。描述上限 800 字，音色描述上限 200 字。不检查是不是英文。
- 图片顺序、音频顺序都是文件名的自然排序（`2.png` 在 `10.png` 之前）。保存时写成两位序号：`01.png`、`02.png`。
- 图片只接受 PNG。音频保存写成 16-bit PCM WAV；读取仍接受手工放入的 `.mp3`（`soundfile`，懒加载）。同一序号的 `.wav` 和 `.mp3` 同时存在则报错。
- 角色至少一张图。音频可以是零条。场景至少一张图。
- 同一资产内的图必须同宽同高。这是因为 ComfyUI 的 `IMAGE` 是一个 `[N,H,W,3]` 张量，不能在一个插座里装两种分辨率。

### 已有库

现在的角色是 `portrait.png` + `voice_sample.wav`（或 `.mp3`），JSON 字段是 `appearance`。场景是一张 `scene.png`。

读取时：

- 有 `images/` 就只用 `images/`。没有 `images/` 时，`portrait.png` 或 `scene.png` 当作第 1 张。
- 有 `audio/` 就只用 `audio/`。没有 `audio/` 时，`voice_sample.wav` 或 `.mp3` 当作第 1 条。
- JSON 没有 `description` 时，用原来的 `appearance`。

下一次成功保存会写成新布局，并删掉已被取代的 `portrait.png`、`scene.png`、`voice_sample.*`，避免下次读取数成两份。只读不删。

## 5. 节点输出

菜单仍是 `🐑 MieNodes/🐑 Assets`。类型只有 `STRING`、`IMAGE`、`AUDIO`、`INT`。

### 读取角色 `LoadCharacterAsset`

| 插座 | 类型 | 内容 |
|---|---|---|
| `name` | STRING | 磁盘上的目录名 |
| `description` | STRING | 描述 |
| `images` | IMAGE | `[N,H,W,3]`，N 为 1–8，顺序与文件名一致 |
| `voice` | STRING | 音色描述 |
| `audio_1` … `audio_8` | AUDIO | 第 1 条到第 8 条。没有该条时，这个输出是 `ExecutionBlocker`，文案写明这个角色实际有几条。没人接的空插座不报错 |

`IMAGE` 与现在一样是 float32 RGB、0–1。`AUDIO` 与现在一样是 `{"waveform": [1,C,T], "sample_rate": int}`。

### 读取场景 `LoadSceneAsset`

| 插座 | 类型 | 内容 |
|---|---|---|
| `name` | STRING | 目录名 |
| `description` | STRING | 描述 |
| `images` | IMAGE | `[N,H,W,3]`，N 为 1–8 |

### 保存

保存角色的输入：名字下拉、`new_name`、`description`、`voice`、可选 `images`、可选音频。

- 音频输入用本仓库 `js/stringFormatAutogrow.js` 的同一套做法：先露出 `audio_1`，接上之后再露出下一条，最多 8 条。
- 新建角色必须有至少一帧图像。音频可以不接。
- 覆盖时，`images` 没接表示保留磁盘上的图；接了表示用这批图整组替换（1–8 帧）。音频同理：一条都没接表示保留；接了任意一条表示用接上的那几条整组替换。这样改一句描述不用重接图和音频。
- 保存场景只有描述和可选 `images`，没有音色，没有音频。

覆盖前，选中已有名字要把磁盘上的描述、音色描述和第一张缩略图填回节点。将要写下去的字和库里不一致时，节点上标「将覆盖」。

`IS_CHANGED`：保存和列出仍返回 `float("nan")`。读取的缓存键覆盖 JSON、每一张图、每一条音频。缺文件时仍返回 `MISSING_CACHE_MARKER`，函数不抛错。

## 6. 批量加载

`LoadAssetBatch` 仍是按顺序挑选若干资产。界面是选择弹窗加一条可拖动的槽位条。顺序就是槽位顺序。

去掉「✎ 名单」和可见的多行 `names`。`names` widget 保持隐藏，仍是工作流里的顺序记录，加载老工作流时能还原槽位条。用户不能打开它。

批量节点的 `images` 只取每个资产的第 1 张，按槽位拼成一个批次。不把一个角色的第 2 张、第 3 张展开进这个批次，否则 `<Picture N>` 会在用户没选择的情况下错位。一个角色的全部图和全部音频从「读取角色」取。

这一批第 1 张图的分辨率必须相同，否则按现在的文案报错（点名是哪几张、各是多大），不缩放。

`roster_json` 每个条目仍有 `name`、`kind`、`slot`。角色另有 `description`、`voice`、`image_count`、`audio_count`。场景另有 `description`、`image_count`。角色条目同时写 `appearance`，内容与 `description` 相同，这样已经接上的花名册解析不用在本计划里改。

## 7. 弹窗里的图

缩略图接口保持最长边 256，不改像素比例。显示不全是样式造成的：卡片把图放进约 150×104 的横框，并且 `object-fit: cover`。

弹窗卡片、节点上的小预览、槽位条上的图，都改为完整显示：

- `object-fit: contain`
- `object-position: center`
- 底色保持现在的深色
- 弹窗卡片的图框用竖幅比例（宽 100%，`aspect-ratio: 3 / 4`），不再写死 104 像素高

卡片上用第一张图，角标写「3 张」或「2 条音频」。不要为了弹窗去解码原图。

## 8. 性能

- 打开文件夹不进队列，不进 `IS_CHANGED`。
- 名单接口仍只返回名字和数量，不返回图片字节。
- 弹窗只请求第一张的缩略图。缓存键继续带源文件的大小和修改时间。
- 读取节点在执行时才解码该资产的 PNG 和音频。这是输出本身的成本，不在刷新界面时做。
- 同一目录的扫描继续共用一个在途请求。保存成功后作废该缓存。

## 9. 验收

1. 任一资产节点上看不到可编辑的路径。点「打开文件夹」打开的是 `MIE_ASSET_LIBRARY` 或 `~/mie_asset_library`，不是工作流里旧的 `library_dir`。
2. 批量加载节点上没有名单按钮，也没有多行名字框。用弹窗选两个角色、拖动排序后，重新打开工作流，顺序还在。
3. 一张竖幅定妆照在弹窗里从头到脚都看得见。
4. 一个有 2 张同尺寸图和 2 条音频的角色，读取节点给出名称、描述、`[2,H,W,3]`、音色描述、`audio_1`、`audio_2`。`audio_3` 被接上时失败，并写明只有 2 条。
5. 一个有 2 张图的场景，读取节点给出名称、描述和 `[2,H,W,3]`。
6. 只改描述再保存，磁盘上的图和音频字节不变。
7. 只有 `portrait.png` 和 `voice_sample.wav` 的旧角色仍能读取，图数为 1，音频为 1。
8. 同一角色两张图分辨率不同，保存和读取都失败，错误里有两个文件名和尺寸。
