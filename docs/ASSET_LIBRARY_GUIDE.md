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
- 保存节点：下拉只列本类型；选中已有名字自动回填正文并标「与库一致 / **将覆盖**」；图/音频插座可不接（覆盖时保留磁盘文件）；「✏️ 新名字…」新建。

**Prompt Generator V2 的接法**：`character_1..9` / `scene_1..9` 是**插座**——把 Load 节点的 `asset` 输出一根线接过来；接上 `character_1` 才露出 `character_2`。节点上没有任何选择控件；生成器从线上的资产包组装 roster 与 manifest（零视觉调用），槽位顺序（角色在前、场景在后）就是 `<Picture N>`。第 4 个输出 `images` 是按槽位拼好的图片批次，直接接视频链。

```text
Load Character Asset ──asset──► V2.character_1
Load Character Asset ──asset──► V2.character_2
Load Scene Asset     ──asset──► V2.scene_1
V2.plan_json ─────────────────► MiniMaxH3ChainPlanModern.plan_json_input
V2.images ──► Asset Pictures → Slots ──picture_1──► ref2v.ref_image_0
                                  ├──picture_2──► ref2v.ref_image_1
                                  └──picture_3──► ref2v.ref_image_2
```

**顺序保障（Asset Pictures → Slots 节点）**：V2 的 `images` 输出本来就是槽位序（角色1..N → 场景1..M = `<Picture N>`）。这个节点把它按帧拆成 `picture_1..picture_9` 根明确编号的输出线——`picture_i` 恒等于批次第 i 张，接 `ref_image_{i-1}` 即可，**顺序由构造保证**，不再依赖两条链各自手工对齐。`count` 插件控制可见输出数量（接线了的输出不会被裁）；超出批次的槽输出 None，H3 跳过且**不占图片编号**。`count` 输出（INT）告诉你实际有几张。注意：`MiniMaxH3TaggedReferenceScene` 那条路按 @tag 链重编号、只激活 prompt 引用的 tag——V2 的 plan 用原生 `<Picture N>` 无 @别名，直连 ref_image_N 才是对口路径。

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
