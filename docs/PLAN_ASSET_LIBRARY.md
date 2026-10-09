# 角色与场景资产库实施计划

> 目标：在 MieNodes 里加一套独立的角色、场景存储，方便反复取出同一张定妆图、同一句音色、同一句场景说明。  
> 范围：磁盘上的库，加上保存、读取、列出这三类节点。  
> 硬边界：资产模块不认识 H3，不改任何 H3 节点、提示词或解析器。

评审人只需读这一份。对话里曾经有一版把资产做成 H3 概念稿拼装器，并给循环节点加音色插座。那一版已放弃，理由见第 2 节。

本文已按两次外部评审修订。第一次补上列出节点的双输出、双文件失败窗口、大小写合并、读取失败、缓存、路径和张量往返。第二次把组批节点改成核心里的 `BatchImagesNode`。第三次是用户决策（2026-10-07）：角色字段加入音色样本音频——只存样本本身，不存 `voice_id` 或音色枚举；原第 2 节「不做音色音频」一条撤销。

第四次是用户三项决策（2026-10-09，实现当天作出，推翻本文前两节的四条边界）：

1. **角色 JSON 必填 `appearance`（手写英文人物描述，1–800 字）**，与 `voice` 同为「作者写的英文正文，原样注入提示词、不翻译」。外貌真相从「定妆图 + 视觉打标」改为「手写正文」。理由：视觉打标是链路里最贵（一次多模态调用 + 磁盘缓存）也最不可控的一环（改一次打标提示词整批缓存失效），且外貌与音色在 schema 层分开后，音色描述里混入衣着这类错误从结构上不可能。旧 JSON 缺 `appearance` 读取时大声报错，不静默回退打标——一份 plan 里混「手写真相」和「机器猜的真相」正是第三次评审要治的身份分裂。
2. **场景升级为「一张场景图 + 一句手写英文描述」**：`scenes/<名字>/scene.json` + `scene.png`，字段 `setting` 改名 `description`（1–800 字），目录结构、两文件提交点与角色完全一致。场景是「角色减去音色」。纯一句话、无图的场景不再是资产——那种情况写在 user_input 散文里。
3. **库通过结构化花名册接进 H3 提示词生成**（原第 2 节「不修改 H3」撤销）：新增 `MiniMaxH3LoopPromptGeneratorV2`（继承 V1，只多一个可选 `asset_roster` JSON 插座）和 `LoadAssetBatch`（角色与场景混装，输出 IMAGE 批次 + `roster_json`）。合同是一条 JSON 字符串，`nodes/assets/` 与 `nodes/llm/` 之间仍无 Python import。花名册槽位跳过视觉打标；场景槽位 manifest role 为 `environment`（不占 `<Subject N>`、不进身份钉、不进音色表）。配套改了三份 H3 提示词文件，其中 `shot_user_template.txt` 在空花名册下渲染与改动前逐字节相同（测试冻结），`prefix_synth_system.txt` / `ref2v_addendum.txt` 的措辞变化由 V1/V2 共享、无花名册时语义等价。

按第四次修订实现后，第一、二节的原文保留在下方作为历史边界，冲突处以本段为准。

第五次是用户决策（2026-10-09 晚）：**撤销第 2 节「不做前端扩展」中关于输入辅助的部分**。字段集与 slot 契约已在真实使用中稳定（当日已提交两版），用户以实际体验提出：手打名字和路径不可用也不可读。落地为「输入辅助」而非「检索界面」——新增 `POST /mie_nodes/assets/list` 端点（只返回名字和类型，不返回文件内容）和 `js/assetLibraryUI.js`：name 由手打 STRING 改为下拉（类级占位 + 按实例取真实列表；`VALIDATE_INPUTS` 豁免类级成员校验，错误交给执行层报）；Save 节点尾追 `new_name`（选 `（✏️ 新名字…）` 时才填写，选已有名即明确覆盖）；`LoadAssetBatch` 增加槽位行（纯 UI 镜像，`serialize:false` 永远排在 widgets 末尾，序列化载体仍是 `names` 多行文本，工作流 JSON 逐字节兼容）。搜索框、管理面板、删除界面仍然不做。

第六次是用户决策（2026-10-09 深夜）：**库目录每台机器固定，不再逐节点手填**。空 `library_dir` 是「用本机默认」标记：解析为 `$MIE_ASSET_LIBRARY`，未设则 `~/mie_asset_library`（用户现有库就在此处，零迁移）。非空值仍然生效——它是老工作流的逃生口。前端把 `library_dir` widget 隐藏（仍序列化、仍参与位置对齐），节点上的「📁 库目录」按钮负责显隐并显示解析结果。空值解析收敛在 `library.resolve_library_dir` + `validate_library_dir` 一处，所有入口（保存/读取/列出/缓存键/端点）都经过它。

---

## 1. 要解决的事

短剧要跨镜头重复使用同一个人、同一个房间。现在这些材料写在各条工作流的概念稿里，图也散落在各处。下一次生成要靠人再找图、再抄音色和房间说明。

这一版只把两样东西做成可复用的库：

| 资产 | 库里有什么 | 取出来给谁用 |
|---|---|---|
| 角色 | 名字、一句手写人物描述（`appearance`）、一句手写音色描述、一张定妆图、一段音色样本音频 | 任何接受 `STRING` / `IMAGE` / `AUDIO` 的节点；经 `LoadAssetBatch` + V2 直喂 H3 链 |
| 场景 | 名字、一张场景图、一句手写场景描述 | 同上 |

> **第四次修订后**：外貌真相是手写的 `appearance` 正文（英文，原样注入），不再由视觉打标从定妆图猜；「不在 JSON 里再写一套毛色和衣服」一条随之撤销——写的就是那套正文。两个 text 字段的英文要求是文档约定，代码只查类型和长度。站位、身份、台词、参考图编号仍然不是资产字段。

## 2. 不做什么

以下内容不在本计划里，实现时不要顺手加进去。

- 不新增 `H3_CHARACTER` 之类只有某个生成节点能接的类型。
- 不生成 `参考图 1 → 名字（画面左）`，不拼接台词，不产出 `speaker_voices` JSON。（第四次修订后仍然成立：花名册是数据契约——名字、槽位、两段正文——绑定句与音色句的**拼装**仍由 H3 管线自己完成。）
- ~~不修改 `MiniMaxH3LoopPromptGenerator`、`minimax_h3_loop_prompts.py`、H3 提示词和用户输入增强器。~~（第四次修订撤销：V1 节点类本身未改；新增 V2 子类，`minimax_h3_loop_prompts.py` 为纯新增（roster 解析与文本块），三份提示词文件的改动见头部第四次修订说明。）
- `nodes/assets/` 不 import `nodes/llm/` 下的任何模块。
- 不做空镜图、多套衣服、道具、表演视频、`voice_id` 或音色枚举字段。音色以样本音频本身入库（第三次修订的决策）；枚举名和克隆出来的 id 属于某一家 TTS 服务，由使用方自己维护，不进 schema。
- 不做删除节点，不做库的检索界面和前端扩展。（第五次修订缩小范围：输入辅助类前端扩展——下拉与刷新按钮——已做，见头部第五次修订；搜索框与管理面板仍不做。）
- 不把库目录放进插件包内。`scripts/sync_to_comfyui.py` 默认是镜像同步：目标端有、源端没有的文件会被删掉。库存若写在插件包里，下一次同步会直接消失，不只是被盖住。

使用方以后可以自己把「读取角色」的图接到 H3 循环节点的 `images`，把音色和场景说明抄进概念稿。那是工作流的接法，不是这个库的职责。（第四次修订后：这是默认路径之外的新增路径——`LoadAssetBatch` 的 `roster_json` 把「谁绑哪张图、长相、音色、场景是什么」作为数据交给 V2，散文抄写不再必要；不接花名册的旧接法继续可用。）

## 3. 磁盘格式

节点上有一个必填字符串 `library_dir`，指向库存根目录。

路径规则：

- 空白或空字符串直接拒绝。空串会落到进程的当前工作目录，然后在那里建出 `characters/`。
- 必须是绝对路径。相对路径按哪种工作目录解析，取决于 ComfyUI 的启动方式，不接受。
- 解析后的路径若位于插件包根以内，保存和读取都拒绝，并说明应改到包外。插件包根由资产模块的 `__file__` 向上找到包含插件 `__init__.py` 的那一层。开发仓库和 `custom_nodes` 里的部署副本都用这一条，不写死某台机器上的仓库路径。镜像同步会删掉的就是这一层目录里的多余文件。

```text
<library_dir>/
  characters/<名字>/character.json    （name + appearance + voice）
  characters/<名字>/portrait.png
  characters/<名字>/voice_sample.wav
  scenes/<名字>/scene.json             （name + description，第四次修订）
  scenes/<名字>/scene.png
```

`characters/<名字>/character.json`：

```json
{
  "name": "虎斑",
  "voice": "成年男性，低音，平稳"
}
```

同目录下的定妆图固定叫 `portrait.png`。JSON 不记录图片路径，避免改名或搬库之后字段和文件对不上。

`scenes/<名字>.json`：

```json
{
  "name": "便利店",
  "setting": "深夜便利店打烊后，货架灯白光，冷柜嗡嗡作响，料理台边摊着一本账本。"
}
```

写入规则：

- UTF-8，`ensure_ascii=false`，末尾一个换行。中文名字在文件里保持原样。
- 先写同目录临时文件，再替换目标文件。
- 名字只作为单一路径分量。实现用「校验后的名字」拼接路径，不把用户输入直接 `joinpath`。

角色要写三个文件，顺序固定：

1. 把新的 PNG、新的音色样本 WAV 和新的 JSON 都写到临时文件。临时文件写失败时，三个目标文件都保持原样。
2. 用临时 PNG 替换 `portrait.png`。这一步失败时，三个目标文件都保持原样。
3. 用临时 WAV 替换 `voice_sample.wav`，并删掉目录里残留的 `voice_sample.mp3`（见 3.2）。这一步失败时，`portrait.png` 已是新图，音色样本和 JSON 保持原样。
4. 用临时 JSON 替换 `character.json`，这是提交点。

第 4 步失败时，磁盘上是新图加新样本加旧音色。此时抛出错误，错误文本写明定妆图和样本已经换成新的、音色文件还是旧的，下次成功保存同一个名字会把三份一起写好。这一次保存不算成功，不返回 `created` 或 `overwritten`。调用方看到异常即可。测试要分别注入第 2、3、4 步的失败，见第 7 节。

场景只有一个 JSON，替换失败时旧文件保持原样。

### 3.1 名字

保存前对名字做 `strip`。通过校验的名字就是目录名或文件名。

| 规则 | 说明 |
|---|---|
| 长度 | 1 到 40 个字符 |
| 允许 | 汉字、字母、数字，以及 `-` `_` |
| 拒绝 | 空白、`/ \ : * ? " < > \|`、控制字符、`.`、`..`、以 `.` 开头 |
| 拒绝 | Windows 设备名：`CON` `PRN` `AUX` `NUL` `COM1`–`COM9` `LPT1`–`LPT9`，大小写不敏感 |

资产库没有保留词表。不要为 H3 的「场景」「镜头」「旁白」再维护一份黑名单。

大小写合并不靠 `exists()`。Linux 上 `exists("tiger")` 找不到已经存在的 `Tiger`。保存前列出父目录，用 `casefold` 比较：

- 没有大小写折叠后相同的条目：按本次名字新建。
- 已有一条：在那条的现有目录或文件上覆盖，不另建一条。磁盘上的目录名保持先创建时的拼写。JSON 里的 `name` 与磁盘名一致，所以仍是先创建时的拼写。返回的 `name` 也是这个磁盘名，`status` 为 `overwritten`。
- 用任一拼写读取，都落到这一条。

Windows 上不能可靠地把 `Tiger` 重命名成 `tiger`，所以本计划不要求覆盖后改成新拼写。

### 3.2 文字和图片

| 字段 | 规则 |
|---|---|
| `voice` | `strip` 后 1 到 200 字符。不检查是否像音色，不拒绝冒号 |
| `setting` | `strip` 后 1 到 500 字符。允许冒号。库不解释这句文字 |
| 定妆图 | 恰好一张。输入批次的帧数不是 1 时拒绝，提示先选出一帧。保存为 PNG |
| 音色样本 | 恰好一段。节点写入 `voice_sample.wav`（16-bit PCM，用标准库 `wave` 编码）。手工放的 `voice_sample.mp3` 原样读取。两个文件同时存在时读取报错（歧义），不猜。节点保存成功后删除残留的 `.mp3` |

图片张量沿用本仓库已有约定：`IMAGE` 为 float32、`[1, H, W, 3]`、RGB、数值 0 到 1。`nodes/media/image.py` 里张量转 PIL 的方向是 clip 到 0–1 再乘 255 变成 uint8。转换函数抄这个方向，不新写颜色约定。

保存节点返回的 `image` 是 8-bit 量化之后再读回的值，和输入张量不是逐位相同。这是期望行为：下游拿到的就是以后「读取角色」会给出的那张图。读取时按同一形状返回。

音频沿用本仓库 `AUDIO` 约定（`QwenTTSNode` 产出、`WavConcat` 消费的形状）：`{"waveform": float32 [1, C, N]，数值 -1 到 1，"sample_rate": int}`。节点保存时编码为 16-bit PCM WAV，往返差不超过 `1/32767`。`.mp3` 解码用 `soundfile`，和 `QwenTTSNode` 同一加载器，懒加载——只有手放 `.mp3` 才会 import。空样本（0 帧）拒绝。

不接受缺图或缺音色样本的角色目录。JSON 里的 `name` 必须和目录名一致；不一致时读取失败并写出两条名字。

## 4. 节点

新包 `nodes/assets/`。菜单分类 `🐑 MieNodes/🐑 Assets`。类名和显示名走现有的 `add_suffix` / `add_emoji`（`core/utils.py`），类名后缀是 `|Mie`，显示名末尾是 `🐑`。

六个节点都使用 ComfyUI 原有的 `STRING`、`IMAGE` 和 `AUDIO`。没有自定义类型。

| 类名 | 显示名 | 作用 |
|---|---|---|
| `SaveCharacterAsset` | Save Character Asset | 把名字、音色、一张图写入库 |
| `LoadCharacterAsset` | Load Character Asset | 按名字读出角色 |
| `ListCharacterAssets` | List Character Assets | 列出角色名 |
| `SaveSceneAsset` | Save Scene Asset | 把名字和场景说明写入库 |
| `LoadSceneAsset` | Load Scene Asset | 按名字读出场景 |
| `ListSceneAssets` | List Scene Assets | 列出场景名 |

### 4.1 保存角色

输入：

| 插座 | 类型 | 说明 |
|---|---|---|
| `library_dir` | `STRING` | 库存根目录，绝对路径 |
| `name` | `STRING` | 角色名 |
| `voice` | `STRING`，多行 | 音色 |
| `image` | `IMAGE` | 一张定妆图 |
| `voice_sample` | `AUDIO` | 一段音色样本，恰好一轨 |

输出：

| 插座 | 类型 | 内容 |
|---|---|---|
| `name` | `STRING` | 磁盘上的名字。大小写合并时是已有目录的拼写 |
| `voice` | `STRING` | 实际写入的音色 |
| `image` | `IMAGE` | 从刚写入的 PNG 读回的图，已经过 8-bit 量化 |
| `status` | `STRING` | 新建为 `created`，同名已存在为 `overwritten` |

保存节点不返回音频输出。要确认落盘的样本，接一个读取节点。

`OUTPUT_NODE = True`。下游没接线时，保存仍然执行。

`IS_CHANGED` 返回 `float("nan")`，与 `nodes/common/general.py` 里强制重跑的写法相同。输入没变时，ComfyUI 否则会命中缓存并且不再落盘；用户删了库文件再排队，缓存不会把文件写回去。保存本身便宜，重复写同一内容也是覆盖。

同名覆盖是既定行为，没有「是否覆盖」开关。`status` 用来区分新建和覆盖。

### 4.2 读取角色

输入：`library_dir`、`name`。

输出：`name`、`voice`、`image`、`voice_sample`（`AUDIO`）。`name` 是磁盘上的拼写。样本按磁盘上的实际格式解码（`.wav` 用标准库，`.mp3` 用 `soundfile`）。

下面六种情况都抛出错误，错误文本包含该角色目录或文件的绝对路径：

| 情况 | 错误里还要有的内容 |
|---|---|
| 名字不存在 | 绝对路径 |
| JSON 损坏 | 绝对路径 |
| 缺 `portrait.png` | 绝对路径 |
| 缺音色样本（两个候选文件名都不存在） | 绝对路径，以及两个候选文件名 |
| `voice_sample.wav` 和 `voice_sample.mp3` 并存 | 绝对路径，说明是歧义、保留一个 |
| JSON 的 `name` 和目录名不一致 | 绝对路径，以及这两条名字 |

`IS_CHANGED` 返回 `character.json`、`portrait.png` 与音色样本三个文件的内容哈希。文件一变，下游缓存失效。文件不存在时返回一个固定的缺失标记，避免缓存住一次失败。读取节点不用 `float("nan")`，文件没变时不必重读。

### 4.3 列出角色

输入：`library_dir`。

输出只有两个插座：

| 插座 | 类型 | 内容 |
|---|---|---|
| `names` | `STRING` | 每个角色名一行，按名字排序。没有可列出的角色时为空字符串 |
| `status` | `STRING` | 全部可读时为 `ok`。有跳过项时逐条写明目录名和原因 |

没有 `characters/` 目录时，`names` 为空字符串，`status` 为 `ok`。

只列出同时具备合法 `character.json`、`portrait.png` 和音色样本的目录。缺图、缺样本、样本文件并存、名字不合法、JSON 损坏、JSON 名字与目录名不一致的目录跳过，不让整次列出失败。

`IS_CHANGED` 返回 `float("nan")`。库可能在 ComfyUI 之外被改过，列出结果不能吃缓存。

### 4.4 保存场景、读取场景、列出场景

与角色对应，只是没有图片。

保存场景的输入是 `library_dir`、`name`、`setting`。输出是 `name`、`setting`、`status`。`status` 同样是 `created` 或 `overwritten`。`OUTPUT_NODE = True`，`IS_CHANGED` 为 `float("nan")`。

读取场景的输出是 `name`、`setting`。不存在或 JSON 损坏时，错误含绝对路径。JSON 的 `name` 和文件名不一致时，错误含绝对路径和这两条名字。`IS_CHANGED` 哈希该 JSON 文件；文件不存在时返回固定缺失标记。

列出场景的输出是 `names`、`status`。损坏或不合法的文件跳过，并写进 `status`。`IS_CHANGED` 为 `float("nan")`。

## 5. 代码落点

| 路径 | 职责 |
|---|---|
| `nodes/assets/library.py` | 校验、原子写入、读取、列出、内容哈希。不 import ComfyUI、torch、`nodes.llm`。图片和音频都以字节进出 |
| `nodes/assets/nodes.py` | 六个节点。负责张量和 PNG/WAV 的转换，然后调用 `library.py` |
| `nodes/assets/__init__.py` | 导出六个节点类 |
| `tests/test_asset_library.py` | 只测 `library.py` 和纯转换函数 |
| `__init__.py` | 注册到 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` |
| `README.md`、`README_CN.md` | 各加一小节：库放在插件包外，六个节点各自做什么 |

`library.py` 对图片只接受 PNG 字节或本地 `portrait.png` 路径，对音频只接受 WAV 字节或库里的样本文件路径。torch 张量留在 `nodes.py`，测试不必启动 ComfyUI。

张量与 PNG 的转换做成纯函数：float RGB 0–1 的数组进，uint8 PNG 字节出；再读回 float。数值差在 `1/255` 以内。保存节点返回的图就是这个读回值。

`AUDIO` 与 WAV 的转换也做成纯函数：`{"waveform": [1, C, N] float, "sample_rate"}` 进，16-bit PCM WAV 字节出（标准库 `wave`，无编码器依赖）；读回 float，数值差在 `1/32767` 以内。`.mp3` 只在读取路径出现，用 `soundfile` 懒加载解码。

根 `__init__.py` 的注册形式与现有节点一致：

```python
from _mienodes_internal.nodes.assets import (
    SaveCharacterAsset,
    LoadCharacterAsset,
    ListCharacterAssets,
    SaveSceneAsset,
    LoadSceneAsset,
    ListSceneAssets,
)
```

映射键用 `add_suffix("SaveCharacterAsset")`，显示名用 `add_emoji("Save Character Asset")`。其余五个类同理。`_mienodes_internal` 是根 `__init__.py` 放进 `sys.modules` 的合成包，新包沿用这条导入路径。

`tests/test_h3_loop_docs.py` 只守护 H3 文档，不会因为 README 缺资产库小节而变红。两边 README 仍要一起改，和仓库现有的中英对照习惯一致。

## 6. 使用时的接线

库不拼概念稿。一条工作流若要拿这些材料去生成，自己接线：

```text
Load Character Asset ── image ──► 需要图片的节点
                     ├─ name
                     └─ voice ──► 文本节点或概念稿

Load Scene Asset ── setting ──► 文本节点或概念稿
```

多个角色的定妆图要合成一个 `IMAGE` 批次时，用 ComfyUI 核心的 Batch Images（`BatchImagesNode`，`comfy_extras/nodes_post_processing.py`，输入可随图片数量增加），按取用顺序接。旧工作流里的 `ImageBatch` 已标成 Batch Images (DEPRECATED)，核心会把它迁移到 `BatchImagesNode`。本计划不新增批次节点。

仓库里的 `LoadImageBatch` / `SaveImageBatch`（`nodes/common/general.py`）读写的是 tensor_store 的 `.pt` 文件，不是把多个 `IMAGE` 插座拼成一批。不要用它们来组定妆图。

## 7. 测试

`tests/test_asset_library.py` 使用 pytest 的 `tmp_path`。不调用模型，不依赖 ComfyUI 进程。

必须覆盖：

1. 保存角色后再读取，名字、音色、PNG 字节和音色样本字节一致。
2. 同名再保存，`status` 为 `overwritten`，新音色、新图和新样本替换旧内容。
3. 保存场景、读取场景、同名覆盖，行为与角色对应。
4. 列出时按名字排序；空库的 `names` 为空字符串，`status` 为 `ok`。
5. 缺图、缺样本、样本并存、损坏 JSON、JSON 名字与目录名不一致的角色目录被列出跳过，`status` 含该目录名。
6. 非法名字（空、超长、含 `..`、含斜杠、设备名 `CON`）在保存时拒绝，且目录没有被创建。空字节串样本同样拒绝。
7. `library_dir` 解析后位于插件包根内时拒绝写入。测试把一个临时目录当作包根传进校验函数，不依赖真实仓库路径。空字符串、纯空白、相对路径也都拒绝。
8. 音色或场景说明为空、只含空白、超过长度上限时拒绝。
9. 角色三文件替换的三个失败点：
   - 替换 PNG 失败：三个目标文件的字节都不变。
   - 替换音色样本失败：PNG 是新图，样本和 JSON 是旧的，调用抛错。
   - 替换 JSON 失败：PNG 和样本是新的，JSON 仍是旧音色，调用抛错，不返回成功状态。
10. 内容哈希随任一文件内容变化（含音色样本）。同一内容两次哈希相同。文件不存在时返回固定缺失标记。
11. 先保存 `Tiger`，再保存 `tiger`：父目录里只有一个角色目录，音色和图是第二次的内容。用 `tiger` 和 `Tiger` 读取都得到这一条。磁盘目录名保持第一次的 `Tiger`。这条测试在区分大小写的文件系统上也能发现「只调用了 `exists()`」的实现。
12. 读取失败，错误文本都含绝对路径：
    - 名字不存在。
    - JSON 损坏。
    - 缺 `portrait.png`。
    - 缺音色样本，错误里有两个候选文件名。
    - 样本 `.wav` 与 `.mp3` 并存，错误说明歧义。
    - JSON 的 `name` 与目录名不一致，且错误里同时有这两条名字。
13. 张量转 PNG 再读回：每个通道与原值的差不超过 `1/255`。返回值不必与输入逐位相同。
14. 音频：`AUDIO` 转 WAV 再读回，每个采样差不超过 `1/32767`；非 `[1, C, N]` 形状、空样本、非 16-bit WAV、未知扩展名都拒绝。手放 `voice_sample.mp3` 可读（`soundfile` 解码），两个样本文件并存时读取报错、列出跳过，节点保存后残留的 `.mp3` 被删除。

节点类本身不做启动 ComfyUI 的测试。读取用的 `IS_CHANGED` 抽成 `library.py` 的内容哈希，由第 10 条覆盖。保存和列出的 `float("nan")` 是节点上的一行返回值，不另做运行时测试。

## 8. 实施顺序

1. 写 `library.py` 和 `tests/test_asset_library.py`，先让第 7 节的测试通过。
2. 写六个节点和 PNG 转换，注册到根 `__init__.py`。
3. 在 `README.md` 和 `README_CN.md` 各增加资产库小节：目录放在插件包外、六个节点的输入输出、同名会覆盖、返回的图是 8-bit 量化后的值。
4. 跑 `tests/test_asset_library.py`。本改动不碰 H3，不把整份 H3 测试套件当验收条件。

## 9. 已定决策

评审时若要改，请直接针对下面这些决定。

| 决定 | 选择 |
|---|---|
| 和 H3 的关系 | ~~完全分开。本计划不改 H3 代码~~ → 第四次修订：经 `roster_json`（纯 JSON 字符串合同，无 Python import）接进 V2 生成器；`plan_json` schema、`validate_plan`、标签策略不动 |
| 输出类型 | 只用 `STRING`、`IMAGE` 和 `AUDIO`（花名册是 STRING） |
| 角色字段 | 名字、`appearance`（必填，1–800，手写英文）、音色（1–200）、一张 PNG、一段音色样本（第四次修订） |
| 音色样本形态 | 只存音频本身（节点写 `.wav`，手放 `.mp3` 可读，并存报错）。不存 `voice_id` 或音色枚举 |
| 场景字段 | 名字、`description`（必填，1–800，手写英文）、一张 `scene.png`。没有音频（第四次修订） |
| 两个 text 字段的语言 | 英文、原样注入、不翻译——文档约定，代码只查类型和长度 |
| 同名跨类 | 一个名字同时存在于 `characters/` 和 `scenes/` 下报错 |
| 场景槽位在 H3 里的角色 | manifest role=`environment`：不占 `<Subject N>`、不进身份钉、不进音色表 |
| 库存位置 | ~~绝对路径，且必须在插件包根外。空串和相对路径拒绝~~ → 第六次修订：空串 = 本机默认（$MIE_ASSET_LIBRARY，再 ~/mie_asset_library）；相对路径仍拒绝；插件包根外仍强制 |
| 同名 | 覆盖。仅大小写不同视为同名，保留先创建的磁盘拼写 |
| 三文件失败 | PNG 失败三份都不变；样本失败留新图；JSON 失败留新图新样本加旧音色，抛错 |
| 删除 | 这一版没有 |
| 前端 | 输入辅助（下拉/刷新/槽位行）已做（第五次修订）；检索界面/管理面板不做 |
| name widget | combo（类级占位 + 前端按实例填充 + VALIDATE_INPUTS 豁免） |
| 坏目录 | 列出时跳过并写明原因；读取时失败 |
| 保存和列出的缓存 | `IS_CHANGED` 为 `float("nan")`，每次排队都执行 |
| 读取的缓存 | 内容哈希（JSON + PNG + 样本）；缺失用固定标记 |
| 名字长度 | 1 到 40 |
| 音色长度 | 1 到 200 |
| 场景说明长度 | ~~1 到 500~~ → 1 到 800（与 `appearance` 对齐，第四次修订） |
| 文档 | `README.md` 与 `README_CN.md` 一起改 |

## 10. 验收

完成后应同时满足：

- 菜单 `🐑 MieNodes/🐑 Assets` 下能看到六个节点。
- 保存「虎斑」和「便利店」后，磁盘上只有第 3 节描述的文件。
- 重新打开工作流再读取，得到同一句音色、同一张图、同一段音色样本、同一句场景说明。返回的图与当时保存的输入相差不超过 `1/255`，音频不超过 `1/32767`。
- `nodes/assets/` 对 `nodes/llm/` 没有 import。
- 本计划产生的新 commit 不出现 `minimax_h3_loop_prompt_generator.py` 和 `minimax_h3_loop_prompts.py`。工作区里这两份文件可能已有别的未提交改动，不用整份 `git diff` 当这条的判断依据。
