# 分镜资产 + H3 视频 · 外部测试说明（v1.5.2）

核心流程：**装插件 → 准备角色资产 → 导入分镜资产 → 跑出成片**。
本管线是"分镜与渲染解耦"设计：**后半段（分镜资产 → 编译提示词 → 视频生成）完全零 LLM，不需要任何 API Key**。

---

## 1. 环境准备

1. **安装本插件**
   - ComfyUI Manager：搜索 `ComfyUI_MieNodes`（发布者 `mie`），装 **v1.5.2 或更新**。
   - 手动：克隆 `https://github.com/MieMieeeee/ComfyUI-MieNodes` 到 `custom_nodes/comfyui_mienodes`。
2. **安装 H3 依赖与模型**
   - 需要 `ComfyUI-MiniMax-H3-Turbo`、`ComfyUI-MiniMaxH3-Context-Loop` 等依赖插件，以及 H3 模型文件。
   - 画布上出现红框/缺节点，按报错在 Manager 里补装即可。
3. **重启 ComfyUI**（装完或改过插件代码必须重启）。

### 参考工作流（建议从这里开始）

仓库自带一份可直接拖进画布的完整工作流：

```
docs/examples/storyboard-generation/storyboard-h3-ref2va.json
```

（相对本文件的路径是 `../../examples/storyboard-generation/storyboard-h3-ref2va.json`）

它包含整条链路（载入分镜资产 → V3 编译 → 资产分槽 → H3 ref2va → 链路循环 → 合成成片），
画布上还有一个 Note 节点写着每一步。拖进 ComfyUI 后只需要改两处（见第 2 节）。

## 2. 准备角色资产（测试者自建，仓库里不提供）

仓库只提供**分镜资产 JSON**，**角色资产（参考图 + 声音）不在仓库里**，需要自己建：

1. 画布上放 **「保存角色资产 · Save Character Asset」**（`SaveCharacterAssetV2|Mie`）：
   - 填 `name`（资产名）、`description`（外观描述，英文，越具体越稳）、
     `voice`（声线要求，英文；只写音色/音区/语速，别写表演行为）；
   - 接一张**角色参考图**（IMAGE）；有参考音频就再接一条（AUDIO），没有也能跑（声音由 H3 生成）。
2. 运行它，资产就进库了。
3. 之后在 **「加载资产 · Load Asset」**（`LoadAssetV2|Mie`）里按名字选它。

> 参考工作流里那两个 **「角色1 / 角色2 · 填你自己的角色资产名」** 节点就是第 3 步：
> 把它们的 `name` 改成你建好的角色资产名。分镜里**有几位画内角色就接几张角色图**，
> 少接一张 H3 会自己编那个角色的长相；只有一个角色的资产时，把角色2 那根连线删掉即可
> （H3 会忽略空的参考槽位）。

## 3. 拿到分镜资产（三条路线，建议按顺序试）

### 路线 1：直接用仓库里的现成资产（零 LLM · 先跑通管线）

仓库 `docs/storyboard-assets/` 下有已实跑验证过的资产：

| 资产名 | 需要的角色资产 | 场景 |
|---|---|---|
| `班上第三名` | 2 个（女/男各一） | 可选 |
| `同窗请帖` | 2 个 | 可选 |
| `下班前上线` | 2 个 | 可选（不接时环境由分镜 setting 决定） |
| `白披风骑士猫的独白` | 1 个 | 可选 |

导入二选一：
- 命令行（插件目录下）：`python scripts/import_storyboard_assets.py docs/storyboard-assets --overwrite`
- 手动：打开 JSON 全文，粘进 **「保存分镜资产 · Save Storyboard Asset」** 的 `json_text`，
  节点的 `name` 必须与 JSON 里的 `"name"` 完全一致。

然后把参考工作流里 **「① 分镜资产」** 节点的 `name` 改成对应资产名，跑。
（参考工作流默认用 `班上第三名`，需要 2 个角色资产；用 `白披风骑士猫的独白` 只需 1 个。）

### 路线 2：用外部网页大模型生成分镜（免配 API Key · 写自己的故事）

1. 取系统指令全文：仓库 **`docs/storyboard_asset_system_prompt.txt`**
   （节点内置版在 `nodes/llm/prompts/h3_storyboard/system_asset.txt`，两者是同一契约的两种写法）。
2. 把它发给任意大模型（网页聊天即可），然后按下面的格式逐行发你的剧本（**一行一个事件**）：

```text
角色1入场：推门走进咖啡厅，四处张望
角色1：请问有人吗？
角色1动作：走到吧台前敲了敲桌面
角色2：来啦！刚才在后面磨豆子。
角色1离场：拿着咖啡转身走出门外
```

3. 模型会输出**一个完整的 JSON 对象**。复制它，粘进 **「保存分镜资产」** 的 `json_text`，
   并把节点 `name` 设成与 JSON 内部 `"name"` 一致。

### 路线 3：用插件内置节点生成（需要配 LLM API Key）

画布上放 **「MiniMax H3 分镜资产生成器」**（`MiniMaxH3StoryboardGenerator|Mie`）：

| 输入 | 说明 |
|---|---|
| `concept` | 逐行事件/台词（格式同路线 2） |
| `character_settings` | 每角色一行：`名字 \| 画面内/画外/稍后入场 \| 左/右/中/无 \| 声线要求` |
| `shot_count` | **能装进一段就填 1** |
| `target_duration_seconds` | 中文台词按 **5–6 字/秒** 估，每镜再留 1 秒余量 |
| `camera_framing/movement/perspective` | 机位三项 |
| `continuation_overlap_frames` | 多段时保持 22（与下游 Loop 裁切一致） |
| `audio_mode` | `generated`（默认，H3 生成声音，**角色参考音会被丢弃**）/ `reference`（接上角色资产声线样音，≤3 个角色） |

输出四个口：`storyboard`（接 V3）、`storyboard_json`（接「保存分镜资产」的 json_text）、
`storyboard_text`（人读的预览）、`shot_count`。

## 4. 接线铁律（跑视频，全程零 LLM）

管线结构（节点中文名 ↔ 类名）：

```text
载入分镜 · Load Storyboard Asset            (LoadStoryboardAsset|Mie)
  → H3 循环提示词生成器 V3                 (MiniMaxH3LoopPromptGeneratorV3|Mie)  ← 零 LLM，编译成 plan_json
  → 资产转槽位 · Asset → Slots              (AssetToSlots|Mie)                    ← 拆成 picture_1.. / audio_1..
  → H3 参考生视频                           (MiniMaxH3ReferenceToVideo)
  → MiniMaxH3ChainPlanModern → LoopStart → … → Assemble（合成最终成片）
```

- **画内角色必须接图**：`character_1`、`character_2`… 一对一连接角色资产；
  **剧本里有几位画内角色就接几张 `ref_image`**，少接一张 H3 就会现编那个角色的长相。
- **画外旁白**：只接音频不接图（接了图也不会入画）。
- **场景资产 `scene_1` 可留空**：留空时环境完全由分镜的 `setting` 文本决定；
  接上则以场景图为环境（此时 setting 只写光锚，别写房型家具，否则会盖过场景图）。
- **故事必需道具要写进资产**（不是管线能解决的）：例如"新版本上线"需要一台电脑，
  就得在分镜的 `setting` / `visible_state` 里写明"桌上一台亮着的电脑显示器"——H3 不会从"一张书桌"推断职业场景。
- **分辨率**：`width`/`height` 建议横屏 960×544、竖屏 544×960（32 的倍数即可）。
- **`run_name` 只能 ASCII**：中文资产名要另起一个英文代号。

## 5. 验收方法（很重要）

1. **台词用 ASR 核对，不要看烧录字幕**：H3 的烧录字幕有无不稳定（同一资产换个 seed 可能就变了）。
   `python -m pip install faster-whisper` 后对成片跑一遍即可。
   *注：同音字会被 ASR 误识（"驾校"→"假校"、"随礼"→"随理"），这类**不是吞字**，结合语境判断。*
2. **看真正喂给 H3 的提示词**：`output/h3_chains/<run_name>/segments/*.prompt.txt`
   （`prompt_history/*.json` 里存的是分镜正文，不是最终提示词）。
3. **帧检三处**：角色身份（是否参考图那个人）、道具（数量/归属）、姿态（有没有被参考图手势冻住）。

## 6. 已知模型边界（实测摸清的脾气）

1. **续接段的前 0.9 秒是盲区**：多段生成时管线固定裁掉每段开头的 22 帧（≈0.9 秒），
   这段里的事件不会进成片 → "必须被观众看到的关键动作"不能写在续接段第 1 秒内。
2. **塞满的片段会丢最短的一句**：H3 念不完时丢的是最短那句，不是念慢 → 按 5–6 字/秒预算并留余量。
3. **参考图姿势会被冻结**：若 `visible_state` 里写了参考图的姿势（敬礼/抱臂），H3 会整段保持那个手势。
   写自然姿态；**道具是例外**（道具保持参考图姿态，但别要求角色改变它的姿态——改了会被重绘崩坏）。
4. **voice 文字与参考音会叠加生效**：reference 模式下提示词里的音区词仍参与引导。
   样音 126 Hz 而提示词写了 `mid-low`，出来会被压到 96 Hz 左右 → 音区词照角色资产声线规格原样抄。
5. **场景图里已有的字样改不掉**：文字语言锚点只管模型自己编的字
   （咖啡厅黑板上本来就印着英文 `MENU`，提示词改不掉它）。
6. **同音字**：见第 5 节的注。

## 7. 反馈

带这四样回来最好复现：分镜资产 JSON、`segments/*.prompt.txt`、成片 mp4、ASR 文本。
已验证的成片与 QA 记录在 `docs/qa/`（`sketches-20261027/`、`knight-cat-20261026/`、`storyboard-gags-20261026/`）。
