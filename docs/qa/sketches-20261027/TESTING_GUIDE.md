# 分镜资产 + H3 视频 · 外部测试说明（v1.5.2）

给来试这个功能的人：**装插件 → 连大模型 → 生成一条分镜资产 → 跑一条视频**。
全部零 LLM 的后半段（资产 → 提示词 → 成片）不需要任何 API Key。

## 1. 安装

- ComfyUI Manager 里搜 **ComfyUI_MieNodes**（发布者 `mie`），装 **1.5.2 或更新**。
  - 或手动：克隆 `https://github.com/MieMieeeee/ComfyUI-MieNodes` 到 `custom_nodes/comfyui_mienodes`，重启 ComfyUI。
- 需要 H3 模型与它依赖的插件（`ComfyUI-MiniMax-H3-Turbo` / `ComfyUI-MiniMaxH3-Context-Loop` 等），
  缺哪个节点会直接报缺节点，按报错提示补装即可。
- 重启 ComfyUI（改了插件代码必须重启）。

## 2. 两条路：生成分镜资产 / 直接跑现成资产

**路 A：用系统指令让任意大模型生成分镜资产（推荐先试这条）**
1. 取系统指令全文：[`docs/storyboard_asset_system_prompt.txt`](../../docs/storyboard_asset_system_prompt.txt)
   （仓库里；节点版在 `nodes/llm/prompts/h3_storyboard/system_asset.txt`，两者是同一契约的两种写法）。
2. 粘进任意大模型（网页聊天也行），把你的故事按
   `角色1：台词 / 角色1动作：动作 / 角色1入场：动作 / 角色1离场：动作` 逐行给它。
3. 模型会输出**一个 JSON 对象**。复制它，粘进「保存分镜资产」节点的 `json_text`，
   资产名（`name`）要和 JSON 里的 `name` 一致。

**路 B：用插件里的分镜生成节点（需要配大模型 Key）**
- 节点：「MiniMax H3 分镜资产生成器」（`MiniMaxH3StoryboardGenerator`）
  - 填 `concept`（逐行事件）、`character_settings`（每角色一行：名字 | 画面内/画外/稍后入场 | 左/右/中/无 | 声线要求）、
    `shot_count`、`target_duration_seconds`、机位三项。
  - 输出 `storyboard` / `storyboard_json` / `storyboard_text`（预览）/ `shot_count`。
- **关键参数建议**：
  - `shot_count`：**能装进一段就填 1**。每多一段就多一次续生成，续生成片段有已知的角色消失风险。
  - `target_duration_seconds`：中文台词按 **5–6 字/秒** 估（不是 8），每镜再留 1 秒余量。
    片段塞满时 H3 会**丢掉最短的那一句**，不是念慢。
  - `continuation_overlap_frames`：多段时保持 22（与 Loop 续接裁切一致）。
  - `audio_mode`：
    - `generated`（默认）= 声音由 H3 生成，**角色资产里的参考音会被丢弃**；
    - `reference` = 接上角色资产的声线样音（≤3 个角色，每个都要有 voice + 已接音频）。
      想用角色自己的声音就选它。

**路 C：直接用现成资产试跑（零 LLM，最快）**
仓库 `docs/storyboard-assets/` 里有已验证的资产，可直接导入：
```text
python scripts/import_storyboard_assets.py docs/storyboard-assets --overwrite
```
或手动把 JSON 粘进「保存分镜资产」。

## 3. 跑视频（零 LLM）

管线（节点名以画布上的中文名为准）：
```
加载分镜资产(LoadStoryboardAsset)
  → MiniMaxH3LoopPromptGeneratorV3        ← 零 LLM，编译成 plan_json
  → AssetToSlots(Asset → Slots)            ← 拆成 picture_1.. / audio_1..
  → MiniMaxH3ReferenceToVideo              ← H3 ref2va
  → MiniMaxH3ChainPlanModern → LoopStart → … → Assemble
```
接线要点：
- 画内角色**必须**各接一个图像角色资产（`character_1`、`character_2`…）；
  场景资产 `scene_1` **可留空**（留空时环境由分镜的 setting 文案决定）；
  画外旁白可以只接**音频**不接图（接了也不会入画）。
- **参考图槽位要接全**：角色有几位就接几张 `ref_image`，少接一张 H3 就会现编那个角色的长相。
- 分镜里出现的**故事必需道具**（例如“上线”需要一台电脑）要写进 setting/visible_state，
  H3 不会从“一张书桌”推断职业场景。
- 尺寸：横屏 960×544，竖屏 544×960（链路的 `width`/`height`）。
- `run_name` 只能是 ASCII（中文资产名要另起 slug）。

## 4. 怎么验收（很重要）

- **台词一律用 ASR 核对，不要看烧录字幕**——H3 的字幕有无不稳定，同一资产不同 seed 表现不同。
  例：`python -c "import faster_whisper; ..."` 或用任意 ASR 工具跑一遍成片。
- **看最终提示词**看 `output/h3_chains/<run_name>/segments/*.prompt.txt`；
  `prompt_history/*.json` 里存的是分镜正文，不含最终提示词。
- 帧检看三处：角色身份（是不是参考图那个人）、道具（数量/归属对不对）、姿态（有没有冻住的参考图手势）。

## 5. 已知边界（都是实测出来的，交给规则了，但值得先知道）

1. **续接段第一秒是盲区**：链路固定裁掉每个续生成片段的前 22 帧（≈0.9 秒），
   这段里的事件不会出现在成片里 → 多段资产的“必须被看见的事件”要放在首秒之后。
2. **塞满的片段会丢最短的一句**：H3 念不完时丢的是最短那句，不是念慢 → 按 5–6 字/秒预算。
3. **参考图的姿势会被继承**：visible_state 里写“敬礼/抱臂”这类参考图姿势，H3 会整段冻住那个手势。
   写自然姿态；**道具是例外**（道具保持参考图姿态，但别要求角色改变它的姿态——改了会被重绘）。
4. **voice 描述与样音会同时生效**：参考模式下提示词里的音区词仍参与引导，
   写“mid-low”而样音是 126 Hz，出来会被压到 96 Hz 左右 → 照角色资产声线规格的音区词抄。
5. **同音字陷阱**：“驾校/假校”这类会被 TTS 念成同音字，验收时容易误判。
6. **场景图里已有的字样改不掉**：文字语言锚点只管模型自己编的字（咖啡厅黑板的英文 MENU 不会被改成中文）。

## 6. 反馈

带这四样回来最好复现：资产 JSON、运行的 `segments/*.prompt.txt`、成片、ASR 文本。
已知的成片与 QA 记录见 `docs/qa/`（`sketches-20261027/`、`knight-cat-20261026/`、`storyboard-gags-20261026/`）。
