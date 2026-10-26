# 三个新分镜资产的实跑验收（2026-10-26）

三个梗（`奔驰车主的月充一万` / `七岁养伤的一年` / `我叫张同`）从 v2 分镜契约走到 H3 成片的
**第一次完整实跑**，选角用角色资产里的 `Claude娘-日式赛璐璐` 与 `DeepSeek娘-日式赛璐璐`。

## 管线（全部零 LLM）

```
LoadStoryboardAsset(分镜资产)
  → MiniMaxH3LoopPromptGeneratorV3   ← 零 LLM，编译成 plan_json
  → AssetToSlots                      ← 拆成 picture_1.. / audio_1..
  → MiniMaxH3ReferenceToVideo         ← H3 ref2va，图片=角色+场景，音频=角色声线
  → MiniMaxH3ChainPlanModern → LoopStart → … → Assemble
```

基准图是用户自己的 API 格式画布导出 `C:\Users\administered\Downloads\c1.json`
（与 2026-10-20 POV 实跑同一条链）。驱动脚本 `_live_three_storyboards.py`。

### 接线

| 分镜 | character_1 | character_2 | scene_1 | 图片/音频位 |
|---|---|---|---|---|
| 奔驰车主的月充一万 | Claude娘-日式赛璐璐 | DeepSeek娘-日式赛璐璐 | 不接 | ref_image_0/1、ref_audio_0/1 |
| 七岁养伤的一年 | Claude娘-日式赛璐璐 | — | 室内客厅沙发 | ref_image_0/1、ref_audio_0 |
| 我叫张同 | Claude娘-日式赛璐璐 | DeepSeek娘-日式赛璐璐 | 不接 | ref_image_0/1、ref_audio_0/1 |

场景留空的两条由 setting 文案承担环境；场景是可选资产，V3 编译器会把接线场景写成
`Scene reference: <Picture N> provides only the environment, materials and lighting.`
（这一行在链的 `plan.json` / 片段 `*.prompt.txt` 里；`prompt_history/*.json` 存的是分镜正文，
不含前缀——查证时别被它误导）。

## 结果：三条全部通过（最终成片在 `final/`）

| 分镜 | 时长 | 帧 | 台词核对（faster-whisper medium, CPU） | 结论 |
|---|---|---|---|---|
| 我叫张同 | 7.29s | 175 | 三句逐字命中，顺序正确 | 通过 |
| 七岁养伤的一年 | 10.83s | 260 | 七句逐字命中，顺序正确 | 通过 |
| 奔驰车主的月充一万 | 15.79s | 379 | 九句逐字命中，顺序正确 | 通过 |

证据：`final/*.mp4`（最终成片）、`final/asr-*.json`（逐句带时间轴）、
`final/prompt-*.json`（计划正文）、`run1/`（第一轮证据，用于对照问题）。

### 三轮

- **run 1**（seed 2026xxxx01/02/03）：首跑，发现四个问题（见下）。
- **run 2**（+1000）：修掉 voice 句号、道具数量、手势细节、setting 与场景打架；重跑全部三条。
- **run 3**（+7）：按用户看片反馈再修「七岁养伤」（机位改 observer、手势改成逐根收指）与
  「奔驰」（车锚成窗户正下方俯视），各重渲一条。

用户看片反馈与对应修订：

| 反馈 | 修订（改契约，不是手修渲染） | 验证 |
|---|---|---|
| 人物手势跟数字没关系 | direction 改成「手心朝向镜头，每说一个岁数就收起一根手指」 | `final/frame-七岁养伤-手指逐根收起序列.png`：五指 → 逐根收起 → 一指 |
| 画面下方有第一人称的手 | `camera.perspective` 由 `first_person` 改 `observer`，并把「画面里只有讲述者一个人」写进 constraints 与 execution.setting | 复渲后画面下方干净 |
| 脚本说楼下奔驰、场景在窗户外 | setting / execution.setting / visible_state 改成「高层公寓，从上往下看，车停在窗户正下方的路边，俯视能看到车顶」 | `final/frame-奔驰-车在窗下俯视.png`：车就在窗下 |

残留（已如实记录，未再改）：「奔驰」终版画面里茶几上仍多出一部手机（角色手里那部之外），
「全场唯一的那部手机」没能完全约束住 H3 的道具复制；「七岁养伤」的手指收拢到后期略显粘连，
且最后只剩一根手指，与她列出的五个岁数不严格对应。


## 这轮跑出来的四个真实问题（已回流成契约修订，不逐条手修渲染）

1. **setting 文案会压过场景参考图。** `七岁养伤的一年` 第一版 setting 写「现代公寓客厅…白色墙面把光反射回来」，
   而接线场景 `室内客厅沙发` 是暖色乡村客厅。结果同一份提示词两次渲染给出两个房间（一次场景赢、一次文案赢）。
   修订：setting 只留光锚（「正午的客厅：窗外是正午的强烈日光…」），不写房型/墙面/家具。
2. **第一人称机位会把"观众的手"画进画面。** `七岁养伤的一年` 原来用 `first_person`，
   成片画面下方多出一双 POV 观众的手（还端着一个杯子）。修订：改 `observer`，并把"画面里只有讲述者一个人"
   写进 constraints 与 execution.setting。
3. **手势不写细节就渲染不出语义。** 「掰着手指头数数」只渲染成一只抬起的手，跟数字无关。
   修订：「手心朝向镜头，每说一个岁数就收起一根手指」，观众能看清她在数。
4. **垂直关系必须写明视角。** 台词说「楼下停的那辆奔驰」，第一版成片把车平铺在窗外街对面。
   修订：setting 与 visible_state 写「高层公寓，从上往下看，车停在窗户正下方的路边，俯视能看到车顶」。

另外两处小修：角色 `voice` 字符串不许以句号结尾（会被拼成 `…pace.: <d>`）；道具要写成
「全场唯一的那部手机」，否则 H3 会在茶几上再加一部。

## 建议回写 `docs/storyboard_asset_system_prompt.txt` 的规则（本次未擅自改指令文件）

- **ASSET INDEPENDENCE**：setting 文案会盖过场景参考图 → setting 只带光锚，不写房型/墙面/家具；
  房型只有故事必需时才写（诊室、看得见街面的窗）。
- **CAMERA**：`first_person` 会让观众的手/身体入画 → 只有当"听者的身体"本身是剧情时才用，
  独白类一律 `observer`。
- **CRAFT RULES**：手势要写「手心朝向镜头 / 每句收起一根手指」这类可执行的细节，否则只剩一个抬手的动作；
  垂直关系（楼下/楼上/正下方）必须写明观察角度，否则渲染成平铺的远景。
- **ROLES**：`voice` 不得以句号结尾——它会被拼进提示词成 `<voice>: <d>…`。
- **ENGLISH EXECUTION COPY**：要独占的道具写成「全场唯一的那一个」，只写"一个"会出现复制品。

## 复现

```bash
python scripts/validate_storyboard_asset.py docs/storyboard-assets/奔驰车主的月充一万.json
python _live_three_storyboards.py          # 三条依次实跑（run_name 必须 ASCII）
python _live_qc_one.py <slug> <video.mp4> 0,40,90,130,170   # 抽帧 + 字幕条 + ASR
```

注意：**改过分镜 JSON 后必须换 seed / run_name 重提**，同 id 的 api_prompt 会复用旧 bundle 与旧计划。
