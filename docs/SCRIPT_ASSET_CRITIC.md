# 剧本资产 · 连续性审片（critic）

**一句话：** 资产过了**契约校验**之后，再让一个"读者"跨镜头复核一遍；发现观众看得见的矛盾就回修重摇。
**One line:** after the asset passes the CONTRACT check, a second reader looks across the shots and repairs what a viewer would notice.

## 为什么有它

契约校验只看**一份资产本身**：字段形状、台词逐字一致、出入场重放、时长预算。它看不见这些：

- 角色 1 在 scene_01 穿红外套，在 scene_03 穿蓝外套；
- 那颗"唯一的苹果"同时在两个人手里；
- 同一拍里"手里拿着苹果"和"双手空空"互相打架；
- 夜戏的光锚预设配上一句"明亮的日光"；
- 某角色在还没上场的镜头里被写进了动作说明。

这些正是 2026-10/11 真机渲染里反复出现的缺陷形态。审片把"人眼看出来的"变成"机器拦下来的"。

## 两层

| 层 | 位置 | 成本 | 干什么 |
|---|---|---|---|
| 确定性 critic | `nodes/assets/storyboard_critic.py` | 0（纯规则） | 服装换色、道具易主/双持、空手与持物打架、角色不在场、时段与光锚冲突、一镜多动作、开场状态写了还没出场的人 |
| 模型 critic | `nodes/llm/storyboard_critic.py` + `prompts/h3_storyboard/critic_asset.txt` / `critic_shots.txt` | 每次生成 +1 次调用 | 覆盖（剧本每一行都在吗）、连续性、单镜可演、节奏——只有文本，不看渲染 |

**模型 critic 不看画面**：它读的是资产 JSON。所以它不把模型拉回渲染循环，"资产生成一次、渲染零模型"这条卖点不变。

## 协议（值得记住的三条）

1. **没有 path 的发现一律丢弃**——指不出位置的批评是噪音。
2. **读不懂的回复算 unsure，unsure 永不拦截**：资产照常交付，附一条"这次没做连续性复核"的警告。
3. **只有 critical 才买一次回修**：minor 只是笔记。

契约失败仍然是硬失败（资产就是契约）；**critic 失败不是**——3 次修不好就交付"通过契约校验的那一版 + 警告"，绝不把整条节点变成失败。

## 覆盖范围（v1 / v2 / v3）

| 轨 | 节点 | 审片方式 |
|---|---|---|
| V1 | `MiniMaxH3LoopPromptGenerator`（自动分镜） | 确定性 + 模型 critic，critical 买一次重摇 |
| V2 | `MiniMaxH3LoopPromptGeneratorV2` | 同上（继承 V1） |
| V2 资产 | `MiniMaxH3StoryboardGenerator`（剧本资产生成器） | 确定性 + 模型 critic，回修走已有的修复循环 |
| V3 | `BuildPlanFromStoryboard` / `MiniMaxH3LoopPromptGeneratorV3` | 只跑确定性层（零模型），结论写进 run summary |

三个节点都有 `critic` 开关（`on` / `off`，默认 `on`）。关掉 = 只做契约校验，快、零额外调用。

## 怎么关、怎么开

- 节点上 `critic` 选择器：`off` 即刻退回旧行为（契约校验），不额外调用模型。
- `critic` 在缓存键里（`IS_CHANGED`），改开关会重算。

## 提示词侧的对齐

SYNC PAIR 的两份系统提示（`nodes/llm/prompts/h3_storyboard/system_asset.txt` 与 `docs/script_asset_system_prompt.txt`）同步补了四条 craft 规则 20–23，让模型一开始就少犯 critic 会抓的错：

- 20 一个人、一件衣服、一个道具归属，全片一致（换装/易主必须写成动作）；
- 21 一镜一拍，多动作就 `camera_cuts`；
- 22 一只手要么拿着要么空着，不能同时；
- 23 复制里的时段必须和 light 预设一致。

## 已知边界

- 确定性层靠英文词表（执行文本按契约就是英文），中文写法的道具/服装不会被逐字识别——模型 critic 负责那一层。
- 模型 critic 只看资产，看不见渲染；渲染期的连续性（尾帧 vs 首帧、色偏）不在本轮范围。
- 旧轨（V1/V2）的分镜板没有执行文本，确定性层能查的很少，主要靠模型 critic。
