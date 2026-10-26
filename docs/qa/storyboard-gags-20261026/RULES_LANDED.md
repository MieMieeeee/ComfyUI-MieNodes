# 本次实跑回流到系统指令的规则（2026-10-26 落地记录）

提案原文见 [INSTRUCTION_UPDATE_PROPOSALS.md](INSTRUCTION_UPDATE_PROPOSALS.md)。
本文件记录**哪些落地、落在哪、哪些按用户裁定不采纳**，避免以后重复讨论。

## 已落地

| 提案 | 落点 | 形式 |
|---|---|---|
| A1 单一角色独占台词的续生成片段 = 禁用形态 | 两份 prompt | 提示词（docs CRAFT 8 / 节点版 SHOTS 段） |
| A2 visible_state 开头 = 首帧状态 | 两份 prompt | 提示词 |
| A3 环境文字语言锚点必须进 setting | 两份 prompt | 提示词（随 constraints 不被编译那条一起） |
| **A4 voice 结尾句号拼出 `…pace.: <d>`** | `nodes/assets/storyboard_contract.py` | **代码归一化**（`_bare_phrase`：role.voice 与 execution.voices 统一去尾标点），提示词补一条"bare phrase, no trailing punctuation" |
| A5 voice 需匹配实际接线的角色资产，否则留空 | 两份 prompt ROLES | 提示词 |
| A6 自改脚本的数字口语化 | 两份 prompt | 提示词（上轮已落） |
| A7 时长贴可行下限 | 两份 prompt DURATION | 提示词 + `storyboard_warnings` 零缓冲预警（上轮已落） |
| **B1 setting 文案盖过接线的场景参考图** | 两份 prompt | 提示词：可接场景资产时 setting 只留光锚；房型仅在故事必需时写且不接场景资产 |
| **B2 `first_person` 会画出观众的手** | 两份 prompt CAMERA | 提示词：独白一律 observer；`七岁养伤的一年` 已改 observer |
| **B4 垂直关系（楼下）必须写观察角度** | 两份 prompt | 提示词（docs CRAFT 9 / 节点版 LIVE RENDER NOTES） |
| **B5 道具要写"全场唯一那一个"** | 两份 prompt | 提示词（docs CRAFT 10 / 节点版 LIVE RENDER NOTES），并如实写明 H3 仍可能复制小道具，必要时去掉道具而非硬扛 |
| B7 H3 烧录字幕不稳定 | — | 不进指令；QA 流程按 ASR 验收（本次已用 faster-whisper 逐句核对，三条成片台词正确） |
| C1 `run_name` 只能 ASCII | — | 不进指令；接线背景知识 |
| C2 改过分镜 JSON 必须换 seed/run_name 重提 | — | 不进指令；已有记忆 `storyboard-hand-edit-node-cache-gotcha` |
| C3 查最终提示词看 `segments/*.prompt.txt` | — | 不进指令；`prompt_history/*.json` 存的是分镜正文不含前缀 |

## 按用户裁定不采纳

| 提案 | 理由 |
|---|---|
| **B3 手势要写可执行细节（掌心朝向、每说一个岁数收起一根手指）** | 用户裁定：不追求手指渲染，"加了反而容易出错"。让 H3 逐根收手指，手部形变风险高，而"数数"这个笑点由台词承载即可 |
| **B6 手指收拢到后期会变形、数量与台词不对应** | 与 B3 同源；不引入手指细节即不触发该类形变 |

按此裁定，`docs/storyboard-assets/七岁养伤的一年.json` 的手势文案已从"逐根收起手指"
改为不依赖手指细节的胸前点数示意动作（`event_001` direction 与 constraints 同步），
资产重新校验通过、零预警。

## 验收方式

成片台词一律用 ASR 核对，不看烧录字幕——H3 的字幕有无不稳定（同一资产不同 seed 表现不同），
拟声词（如"嗯？？？"）只在音频里。
