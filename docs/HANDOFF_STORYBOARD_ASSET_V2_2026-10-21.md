# 分镜资产 V2 开发交接（2026-10-21）

## 当前状态

- 当前开发分支：`codex/storyboard-assets-v2`。
- 基线：`c6ccec8`（1.4.3）。分支已包含 Issue #20 修复提交 `8347f53`；对应远端独立提交为 `a40efa2`。
- 分镜功能仍是未提交修改。工作区还存在其他任务和用户文件；禁止 `git add -A`，不要删除未确认的 untracked 文件，不要纳入 `tests/fixtures/storyboard_pov_xiaod_cloud.json` 的既有删除。
- ComfyUI 已同步并重启，端口 `8188`，队列为空。
- 主工作区测试：`1850 passed, 10 skipped`；部署目录相关测试：`84 passed`。

## 已实现

直接扩展现有 `MiniMaxH3StoryboardGenerator|Mie`，没有新增平行生成节点。输入包括角色数量和设置、有序动作/对白、镜头数、目标时长、机位、风格、场景和导演约束；输出包括 `MIE_STORYBOARD`、完整 JSON、可读预览和镜头数。

未发布的 version=2 资产支持 1–8 个稳定编号角色、画面内/画外/稍后入场、dialogue/action/enter/exit、有序原文锁定、可选场景资产。用户原始 look、setting、constraints、source_script 保持不变；独立英文 `execution` 包含 look、setting、visible_state、voices、directions。V3 确定性编译，不再次调用 LLM，角色稳定映射到 `<Picture N>` 和 `S<N>`。

核心文件：

- `nodes/assets/storyboard_contract.py`：v2 校验、帧数分配、V3 编译。
- `nodes/llm/storyboard_asset_generator.py`：LLM 生成、最多两次语义纠正、失败关闭。
- `nodes/llm/minimax_h3_storyboard_generator.py`：公开节点扩展。
- `nodes/llm/prompts/h3_storyboard/system_asset.txt` 和 `asset.schema.json`：ComfyUI/外部聊天共用指令与 Schema。
- `scripts/validate_storyboard_asset.py`：脱离 ComfyUI/torch 的校验器。
- `docs/STORYBOARD_GENERATION.md`：用户说明。

## 关键设计结论

### 禁用内容

用户可在中文 constraints 中写“不出现、不接、不吃”，但 H3 执行稿不逐字翻译否定句，而是生成英文正向可见状态。例如：男猫不接苹果 → 女猫是唯一持有者、男猫双手可见且空着；不出现刺猬/回忆 → 两名角色留在当前沙发对话，故事只通过对白表达。

`execution.visible_state` 和动作 directions 拒绝 no、not、never、without、avoid 等否定控制词。送给 H3 的导演内容全部为英文；中文只在 `<d>[Chinese] ...</d>` 内。

### 道具与参考图

指定道具保持数量和持有者；没有明确接受的递物仍归递出者。苹果案例已因此稳定。角色参考图是强条件：刺猬案例参考图自带杯子和书，出现递杯动作，用户确认可接受。编译器后来补充“纯对白时参考图附属物留在初始持有者”。最稳妥的角色资产仍应手部清楚、没有无关手持物。

### 短对白补说

刺猬初次成片最后一镜只有一句短对白，却分到约 7 秒，正确句子后出现无意义语音。根因是空余音频过长，不是声线映射。

`allocate_frames` 现在把单一短对白镜头限制为估算语音预算加两档 H3 帧网格余量，多余总时长分给多事件/动作镜头；若全部镜头都短而目标时长过长则明确报错。收尾从 192 原始帧降到 90 帧（3.75 秒），单镜复测只有目标句，没有后续补说；爷爷开口，小猫闭嘴。

## 真实验证

### 苹果

- MiniMax 节点 3 seed：3/3 有效；外部聊天 + 系统指令也有效。
- 完整视频 20.041667 秒、481 帧；0–20 秒逐秒抽帧始终只有一颗苹果，由女猫持有，男猫没有接到。
- ASR 覆盖 7 句原始对白，没有念导演指示。
- 视频：`C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/output/storyboard_generation_20261021/storyboard_v2_apple_positive_20261021.mp4`。

### 刺猬

- MiniMax 节点 3 seed：3/3 有效；外部聊天 + 系统指令也有效。
- 初次完整成片四段均为沙发上的两名角色，没有刺猬、板栗、踩踏或回忆画面。
- 初次完整片末尾有补说杂音；90 帧修复后收尾已通过单镜音频/画面检查。
- 修复收尾：`C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/output/h3_chains/storyboard_v2_hedgehog_tail_audiofix_v2_20261021/final/storyboard_v2_hedgehog_tail_audiofix_v2_20261021.mp4`。
- 尚未用新帧分配重跑完整约 30 秒刺猬视频。当前证据分别覆盖：旧完整片的画面安全性，和新 90 帧收尾的音频修复。

详细报告：`docs/qa/storyboard-generation-20261021/report.md`。

## 外部交付

目录：`C:/Users/administered/Downloads/Mie-分镜资产生成-20261021`。包含系统指令、Schema、独立校验器、输入模板、节点/外部生成样例、生成与渲染工作流，已更新到 visible_state 和短对白时长规则。

## Issue #20

- 远端分支：`codex/fix-save-auto-ambience`；提交：`a40efa2 Fix auto asset ambience input`。
- 三个 Auto 保存节点均接收 `ambience`，场景值继续透传保存。
- 新增 INPUT_TYPES↔FUNCTION 签名机械守卫和实际场景氛围保存测试。
- 相关测试 73 passed。隔离 worktree 中另一个既有测试因 ComfyUI import 环境未加载被排除；主工作区全量测试通过。
- 已 cherry-pick 到当前分镜分支为 `8347f53`。

## 提交范围与风险

本任务文件至少包括：`__init__.py`；`nodes/assets/nodes.py`、`storyboard.py`、`storyboard_contract.py`；`nodes/llm/minimax_h3_storyboard_generator.py`、`storyboard_builder.py`、`storyboard_asset_generator.py`；系统指令/Schema；独立校验器；两个新测试和两个 v2 fixture；节点契约基线；用户文档、输入模板、设计文档与最终 QA 报告。

工作区大量 `_live_*`、旧 QA、备份、`.zcodeignore` 和其他未跟踪文档不应自动纳入。先用 `git status --short`、`git diff --stat` 和逐文件 diff 确认。

## 建议接续顺序

1. 用新分配完整重跑刺猬视频；预期帧数 `[141, 209, 345, 90]`，总时长约 30 秒。
2. 验收 7 句对白、结尾无补说、全片无台词故事可视化。
3. 更新刺猬渲染工作流的展示用 fallback plan；实际执行已经由 V3 动态计划覆盖。
4. 跑 `.venv/Scripts/python.exe -m pytest -q tests`，预期 `1850 passed, 10 skipped`。
5. 核对源码与部署运行文件 SHA-256，确认重启后的节点定义。
6. 只暂存本任务文件，审查后提交并推送 `codex/storyboard-assets-v2`。
7. 完成分镜资产生成节点后，再与用户讨论下一阶段。
