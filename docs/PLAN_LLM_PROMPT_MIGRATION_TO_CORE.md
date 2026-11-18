# PLAN：LLM/Prompt 全量上移 Core（MieNodes 只留壳）

- 日期 2026-11-17。施工仓：本仓（ComfyUI-MieNodes，main `c0a06a4`）+ `F:\ComfyUI-Mie-Prompt`。
- 测试基线（2026-11-17，main c0a06a4）：**2055 passed, 10 skipped（20s）**；engine 仓 272 passed。
- 发版前基线：mie-prompt 0.1.6 / mie-craft 0.1.6 / mie-llm 0.1.5 均已上 PyPI 并双端安装。
- 上游依据：engine 仓 PLAN.md 的 D11/D13（生成器全族进 Core；MieNodes 只留节点壳）——
  Phase 4 只迁了 H3 族，本计划收尾其余全部。
- 已拍板的产品决定：loader 合一以 Core（mie_prompt）为准；webapp 侧的"任意族提示词增强"
  接口**暂不做**（迁移不为其预留接口形状，但结果天然解锁它）。

> **2026-11-18 核对**：批 1、2a、2b、2c 已实施；批 3 按下文修正为核实既有接口。
> 当前 `mie_prompt/prompts/` 有 21 个族目录（含后加的 h3_image），本仓只保留 loader 与
> package shim。完成标准里的“目录清空”应理解为移除族文本，不删除兼容入口；
> translator、storyboard_builder、custom_system_prompt 等节点壳按职责保留。
> 后续已完成全族 enhancer、十个通用生成方法和 H3 loop/storyboard 编排迁移，三个共享包已发布并安装 0.2.0。收尾记录见
> `F:/ComfyUI-Mie-H3-Engine/notes/split-completion-2026-11-18.md`。首轮审计边界见
> `F:/ComfyUI-Mie-H3-Engine/notes/split-implementation-audit-2026-11-18.md`。

## 1. 现状盘点（2026-11-17 实测）

`nodes/llm/` 共 35 个 py：**7 个已是指向 mie_prompt/mie_llm 的 shim**，28 个实体。
`nodes/llm/prompts/` 下 **20 个族目录**（bernini、flux2、flux_klein、flux3_klein、
frame_transition、h3、h3_loop、h3_storyboard、hunyuan、ideogram4、krea2、ltx25、
moss、qwen…其中 h3 三族已在 Core）。tensor 耦合实测：krea2/translator/
user_input_enhancer/prompt 家族文本为 **0**；moss_image generator 14 处、
prompt_generator 底座 8 处（`import folder_paths` + `core.utils.image_tensor_to_data_url`）。

## 2. 拆分线（不可逾越）

**Core（mie_prompt/mie_llm）不得 import ComfyUI**（engine PLAN §202）。三层去处：

| 层 | 内容 | 去处 |
| --- | --- | --- |
| 提示词文本 + 族目录 | `nodes/llm/prompts/<族>/` 的 txt、`*_prompts.py` 里的模板常量与解析 | `mie_prompt`（`prompts/<族>/` + 模块） |
| 纯文本逻辑 | 构造 prompt、解析回复、增强改写、预算/翻译/分段 | `mie_prompt` 同模块（h3_prompts 先例） |
| 节点壳 | `INPUT_TYPES`、IMAGE tensor→dataURL、`|Mie` 命名、ComfyUI 前端 | **留本仓**，改 import Core |

tensor 工具（`core/utils.py` 的 image_tensor_*）按 engine PLAN §618 **留在本仓**，
Core 的多模态接口以 "dataURL 字符串进出" 为界（webapp/autodraft.py `_invoke` 是
现成范式）。

## 3. loader 合一（已拍板：以 Core 为准）

- 本仓 `nodes/llm/prompts/loader.py` 与 `mie_prompt.prompts.loader` 并存。合一后
  **逻辑名解析、逐字读取、永不 .format()** 以 Core 为准；本仓 loader 删除，
  原调用点改 `from mie_prompt.prompts.loader import ...`。
- 本仓按**文件路径**加载 prompt 的历史习惯（engine PLAN §1103 的教训：产品层
  因此无法干净 import）在迁移中一并消灭：Core 侧只暴露逻辑名 API。

## 4. 批次

### 批 1 · 零 tensor 纯文本逻辑（最小风险，先打通"webapp 可 import"）

> **执行修正（2026-11-17）**：
> 1. enhancer 落 **mie_craft**（`mie_craft/loop_enhancer.py`）而非 mie_prompt——它依赖
>    `mie_craft.h3_loop_prompts.parse_reference_mode`，放 mie_prompt 会造成 Prompt→Craft
>    反向依赖（依赖方向 LLM ← Prompt ← Craft 不许倒）。
> 2. **loader 合一挪到批 2c 收尾**：本地 prompts/ 还有 17 族未上移，过早 shim 化会 404。
>    批 2 每迁一族改该族 import 为 `from mie_prompt.prompts.loader import ...`，全部迁完
>    删除本地 loader。
> 3. `custom_system_prompt.py` **留本仓**：其存取层绑定宿主仓路径（user_system_prompts.json
>    在仓根）+ tensor 采样，按拆分线不迁；Core 侧等 webapp 需求牵引再定。
> 4. 实现侧两处行为修复随迁：`_CACHE_ROOT_OVERRIDE` 优先级提到最高（原实现里调用方
>    cache_root 参数压过全局 override，隔离失效）；ComfyUI 缓存根由壳解析显式传入。
>    MieNodes 测试三处 fixture 改 patch `mie_craft.loop_enhancer`（shim 上 patch 无效，
>    与 h3_prompts 迁移时的教训一致）。
- 对象：`translator.py`、`custom_system_prompt.py`、`dialogue_segmenter.py`、
  `invoke_budget.py`、`minimax_h3_loop_user_input_enhancer.py` 的文本构造/解析部分。
- 动作：逻辑进 mie_prompt（新模块或并入现有），本仓文件变 shim
  （module-level `__getattr__`，照 `h3_prompts.py` 先例——星号导入会丢下划线名）。
- 验收：本仓全量 tests 基线不变；mie_prompt selftest；`python -c "from mie_prompt import ..."`
  在系统与 embedded 双端成立。

### 批 2 · 提示词族上移（2a→2b→2c，按使用热度）
- 2a：qwen_image_21、ideogram4（近期活跃，含 UPSTREAM.md/构建脚本）。
- 2b：bernini、flux_klein、flux3_klein、flux2、krea2、ltx25、scail2。
- 2c：moss、hunyuan、frame_transition 及 prompts/ 其余族 + `storyboard_builder.py`。
- 每族动作：`prompts/<族>/` 文本 → `mie_prompt/prompts/<族>/`；`<族>_prompts.py`
  模板/解析 → mie_prompt 模块；本仓同名文件变 shim；`<族>_prompt_generator.py`
  改 import（此批不拆 tensor——generator 里现成的 tensor 调用保留，只换文本来源）。
- 版权：bernini 等 Apache 文件的版权头 **原样随文迁移**；`UPSTREAM.md` 跟随族目录。
- 验收：每族现有 tests 全绿；`_node_contracts_baseline.json` 节点名零变化（只迁实现，
  不动节点契约——D9 的冻结资产节点不在本计划范围）。

### 批 3 · 多模态 generator 拆接口

> **执行结论（2026-11-17）：无需拆分——接口早已在 Core。**
> `build_multimodal_user_content`（文本 + dataURL 列表 → OpenAI 风格 content）
> 与 `image_tensor_batch_to_data_urls`（IMAGE tensor → dataURL 列表）本来就住在
> **mie_llm.media**；MieNodes 的 `core/utils.py` 只是 re-export，各 generator 壳
> 已经在用它。计划里"在 mie_prompt 定义 multimodal 接口"的前提不成立——接口在
> 更下层的 mie_llm，位置更对：Core 不依赖 ComfyUI，media 只认 dataURL 与 tensor
> 形状，不认 ComfyUI。
>
> 批 3 因此收敁为**核实**（接口存在、两端可用、各 generator 已在用），
> 而不是新写一层。若日后 webapp 要"任意族提示词增强"，直接用
> `mie_llm.media.build_multimodal_user_content` + `mie_prompt` 的族文本即可。
>
> 遗留的真实耦合只剩 tensor 工具本身（`image_tensor_*`），按 engine PLAN §618
> 留在 MieNodes ——它依赖 ComfyUI 的 IMAGE 约定。
- 对象：`prompt_generator.py` 底座、`moss_image_prompt_generator.py` 等 tensor 节点。
- 动作：mie_prompt 定义 multimodal 调用接口（文本+dataURL 列表进、回复出）；
  本仓壳只做 IMAGE→dataURL（继续用本仓 core.utils）+ INPUT_TYPES。
- 验收：同批 2，另加一条端到端：任一 multimodal 节点在本地 ComfyUI 起图走通
  （真 LLM 一次）。

## 5. 发布与部署纪律（每批一次）

1. 本仓 commit（可独立审阅）；mie_prompt 功能 commit。
2. mie_prompt bump → build → twine（凭据走 mie-credentials 的 pypi.prod，token 不进
   命令行/日志）→ PyPI → push。
3. 双端升级：embedded + 系统 python `pip install -U -i https://pypi.org/simple`。
4. `python scripts/sync_to_comfyui.py --target local` 部署到 custom_nodes。
5. 回归：本仓全量 tests；engine 仓 `webapp_tests`（批 2a 与批 3 必跑全量 272，
   其余批跑 test_api+test_storyboard_intake）。
6. ComfyUI 重启后 8199 健康检查版本不变（webapp 代码不动则无需重启 8199）。

## 6. 风险与既知坑

- **shim 上 patch 失效**：MieNodes 测试 monkeypatch 打在 shim 上碰不到实现
  （h3_prompts.shim 头部已写明）——迁移时逐个把测试的 patch 目标改指
  `mie_prompt.<模块>`。
- **prompt 文本路径加载**：迁移中发现的相对路径/脚本目录加载一律改为逻辑名。
- **族目录里的构建脚本**（如 `scripts/build_ideogram4_system_prompts.py`）随族走，
  产物进 Core，脚本可留本仓（生成工具）但输出目录改指 Core 源码树。
- `nodes/web/short_drama_webui.py`（未跟踪）与本计划无关，不动。

## 7. 完成标准

- `nodes/llm/` 内除 `*_prompt_generator.py`（节点壳）与 `__init__.py` 外全部为 shim；
  `nodes/llm/prompts/` 目录清空（文本全部在 mie_prompt）。
- 本仓全量 tests 与基线同数全绿；engine 仓 272 全绿（批 2a/批 3 后）。
- `python -c "import mie_prompt, mie_craft, mie_llm"` 三包在双端可用，
  本仓对 prompts 的 import 只经 mie_prompt。
