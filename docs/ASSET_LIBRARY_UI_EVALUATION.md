# 资产库节点 UI 可用性评估：动态下拉与前端 JS 的可行性

> 日期：2026-10-09。评估「把 Load Asset Batch / Save Scene Asset / Load Character Asset 等节点从『手打名字』改成『下拉可选、一眼能读』」的实现路径。只评估，不实现。
> 环境：本地 ComfyUI 0.38.0（frontend package 1.32.5），LAN 机 0.39.0。插件已有 `WEB_DIRECTORY = "./js"`，里面有 7 个在用的前端扩展。

---

## 结论

**可行，而且比预想的便宜。** 推荐方案是「轻 JS」：Python 侧加一个扫描端点 + `VALIDATE_INPUTS` 豁免，前端一个约 300–400 行的扩展，把每个手打名字的地方换成跟随 `library_dir` 动态刷新的下拉框。本地装的 comfyui-ollama 就是这个模式的一比一模板（它的 url 输入框一变 → POST 端点 → 刷新 model 下拉），我们自己插件里的 `classicAspectRatio.js` 也已经做过「combo 跟随同节点另一个 widget」。不需要引入任何新依赖。

预估：Python ~120 行（端点 + widget 调整）、JS ~350 行、计划第五次修订、一轮真机验证（本地 + LAN）。一次会话内可以完成。

---

## 1. 现在的痛点

| 节点 | 痛点 |
|---|---|
| `LoadAssetBatch` | `names` 是多行文本，名字必须**逐字手打**（还要和磁盘 casefold 对上）；想知道库里有什么得另接一个 List 节点看输出 |
| `Load/Save Character Asset`、`Load/Save Scene Asset` | `name` 同样手打；保存时看不到已有名字，**静默覆盖**的风险全靠记性 |
| 所有 7 个节点 | `library_dir` 每个节点都要手打一遍绝对路径 |
| `MiniMaxH3LoopPromptGeneratorV2` | `asset_roster` 接线时不可见；手写时是一坨 JSON |
| List 节点 | 它存在的理由就是「给别的节点查名字」——如果下拉到位，它的使用频率会归零（保留不动，不碍事） |

## 2. 技术事实（决定方案选型）

调研结论（本地 ComfyUI 0.38 源码，行号在括号里）：

1. **combo 选项在 Python 侧不能依赖节点实例的其他 widget。** `INPUT_TYPES()` 是类级调用（`server.py:764`），拿不到某个节点实例的 `library_dir`。这是核心限制，绕不开。
2. **`/object_info` 每次请求都重新执行 `INPUT_TYPES()`**（`server.py:808-819`），所以按 R 能刷新**类级**选项——但只在「库目录是全局固定」的前提下才有用。
3. **前端 JS 完全支持「widget A 变化 → fetch 自定义端点 → 改 widget B 的选项」**：combo 菜单打开时实时读 `widget.options.values`（前端 ComboWidget.ts:143-144），直接赋值数组即可生效；`widget.callback` / `node.onWidgetChanged` 提供变化钩子。
4. **Python 加自定义端点的标准写法**：`@PromptServer.instance.routes.post("/mie/...")`（Impact Pack `impact_server.py:60` 同款）。
5. **本地一比一模板**：`custom_nodes/comfyui-ollama/web/js/OllamaNode.js:12-94` —— url 输入变化 → `fetch("/ollama/get_models", {body: {url}})` → `modelWidget.options.values = models` → 越界回退 `models[0]` → 加刷新按钮。和「`library_dir` 变 → 扫描 → 刷新 `name` 下拉」结构完全同构。
6. **combo 值不在类级选项列表里会被执行校验拒绝**（`execution.py:1047-1079`），插件的标准解法是 `VALIDATE_INPUTS` 覆盖豁免（VHS `load_video_nodes.py:471-474` 同款）。
7. **多选没有省心的原生方案**：0.38 的 `MultiCombo` 是 Vue DOM 组件，不支持 remote 选项、JS 操纵它的选项要走组件内部状态（useComboWidget.ts:70-71 明确 TODO）。生态里多选的通用做法是「隐藏数据 widget 存结果 + JS 复选列表」（KJNodes）或「一排单选下拉」（rgthree）。
8. **新旧前端 API 并存但旧写法稳**：`import {app} from "/scripts/app.js"` 在 0.38/0.39 都由内置 shim 转发到新全局。我们仓库现有 7 个扩展全部用的旧写法，都在两台机器上工作正常。应继续用旧写法，避开 `options.values` 传函数（已标废弃）和 `MultiCombo`。

## 3. 三个方案

### 方案 A：零 JS，类级下拉（固定库目录约定）

库目录不再是每节点的 widget，改为全局一处（环境变量或配置文件），`INPUT_TYPES()` 启动时扫描出选项，`name` 变原生 combo。

- 优点：纯 Python，约 60 行；按 R 刷新。
- 缺点：**放弃「每个节点可以指自己的库」这个已有设计**；只有类级刷新（在资源管理器里新建了资产要按 R）；多选依旧没有；配置文件必须放插件目录外（镜像同步会删），又要多解释一条路径规则。
- 判定：能用，但把灵活性换成了方便，方向不对。

### 方案 B：轻 JS，逐节点动态下拉（推荐）

`library_dir` 保持每节点 widget；Python 加 `POST /mie/assets/list` 端点（收 `{library_dir}`，回 `{characters: [...], scenes: [...]}`，直接复用 `library.list_characters/list_scenes`，路径校验照旧）；前端扩展做四件事：

1. 监听 `library_dir` 变化（防抖 500ms）→ POST 端点 → 刷新本节点所有资产下拉的选项。
2. 节点上加一个「🔄 刷新」按钮 widget，随时手动重扫（解决「资源管理器里刚建了资产」）。
3. `LoadAssetBatch` 的多选：**一排下拉槽位**（Slot 1: `[夏栀 ▾]`、Slot 2: `[便利店 ▾]`、Slot 3: `[＋ 添加 ▾]`），选项列表区分 `角色 / 场景` 分组（前缀或分组标题），JS 把槽位选择按顺序写回隐藏的 `names` widget（序列化不变，**工作流文件完全兼容**，老工作流的手打 names 照常工作，加载时反向渲染成槽位）。槽位顺序 = `<Picture N>` 顺序，界面上直接读出绑定关系——「容易读」就落在这。
4. `Save *` 节点的名字框：下拉列出已有名字 + 首项「✏️ 新名字…」，选它时显示一个文本框（`mieForStart.js` 已有现成的 widget 显隐模式）。选已有名字 = 明确的覆盖意图，静默覆盖的担忧消除。

- 优点：零灵活性损失；即时刷新；下拉处处可选；老工作流逐字节兼容（数据 widget 不变）；模板和先例全在本地。
- 缺点：约 470 行新代码；需要一轮双机真机验证；前端扩展是新的维护面。
- 工作量：Python ~120 行、JS ~350 行。

### 方案 C：重 JS，完整 UI（弹窗选择器 + 图片预览 + roster 表格）

在 B 之上：点击槽位弹 Impact-segs-picker 式选择器，里面渲染 portrait 缩略图（端点加一个 `GET /mie/assets/portrait?dir&name` 返回 PNG）；V2 的 roster 接线状态在节点上渲染成小表格。

- 优点：体验最好，选角色时直接看脸。
- 缺点：JS 量翻倍以上（~800 行）；DOM overlay 是前端升级时最容易碎的部分；缩略图端点要处理路径校验和缓存头。
- 判定：作为 B 之后的二期，不 blocking。

### 对比

| | A 零JS | B 轻JS（推荐） | C 重JS |
|---|---|---|---|
| 每节点自己的库目录 | ✗ | ✓ | ✓ |
| 下拉选择 | 类级、按R | 逐节点、即时 | 弹窗+预览 |
| 多选（批量加载） | ✗ | 下拉槽位行 | 复选弹窗 |
| 老工作流兼容 | 破坏（widget 变 combo） | ✓（数据 widget 不变） | ✓ |
| 新代码量 | ~60 行 py | ~120 py + ~350 js | +~450 js |
| 新维护面 | 无 | 一个 JS 扩展 | DOM overlay 依赖 |

## 4. 方案 B 的几个具体设计点

- **端点安全**：`POST /mie/assets/list` 只接受 `{"library_dir": "..."}`，内部走 `library.list_characters` / `list_scenes`，沿用 `validate_library_dir`（拒绝插件目录内、相对路径）。返回的名字附带 kind，供下拉分组。不返回文件内容。
- **校验豁免**：`name` 改成 combo 后（占位选项 `["（先填 library_dir）"]`），Python 加 `VALIDATE_INPUTS(name=...)` 直接返回 True，由执行层的 `load_*` 自己报「找不到该名字」——错误信息本来就更具体。
- **`library_dir` 本身的便利**：可以再加一个 ComfyUI 设置项（`app.extensionManager.settings` 注册 `MieNodes.AssetLibrary.DefaultDir`）+ 端点读写；节点留空 = 用默认值。注意默认值**不能**存插件目录内（镜像同步会删），存 `<ComfyUI>/user/<user>/mienodes/asset_library.json`。这一条是独立小改进，可做可不做。
- **V2 的可读性**：`asset_roster` 接线时本来就是数据不可见——B 不解决它，也不需要解决（信息在 LoadAssetBatch 的槽位行上已经可读）。手写 JSON 的场景在 B 之后基本消失。
- **`IS_CHANGED` 不受影响**：数据 widget（`names` 字符串）没变，缓存键照旧。

## 5. 边界与计划修订

`PLAN_ASSET_LIBRARY.md` 第 2 节仍写着「不做库的检索界面和前端扩展」，§2 的 v1 取舍清单里也有「动态下拉需要前端 JS，越界了」。方案 B 需要第五次修订撤销这一条，理由记录为：字段集与 slot 契约在真实使用中已经稳定（2026-10-09 已提交两版），当时「等字段集稳定再议」的前提已满足；用户以实际使用体验提出了需求。检索界面仍然不做（没有搜索框、没有管理面板），做的只是「输入辅助」——下拉和刷新，这与「检索界面」是两回事。

## 6. 风险

| 风险 | 评估 |
|---|---|
| 前端版本差（本地 0.38 / LAN 0.39） | 低。全部用 legacy `app.registerExtension` + `options.values` 数组赋值 + `addWidget`，两版本都有 shim；避开 MultiCombo / remote combo / 函数式 values |
| 前端大版本升级碎掉 | 中。DOM overlay 越少越稳——这也是把 C 推迟的原因；B 的实现全部走 canvas widget 标准 API |
| 下拉值绕过校验造成错绑 | 低。豁免的只是「值 ∈ 类级选项」这一层；执行层 `load_character_batch` 对不存在/跨类重名的报错原样生效，且槽位行让绑定关系在界面上可见，反而更难接错 |
| 无 JS 测试基建 | 现实约束。仓库没有 JS 测试；验证走真机双端（本地 + LAN）+ 老工作流加载回归（手打名字的老工作流必须照常跑），按 h3_loop_live_e2e_runbook 的路数 |
| mirror 同步 | 无新增风险；新增文件都在 `js/` 和 `nodes/assets/`，随脚本正常同步 |

## 7. 分期建议

1. **第一期（方案 B 主体）**：端点 + Load/Save 四个节点的下拉 + LoadAssetBatch 槽位行 + 刷新按钮 + 计划第五次修订。验收：双机真机上，纯鼠标完成「选库 → 选角色和场景 → 生成 roster」全流程；三个 2026-10-09 之前保存的老工作流加载后照常运行。
2. **第二期（可选增强）**：默认库目录设置项；portrait 缩略图弹窗（方案 C 的部分）；V2 roster 只读摘要。

---

## 附：本评估引用的先例与核心代码位置

| 用途 | 位置 |
|---|---|
| 一比一模板（widget→端点→combo） | `custom_nodes/comfyui-ollama/web/js/OllamaNode.js:12-94` |
| 本插件已有的 combo 跟随 widget | `js/classicAspectRatio.js:54-82` |
| 本插件已有的 widget 显隐切换 | `js/mieForStart.js` |
| 隐藏数据 widget + 复选列表（多选备选） | `custom_nodes/ComfyUI-KJNodes/web/js/ideogram4_prompt_builder.js:414-444` |
| 自定义端点写法 | `custom_nodes/comfyui-impact-pack/modules/impact/impact_server.py:60` |
| VALIDATE_INPUTS 豁免 combo 校验 | `custom_nodes/comfyui-videohelpersuite/videohelpersuite/load_video_nodes.py:471-474` |
| /object_info 每次重跑 INPUT_TYPES | `ComfyUI/server.py:808-819` |
| combo 校验拒绝点 | `ComfyUI/execution.py:1047-1079` |
