# 项目计划：H3 引擎 Fork 与短剧平台

> **状态**：2026-11-05，已并入**三轮**外部评审意见。评审处置台账见 §16。
> **本文件是唯一真源**，取代 `docs/PLAN_H3_ENGINE_FORK_AND_DRAMA_PLATFORM.md`（v3）与
> `docs/qa/h3-engine-fork-20261105/REVIEW_BRIEF.md`。
> **上游**：`ethanfel/ComfyUI-MiniMaxH3-Context-Loop` @ `43afeff`（GPL-3.0）。
> **前置**：`docs/PLAN_STORYBOARD_ASSET.md`、`docs/PLAN_STORYBOARD_FIRST_ASSET_V3.md`、
> `docs/PLAN_ASSET_LIBRARY_V2.md`。
> 不改动已发布的 MieNodes v1/v2/v3 节点契约。

---

## 0. TL;DR

> ### 治本原则（2026-11-06 定调）
>
> **本项目的核心是尽可能吸纳上游已测过的代码；凡是已经发现的问题，
> 就选最优解，而不是最小补丁。**
>
> 落法：
> - **上游的测试是契约，不是障碍**——上游有测试覆盖的地方先读那个测试，
>   它往往已经写清了上游想要的语义（例如 reference cache 的
>   *"ambiguous semantic recipes must not be chosen by mtime"*），
>   而实现却没做到；此时**让实现去满足上游的测试**。
> - **上游的守卫若是对的，原样保留**；只在真正有缺陷的地方改。
> - **每处改动配一条负对照**：原实现必须报红，否则不是真修。
> - **发现上游的前提是错的，当场纠正并写进文档**。

上游的 H3 长视频生成插件**已停更 22 天**。我们要：

1. **fork** 它成 `ComfyUI-Mie-H3-Engine`（保住 148k 行难啃的生成核心，并修好它在 Windows 上的缺陷）；
2. 把自己的**生成逻辑 / 剧本契约 / 资产库 / LLM 连接器**抽成一个**可 pip 安装的共享核心包**
   `ComfyUI-Mie-Core`——引擎包、MieNodes、产品层三者都是它的壳；
3. 起一个**独立 FastAPI 短剧平台**（网页写剧本 → 生成 → 出片，不打开 ComfyUI 界面）；
4. 用一条 `plan_json` 字符串缝合"剧本 → 生成"；
5. 给 ComfyUI 侧的 Scene Prompt Editor 一个**只读预览**通道。

**核心判断：生成引擎不重写。**

**最紧急的一件事**（§8）：上游的缓存守卫把 `(size, mtime)` 组合当成内容身份，
在 NTFS 与 SMB 上都可能**静默用错内容渲染**。这是 P0 生产缺陷，不是测试问题。

---

## 1. 背景

### 1.1 上游停更（2026-11-05 实测）

```
GitHub API   pushed_at = 2026-10-14T11:08:20Z      (距今 22 天)
             stars 563 / forks 39 / open issues 6 / watchers 5
             git tag: 0 个（从来没有）
```

| 周 | W33 | W34 | W35 | W36 | W37 | W38 | W39 | W40 | W41 | W42 |
|---|---|---|---|---|---|---|---|---|---|---|
| commit | 154 | 147 | 180 | 163 | 163 | 100 | 13 | 5 | 12 | 3 |

- 最后 14 天 **0** commit；最后 30 天 main 13 个、全 ref 20 个
- 7 个 `sceneweaver-*` 分支最后 30 天**全部 0**
- 主导作者 `Ethanfel` 843/954（88%）

**决定性的是 issue：**

- **#117**（2026-10-14 建立，**当天关闭**）——老用户原话：
  *"your project has been unusably buggy… It's an absolute clusterfuck.
  Nothing you've produced in the past month has been stable… it's time to start
  polishing and stop adding."* 作者当天关闭，此后**未再推送任何 commit**。
- **#120**（2026-11-02）*"Are you planning any updates?"* —— **0 条回复**。
- **#118**（2026-10-25）、**#121**（2026-11-04）同样无人应答。

### 1.2 上游规模

| 项 | 数量 |
|---|---|
| Python | 107,696 LOC / 252 文件 |
| 前端 `web/` | 40,585 LOC / 72 文件 |
| 测试 | 532 unittest + 103 `.mjs`（65k LOC） |
| 节点类型 | **87** |
| `chain_nodes.py` | **31,538 行**（顶层生产代码 29.3%） |
| CI | **无** |

### 1.3 我们的耦合性质：数据耦合

全仓**没有一处 import 上游代码**，接口只有一个字符串：
`plan_json` → 上游 `plan_json_input` socket。

**这意味着 fork 的是一个我们没有代码耦合的依赖**，风险面比"改造一个深度耦合的库"小得多。

**但有两处影子实现必须合并**：

1. 上游 `chain_nodes.py:7362 _normalize_plan`（帧网格 / seed 值域 / context_length 集合 /
   continuation_mode 跨字段约束）；
2. 我们的 `validate_plan`（`nodes/llm/minimax_h3_loop_prompts.py:1775`，prompt 三段/六段 schema）。

两者是**同一契约的两个半实现**，各自漂移。fork 后合并成一个。

### 1.4 为什么不是"wrapper 包"

一个 50 行的 wrapper 能解决改名，但不能碰真正要改的三样：

| 要修的东西 | 在哪 | wrapper 能碰吗 |
|---|---|---|
| Windows 缺陷（symlink 权限 / mtime / 路径） | 上游**生产线代码** | ❌ |
| Node 24 测试 harness 不兼容 | 上游 **`tests/_*.mjs`** | ❌ |
| Scene Prompt Editor 读 widget 不读 socket | 上游 **`web/*.js`** | ❌ |

**逼着 fork 的是"必须改上游内部"，不是改名。**

### 1.5 目标

一个**短剧的完整流程**：资产管理 → LLM 生成剧本 → 分镜预览 → 长视频生成 → 成片，
其中界面部分逐步脱离 ComfyUI。

---

## 2. 决策记录

| # | 决策 | 落点 |
|---|---|---|
| D1 | 引擎 = **独立包** `ComfyUI-Mie-H3-Engine` | §3.1 |
| D2 | 产品层 = **独立 FastAPI 进程** | §6 |
| D3 | 分镜资产**对接 Scene Prompts 做预览**；Plan Studio 不参与修改 | §5 |
| D4 | 节点类型名**一律加我们的后缀** | §4 |
| D5 | 共享核心包 `ComfyUI-Mie-Core` | §3.4 |
| **D6** | **不再从上游拉代码**（2026-11-06 裁定）：`upstream` remote **只作溯源参考，永不合并**。原策略“用 merge 不用 rebase”随之作废 | §12 R1 |
| D7 | 产品层 **v1 包含站内剧本生成** | §6.2 |
| D8 | 部署拓扑 **v1 同机** | §6.1 |
| D9 | MieNodes 旧资产节点**弃用 + 冻结** | §14 Q-G |
| D10 | 引擎包**不发 Comfy Registry** | §14 Q-E |
| D11 | **生成器全族（本体 + prompts + schema + critic）进 Core**；MieNodes 只留节点壳 | §7 |
| D12 | **Core 是纯 pip 包**，不是 ComfyUI 自定义节点包 | §3.4 |
| D13 | **`services/llm.py` 拆分**：纯客户端进 Core，26 个节点类留 MieNodes | §7 |
| D14 | **删除 `services/llm.py` 的三个死重依赖**（numpy / cv2 / torch） | §6.1 |
| D15 | **P0 分六步**：配置面 strict + **恢复逻辑按 §8.6.1 的三条规则选源** + size 分层 + 全量清单 + 文件系统识别 + write-once | §8 |
| D16 | **写协议的 CAS 基准用内容哈希**，不用 mtime | §10 |
| **D17** | **`st_ino` 带降级规则**：**按文件系统类型判定**（FAT/exFAT 的 file index 来自目录项位置 ⇒ 原地覆写不变）⇒ 无身份信号 ⇒ 强制 strict | §8.5 |
| **D18** | **分级判据**：UNC 前缀 + `DRIVE_REMOTE` 映射盘 + **文件系统类型** + `ino==0` 运行时信号 + 配置白名单 | §8.5 |
| **D19** | **P0-c 按四类分类排查**：缓存跳过 / 身份等值比较 / 最新者胜 / 年龄清理+报告 | §8.6 |
| **D20** | **第三条腿：对 run 输出目录强制 write-once**，把"不可变"从未强制的不变量变成事实 | §8.7 |
| **D21** | **size 分层**：`<64MB` 永远直接哈希，只有多 GB 文件才用身份缓存 | §8.7 |
| **D22** | **bundle 原子性**：媒体内容寻址 + json 清单最后翻转 | §10.2 |
| **D23** | **冻结节点写路径改走 Core 写入口**；`server.py` 写路由返 405/501 明示去向 | §10.3 |
| **D24** | **FastAPI：`uvicorn workers=1`**，生产关 `--reload`，端口 8199 | §6.1 |
| **D25** | **第七坑：共享队列**——`/queue` 深度探测，检测到外来任务即暂停提交 | §11 |
| **D26** | `asset_auto_describe.py` **留 MieNodes 作壳**，其 `library` import 重指 Core | §7.2 |

### 2.1 与上游的关系（2026-11-06 裁定，一句话）

> **上游 = 溯源参考，不是依赖。**
> `upstream` remote **只用来对照与查证，永不合并**；base commit 固定为 `43afeff`。
> **不向上游回流任何改动**（Q-F 关闭）。我们与上游从此**不再有代码往来**。

由此确定的三件事：

| # | 事项 | 结论 |
|---|---|---|
| 1 | 上游的修复 | **拿不到，也不打算拿**。上游的 Windows 缺陷、ComfyUI 兼容性补丁一律由我们自己写 |
| 2 | `upstream` remote | **保留**，但角色降为**只读溯源**：base commit、对照 diff、"这段代码本来长什么样"的唯一依据。**不合并**。已把它的 **push 地址置为 `DISABLED`**（`git push upstream` 直接失败），杜绝误推 |
| 3 | 风险转移 | 风险从"同步成本"转到**"ComfyUI 内核升级的兼容性要自己扛"**（`comfy.ldm.minimax.model` 等内部 API，见 R8）⇒ Phase 1 的 Windows 修复与"ComfyUI 版本兼容矩阵"检查**从可选变成必做** |

---

## 3. 架构

### 3.1 四层

```
┌─ 产品层：短剧平台（新增，独立 FastAPI 进程，8199）─────────────────┐
│  浏览器前端 + REST 路由（剧本 CRUD / 站内生成 / 预览 / critic /     │
│  渲染队列 / 产物）                                                 │
└───────────────────────────┬───────────────────────────────────────┘
                            │ import
┌─ 引擎层：ComfyUI-Mie-H3-Engine（新增包，fork upstream）──────────┐
│  H3 生成 / 上下文续接 / 音频 / checkpoint / 装配 / 审核             │
│  + 资产/剧本/plan 的节点壳（薄壳，逻辑全在 Core）                   │
│  + Scene Prompt Editor 只读预览通道（§5）                          │
└───────────────────────────┬───────────────────────────────────────┘
                            │ import
┌─ 共享核心：ComfyUI-Mie-Core（纯 pip 包，零 ComfyUI 依赖）─────────┐
│  llm/     纯客户端 + 22 provider 连接器 + 预算                      │
│  script/  剧本契约 + critic 两半 + 生成器本体 + prompt 家族 + schema │
│  assets/  资产库读写 + 剧本资产读写                                 │
└───────────────────────────┬───────────────────────────────────────┘
                            │ import
┌─ MieNodes（现有 306 节点）─────────────────────────────────────────┐
│  节点壳：storyboard generator / loop prompt generator / connector  │
│  资产节点壳 + 资产库前端 UI（冻结）                                 │
│  其它 20+ 提示词生成器与工具节点                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 依赖方向

| 边 | 方向 | 理由 |
|---|---|---|
| 引擎 → Core | 引擎依赖核心 | 资产节点与 plan builder 用 Core 的契约与库读写 |
| MieNodes → Core | MieNodes 依赖核心 | 节点壳调 Core 的生成逻辑与 LLM 客户端 |
| 产品层 → Core | 产品层依赖核心 | 剧本 CRUD、生成、critic、plan 编译全用 Core |
| Core → 任何包 | **无** | Core 不依赖引擎、不依赖 MieNodes、不依赖产品层 |

**契约面的依赖已实测**：`nodes/llm/storyboard_asset_generator.py` 对资产层的 import 是
**3 个模块、5 条 import 行**：

| 行 | 目标模块 |
|---|---|
| `:25` | `..assets.storyboard_contract` |
| `:176` | `..assets.storyboard_review` |
| `:198` | `..assets.storyboard_critic` |
| `:297` | `..assets.storyboard_contract` |
| `:308` | `..assets.storyboard_contract` |

### 3.3 一份实现，多处调用

> 编译器（剧本资产 → `plan_json`）在 **Core**，只有一份。
> 不是"产品层通过 HTTP 取 plan_json"——那只解决取值，解决不了"编译逻辑分叉"。

调用点：产品层编译器、引擎 plan builder、引擎 PromptServer 预览路由、
引擎 `MieH3ScriptToPlan` 节点壳。**四处，一段代码。**

### 3.4 共享核心包 `ComfyUI-Mie-Core`（D5 / D12）

**形态：Core 是纯 pip 包，不是 ComfyUI 自定义节点包。**

> Manager 的安装方式是**复制进 `custom_nodes/`**，目录名带连字符，
> **不会**进 site-packages ⇒ MieNodes / 引擎照样 `import mie_core` 失败。
> 而且 Core 没有任何节点，"可被 Manager 安装"没有收益。

**因此：**

- [ ] Core 仓是**标准 Python 包**（`pyproject.toml` 声明包布局 `mie_core/`），
      **不是** ComfyUI 自定义节点包
- [ ] Core 仓**同时**放在 `custom_nodes/` 目录下（蹭既有的**局域网目录同步**机制）
- [ ] **ComfyUI 的 python 环境里 `pip install -e <Core 目录>`**——本机、局域网各一次
- [ ] 产品层 venv 里 `pip install -e <Core 目录>`——同样各一次
- [ ] 部署脚本把这几步作为**一次性安装动作**

**内容清单见 §7.1。不搬 `core/utils.py`**（`|Mie` 是品牌专属，引擎有自己的 `|MieH3` utils）。

**时序（不双线重构）：**

- Phase 0–3：引擎 = 纯 fork，**不碰资产层** → 无耦合问题
- **Phase 4 开头：一次性抽出 Core**，三仓在同一变更窗口内切换
- Phase 5：产品层 `pip install`

**"同一个 commit"做不到**（跨仓）。改成**同一变更窗口**：
Core 先建并发布（可安装）→ MieNodes 与引擎**各一个 commit** → 部署脚本一次性切换 + 双机各装一次。

---

## 4. 命名与注册（D4）

### 4.1 为什么必须改名——比报错更糟

```python
# ComfyUI/nodes.py:2305-2308
for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
    if name not in ignore:
        NODE_CLASS_MAPPINGS[name] = node_cls        # ← 静默覆盖，无告警
```

`/object_info` 以类型名为键，只有**一份**实现会被暴露。
两个包同装 ⇒ 只有一个生效，**无错误无警告** ⇒ 工作流执行的是另一份实现。

#### 4.1.1 本机实测（2026-11-06，两包同时启用）

```
启动日志：ComfyUI-MiniMaxH3-Context-Loop  导入成功
          ComfyUI-Mie-H3-Engine            导入成功
          重复注册 / 覆盖告警：0 条
/object_info 里 MiniMaxH3* 类型数 = 148（上游单独时也是 148；若叠加会是 296）
```

⇒ **确认为静默覆盖，不是叠加。**

**但比本文原先写的更脆弱。** 日志里的 "Import times for custom nodes" 一节是
`sorted(node_import_times)`，**按耗时排序**，不是导入顺序；
真正的导入顺序是 `os.listdir(custom_nodes)` 的**文件系统目录序**（`nodes.py:2371`）。
实测中 `ComfyUI-Mie-H3-Engine` **先于** `ComfyUI-MiniMaxH3-Context-Loop` 被导入，
于是**上游赢了**——尽管我们的包名字看起来更"新"。

⇒ **两个包同时启用时谁生效，取决于一个完全偶然的文件系统顺序因素，且没有任何提示。**
这比"后加载者胜"更糟：**Phase 2 的改名不是可选项，是必须项**——
在改名完成之前，本机必须只启用其中一个（实测做法：启动器 `plugins disable`）。

### 4.2 命名方案

**实施后补记（2026-11-06）**：后缀在**模块级**施加 —— 每个模块的
`*_NODE_CLASS_MAPPINGS` / `*_NODE_DISPLAY_NAME_MAPPINGS` 各自加后缀，
而不是只在 `__init__.py` 的聚合处加。

原因是聚合处加会留下**两个名字空间**：模块字典的键是旧名，ComfyUI 注册的是新名，
上游自己的测试大量直接查模块字典 ⇒ 23 个测试当场变红。
模块级加则**全仓只有一个名字空间**：模块字典里的键 = ComfyUI 注册的名字
= 工作流/提示词里出现的名字。

沿用 MieNodes 既有约定（`core/utils.py:12` `LOGO_SUFFIX = "|Mie"`）：

```python
# ComfyUI-Mie-H3-Engine/core/utils.py
LOGO_SUFFIX = "|MieH3"
```

全部 87 个上游类型加后缀；显示名同步加后缀并遵循"中文 · English"惯例。
我们的新节点独立命名：`MieH3ScriptToPlan`、`MieH3LoadScriptAsset`、`MieH3LoadAssetRoster`。

**为什么用后缀不用前缀**：保留 `MiniMaxH3` 前缀让 87 个类型的批量替换不易漏；
节点面板是子串搜索，搜 `MieH3` 也能命中。

### 4.3 存量工作流迁移（Q-B：**A**）

`tools/migrate_workflows.py` 要求：处理 **subgraph 嵌套**的 `type`；**幂等**；
支持**按目录批量**（`docs/qa/` 被 gitignore，局域网另有一批副本）；只改 `type`，不碰 `widgets_values`。

### 4.4 改名表必须自动生成

`tools/node_rename_map.json` 由脚本从各模块 `NODE_CLASS_MAPPINGS` **自动生成**，
不允许手写（87 条手写必漏）。生成器与迁移脚本共用同一张表。

**CI 断言**：`len(rename_map) == len(引擎包注册类型数)`，
且 `set(rename_map.values()) == set(引擎包注册类型名)`。

### 4.5 改名范围：Python 之外，**`web/` 必须一起扫**（第四轮补的真缺口）

上游前端**用字符串比对类型名**。只改 Python 的 `NODE_CLASS_MAPPINGS`，
这些扩展**挂不上 `|MieH3` 节点**；两个包同时装时，
它们还会**挂到上游原名节点上**——而那正是我们要避免的静默错配。

实测上游 `web/` 里**硬编码了 47 个类型名，散在 20+ 个文件**：

| 类型名 | 出现次数 |
|---|---|
| `MiniMaxH3ChainPlan` | 14 |
| `MiniMaxH3ChainPlanModern` | 13 |
| `MiniMaxH3ChainPlanStudio` | 11 |
| `MiniMaxH3ChainLoopStart` | 9 |
| `MiniMaxH3ProjectAssetManager` | 6 |
| `MiniMaxH3ProjectAssetTree` | 5 |
| `MiniMaxH3ChainLoopEnd` / `MiniMaxH3ChainCurrent` | 各 5 |
| `MiniMaxH3ChainReview` | 4 |
| `MiniMaxH3CurrentTaggedReferenceScene` | 3 |
| `MiniMaxH3ChainScenePromptEditor` / `MiniMaxH3ChainRichScenePromptEditor` | 各 3 |
| …其余 37 个类型，各 1–2 次 | |

涉及文件（部分）：
`h3_chain_plan_editor.js`、`h3_chain_plan_studio.js`、
`h3_chain_scene_prompt_editor.js`、`h3_chain_rich_scene_prompt_editor.js`、
`h3_chain_checkpoint_manager.js`、`h3_chain_run_manager.js`、
`h3_chain_cancel_reroll.js`、`h3_chain_review_final.js`、
`h3_chain_top_level_requeue.js` + 多个 `.mjs` core。

**改名规则（第四轮明确）：**

- [ ] 用 `tools/node_rename_map.json` 做**全字（whole-word）匹配替换**
- [ ] **不要按 `MiniMaxH3` 前缀扫**——那会误伤
      **节点 id**（如 `MiniMaxH3ContexLoop.cancelRerollControl`）、
      **扩展 id**、**仓库 URL**、**注释**
- [ ] `web/` 与 Python 同批改；改完跑一次"前端还能不能挂上 `|MieH3` 节点"的检查

**实施后补记（2026-11-06）—— 改名有两个工具查不出来的陷阱：**

1. **JS 里的类型索引表用的是裸标识符键**（`MiniMaxH3ChainPlan: [...]`），
   不是引号字面量 ⇒ 只扫引号字面量会漏掉它们，表现为“前端拿不到高级输出名”。
   而**裸键装不下 `|`**（不是合法标识符字符）⇒ 改名时必须**把键加上引号**。
2. **有些地方出现的旧名，语义上就是“历史名”**：退役检测（`removed` 集合）
   与旧版检测（`legacy` 集合）里的名字**必须保持旧名**，改了就把那条检查改坏了。
   ⇒ 改名工具需要一张**例外表**。

**完整的改名范围：**

| 目标 | 处理 |
|---|---|
| **`web/**`**（47 个类型名 / 20+ 文件） | **一并改名**（第四轮补充） |
| Python 源码里的 `NODE_CLASS_MAPPINGS` | 由 `add_suffix()` 统一处理 |
| `tests/**`（golden / portability / POV 等引用旧名的测试） | 一并改名 |
| `tests/_node_contracts_baseline.json` | 随改名更新，并随 D9 弃用标记更新 |
| `docs/examples/**`、`workflows/**`（本仓） | 一并改名 |
| 上游 `example_workflows/**`（22 个 JSON） | 一并改名 |

---

## 5. Scene Prompts 预览通道（D3）

### 5.1 现状：Editor 读 widget 不读 socket

```javascript
// web/h3_chain_rich_scene_prompt_editor.js:2383-2395
const planWidget = planNode?.widgets?.find((item) => item.name === "plan_json");
const value = String(planWidget.value ?? "");     // ← 只读 widget
```

```javascript
// 同文件 :744 与 :804（写回）
state.planWidget.value = pending.value;                // ← 只写 widget
```

上游文档还教用户把我们的接法拆掉：
`docs/PROJECT_ASSETS.md:39` — *"External `plan_json_input` must be disconnected
because it overrides the editable Plan."*

### 5.2 硬约束：前端读不到 socket 的运行时值

- `execution.py:583` 在执行循环里给**每个**节点发 `executed` 消息 → 前端有全部节点输出；
- 但 `execution.py:601` 的 `ui_outputs`（即 `/history` 的 `outputs`）**只收 `OUTPUT_NODE`**。

所以"事后缓存"救不了预览：它要求**先跑一次**，而预览的意义就是跑之前。

### 5.3 方案：PromptServer 编译路由 + 前端 provider

**编译留在 Python，前端只 fetch**（否则"一份实现"会在前端破功）。

```
POST /mieh3/plan_preview
  请求：{"script_ref": "<剧本资产 id 或资产库根目录下的相对路径>"}
  响应：{"plan_json": "<字符串>", "plan_sha256": "<前 8 位>",
         "source_mtime_ns": 123, "source_etag": "..."}
```

两条硬要求：

1. **响应带 `plan_sha256` 前 8 位**，编辑器横幅显示"当前预览 = xxxx1234"，
   与产品侧一望即知是否同版。
2. **`script_ref` 必须限定在资产库根目录下**——否则 LAN 上任何人都能让 ComfyUI
   进程读任意路径（**路径穿越**）。实现：只接受资产 id 或根目录下相对路径；
   `os.path.realpath` 后必须仍在根目录内（`commonpath` 检查）；拒绝符号链接逃逸。

**前端 provider 约定：**

```javascript
node.mieH3PlanProvider = () => ({
    fetchPlan: () => fetch("/mieh3/plan_preview", {method: "POST", body: ...})
});
```

引擎包前端逻辑：socket 已连接 → 找生产节点 → 有 provider 就 fetch；
否则回落 widget 并显示**明确横幅**。

### 5.4 Plan Studio 的处置

保留（对纯 widget 驱动的工作流仍有价值），但 socket 模式下**不作编辑入口**，
顶部加同样横幅，明确"外部 plan 接管，本面板只读"。

### 5.5 新鲜度

响应带 `source_mtime_ns` / `source_etag` / `plan_sha256`；
前端记录，**每次打开编辑器必校验**（不静默显示旧内容）；
源变更 → 显示"预览已过期，点击刷新"；提供刷新按钮；不做轮询。

### 5.6 验收

- [ ] 接上 `plan_json_input` 后，Scene Prompt Editor 显示的是**我们的**分镜
- [ ] Editor 内编辑不改变执行结果，界面明示只读
- [ ] 产品层改剧本资产后，ComfyUI 侧出现"预览已过期"提示，刷新后显示新内容
- [ ] 横幅显示 `plan_sha256` 前 8 位，与产品侧一致
- [ ] `script_ref` 越权路径被拒绝（**安全测试**）
- [ ] 未接外部输入时，Plan Studio 行为与上游完全一致（回归）

---

## 6. 产品层：短剧平台（D2 / D7 / D8 / D14 / D24）

### 6.1 进程模型、导入机制与部署拓扑

- 独立 FastAPI 进程（`uvicorn`），**不挂在 ComfyUI 的 PromptServer 上**（D2）。
- **端口**：ComfyUI 继续占 **8188** 不动；FastAPI 挑空闲口 **8199**，绑 `0.0.0.0` 供局域网浏览器用。
  **无冲突可言。**
- **`uvicorn workers=1`**（D24）——多 worker 会破坏渲染队列串行化（坑 4 的前提）
  与内存态任务表；**生产模式关 `--reload`**（否则又是双进程陷阱）。
  ⚠️ 但 **`workers=1` 只保证单进程，异步请求仍会交错** ⇒
  **产品层自己的队列要另加一把进程内锁**（R21/R22），不能依赖"单 worker 所以串行"的假设。
- **导入机制**：产品层 venv `pip install -e <Core 目录>`，`import mie_core`。
  不做 sys.path 注入，不要求产品层机器 clone 引擎仓。
- **依赖体量（D14）**：`services/llm.py` 顶层 `import numpy, cv2, torch`
  ——**实测三个都是死 import**（AST 审计 + 原始 grep 双重确认：三个名字在全文
  **各只出现 1 次**，即只有 import 行本身）。⇒ **直接删除**，不需要惰性导入技巧。
  ⇒ Core 的 LLM 模块依赖只剩 **`requests` + stdlib**，
  产品层 venv 不再需要 torch / numpy / cv2 这几个数 GB 级依赖。
- **部署拓扑（D8）：v1 同机。** 角色立绘与音色样本必须落到 ComfyUI 的 `input/`，
  而 `/upload/image` 名义上只收图片、不收音频 ⇒ 同机直接写盘最省事。
  **跨机是 v2 的问题**。
- **⚠️ 同步 `requests` 会卡住事件循环（第四轮补）**：Starlette 把普通 `def` 路由
  丢进线程池，但把 `async def` **直接在事件循环里跑**。
  若生成与 critic 写成 `async def` 再调同步的 `mie_core` / `requests`，
  **几十秒内预览、任务轮询、空闲标志都会停**。
  ⇒ **这两条路由用同步 `def`**，或在 `async def` 里用 `anyio.to_thread.run_sync` 包起来。
- **锁的边界（第四轮补）**：产品层那把锁**只包住"看 `/queue` + 提交"那一小段**。
  生成路由若在线程池里跑 ⇒ 用 `threading.Lock`，
  **不要在等 LLM 的几十秒里握着它**。
  这把锁**仍只串行本进程**；人手在 8188 上的提交**不进这把锁**，
  两次 HTTP 之间的空窗仍在（见 §11 坑 7）。

### 6.2 路由清单（首版）

| 方法 | 路径 | 作用 |
|---|---|---|
| GET/POST | `/api/projects` | 剧集/项目 CRUD |
| GET/POST | `/api/projects/{id}/scripts` | 剧本资产 CRUD（source of truth） |
| **POST** | **`/api/scripts/{id}/generate`** | **站内剧本生成（concept/revise）**（D7） |
| GET | `/api/scripts/{id}/preview` | 分镜预览数据 |
| GET | `/api/scripts/{id}/plan` | 编译出的 `plan_json`（调试用） |
| POST | `/api/scripts/{id}/critic` | 连续性 critic（**两半**） |
| GET/POST | `/api/assets` | 角色/场景资产 |
| POST | `/api/render/submit` | 提交渲染 |
| GET | `/api/render/{job}` | 轮询任务状态 |
| GET | `/api/render/{job}/artifacts` | 产物 |
| GET | `/app` | 前端 SPA |

**站内生成为什么必须保留（D7）**：既有 concept/revise 模式 + 报错教学
就是为了让人**填得动**；砍掉只留结构化表单是**体验退步**。
**它的可行性依赖 D11**——生成器本体必须是 Core 里的纯 Python。

### 6.3 数据模型

```
Project（剧集）
  ├─ scripts: ScriptAsset[]        # 剧本资产 = source of truth
  │    ├─ roles[]                  # id / name / presence / position / voice
  │    ├─ camera{}                 # framing / movement / perspective
  │    ├─ execution{}              # look / setting / scene / light / visible_state
  │    ├─ shots[]                  # 每镜 events[]（action/dialogue/enter/exit）
  │    └─ assets[]                 # 接线的角色/场景资产 id
  ├─ assets: CharacterAsset[] / SceneAsset[]
  └─ renders: RenderJob[]
```

**复用而非重写**：结构沿用 Core 的 `core/script/contract.py`（v2 契约，707 行）与 `to_runtime()`。

### 6.4 前端（首版）

1. **剧本**：结构化表单 + 站内生成（concept/revise），右侧实时显示编译后的每镜 prompt；
2. **分镜预览**：逐镜卡片；
3. **渲染**：提交、进度、产物回放。

**首版不做**：视频剪辑器、时间线、多轨混音、字幕编辑器。

**资产 UI 的种子（第三轮发现）**：MieNodes 已有 `js/assetLibraryUI.js`
（**1,551 行 / 74KB**，调用 6 个 `/mie_nodes/assets/*` 端点）——
一个**能用的资产浏览器**。产品层的资产页应以它为起点移植，而不是从零写。

---

## 7. 资产层与剧本层的完整归属

### 7.1 进 Core

| 来源 | LOC | 内容 |
|---|---|---|
| `services/llm.py`（**拆分**，D13） | 2,655 | 50 个类 → **24 个纯客户端/连接器**进 Core，**26 个带 `INPUT_TYPES` 的节点类**留 MieNodes |
| `nodes/llm/invoke_budget.py` | — | 调用预算 |
| `nodes/assets/storyboard_contract.py` | 707 | v2 契约 + `to_runtime()` |
| `nodes/assets/storyboard_critic.py` | 484 | critic **确定性半** |
| `nodes/llm/storyboard_critic.py` | 197 | critic **模型半** |
| `nodes/assets/storyboard_review.py` | 41 | 资产独立审阅 |
| **`nodes/llm/storyboard_asset_generator.py`** | 343 | **生成器本体**（D11） |
| **`nodes/llm/minimax_h3_storyboard_prompts.py`** | 791 | **生成器 prompt/白名单**（D11） |
| **`nodes/llm/prompts/h3_storyboard/*`**（7 个文件，含 `asset.schema.json` 与 `UPSTREAM.md`） | — | **prompt 家族**（D11） |
| `nodes/assets/library.py` | 1,284 | 资产库（纯 stdlib） |
| `nodes/assets/storyboard.py` | 1,224 | 剧本资产读写 |

**不搬 `core/utils.py`**（`|Mie` 是品牌专属）。

### 7.2 留 MieNodes（全是**壳**）

| 来源 | LOC | 内容 |
|---|---|---|
| `nodes/llm/minimax_h3_storyboard_generator.py` | 408 | **生成器节点壳** |
| `nodes/llm/minimax_h3_loop_prompt_generator.py` | 5,503 | loop 生成器（**LLM 连接器是注入的**，不是 import） |
| **`nodes/llm/asset_auto_describe.py`** | **352** | **Auto 节点壳**（D26）——它包装的是 Save 节点 |
| `services/llm.py` 的 26 个节点类 | — | `Set*LLMServiceConnector` 等 |
| `nodes/assets/nodes.py` | 1,151 | 资产节点壳 |
| `nodes/assets/server.py` | 277 | aiohttp 路由（写路由按 D23 返 405/501） |
| `js/assetLibraryUI.js` 等 13 个前端文件 | — | 资产库 UI（冻结；作为产品层资产页的种子） |
| `nodes/llm/prompts/h3_loop/*` | — | H3 loop prompt 家族 |

**为什么生成器本体必须搬（D11）：**

1. `nodes/llm/__init__.py` 做 `from .xxx import *`（实测 20+ 行）⇒
   **import 任何一个 `nodes.llm.*` 模块都会拖起整个包**，其中包含需要 ComfyUI
   运行时的模块（`minimax_h3_loop_prompt_generator.py:115-118` 的 shim 注释明说了这点，
   它正是因此才按**文件路径**加载 prompts/critic）⇒ **产品层无法干净 import MieNodes 的包**。
2. 生成器**本体**已经是纯 Python（import 只有 `..assets.*` 契约/审阅/critic
   + **注入的 llm 服务**），**408 行的节点壳**才是 ComfyUI 侧——**本体/壳早已分层**。
3. **critic 模型半的口粮在同一个目录家族里**：
   `nodes/llm/storyboard_critic.py:29`
   `PROMPT_DIR = Path(__file__).parent / "prompts" / "h3_storyboard"`
   ⇒ critic prompts、生成器 prompts、`asset.schema.json` 是**同一个目录的家族**。
   **代码走、口粮不留 ⇒ 搬过去读不到指令。**

**`asset_auto_describe.py` 的归属（D26）**：它自己的注释写着
*"The Auto nodes ARE the Save nodes plus one `llm_service_connector` input"*，
import 面横跨三方：

| import | 目标 | 去向 |
|---|---|---|
| `..assets.nodes`（`:37`/`:47`） | Save 节点壳 | 留 MieNodes |
| `..assets.library`（`:45`/`:55`，取 `DESCRIPTION_MAX_CHARS`） | 资产库 | **→ Core，import 必须重指** |
| `nodes/llm/h3_prompts.caption_reference_prompt`（`:32`/`:34`） | H3 prompt | 留 MieNodes |
| `core.utils.image_tensor_batch_to_data_urls`（`:27`/`:29`） | 通用工具 | 留 MieNodes（不搬 `core/utils.py`） |

⇒ **留 MieNodes 作壳**，但 Phase 4 必须加一行"其 `library` import 重指 Core"，否则编译期就炸。

### 7.3 critic 的两半（口径统一）

| 半 | 文件 | 需要 LLM？ | 去向 |
|---|---|---|---|
| 确定性半 | `nodes/assets/storyboard_critic.py` (484) | 否 | **Core** |
| 模型半 | `nodes/llm/storyboard_critic.py` (197) | 是（经 Core 的 LLM 抽象） | **Core** |

**两半同住 Core**（D11 消解了原 Q-H）。
`/api/scripts/{id}/critic` 路由需要**同时**用两半。

### 7.4 搬进引擎包的 ComfyUI 外壳（薄）

| 新节点 | 作用 |
|---|---|
| `MieH3LoadAssetRoster` | 从 Core 资产库读角色/场景 → roster |
| `MieH3LoadScriptAsset` | 从 Core 读剧本资产 |
| `MieH3ScriptToPlan` | Core 的零 LLM 编译器 → `plan_json`（**并实现 §5.3 的 provider**） |
| `MieH3SaveAsset` / `MieH3SaveScriptAsset` | 写回 Core |

### 7.5 与上游 `project_assets.py` 的边界

上游管"这次渲染用了哪些媒体文件"（run 作用域媒体卡片），
Core 管"这个角色长什么样、声音什么样、在哪几镜出现"（跨项目具名实体）。
**正交，不合并。** 产品层负责把 Core 的角色/场景翻译成上游 Carousel 的引用关系。

---

## 8. P0：缓存与 mtime 真相

> 这是整个计划里**唯一**"看不见的错片"来源。Phase 1 未完成不得进 Phase 3。

### 8.1 守卫本体

```python
# 上游 chain_nodes.py:24681-24713
def _verify_png_export_checkpoint(
        checkpoint: str, expected_hash: str, verification: str,
        cache: dict[str, Any]) -> tuple[str, bool]:
    mode = str(verification or "cached").strip().lower()
    if mode not in ("cached", "strict"):
        raise ValueError(...)
    expected = str(expected_hash or "").strip().lower()
    ...
    if mode == "cached" and isinstance(entry, dict):
        cache_matches = (
            str(entry.get("sha256") or "").lower() == expected      # ① 上次实测 vs 调用方期望
            and int(entry.get("size", -1)) == int(stat.st_size)     # ②
            and int(entry.get("mtime_ns", -1)) == int(stat.st_mtime_ns))  # ③
    if cache_matches:
        return "cached SHA-256", False          # ← 直接返回，不重算文件哈希
    actual = _file_sha256(checkpoint)          # ← 真正的完整性校验在这里
    if actual.lower() != expected:
        raise ValueError("H3 PNG export checkpoint failed its SHA-256 integrity check: %s")
    cache.setdefault("entries", {})[key] = {
        "sha256": actual.lower(),              # ← 写回【实测值】
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
```

**`expected_hash` 来自调用方**（实测三处调用全部是
`segment.get("checkpoint_sha256", "")`，`chain_nodes.py:25395 / 25586 / 25614`），
来自装配记录；而 `entry["sha256"]` 是**上次真正算过的文件哈希**。

⇒ 条件 ① 是 **"上次实测 == 调用方期望"**，**不是同源恒真**。
**如果按"删掉冗余条件"去修，会什么都没修好。**

### 8.2 真正的门

> 一旦某个 `(size, mtime_ns)` 组合被实测通过过，它就成了**该路径的内容身份**。
> 此后内容变了而 size/mtime 不变 ⇒ **永不复检** ⇒ 静默用错内容渲染。

### 8.3 上游信任的是一个从未被强制的不变量

widget 的 tooltip 自己写着：

> *"Cached verifies each **immutable** checkpoint with SHA-256 once, then trusts
> matching size and modification time on later exports. Strict re-hashes every
> checkpoint on every export."*

上游信任"checkpoint 不可变"——而这个不变量**从未被强制**。

### 8.4 修法一：配置面（今天就能做）

`checkpoint_verification` **已经是 widget**（`chain_nodes.py:25166`，默认 `"cached"`），
**strict 模式永不走缓存**（`mode == "cached"` 才查 entry）。
⇒ **关键路径可以立刻切 strict**，不需要改一行代码。

### 8.5 修法二：`st_ino` 入 cache key —— **带降级规则**（D17）

**加 `st_ino` / `st_dev`（NTFS file index）到 cache key**，因为 `os.replace`
原子替换后 file ID 必变，比 mtime 硬。

**但 file index 的可靠性是分层的，不能无脑入 key：**

| 介质 | `st_ino` 行为 | 结论 |
|---|---|---|
| **NTFS 本地** | 好（MS 文档只保证句柄打开期间唯一，defrag/复用可变；实践中稳） | 加分项，不是保证项 |
| **FAT / exFAT** | file index 由**目录项位置**派生 ⇒ **原地覆写不变** | **恰好防不住我们要防的场景**（SD 卡 / U 盘上的缓存目录直接失效） |
| **UNC / SMB** | 重定向器可能返回 0 或服务器相关值 | **不可信** |

**降级规则（D17，自校准）：拿不到可靠 ino 就不放行。**

> ⚠️ **第三轮纠正**：上一版的 `identity_signal` **没有执行它自己给出的 FAT 结论**。
> 表里写明 FAT/exFAT 的 file index 来自目录项位置、原地覆写不变，
> 而函数只在"远程路径"或"`ino == 0`"时拒绝——
> **本地 exFAT（U 盘 / SD 卡）给出的是非 0 的目录项编号，函数会把它当成身份放行。**
> ⇒ 必须**认出文件系统**，不能只看 ino 是否为 0。

```python
FAT_TYPES = {"FAT", "FAT32", "exFAT"}

def volume_fs_type(path):
    """Windows: 返回卷的文件系统类型；非 Windows 返回 None。"""
    import ctypes, os
    root = os.path.splitdrive(os.path.abspath(path))[0] + os.sep
    if not root:
        return None
    name = ctypes.create_unicode_buffer(64)
    fs = ctypes.create_unicode_buffer(64)
    serial = ctypes.c_uint(); maxc = ctypes.c_uint(); flags = ctypes.c_uint()
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(root), name, 64, ctypes.byref(serial),
        ctypes.byref(maxc), ctypes.byref(flags), fs, 64)
    return fs.value if ok else None

def identity_signal(path, st):
    """返回可用于内容身份判断的信号；不可信时返回 None。"""
    if is_remote_path(path):        # UNC 前缀 或 DRIVE_REMOTE 映射盘
        return None
    if volume_fs_type(path) in FAT_TYPES:
        return None                 # file index 由目录项位置派生 ⇒ 原地覆写不变
    if not st.st_ino:               # 重定向器返回 0
        return None
    return (st.st_dev, st.st_ino)
```

（本机已验证 `GetVolumeInformationW` 可用，`C:\` 返回 `NTFS`。）

**分级判据（D18），四条并列：**

1. UNC 前缀（`\\` 开头）；
2. **映射盘符**（`Z:\` 映射到共享不带 `\\`）⇒ 用 `GetDriveTypeW == DRIVE_REMOTE` 判定；
3. **文件系统类型** ∈ {FAT, FAT32, exFAT} ⇒ 无身份信号（D17）；
4. **运行时信号** `st_ino == 0` 兜底；
5. 配置白名单覆盖（用户可手动把某目录标成"强制 strict"）。

**分级策略**：可信的本地 NTFS = `ino + size + mtime`；其余 = **强制 strict**。

**探针要测"危险的那条路"（第三轮纠正）：**

上一版的探针是"`os.replace` 之后 `st_ino` 变了"——**这测的是已经能防住的那条路径**
（`os.replace` 换新的 MFT 记录 ⇒ ino 必变）。

**真正危险的是原地覆写**：robocopy / 镜像同步是原地覆写，
`ino`、size、mtime **都可以不变**。
⇒ 探针必须是：**原地覆写一个同 size、保留 mtime 的文件，
断言身份信号确实检测不到变化**（即：这种场景必须走 strict，不能靠 ino）。

**并且要注意这条缓存管的是 PNG（`:24681`）**：
配合 D21（`<64MB` 一律直接哈希），**这段缓存基本整段落到哈希上，轮不到 ino**。
⇒ **P0 的工时应该花在 P0-a（关键路径切 strict）与 §8.6.1 的恢复逻辑上，
而不是花在做 ino 这一层。**

### 8.6 修法三：全量清单，按**四类**分类排查（D19）

**完整清单（实测 39 处 / 13 个生产文件）：**

| 文件 | 处数 |
|---|---|
| `chain_nodes.py` | 16 |
| **`storage_inventory.py`** | **7**（全包最多） |
| `run_manager.py` | 3 |
| `checkpoint_manager.py` | 3 |
| `chain_layout_conversion.py` | 2 |
| `working_branch_cleanup.py` | 1 |
| `selflift_runtime/alternative_lifts.py` | 1 |
| `processing_checkpoint_delete.py` | 1 |
| `png_video_export.py` | 1 |
| `png_export_transaction.py` | 1 |
| `handoff_state.py` | 1 |
| `chapter_snapshot_retirement.py` | 1 |
| `assembly_checkpoint_cleanup.py` | 1 |
| **合计** | **39** |

**排查按四类分类（D19）：**

| 类 | 危险？ | 说明 |
|---|---|---|
| **① 缓存跳过** | **危险** | 内容变了但 size/mtime 未变 ⇒ 跳过哈希（§8.1 本体） |
| **② 身份等值比较** | **危险** | 用元组判"变了没有"，元组漏项 ⇒ 漏检 |
| ③ 最新者胜 | **危险（另一条线）** | 见 §8.6.1 |
| ④ 年龄清理 + 报告 | 基本良性 | `checkpoint_manager.py` 的 3 处全在记录/格式化时间戳 |

**已知归类结果：**

- `checkpoint_manager.py`（3 处）→ ④ 良性；
- `chain_layout_conversion.py`（2 处）→ 待归 ②/④；
- `storage_inventory.py`（7 处）→ **②，但第三轮纠正了我上一版的说法**：
  - `:212-213`、`:371-372` 用 `(st_dev, st_ino, st_size, st_mtime_ns)` 四元组
    做 before/after 检测——但**上游自己在 `:367-368` 写着**：
    *"This is not an atomic snapshot and **does not detect same-stat byte edits**."*
    ⇒ 它是**扫描过程中的竞态检测**，**不是内容身份机制**。
    **这条注释恰恰加强了我们的论据**：连上游自己都承认这个四元组抓不到
    "stat 不变但字节变了"——而那正是我们要防的场景。
    （与上游惯用法同形，价值在**读懂上游这段代码**，不在"修法同构"。）
  - `:380` 是**目录那一半**，不是文件四元组的弱化版 ⇒ **不必对齐成四元组**。

#### 8.6.1 "较新的 mtime 变成正式文件"——真正的洞在这里（第三轮纠正，第四轮细化）

上一版把 **`run_manager.py:534`** 当成危险线。**降级：它只是显示层。**

```python
# 上游 run_manager.py:534 → 结果进 modified_at
newest = max(modified_paths, key=os.path.getmtime)
# 上游 run_manager.py:558
runs.sort(key=lambda item: item["modified_at"], reverse=True)
```

⇒ 它会把**错误的 run 排到列表最前**，**不改 checkpoint 的字节**。
是排序 bug，不是渲染正确性 bug。

**真正把"较新的 mtime"写成正式文件的是这三处。但三处的修法各不相同**
（第四轮纠正：**哈希只判等值，不能当排序依据；没有"父级期望哈希"就不猜**）：

---

**① `chain_nodes.py:28453-28458` —— 恢复路径**

关键事实：**`:28395-28421` 的 journal 路径已经是安全的**——
它按 journal 里记录的 `temporary` 文件名做 `os.replace`，
journal 损坏时走 `else` 分支**直接删掉临时文件、不做猜测**。
上游自己的注释写明：*"Builds **predating the journal** can leave one renamed pointer behind…"*
⇒ **mtime 只用于 journal 出现之前的遗留兜底。**

- ⚠️ **实施时纠正**：文件名里那 32 位 hex **不是内容摘要**，而是
  `uuid.uuid4().hex`（**随机事务号**，`chain_nodes.py:28650`）。
  ⇒ 盘上**没有任何信号**能指出哪一份才是对的，
  **"按内容选源"在这一处不可得**，只能拒绝猜测；
- 没有 journal 又有多份临时文件时 ⇒ **按 journal 损坏的同一条路拒绝自动恢复**。

**修法（已落地）**：遗留兜底里，若同一 canonical 指针对应**多份**临时文件 ⇒
**拒绝猜测（不按 mtime 挑）**，报错并**保留全部文件**
（不提升、不删除——上游注释承诺"no media is deleted here"）；
**恰好一份** ⇒ 用它（行为不变）。
**不要加 `_mie_seq`**——那会碰到现成的身份摘要。

---

**② `chain_nodes.py:22055` —— 已封印 Chapter 清单**

关键事实（实测 `:21818-21832` / `:21868-21878`）：

```python
def _chapter_manifest_identity(manifest):
    identity = _json_document(manifest) or {}
    for key in ("sealed_at", "chapter_manifest_id", "chapter_manifest_path"):
        identity.pop(key, None)      # ← 这三个字段被摘出摘要
    return identity

def _chapter_manifest_digest(manifest):
    return hashlib.sha256(_canonical_json(
        _chapter_manifest_identity(manifest)).encode("utf-8")).hexdigest()[:32]
```

- `sealed_at` 是**写在文件内容里的时间**（`datetime.now(timezone.utc)`），
  robocopy `/COPYALL` 会**连它一起拷走**，**不受文件系统 mtime 影响**；
- 摘要**排除了** `sealed_at` / `chapter_manifest_id` / `chapter_manifest_path`；
- 调用方传入 `snapshot_id` 时，**文件名就是这份摘要** ⇒ 按 id 取；
- id 为空、要"当前封印"时 ⇒ 比**文件里的 `sealed_at`**。

**修法**：调用方给了 `snapshot_id` ⇒ **按 id 取**；
没给 ⇒ **比文件内的 `sealed_at`**。
**绝不要往清单正文写 `_mie_seq`**——那会改变摘要和文件名，等于**另造一份快照**。

---

**③ `chain_nodes.py:4940` —— reference cache 候选**

关键事实：`_find_reference_cache(fingerprint, scene, scene_count, prompt, width, height, length)`
—— **调用方没有文件哈希**，函数做的是**语义查找**；
候选已按契约（fingerprint/scene/prompt/尺寸）过滤过。

> ⚠️ **实施时纠正**：上游**已经**有歧义守卫
> （不同 `reference_fingerprint` ⇒ 抛
> *"different semantic-anchor settings"*），而且**上游自己的测试**
> `tests/_reference_cache_fingerprint_unit_test.py` 就在守它
> （断言语是 *"ambiguous semantic recipes must not be chosen by mtime"*）。
> 该测试在本机因 `comfy_aimdo` 缺失跑不到那一步，但用 ComfyUI 自带 python 跑**通过**。
> ⇒ **上游这条守卫是对的，必须原样保留**；
> 也不该把"不同 signature 就拒绝"套上去——上游测试明确要求
> *"An exact fingerprint match remains preferred, including legacy V1 caches"*，
> 同一 fingerprint 下多个对象（格式迁移）是**合法状态**，
> 上游的"最后写入者胜"在那个场景里是有意义的。

**修法（已落地）**：**上游守卫原样保留**；只补一条**内容完整性校验**——
缓存文件名带 `signature[:24]`，描述符内容里的 `signature` 必须与它相符，
不符则**跳过该候选**。这样"把 A 的内容换成 B 变体的内容"这种原地覆写
会让描述符自相矛盾而被识破，而 mtime 不再有机会决定谁被选中。
`expected_sha256` 对上号，**只适用于调用方事先记下了目标哈希的路径**
（例如 PNG 导出守卫 `:24681`）——**那三处不适用**。

---

**贯穿三条的通则：**

> **哈希只判等值，不作排序依据。没有"父级期望哈希"就不猜——拒绝自动恢复，
> 走人工。** 绝不用新增序号字段（`_mie_seq` 之类）去"造一个可排序的键"：
> 那会改变摘要与文件名，等于制造另一份快照。

⇒ **D20（write-once）会与 `:28453` 的恢复路径撞车**：
恢复是 write-once 的例外，而**例外本身就是这个洞**；
**write-once 只约束走我们写入口的进程，挡不住外部覆写**（robocopy、镜像同步）。
⇒ D20 必须写成：**write-once + 恢复路径显式豁免 + 恢复时按上述三条规则选源**。

#### 8.6.2 实测：56 个失败里有环境成分，"全绿"要先解决解释器

套件**用哪个 python 跑**，结果差很多：

| 解释器 | 结果 |
|---|---|
| 系统 python 3.12.10 + `PYTHONPATH=<repo>` | **192 / 56**（本文所有基线都是这个） |
| ComfyUI 自带 python 3.13.12（隔离模式） | **80 / 168** |

- ComfyUI 自带的 python 是**隔离模式**（`python313._pth` 固定 sys.path：
  不加脚本目录、**忽略 `PYTHONPATH`**）⇒ 上游测试互相 import 的 fixture 全找不到；
- 系统 python 缺 `comfy_aimdo`（装在 ComfyUI 的 3.13 site-packages，主体纯 Python）⇒ 30 处失败行提到它。

⇒ **"56 个失败"是环境产物，不是代码真相。**
已验证可行的组合：**ComfyUI 自带 python + 一个把 `tests/` 插进 `sys.path` 的引导脚本**
⇒ `_reference_cache_fingerprint_unit_test.py` 在该组合下**通过**。
这是 Phase 1 的**环境子任务**，与 P0 的代码修复是两条线。

### 8.7 修法四：把"不可变"从未强制的不变量变成事实（D20）+ size 分层（D21）

**第三条腿：对 run 输出目录强制 write-once**
——存在即拒绝或改名，把"checkpoint 不可变"变成被强制的事实。

> ⚠️ **第三轮纠正**：write-once **会与 `chain_nodes.py:28458` 的恢复路径撞车**
> （恢复时把 `.restore.<hash>.tmp` 的 mtime 最大者 `os.replace` 成正式 checkpoint）。
> **恢复是 write-once 的例外，而例外本身就是这个洞**；
> 而且 **write-once 只约束走我们写入口的进程，挡不住外部覆写**（robocopy、镜像同步）。
> ⇒ D20 的正确形态是三条一起：
> **① write-once ② 恢复路径显式豁免 ③ 恢复时按内容而非 mtime 选源。**

**size 分层（D21）**：PNG 段 checkpoint 多半是 MB 级，
**`<64MB` 永远直接哈希**（毫秒级，对分钟级渲染可忽略），
**只有多 GB 文件才需要身份缓存**。
⇒ 做完这一条，P0 的暴露面**缩到只剩大文件**。
（也正因如此，`:24681` 那段 PNG 导出缓存基本整段落到直接哈希上——
**ino 这一层在那条路径上根本用不上**，见 §8.5 末尾。）

---

## 9. 实施阶段

> **从哪里切（第四轮明确）**：
> **Phase 0 → 然后只落地 P0-a**（关键路径 `checkpoint_verification` 改 `strict`）。
> 这是**当天能做的生产修复**，**不等 Q-F′**，也**不动那三处恢复逻辑**。
> **P0-b 要等 §8.6.1 的三条选源规则定稿后再动。**
> **Phase 4 的 `services/llm.py` 拆分不修静默错渲染**——那是资产层重组，与 P0 无关。
> **改名脚本可以在 fork 之后做，但第一次扫描就必须包含 `web/`（§4.5）。**

### 进度（2026-11-06）

| 阶段 | 状态 | 证据 |
|---|---|---|
| Phase 0 建 fork | ✅ **已完成** | 分支直接建立在 `43afeff` 之上（上游 954 个 commit 原样保留），我们的提交叠在上面；`upstream` 为永久第二 remote；`LICENSE` / `THIRD_PARTY_NOTICES.md` / `docs/FEATURE_TRACEABILITY.md` 逐字未改 |
| Phase 0 / P0-a | ✅ **已完成** | 三处默认切 `strict`（widget / `export()` 签名 / `or` 兜底）+ tooltip 重写；本机 ComfyUI 实测 `/object_info` 的默认值已是 `strict` |
| Phase 0 / 缓存契约修复 | ✅ **已完成** | 版本号 `0.7.3+mie.1` 触发上游 `_web_cache_bust_unit_test.py` 失败（`Invalid web cache token`）⇒ 放宽其正则接受 `+local`，**排序语义不变**（换回 `0.7.3` 同样通过） |
| Phase 1 / 回归测试 | ✅ **已新增** | `tests/_mie_png_verification_strict_unit_test.py`：证明 cached 看不见"字节变、size/mtime 不变"、strict 能拦住、默认是 strict；**负对照已做**（改回 cached → 测试报红） |
| 测试套件 | **193 passed / 56 failed** | 改动前基线 **189 / 58** ⇒ **零新增失败**；56 个与上游克隆同机失败数一致，全是既有失败（Windows 环境类 + Node 24 harness 类）。**注意 56 这个数字有环境成分，见 §8.6.2** |
| 节点名冲突 | **已实测** | 两包同装：148 而非 296、**零告警** ⇒ 静默覆盖确认。**且胜者由 `os.listdir` 目录序决定**（实测上游赢了）⇒ **Phase 2 的改名是必须项**，见 §4.1.1 |
| Phase 1 / P0-b | ✅ **已做**（三处各按自己的规则改） | ① 遗留恢复：多份 ⇒ **拒绝猜测并保留全部文件**（单份行为不变）；② 封印清单：改按**内容里的 `sealed_at`** 排序；③ reference cache：**描述符的 signature 必须与文件名一致**，上游已测的歧义守卫**原样保留**。<br>**逐项负对照**：上游原实现三条**全红**，我们的实现三条**全绿**（`tests/_mie_mtime_selection_unit_test.py`） |
| Phase 1 / P0-c…P0-f | ✅ **已做** | **P0-c** size 分层（<64MB 一律直接哈希）；**P0-d** 36 处 / 13 文件四类排查（①缓存跳过 2、②身份等值比较 13、③最新者胜 6、④年龄清理+报告 15）—— 真正危险的两类共 8 处**全部处理完毕**；**P0-e** `checkpoint_identity.py`（`GetVolumeInformationW` 识别 FAT/exFAT、映射盘、ino=0 ⇒ 信号不可信就不用）；**P0-f** 已发布修订产物 write-once + 恢复路径显式豁免。**排查中另发现两处此前未列入清单的**：第二个缓存守卫（导出 PNG/WAV，`chain_nodes.py:24923`）与第五处"最新者胜"（`run_manager.py:447`） |
| Phase 2 改名 | ✅ **已做** | 86 个类型加 `\|MieH3`；模块级施加（全仓一个名字空间）；新增 `fork_identity.py`、`tools/gen_rename_map.py`（自动生成 + `--check`）、`tools/apply_rename.py`、`tools/migrate_workflows.py`；**本机实测：两包同装，148（上游）+ 86（我们）= 234，零冲突** ✓ |
| Q-F′ | ✅ **已裁定** | 本仓未来 **public** ⇒ CI 上云。落地：§10.3 的基线闸门（`ubuntu-latest` + `windows-latest`）。**发布到 Comfy Registry 仍暂缓**（Q-E） |

### Phase 0 — 建 fork（0.5 天）

- [x] `git clone` 上游 → 新仓 `ComfyUI-Mie-H3-Engine`，base commit `43afeff`
      —— **实做**：分支直接建立在 `43afeff` 之上（上游 954 个 commit 原样保留）
- [x] 保留 `upstream` remote 指向上游原仓，**角色是只读溯源参考**（D6：永不合并）；
      并把它的 **push 地址置为 `DISABLED`**，杜绝误推
- [x] `LICENSE`（GPL-3.0）+ `THIRD_PARTY_NOTICES.md` + `docs/FEATURE_TRACEABILITY.md` 全部保留
      —— **已用 `git diff upstream/main` 逐项验证为逐字未改**
- [x] 新增 `MODIFICATIONS.md`：**逐条台账**（改了什么/为什么/影响哪些文件）
- [x] 保留作者署名，**不改上游版权声明**（上游两位作者**追加**，不替换）
- [x] `version` = **`0.7.3+mie.1`**（Q-I）—— ⚠️ **实做时发现它触发了一条上游契约，见下**
- [ ] 生成 `tools/node_rename_map.json`（§4.4，自动生成）—— 属 Phase 2

**判据**：`git log --oneline origin/main` 仍可解析；`LICENSE` 逐字未改。 ✅

> ⚠️ **Phase 0 的附带发现：`version` 是一条上游契约，不是自由的字符串。**
>
> 上游把 `pyproject.toml` 的 `version` **同时当作浏览器 cache-bust token 的默认下限**，
> `tests/_web_cache_bust_unit_test.py` 的正则 `^(\d+)\.(\d+)\.(\d+)$` **不接受 `+` 后缀**，
> 于是 `0.7.3+mie.1` 让它直接失败（`AssertionError: Invalid web cache token '0.7.3+mie.1'`）。
>
> `0.7.3+mie.1` 在 PEP 440 / SemVer 里正是"上游版本 + 本地改动"的标准写法
> （`0.7.3 < 0.7.3+mie.1 < 0.7.4`），**版本号本身是对的，是上游的正则太窄**。
> 修法只放宽"能否解析"、**不动排序语义**：正则接受可选后缀，`version_key()` 取前三个数字组。
> 回归验证：把版本号换回 `0.7.3` 也同样通过 ⇒ 排序保证未变。
>
> **教训：版本号格式是一个被测试守着的契约。** 动它必须连带动守它的测试，
> 并把改动写进 `MODIFICATIONS.md`。

### Phase 1 — Windows 正确性修复（**P0**）（3–5 天，**下限不是上限**）

- [x] **P0-a（配置面，今天就能做，最高优先）**：关键路径 `checkpoint_verification` 设为 **`strict`**
      —— **本阶段的落地目标就这一条**，其余按 §8.6.1 的规则推进
      —— **实做**：widget 默认 / `export()` 签名默认 / `or` 兜底三处 + tooltip 重写；
      本机 ComfyUI 实测 `/object_info` 的默认值已是 `strict`；
      新增 `tests/_mie_png_verification_strict_unit_test.py` 把"cached 漏检 / strict 拦截 /
      默认是 strict"钉成可执行事实（负对照已做）
- [ ] **P0-b（恢复逻辑，真正的洞）**：三处"按 mtime 选源"各按自己的规则改
      （`chain_nodes.py:28453-28458` 遗留兜底 / `:22055` 封印清单 / `:4940` reference cache）
      ⇒ 规则见 **§8.6.1**（遗留多份 ⇒ 拒绝猜测；按 id 或文件内 `sealed_at`；
      语义查找剩多份 ⇒ 拒绝猜测）。**通则：哈希只判等值不作排序；没有"父级期望哈希"就不猜；
      不加 `_mie_seq`。**
- [ ] **P0-c（size 分层）**：`<64MB` 一律直接哈希（D21）——**做完这条，
      `:24681` 那段 PNG 缓存基本整段落到哈希上，ino 用不上**
- [ ] **P0-d（排查）**：按 **39 处 / 13 文件**、**四类**逐个判定（D19）
- [ ] **P0-e（文件系统识别）**：`GetVolumeInformationW` 识别 FAT/exFAT ⇒ 无身份信号（D17/D18）；
      UNC / `DRIVE_REMOTE` 映射盘 / `ino==0` / 配置白名单
- [ ] **P0-f（write-once）**：run 输出目录 write-once **+ 恢复路径显式豁免 +
      恢复按内容选源**（D20，见 §8.7 的纠正）
- [ ] **探针测试**：**原地覆写**一个同 size、保留 mtime 的文件，
      断言身份信号检测不到变化 ⇒ 该场景必须走 strict
      （**不是**测 `os.replace`——那条路径 ino 本来就会变）
- [ ] Windows 无符号链接/硬链接权限（WinError 1314）→ 代码侧降级而非崩
- [ ] 修 Node 24 的 `vm.runInContext` ESM 兼容（10 个 `.mjs`）
- [ ] 修剩余真 Windows 缺陷
- [x] CI 处置：**基线闸门**已落地（`tools/ci_gate.py` + `.github/workflows/ci.yml`）——Q-F′ 已裁定，§10.3

**判据**：`SUMMARY: N passed; 0 failed`（Windows）；
**外加**：构造"内容变、size 与 mtime 不变"的用例，断言缓存**不**命中。

> **工时分配提示（第三轮）**：P0 的时间应该花在 **P0-a（切 strict）** 与
> **P0-b（恢复逻辑）** 上；**ino 那一层不是重点**——
> PNG 导出缓存配合 `<64MB` 直接哈希后基本用不上 ino，
> 而真正让"较新的 mtime"变成正式文件的是恢复路径。

### Phase 2 — 命名与注册（1 天）

- [ ] 引擎包注册统一走 `add_suffix()`；显示名同步加后缀
- [ ] `tools/migrate_workflows.py`：subgraph 嵌套、幂等、目录批量（§4.3）
- [ ] 改名范围含 `tests/**`、`tests/_node_contracts_baseline.json`、
      `docs/examples/**`、`workflows/**`、上游 `example_workflows/**`（§4.5）
- [ ] CI 断言改名表条目数与注册类型数一致

**判据**：引擎包与上游原包**可同时安装**，`/object_info` 类型名无重叠。

### Phase 3 — plan 契约合并与 Scene Prompts 预览（2–3 天）

- [ ] 我们的 `validate_plan` 与上游 `_normalize_plan` 合并为**一份**（Core）
- [ ] 引擎包新增 `POST /mieh3/plan_preview`（§5.3）——含 `plan_sha256` 前 8 位 +
      **`script_ref` 路径限制**
- [ ] 引擎包前端实现 `mieH3PlanProvider` 约定 + fetch 预览
- [ ] Scene Prompt Editor：provider 优先，缺失时回落 widget + 明确横幅
- [ ] Plan Studio：socket 模式下只读 + 同样横幅
- [ ] `MieH3ScriptToPlan` 实现 provider 访问器

**判据**：§5.6 六条验收全过。

### Phase 4 — 抽 Core + 搬资产层（4–5 天，**含安装与配套，不只是搬文件**）

- [ ] 建 `ComfyUI-Mie-Core` 仓（**纯 pip 包**，D12）
- [ ] 按 §7.1 搬 11 项（含生成器全族 D11；LLM 拆分 D13）
- [ ] **删掉 `services/llm.py` 的三个死重依赖**（numpy / cv2 / torch，D14）
- [ ] **重指 `asset_auto_describe.py` 的 `library` import 到 Core**（D26）
- [ ] **冻结节点的写路径改走 Core 写入口**（D23）
- [ ] 按 §4.5 迁移 `tests/**`、SYNC PAIR 守护测试、critic 实测测试
- [ ] 引擎包加 §7.4 的薄壳节点
- [ ] 端到端：剧本资产 → `MieH3ScriptToPlan` → `plan_json` →
      `MiniMaxH3ChainPlanModern|MieH3` → 渲染
- [ ] **零 LLM 渲染路径保持字节级可复现**
- [ ] **部署**：ComfyUI 环境 `pip install -e Core`（本机 + 局域网）；产品层 venv 同样安装
- [ ] **变更窗口**：Core 先建可装 → MieNodes 与引擎**各一个 commit** → 部署脚本一次性切换

**判据**：
- 同剧本 + 同资产 + 同 seed ⇒ 同 `plan_json` 字节
- 真机跑通一条完整短剧（≥3 镜，含台词与参考图）
- **live combo matrix 改跑后缀名节点**（`MIE_LIVE_E2E=1`）
- Core 安装后，**ComfyUI 侧与产品层侧都能 `import mie_core`**

> **import 重指的实际范围**（第三轮核实并修正）：生产代码里只有**两处**真正 import 它：
> **根 `__init__.py:19`**（注册 26 个节点类）与 **`services/__init__.py:1`
> `from .llm import *`**（搜 `services/llm` 搜不到它，因为是相对导入）。
> ——`nodes/llm/invoke_budget.py:21` 只是**docstring 里提到**它。
> ⇒ D13 的重指范围 = **根 `__init__.py` + `services/__init__.py` + tests**
> （仍然不是"7 处生产 import"）。
> **Phase 4 里占时间的是 2,655 行 / 50 个类的拆分本身，不是这几个 import 行。**

### Phase 5 — 产品层后端（3–5 天）

- [ ] FastAPI 骨架 + §6.2 路由；`uvicorn workers=1`（D24）、端口 8199
- [ ] `pip install -e Core`，`import mie_core`
- [ ] 资产库读写**进程内直接调用**，不走 HTTP
- [ ] 站内剧本生成路由（调用 **Core 的生成器本体**）
- [ ] critic 路由：两半
- [ ] 渲染任务队列：提交 `/prompt`，轮询 `/history`，产物落盘 + 回写
- [ ] **渲染链路七坑**（§11）作为设计输入
- [ ] **资产目录写协议**（§10）

**判据**：产品层在 ComfyUI **未启动**时，剧本编辑/**生成**/预览/编译全部可用；
ComfyUI 启动后，提交渲染 → 出片 → 产物回显一条走通。

### Phase 6 — 产品层前端（3–5 天）

- [ ] 三页签 SPA：剧本 / 分镜预览 / 渲染
- [ ] 剧本编辑器：字段级覆盖
- [ ] 站内生成：concept/revise 两模式 + 报错教学
- [ ] 分镜预览：逐镜卡片，与引擎侧 Editor 显示一致（含 `plan_sha256` 对照）
- [ ] 渲染页：提交、进度、产物回放
- [ ] **资产页以 `js/assetLibraryUI.js` 为种子移植**（§6.4）

**判据**：一个不了解 ComfyUI 的用户，只用网页完成
**写剧本 → 生成/修改 → 出片**，全程不打开 ComfyUI 界面。

---

## 10. 资产目录写协议

### 10.1 已验证的事实

```
nodes/assets/library.py:298    fd, tmp_name = tempfile.mkstemp(...)
nodes/assets/library.py:583,585 os.replace(tmp, final) / os.replace(tmp_json, json_path)
nodes/assets/storyboard.py:799  tmp_fd, tmp_name = mkstemp(...)
nodes/assets/storyboard.py:806  os.replace(tmp_path, json_path)
```

**原子写有，但没有任何锁**（无 `fcntl` / `msvcrt` / `filelock`）。
原子写只保证**不写坏**，不保证**不互相覆盖**（last-writer-wins 静默丢更新）。

**写者有四个**：产品层、MieNodes `server.py`（留在 MieNodes！）、引擎节点、
**镜像同步脚本**（绕锁）。

### 10.2 bundle 原子性（D22）——第三轮发现的关键缺口

**一个资产 = 多个文件**（json + 立绘 + 音色样本），
`os.replace` **只保单文件原子**。
读者（引擎 roster 加载、预览路由）会看到**新 json + 旧图**的撕裂瞬间。

**方案：**

- [ ] **媒体文件用内容寻址命名**（不可变，先落盘）
- [ ] **json 清单最后原子翻转** ⇒ 单文件原子升级为 **bundle 原子**
- [ ] 读者按 json 里的内容寻址路径取媒体 ⇒ 撕裂窗口关闭

### 10.3 协议

- [ ] 所有写入**统一走 Core 的一个写入口**，该入口持有 advisory lockfile
- [ ] **CAS 的比较基准用「读时记录的内容哈希」，不用 mtime**（D16）——
      §8 已经证明 mtime 在 NTFS/SMB 上都不可靠；资产 JSON 都是小文件，
      直接比较内容哈希最稳
- [ ] 冲突语义：**409 + 返回对方版本**，由前端裁决，**不静默覆盖**
- [ ] **冻结的旧节点写路径也改走 Core 写入口**（D23）——
      否则统一写入口就漏了
- [ ] **`server.py` 写路由返 405/501 + 明示"去向产品层"**（D23）——
      **静默降级是又一种骗人界面**；真删除留给 MieNodes 下个大版本
- [ ] 镜像同步与产品层写入并发的运营规则（可执行）：
      产品层暴露**"空闲标志"**，同步脚本先查再动；
      双机 `-e` 同一副本的同步时滞 = **Core 版本偏斜** ⇒ 规则定为
      **"先同步 Core，再重启两侧服务"**
- [ ] **不上 SQLite**：资产库是"少文件、人频、镜像同步"形态，
      MIRROR-delete 直接复制文件树，SQLite 的 wal/journal 在镜像复制时机不对时会损坏，
      **比 JSON 更脆**

---

### 10.3 CI：基线闸门（Q-F′ 已裁定，2026-11-06）

本仓未来 **public**，CI 可以上云。但**闸门不能是"零失败"**——
上游套件在任何平台上都不干净：Windows 丢 56 个脚本（符号链接权限、NTFS 时间戳粒度、
Node 24 harness），Linux 丢另一批。一个要求零失败的闸门**从第一次跑起就是红的，
然后就再也没人看它**。

⇒ 落地是**基线闸门**（`tools/ci_gate.py`）：

| 规则 | 说明 |
|---|---|
| 基线文件 | `tools/known_failures/<platform>.txt`，按平台分别记录 |
| **新失败 ⇒ 红** | 不在基线里的失败才让构建失败 |
| 修好了 ⇒ 不红 | 基线条目开始通过时报告为"已修复"，可从基线里删掉；基线只会缩小，不会挡住 |
| **串行复跑** | 并行跑完后再把失败项**逐个串行复跑**，把时序敏感的 flaky 排除在基线外 |
| **我们的回归测试是硬门槛** | `_mie_png_verification_strict_unit_test.py` 与 `_mie_mtime_selection_unit_test.py` **无论基线怎么说都必须通过**，否则直接红 |
| 首次运行 | 没有基线时**写出基线并放行**（不假装绿），提示提交该文件 |

工作流：`.github/workflows/ci.yml`（`ubuntu-latest` + `windows-latest`）。
测试需要 ComfyUI 检出（`comfy/options.py`）与 torch/numpy/av/safetensors/pillow —— 轻量。

**明确不在 CI 里的（必须在本机与局域网的真实盘上跑）：**

- Windows 卷类型（FAT / exFAT）——托管 runner 只有一块盘；
- 映射盘（`GetDriveTypeW == DRIVE_REMOTE`）——角色资产在 LAN 的 UNC 共享上；
- 符号链接权限（WinError 1314）——开发者模式默认关；
- **原地覆写且保留 size/mtime**——证明缓存守卫失效模式的那个探针，需要真实文件系统；
- `MIE_LIVE_E2E=1`——对着 H3 模型与 60GB 卡的**真机渲染矩阵**，不是"几分钟巡检"。

**顺带处理的一个隐患**：上游带来的 `.github/workflows/publish_action.yml` 在
"push 到 main 且改动 `pyproject.toml`"时**自动**触发 Comfy Registry 发布。
我们 Phase 0 正好改了 `pyproject.toml` ⇒ 第一次 push 就会触发它。
已改成**只保留手动触发**（见 §10.3 与该文件顶部说明），等 Q-E 真正拍板再开。

**公开前的卫生扫描（2026-11-06，已做）**：仓库内**无**密钥/令牌、
**无**内网地址或 UNC 路径、**无**个人路径、**无**未跟踪残留；
大于 2MB 的文件只有上游自带的示例工作流素材（本就公开）。

---

## 11. 渲染链路七坑（Phase 5 的设计输入）

全部是本项目真机踩过的：

| # | 坑 | 产品层的应对 |
|---|---|---|
| 1 | `/history` **重启即清空** | 任务状态**必须落盘可恢复**；`output/h3_chains/<run>/` 取证模式是既定事实 |
| 2 | **RECOVERY 节点在全新运行必炸** | 产品层模板去掉 RECOVERY 节点，或改 `run_name` |
| 3 | **手改编译产物/资产的节点缓存坑** | 同 id 重提复用旧 bundle ⇒ **每次渲染自带全新 `run_name` + 新 node id** |
| 4 | **60GB 显存下单任务排队** | 渲染队列**串行化** |
| 5 | **双实例 / 谎报的 restart** | launcher 的 restart 可谎报成功，旧进程继续占 8188 ⇒ 提交进 A 实例、轮询 B 实例 ⇒ **产物错配**。**提交前探 `/system_stats` 取实例指纹**（pid / 启动时间），轮询时比对，不一致即拒 |
| 6 | **`/history` 条目存在 ≠ 成功** | 必须按条目的 **`status` 字段**判失败 |
| **7** | **共享队列**（第三轮新增） | 坑 4 让产品层串行化**自己的**队列，但 **8188 的队列是全局的**——串行化不了人手提交。本项目的现实使用方式恰是产品层与用户手工驱动同一台 8188 ⇒ 产品层任务执行中有人插队 ⇒ 60GB 卡上交错执行 ⇒ **OOM 或双双失败**。**轻量应对**：提交前后探 `/queue` 深度，检测到外来任务即**报警并暂停提交**。**重量应对（v2）**：产品层独占第二个 ComfyUI 实例。⚠️ **探测与提交之间有空窗**（第三轮指出）⇒ 把"探测 + 提交"放进**同一把进程内锁**的临界区（R21） |

**荣誉提名两条**（第三轮附加）：

- **产物文件写入中就被读取** ⇒ 等 `status=success` **且大小稳定**再取；
- **中文 `run_name` × 深目录 ⇒ Windows 260 字符 `MAX_PATH`** ⇒ 启用长路径或限制长度。

---

## 12. 风险

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | **上游不再回灌**（2026-11-06 裁定：D6） | 上游的修复我们拿不到 | 接受：我们自担全部维护。<br>**但保留 `upstream` remote 作溯源参考**（base commit `43afeff`、随时可 diff 对照），**不再周期性合并**。<br>⇒ 风险从“同步成本”转移到**“ComfyUI 兼容性要自己扛”**（见 R8）——这让 Phase 1 的 Windows 修复与“ComfyUI 版本兼容矩阵”检查**更重要** |
| R2 | 批量改名漏改 | 节点注册不全，工作流红 | 改名表**自动生成**（AST 扫 13 个模块，86 条）+ CI 断言条目数相等 + **改名范围含 `tests/**` 与 baseline json**。**实施后补记**：还有两个工具查不出来的陷阱——JS 的**裸标识符键**（须加引号，`|` 不是合法标识符字符）与**语义上的历史名**（退役/旧版检测集合，须保持旧名） |
| R3 | Phase 1 的 P0 排查拖太久 | 阻塞后续 | P0 分五条；**P0-a 今天就能做**；P0 未完成不得进 Phase 3 |
| R4 | plan 编译逻辑分叉 | 预览与实际渲染不一致 | **一份实现多处调用**（§3.3） |
| R5 | FastAPI 与 ComfyUI 的状态竞争 | 产物错配 | `run_name` 唯一键；产物由 ComfyUI 写，产品层只索引；**实例指纹校验**（坑 5） |
| R6 | 概念边界混乱 | 实现走偏 | 边界：**逻辑在 Core，MieNodes 与产品层都是壳**（§7.2） |
| R7 | 引擎包 157k 行无人 review | 质量下滑 | 保留上游 65k 行测试 + 新增契约测试 + CI |
| R8 | 上游依赖 `comfy.ldm.minimax.model` 内部 API | 引擎不可用 | fork 后由我们承担；加"ComfyUI 版本兼容矩阵"检查 |
| R9 | 产品层 import GPL 核心 ⇒ GPL 衍生品 | 发布策略受限 | 与 Q-E 联动：内部 LAN 无碍；**公开发布产品层必须同发源码** |
| R10 | 多写者丢更新 | 资产被静默覆盖 | 单一写入口 + advisory lock + **内容哈希 CAS**；409 + 返回对方版本 |
| R11 | Phase 1 就已破"只新增文件"的约束 | R1 缓解强度下降 | **D6 已换策略**；原地修改外科手术化并逐条进 `MODIFICATIONS.md` |
| R12 | **UNC 上的 mtime 粒度更粗** | 静默错渲染（比本地更易发生） | P0-d + D18 分级；远程强制 strict |
| R13 | 生成器搬到 Core 后 prompt 家族与代码分家 | critic/生成器读不到指令 | D11 一并搬；SYNC PAIR 守护改为**跨仓内容一致性断言** |
| R14 | 产品层 venv 装不下 torch/cv2 | Phase 5 第一天就撞 | D14 删死依赖 ⇒ Core 只依赖 `requests` |
| **R15** | **bundle 撕裂**（新 json + 旧图） | 读到不一致的资产 | **D22** 内容寻址 + 清单最后翻转 |
| **R16** | **共享队列插队**（第三轮） | OOM 或双双失败 | **D25** `/queue` 深度探测 + 暂停提交；v2 独占实例 |
| **R17** | **冻结节点绕过统一写入口** | 统一写入口漏写者 | **D23** 冻结节点写路径改走 Core 入口 |
| **R18** | **`st_ino` 在 FAT/exFAT/UNC 上不可靠**（第三轮） | P0-b 修法失效 | **D17/D18**：`GetVolumeInformationW` 识别 FAT/exFAT ⇒ 无身份信号（**本地 exFAT 给的是非 0 目录项编号，只看 `ino==0` 会误放行**）；UNC / `DRIVE_REMOTE` ⇒ 强制 strict；`ino==0` 兜底 |
| **R19** | **双机 Core 版本偏斜**（第三轮） | 产品层与引擎用不同 Core | 规则：**先同步 Core，再重启两侧服务**（§10.3） |
| **R20** | **"按 mtime 选源"让外部覆写变成正式产物**（第三/四轮） | 静默用错内容渲染 | **P0-b**：三处各按自己的规则选源（§8.6.1）——① 遗留多份临时文件 ⇒ **拒绝猜测**；② 按 `chapter_manifest_id` / 文件内 `sealed_at`；③ 剩多份 ⇒ **拒绝猜测**。**通则：哈希只判等值不作排序；没有"父级期望哈希"就不猜；不加 `_mie_seq`** |
| **R23** | **同步 `requests` 卡死事件循环**（第四轮） | 预览/轮询/空闲标志停摆 | 生成与 critic 路由用**同步 `def`**，或 `anyio.to_thread.run_sync`（§6.1） |
| **R24** | **锁的粒度与持锁时长**（第四轮） | 队列被拖死 / 锁形同虚设 | 锁**只包住"看 `/queue` + 提交"**；生成路由在线程池里用 `threading.Lock`，**不在等 LLM 时持锁**（§6.1） |
| **R21** | **`/queue` 探测与提交之间有空窗**（第三轮） | 仍可能被插队 | `workers=1` 只保证单进程；**产品层自己的队列另加一把进程内锁**，把"探测 + 提交"合成一个临界区 |
| **R22** | **`uvicorn workers=1` 仍会异步交错**（第三轮） | 渲染队列串行化被破坏 | 产品层队列**进程内锁**（R21），不依赖 worker 单例假设 |

---

## 13. 不做什么

- **不重写上游生成核心**（§1.4）。
- **不把引擎包和 MieNodes 合成一个包**（D1）。
- **不做视频剪辑 / 时间线 / 多轨混音 / 字幕编辑器**。
- **不在 ComfyUI 里做分镜图形编辑器**。分镜编辑在产品层；ComfyUI 侧只读预览。
- **不改已发布的 MieNodes 节点契约**。v1/v2/v3 类型名、输入顺序、输出顺序不动。
- **不合并上游 `project_assets` 与 Core 资产库**（§7.5）。
- **不改 `plan_json` 的顶层形状**。
- **v1 不做跨机部署**（D8）。
- **不把 Core 做成依赖 ComfyUI 的包**（否则产品层装不了）。
- **不把 `core/utils.py` 搬进 Core**（品牌专属）。
- **不把资产库改成 SQLite**（§10.3）。
- **不删除 MieNodes 的 `server.py` 写路由**——降级为 405/501，真删除留到下个大版本（D23）。

---

## 14. 待裁定

| # | 问题 | 状态 / 倾向 |
|---|---|---|
| **Q-F′** | **CI 与"不再推送"冲突** | **已裁定（2026-11-06）：本仓未来为 public，CI 可以上云。** 落地见 §10.3：`ubuntu-latest` + `windows-latest` 双平台跑 `tools/ci_gate.py`，**基线闸门**而非"零失败"闸门；发布到 Comfy Registry 仍**暂缓**（Q-E），上游那个自动触发的发布工作流已改成手动 |
| Q-A′ | 共享核心形态 | **(a) 独立 pip 包** + D12 的两条机械约束 |
| Q-B | 存量工作流迁移 | **A**（纯改名 + 迁移脚本） |
| Q-C | 后缀 vs 前缀 | **`\|MieH3`** |
| Q-D | Scene Prompt Editor 只读 | **完全只读** + 新鲜度验收（§5.5） |
| Q-E | 引擎包发不发 Comfy Registry | **不发**（D10） |
| **Q-F** | Windows 修复要不要回流上游 | **已裁定：不回流**（2026-11-06）。上游是原仓库 `ethanfel/ComfyUI-MiniMaxH3-Context-Loop`；作者不怎么维护，**我们无视它，只在自有环境实现**。⇒ 那些修复**只存在于我们的 fork**，不进上游 |
| Q-G | MieNodes 旧资产节点 | **弃用 + 冻结**（D9） |
| Q-G′ | 部署拓扑 | **v1 同机已定**（D8）；跨机是 v2 |
| Q-H | critic 模型半放哪 | **已消解**：两半同住 Core（D11） |
| Q-I | `version = 0.7.3+mie.1` | **接受** |

---

## 15. 证据索引

### 上游（`C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-Context-Loop`）

| 事实 | 位置 |
|---|---|
| Production Plan 节点 | `chain_nodes.py:15352` |
| `plan_json_input` socket 声明 | `chain_nodes.py:15431-15445` |
| socket 覆盖 widget | `chain_nodes.py:15322-15326` |
| plan 归一化 / 校验 | `chain_nodes.py:7362`（per-shot `:7484+`） |
| `IS_CHANGED` 对普通编辑恒 False | `chain_nodes.py:15284-15305` |
| **缓存守卫本体** | `chain_nodes.py:24697-24713` |
| **`checkpoint_verification` 是 widget，默认 `cached`；tooltip 说"immutable"** | `chain_nodes.py:25166-25172` / `:25222` |
| 守卫的三个调用点（`expected_hash` 来自 `segment["checkpoint_sha256"]`） | `chain_nodes.py:25395` / `:25586` / `:25614` |
| **"按 mtime 选源 ⇒ 变成正式文件"：遗留恢复兜底（journal 之前的旧构建）** | `chain_nodes.py:28453-28458`（注释 `:28439-28442` 明说是 journal 之前的遗留） |
| **journal 路径本身是安全的**：按记录的 `temporary` 文件名 `os.replace`；journal 损坏 ⇒ 拒绝（删临时文件，不猜） | `chain_nodes.py:28395-28421`（`else` 分支 `:28424-28431`） |
| 已封印 Chapter 清单按 mtime 选源 | `chain_nodes.py:22055` |
| **清单摘要排除 `sealed_at` / `chapter_manifest_id` / `chapter_manifest_path`** | `chain_nodes.py:21818-21832` |
| **`sealed_at` 写在文件内容里**（robocopy /COPYALL 会连它一起拷走） | `chain_nodes.py:21868-21878` |
| reference cache 候选按 mtime 选源（候选已按契约过滤） | `chain_nodes.py:4926`（append mtime）、`:4940`（取 max） |
| **reference cache 是语义查找，调用方没有文件哈希** | `chain_nodes.py:4936`（`_find_reference_cache` 签名） |
| **`web/` 里硬编码的类型名（47 个 / 20+ 文件）** | 见 §4.5 表 |
| **仅用于排序的 "最新者胜"（显示层，非渲染正确性）** | `run_manager.py:534` → `:558` |
| **上游自述：四元组抓不到 same-stat byte edits** | `storage_inventory.py:367-368` |
| 竞态检测用的四元组 | `storage_inventory.py:212-213`、`:371-372` |
| **目录那一半**（非文件四元组的弱化版） | `storage_inventory.py:380` |
| mtime 参与点 **39 处 / 13 文件** | 见 §8.6 表 |
| Plan Studio 横幅 / 页脚 | `web/h3_chain_plan_editor.js:1970-1978` / `:2112` |
| Scene Prompt Editor 读 / 写 widget | `web/h3_chain_rich_scene_prompt_editor.js:2383-2395` / `:744`、`:804` |
| 上游教用户断开 `plan_json_input` | `docs/PROJECT_ASSETS.md:39` |
| plan JSON schema 规范 | `H3_CHAIN_FORMAT_GUIDE.md:204-290` |
| 测试运行器 / 无 CI | `tools/check_release.py` / `.github/workflows/` 仅 `publish_action.yml` |
| 停更证据 | GitHub API `pushed_at=2026-10-14T11:08:20Z`；issue #117 / #120 |

### ComfyUI

| 事实 | 位置 |
|---|---|
| 节点名静默覆盖 | `nodes.py:2305-2308` |
| `executed` 发给每个节点 / `/history` 只收 OUTPUT_NODE | `execution.py:583` / `:601` |

### 我们（MieNodes）

| 事实 | 位置 |
|---|---|
| 后缀约定 | `core/utils.py:12`、`:23` |
| 节点注册（306 个） | `__init__.py:55` 起 |
| **`services.llm` 的真正生产 import（两处）** | `__init__.py:19`、`services/__init__.py:1`（`from .llm import *`，搜 `services/llm` 搜不到） |
| `invoke_budget` 只是 docstring 提到 services/llm.py | `nodes/llm/invoke_budget.py:21` |
| `nodes/llm/__init__.py` 拖起整包（`from .x import *` × 20+） | `nodes/llm/__init__.py` |
| LLM 连接器是**注入**的（loop generator 5,503 行不 import services.llm） | `nodes/llm/minimax_h3_loop_prompt_generator.py:4774`（socket 声明） |
| `services/llm.py` 50 类 / 26 个带 `INPUT_TYPES` / 零 `folder_paths` | `services/llm.py:1264`、`:2185` |
| **numpy / cv2 / torch 是死 import**（各只出现 1 次） | `services/llm.py:6-8` |
| 生成器对资产层的 import：**3 模块 / 5 行** | `nodes/llm/storyboard_asset_generator.py:25/176/198/297/308` |
| **critic 模型半的 PROMPT_DIR 指向 prompt 家族** | `nodes/llm/storyboard_critic.py:29` |
| **asset_auto_describe 的三方 import 面** | `nodes/llm/asset_auto_describe.py:27/32/37/45` |
| 原子写但无锁 | `nodes/assets/library.py:298/583/585`、`nodes/assets/storyboard.py:799/806` |
| file-path shim 机制（先例） | `nodes/llm/minimax_h3_loop_prompt_generator.py:104-125` |
| 我方 plan 校验器 | `nodes/llm/minimax_h3_loop_prompts.py:1775` |
| **资产库前端 UI（74KB，6 个端点）** | `js/assetLibraryUI.js`（1,551 行）+ `js/` 共 13 个文件 |
| 节点契约基线 | `tests/_node_contracts_baseline.json` |
| live combo matrix | `tests/test_h3_loop_live_combo_matrix.py`（`MIE_LIVE_E2E=1`） |
| critic 实测测试（default-on） | commit `eb2e7e1` |

---

## 16. 评审处置台账（三轮）

### 16.1 第一轮（约 25 条，全部采纳）

契约依赖方向矛盾（→ Core 包）、搬迁清单不全、critic 是两半而非两份、
provider 的 planJson 来源（→ 后端路由）、R4/Phase 5 矛盾（→ 一份实现多处调用）、
缺"生成剧本"路由、缺"为什么不 wrapper"、R1 在 Phase 1 就破约（→ 改 merge）、
改名漏改（含 `tests/**`）、双进程写同目录无锁、渲染四坑、CI 与不推送冲突、
部署拓扑隐式、mtime 可能是生产问题（→ P0）、显示名惯例、`version` 形态、live matrix。

### 16.2 第二轮（约 25 条，全部采纳）

| 意见 | 处置 |
|---|---|
| **P0 机制描述被读错** | 纠正：① 是"上次实测 vs 调用方期望"，**不是恒真**；照错误读法修会什么都没修好 |
| strict 逃生口未被利用 | `checkpoint_verification` 已是 widget ⇒ P0-a 纯配置面 |
| cache key 补 `st_ino`/`st_dev` | 采纳（见 §8.5，带降级规则） |
| 清单不全 | 实测 **39 处 / 13 文件**，逐文件补齐 |
| UNC/SMB 粒度 | 新增 P0-d + R12 |
| **生成器归属 vs Phase 5 判据矛盾** | **D11：生成器全族进 Core**，MieNodes 只留 408 行壳 |
| critic 的 PROMPT_DIR 指向 prompt 家族 | 一并搬；**Q-H 消解** |
| R6 表述改为"逻辑在 Core，两边都是壳" | 采纳 |
| SYNC PAIR 守护变跨仓 | 降级为**跨仓内容一致性断言**（R13） |
| Core 作为 ComfyUI 包解决不了 import | **D12** 纯 pip 包 + `pip install -e` |
| "同一个 commit" 做不到 | 改**同一变更窗口** |
| Phase 4 依赖面漏 cv2 | 采纳并加强：三个都是**死 import** ⇒ 直接删（D14） |
| `core/utils.py` 品牌专属 | 不搬 |
| 预览路由加 `plan_sha256` | 采纳 |
| `script_ref` 路径穿越 | 采纳 + 列入验收 |
| CAS 基准别用 mtime | **D16** 内容哈希 |
| 别上 SQLite | 采纳 |
| 第五坑：双实例 / 谎报 restart | 坑 5 + 实例指纹 |
| 第六小坑：history 有条目 ≠ 成功 | 坑 6，按 `status` 判 |
| Phase 4 缺"测试随模块走" | 采纳；CI 里可关 |
| Phase 4 估时只算搬文件 | 改 **4–5 天** |
| "四处" 实为 3 模块 5 行 | §3.2 已分列 |
| §7.3 三处口径不一 | 统一为"两半同住 Core" |

### 16.3 第三轮（全部采纳）

| 意见 | 处置 |
|---|---|
| **`st_ino` 的可靠性分层**（NTFS 好 / FAT 覆写不变 / UNC 不可信） | **D17** 降级规则：`ino==0` 或远程 ⇒ 无身份信号 ⇒ 强制 strict |
| **分级判据**：UNC 前缀 + **映射盘符**（`GetDriveTypeW == DRIVE_REMOTE`）+ `ino==0` + 配置白名单 | **D18** 四条并列 |
| **第二类用法**：`run_manager.py:534` "最新者胜" | §8.6.1 独立危险线 |
| **惊喜**：`storage_inventory.py` 已用 `st_ino` 四元组（`:212-213`/`:371-372`），`:380` 是 mtime-only 孪生 | 修法与上游惯用法**同构** ⇒ 回流顺、冲突风险低；`:380` 对齐成四元组 —— ⚠️ **此条第四轮已修正，见 §16.4**（四元组不是内容身份；`:380` 是目录那一半，不必对齐） |
| **P0-c 按四类分类**（缓存跳过 / 身份等值比较 / 最新者胜 / 年龄清理+报告） | **D19** |
| **威胁模型**：上游信任"checkpoint 不可变"这个从未强制的不变量 | §8.3 写明 |
| **第三条腿：run 输出目录 write-once** | **D20** |
| **size 分层**：`<64MB` 永远直接哈希 | **D21** |
| **搬迁清单缺 `asset_auto_describe.py`** | **D26** 留 MieNodes 作壳 + `library` import 重指 Core |
| prompt 家族 7 个文件（含 `UPSTREAM.md`） | §7.1 已含 |
| "MieNodes 全仓没有任何前端 `.js`" | **⚠️ 此断言不成立**：MieNodes 有 `js/` 共 **13 个文件**，含 `assetLibraryUI.js`（**1,551 行 / 74KB**，调用 6 个 `/mie_nodes/assets/*` 端点）。⇒ **"非 .py 资源漏搬"不能关闭**；但它是资产 UI 的**冻结件**，且正好是产品层资产页的**种子**（§6.4） |
| **§12 "唯一 importer = invoke_budget.py" 是错的** | 核实：`invoke_budget.py:21` 只是 **docstring 提到**；生产代码**零处** import `services/llm.py`，真正注册 import 在**根 `__init__.py:19`**。⇒ Phase 4 的重指范围 = 根 `__init__.py` + tests —— ⚠️ **此条第四轮已修正，见 §16.4**（`services/__init__.py:1` 也是真 import） |
| **写协议漏 bundle 原子性** | **D22** 内容寻址 + 清单最后翻转 |
| 冻结节点写路径要并入统一入口 | **D23** |
| `server.py` 写路由返 405/501 而非静默只读 | **D23**；真删除留到下个大版本 |
| 锁只在同机有效；镜像同步是第四写者 | 产品层暴露**空闲标志**；规则：**先同步 Core，再重启两侧服务**（R19） |
| 端口无冲突；`uvicorn workers=1`、关 `--reload` | **D24**（端口 8199） |
| **第七坑：共享队列** | **D25** `/queue` 深度探测 + 暂停提交；v2 独占实例（R16） |
| 荣誉提名：产物写入中就被读 / 中文 run_name × 深目录 ⇒ MAX_PATH | §11 荣誉提名两条 |

### 16.4 第四轮（三处纠错，全部采纳）

| 意见 | 处置 |
|---|---|
| **① 「生产代码零命中」漏了一个真 import** | 核实成立：`services/__init__.py:1` 是 `from .llm import *`（相对导入，搜 `services/llm` 搜不到）。⇒ D13 重指范围 = 根 `__init__.py:19` + `services/__init__.py:1` + tests；**Phase 4 占时间的是 2,655 行 / 50 个类的拆分，不是这几个 import** |
| **② D17 的代码没有执行它自己的 FAT 结论** | 核实成立：本地 exFAT 给的是**非 0 的目录项编号**，原 `identity_signal` 会误放行。⇒ 改为用 `GetVolumeInformationW` 识别 FAT/FAT32/exFAT（**本机已验证可用**，`C:\` 返回 `NTFS`）⇒ 无身份信号 ⇒ 强制 strict |
| ② 上游 `storage_inventory` 的四元组**不是内容身份** | 核实成立：上游 `:367-368` 自述 *"does not detect same-stat byte edits"* ⇒ 它是扫描期竞态检测。⇒ 改写为**加强论据**（连上游都承认它抓不到 stat 不变的字节改写），不再声称"修法同构"；`:380` 是**目录那一半**，**不必**对齐成四元组 |
| ② 探针测的是已经能防住的路径 | 采纳：探针改为**原地覆写**同 size、保留 mtime 的文件，断言身份信号检测不到变化 |
| ② `:24681` 那段缓存管的是 PNG，配合 `<64MB` 直接哈希后**轮不到 ino** | 采纳：**P0 工时应花在 P0-a（切 strict）与恢复逻辑上**，ino 不是重点 |
| **③ 「最新者胜」点错了更危险的位置** | 核实成立：`run_manager.py:534` 的结果进 `modified_at`，`:558` 用于**排序** ⇒ 只把错误的 run 排到最前，**不改 checkpoint 字节** ⇒ 降级为显示层。**真正的洞**是三处"按 mtime 选源"：`:28453-28458`（恢复 temp → 正式 checkpoint，最危险）、`:22055`（封印章节清单）、`:4940`（reference cache 候选） |
| ③ D20 write-once 会与 `:28458` 的恢复路径撞车 | 采纳：D20 改为三条一起——**write-once + 恢复路径显式豁免 + 恢复时按内容而非 mtime 选源**；并写明 **write-once 只约束走我们写入口的进程，挡不住外部覆写** |
| `workers=1` 只保证单进程，异步请求仍交错 | 采纳：**产品层队列另加一把进程内锁**（R21/R22），不依赖 worker 单例假设 |
| `/queue` 探测与提交之间有空窗 | 采纳：把"探测 + 提交"放进同一临界区（R21） |
| 文档结构说明少写 §13 与 §17 | 属摘要表述问题，**文档本身 §13「不做什么」与 §17「变更记录」均完整**，无需改动 |

### 16.5 第五轮（三条实操暗礁 + Q-F′ 约束，全部采纳）

| 意见 | 处置 |
|---|---|
| **暗礁 1：哈希不能排序，但不要加 `_mie_seq`** | **采纳并按三条路径分别改写 §8.6.1**：① `:28453-28458` 是 **journal 之前的遗留兜底**（上游注释 `:28439-28442` 明说），journal 路径 `:28395-28421` **已经安全**（按记录的 `temporary` 名 `os.replace`，journal 损坏 ⇒ 拒绝）⇒ 遗留兜底里多份临时文件 ⇒ **拒绝猜测**，恰好一份才用；文件名里的 32 位 hex 是**该临时文件自己的摘要**，**没有"父级期望哈希"**；② `:22055` 的清单**内容里带 `sealed_at`**（`:21868-21878`），摘要**排除** `sealed_at`/`chapter_manifest_id`/`chapter_manifest_path`（`:21818-21832`）⇒ 调用方给了 `snapshot_id` 就**按 id 取**，没给就**比文件内的 `sealed_at`**；③ `:4940` 是**语义查找**（调用方没有文件哈希，候选已按契约过滤）⇒ **剩一份就用，剩多份拒绝猜测**。**通则写入文档：哈希只判等值不作排序；没有"父级期望哈希"就不猜；绝不加 `_mie_seq`（会改变摘要与文件名，等于另造一份快照）。** `expected_sha256` 对上号**只适用于调用方事先记下目标哈希的路径**（如 PNG 导出守卫 `:24681`） |
| **暗礁 2：改名必须扫 `web/`** | **采纳，§4.5 重写**。实测上游 `web/` 硬编码 **47 个类型名、20+ 个文件**（`MiniMaxH3ChainPlan`×14、`MiniMaxH3ChainPlanModern`×13、`MiniMaxH3ChainPlanStudio`×11 …）。只改 Python ⇒ 扩展挂不上 `|MieH3` 节点，两包同装时还会**挂到上游原名节点上**。规则：用 `node_rename_map.json` 做**全字匹配**；**不要按 `MiniMaxH3` 前缀扫**（会误伤节点 id 如 `MiniMaxH3ContexLoop.cancelRerollControl`、扩展 id、仓库 URL、注释） |
| **暗礁 3：同步 `requests` 会卡住事件循环** | **采纳，写入 §6.1 + R23**：Starlette 把 `def` 路由丢线程池、把 `async def` **直接在事件循环里跑** ⇒ 生成与 critic 写成 `async def` 再调同步 `mie_core`/`requests`，**几十秒内预览、轮询、空闲标志全停** ⇒ 这两条路由用**同步 `def`**，或 `anyio.to_thread.run_sync`。**锁只包住"看 `/queue` + 提交"那一小段**；生成路由在线程池里用 `threading.Lock`，**不在等 LLM 的几十秒里持锁**；锁仍只串行本进程，人手在 8188 的提交不进这把锁，两次 HTTP 之间的空窗仍在（R24） |
| **Q-F′ 的三条硬约束** | **采纳，写入 §14**：① **GitHub 托管 runner 是虚拟盘** ⇒ Windows 卷类型 / 映射盘 / 符号链接权限 / 原地覆写**测不到**，这几组探针**必须在本机和局域网真实盘上跑**；② **Git hook 顶不上**——`pre-push`/`post-commit` 可被 `--no-verify` 跳过，且**不随 clone 走**；③ **`MIE_LIVE_E2E=1` 是真机渲染矩阵**，与跨仓内容一致性断言**不是一个量级**，不算进"几分钟巡检"；**纯逻辑单测不依赖那块盘**。⇒ 两支：**① 允许私有仓** ⇒ 只把 Core 上 Actions、引擎包留局域网跑 `tools/check_release.py`；**② 任何代码都不上云** ⇒ **没有 Actions 这一档**。**仍只有用户能拍** |
| **从哪切** | **采纳，写入 §9 开头**：**Phase 0 → 只落地 P0-a**（`checkpoint_verification` → `strict`）——当天能做的生产修复，**不等 Q-F′**、**不动三处恢复逻辑**；**P0-b 等 §8.6.1 三条规则定稿后再动**；**Phase 4 的 `services/llm.py` 拆分不修静默错渲染**；**改名脚本可在 fork 之后做，但第一次扫描必须含 `web/`** |

---

## 17. 变更记录

| 日期 | 修订 |
|---|---|
| 2026-11-05 | 初稿（`docs/PLAN_H3_ENGINE_FORK_AND_DRAMA_PLATFORM.md` v1）：D1–D4；上游停更实测；Editor 缺陷定位；plan provider 方案 |
| 2026-11-05 | v2：第一轮评审。补 Core 包（D5）、provider 改后端路由、mtime 提为 P0、R1 改 merge、写协议、四坑 |
| 2026-11-05 | v3：第二轮评审。**修正 P0 机制**、清单补到 39 处/13 文件、UNC 新线、**生成器全族进 Core**（D11）、Core 改纯 pip 包（D12）、LLM 拆分（D13）、删死依赖（D14）、P0 分四条（D15）、CAS 改内容哈希（D16）、第五/第六坑 |
| 2026-11-05 | **本文件**：第三轮评审 + 合并。**P0 补分层真相与两条新腿**（D17–D21）、**bundle 原子性**（D22）、**冻结写路径与 405**（D23）、**uvicorn 单进程**（D24）、**共享队列**（D25）、**asset_auto_describe 归属**（D26）；**纠正评审两处不成立的断言**（前端 .js 存在、invoke_budget 非 importer）；合并 v3 计划与审阅简报为单一真源 |
| 2026-11-05 | **第四轮纠错**：**D17 改为按文件系统类型判定**（`GetVolumeInformationW`；本地 exFAT 给非 0 目录项编号，只看 `ino==0` 会误放行）；**探针改测原地覆写**；**"上游惯用法同构"改写为"上游自述抓不到 same-stat byte edits"**（加强论据）；**"最新者胜"降级为显示层**，真正的洞改为三处"按 mtime 选源"（`:28453-28458` 恢复路径最危险 / `:22055` / `:4940`）；**D20 write-once 补恢复豁免 + 按内容选源**；**补 `services/__init__.py:1` 这个真 import**；**补进程内队列锁**（R21/R22） |
| 2026-11-05 | **第五轮实操暗礁**：**§4.5 补 `web/`**（实测 47 个硬编码类型名 / 20+ 文件；全字匹配、不按前缀扫）；**§8.6.1 按三条路径分别写清选源规则**（遗留多份⇒拒绝猜测 / 按 id 或文件内 `sealed_at` / 语义查找剩多份⇒拒绝猜测；**通则：哈希只判等值、绝不加 `_mie_seq`**）；**§6.1 补同步 `requests` 卡死事件循环**（生成与 critic 用同步 `def` 或 `anyio.to_thread`）+ **锁只包住"探测+提交"**；**§14 补 Q-F′ 的三条硬约束**（runner 虚拟盘 / hook 可绕 / live matrix 不同量级）与两支；**§9 开头写明从 Phase 0 → 只落地 P0-a** |
