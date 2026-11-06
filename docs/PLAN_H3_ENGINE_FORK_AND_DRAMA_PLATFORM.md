# H3 引擎 Fork + 短剧平台 实施计划

> ⚠️ **本文件已被取代（2026-11-05）**。唯一真源是
> [`docs/PROJECT_H3_ENGINE_AND_DRAMA_PLATFORM.md`](PROJECT_H3_ENGINE_AND_DRAMA_PLATFORM.md)——
> 它合并了本文件（v3）与 `docs/qa/h3-engine-fork-20261105/REVIEW_BRIEF.md`，
> 并并入了第三轮评审意见。本文件仅作历史留存。

> **状态**：第三版（2026-11-05），已并入**第二轮**外部评审意见（处置台账见 §15）。
> **前置**：`docs/PLAN_STORYBOARD_ASSET.md`、`docs/PLAN_STORYBOARD_FIRST_ASSET_V3.md`、
> `docs/PLAN_ASSET_LIBRARY_V2.md`。
> **上游**：`ethanfel/ComfyUI-MiniMaxH3-Context-Loop` @ `43afeff`（GPL-3.0）。
> 本文档描述**新增引擎包** + **新增共享核心包** + **新增产品层**。
> 不改动已发布的 MieNodes v1/v2/v3 节点契约。

---

## 0. 一句话

把上游 `ComfyUI-MiniMaxH3-Context-Loop` fork 成 `ComfyUI-Mie-H3-Engine`
（保住 H3 长视频生成能力并修好它在 Windows 上的缺陷），
把 MieNodes 现有的**生成逻辑 + 剧本契约 + 资产库 + LLM 连接器**抽成一个
**可 pip 安装的共享核心包 `ComfyUI-Mie-Core`**，
引擎包、MieNodes、产品层（独立 FastAPI）三者都是它的壳，
再用一条 `plan_json` 字符串把"剧本 → 生成"这条缝接上，
并给 ComfyUI 侧的 Scene Prompt Editor 一个**只读预览**通道。

**生成引擎我们不重写。** 上游那 148k 行里最难啃的部分
（ComfyUI H3 内部耦合、六种续接模式、音频窗口、checkpoint 血缘）
是 6 周 GPU 调试的产物，重写等于重走一遍。

---

## 1. 决策摘要

### 1.1 已定（用户 2026-11-05 裁定）

| # | 决策 | 落点 |
|---|---|---|
| D1 | 引擎 = **独立包** `ComfyUI-Mie-H3-Engine` | §4.1 |
| D2 | 产品层 = **独立 FastAPI 进程** | §6 |
| D3 | 分镜资产**对接 Scene Prompts 做预览**；Plan Studio 不参与修改 | §5 |
| D4 | 节点类型名**一律加我们的后缀** | §3 |

### 1.2 第二版新增（第一轮评审后）

| # | 决策 | 落点 |
|---|---|---|
| D5 | 共享核心包 `ComfyUI-Mie-Core` | §4.4 |
| D6 | 同步策略 = **`git merge upstream/main`**（不用 rebase） | §10 R1 |
| D7 | 产品层 **v1 包含站内剧本生成** | §6.2 |
| D8 | 部署拓扑 **v1 同机** | §6.1 |
| D9 | MieNodes 旧资产节点**弃用 + 冻结** | §11 Q-G |
| D10 | 引擎包**不发 Comfy Registry** | §11 Q-E |

### 1.3 第三版新增（第二轮评审后）

| # | 决策 | 落点 |
|---|---|---|
| **D11** | **生成器全族（本体 + prompts + schema + critic）进 Core**；MieNodes 只留节点壳 | §7.1 / §7.2 |
| **D12** | **Core 是纯 pip 包，不是 ComfyUI 自定义节点包**；ComfyUI 的 python 环境需 `pip install -e` | §4.4 |
| **D13** | **`services/llm.py` 拆分**：纯客户端进 Core，26 个节点类留 MieNodes | §7.1 |
| **D14** | **删除 `services/llm.py` 的三个死重依赖**（numpy / cv2 / torch） | §7.1 |
| **D15** | **P0 缓存修复走三条路**：配置面（strict 已是 widget）+ `st_ino` 入 cache key + 全量清单 | §2.5 |
| **D16** | **写协议的 CAS 基准用内容哈希**，不用 mtime | §8.2 |

### 1.4 本文给出的评估结论

| 问题 | 结论 |
|---|---|
| 上游更新频率？ | **事实停更**：最后 push `2026-10-14`（22 天前），此后 0 commit、0 issue 回复。§2.1 |
| fork 还是"自己写全新插件"？ | **fork**。§2.3 |
| 为什么不做一个 50 行的 wrapper 包？ | wrapper 改不动**上游内部**：Windows 缺陷在生产线代码、Node 24 在测试 harness、Editor 行为在上游 JS。§2.4 |
| Scene Prompts 预览今天能用吗？ | **不能**。Editor 读 `plan_json` **widget**，不读 socket。§5.1 |
| 改名会不会报错？ | **比报错更糟：静默覆盖**。§3.1 |
| mtime 相关失败只是测试问题吗？ | **不是**。**缓存守卫一旦把某个 (size, mtime) 组合认定为内容身份，就永不复检** ⇒ 静默错渲染。**P0**。§2.5 |
| 产品层能自己生成剧本吗？ | **能，但必须把生成器本体搬进 Core**（否则 MieNodes 的 `nodes/llm/__init__` 会拖起 ComfyUI 运行时）。§7.2 |

---

## 2. 现状事实

### 2.1 上游状态：事实停更（2026-11-05 实测）

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

### 2.2 上游规模

| 项 | 数量 |
|---|---|
| Python | 107,696 LOC / 252 文件 |
| 前端 `web/` | 40,585 LOC / 72 文件 |
| 测试 | 532 unittest + 103 `.mjs`（65k LOC） |
| 节点类型 | **87** |
| `chain_nodes.py` | **31,538 行**（顶层生产代码 29.3%） |
| CI | **无** |

### 2.3 我们侧规模与耦合性质

| 项 | 数量 |
|---|---|
| MieNodes 全仓 Python | 102,329 LOC |
| 节点类型 | 306 |
| `services/llm.py` | 2,655 LOC / **50 个类**，其中 **26 个带 `INPUT_TYPES`**（即 ComfyUI 节点类） |

**耦合是数据耦合，不是代码耦合**：全仓没有一处 import 上游代码，
接口只有一个字符串 `plan_json` → 上游 `plan_json_input` socket。

**两处影子实现必须合并**：

1. 上游 `chain_nodes.py:7362 _normalize_plan`（帧网格 / seed 值域 / context_length 集合 /
   continuation_mode 跨字段约束）；
2. 我们的 `validate_plan`（`nodes/llm/minimax_h3_loop_prompts.py:1775`，prompt 三段/六段 schema）。

### 2.4 为什么不是"wrapper 包"

一个 50 行的 wrapper 能解决改名，但不能碰真正要改的三样：

| 要修的东西 | 在哪 | wrapper 能碰吗 |
|---|---|---|
| Windows 缺陷（symlink 权限 / mtime / 路径） | 上游**生产线代码** | ❌ |
| Node 24 测试 harness 不兼容 | 上游 **`tests/_*.mjs`** | ❌ |
| Scene Prompt Editor 读 widget 不读 socket | 上游 **`web/*.js`** | ❌ |

**逼着 fork 的是"必须改上游内部"，不是改名。**

### 2.5 P0：缓存守卫把 (size, mtime) 组合当成内容身份（**第三版已修正机制描述**）

**第二轮评审纠正了第二版的机制误读，纠正后成立，且更精确。**

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

**调用方传的 `expected_hash` 来自装配记录**（实测三处调用全部是
`segment.get("checkpoint_sha256", "")`，`chain_nodes.py:25395 / 25586 / 25614`），
而 `entry["sha256"]` 是**上次真正算过的文件哈希**。

**所以条件 ① 不是"记录值 vs 记录值"的同源恒真，而是"上次实测 == 调用方期望"。**
第二版把它读成恒真是错的——**照那个读法去修（删冗余条件）会什么都没修好。**

**真正的门是：**

> 一旦某个 `(size, mtime_ns)` 组合被实测通过过，它就成了**该路径的内容身份**。
> 此后内容变了而 size/mtime 不变 ⇒ 永不复检 ⇒ **静默用错内容渲染**。

**修复方向（三条，见 D15）：**

1. **配置面先上——上游已有逃生口没被利用。**
   `checkpoint_verification` **是一个 widget**（`chain_nodes.py:25166`
   `("checkpoint_verification": (["cached", "strict"], {...})`），
   默认 `"cached"`（`chain_nodes.py:25222`）。
   **strict 模式永不走缓存**（`mode == "cached"` 才查 entry）⇒ 关键路径可以立刻切 strict。
   这是**今天就能做的配置面修复**，不需要改代码。
2. **cache key 补 `st_ino` / `st_dev`**（NTFS file index）。
   `os.replace` 原子替换后 file ID 必变 ⇒ 比 mtime 硬得多。
3. **清单补全**（第二版说"~18 处"，**实测错了**）：

   | 文件 | 生产代码中 `st_mtime_ns`/`getmtime` 处数 |
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
   | **合计** | **39 处 / 13 个文件** |

   第二版漏掉了 `storage_inventory.py`（7 处）、`checkpoint_manager.py`、
   `chain_layout_conversion.py`、`png_video_export.py`、
   `chapter_snapshot_retirement.py`、`run_manager.py`、`working_branch_cleanup.py`、
   `processing_checkpoint_delete.py`。

4. **多挖一条线：SMB/UNC 的 mtime 粒度。**
   本项目的角色资产源头在 `\\192.168.31.150\k\...` 共享上。
   SMB 协商出的 mtime 粒度可能粗到 1–2 秒 ⇒
   **"同秒内、同 size、不同内容"在 UNC 路径上比 NTFS 本地更易命中陈旧缓存**。
   Phase 1 必须回答：**被缓存守卫的文件（参考图 / 音色样本 / checkpoint）
   有没有可能位于 UNC 路径**；若有 ⇒ 分级策略：
   **本地 = ino+size+mtime，远程 = 强制 strict**。

> 结论不变但更硬：**测试红是看得见的，这个是看不见的错片**。
> Phase 1 必须把它当 **P0 生产缺陷**，不是"让测试变绿"的附带工作。

---

## 3. 命名与注册（D4）

### 3.1 为什么必须改名

```python
# ComfyUI/nodes.py:2305-2308
for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
    if name not in ignore:
        NODE_CLASS_MAPPINGS[name] = node_cls        # ← 静默覆盖，无告警
```

`/object_info` 以类型名为键，只有**一份**实现会被暴露。
两个包同装 ⇒ 谁后加载谁赢，**无错误无警告** ⇒ 工作流执行的是另一份实现。

### 3.2 命名方案

沿用 MieNodes 既有约定（`core/utils.py:12` `LOGO_SUFFIX = "|Mie"`），
引擎包用独立后缀：

```python
# ComfyUI-Mie-H3-Engine/core/utils.py
LOGO_SUFFIX = "|MieH3"
```

| 上游类型 | 引擎包类型 |
|---|---|
| `MiniMaxH3ChainPlanModern` | `MiniMaxH3ChainPlanModern\|MieH3` |
| `MiniMaxH3ChainScenePromptEditor` | `MiniMaxH3ChainScenePromptEditor\|MieH3` |
| …（全部 87 个） | …全部加后缀 |

显示名同步加后缀，并遵循既有"中文 · English"惯例。
我们的新节点独立命名：`MieH3ScriptToPlan`、`MieH3LoadScriptAsset`、`MieH3LoadAssetRoster`。

**为什么用后缀不用前缀**：保留 `MiniMaxH3` 前缀让 87 个类型的批量替换不易漏；
节点面板是子串搜索，搜 `MieH3` 也能命中。

### 3.3 存量工作流迁移（Q-B：**A**）

**A（纯改名 + 迁移脚本）**。脚本 `tools/migrate_workflows.py` 要求：

- [ ] 处理 **subgraph 嵌套**的 `type` 字段
- [ ] **幂等**
- [ ] 支持**按目录批量**（`docs/qa/` 被 gitignore，局域网另有一批副本）
- [ ] 只改 `type`，**不碰 `widgets_values`**

### 3.4 改名表必须自动生成

`tools/node_rename_map.json` 由脚本从各模块 `NODE_CLASS_MAPPINGS` **自动生成**，
不允许手写（87 条手写必漏）。生成器与迁移脚本共用同一张表。

**CI 断言**：`len(rename_map) == len(引擎包注册类型数)`，
且 `set(rename_map.values()) == set(引擎包注册类型名)`。

### 3.5 改名范围包含测试与基线

| 目标 | 处理 |
|---|---|
| `tests/**`（golden / portability / POV 等引用旧名的测试） | 一并改名 |
| `tests/_node_contracts_baseline.json` | 随改名更新，并随 D9 弃用标记更新 |
| `docs/examples/**`、`workflows/**`（本仓） | 一并改名 |
| `example_workflows/**`（上游自带 22 个 JSON） | 一并改名 |

---

## 4. 架构（D1 / D2 / D3 / D5 / D12）

### 4.1 四层

```
┌─ 产品层：短剧平台（新增，独立 FastAPI 进程）──────────────────────┐
│  浏览器前端 + REST 路由（剧本 CRUD / 站内生成 / 预览 / critic /    │
│  渲染队列 / 产物）                                                │
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
│  其它 20+ 提示词生成器与工具节点                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 依赖方向

| 边 | 方向 | 理由 |
|---|---|---|
| 引擎 → Core | 引擎依赖核心 | 资产节点与 plan builder 用 Core 的契约与库读写 |
| MieNodes → Core | MieNodes 依赖核心 | 节点壳调 Core 的生成逻辑与 LLM 客户端 |
| 产品层 → Core | 产品层依赖核心 | 剧本 CRUD、生成、critic、plan 编译全用 Core |
| Core → 任何包 | **无** | Core 不依赖引擎、不依赖 MieNodes、不依赖产品层 |

**契约面的依赖已核实**（第二轮评审指出第二版口径不一，此处统一）：

`nodes/llm/storyboard_asset_generator.py` 对资产层的 import 是
**3 个模块、5 条 import 行**：

| 行 | 目标模块 |
|---|---|
| `:25` | `..assets.storyboard_contract` |
| `:176` | `..assets.storyboard_review` |
| `:198` | `..assets.storyboard_critic` |
| `:297` | `..assets.storyboard_contract` |
| `:308` | `..assets.storyboard_contract` |

（第二版写"四处"不准确——按模块数是 3 个，按 import 行数是 5 条。）

### 4.3 契约只有一份实现

> **一份代码，多处调用。**
> 不是"产品层通过 HTTP 取 plan_json"——那只解决取值，解决不了"编译逻辑分叉"。

- 编译器（剧本资产 → `plan_json`）在 **Core**，只有一份；
- 产品层 import 它编译，把结果 POST 给引擎；
- 引擎的 plan builder 也 import 它——**同一段代码**；
- 引擎的 PromptServer 预览路由（§5.3）也调用它——**同一段代码**；
- 引擎的 `MieH3ScriptToPlan` 节点壳也调用它——**同一段代码**。

### 4.4 共享核心包 `ComfyUI-Mie-Core`（D5 / D12）

**形态（第三版修正）：Core 是纯 pip 包，不是 ComfyUI 自定义节点包。**

第二轮评审指出："既是自定义节点包（Manager 可装）"这个形态**解决不了问题**——
Manager 的安装方式是**复制进 `custom_nodes/`**，目录名带连字符，
**不会**进 site-packages ⇒ MieNodes / 引擎照样 `import mie_core` 失败。
而且 Core 没有任何节点，"可被 Manager 安装"没有收益。

**因此：**

- [ ] Core 仓是一个**标准 Python 包**（`pyproject.toml` 声明包布局 `mie_core/`），
      **不是** ComfyUI 自定义节点包
- [ ] Core 仓**同时**放在 `custom_nodes/` 目录下（蹭既有的**局域网目录同步**机制，
      省掉一套新的同步通道）
- [ ] **ComfyUI 的 python 环境里要 `pip install -e <Core 目录>`**——本机、局域网各一次
- [ ] 产品层的 venv 里 `pip install -e <Core 目录>`——同样各一次
- [ ] 部署脚本把这三步（ComfyUI 环境 ×2 机 + 产品层 venv ×2 机）作为**一次性安装动作**

**内容清单：**

| 模块 | 当前路径 | LOC | 去向 |
|---|---|---|---|
| LLM 纯客户端 + 连接器 | `services/llm.py`（**拆分**，见 D13） | 2,655 | `core/llm/client.py` |
| LLM 预算 | `nodes/llm/invoke_budget.py` | — | `core/llm/budget.py` |
| 剧本契约 | `nodes/assets/storyboard_contract.py` | 707 | `core/script/contract.py` |
| critic 确定性半 | `nodes/assets/storyboard_critic.py` | 484 | `core/script/critic.py` |
| critic 模型半 | `nodes/llm/storyboard_critic.py` | 197 | `core/script/critic_llm.py` |
| 资产独立审阅 | `nodes/assets/storyboard_review.py` | 41 | `core/script/review.py` |
| **生成器本体** | `nodes/llm/storyboard_asset_generator.py` | 343 | `core/script/generator.py` |
| **生成器 prompt/白名单** | `nodes/llm/minimax_h3_storyboard_prompts.py` | 791 | `core/script/storyboard_prompts.py` |
| **prompt 家族 + schema** | `nodes/llm/prompts/h3_storyboard/*` | — | `core/script/prompts/` |
| 资产库读写 | `nodes/assets/library.py` | 1,284 | `core/assets/library.py` |
| 剧本资产读写 | `nodes/assets/storyboard.py` | 1,224 | `core/assets/script_asset.py` |

**不搬 `core/utils.py`**（第二轮评审意见）：里面是 `|Mie` 后缀这类
**MieNodes 专属品牌逻辑**，放 Core 语义别扭；引擎有自己的 `|MieH3` utils。
Core 只留**真正通用**的部分；通用工具在 Core 内新建一个不带品牌的模块。

**时序（不双线重构）：**

- Phase 0–3：引擎 = 纯 fork，**不碰资产层** → 无耦合问题
- **Phase 4 开头：一次性抽出 Core**，三仓在同一变更窗口内切换
- Phase 5：产品层 `pip install`

**"同一个 commit"做不到（第二轮评审纠正）**——跨仓没有同一个 commit。
改成：**同一变更窗口**——Core 先建并发布（可安装），MieNodes 与引擎
**各一个 commit**，部署脚本一次性切换 + 双机各装一次。

---

## 5. Scene Prompts 对接（D3）

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

**编译留在 Python，前端只 fetch**（否则 R4 的唯一实现会在前端破功）。

```
POST /mieh3/plan_preview
  请求：{"script_ref": "<剧本资产 id 或资产库根目录下的相对路径>"}
  响应：{"plan_json": "<字符串>", "plan_sha256": "<前 8 位>",
         "source_mtime_ns": 123, "source_etag": "..."}
  行为：后端调用 Core 的同一个编译器，返回编译产物 + 内容指纹 + 源新鲜度标记
```

**第二轮评审补充的两条，本版采纳：**

1. **响应带 `plan_sha256` 前 8 位**，编辑器横幅显示"当前预览 = xxxx1234"，
   与产品侧一望即知是否同版。
2. **`script_ref` 必须限定在资产库根目录下**——否则 LAN 上任何人都能
   让 ComfyUI 进程读任意路径（**路径穿越**）。实现要求：
   - 只接受**资产库根目录下的相对路径**或**资产 id**；
   - 解析后 `os.path.realpath` 必须仍在根目录内（`commonpath` 检查）；
   - 拒绝符号链接逃逸。

**前端 provider 约定：**

```javascript
node.mieH3PlanProvider = () => ({
    fetchPlan: () => fetch("/mieh3/plan_preview", {method: "POST", body: ...})
});
```

引擎包前端逻辑：socket 已连接 → 找生产节点 → 有 provider 就 fetch →
否则回落 widget 并显示**明确横幅**。

### 5.4 Plan Studio 的处置

保留（对纯 widget 驱动的工作流仍有价值），但 socket 模式下**不作编辑入口**，
顶部加同样横幅，明确"外部 plan 接管，本面板只读"。

### 5.5 新鲜度

- 响应带 `source_mtime_ns` / `source_etag` / `plan_sha256`；
- 前端记录，每次打开编辑器**必校验**（不静默显示旧内容）；
- 源变更 → 显示 **"预览已过期，点击刷新"**；
- 提供刷新按钮；不做轮询。

### 5.6 验收

- [ ] 接上 `plan_json_input` 后，Scene Prompt Editor 显示的是**我们的**分镜
- [ ] Editor 内编辑不改变执行结果，界面明示只读
- [ ] 产品层改剧本资产后，ComfyUI 侧出现"预览已过期"提示，刷新后显示新内容
- [ ] 横幅显示 `plan_sha256` 前 8 位，与产品侧一致
- [ ] `script_ref` 越权路径被拒绝（**安全测试**）
- [ ] 未接外部输入时，Plan Studio 行为与上游完全一致（回归）

---

## 6. 产品层：短剧平台（D2 / D7 / D8）

### 6.1 进程模型、导入机制与部署拓扑

- 独立 FastAPI 进程（`uvicorn`），**不挂在 ComfyUI 的 PromptServer 上**（D2）。
- **导入机制**：产品层 venv `pip install -e <Core 目录>`，`import mie_core`。
  不做 sys.path 注入，不要求产品层机器 clone 引擎仓。§4.4。
- **依赖体量（第二轮评审纠正，见 D14）**：
  `services/llm.py` 顶层 `import numpy, cv2, torch`——**实测三个都是死 import**
  （AST 审计 + 原始 grep 双重确认：`np` / `cv2` / `torch` 三个名字在全文
  **各只出现 1 次**，即只有 import 行本身）。
  ⇒ **直接删除**，不需要惰性导入技巧。
  ⇒ Core 的 LLM 模块依赖只剩 **`requests` + stdlib**，
  产品层 venv 不再需要 torch / numpy / cv2 这几个数 GB 级依赖。
- **部署拓扑（D8）：v1 同机。** 角色立绘与音色样本必须落到 ComfyUI 的 `input/`，
  而 `/upload/image` 名义上只收图片、不收音频 ⇒ 同机直接写盘最省事。
  **跨机是 v2 的问题**（需要一条资产同步通道），v1 明确不做。

### 6.2 路由清单（首版）

| 方法 | 路径 | 作用 |
|---|---|---|
| GET/POST | `/api/projects` | 剧集/项目 CRUD |
| GET/POST | `/api/projects/{id}/scripts` | 剧本资产 CRUD（source of truth） |
| **POST** | **`/api/scripts/{id}/generate`** | **站内剧本生成（concept/revise）**（D7） |
| GET | `/api/scripts/{id}/preview` | 分镜预览数据 |
| GET | `/api/scripts/{id}/plan` | 编译出的 `plan_json`（调试用） |
| POST | `/api/scripts/{id}/critic` | 连续性 critic（**两半**，§7.3） |
| GET/POST | `/api/assets` | 角色/场景资产 |
| POST | `/api/render/submit` | 提交渲染 |
| GET | `/api/render/{job}` | 轮询任务状态 |
| GET | `/api/render/{job}/artifacts` | 产物 |
| GET | `/app` | 前端 SPA |

**站内生成为什么必须保留（D7）**：既有 concept/revise 模式 + 报错教学
就是为了让人**填得动**；砍掉只留结构化表单是**体验退步**。
**它的可行性依赖 D11**——生成器本体必须是 Core 里的纯 Python（§7.2）。

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

**复用而非重写**：结构直接沿用 Core 的 `core/script/contract.py`（v2 契约，707 行）
与 `to_runtime()`。

### 6.4 前端（首版）

1. **剧本**：结构化表单 + 站内生成（concept/revise），右侧实时显示编译后的每镜 prompt；
2. **分镜预览**：逐镜卡片；
3. **渲染**：提交、进度、产物回放。

**首版不做**：视频剪辑器、时间线、多轨混音、字幕编辑器。

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
| **`nodes/llm/prompts/h3_storyboard/*`** | — | **prompt 家族 + `asset.schema.json`**（D11） |
| `nodes/assets/library.py` | 1,284 | 资产库（纯 stdlib） |
| `nodes/assets/storyboard.py` | 1,224 | 剧本资产读写 |

**不搬 `core/utils.py`**：`|Mie` 是品牌专属逻辑（第二轮评审意见）。

### 7.2 留 MieNodes（全是**壳**）

| 来源 | LOC | 内容 |
|---|---|---|
| `nodes/llm/minimax_h3_storyboard_generator.py` | 408 | **生成器节点壳** |
| `nodes/llm/minimax_h3_loop_prompt_generator.py` | 5,503 | loop 生成器（**LLM 连接器是注入的**，见下） |
| `services/llm.py` 的 26 个节点类 | — | `Set*LLMServiceConnector` 等 |
| `nodes/assets/nodes.py` | 1,151 | 资产节点壳 |
| `nodes/assets/server.py` | 277 | aiohttp 路由（写路由按 D9 降级只读） |
| `nodes/llm/prompts/h3_loop/*` | — | H3 loop prompt 家族 |

**为什么生成器本体必须搬（D11，第二轮评审的核心发现）：**

1. `nodes/llm/__init__.py` 做 `from .xxx import *`（实测 20+ 行）⇒
   **import 任何一个 `nodes.llm.*` 模块都会拖起整个包**，其中包含需要
   ComfyUI 运行时的模块（`minimax_h3_loop_prompt_generator.py:115-118`
   的 shim 注释明说了这点，它正是因此才按**文件路径**加载 prompts/critic）。
   ⇒ 产品层**无法干净地 import MieNodes 的包**。
2. 生成器**本体**已经是纯 Python（import 只有 `..assets.*` 契约/审阅/critic
   + **注入的 llm 服务**），**408 行的节点壳**才是 ComfyUI 侧——
   **本体/壳早已分层**，搬本体是顺着现有结构走。
3. **critic 模型半的口粮在同一个目录家族里**：
   `nodes/llm/storyboard_critic.py:29`
   `PROMPT_DIR = Path(__file__).parent / "prompts" / "h3_storyboard"`
   ⇒ critic prompts、生成器 prompts、`asset.schema.json` 是**同一个目录的家族**。
   **代码走、口粮不留 ⇒ 搬过去读不到指令。**
   第二版曾想把 critic 模型半搬 Core 却把 prompt 家族留 MieNodes，
   那是直接打架——本版一并解决。

**R6 的表述相应改为**：
不是"MieNodes 管 LLM 生成"，而是——**生成逻辑在 Core，MieNodes 与产品层都是壳**。

### 7.3 critic 的两半（口径统一）

| 半 | 文件 | 需要 LLM？ | 去向 |
|---|---|---|---|
| 确定性半 | `nodes/assets/storyboard_critic.py` (484) | 否 | **Core** |
| 模型半 | `nodes/llm/storyboard_critic.py` (197) | 是（经 Core 的 LLM 抽象） | **Core** |

**两半同住 Core**（D11 消解了原 Q-H）。
`/api/scripts/{id}/critic` 路由需要**同时**用两半：产品层一次 import 拿到完整 critic。

### 7.4 搬进引擎包的 ComfyUI 外壳（薄）

| 新节点 | 作用 |
|---|---|
| `MieH3LoadAssetRoster` | 从 Core 资产库读角色/场景 → roster |
| `MieH3LoadScriptAsset` | 从 Core 读剧本资产 |
| `MieH3ScriptToPlan` | Core 的零 LLM 编译器 → `plan_json`（**并实现 §5.3 的 provider**） |
| `MieH3SaveAsset` / `MieH3SaveScriptAsset` | 写回 Core |

### 7.5 与上游 `project_assets.py` 的边界（不变）

上游管"这次渲染用了哪些媒体文件"（run 作用域媒体卡片），
Core 管"这个角色长什么样、声音什么样、在哪几镜出现"（跨项目具名实体）。
**正交，不合并。** 产品层负责把 Core 的角色/场景翻译成上游 Carousel 的引用关系。

---

## 8. 实施阶段

### Phase 0 — 建 fork（0.5 天）

- [ ] `git clone` 上游 → 新仓 `ComfyUI-Mie-H3-Engine`，base commit `43afeff`
- [ ] 保留 `origin` 指向上游（D6 用 merge 不用 rebase）
- [ ] `LICENSE`（GPL-3.0）+ `THIRD_PARTY_NOTICES.md` + `docs/FEATURE_TRACEABILITY.md` 全部保留
- [ ] 新增 `MODIFICATIONS.md`：**逐条台账**（改了什么/为什么/影响哪些文件）
- [ ] 保留作者署名，**不改上游版权声明**
- [ ] `version` = **`0.7.3+mie.1`**（Q-I）
- [ ] 生成 `tools/node_rename_map.json`（§3.4，自动生成）

**判据**：`git log --oneline origin/main` 仍可解析；`LICENSE` 逐字未改。

### Phase 1 — Windows 正确性修复（**P0**）（3–5 天，**下限不是上限**）

- [ ] **P0-a（配置面，今天就能做）**：把关键路径的 `checkpoint_verification`
      设为 **`strict`**（widget 已有，`chain_nodes.py:25166`，默认 `cached`），
      使其永不走缓存（§2.5 修复方向 1）
- [ ] **P0-b（根治）**：cache key 补 `st_ino` / `st_dev`（§2.5 修复方向 2）
- [ ] **P0-c（排查）**：按 **39 处 / 13 文件**的完整清单逐个判定
      "内容变了但 size/mtime 未变"是否**都**成立（§2.5 修复方向 3）——
      特别查 `storage_inventory.py`（7 处，全包最多）
- [ ] **P0-d（新线）**：UNC/SMB 的 mtime 粒度——确认被缓存守卫的文件
      （参考图 / 音色样本 / checkpoint）**是否可能位于 UNC 路径**；
      若是 ⇒ **本地 = ino+size+mtime，远程 = 强制 strict**
- [ ] Windows 无符号链接/硬链接权限（WinError 1314）→ 代码侧对权限不足降级而非崩
- [ ] 修 Node 24 的 `vm.runInContext` ESM 兼容（10 个 `.mjs`）
- [ ] 修剩余真 Windows 缺陷
- [ ] CI 处置见 Q-F′

**判据**：`SUMMARY: N passed; 0 failed`（Windows）；
**外加**：构造"内容变、size 与 mtime 不变"的用例，断言缓存**不**命中
（ino/dev 改了就命中不了 ⇒ 走重算）。

### Phase 2 — 命名与注册（1 天）

- [ ] 引擎包注册统一走 `add_suffix()`；显示名同步加后缀
- [ ] `tools/migrate_workflows.py`：subgraph 嵌套、幂等、目录批量（§3.3）
- [ ] 改名范围含 `tests/**`、`tests/_node_contracts_baseline.json`、
      `docs/examples/**`、`workflows/**`、上游 `example_workflows/**`（§3.5）
- [ ] CI 断言改名表条目数与注册类型数一致

**判据**：引擎包与上游原包**可同时安装**，`/object_info` 类型名无重叠。

### Phase 3 — plan 契约合并与 Scene Prompts 预览（2–3 天）

- [ ] 我们的 `validate_plan` 与上游 `_normalize_plan` 合并为**一份**（Core）
- [ ] 引擎包新增 `POST /mieh3/plan_preview`（§5.3，调用 Core 的编译器）
      —— 含 `plan_sha256` 前 8 位 + **`script_ref` 路径限制**
- [ ] 引擎包前端实现 `mieH3PlanProvider` 约定 + fetch 预览
- [ ] Scene Prompt Editor：provider 优先，缺失时回落 widget + 明确横幅
- [ ] Plan Studio：socket 模式下只读 + 同样横幅
- [ ] `MieH3ScriptToPlan` 实现 provider 访问器

**判据**：§5.6 六条验收全过。

### Phase 4 — 抽 Core + 搬资产层（4–5 天，**含安装，不只是搬文件**）

- [ ] 建 `ComfyUI-Mie-Core` 仓（**纯 pip 包**，D12）
- [ ] 按 §7.1 搬 11 项（含生成器全族，D11；LLM 拆分，D13）
- [ ] **删掉 `services/llm.py` 的三个死重依赖**（numpy / cv2 / torch，D14）
- [ ] 按 §3.5 迁移 `tests/**`、SYNC PAIR 守护测试、critic 实测测试
- [ ] 引擎包加 §7.4 的薄壳节点
- [ ] 端到端：剧本资产 → `MieH3ScriptToPlan` → `plan_json` →
      `MiniMaxH3ChainPlanModern|MieH3` → 渲染
- [ ] **零 LLM 渲染路径保持字节级可复现**
- [ ] **部署**：ComfyUI 环境 `pip install -e Core`（本机 + 局域网）；
      产品层 venv 同样安装
- [ ] **变更窗口**：Core 先建可装 → MieNodes 与引擎**各一个 commit** → 部署脚本一次性切换

**判据**：
- 同剧本 + 同资产 + 同 seed ⇒ 同 `plan_json` 字节
- 真机跑通一条完整短剧（≥3 镜，含台词与参考图）
- **live combo matrix 改跑后缀名节点**（`MIE_LIVE_E2E=1`）
- Core 安装后，**ComfyUI 侧与产品层侧都能 `import mie_core`**

> 第二轮评审提醒：3–4 天的估时**只算了搬文件**。
> 本版把"7 处 import 重指 + 双机安装 + 死依赖清理 + 测试迁移"计入，给 **4–5 天**。

### Phase 5 — 产品层后端（3–5 天）

- [ ] FastAPI 骨架 + §6.2 路由
- [ ] `pip install -e Core`，`import mie_core`（§6.1）
- [ ] 资产库读写**进程内直接调用**（Core 的库读写函数），不走 HTTP
- [ ] 站内剧本生成路由（concept/revise，调用 **Core 的生成器本体**）
- [ ] critic 路由：两半（§7.3）
- [ ] 渲染任务队列：提交 `/prompt`，轮询 `/history`，产物落盘 + 回写
- [ ] **渲染链路六坑**（§8.1）作为设计输入
- [ ] **资产目录写协议**（§8.2）

**判据**：产品层在 ComfyUI **未启动**时，剧本编辑/**生成**/预览/编译全部可用；
ComfyUI 启动后，提交渲染 → 出片 → 产物回显一条走通。

### Phase 6 — 产品层前端（3–5 天）

- [ ] 三页签 SPA：剧本 / 分镜预览 / 渲染
- [ ] 剧本编辑器：字段级覆盖
- [ ] 站内生成：concept/revise 两模式 + 报错教学
- [ ] 分镜预览：逐镜卡片，与引擎侧 Editor 显示一致（含 `plan_sha256` 对照）
- [ ] 渲染页：提交、进度、产物回放

**判据**：一个不了解 ComfyUI 的用户，只用网页完成
**写剧本 → 生成/修改 → 出片**，全程不打开 ComfyUI 界面。

### 8.1 渲染链路六坑（Phase 5 的设计输入）

全部是本项目真机踩过的：

| # | 坑 | 产品层的应对 |
|---|---|---|
| 1 | `/history` **重启即清空** | 任务状态**必须落盘可恢复**；`output/h3_chains/<run>/` 取证模式是既定事实 |
| 2 | **RECOVERY 节点在全新运行必炸** | 产品层模板去掉 RECOVERY 节点，或改 `run_name` |
| 3 | **手改编译产物/资产的节点缓存坑** | 同 id 重提复用旧 bundle ⇒ **每次渲染自带全新 `run_name` + 新 node id** |
| 4 | **60GB 显存下单任务排队** | 渲染队列**串行化** |
| **5** | **双实例 / 谎报的 restart**（第二轮评审新增） | launcher 的 restart 可谎报成功，旧进程继续占 8188 ⇒ 提交进 A 实例、轮询 B 实例 ⇒ **产物错配**。应对：**提交前探 `/system_stats` 取实例指纹**（pid / 启动时间），轮询时比对，不一致即拒 |
| **6** | **`/history` 条目存在 ≠ 成功**（第二轮评审新增） | 必须按条目的 **`status` 字段**判失败，不能只看"有条目" |

### 8.2 资产目录写协议（第二轮评审补充）

**已验证的事实：**

```
nodes/assets/library.py:298    fd, tmp_name = tempfile.mkstemp(...)
nodes/assets/library.py:583,585 os.replace(tmp, final) / os.replace(tmp_json, json_path)
nodes/assets/storyboard.py:799  tmp_fd, tmp_name = mkstemp(...)
nodes/assets/storyboard.py:806  os.replace(tmp_path, json_path)
```

**原子写有，但没有任何锁**（无 `fcntl` / `msvcrt` / `filelock`）。
原子写只保证**不写坏**，不保证**不互相覆盖**（last-writer-wins 静默丢更新）。
写者有**三个**：产品层 + MieNodes `server.py`（留在 MieNodes！）+ 引擎节点。

**协议：**

- [ ] 所有写入**统一走 Core 的一个写入口**，该入口持有 advisory lockfile
- [ ] **CAS 的比较基准用「读时记录的内容哈希」，不用 mtime**（D16）——
      §2.5 已经证明 mtime 在 NTFS/SMB 上都不可靠；资产 JSON 都是小文件，
      直接比较内容哈希最稳
- [ ] 冲突语义：**409 + 返回对方版本**，由前端裁决，**不静默覆盖**
- [ ] MieNodes 旧 `server.py` 的写路由按 D9 **降级为只读**
- [ ] 镜像同步（MIRROR-delete）与产品层写入并发的运营规则写进文档
- [ ] **不上 SQLite**（第二轮评审意见）：资产库是"少文件、人频、镜像同步"形态，
      MIRROR-delete 直接复制文件树，SQLite 的 wal/journal 在镜像复制时机不对时会损坏，
      **比 JSON 更脆**

---

## 9. 不做什么

- **不重写上游生成核心**（§2.4）。
- **不把引擎包和 MieNodes 合成一个包**（D1）。
- **不做视频剪辑 / 时间线 / 多轨混音 / 字幕编辑器**。
- **不在 ComfyUI 里做分镜图形编辑器**。分镜编辑在产品层；ComfyUI 侧只读预览。
- **不改已发布的 MieNodes 节点契约**。v1/v2/v3 类型名、输入顺序、输出顺序不动。
- **不合并上游 `project_assets` 与 Core 资产库**（§7.5）。
- **不改 `plan_json` 的顶层形状**。
- **v1 不做跨机部署**（D8）。
- **不把 Core 做成依赖 ComfyUI 的包**（否则产品层装不了）。
- **不把 `core/utils.py` 搬进 Core**（品牌专属）。
- **不把资产库改成 SQLite**（§8.2）。

---

## 10. 风险与缓解

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 上游复活，fork 落后 | 需同步 148k 树 | **D6：`git merge upstream/main`**，不用 rebase；原地修改外科手术化；`MODIFICATIONS.md` 逐条台账；季度同步 |
| R2 | 87 个类型批量改名漏改 | 节点注册不全，工作流红 | 改名表**自动生成** + CI 断言条目数相等 + **改名范围含 `tests/**` 与 baseline json** |
| R3 | Phase 1 的 P0 排查拖太久 | 阻塞后续 | P0 分四条（a 配置面 / b 根治 / c 清单 / d UNC），**P0-a 今天就能做**；P0 未完成不得进 Phase 3 |
| R4 | plan 编译逻辑分叉 | 预览与实际渲染不一致 | **一份代码多处调用**（§4.3） |
| R5 | FastAPI 与 ComfyUI 的状态竞争 | 产物错配 | `run_name` 唯一键；产物由 ComfyUI 写，产品层只索引；**实例指纹校验**（坑 5） |
| R6 | 概念边界混乱 | 实现走偏 | 边界改为：**逻辑在 Core，MieNodes 与产品层都是壳**（§7.2） |
| R7 | 引擎包 157k 行无人 review | 质量下滑 | 保留上游 65k 行测试 + 新增契约测试 + CI |
| R8 | 上游依赖 `comfy.ldm.minimax.model` 内部 API | 引擎不可用 | fork 后由我们承担；加"ComfyUI 版本兼容矩阵"检查 |
| R9 | 产品层 import GPL 核心 ⇒ 成为 GPL 衍生品 | 发布策略受限 | 与 Q-E 联动：内部 LAN 无碍；**公开发布产品层必须同发源码**。Core 与引擎均 GPL-3.0，产品层沿用 GPL-3.0 |
| R10 | 多写者丢更新 | 资产被静默覆盖 | Core 单一写入口 + advisory lock + **内容哈希 CAS**；409 + 返回对方版本；旧 `server.py` 写路由降级只读 |
| R11 | Phase 1 就已破"只新增文件"的约束 | R1 缓解强度下降 | **D6 已换策略**；原地修改外科手术化并逐条进 `MODIFICATIONS.md` |
| **R12**（新） | **UNC 上的 mtime 粒度更粗** → 陈旧缓存命中率更高 | 静默错渲染（比本地更易发生） | P0-d：确认被守卫文件是否可能在 UNC；远程强制 strict |
| **R13**（新） | **生成器搬到 Core 后，prompt 家族与代码分家** | critic/生成器读不到指令 | D11 一并搬；SYNC PAIR 守护测试改为**跨仓内容一致性断言**（§8 Phase 4） |
| **R14**（新） | **产品层 venv 装不下 torch/cv2** | Phase 5 第一天就撞 | D14 删死依赖 ⇒ Core 只依赖 `requests` |

---

## 11. 待第三方裁定的问题

| # | 问题 | 状态 / 倾向 |
|---|---|---|
| **Q-A'** | 共享核心形态：(a) 独立 pip 包 / (b) 留在 MieNodes 按路径加载 | **(a)**，并按第二轮补上 §4.4 的两条机械约束（ComfyUI 环境 `pip install -e` + 放 `custom_nodes/` 蹭 LAN 同步）；**"同一个 commit"改为"同一变更窗口"** |
| **Q-B** | 存量工作流迁移：A 还是 B | **A** |
| **Q-C** | 后缀 `\|MieH3` vs 前缀 | **`\|MieH3`** |
| **Q-D** | Scene Prompt Editor 完全只读 | **完全只读** + 新鲜度验收（§5.5） |
| **Q-E** | 引擎包要不要发 Comfy Registry | **不发**（D10） |
| **Q-F** | Windows 修复要不要回流上游 | **尝试回流、零依赖**；不依赖它 |
| **Q-G** | MieNodes 旧资产节点：删还是弃用 | **弃用 + 冻结**（D9） |
| **Q-F'** | **CI 与"不再推送"冲突** | **需裁定**。倾向：引擎/Core 两仓**私有化 + 允许 GitHub Actions**（纯代码无资产无密钥，GPL 本就要求源码可得；换来 Windows+Linux 双平台矩阵）。若"不再推送"的本意是任何代码都不上云 ⇒ 本地定时跑。**只有你能拍** |
| **Q-G'** | 部署拓扑 | **v1 同机已定**（D8）；跨机是 v2 |
| **Q-H** | critic 模型半放哪 | **已消解**：两半同住 Core（D11） |
| **Q-I** | `version = 0.7.3+mie.1` | **接受** |

---

## 12. 证据索引

### 上游（`C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-Context-Loop`）

| 事实 | 位置 |
|---|---|
| Production Plan 节点 | `chain_nodes.py:15352` |
| `plan_json_input` socket 声明 | `chain_nodes.py:15431-15445` |
| socket 覆盖 widget | `chain_nodes.py:15322-15326` |
| plan 归一化 / 校验 | `chain_nodes.py:7362`（per-shot `:7484+`） |
| `IS_CHANGED` 对普通编辑恒 False | `chain_nodes.py:15284-15305` |
| **缓存守卫本体** | `chain_nodes.py:24681-24713` |
| **`checkpoint_verification` 是 widget，默认 `cached`** | `chain_nodes.py:25166` / `:25222` |
| 守卫的三个调用点（`expected_hash` 来自 `segment["checkpoint_sha256"]`） | `chain_nodes.py:25395` / `:25586` / `:25614` |
| mtime 参与点 **39 处 / 13 文件** | 见 §2.5 表 |
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
| `nodes/llm/__init__.py` 拖起整包（`from .x import *` × 20+） | `nodes/llm/__init__.py` |
| LLM 连接器是**注入**的（loop generator 5,503 行不 import services.llm） | `nodes/llm/minimax_h3_loop_prompt_generator.py:4774`（socket 声明） |
| `services/llm.py` 50 类 / 26 个带 `INPUT_TYPES` / 零 `folder_paths` | `services/llm.py:1264`、`:2185` |
| **numpy / cv2 / torch 是死 import**（各只出现 1 次） | `services/llm.py:6-8` |
| `services/llm` 生产侧唯一 importer | `nodes/llm/invoke_budget.py`（+ `__init__.py:19`） |
| 生成器对资产层的 import：**3 模块 / 5 行** | `nodes/llm/storyboard_asset_generator.py:25/176/198/297/308` |
| **critic 模型半的 PROMPT_DIR 指向 prompt 家族** | `nodes/llm/storyboard_critic.py:29` |
| 原子写但无锁 | `nodes/assets/library.py:298/583/585`、`nodes/assets/storyboard.py:799/806` |
| file-path shim 机制（先例） | `nodes/llm/minimax_h3_loop_prompt_generator.py:104-125` |
| 我方 plan 校验器 | `nodes/llm/minimax_h3_loop_prompts.py:1775` |
| 节点契约基线 | `tests/_node_contracts_baseline.json` |
| live combo matrix | `tests/test_h3_loop_live_combo_matrix.py`（`MIE_LIVE_E2E=1`） |
| critic 实测测试（default-on） | commit `eb2e7e1` |

---

## 13. 变更记录

| 日期 | 修订 |
|---|---|
| 2026-11-05 | 初稿。D1–D4；上游停更实测；Editor 缺陷定位；plan provider 方案 |
| 2026-11-05 | **v2**：并入第一轮评审。补 Core 包（D5）、provider 改后端路由、mtime 提为 P0、R1 改 merge、写协议、四坑、Q-A 扩大 |
| 2026-11-05 | **v3**：并入第二轮评审。**修正 P0 机制描述**（不是恒真比较，是"上次实测 vs 调用方期望"）、**mtime 清单补到 39 处/13 文件**、**新增 UNC/SMB 线**、**生成器全族进 Core**（D11）、Core 改为纯 pip 包（D12）、LLM 拆分（D13）、**删三个死重依赖**（D14）、P0 分四条（D15）、CAS 基准改内容哈希（D16）、**新增第五/第六坑**、Phase 4 估时含安装、§7.3 口径统一 |

---

## 14. 第一轮评审意见处置台账

见 Git 历史中 v2 提交；要点：契约依赖方向矛盾（→ Core 包）、搬迁清单不全（→ §7.1/§7.2）、
provider 的 planJson 来源（→ 后端路由）、R4/Phase 5 矛盾（→ 一份代码多处调用）、
缺生成路由（→ D7）、缺"为什么不 wrapper"（→ §2.4）、
R1 在 Phase 1 就破约（→ D6 改 merge）、改名漏改含 tests（→ §3.5）、
双进程写同目录无锁（→ §8.2）、渲染四坑（→ §8.1）、CI 与不推送冲突（→ Q-F′）、
部署拓扑隐式（→ D8）、mtime 可能是生产问题（→ P0）。

## 15. 第二轮评审意见处置台账

| 评审意见 | 处置 | 落点 |
|---|---|---|
| **§一 P0 机制描述读错** | **完全采纳，纠正我方误读**。`expected_hash` 来自调用方 `segment["checkpoint_sha256"]`，`entry["sha256"]` 是上次实测值 ⇒ 条件 ① 是"上次实测 vs 调用方期望"，**不是恒真**。真正的门是"(size, mtime) 组合成为内容身份后永不复检"。**第二版若照那个读法去修会什么都没修好** | §2.5 |
| §一 修法 1：`verification: cached\|strict` 逃生口未被利用 | **采纳**。`checkpoint_verification` 是 widget（`:25166`，默认 `cached`）⇒ **P0-a 纯配置面，今天就能做** | §2.5 方向 1 / Phase 1 |
| §一 修法 2：cache key 补 `st_ino`/`st_dev` | **采纳**。`os.replace` 后 file ID 必变，比 mtime 硬 | §2.5 方向 2 / Phase 1 |
| §一 修法 3：清单不全 | **采纳**。我方实测 **39 处 / 13 文件**（评审说 31/11，我方数字以实测为准），补齐 `storage_inventory.py`(7)、`checkpoint_manager.py`、`chain_layout_conversion.py`、`png_video_export.py`、`chapter_snapshot_retirement.py`、`run_manager.py`、`working_branch_cleanup.py`、`processing_checkpoint_delete.py` | §2.5 表 / Phase 1 |
| §一 修法 4：UNC/SMB mtime 粒度 | **采纳**。新增 **P0-d**：确认被守卫文件是否可能在 UNC；本地 = ino+size+mtime，远程 = 强制 strict。**新增 R12** | §2.5 方向 4 / Phase 1 |
| **§二 生成器归属 vs Phase 5 判据矛盾** | **完全采纳**。`nodes/llm/__init__.py` 的 `from .x import *` 拖起整包（含 ComfyUI 运行时）⇒ 产品层无法干净 import MieNodes。**D11：生成器本体 + prompt 家族 + schema 全族进 Core**，MieNodes 只留 408 行节点壳 | §7.1 / §7.2 |
| §二 critic 模型半的 PROMPT_DIR 指向 prompt 家族 | **采纳**。"代码走、口粮不留" —— 一并搬，**Q-H 顺势消解**（两半同住 Core） | §7.1 / §7.3 |
| §二 R6 表述改为"逻辑在 Core，两边都是壳" | **采纳** | §7.2 / R6 |
| §二 SYNC PAIR 守护是同仓断言，搬后成跨仓 | **采纳**。守护测试降级为**跨仓内容一致性断言**；docs/ 同步副本一并处理 | Phase 4 / R13 |
| §三 Core 作为 ComfyUI 包解决不了 import | **采纳**。**D12：Core 是纯 pip 包**；放 `custom_nodes/` 蹭 LAN 同步；**ComfyUI 的 python 环境要 `pip install -e`**（双机各一次） | §4.4 |
| §三 "同一个 commit" 做不到 | **采纳**。改为**同一变更窗口**：Core 先建可装 → 两仓各一 commit → 部署脚本一次性切换 + 双机各装一次 | §4.4 / Phase 4 |
| §三 Phase 4 依赖面漏 cv2，torch/cv2 是重依赖 | **采纳并加强**。实测 **numpy / cv2 / torch 三个都是死 import**（各只出现 1 次）⇒ **直接删除**而非惰性导入 ⇒ Core 只依赖 `requests`。**D14** | §6.1 / D14 / R14 |
| §四-1 `core/utils.py` 是品牌专属 | **采纳**。**不搬 `core/utils.py`** | §4.4 / §9 |
| §四-2 预览路由是对的；加 `plan_sha256` 前 8 位 | **采纳** | §5.3 / §5.6 |
| §四-2 `script_ref` 必须限定在资产库根目录下 | **采纳**。**路径穿越风险**，加 `realpath` + `commonpath` 检查 + 拒绝符号链接逃逸，并列入验收 | §5.3 / §5.6 |
| §四-3 mtime 还往哪挖 | **采纳**。清单补全 + UNC 新线 + 三个修法 | §2.5 |
| §四-4 写协议 CAS + 单写入口够，别上 SQLite | **采纳**。SQLite 的 wal/journal 在镜像复制时机不对时会损坏，比 JSON 更脆 | §8.2 / §9 |
| §四-4 CAS 基准别用 mtime，用内容哈希 | **采纳**。**D16**（与 §2.5 的 mtime 不可靠论证自洽）；冲突语义 = 409 + 返回对方版本，前端裁决 | §8.2 |
| §四-5 第五坑：双实例 / 谎报 restart | **采纳**。提交前探 `/system_stats` 取实例指纹，不一致即拒 | §8.1 坑 5 / R5 |
| §四-5 第六小坑：`/history` 有条目 ≠ 成功 | **采纳**。按 `status` 字段判失败 | §8.1 坑 6 |
| §五 Q-A' (a) 带两个补充 | **采纳** | §4.4 |
| §五 Q-H 升级为"生成器全族进 Core" | **采纳**（D11） | §7.1 / §7.3 |
| §五 Q-F' 私有仓例外 | **倾向采纳，标为待裁定**——只有用户能拍 | §11 Q-F' |
| §五 Q-I `0.7.3+mie.1` | **采纳** | Phase 0 |
| §六 §7.3 三处口径不一 | **采纳**。本版统一为"两半同住 Core" | §7.3 |
| §六 Phase 4 缺"测试随模块走" | **采纳**。critic 实测（default-on）与 SYNC PAIR 归属写明；**CI 里要可关** | Phase 4 |
| §六 `services.llm` 迁移爆炸半径 | **采纳**。Phase 4 估时改 **4–5 天**并写明含"7 处 import 重指 + 双机安装 + 死依赖清理 + 测试迁移" | Phase 4 |
| §六 "四处" 实为 3 模块 5 行 | **采纳**。§4.2 已按模块/行数分列 | §4.2 |
