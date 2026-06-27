# MieLoop Expand ID 扁平化实施计划（方案 B + A）

> 目标：消除长跑循环时 `#453.0.0.453.0.0.453...` 式嵌套 ID，降低深层 expand 的不稳定风险。  
> 范围：**仅方案 B（Recurse 命名）+ 方案 A（Flat Prefix）**，不含 Chunk / Executor 内循环。  
> 首要约束：**已保存工作流零回归**——用户 JSON/API prompt 不需修改，行为与输出语义保持一致。

---

## 1. 背景与问题

### 1.1 现状

每轮 `MieLoopEnd` expand 会 clone 协议节点（BodyOut、End）和业务节点。Clone 时使用**模板节点 id**（如 `"453"`），ComfyUI `GraphBuilder` 再叠加父级 prefix，形成：

```text
453
└─ 453.0.0.453          ← 第 1 轮 expand 的 End
   └─ 453.0.0.453.0.0.453   ← 第 2 轮
      └─ …（每轮 + `.0.0.{template_end_id}`）
```

49 轮 SCAIL 类工作流会出现极长 execution id，日志难读，且深层 ephemeral 子图对外链解析、cache 压力更大。

### 1.2 参考实现

ComfyUI 官方 `while_loop_close`（`tests/execution/testing_nodes/testing-pack/flow_control.py`）对**循环关闭节点**使用固定 id `"Recurse"`，避免把关闭节点模板 id 重复嵌进路径。

MieNodes 已有类似先例：`__mie_loop_resume__`（Resume 节点固定内部 id + `override_display_id` 可选）。

---

## 2. 不变量（已保存工作流零回归契约）

以下条目为**硬性约束**，任何 PR 不得破坏。

### 2.1 用户可见 / 持久化层（不得改）

| 项目 | 要求 |
|------|------|
| 已保存 workflow JSON | 节点 id、连线、widgets 完全不变；**不要求用户重存或迁移** |
| 协议节点 INPUT/OUTPUT | `MieLoopStart/BodyIn/BodyOut/End` 的 socket 类型、名称、widget 字段不增删（hidden 除外） |
| `loop_ctx` schema | 保持 `version: 3`；字段名、语义不变 |
| `ctx.meta.body_in_id / body_out_id / end_id` | 始终为**原始模板 id**（如 `"453"`），expand 轮次不得写回带 `.` 的 clone id |
| `resume_loop_ctx` | 行为与 JSON 格式不变；resume 后续跑结果与改前一致 |
| `collect_loop_body()` | 仍基于**原模板图**做节点发现，不依赖 expand 图结构 |
| 外链输入 | expand 图内指向循环外的 link 仍为 `["470", 0]` 等**模板 id**，不得改为 ephemeral id |
| Finalize / Collect | 收集器 ref、Finalize 输出 batch 语义不变 |
| UI 节点编号 | 通过 `override_display_id` 继续显示模板 id（453 等） |

### 2.2 允许变化（仅运行时内部）

| 项目 | 说明 |
|------|------|
| ephemeral 节点 execution id | 由 `453.0.0.453...` 变为 `453.r5.__mie_loop_recurse_end__` 等 |
| 日志中的 `LoopEndExpand end_node_id` | 字符串变短 |
| `ctx.meta.expand_root` | **新增** optional 字段，旧 ctx 无此字段时自动回填，不影响已存 workflow |

### 2.3 显式不做

- 不改 `MieLoopStart` 的 `int_range_*` / `count` 计算逻辑  
- 不改 Chunk / `max_rounds_per_prompt`（留给后续计划）  
- 不修改 ComfyUI 引擎  
- 不强制用户升级 workflow 文件版本号  

---

## 3. 方案 B：Recurse 固定内部 id（先做，低风险）

### 3.1 改动点

**文件**：`nodes/loop/loop.py` → `_build_expand_graph_for_next_round()`

1. 增加常量（与 Resume 并列）：

```python
RECURSE_END_ID = "__mie_loop_recurse_end__"
RECURSE_BODY_OUT_ID = "__mie_loop_recurse_bodyout__"
```

2. 在 `build_node(oid)` 中，clone 协议节点时使用固定内部 id，并设置 display override：

```python
clone_id = oid
if oid == str(end_id):
    clone_id = RECURSE_END_ID
elif oid == str(body_out_id):
    clone_id = RECURSE_BODY_OUT_ID

new_node = graph.node(class_type, clone_id, **new_inputs)
new_node.set_override_display_id(oid)  # UI 仍显示 "453" / "450"
```

3. `built_nodes` 键、`is_loop_end_node` / `is_body_out_node` 判断改为基于 `oid`（模板 id），**不是** clone_id。

4. End 强依赖 BodyOut 的逻辑（约 1090–1094 行）仍通过 `build_node(str(body_out_id))` 解析，内部映射到 `RECURSE_BODY_OUT_ID`。

5. 碰撞检测：Resume 已用 `__mie_loop_resume__`；单轮 expand 图内 End/BodyOut 各 1 个，与现有 `test_build_expand_graph_resume_node_does_not_collide_with_end_id_1` 同类断言需更新为 Recurse id。

### 3.2 预期效果

| 轮次 | 改前 End id（示意） | 改后 End id（示意） |
|------|---------------------|---------------------|
| 1 | `453.0.0.453` | `453.0.0.__mie_loop_recurse_end__` |
| 2 | `453.0.0.453.0.0.453` | `453.0.0.__mie_loop_recurse_end__.0.0.__mie_loop_recurse_end__` |

路径仍随 `.0.0.` 变长，但**不再重复叠加以模板 End id 命名的段**，长度约减半，对齐 ComfyUI 官方 while 循环。

### 3.3 回归风险与缓解

| 风险 | 缓解 |
|------|------|
| 测试硬编码 `"453"` 作为 expand 图 End 节点 key | 更新断言：按 `class_type == MieLoopEnd` 查找；或断言 `override_display_id == "453"` |
| `_should_record_protocol_node_id` 误判 | Recurse id 含 `.` 的 current_node_id 仍不写入 meta（逻辑已满足） |
| BodyOut→End 强依赖链断裂 | 保留现有 `test_end_loop_ctx_from_current_cloned_bodyout` 并加 Recurse 命名断言 |

---

## 4. 方案 A：Flat Prefix（在 B 稳定后合入）

### 4.1 改动点

**文件**：`nodes/loop/loop.py`

#### 4.1.1 记录 expand_root

在 **`MieLoopEnd.execute()` 首轮**（`curr_index == 0` 且 expand 即将发生，或 Start 完成时）写入：

```python
if "expand_root" not in ctx.get("meta", {}):
    # 优先用模板 end_id；若无则用当前 End 的「无点号」id
    ctx["meta"]["expand_root"] = str(ctx["meta"].get("end_id") or end_template_id)
```

- `expand_root` 一旦写入，**整个 run_id 生命周期内不变**（含 resume）。  
- 旧 workflow / 旧 `loop_ctx` 无此字段：在首次 expand 时 lazy 回填，行为等价于今天。

#### 4.1.2 显式 Flat Prefix 构造 GraphBuilder

在 `_build_expand_graph_for_next_round()` 开头：

```python
expand_root = str(next_ctx.get("meta", {}).get("expand_root") or end_id)
round_idx = int(next_ctx.get("index", 0))
flat_prefix = f"{expand_root}.r{round_idx}."
graph = GraphBuilder(prefix=flat_prefix)
```

- **禁止**使用默认 `GraphBuilder()`（会继承 ComfyUI 嵌套 prefix）。  
- 每轮 id 形如：`453.r5.__mie_loop_recurse_end__`、`453.r5.369`（业务节点仍用模板 id `"369"`）。

#### 4.1.3 与方案 B 组合

- 业务节点：继续 `set_override_display_id(模板id)`。  
- Resume：仍为 `__mie_loop_resume__`（全图唯一，每轮在新 prefix 下不冲突）。  
- End/BodyOut：使用 B 的 Recurse id。

### 4.2 预期效果

```text
改前（49 轮）: 453.0.0.453.0.0.453.0.0.453.0.0.453.0.0.369
改后（第 5 轮）: 453.r5.369
```

Expand 深度与轮数解耦；49 轮 id 长度 O(1) 级别。

### 4.3 回归风险与缓解

| 风险 | 缓解 |
|------|------|
| 跨轮 ephemeral id 冲突 | prefix 含 `round_idx`，每轮唯一 |
| 外链 `["470",1]` 在 flat id 下失效 | 专项测试：10+ 轮 expand，490 类节点仍保留外链；必要时跑 ComfyUI execution 集成测 |
| ComfyUI cache key 变化导致行为差异 | VHS 等含 `UNIQUE_ID` 的节点会获新 id → **仅影响 cache 命中，不影响正确性**；长循环反而减少错误复用 |
| `expand_root` 与 `end_id` meta 不一致 | 以 `end_id` 为准初始化；单元测试覆盖 meta 已有 end_id 的场景 |
| resume 跨 prompt 续跑 | resume ctx 携带 `expand_root`；同一 `run_id` + `loop_id` 后续轮 prefix 连续 |

---

## 5. 实施顺序与 PR 拆分

### PR-1：方案 B（Recurse 命名）

1. 实现 §3.1  
2. 更新/新增测试（§6.1）  
3. 跑全量 `tests/test_loop*.py`  
4. 手动：用 `workflows/loop7.json` 跑 5 轮，对比 Finalize 输出与改前一致  

**合并门槛**：无测试失败；workflow JSON 无需改动。

### PR-2：方案 A（Flat Prefix）

1. 在 PR-1 基础上实现 §4.1  
2. 更新/新增测试（§6.2）  
3. 跑全量 loop 测试 + 可选 ComfyUI execution `test_for_loop` 对照  
4. 手动：mock 49 轮 expand 图 id 快照（不跑真实 GPU）  

**合并门槛**：§6.3 工作流快照清单全部通过。

---

## 6. 测试计划

### 6.1 PR-1 新增/更新测试

**文件**：`tests/test_loop_graph.py`

| 用例 | 断言 |
|------|------|
| `test_expand_recurse_end_id` | expand 图 End 的 key 含 `__mie_loop_recurse_end__`；`override_display_id == end_id` |
| `test_expand_recurse_bodyout_id` | 同上 BodyOut |
| `test_expand_no_template_end_id_in_path` | 任意 expand 节点 id **不包含** `{end_id}.0.0.{end_id}` 子串 |
| 更新 `test_build_expand_graph_resume_node_does_not_collide_*` | Resume ≠ Recurse End |

**文件**：`tests/test_loop_nodes.py`

| 用例 | 断言 |
|------|------|
| 已有 `test_cloned_round_keeps_original_end_id` | 仍为模板 id（加强：expand 后 meta 不含 Recurse 字符串） |

### 6.2 PR-2 新增/更新测试

**文件**：`tests/test_loop_graph.py`

| 用例 | 断言 |
|------|------|
| `test_flat_prefix_format` | 所有 expand 节点 id 匹配 `^{expand_root}\.r\d+\.` |
| `test_flat_prefix_round_increments` | round 3 → prefix 含 `.r3.` |
| `test_flat_prefix_no_nested_template_end` | 49 轮模拟 build expand（不调 ComfyUI）→ id 中无重复 `.0.0.453` 链 |
| `test_external_links_preserved_under_flat_prefix` | KSampler model 仍为 `["16",0]` 等（沿用 v3_fixes 夹具） |
| `test_expand_root_lazy_init` | 无 expand_root 的 ctx 首次 expand 后写入且等于 end_id |

**文件**：`tests/test_loop_nodes.py`

| 用例 | 断言 |
|------|------|
| `test_resume_preserves_expand_root` | resume_loop_ctx 后续 expand prefix 仍用同一 root |

### 6.3 已保存工作流快照回归（手动 + 可选 CI）

对以下文件 **原样加载**，不得修改 JSON：

| 文件 | 验证项 |
|------|--------|
| `workflows/loop2.json` ~ `loop7.json` | Queue 成功；轮数与 Finalize 输出一致 |
| 外部 SCAIL 循环工作流（如 `wan21_scail2_loop_fixed_canvas.json`） | 短跑 count=5 与改前输出帧数一致；日志 id 已为 flat/recurse 格式 |

**对比维度**（改前改后）：

- Finalize 图片 batch 尺寸 / 文本列表长度  
- `loop_ctx.count`、`index` 推进顺序  
- Collect 计数日志 `LoopCollectImage count=N`  
- **不对比** ephemeral execution id（预期变化）

### 6.4 全量自动化命令

```bash
cd C:\Users\administered\PycharmProjects\ComfyUI-MieNodes
python -m pytest tests/test_loop_graph.py tests/test_loop_nodes.py tests/test_loop_v3_fixes.py tests/test_loop_optimization.py tests/test_loop_v31_*.py tests/test_loop_runtime.py tests/test_loop_pure.py -q
```

---

## 7. 文档更新（随 PR 合入）

**文件**：`docs/LOOP_USAGE.md`

新增小节 **「Expand 运行时 ID」**：

- 说明用户保存的 workflow 仍用模板 id（453 等）  
- 日志中可能出现 `453.r5.369`，属正常现象  
- `override_display_id` 保证 UI 编号不变  
- 不涉及用户操作  

---

## 8. 回滚策略

| 阶段 | 回滚 |
|------|------|
| PR-1 出问题 | Revert Recurse 映射；expand 恢复旧 id 规则 |
| PR-2 出问题 | Revert flat prefix，保留 PR-1 Recurse（仍优于现状） |
| 生产紧急 | 两 PR 均可独立 revert，**无需用户改 workflow** |

---

## 9. 验收标准（Definition of Done）

- [ ] PR-1 + PR-2 合并  
- [ ] §6.4 全量 pytest 绿  
- [ ] §6.3 工作流快照回归通过  
- [ ] 49 轮 expand 模拟：最长节点 id 长度 < 80 字符（含 `453.r48.369` 量级）  
- [ ] `ctx.meta.end_id / body_out_id / body_in_id` 在任意轮次仍为模板 id  
- [ ] `LOOP_USAGE.md` 已更新  
- [ ] 无 `MieLoopStart/End/BodyIn/BodyOut` 的 breaking INPUT 变更  

---

## 10. 时间估算

| 阶段 | 工作量 |
|------|--------|
| PR-1 实现 + 测试 | 0.5–1 天 |
| PR-2 实现 + 测试 | 1–1.5 天 |
| 工作流手动回归 + 文档 | 0.5 天 |
| **合计** | **约 2–3 天** |

---

## 附录 A：关键代码锚点

| 位置 | 说明 |
|------|------|
| `loop.py:_build_expand_graph_for_next_round` | B/A 主要改动 |
| `loop.py:MieLoopEnd.execute` | A：`expand_root` 初始化 |
| `loop.py:_should_record_protocol_node_id` | 协议 id 防漂移（勿破坏） |
| `loop.py:collect_loop_body` | 模板图发现（勿改为读 expand 图） |
| `comfy_execution/graph_utils.py:GraphBuilder.__init__(prefix=...)` | A 依赖显式 prefix |
| `flow_control.py:while_loop_close` | B 的设计参考 |

## 附录 B：与 SCAIL 双视频问题的关系

本计划**不解决**工作流内 469/369 双 `LoadVideo` 路径不一致问题（属工作流设计/文档范畴）。Flat expand 仅提升长跑稳定性；SCAIL 用户仍需保证两个视频输入为同一文件。
