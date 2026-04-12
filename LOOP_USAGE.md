# MieLoop 使用指南

## 概述

MieLoop 是 ComfyUI-MieNodes 的 For 循环节点组，支持多轮迭代执行、状态管理、图片收集等功能。

### 核心设计

- **v3 协议**: 基于 `MIE_LOOP_CTX` 上下文传递
- **expand 机制**: 使用 ComfyUI 的图展开实现循环（`GraphBuilder` + `is_link()` 结果）
- **BodyIn/BodyOut/End**: 三个协议节点定义循环边界
- **自动节点发现**: 通过图遍历识别循环体业务节点
- **CollectImage**: 收集器节点不参与 clone，通过 `RUNTIME_STORE` 跨轮传递

---

## 节点一览

### 协议节点（必须使用）

| 节点 | 用途 | 关键输出 |
|------|------|----------|
| **🐑 MieLoopStart** | 循环入口，定义参数列表 | `loop_ctx`, `index`, `count`, `is_last` |
| **🐑 MieLoopBodyIn** | 循环体入口锚点 | `loop_ctx`, `anchor` |
| **🐑 MieLoopBodyOut** | 循环体出口，传递状态和值 | `loop_ctx`, `value_image`, `state_json`, `value_any_1~5` |
| **🐑 MieLoopEnd** | 循环结束，判断继续或完成 | `loop_ctx`, `done`, `final_images`, `final_grid` |

### 参数节点（循环体内使用）

| 节点 | 用途 | 输出类型 |
|------|------|----------|
| **🐑 MieLoopParamGetInt** | 获取当前轮整数参数 | `INT` |
| **🐑 MieLoopParamGetFloat** | 获取当前轮浮点参数 | `FLOAT` |
| **🐑 MieLoopParamGetString** | 获取当前轮字符串参数 | `STRING` |
| **🐑 MieLoopParamGetBool** | 获取当前轮布尔参数 | `BOOLEAN` |

### 状态节点（循环体内使用）

| 节点 | 用途 | 输出类型 |
|------|------|----------|
| **🐑 MieLoopStateGetInt** | 获取状态整数 | `INT` |
| **🐑 MieLoopStateGetFloat** | 获取状态浮点 | `FLOAT` |
| **🐑 MieLoopStateGetString** | 获取状态字符串 | `STRING` |
| **🐑 MieLoopStateGetBool** | 获取状态布尔 | `BOOLEAN` |
| **🐑 MieLoopStateSet** | 设置状态值 | `STRING` (state_json) |

### 图片收集节点

| 节点 | 用途 | 输出类型 |
|------|------|----------|
| **🐑 MieLoopCollectImage** | 收集图片到 batch | `MIE_LOOP_CTX` |
| **🐑 MieLoopFinalizeImages** | 合并所有收集的图片 | `IMAGE` |
| **🐑 MieLoopCleanupImages** | 清理图片缓存 | `MIE_LOOP_CTX`, `BOOLEAN` |
| **🐑 MieImageGrid** | 将多张图片拼接成网格 | `IMAGE` |

---

## 基础工作流结构

```
┌─────────────┐
│ MieLoopStart│──►loop_ctx──►┌──────────────┐
└─────────────┘              │ MieLoopBodyIn │──►loop_ctx
                             └───────┬───────┘
                                     │
                          ┌──────────▼──────────┐
                          │   [业务节点]         │
                          │  (ParamGet/StateGet│
                          │   /KSampler/etc)   │
                          └──────────┬──────────┘
                                     │
                            ┌────────▼────────┐
                            │ MieLoopBodyOut  │──►loop_ctx + value_image + state_json
                            └────────┬────────┘
                                     │
                             ┌──────▼──────┐
                             │  MieLoopEnd  │──►done + final_images + final_grid
                             └──────────────┘
```

---

## 参数配置详解

### MieLoopStart 参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `loop_id` | STRING | 循环标识符 | `"scan_steps"` |
| `params_mode` | 选择 | 参数模式 | `int_list` / `string_list` / `json_list` |
| `int_list` | STRING | 整数列表（逗号分隔） | `"8,9,10"` |
| `string_list` | STRING | 字符串列表（每行一个） | `"cat\ndog\ncar"` |
| `json_list` | STRING | JSON 对象列表 | `'[{"steps":8},{"steps":9}]'` |
| `initial_state_json` | STRING | 初始状态 JSON | `'{"count":0}'` |

### params_mode 示例

**int_list** (推荐用于数值参数):
```
8,9,10
→ 第0轮: 8, 第1轮: 9, 第2轮: 10
```

**string_list** (推荐用于字符串参数):
```
cat
dog
car
→ 第0轮: "cat", 第1轮: "dog", 第2轮: "car"
```

**json_list** (推荐用于多值参数):
```json
[{"steps":20, "cfg":7.5}, {"steps":30, "cfg":8.0}]
→ 第0轮: {"steps":20, "cfg":7.5}
→ 第1轮: {"steps":30, "cfg":8.0}
```

---

## 状态管理

### 基础概念

- **State**: 跨轮次持久化的状态字典
- **Current Params**: 当前轮次的参数（来自 params_list）
- **Initial State**: 循环开始前的初始状态

### 使用示例

```python
# MieLoopStart - 设置初始状态
initial_state_json = '{"total":0}'

# MieLoopStateSet - 在循环内更新状态
state_json = '{"total":${total}+1}'

# MieLoopStateGetInt - 获取状态值
key = "total"
default_value = 0
→ 返回当前轮的 total 值
```

### 状态更新时机

状态在 `MieLoopBodyOut` 中通过 `state_json` 参数更新：

```python
# MieLoopBodyOut 输入
state_json = '{"steps":25, "result":"done"}'

# 循环结束时，ctx.state 包含所有轮次的状态合并
```

---

## 图片收集与合并

### 推荐模式

```
┌──────────────────────────────────────────────────────┐
│                    循环体                              │
│  ... → KSampler → Decode → LoopCollectImage         │
│                               ↓                       │
│                          loop_ctx                     │
└──────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  MieLoopFinalizeImages │ (done=True 时)
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    MieImageGrid     │
                    └─────────────────────┘
```

### 收集机制

- `MieLoopCollectImage`: 将图片添加到当前 run 的缓存
- 缓存在 `RUNTIME_STORE` 中，跨轮次持久化
- `MieLoopFinalizeImages`: 当 `done=True` 时，合并所有收集的图片
- `MieImageGrid`: 将多张图片拼接成网格

### 示例

```python
# MieLoopStart
params_mode = "int_list"
int_list = "1,2,3"  # 3轮循环

# 每轮收集一张图
MieLoopCollectImage: image= decoded_image

# 循环结束后
MieLoopFinalizeImages: done=True
→ final_images = [img1, img2, img3]  # 3张图

# 可选：拼接成网格
MieImageGrid: cols=3
→ final_grid = 3列网格图
```

---

## value_any 跨轮传递

### 用途

`value_any_1` ~ `value_any_5` 用于在轮次间传递任意类型的值。

### 优先级规则

当 `MieLoopEnd` 收到 `value_any` 时：
- 如果新值不为 None，使用新值
- 如果新值为 None，保留上一轮 BodyOut 存储的值

### 示例

```python
# 第0轮 BodyOut
value_any_1 = tensor_data

# 第0轮 LoopEnd (不传 value_any_1)
# → ctx.value_any[0] = tensor_data

# 第1轮 LoopEnd (传新的 value_any_1)
value_any_1 = new_tensor
# → ctx.value_any[0] = new_tensor

# 第2轮 LoopEnd (不传 value_any_1)
# → ctx.value_any[0] = new_tensor (保留)
```

---

## Expand 机制原理

MieLoop 的循环复用 ComfyUI 内置的 **Graph Expansion** 机制。理解其原理对调试至关重要。

### 执行流程

```
第0轮（原始图）:
  LoopStart → BodyIn → [业务节点...] → BodyOut → LoopEnd(done=False)
                                                          │
                                                    返回 {"result": [is_link()...], "expand": graph}
                                                          │
                                                    ComfyUI 引擎:
                                                    1. 将 expand 图注册为临时节点
                                                    2. result 中的 is_link() 创建 strong_link
                                                    3. 将子图节点加入执行队列
                                                    4. 返回 PENDING，等待子图完成
                                                          │
第1轮（expand 图）:                                       ▼
  Resume → [业务节点...] → BodyOut → LoopEnd(done=False)
                                          │
                                    再次 expand ...
                                          │
最后一轮:                                   ▼
  Resume → [业务节点...] → BodyOut → LoopEnd(done=True)
                                          │
                                    返回具体值 (ctx, True, final_images, final_grid)
```

### is_link() 的关键作用

ComfyUI 引擎（`execution.py` lines 572-576）检测 `result` 中的 `is_link()` 值来决定是否执行 expand 图：

- **有 `is_link()`**：引擎创建 `strong_link`，将子图节点拉入执行队列，等待子图完成后才解析父节点输出
- **无 `is_link()`**：引擎立即用具体值解析父节点输出，expand 图注册但**永远不会执行**

MieLoopEnd 在 `done=False` 时返回：
```python
{
    "result": tuple([end_built_node.out(i) for i in range(4)]),  # 全是 is_link()
    "expand": expand_graph,
}
```

`end_built_node.out(i)` 返回 `[prefixed_node_id, output_index]`，指向 expand 图中克隆的 LoopEnd 节点输出。

### 节点分类与 Expand 图构建

| 节点类型 | 是否 clone | 说明 |
|----------|-----------|------|
| **协议节点** (BodyOut, LoopEnd) | ✅ clone | 定义循环体出口 |
| **业务节点** (KSampler, Decode, etc.) | ✅ clone | 循环体核心逻辑 |
| **收集器节点** (CollectImage) | ❌ 不 clone | 使用 `RUNTIME_STORE` 跨轮传递 |
| **入口锚点** (BodyIn) | ❌ 不 clone | 输出被 Resume 节点替代 |
| **外部节点** (ModelLoader, etc.) | ❌ 不 clone | 保持原始 link 引用 |

### 与 EasyUse 的对比

| 特性 | EasyUse whileLoopEnd | MieLoopEnd |
|------|---------------------|------------|
| 循环控制 | while 条件 | for_each 计数 |
| result 格式 | `tuple([my_clone.out(i)])` | `tuple([end_built_node.out(i)])` |
| 上下文传递 | rawLink + value | MIE_LOOP_CTX dict |
| 图片收集 | 无内置 | RUNTIME_STORE + CollectImage |

---

## debug 模式

### 启用方式

```
MieLoopEnd → debug = True
```

### 输出信息

启用后，`MieLoopEnd` 会输出详细的循环体发现信息：

```
LoopEndDetect: loop_id=scan, run_id=abc123, 
  body_in_id=NodeID.1, body_out_id=NodeID.5, end_id=NodeID.6,
  forward_set=[...], backward_set=[...],
  body_nodes_raw=[...], body_nodes_filtered=[...],
  protocol_nodes=[...], collector_nodes=[...],
  body_nodes_business=[...], excluded_nodes=[...]
```

### 用途

- 排查循环体节点发现是否正确
- 验证业务节点是否被正确识别
- 检查 collector 节点是否被正确排除

---

## 常见工作流示例

### 示例 1: 基础参数循环

```
┌─────────────┐
│ MieLoopStart│ params_mode="int_list", int_list="1,2,3"
└──────┬──────┘
       │
   loop_ctx, index, count, is_last
       │
       ▼
┌──────────────┐
│ MieLoopBodyIn │ anchor=anything
└──────┬───────┘
       │
   loop_ctx
       │
       ▼
┌─────────────────────┐
│ MieLoopParamGetInt  │ key="value"
└─────────┬───────────┘
          │
       INT (current param)
          │
          ▼
    [KSampler etc.]
          │
          ▼
┌──────────────┐
│ MieLoopBodyOut│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  MieLoopEnd  │ done + final_images
└──────────────┘
```

### 示例 2: 带状态和图片收集

```
┌─────────────┐
│ MieLoopStart│ params_mode="json_list", 
│             │ json_list='[{"seed":1},{"seed":2},{"seed":3}]'
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ MieLoopBodyIn │
└──────┬───────┘
       │
   loop_ctx
       │
       ▼
┌─────────────────────┐
│ MieLoopParamGetInt  │ key="seed"
└─────────┬───────────┘
          │
       seed
          │
          ▼
    [KSampler → Decode]
          │
          ├─────────────────┐
          │                 │
          ▼                 ▼
┌─────────────────┐  ┌──────────────┐
│MieLoopCollectImg│  │ MieLoopBodyOut│
└────────┬────────┘  └──────┬───────┘
         │                  │
         ▼                  ▼
   loop_ctx (updated)  loop_ctx
                              │
                              ▼
                    ┌──────────────────┐
                    │   MieLoopEnd     │
                    │ debug=True       │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
        loop_ctx         done           final_images
            │                │                │
            │                │                ▼
            │                │         ┌────────────┐
            │                │         │MieImageGrid│
            │                │         └────────────┘
            ▼                ▼
        (继续下一轮)     (最后一轮)
```

---

## 日志输出

### 关键日志

| 日志前缀 | 触发时机 | 关键信息 |
|----------|----------|----------|
| `LoopStart:` | 循环初始化 | `loop_id`, `run_id`, `count` |
| `LoopEnd: loop continue:` | 继续下一轮 | `round=X -> Y`, `count=N` |
| `LoopEnd: loop completed:` | 循环结束 | `final_round=X`, `count=N` |
| `LoopEndExpand:` | 展开图构建 | `next_index`, `expand_nodes` |
| `LoopEndDetect:` | 循环体发现 (debug) | 节点集合详情 |
| `LoopCollectImage:` | 图片收集 | `loop_id`, `ref`, `count` |

### 查看日志

在 ComfyUI 终端或日志文件中查看。

---

## 注意事项

### 1. 循环体节点发现

- `MieLoopBodyIn` 和 `MieLoopBodyOut` 必须连接到工作流的对应位置
- ComfyUI 自动通过图遍历发现循环体中的业务节点
- `SaveImage`、`PreviewImage` 等输出节点会被自动排除

### 2. RUNTIME_STORE 清理

- 图片缓存在 `RUNTIME_STORE` 中管理
- 正常情况下 `MieLoopEnd` 在 `done=True` 时自动清理
- 如需手动清理，使用 `MieLoopCleanupImages`

### 3. 循环次数限制

- 单次循环最多 30 轮（`count >= 30` 会输出警告）
- 超过限制建议分段执行

### 4. debug 模式

- 正常运行时 `debug=False`
- 排查问题时启用 `debug=True`
- 生产环境建议关闭（减少日志输出）

---

## 错误排查

### 循环只执行一轮

这是最常见的问题，通常由以下原因导致：

**1. expand 图未执行（最可能）**

症状：日志显示 `LoopEndExpand: expand_nodes=N` 但没有后续轮次执行。

检查 `ExpandGraphBuild` 日志：
- `expand_node_ids` 中是否有节点
- 如果日志显示 expand_nodes > 0 但下一轮不执行，说明 result 中缺少 `is_link()` 值

**2. done 立即为 True**

检查 `LoopEnd` 日志：
- 如果没有 `LoopEndExpand` 日志，说明 `done=True` 立即触发
- 检查 `count` 值是否正确

**3. body_nodes_business 为空**

启用 `debug=True` 查看 `LoopEndDetect` 日志：
- `body_nodes_business` 是否为空
- `protocol_nodes` 是否包含正确的 BodyIn/BodyOut/End ID

### expand 失败

启用 `debug=True` 查看 `LoopEndDetect` 日志：
- `body_nodes_business` 是否为空
- `protocol_nodes` 是否包含正确的 BodyIn/BodyOut/End ID
- `collector_nodes` 是否正确排除（不应出现在 `body_nodes_business` 中）

### 图片收集丢失

检查：
- `MieLoopCollectImage` 是否每轮都连接
- `MieLoopFinalizeImages` 是否在 LoopEnd 之后
- `done=True` 时 `LoopCollectImage` 日志是否显示正确的 count

### 日志阅读指南

正常 3 轮循环的日志序列：
```
LoopStart: initialized loop_id=..., count=3          # 初始化
LoopEnd: loop continue: round=0 -> 1, count=3        # 第0轮完成
LoopEndExpand: next_index=1, expand_nodes=N           # 展开第1轮
LoopEnd: loop continue: round=1 -> 2, count=3        # 第1轮完成
LoopEndExpand: next_index=2, expand_nodes=N           # 展开第2轮
LoopEnd: loop completed: final_round=2, count=3      # 第2轮完成（最后一轮）
```
