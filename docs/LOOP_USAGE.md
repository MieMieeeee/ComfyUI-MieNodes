# MieLoop 使用指南（v3.1+）

## 概述
MieLoop 是 ComfyUI-MieNodes 的 for-each 循环节点组，基于 `MIE_LOOP_CTX` 在多轮之间传递参数、状态与收集器引用。  
当前协议已收口为 **collector-only**：循环体结果通过 `Collect* / Finalize*` 输出，跨轮反馈通过 `state` 传递。

## 核心原则
- 使用 ComfyUI `expand` 机制做循环，不是 while 轮询。
- 协议节点固定为 `Start -> BodyIn -> BodyOut -> End`。
- `BodyOut` 只负责 `loop_ctx + state_json`。
- `End` 只负责 `loop_ctx + done`，并在 `done=False` 时展开下一轮。
- 图片/文本/JSON/音频结果统一走 `Collect* + Finalize*`。

## 节点清单
### 协议节点
| 节点 | 输入 | 输出 |
|---|---|---|
| `MieLoopStart` | loop 配置 | `loop_ctx`, `index`, `count`, `is_last` |
| `MieLoopBodyIn` | `loop_ctx`, `anchor` | `loop_ctx`, `anchor` |
| `MieLoopBodyOut` | `loop_ctx`, `state_json` | `loop_ctx`, `state_json` |
| `MieLoopEnd` | `loop_ctx`, `state_json`, `debug` | `loop_ctx`, `done` |
| `MieLoopResume` | 内部节点（勿手动接） | `loop_ctx` |

### 参数与轮次读取
| 节点 | 作用 |
|---|---|
| `MieLoopGetIndex` | 获取当前轮次，从 `0` 开始 |
| `MieLoopParamGetInt/Float/String/Bool` | 读取 `current_params` |

### 状态节点
| 节点 | 作用 |
|---|---|
| `MieLoopStateGetInt/Float/String/Bool` | 读取 `state[key]` |
| `MieLoopStateSet` | 生成 `state_json`（字符串 patch） |
| `MieLoopStateSetInt` | 直接写 `ctx.state[key]=int` |
| `MieLoopStateSetImage` | 写入图像状态对象（按 ref 存储） |
| `MieLoopStateGetImage` | 读取图像状态对象 |
| `MieLoopStateCleanupImage` | 清理图像状态对象 |

### 收集器节点
| 类型 | Collect | Finalize | Cleanup |
|---|---|---|---|
| 图片 | `MieLoopCollectImage` | `MieLoopFinalizeImages` | `MieLoopCleanupImages` |
| 文本 | `MieLoopCollectText` | `MieLoopFinalizeTextList` | `MieLoopCleanupText` |
| JSON | `MieLoopCollectJSON` | `MieLoopFinalizeJSONList` | `MieLoopCleanupJSON` |
| 音频 | `MieLoopCollectAudio` | `MieLoopFinalizeAudio` | `MieLoopCleanupAudio` |

辅助节点：`MieImageGrid`、`MieImageSelectFrame`。

## Start 参数模式
`MieLoopStart` 使用 `param_type + param_mode`：
- `int`: `list/range`
- `float`: `list/range`
- `string`: `list`
- `json`: `list`

示例：
- `int + range(0,5,1)` -> 5 轮，index 为 `0..4`
- `int + list("8,9,10")` -> 3 轮

## 标准连线模板
```text
Start.loop_ctx -> BodyIn.loop_ctx
BodyIn.loop_ctx -> [业务链]
[业务链输出的 loop_ctx] -> BodyOut.loop_ctx
BodyOut.loop_ctx -> End.loop_ctx
End.loop_ctx/done -> Finalize*
```

关键点：
- `Finalize*` 一定接 `End.done`，否则只会返回空结果。
- 轮次内要跨轮反馈，优先用 `MieLoopStateSetInt` / `MieLoopStateSetImage`。

## 收集器与 offload to disk
`MieLoopCollectImage`、`MieLoopCollectAudio` 支持：
- `offload_to_disk`（默认 `false`）
- `offload_dir`（默认空）

行为：
- 开启后，收集阶段写 `.pt` 到临时目录，内存里只保留 `disk_path` 元信息。
- `Finalize*` / `Cleanup*` / 运行时清理会自动删除对应磁盘缓存。

适用场景：
- 长轮次、大分辨率图片或长音频，降低峰值显存/内存。

## Expand 与协议 ID 约束
MieLoop 使用 expand 图递归执行下一轮。为避免 ID 漂移：
- 协议模板 ID（`body_out_id`, `end_id`）一旦确定必须保持模板值。
- 克隆轮次 ID（包含 `.`）不会回写到 `meta`。

这可以避免类似 `240.0.0.240.0.0...` 的递归拼接问题。

## 日志与 debug
常规（`debug=false`）日志只保留关键流程：
- `LoopStart`
- `LoopEnd: loop continue/completed`
- `LoopEndExpand`
- `LoopCollect*` / `LoopFinalize*`

详细展开图日志（`ExpandGraphBuild`、`ExpandNode`）仅在 `debug=true` 输出。

建议：
- 生产或长任务使用 `debug=false`。
- 排障时短时开启 `debug=true`，复现后关闭。

## 常见问题
### 只跑一轮
- 检查 `Start.count` 是否 > 1。
- 检查 `BodyOut.loop_ctx -> End.loop_ctx` 是否断开。
- 看 `LoopEnd` 是否输出了 `loop continue`。

### Finalize 为空
- `Finalize*` 的 `done` 没接 `End.done`。
- 对应 `Collect*` 没在循环体内执行到。

### 音频拼接异常
- `FinalizeAudio` 要求各轮 `sample_rate` 一致，不一致会报错。
- 如果每轮音频来源相同，会得到重复拼接结果，这是连线语义问题，不是节点 bug。

### 日志体积异常增长
- 先确认 `debug` 是否误开。
- 确认使用的是最新版本（包含协议 ID 冻结与日志截断修复）。

## 推荐最小示例（文本收集）
```text
Start(int_list=1,2,3)
 -> BodyIn
 -> ParamGetInt(value)
 -> ShowAnything
 -> CollectText
 -> BodyOut(state_json={})
 -> End
 -> FinalizeTextList(done=End.done)
```

期望输出：`["1","2","3"]`（或你业务链生成的文本）。
