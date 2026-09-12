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
- `Cleanup*` 与运行时清理会自动删除对应磁盘缓存；`Finalize*` **仅合并成功**时删除，失败时保留以便手动救回（见下「Finalize 崩溃后手动合并」）。

适用场景：
- 长轮次、大分辨率图片或长音频，降低峰值显存/内存。

长跑建议（如 SCAIL 数十段）：
- **强烈建议** `offload_to_disk=true`：生成阶段把每段结果落盘，避免 collect list 持有全部 tensor。
- `FinalizeImages` / `FinalizeAudio` 默认采用**增量合并**（逐段 load → cat → 释放），load 阶段峰值显著低于一次性全量载入；合并失败时磁盘缓存**保留**，便于手动救回（见下「Finalize 崩溃后手动合并」）。
- 增量合并的降峰值收益主要在 `offload_to_disk=true`（逐段从盘载入）；`offload_to_disk=false` 时 collect list 仍持有全部 tensor，Finalize 峰值与改前相近——长跑请务必开启 offload。
- `MieLoopFinalizeImages` 新增 `finalize_to_disk` / `chunk_size` 两个可选入参，详情见「长跑防 OOM 合并」一节。

## Expand 与协议 ID 约束
MieLoop 使用 expand 图递归执行下一轮。为避免 ID 漂移：
- 协议模板 ID（`body_out_id`, `end_id`）一旦确定必须保持模板值。
- 克隆轮次 ID（包含 `.`）不会回写到 `meta`。

### Expand 运行时 ID（v1.2+ 扁平化）
日志中看到的执行 ID 是**扁平化**的，属正常现象，无需任何用户操作：
- **用户保存的 workflow 仍用模板 id**（如 `453`、`369`）。画布编号、连线、JSON 完全不变。
- 运行时每轮 expand 节点的 id 形如 `453.r5.369` —— `{end_id}.r{轮次}.{模板id}`，深度与轮数解耦。
- 循环关闭节点（End / BodyOut）在 expand 图内使用固定内部 id（`__mie_loop_recurse_end__` / `__mie_loop_recurse_bodyout__`），避免每轮把模板 id 重复叠进路径。
- `override_display_id` 保证 UI 上仍显示模板编号（如 `453`），与扁平化前的观感一致。
- `ctx.meta.expand_root` 在首次 expand 时由 End 写入（等于 `end_id`），整个 run_id 生命周期（含 resume）不变；旧 workflow / 旧 loop_ctx 无此字段会自动回填。

这样彻底消除了早期版本长跑循环里 `453.0.0.453.0.0.453...` 式的嵌套拼接，长循环（如 49 轮）的节点 id 长度保持在 O(1) 量级。

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

### Finalize 崩溃后手动合并
`FinalizeImages` / `FinalizeAudio` 合并失败时（如超大批次触发原生崩溃），**不会删除**磁盘缓存，日志会打印 `LoopFinalizeMergeFailed: ... cache_dir=... disk_files_preserved=true`。
- 缓存目录：`{ComfyUI temp}/mie_loop_offload/{run_id}/`（或你指定的 `offload_dir`），形如 `image_*.pt` / `image_*.shards/`（超限批次）/ `audio_*.pt`。
- 手动救回用本仓库自带脚本（无需 ComfyUI 启动）：
  ```bash
  python scripts/manual_merge_offloaded_images.py 5e81e5f1a4b24e58a721371f
  # 或显式指定 offload 目录与输出：
  python scripts/manual_merge_offloaded_images.py       --offload-dir F:/ComfyUI_Mie_2026_V8.0_Base/ComfyUI/temp/mie_loop_offload/5e81e5f1a4b24e58a721371f       --out F:/ComfyUI_Mie_2026_V8.0_Base/ComfyUI/output/merged.pt
  # --dry-run 只列文件；--cleanup 成功合并后删原 batch。
  ```
  脚本和节点走同一条单遍流式合并路径（逐 batch 流式写入分卷，失败也保留输入文件）；输出可被 `LoadImageBatch|Mie` 直接吃（小结果为 `merged.pt` 单文件，超 ~1.5 GiB 时为 `merged.shards/` 目录，两者都会被节点自动识别）。**注意：`LoadAny|Mie` 走 pickle，读不了这些文件，恢复请用 `LoadImageBatch|Mie`。**
- 救回后可自行删除该 `{run_id}` 目录释放空间。
- 注意：若工作流内同一文件被多路 LoadVideo 引用（如 SCAIL 双视频路径不一致），collector 可能混入不属于本循环的段，需结合业务链判断要跳过的前若干文件——这是 workflow 层问题，Finalize 无法自动识别。


### 长跑防 OOM + 单文件不超 2 GB 的合并（`avoid_oom` 已弃用）
`MieLoopFinalizeImages` 现在走**单遍流式 + 字节封顶分卷**的合并路径，同时解决两件事：长跑 OOM、以及"中文安装路径 + `torch.save` 写超 2 GB zip64 文件在 2^31 字节处损坏"（社区报告的视频只剩单帧问题）。

**保证**：
- 任何一次落盘的文件都 ≤ `SHARD_FILE_CAP_BYTES`（1.5 GiB，在 2^31 损坏线下留 ~512 MiB 裕量）；超限张量自动写成 `<名字>.shards/` 目录（`part_00000.pt`、`part_00001.pt`… 按"帧"整切 + 最后原子写入的 `meta.json` 提交标记），dtype（含 bfloat16）无损，单个分卷也可以手动 `torch.load`。
- 合并峰值内存 ≈ 一个输入 batch + 一个分卷（SCAIL-2 30 段 case ~2.3 GB），不再有中间 `_mie_chunk_*.pt` 和 numpy.memmap 暂存。
- `avoid_oom` 参数保留只为旧工作流能加载，**已无效果**（默认 True；传 False 只会打一条弃用日志）。

**输出路径**：自动推到 `<ComfyUI temp>/mie_loop_offload/<run_id>/`，小结果为 `merged.pt`，超限结果为 `merged.shards/`，和每轮写出的批次文件放一起。

**节点输出**：
- `images`（IMAGE）：尽力加载合并产物，**最终 tensor 在系统 RAM 装不下时返回 `EMPTY_IMAGES`**（不抛错、不杀进程）。
- `merged_path`（STRING）：合并产物的实际路径（`merged.pt` 文件或 `merged.shards` 目录），**即使 IMAGE 槽返回了 EMPTY_IMAGES 也有效**，接给 `LoadImageBatch|Mie` 即可恢复（它自动识别单文件与分卷目录，不吃本机 RAM 也能逐卷重组）。

**失败兜底**：
- 任一阶段失败都会清掉中途的输出暂存（隐藏的 `.staging` 目录），但**绝不删原始批次文件**（`image_*.pt` / `image_*.shards/`）——preserve-on-failure 契约保持不变，可继续用 `scripts/manual_merge_offloaded_images.py` 救回。
- 分卷目录以 `meta.json` 为提交标记：进程中途被杀留下的残缺目录（无 meta 或分卷缺失）会被加载方判定为"不存在/损坏"并报 `CorruptShardError`，绝不会把半截数据当正常结果读。

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
