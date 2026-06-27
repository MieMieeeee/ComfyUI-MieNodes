# MieLoop Finalize 增量合并实施计划

> 目标：修复长跑循环（如 SCAIL 49 段）在 **`MieLoopFinalizeImages` / `MieLoopFinalizeAudio`** 阶段因一次性加载全部 disk offload 缓存而崩溃的问题。  
> 范围：**Finalize 合并路径 + 失败可恢复日志**；不改 Collect / expand / workflow JSON。  
> 首要约束：**已保存工作流零回归**——输出 tensor 语义、节点 socket、Collect/Finalize 协议不变。

---

## 1. 背景与问题

### 1.1 现场（SCAIL-2，`count=49`，`offload_to_disk=true`）

| 阶段 | 结果 |
|------|------|
| 49 轮生成 + Collect | ✅ 完成（`LoopCollectImage count=49`） |
| 49 层 `MieLoopEnd` unwind | ✅ 完成（极慢，但与本次 crash 无直接因果） |
| `MieLoopFinalizeImages` → `_merge_images_for_ctx` | ❌ `torch.load` 时 **access violation**（Windows `0xC0000005`） |
| 最终成片 | ❌ 未产出 |

- **run_id**：`fb7f168cb16d4c3291bfd5a2`
- **disk cache**：`{ComfyUI temp}/mie_loop_offload/{run_id}/`，49 个有效 `image_*.pt`，合计约 **10 GB**
- **单段 shape**：`(81, 352, 640, 3)` float32（末段帧数更少）
- **手动救回**：重启后逐文件 `torch.load` + 增量 `torch.cat` → 3717 帧 mp4 合并成功（见 `e:\CC\data\scail2\workflows\merge_mie_loop_cache.py`）

说明：**缓存文件本身完整**，崩溃发生在 Finalize 合并策略，而非生成阶段或 expand ID。

### 1.2 现状代码

**Images**（`loop.py` → `_merge_images_for_ctx`）：

```python
batches = []
for item in raw_batches:
    if _is_disk_cache_item(item):
        loaded = torch.load(..., map_location="cpu")
        batches.append(loaded)
    else:
        batches.append(item)
return torch.cat(batches, dim=0)
```

**Audio**（`MieLoopFinalizeAudio.execute`）：同样先 **全量 load 进 list**，再 `torch.cat(waveforms, dim=-1)`。

### 1.3 问题本质

| 问题 | 说明 |
|------|------|
| **峰值内存过高** | 49 段 × ~209 MB ≈ 10 GB；list 持有全部 batch 引用，`torch.cat` 前峰值接近 **2× 合并体积** |
| **原生崩溃风险** | 大量连续 `torch.load` + 超大 `cat` 在长跑后的 PyTorch 进程中触发 access violation（不一定是 Python OOM） |
| **失败后难抢救** | 崩溃时日志未打印 cache 目录/文件列表；用户需自行搜 `mie_loop_offload/{run_id}` |
| **与 ID 扁平化正交** | `PLAN_LOOP_FLAT_EXPAND.md` 解决 expand/unwind；**不解决** Finalize 读盘合并 |

---

## 2. 目标与非目标

### 2.1 目标

1. **增量合并**：每次只保留「已合并结果 + 当前 batch」，load 后立即 `cat`，释放 batch 引用并 `gc.collect()`。
2. **内存路径统一**：无论 batch 来自 memory 还是 disk，走同一套 `_merge_tensor_batches_incremental()` 逻辑。
3. **失败可恢复**：Finalize 开始前记录 `run_id`、cache 目录、batch 数、路径列表（截断日志）；合并失败时 **不删** disk cache。
4. **行为零回归**：短循环（count≤5）、纯 memory collect、现有 offload 测试全部通过。

### 2.2 非目标（本计划不做）

- 不改 Collect 写盘格式（仍为 `torch.save(tensor)` → `image_*.pt`）
- 不改 overlap 去重（Collect 仍收集全段；与官方 extend 的差异属 workflow 层）
- 不实现 Finalize 流式写 mp4（仍输出 `IMAGE` tensor；下游 `VHS_VideoCombine` 不变）
- 不实现跨 run 断点 Finalize（resume_loop_ctx 续跑属可选能力，见 `PLAN_LOOP_FLAT_EXPAND.md`）
- 不替换 `torch.load` 为自定义 zip 解析（维护成本过高）

---

## 3. 不变量（零回归契约）

| 项目 | 要求 |
|------|------|
| 节点 API | `MieLoopFinalizeImages` / `MieLoopFinalizeAudio` 的 INPUT/OUTPUT 不变 |
| 输出语义 | `torch.cat(batches, dim=0)`（image）/ `torch.cat(waveforms, dim=-1)`（audio）结果与改前 **bitwise 一致**（同 dtype、同 shape） |
| shape 校验 | 仍校验各 batch 的 `shape[1:]` / `sample_rate` 一致；错误信息与现网等价 |
| disk cleanup | **仅在合并成功** 后执行 `_cleanup_disk_cache_paths`（与 today 的 `finally` 一致，但 merge 抛错时不 cleanup） |
| `done=false` | 仍返回 `EMPTY_IMAGES` / `EMPTY_AUDIO` |
| workflow JSON | 不要求用户修改任何节点 widget |

---

## 4. 方案设计

### 4.1 核心：增量 tensor 合并 helper

**新增**（`loop.py`，Collect/Finalize 区域附近）：

```python
def _merge_tensor_batches_incremental(
    raw_batches,
    *,
    load_disk_item,
    validate_batch,
    log_progress=None,
):
    """Load one batch at a time, cat into accumulator, release batch refs."""
    merged = None
    for idx, item in enumerate(raw_batches):
        batch = load_disk_item(item) if _is_disk_cache_item(item) else item
        validate_batch(batch, idx, merged)
        if merged is None:
            merged = batch
        else:
            merged = torch.cat([merged, batch], dim=0)
            del batch
            gc.collect()
        if log_progress:
            log_progress(idx + 1, len(raw_batches))
    return merged if merged is not None else ...
```

- **Image**：`load_disk_item` = `torch.load(path, map_location="cpu")` + isinstance 检查  
- **Memory batch**：直接传入 tensor，不 copy（与 today 一致）  
- **Audio**：单独 helper `_merge_audio_items_incremental()`，波形沿最后一维 cat；disk item load 后取 `waveform`

### 4.2 重构 `_merge_images_for_ctx`

```python
def _merge_images_for_ctx(loop_ctx):
    ...
    disk_paths = _collect_disk_paths(raw_batches)
    _log_finalize_merge_start("image", ctx, raw_batches, disk_paths)
    try:
        def load_item(item):
            loaded = torch.load(str(item["disk_path"]), map_location="cpu")
            if not isinstance(loaded, torch.Tensor):
                raise ValueError("disk cached image is not a torch.Tensor")
            return loaded

        def validate(batch, idx, merged):
            if merged is None:
                return
            if tuple(batch.shape[1:]) != tuple(merged.shape[1:]):
                raise ValueError(...)

        return _merge_tensor_batches_incremental(
            raw_batches, load_disk_item=load_item, validate_batch=validate,
            log_progress=_log_finalize_batch_progress,
        )
    except Exception:
        _log_finalize_merge_failed("image", ctx, disk_paths)
        raise
    finally:
        _cleanup_disk_cache_paths(disk_paths)  # 仅当 try 未抛错时？→ 见 4.4
```

### 4.3 重构 `MieLoopFinalizeAudio.execute`

- 提取与 image 对称的增量逻辑  
- 保持 `sample_rate` 校验顺序：首个 item 定基准，后续逐项比对

### 4.4 disk cache 清理策略（重要）

**Today**：`try/finally` 无论 merge 成功与否都会 `_cleanup_disk_cache_paths`。

**改后**：

```python
merged_ok = False
try:
    merged = _merge_...(raw_batches)
    merged_ok = True
    return merged
except Exception:
    _log_finalize_merge_failed(...)
    raise
finally:
    if merged_ok:
        _cleanup_disk_cache_paths(disk_paths)
```

- merge 失败 → **保留 `.pt` 文件**，便于用户用手动脚本救回（本次 incident 已验证可行）
- merge 成功 → 与 today 一致，清理 cache

> 若担心 temp 目录堆积：可在 `_log_finalize_merge_failed` 中打印完整目录路径 + 「可手动删除或用手动合并脚本处理」。

### 4.5 日志增强

**新增** `_log_finalize_merge_start(kind, ctx, raw_batches, disk_paths)`：

```
LoopFinalizeMergeStart: kind=image, loop_id=..., run_id=fb7f168..., batch_count=49, disk_count=49, cache_dir=E:\...\mie_loop_offload\fb7f168..., paths=[image_a.pt, ...] (truncated)
```

**新增** `_log_finalize_batch_progress(kind, ctx, current, total)`（可选，`debug` 或 batch_count≥10 时启用）：

```
LoopFinalizeMergeProgress: kind=image, run_id=..., progress=12/49
```

**失败日志** `_log_finalize_merge_failed`：

```
LoopFinalizeMergeFailed: kind=image, run_id=..., cache_dir=..., disk_files_preserved=true, batch_count=49
```

- 使用现有 `_truncate_for_log()` 限制 paths 字符串长度  
- **常规 `debug=false` 也输出** Start/Failed（Progress 可仅在 count≥N 或 debug 时输出，避免刷屏）

### 4.6 内存峰值估算（49 段 SCAIL）

| 策略 | 峰值（量级） |
|------|----------------|
| 现状（全 list + cat） | ~10 GB（list）+ 最终 ~10 GB tensor ≈ **15–20 GB** 窗口 |
| 增量 cat | **~10 GB**（仅 merged accumulator + 单 batch ~200 MB） |
| 增量 + 合并后立即 downcast | 非本计划范围（可能影响 VHS 输入精度） |

增量合并 **不能降低最终输出 tensor 体积**（下游仍需要完整 `IMAGE`），但可显著降低 **load 阶段** 的峰值与 native 崩溃概率。

---

## 5. 实施步骤（单 PR）

### PR-1：Finalize 增量合并 + 失败保留 cache + 日志

| 文件 | 改动 |
|------|------|
| `nodes/loop/loop.py` | 新增 helper；重写 `_merge_images_for_ctx`；重构 `MieLoopFinalizeAudio.execute`；日志函数 |
| `docs/LOOP_USAGE.md` | 补充：长跑建议 `offload_to_disk=true`；Finalize 失败时 cache 路径见日志 |
| `tests/test_loop_runtime.py` | 现有 `_merge_images_for_ctx` 用例保持绿 |
| `tests/test_loop_v31_image_offload.py` | 新增多 batch 增量合并用例 |
| `tests/test_loop_v31_audio_collectors.py` | 新增 audio 增量合并用例 |
| `tests/test_loop_finalize_incremental.py`（可选新文件） | 大 batch 数 mock、失败不 cleanup、日志断言 |

**不改动**：Collect、expand、workflow JSON、节点 widget。

---

## 6. 测试计划

### 6.1 单元测试

| 测试 | 断言 |
|------|------|
| `test_merge_images_incremental_matches_one_shot_cat` | 5 个 memory batch：增量结果 == 原 `torch.cat(batches)` |
| `test_merge_images_disk_offload_many_batches` | 20 个 disk batch（小 tensor）：shape、dtype、帧数正确；成功后文件删除 |
| `test_merge_images_disk_failure_preserves_files` | mock `torch.load` 在第 N 个抛错 → 前 N 个 `.pt` 仍存在 |
| `test_merge_images_shape_mismatch_mid_stream` | 第 3 个 batch shape 不对 → ValueError；cache 保留 |
| `test_merge_audio_incremental_matches_one_shot` | 多段 audio disk offload，waveform 长度一致 |
| `test_finalize_logs_cache_dir_on_start` | caplog 含 `run_id` 与 `cache_dir` |

### 6.2 回归

```bash
pytest tests/ -k "loop" -q
pytest tests/test_loop_v31_image_offload.py tests/test_loop_v31_audio_collectors.py -q
```

### 6.3 集成（可选，ComfyUI 实机）

| 场景 | 预期 |
|------|------|
| SCAIL count=5，`offload_to_disk=true` | Finalize 输出帧数与改前一致；日志有 `LoopFinalizeMergeStart` |
| SCAIL count=49 | 不再 access violation；或至少失败时 cache 完整可手动合并 |

---

## 7. 手动救回脚本（文档引用，非本 PR 必须）

已验证脚本：`e:\CC\data\scail2\workflows\merge_mie_loop_cache.py`

- 按 mtime 排序 `image_*.pt`  
- 跳过作废 collector 的前若干文件（**workflow 层问题**，Finalize 无法自动判断）  
- 增量 load + cat → mp4  

计划在 `LOOP_USAGE.md` 「常见问题」增加一节：**Finalize 崩溃后如何从 `mie_loop_offload/{run_id}` 手动合并**。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 增量 `cat` 多次拷贝导致更慢 | 49 段量级可接受；Finalize 相对 49 段生成可忽略 |
| 失败不 cleanup → temp 占盘 | 日志打印目录；文档说明可手动删 |
| `gc.collect()` 过于频繁 | 仅 disk batch 合并后调用；memory path 可省略 |
| 与 today finally-always-cleanup 行为差异 | **仅失败路径变化**；成功路径仍 cleanup；测试覆盖 |
| Audio/image 逻辑分叉 | 共用「逐 item load + validate + accumulate」模式，减少重复 |

---

## 9. 回滚策略

| 情况 | 动作 |
|------|------|
| 增量合并引入 shape/精度回归 | Revert PR-1；恢复 one-shot list+cat |
| 失败不 cleanup 导致磁盘抱怨 | 保留增量合并，恢复 finally 无条件 cleanup（折中） |

---

## 10. 验收标准

- [ ] `_merge_images_for_ctx` / `MieLoopFinalizeAudio` 使用增量合并  
- [ ] 合并失败时 disk cache **保留**，日志含 `run_id` + `cache_dir`  
- [ ] 合并成功时行为与改前一致（tensor shape/dtype/数值）  
- [ ] 全部 loop 相关 pytest 绿  
- [ ] `LOOP_USAGE.md` 更新长跑与救回说明  
- [ ] （可选）SCAIL count=49 实机 Finalize 不再崩溃  

---

## 11. 与 ID 扁平化的关系

| 计划 | 解决问题 |
|------|----------|
| `PLAN_LOOP_FLAT_EXPAND.md` | expand id 嵌套、unwind 慢、日志难读 |
| **本计划** | Finalize 读盘合并峰值内存 / native crash、失败难救回 |

两者 **独立 PR**，可并行开发；建议 **ID 扁平化先合入**，Finalize 增量合并紧随其后，共同支撑 SCAIL 49+ 段长跑。

---

## 12. 改动索引

| 符号 | 位置 |
|------|------|
| `_merge_images_for_ctx` | `loop.py` ~1182 |
| `MieLoopFinalizeAudio.execute` | `loop.py` ~2771 |
| `_offload_payload_to_disk` | 只读，不改 |
| `_cleanup_disk_cache_paths` | 调整调用条件 |
| 参考救回脚本 | `e:\CC\data\scail2\workflows\merge_mie_loop_cache.py` |
