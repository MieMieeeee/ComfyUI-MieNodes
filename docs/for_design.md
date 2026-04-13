## 现在不能再把 v3 理解成“图像循环节点”
而应该把它定义成：

# **一个通用的迭代执行框架，image 只是第一种 collector。**

---

# 先给结论

## 当前阶段建议
**不要立刻把视频 / 音频 / 文字全一口气硬塞进去。**  
正确做法是：

### 把 v3 抽成三层
1. **Loop 调度层**
2. **协议 / 状态层**
3. **Collector / Finalizer 类型层**

然后基于统一 collector 协议，逐步加：
- text/string
- audio
- video
- latent / frame sequence
- json/object metadata

---

# 你现在已经有的基础，实际上非常适合抽象

你现在已经完成了最难的一层：

## 已经有的东西
- `loop_ctx`
- 显式 `BodyIn / BodyOut / End`
- dynprompt body 识别
- expand 调度
- `Resume` 注入
- collector 能跨轮累计
- `BodyOut` 已经开始支持 `value_image / value_string / value_any_*`

这说明：

> **你的 loop 内核其实已经是“通用结果槽”架构的雏形了。**

只是目前真正产品化的 collector 还是只有 image。

---

# 下一步应该怎么抽象

---

## 一、把“结果类型”从 Loop 核心里拆出去
现在 `LoopEnd` 还带着：
- `final_images`
- `final_grid`

这个在 image 路线下很方便，但如果你要支持：
- text
- audio
- video

它就会变成瓶颈。

---

## 我建议的方向
把 `LoopEnd` 的职责重新定义为：

### `LoopEnd` 只负责：
- 推进 index
- merge state
- 识别 body
- expand 下一轮
- 最后一轮发出 `done=True`

### 它不负责：
- image merge
- grid
- text join
- video assemble
- audio concat

这些都交给 collector/finalizer 层。

---

# 二、正式定义 collector 协议
这是下一步最重要的工程动作。

---

## 统一 collector slot 结构
你现在的 `loop_ctx.collectors` 里已经有：

```json
{
  "images": {"ref": "...", "count": 2}
}
```

建议正式升级成：

```json
{
  "image": {"ref": "...", "count": 2},
  "text": {"ref": "...", "count": 3},
  "audio": {"ref": "...", "count": 5},
  "video": {"ref": "...", "count": 1},
  "json": {"ref": "...", "count": 3}
}
```

注意我建议 key 用**类型名单数**：
- `image`
- `text`
- `audio`
- `video`
- `json`

比 `images/strings` 更统一。

---

## 同时把 runtime store 也统一
从现在：

```python
RUNTIME_STORE = {
    "images": {},
    "meta": {},
    "_detect_cache": {},
}
```

升级成：

```python
RUNTIME_STORE = {
    "collectors": {
        "image": {},
        "text": {},
        "audio": {},
        "video": {},
        "json": {},
    },
    "meta": {},
    "_detect_cache": {},
}
```

这样所有 collector 走统一 helper。

---

# 三、增加统一 collector helper
你现在最值得做的代码重构不是“先加视频节点”，而是**先把 collector helper 统一化**。

---

## 建议新增 helper

### 1. `_ensure_collector_slot(ctx, kind)`
确保：
```python
ctx["collectors"][kind] = {"ref": None, "count": 0}
```

### 2. `_collector_make_ref(run_id, loop_id, kind)`
例如：
```python
f"{kind}col_{run_id}_{loop_id}_{uuid.uuid4().hex[:8]}"
```

### 3. `_collector_ensure_store(kind)`
确保：
```python
RUNTIME_STORE["collectors"][kind]
```
存在。

### 4. `_collector_append(kind, ref, value)`
统一 append。

### 5. `_collector_pop_all(kind, ref)`
统一 finalize pop。

### 6. `_collector_cleanup(kind, ref)`
统一 cleanup。

---

# 四、collector 类型扩展路线

我建议不要同时开太多，按照“从最稳定到最复杂”的顺序来。

---

## Phase A：先做 Text
这是最应该先加的。

### 节点
- `MieLoopCollectText`
- `MieLoopFinalizeTextList`
- `MieLoopCleanupText`

### 为什么先做 text
因为 text 最能验证：
- collector 抽象是不是通用
- 非 tensor 类型是不是也能跨轮累计
- finalize 是不是能和 image 解耦

### finalize 输出建议
优先输出：
- JSON array string

比如：
```json
["frame prompt 1", "frame prompt 2", "frame prompt 3"]
```

比简单换行更稳。

---

## Phase B：再做 Audio
### 可能的 collector 形式
音频在 ComfyUI 生态里常见有两种表示：
1. tensor waveform
2. dict / structured audio payload

所以你要先确认你打算支持哪种“音频值协议”。

### 我建议
第一版音频 collector 只支持：
- waveform tensor + sample_rate
或
- 明确约定的 audio dict

不要一上来支持所有插件格式。

---

## Phase C：再做 Video
### 关键问题
“视频”到底是什么：
- 真正的 encoded video file path
- frame sequence
- IMAGE batch
- latent video representation

这几种完全不同。

### 我建议
第一版 video collector 不要直��定义成“最终 mp4 拼装器”，而是先定义成：

## `video = frame sequence collector`
也就是：
- 本质上收集 IMAGE batch 序列 / frame metadata
- 后续由独立 video assemble 节点去编码

这样更通用。

---

## Phase D：再做 JSON / structured metadata
这个也很有价值，比如：
- 每轮 prompt
- 每轮 seed
- 每轮参数结果
- 每轮评分

---

# 五、建议的节点矩阵

---

## 文本
- `MieLoopCollectText`
- `MieLoopFinalizeTextList`
- `MieLoopCleanupText`

## 图像
- `MieLoopCollectImage`
- `MieLoopFinalizeImageBatch`
- `MieLoopCleanupImage`

## 音频
- `MieLoopCollectAudio`
- `MieLoopFinalizeAudioList` / `MieLoopFinalizeAudioConcat`
- `MieLoopCleanupAudio`

## 视频/帧
- `MieLoopCollectFrame`
- `MieLoopFinalizeFrameBatch`
- `MieLoopCleanupFrame`

## 结构化数据
- `MieLoopCollectJSON`
- `MieLoopFinalizeJSONList`
- `MieLoopCleanupJSON`

---

# 六、我建议你重新定义 `BodyOut`
现在 `BodyOut` 已经开始有：

- `value_image`
- `value_string`
- `value_any_1..5`

这说明你已经在试图通用化。

但长期我建议你考虑两条路：

---

## 路线 A：继续保留多口输出
优点：
- 节点连线直观
- ComfyUI 用户易理解

缺点：
- 每增加一种类型，BodyOut 都会变胖

---

## 路线 B：把 BodyOut 简化成“协议出口 + side values”
比如：
- `loop_ctx`
- `state_json`
- `slot_1..slot_n`

然后由具体 collector 节点自己决定吃哪个 slot。

### 我现在建议
短期保留 A，  
中期逐步向 B 演化。

因为现在你的 image loop 已经跑起来，别一下子把 BodyOut 再翻掉。

---

# 七、建议的产品化路线

这是我认为你下一阶段最合理的路线：

---

## 第一步：冻结 Loop 内核
先不要再大改：
- `Start`
- `BodyIn`
- `BodyOut`
- `End`
- `Resume`
- body detect
- expand

把这层视作相对稳定。

---

## 第二步：重构 collector 层
目标：
- 让 image collector 不再是特例
- 所有 collector 走统一 store/helper

这是你下一步最值得做的事情。

---

## 第三步：补 text collector
这是最划算的第二类型支持。

验收标准：
- 一次 loop 同时 collect image + text
- 最终能拿到：
  - image batch
  - text list

这会证明 v3 不再是 image-only。

---

## 第四步：定义 audio/video 数据协议
在正式加节点前，先拍板：
- 音频值是什么结构
- 视频值是什么结构
- frame sequence 与最终编码是否拆开

---

# 八、我对“视频、音频、文字都要支持”的直接建议

## 不建议现在这么说：
> “我们支持所有内容类型”

## 建议改成：
> “Mie Loop v3 是通用循环内核；当前 collector 已验证 image，下一步将扩展 text，然后是 audio/frame/video。”

这会让架构更稳，也更容易维护。

---

# 九、如果现在就要开工，我建议你下一步做什么

## 最值得立刻做的任务
### **Collector 层抽象化 + Text collector**
这是最优先的。

因为它可以直接验证：
- 你的 v3 是否真正通用
- End 是否可以不再 image-special
- runtime store 是否抽象正确

---

# 十、我建议的下一版目标

## v3.1 目标
在当前 image loop 跑通的基础上，实现：

1. 统一 collector helper
2. image collector 迁移到 helper
3. text collector 完整支持
4. End 尽量减少 image 特化输出
5. 一次 loop 能同时输出：
   - image batch
   - text list

---

# 十一、v3 正式重构落地结论

## 1. LoopEnd 是纯控制节点
- `MieLoopEnd` 只负责：
  - 合并 state
  - 推进 index
  - 判断 done
  - 复用 body detect
  - 构造 expand graph
- `MieLoopEnd` 最终只返回：
  - `loop_ctx`
  - `done`
- `MieLoopEnd` 不再负责：
  - image merge
  - image grid
  - text finalize
  - 任何具体类型结果输出

## 2. collector / finalize / cleanup 三层分离
- `CollectX`：每轮追加本轮结果到 runtime store
- `FinalizeX(done)`：`done=True` 时消费 collector 数据并输出最终结果
- `CleanupX`：清理 runtime 残留并重置 `ctx.collectors[kind]`

### 生命周期语义
- `LoopEnd(done=True)`：只表示循环结束，不消费 collector 数据，也不重置 collector slot
- `FinalizeX(done=True)`：负责消费 `RUNTIME_STORE["collectors"][kind][ref]` 中的结果，并从 `run_meta["collector_refs"][kind]` 移除 ref
- `CleanupX`：负责清掉残留 store 数据，并把 `ctx.collectors[kind]` 重置为 `{"ref": None, "count": 0}`
- `_prune_runtime_store()`：负责清理 orphan / stale runtime 数据，是 collector 生命周期的兜底清理机制

## 3. 当前已确认的 collector 协议
- `ctx.collectors[kind] = {"ref": ..., "count": ...}`
- `RUNTIME_STORE["collectors"][kind][ref]` 保存累计结果
- `run_meta["collector_refs"][kind]` 跟踪活动 ref
- 所有 collector 统一通过 helper 操作 runtime store，不允许走 image 专用旁路

## 4. 当前支持的 finalize / cleanup 形态
- image：
  - `MieLoopCollectImage`
  - `MieLoopFinalizeImages(done)`
  - `MieLoopCleanupImages`
- text：
  - `MieLoopCollectText`
  - `MieLoopFinalizeTextList(done)`
  - `MieLoopCleanupText`
- json：
  - `MieLoopCollectJSON`
  - `MieLoopFinalizeJSONList(done)`
  - `MieLoopCleanupJSON`

## 5. 工作流规范
- 图像：
  - `LoopStart -> BodyIn -> 业务节点 -> CollectImage -> BodyOut -> LoopEnd -> FinalizeImages(done) -> ImageGrid -> Preview/Save`
- 文本：
  - `LoopStart -> BodyIn -> 业务节点 -> CollectText -> BodyOut -> LoopEnd -> FinalizeTextList(done)`
- 结构化数据：
  - `LoopStart -> BodyIn -> 业务节点 -> CollectJSON -> BodyOut -> LoopEnd -> FinalizeJSONList(done)`
- 双 collector：
  - `LoopStart -> BodyIn -> 业务节点 -> CollectImage + CollectText -> BodyOut -> LoopEnd -> FinalizeImages(done) + FinalizeTextList(done)`

## 6. 后续扩展边界
- 未来新增：
  - audio
  - frame sequence / video
  - JSON / structured metadata
- 扩展原则：
  - 只新增 `CollectX / FinalizeX / CleanupX`
  - 不修改 `MieLoopEnd`

## 7. 非 JSON 跨轮对象反馈
- collector 不用于反馈当前有效对象；collector 只负责汇总所有轮次结果
- 跨轮图像反馈采用 `state object ref` 机制：
  - 图像本体存 `RUNTIME_STORE["state_objects"]["image"][ref]`
  - `loop_ctx.state` 只保存 `"{key}_ref"`
- 当前 image feedback 节点：
  - `MieImageSelectFrame`
  - `MieLoopStateSetImage`
  - `MieLoopStateGetImage`
  - `MieLoopStateCleanupImage`
- 生命周期语义：
  - `LoopEnd(done=True)` 与 `FinalizeX(done=True)` 默认不清 feedback image
  - feedback image 由 `MieLoopStateCleanupImage` 或 run/prune 清理

---
