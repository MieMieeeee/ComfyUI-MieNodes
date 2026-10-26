# H3 Loop 对白链路未解问题清单（2026-09-22，待 review）

状态：两个问题在 4 轮真机视频验证（C2/C3/C4/C5）中均未根治。
本文件列出全部证据、已试方案及失败原因、当前假设与候选方案，
供详细 review 后定方向。

输入提示词（C2–C5 全部相同）：

```
图1的黑猫是爸爸，图2的白猫是妈妈，图3的小猫是女儿，
三只猫都坐在客厅的沙发上，全部在画面里：小猫坐中间，
黑猫坐小猫左边，白猫坐小猫右边。

图3的小猫：Mommy, daddy is so ugly, why did you marry him.
图2的白猫：I guess I was blind.
图3的小猫：Daddy, why did you marry a blind lady?
图2的白猫：Oh my god.
图1的黑猫：Well, nobody's perfect.
```

期望：小猫问"Mommy…"时面向右边的妈妈；问"Daddy…"时转向左边的
爸爸；妈妈两句（"I guess I was blind." / "Oh my god."）同一个
成人女声；爸爸一句成人男声。

---

## 问题 1：同一说话人跨场声音不一致（未修复）

**现象**：妈妈 scene_01 的 "I guess I was blind." 与 scene_02 的
"Oh my god." 声音不同（C5 用户复核：问题仍在，scene_02 偏童声）。

### 四轮实验记录

| 运行 | 声音描述符位置 | 归属 | 声音 |
|---|---|---|---|
| C2/C3 | 块内、标签前：`白猫, 图说; voice: adult female… (S2): <d>` | ✅ 正确 | ✗ 不一致（描述符被忽略，TTS 按场内语境猜） |
| C4 | 块内、标签后：`白猫 (S2) adult female…: <d>` | ✗ **台词被小猫抢走** | — |
| C5 | 块外散文：beat 句里 `as (S2) adult female, warm mid-range pitch, soft rounded timbre`；块回归 `白猫 (S2): <d>` | ✅ 正确 | ✗ **仍不一致** |
| 苹果版（最早一次成功的直播输出，LLM 手写台词） | 散文内**紧贴 `<d>` 标签**：`mouth opening as (S1) adult female, bright mid-range pitch <d>[Chinese] 吃苹果吗？</d>` | ✅ 正确 | ✅ **一致**（当时用户未报声音问题） |

### 关键差异（当前最强假设）

苹果版与 C5 的散文里都有 `as (S2) adult female…`，唯一结构差异是
**`<d>` 块的位置**：

- 苹果版：`<d>` 内联在散文句中，紧跟声音描述符（同一句、紧邻）。
- C5：`<d>` 块追加在 detailed_description 末尾的独立行，与散文中
  的描述符相隔数段。

**假设 H1（TTS 局部性）**：H3 的音频管线只读 `<d>` 标签**紧邻上文**
（同句/邻接 token）来决定音色；追加在段尾的块周围没有任何声音
线索，于是每场独立猜。C5 的散文描述符虽然写进了 beat 句，但与
`<d>` 不相邻，等于没写。

**假设 H2（短句渲染）**："Oh my god." 只有 3 个词，接近上游
addendum 警告的 <1.5s 短行不可靠区（TTS 会 clip/merge/默认感叹
轮廓偏高）。即使声音绑定正确，短感叹句也可能天然偏离。此假设
可与 H1 并存；若 H1 修复后仍偏，则 H2 是残余因素。

### 候选方案（待 review 定向）

**方案 V-A：`<d>` 内联替换（推荐优先验证）**
让 shot LLM 在散文里为每句台词写占位符（如 `{{LINE_1}}`），节点
把占位符**原位替换**成 `as (S2) adult female, warm mid-range pitch
<d>[English] Oh my god.</d>`。逐字仍由节点注入（不变写），但
`<d>` 与描述符同句紧邻，完全复刻苹果版成功形态。
- 风险：LLM 可能漏写/多写占位符 → 需机械校验（缺占位符重试，
  多余占位符删除）；替换后 prose 句法要自然（指令里给句式模板）。

**方案 V-B：块行自带邻接描述符但不破坏 `名字 (S2):` 结构**
块写成 `白猫 (S2): <d>` 不动，在块**上一行**插一行
`Voice: (S2) adult female, warm mid-range pitch, soft rounded timbre`。
比 V-A 改动小，但"上一行"是否够"紧邻"未知——等于又一次格式
试错，且 C4 教训是块区任何改动都可能破坏归属。

**方案 V-C：短句并场（针对 H2）**
把 <1.5s 的行并进相邻场（妈妈的两句同场生成，同场 TTS 一致性
由 H3 自身保证）。可与 V-A 叠加，但改变切分语义（最小剪切让位
于声音一致性），需要你确认取舍。

---

## 问题 2：台词受话对象搞错——"Daddy…" 面向了妈妈（C5 新暴露）

**现象**：小猫问 "Daddy, why did you marry a blind lady?" 时面朝
右边的妈妈说话，而不是转向左边的爸爸。

**证据（C5 scene_01 Beat 3 原文）**：

> Beat 3 — <Subject 3>, as (S1) child voice …, **snaps her gaze
> left toward <Subject 1> for a beat, then whips her head back to
> <Subject 2>** — eyebrows lifting high, paws separating and
> gesturing outward, mouth shaping the next words …

LLM 让小猫"朝爸爸瞥一眼然后**转回妈妈**说下一句"——而下一句
恰是 "Daddy, why did you marry a blind lady?"。方向写反了。

**根因**：管线只告诉 per-shot LLM **谁在说**（speaker+声音），从不
告诉它**每句对谁说、该朝哪边看**。受话对象完全靠 LLM 从台词猜；
它把整场理解成"小猫↔妈妈"二人对话（妈妈是前两句的应答者），
爸爸被当成背景旁观者，于是"Daddy…"也朝妈妈说了。

C2（同提示词）没有这个问题——说明这是 LLM 概率性行为，每次
重roll可能好可能坏，不是必然。

### 候选方案（待 review 定向）

**方案 A-A：受话对象抽取 + 逐句朝向指令（推荐）**
台词里的呼格（vocative）是最强信号：`Mommy, …` → 对妈妈；
`Daddy, …` → 对爸爸。语义事实抽取器（已有）加一个
`addressees` 字段：LLM 判断每句的受话对象（无呼格时按对话轮次
推断应答者）；机械验证 = 受话对象名必须出现在概念里。注入
per-shot 模板为逐句朝向指令："line 2 (小猫 → 爸爸, at LEFT of
frame): speaker turns toward 爸爸 before delivering"——方向由
空间布局表（黑猫=left）联查得出。

**方案 A-B：纯呼格规则（更机械、覆盖窄）**
只解析行首 `Mommy,` / `Daddy,` / `妈妈，` 等呼格，查角色表得
对象与方位。无呼格的行不注入朝向（维持现状）。零 LLM 依赖、
不会引入新错误源，但覆盖有限。

**方案 A-C：模板指令层兜底**
不抽取，只在 user template 加一条通用规则："每句台词 delivery
前，说话人必须转向该句呼格所称呼的角色（若有）；呼格称谁就
面向谁"。最便宜，但属于 prompt-only（概率性），与 A-A 相比
无保证。

---

## 附：当前链路已验证正常、不要动的部分

以下在四轮真机中均稳定，review 时不需要怀疑：

- 台词逐字（含 "Iguess" 拼写冻结）、语言标签、说话人 S 编号跨场稳定
- 归属（C2/C3/C5 形态 `名字, [身份] (S2): <d>`）；C4 证明块内
  描述符会破坏它——**块格式不要再动**
- 空间布局三方钉（黑猫左/小猫中/白猫右）正确进入 prefix 与两场
- 视觉身份 caption 锚定（黑猫=tabby 非黑、白猫=cream 非白）
- 声音表生成（VOICE sheet 内容本身正确、跨场字符串一致）——
  问题只在"描述符到 `<d>` 的距离"

## 待 review 的决策点

1. 声音：选 V-A（内联替换，改动大但复刻已验证形态）/ V-B（邻接
   行，改动小但又是试错）/ 先 V-C 验证 H2？
2. 受话对象：选 A-A（语义抽取）/ A-B（纯呼格规则）/ A-C（指令兜底）？
3. 是否接受为验证 H1/H2 增加真机渲染轮次（每轮 ~15 分钟 + 一次
   你的听感复核）？
