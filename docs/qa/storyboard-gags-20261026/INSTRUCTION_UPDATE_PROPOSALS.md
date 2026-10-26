# 分镜资产系统指令 · 实跑反馈与更新提案（2026-10-26）

来源：三个新分镜资产（`奔驰车主的月充一万` / `七岁养伤的一年` / `我叫张同`）在真机 ComfyUI 上
用 `Claude娘-日式赛璐璐` + `DeepSeek娘-日式赛璐璐` 完整跑通 H3 成片，三轮渲染、逐句 ASR 核对。
本文件只列**规则层**问题与建议措辞，逐条手修的渲染细节不进指令。

---

## A. 写资产阶段就踩到的（改指令前那版就有的坑）

### A1. 多分了生成片段，且第二段是"单一角色独占全部台词"
- 现象：三个资产我一开始都写成 2 段；`七岁养伤的一年` 的第二段是讲述者独占全部台词——正是
  H3 已知的"续生成片段里说话者可能从画面消失"的触发形态。
- 修复：三个资产全部合并为**单段生成**（15.79s / 10.83s / 7.29s，均在单段 30 秒上限内）。
- 提案：**CRAFT RULES** 已新增第 8 条（最少分段）。建议再加一句把"单一角色独占台词的续段"写死为禁用形态：

  > A continuation clip whose dialogue is owned by one character is a known H3 failure mode
  > (the sole speaker can vanish from the frame); when no split was requested, merge instead.

### A2. `visible_state` 混进了后面才发生的动作状态
- 现象：初版 `七岁养伤的一年` 的 visible_state 里写了"手在数数""笑容收住""摊开手掌"，
  但这些是第 2、7、9 个事件才发生的；首帧其实是双手放在腿上。
- 修复：visible_state 只写**每个片段首帧**的状态，后续动作状态全部移进 directions。
- 提案：**ENGLISH EXECUTION COPY** 已有该条，建议补一句机器可查的判据：

  > The opening portion of `visible_state` describes the first frame of every clip and must match
  > the state before that clip's first event; any prop or action state that first happens at a later
  > event belongs in `directions`.

### A3. 环境文字语言锚点写在 `constraints` 里，永远到不了 H3
- 现象：初版把"环境中任何可见文字都是中文"写在 constraints。新指令已说明编译器不读 constraints，
  锚点必须结构化。
- 修复：锚点搬进 `setting`（再由 execution.setting 译出），三条资产都改。
- 提案：**ASSET INDEPENDENCE** 建议补：

  > Locale text anchors (the language of every legible text in the environment) belong in `setting`
  > — the compiler never reads `constraints`, so a rule parked there never reaches H3.

### A4. 角色 `voice` 以句号结尾，提示词被拼成 `…pace.: <d>`
- 现象：角色声线规格照抄角色资产的 `voice` 字段（带句号），编译出来是
  `Female voice with a bright sweet timbre, … follow each dialogue line for emotion and pace.: <d>[Chinese] …`
- 修复：角色 `voice` 去掉结尾句号。
- 提案：**ROLES** 建议补：

  > `voice` is spliced into the prompt as `<voice>: <d>…`; a trailing period produces a broken line
  > ("…pace.: <d>"). Keep it a bare phrase with no trailing punctuation.

### A5. 角色 `voice` 与接线角色的性别/音色对不上
- 现象：我把两个角色的 voice 写成 "adult male"（先编的），实跑却接了女性赛璐璐角色。
- 修复：把三个资产的 voice 换成对应角色资产的声线规格。
- 提案：**ROLES** 建议补：

  > `voice` is the fallback used when no reference audio is wired; when a character asset with an
  > audio sample is wired, the reference audio wins but the prompt text still names the fallback —
  > so write `voice` to match the character asset that will be wired, or leave it empty.

### A6. 数字口语化（新指令已写，建议加到自改脚本的显式清单里）
- 现象/修复：自改脚本里的 `200多 / 1W / 3岁` 全部改成 `两百多 / 一万 / 三岁…八岁`。
- 提案：**SOURCE SCRIPT GRAMMAR** 已有该条，无需改。

### A7. 时长压在可行下限
- 现象：第一次校验报 `target_duration_seconds 恰好压在各片段最小可行预算上（零缓冲）`。
- 修复：target 15.1 → 15.8（留约 1 秒余量）。
- 提案：**DURATION** 已有"留 0.5 秒余量"，建议补一句：

  > A validator warning that the budget sits at the feasibility floor means the closing beat is what
  > gets clipped first — raise the target by about a second rather than shipping at the floor.

---

## B. 实跑才暴露的渲染侧问题（本次最有价值的部分）

### B1. `setting` 文案会盖过接线的场景参考图 ⚠️ 最严重
- 现象：`七岁养伤的一年` 接了场景资产 `室内客厅沙发`（暖色乡村客厅），但 setting 写的是
  "现代公寓客厅…白色墙面把光反射回来"。同一份提示词两次渲染给出**两个不同的房间**：
  一次场景图赢、一次文案赢。
- 根因：提示词里的文字描述比参考图更强；场景图只有一行
  `Scene reference: <Picture 2> provides only the environment, materials and lighting.`
  这条弱锚点，敌不过 setting 的具体描述。
- 修复：setting 只保留光锚（"正午的客厅：窗外是正午的强烈日光…"），删掉房型、墙面、家具。
  复渲后场景图稳定赢。
- 提案：**ASSET INDEPENDENCE** 建议新增：

  > The setting TEXT overrides the wired scene image. When a scene asset may be wired, `setting`
  > carries the light anchor only — no room type, no wall or surface colours, no furniture.
  > Name the room type only when the room itself is story-required (a clinic, a window that overlooks
  > the street below); then wire no scene asset, so nothing competes.

### B2. `first_person` 机位会把"观众的手"画进画面
- 现象：独白资产用 `first_person`，成片画面下方多出一双 POV 观众的手（还端着一个杯子）。
- 修复：改成 `observer`，并把"画面里只有讲述者一个人"写进 constraints 与 execution.setting。
- 提案：**CAMERA** 建议新增：

  > `first_person` puts the viewer's own hands and body in frame. Use it only when the listener's
  > presence is part of the story; a solo monologue facing the camera belongs to `observer`.

### B3. 手势不写细节就渲染不出语义
- 现象："掰着手指头数数"只渲染成一只抬起的手，和数字毫无关系（用户看片直接指出）。
- 修复：direction 改成可执行细节——"手心朝向镜头，每说一个岁数就收起一根手指"。
  复渲后用连续帧验证：五指张开 → 逐根收起 → 只剩一根，横跨整段报岁数。
- 提案：**CRAFT RULES** 建议新增：

  > Hand gestures need executable detail, not a label: write the palm orientation and what changes
  > per beat ("the palm turned toward the camera, folding one finger for each year she names"), or
  > H3 renders a generic raised hand that carries no meaning.

### B4. 垂直关系（楼下/楼上/正下方）必须写观察角度
- 现象：台词说"楼下停的那辆奔驰"，初版成片把车平铺在窗外街对面（与"楼下"矛盾，用户指出）。
- 修复：setting / execution.setting / visible_state 三处写成"高层公寓，从上往下看，
  车停在窗户正下方的路边，俯视能看到车顶"。复渲后车就在窗下。
- 提案：**CRAFT RULES** 建议新增：

  > A vertical relation (楼下 / above / directly below) must state the viewing angle, or H3 flattens
  > it into a distant view at eye level: say "seen from above, directly below the window".

### B5. 道具数量写"一个"不够，要写"全场唯一那一个"
- 现象：visible_state 写 "holding one phone in one hand"，H3 在茶几上又变出一部手机；
  改成 "holds the only phone in the room in one hand" 后仍出现复制品（终版仍有）。
- 修复（部分有效）：改成"全场唯一"并同步到 direction。
- 提案：**ENGLISH EXECUTION COPY** 建议新增，并说明这只是缓解：

  > State a prop that must not be duplicated as the only one in the room ("the only phone in the room"),
  > in both `visible_state` and its direction. H3 may still duplicate small props on tables; if a
  > duplicate breaks the staging, drop the prop instead of fighting it with wording.

### B6. 手指收拢到后期会变形，且数量与台词不对应
- 现象：数数手势越收越"糊"，最后只剩一根指头，而她实际列了五个岁数（三四五六八）。
- 提案：**CRAFT RULES** 建议新增（同时是一个更好写法的提示）：

  > Folding fingers degrade as the hand closes; prefer a gesture whose meaning survives the last
  > frames — e.g. an open palm with one finger missing at the punchline rather than a full
  > count-down, and keep the number of visible gestures equal to the years named.

### B7. H3 的烧录字幕不稳定，不能用字幕验收台词
- 现象：`我叫张同` 有中文字幕，`七岁养伤的一年` 与 `奔驰车主的月充一万` 没有；同一资产不同
  seed 表现也不同。拟声词"嗯？？？"只在音频里、没进字幕。
- 修复：验收改用 faster-whisper 对成片做 ASR 逐句核对。
- 提案：不进指令文件（属渲染侧行为），但建议在 QA 流程里写死：**验收用 ASR，不用字幕**。

---

## C. 不属于指令文件、但接线/跑批时必须知道的（给指令维护方参考）

1. **`run_name` 只能是 ASCII**：H3 链校验 `run_name must be 1-96 characters, start and end with a
   letter or number, and contain only letters, numbers, '.', '_' or '-'`。中文资产名放不进 run_name，
   跑批时要另起 slug。
2. **改过分镜 JSON 必须换 seed / run_name 重提**：同 id 的 api_prompt 会复用旧 bundle 与旧计划。
3. **查提示词别看 `prompt_history/*.json` 的 `prompt` 字段**：那里存的是分镜正文，不含
   `prompt_prefix`；发出去的全量提示词在 `output/h3_chains/<run>/segments/*.prompt.txt`
   与 `plan.json` 的 `shots[].prompt`。
4. 场景资产确实带文字锚点（`Scene reference: <Picture N> provides only the environment…`），
   不用再担心"场景进了图却没进文本"——但它很弱，扛不住 setting 的具体描述（见 B1）。
