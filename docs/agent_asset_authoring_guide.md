# 剧本资产 / 角色资产 / 场景资产 · Agent 编写指南

给**要用大语言模型产出这些资产**的人（或让 agent 读这份指南去写）。三类资产各司其职，
它们之间的边界决定了成片是否稳定——本文先讲边界，再逐类给可执行的规格与检查清单。

> 术语：面向用户的名称是**剧本资产**（机器层仍是 `storyboard` / `storyboards/`，
> JSON 字段与类名不变，改名只发生在界面文案上）。

---

## 0. 一张图看懂三类资产

| 资产 | 是什么 | 谁决定长相/环境 | 谁写台词 |
|---|---|---|---|
| **剧本资产** | 一出戏：谁在场、按顺序说什么、动作节拍、时长 | 固定剧情和表演，可附默认外观；接入角色资产时覆盖默认外观 | ✓ 台词写在这里 |
| **角色资产** | 一个角色的身份：长相、声线、可选参考音 | ✓ 角色资产决定长相 | ✗ |
| **场景资产** | 一个环境：光、材质、陈设 | ✓ 场景资产决定环境 | ✗ |

**核心原则（资产独立性）**：剧本可以独立运行，也可以任意搭配角色、场景资产。每个槽位独立解析，部分接入也有效。
- 角色接入时，以该角色资产的身份和外观为准；未接入时使用 `character_designs` 中的默认设计；没有默认则由模型设计。
- 场景接入时，以场景资产的环境和光线为准；未接入时使用 `scene` / `light`；空缺由模型补全。
- 默认外观只写 `character_designs`，英文副本放 `execution.character_designs`（键必须一致，只能引用出镜角色）。环境只写 `scene`，光照只写 `light` / `light_text`。
- `setting`、`execution.visible_state` 和动作执行说明只写表演、站位及剧情必需物件，不重复默认外观、陈设与照明。
- `audio_mode=reference` 仍要求每个角色有声音参考；`generated` 不使用参考音频。

例：`"character_designs": {"character_1": "穿红外套的猫"}`；英文副本 `"character_designs": {"character_1": "A cat wearing a red coat."}` 放在 execution 内。接入 character_1 后该默认描述不会进入提示词。

**优先级**：用户资产库 > 插件内置资产（`bundled_assets/`）。同名时用户的版本胜出，
所以内置的角色/剧本随时可以被用户自己的替换。

---

## 1. 剧本资产

**完整系统指令**（交给 LLM 当 system prompt）：
[`docs/script_asset_system_prompt.txt`](script_asset_system_prompt.txt)
（节点内置版：`nodes/llm/prompts/h3_storyboard/system_asset.txt`；两者是同一契约的两种写法，
共享规则必须同改——见该目录的 UPSTREAM.md "SYNC PAIR" 一节。）

它已经写清了：字段结构、角色编号、事件语法、时长分配、execution 的写法，
以及一批**实跑验证过的渲染规矩**（下面第 4 节是摘要）。

**给 agent 的最短指令模板**：

```text
用 docs/script_asset_system_prompt.txt 的契约，把下面这段戏写成一个 version=2 剧本资产 JSON。
只输出一个 JSON 对象，不要解释。

<你的故事，逐行事件>
```

**事件行格式**（一行一个事件，这是 LLM 与契约之间的唯一接口）：

```text
角色1：台词
角色1动作：动作描述
角色1入场：入场动作
角色1离场：离场动作
角色1[en]：English line
```

写完必须过一遍校验（不需要 ComfyUI / torch）：

```bash
python scripts/validate_storyboard_asset.py 你的剧本.json
```

它会检查：角色编号连续、事件逐字复现、可见性（谁在什么时候在场）、时长可行、
execution 的英文与正向措辞、**渲染风险预警**（时长贴地、慢速语速塞满、单人续生成片段等）。
有 warning 就按提示改，别直接上线。

---

## 2. 角色资产

### 2.1 它由什么组成

```
<资产库>/characters/<角色名>/
  character.json      # name / description / voice
  images/01.png …     # 1–8 张，同一分辨率
  audio/01.wav …      # 0–8 段（WAV / MP3）
```

`character.json`：

| 字段 | 要求 | 写什么 |
|---|---|---|
| `name` | ≤40 字符，与**目录名完全一致** | 角色名 |
| `description` | 1–800 字符，**纯英文** | 这个角色是谁、长什么样 |
| `voice` | 1–200 字符，**纯英文** | 这个角色怎么说话 |

### 2.2 description 怎么写（最容易写错的一处）

**只写身份与外观，不写姿势、不写动作、不写镜头。**

- ✅ 身份与外观：`"A small fluffy gray-beige tabby kitten knight with a cream muzzle
  and chest, round gray-blue eyes, a pink nose, fine whiskers and a small pink satin
  bow beside the ear on the viewer right. Upright feline body with furry paws, silver
  riveted plate armor and engraved shoulder guards, brown leather straps and belt,
  chainmail skirt and silver shin guards. Ivory hooded wool cape with a gold neck
  clasp."`
- ❌ 姿势与动作：`"raises one hand in a boyish salute"`、`"sitting with arms crossed"`、
  `"holding a sword upright"` —— 这些姿势会被 H3 整段冻住（见第 4 节第 3 条），
  角色看起来像雕塑。想让角色有动作，写在**剧本资产的 execution** 里，不写在这里。
- ❌ 背景：`"on a white background"` —— 背景交给场景资产；白底只是参考图的拍摄条件。

**参考图的建议**：单主体、干净背景、正面全身、细节清晰（毛发/铠甲/配饰能看清）。
多图时**同一角色、同一套衣服**，不同角度/表情，供 H3 选姿态；**所有图必须同一分辨率**，
否则保存会直接报错。

### 2.3 voice 怎么写

**只写音色、音区、语速——不要写表演行为。**

- ✅ `"young female, sweet mid register, restrained and cool, steady pace, clear articulation"`
- ✅ `"young male, athletic chest-supported voice, stable mid pitch, brisk and even pace"`
- ❌ `"she speeds up and stumbles when she explains"`、`"a laugh at the end"`、
  `"a short humph"`、`"not a high bright boy voice"` —— 这类备注 H3 会**当成指令在台词上执行**：
  实测"一解释就磕巴"让整句长台词被念成乱码；"句尾带笑"会让每句后面多出一声假笑。

**音区词要和样音一致**：样音 126 Hz（中音）就别写 `mid-low`，写了会把音高压到 96 Hz 左右。
写之前量一下样音基频，或照角色资产原声线规格的音区词抄。

### 2.4 参考音（可选但推荐）

`audio/01.wav` 用来给 H3 做**音色条件**。有它就用 `audio_mode: "reference"`
（最多 3 个角色，每个角色都要有 voice + 已接音频）；没有就保持 `generated`，
H3 自己生成声音。**注意**：`generated` 模式下插件会**丢弃所有参考音**，不是"备用"。

### 2.5 怎么建

画布上放 **「保存角色资产 · Save Character Asset」**：
填 `name` / `description` / `voice`，接参考图（IMAGE）与参考音（AUDIO），
运行即入库。之后用 **「加载角色资产 · Load Character Asset」** 按名字取出。
命令行批量导入：`python scripts/import_storyboard_assets.py` 只管剧本；
角色/场景请用画布节点或直接拷目录。

---

## 3. 场景资产

### 3.1 它由什么组成

```
<资产库>/scenes/<场景名>/
  scene.json          # name / description / ambience(可选)
  images/01.png …     # 1–8 张，同一分辨率
```

| 字段 | 要求 | 写什么 |
|---|---|---|
| `name` | ≤40 字符，与目录名一致 | 场景名 |
| `description` | 1–800 字符，**纯英文** | 这个空间长什么样 |
| `ambience` | ≤400 字符，英文一行，可空 | 氛围/环境音（如 "Soft wind through leaves"） |

### 3.2 两条硬规矩

1. **场景图里不能有人**。不许出现人物、脸、剪影、倒影里的人形——场景图里的人物会**覆盖**
   角色资产的身份（这是明确的缺陷，不是风格问题）。场景资产是**纯环境**。
2. **剧本默认环境和光照独立存储**：环境及固定陈设放 `scene`，时段光锚放 `light` / `light_text`，`setting` 只写表演站位。接入场景资产时，默认场景与光照都会被覆盖；未接入时它们才生效。剧情明确使用的道具和空间关系保留在事件或表演说明里。

### 3.3 怎么建

画布上放 **「保存场景资产 · Save Scene Asset」**：填 `name` / `description`（/ `ambience`），
接场景图（IMAGE），运行即入库。

---

## 4. 跨三类资产的渲染规矩（实跑验证，写资产前必读）

1. **续接段开头是盲区（宽度=重叠帧数）**：多段生成时管线裁掉每段开头
   `continuation_overlap_frames` 帧——剧本 JSON 存默认值（标准引导 Loop=22 帧≈0.9 秒），
   **实际值由组装节点（V3 / Build Plan From Script）的同名输入决定（-1=沿用剧本默认，0..120=覆盖）**，
   且须与下游 Loop 的裁剪一致。盲区秒数=实际帧数÷24：里面的事件不会进成片 →
   "必须被看见的关键动作"不能放在续接段盲区内。
2. **塞满的片段会丢最短的一句**：H3 念不完时丢的是**最短那句**（不是念慢）。
   中文按 **5–6 字/秒** 估（不是 8），每镜再留 1 秒余量。校验器的
   "慢速语速" 预警就是查这个。
3. **参考图的姿势会被继承**：`visible_state` 里若写了参考图的姿势（敬礼/抱臂），H3 整段保持那个手势。
   写自然姿态。**道具是例外**：道具保持参考图姿态，但也别要求角色改变它的姿态（改了会被重绘崩坏）。
4. **故事必需道具要显式写**：H3 不会从"一张书桌"推断职业场景。"版本上线"就得写"桌上一台亮着的电脑显示器"。
5. **voice 文字与参考音叠加生效**：见 2.3。
6. **同音字**："驾校/假校"会被 TTS 念成同音，ASR 也分不出——不是吞字，验收时结合语境判断。
7. **场景图自带字样改不掉**：文字语言锚点只管模型自己编的字（咖啡厅黑板上的英文 MENU 不会被改成中文）。

---

## 5. Agent 交付前自查清单

**角色资产**
- [ ] `description` 是纯英文、只写身份与外观，没有姿势/动作/背景
- [ ] `voice` 是纯英文、只写音色/音区/语速，没有"会怎样演"的备注
- [ ] `name` 与目录名一致；参考图 1–8 张同分辨率、单主体干净背景
- [ ] 参考音与 `voice` 的音区一致（量过基频）

**场景资产**
- [ ] `description` 是纯英文、只写环境
- [ ] 场景图里没有任何人物/剪影
- [ ] `ambience` 一行英文或留空

**剧本资产**
- [ ] 事件逐行、一行一个，角色编号连续
- [ ] 台词逐字保留，未被"润色"
- [ ] `execution` 全英文、无否定控制词（no/never/without…）
- [ ] `visible_state` 写自然姿态 + 首帧状态 + 故事必需道具
- [ ] 每镜时长按 5–6 字/秒留了余量；多段时第一拍不是"必须被看见的事件"
- [ ] `python scripts/validate_storyboard_asset.py` 无 error、warning 已逐条处理

**整体**
- [ ] 剧本里的角色名与角色资产名对得上；场景名与场景资产名对得上
- [ ] 不在剧本里写长相与家具（那些属于角色/场景资产）

---

## 6. 参考

- 剧本资产系统指令：[docs/script_asset_system_prompt.txt](script_asset_system_prompt.txt)
- 节点内置版与"两份必须同步"的约定：`nodes/llm/prompts/h3_storyboard/UPSTREAM.md`
- 已验证的成片与 QA 记录：`docs/qa/`（`sketches-20261027/`、`knight-cat-20261026/`、`storyboard-gags-20261026/`）
- 外部测试者上手说明：[docs/qa/sketches-20261027/TESTING_GUIDE.md](qa/sketches-20261027/TESTING_GUIDE.md)
- 插件内置资产（随包分发）：`bundled_assets/README.md`
