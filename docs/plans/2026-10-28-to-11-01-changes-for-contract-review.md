# 剧本资产体系改动清单（2026-10-28 → 2026-11-01，供契约复审）

基线：`16ae29a`（改动前）→ `058fcb5`（当前），共 17 个提交。
给第三方复审者：**A/B 两节是契约与编译语义的变化**，需要对照你们的实现/理解逐条核对；C/D/E 是工具面与资料面，不影响契约本身。

---

## A. 剧本资产契约（JSON schema 与字段语义）变化

| # | 变化 | 说明 | 提交 |
|---|---|---|---|
| A1 | **setting 收窄** | 从「环境+初始调度」收窄为**纯调度**（站位/姿态/剧情必需空间事实）；环境散文迁出 | dfd78fa |
| A2 | **新增 scene 字段**（可选，用户语言 + execution.scene 英文镜像） | 语义=**默认环境**：不接场景资产时进提示词；接了被整段丢弃 | dfd78fa |
| A3 | **新增 light 字段**（预设枚举 follow_image/daylight/golden_hour/night_lamp/night_screen/late_night/morning/custom + light_text + execution.light） | 语义=**默认时段**：不接场景时预设句/自定义文案进提示词；接场景即丢；custom 须 light_text（空=响亮报错） | dfd78fa, 609d1e3, 058fcb5 |
| A4 | **新增 character_designs**（可选，仅出镜角色 ID→默认外观，用户语言 + execution.character_designs 英文镜像，key 精确一致） | 第三方契约（已认定）：未接该角色资产时默认外观进提示词；接了丢弃 | a05a8c9 |
| A5 | **look 语义降级** | 从「逐字进每镜、永不被覆盖」降为**默认画风**：不接场景时生效；接场景即整行丢弃 | 058fcb5 |
| A6 | **role voice 上限统一 200**（原 300，与角色资产 voice 一致） | 现有资产实测最长 180，无迁移影响 | 0511dff |
| A7 | **解析宽容化**（源事件行）：说/问/喊/回答等口语引导词=对白标签；settings 分隔符 `\|`/，/,/、（maxsplit=3 保 voice 内逗号）；全部报错回显原文+格式+指 concept 逃生口 | 校验更宽松，**事件语义不变** | cbbbc90 |
| A8 | **校验错误细化**：逐字不一致报首个 mismatch（第几条/两侧内容）；"A word placed in the wrong field either leaks into every prompt (look) or dies at assembly (scene, light)" 四字段内容契约写入两份系统指令 | a9868e9, 9e2b9cb |
| A9 | **诊断告警新增**（只预警不改变编译）：感叹号密度≥50%→喊叫语速 ~2.5 字/秒预警；同人连珠炮≥5 句→末句口型漂移预警；预算贴地预警（原有） | a9868e9 |
| A10 | **version=2 保持**；v1 轨道**冻结**（可存可载可编译，新规则只进 v2） | 0511dff |

**不变的契约要素**：version=2 编号、角色编号连续（character_1..N）、台词/事件/顺序逐字锁定、可见性状态机、audio_mode generated/reference 语义（reference 仍要求完整接线、不静默降级）、constraints 永不编译。

## B. 编译/装配行为变化（零 LLM 编译器）

**接线资产覆盖矩阵（终态）——「剧本=可替换默认值，接线必胜」**：

| 通道 | 不接线时 | 接线时 | 提交 |
|---|---|---|---|
| 角色身份/外观 | character_designs 默认或模型生成 | 接线角色资产图 | a05a8c9 |
| 角色音色文字 | 剧本 execution.voices | **接线资产 voice 替换** | 96dbcbc |
| 场景文本 | scene 进提示词 | **整段丢弃**，场景图+ambience 生效 | dfd78fa |
| 时段光 | light 预设句/文案 | **丢弃**，场景图光线生效 | 96dbcbc |
| 画风 | look 逐字进每镜 | **丢弃**，场景图画风生效 | 058fcb5 |
| 环境声 | 固定 fallback | 场景 ambience 替换 | 0511dff |
| 默认外观 | character_designs | 逐角色丢弃 | a05a8c9 |

其它编译行为：
- **Picture 编号只数实际接线的出镜角色**（+可选场景），无接线=纯文本无图片引用；角色编号/说话人标签稳定（a05a8c9，第三方契约）；
- **续接重叠归组装侧**：V3/BuildPlanFromStoryboard 新增 `continuation_overlap_frames` INT（-1=沿用剧本 JSON 默认；0..120=覆盖并重算时长网格+summary 注记）；预览不再显示该值（058fcb5）；
- v1 剧本的模板组装路径与 missing-slot 硬错误**不变**。

## C. 生成器节点变化（创作工具面）

- **三种输入模式**（cbbbc90/0134be9）：`script` 锁定剧本（默认，旧行为）/ `concept` 概念创作（任意文本→AI 作者化完整资产，接线事实 name/audio_mode/overlap caller-owned，其余偏好）/ `revise` 修订（接上一版 storyboard + 修改要求→带机械 diff 的新版本；未点名台词逐字保留）；
- `shot_count` 0=自动（新节点默认）；camera 三枚举/look/scene/light/duration 等 tooltip 全部双语化并注明编译与覆盖行为；
- 系统指令（节点+docs **SYNC PAIR**）：INPUT AND AUTHORITY 双分支+revise 契约、CRAFT RULES 1-13（含收尾拍/喊叫语速/单手/光锚/口型漂移等真机回流）、四字段内容契约；`tests/test_script_asset_prompt_sync.py` 整段相等+标记清单守护两份同步。

## D. 资产库与节点族变化（资料面）

- **剧本资产可删除**：DeleteAsset kind 三态（character/scene/storyboard），server /delete 支持；两击确认不变（0511dff）；
- **库清理**（4e3a731/d86091d）：单一故事只留一版（删 002/003 平铺+四个 003 表演变体+001-生成音；部署库删高德吵架 D/F/G/H）；004/005 时长按语速校准上调；奔驰/我叫张同/七岁 setting→scene 迁移；骑士猫剧本改名「拿着剑我就没法拥抱你」并去默认外观；
- **第三方契约迁移对齐**（a05a8c9）：repo 开发库 18 份+双部署库全部按新契约迁移（LAN 独有 4 份中 语无伦次 已合规、板栗对话×2 为 v1 不动）。

## E. UI/预览变化（不影响契约）

- 加载剧本资产节点**选中即分栏显示全部内容**（九栏：概要[对白连读剧情]/时长/相机/角色[带 角色N（character_N） 编号]/默认角色设计/事件/画风/场景/光/约束[不进编译]/⚠风险[诊断·不进编译]，每栏标题注明编译与覆盖命运）；纵向堆叠、460px 滚动；旧的运行后文本框移除（53966d4/d70aaf8/b2ec7b1/069d355/ec0d440）；
- 验证工具：`_live_sb_preview_check.py`（playwright 量 DOM 几何）、`_live_f0_check.py`（F0 声纹归属）。

---

**复审建议关注点**（契约敏感处）：
1. A5 look 降级与 a05a8c9 可替换契约的 ASSET INDEPENDENCE 段落是否自洽（我们已在两份 prompt 同步改写）；
2. B 覆盖矩阵与你们的 character_designs 条件编译实现是否一致（voice 替换发生在你们的设计丢弃之前，顺序固定）；
3. C 的 revise 模式对 base_asset 的 caller-owned 继承（audio_mode/overlap）与你们的 reference 要求是否冲突（实测：reference 基线未全接线仍响亮报错，不静默降级）。
