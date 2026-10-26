# 资产库实现评审

> 日期：2026-10-09
> 对照：`docs/PLAN_ASSET_LIBRARY.md`（含 2026-10-07 音色样本修订）、`docs/ASSET_LIBRARY_GUIDE.md`、未提交的 `nodes/assets/` 与 H3 循环改动。
> 基线提交：`b1df1f6`（提示词增强已提交。资产库与本次 H3 花名册接线都还在工作区里）。
> 结论给实现方和第三方看。本文不改代码。

## 结论

这次更新作为资产库不合理。磁盘读写是扎实的，但角色和场景的定义，以及它和 H3 的关系，都已经离开计划里写死的边界。`ASSET_LIBRARY_GUIDE.md` 把 `PLAN_ASSET_LIBRARY.md` 称作决策依据，两份文档现在互相矛盾。

测试：`tests/test_asset_library.py` 与 `tests/test_h3_loop_asset_roster.py` 一起跑，72 项通过（2026-10-09）。通过只说明新代码按它自己的测试是自洽的，不说明它实现了计划。

## 计划里现在仍写着的边界

`PLAN_ASSET_LIBRARY.md` 开头和前两节：

- 资产模块不认识 H3，不改任何 H3 节点、提示词或解析器。
- 角色是名字、一句音色、一张定妆图、一段音色样本。外貌以定妆图为准，JSON 里不写毛色和衣服。
- 场景是 `scenes/<名字>.json`，字段是 `name` 和 `setting`，没有场景图。
- 六个节点：角色和场景各一组保存、读取、列出。
- `nodes/assets/` 不 import `nodes/llm/`。
- 使用方以后自己把图接到 H3，把音色和场景说明抄进概念稿。那是工作流的接法，不是这个库的职责。

音色样本是 2026-10-07 的用户决策，计划已经收进去：只存样本本身，不存 `voice_id`。这一条实现遵守了。

## 和计划一致、可以留下的部分

`nodes/assets/library.py` 只用标准库，不引用 ComfyUI、torch、`nodes.llm`。库必须是插件包外的绝对路径。大小写合并靠列目录，不靠 `exists()`。角色按 PNG、音色样本、JSON 的顺序提交，失败时错误写明磁盘上剩的是哪一版。保存和列出的 `IS_CHANGED` 返回 `float("nan")`，读取用内容哈希。`LoadAssetBatch` 遇到分辨率不一致会点名是哪几张图、各是多大。同一个名字同时出现在 `characters/` 和 `scenes/` 里会直接报错。

音色样本写入 `voice_sample.wav`，手工放入的 `voice_sample.mp3` 可以读，两个文件并存时报歧义。V2 不把样本写成 `<Audio N>`。使用说明把这一点标成未接入，这个判断是对的：ref2va 路径上这个标签仍被禁止。

`MiniMaxH3LoopPromptGeneratorV2` 的形状本身干净。它继承 V1，只多一个可选的 `asset_roster` 字符串插座。花名册是 JSON。下游 `plan_json` 没有另起一套 schema。场景槽标成 `environment`，不进身份钉，不占 `<Subject N>`。这是在回答「全能参考里每张图都会被当成角色」。

## 越过已定边界的三处

### 1. 外貌改成必填正文，并跳过看图

计划：外貌只留在定妆图里。

实现：`character.json` 必填 `appearance`（去空白后 1–800 字）。`validate_appearance` 只查类型和长度，不检查是不是英文。有花名册的槽位在 `MiniMaxH3LoopPromptGenerator._caption_images` 里跳过视觉打标，这段文字被当作外貌真相钉进 CAST。使用说明要求写英文，代码不拒绝中文，中文段落也会原样进提示词。

名字像「虎斑」「小猫」时，照片才是长相。再写一套毛色和衣服，计划里要避免的就是两套身份。使用说明的理由是「手写真相和机器看图不能混在一份 plan 里」。这个理由在「文字才是真相」的前提下成立，和「照片才是真相」是相反的决定。

### 2. 场景从一句话变成图加描述

计划：`scenes/<名字>.json`，字段 `setting`，1–500 字，没有图。

实现：`scenes/<名字>/scene.json` 加 `scene.png`。字段改成 `description`，1–800 字。缺图的目录读不出来，列目录时会被跳过。按计划建好的单文件场景读不出来。

### 3. 库又负责喂 H3

计划第 2 节禁止修改 `MiniMaxH3LoopPromptGenerator`、`minimax_h3_loop_prompts.py` 和 H3 提示词。

相对 `b1df1f6` 的未提交 diff：

| 文件 | 规模 |
|---|---|
| `nodes/llm/minimax_h3_loop_prompt_generator.py` | 约 +302 行 |
| `nodes/llm/minimax_h3_loop_prompts.py` | 约 +190 行 |
| `nodes/llm/prompts/h3_loop/prefix_synth_system.txt` | CAST 规则从「图片说明是真相」改成「手写花名册或图片说明都是真相」 |
| `nodes/llm/prompts/h3_loop/ref2v_addendum.txt` | 同上，并写明手写花名册压过图片说明 |
| `nodes/llm/prompts/h3_loop/shot_user_template.txt` | 概念和前缀之间的空行换成 `{asset_roster_block}` |

V1 和 V2 共用增强器和这三份提示词。空花名册时，镜头用户模板里概念和前缀之间的那行空行会消失。前缀提示词和 ref2va 附录的用词对所有走这套循环的生成都变了。

使用说明写「不接 roster 时 V1 逐字节不变」。`test_empty_roster_is_byte_identical_to_the_pre_roster_path` 比较的是新代码里「不传 roster」和「传空字符串」，两边用的是同一套新模板。它没有和 `b1df1f6` 的输出对比。

资产包和 H3 包之间没有 Python import，合同是一条 JSON 字符串。这比把库写进 `llm.py` 干净。它仍然是计划明确排除的职责：库决定谁绑定哪张图、外貌句、音色句，以及场景图是环境。

## 文档和报错对不上

注册的类是 `LoadAssetBatch`，显示名是 Load Asset Batch（`__init__.py`）。下面这些地方写成了没有注册的 `LoadCharacterAssetBatch` / “Load Character Asset Batch”：

- `docs/ASSET_LIBRARY_GUIDE.md` 场景 D、场景 E、排查表两行。
- `nodes/llm/minimax_h3_loop_prompt_generator.py` 里「images 插座是空的」和「slot 超出实际图片数」的报错。用户会去菜单里找一个不存在的节点。

使用说明还写：V2 删掉了原来的 `scene_setting` 插座。V1 没有这个插座。V2 的 `INPUT_TYPES` 从 V1 派生，只加了 `asset_roster`。测试只断言这个名字不在 V2 的 optional 里。

使用说明写资产库测试 42 个。2026-10-09 把资产库测试和花名册测试一起跑，合计 72 项通过。

## 需要拍板的一件事

存储层可以留：路径、原子写、大小写合并、音色样本、内容哈希、包外目录。

两条路：

1. 要留外貌正文、场景图和 V2 花名册。先改 `PLAN_ASSET_LIBRARY.md` 第 1、2 节，让它承认这是新决定，并改掉错误的节点名和「V1 逐字节不变」这句。H3 提示词改动要单独说明对旧工作流的影响。
2. 外貌仍然只认照片，场景仍然只是一句文字，库仍然不碰 H3。撤出循环节点和三份提示词里的花名册接线，以及必填的 `appearance` 和 `scene.png`。音色样本可以按计划留下。

在这件事情定下来之前，不把这批改动当成按计划完成，也不把它和已经提交的提示词增强混成一个提交。
