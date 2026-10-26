# 分镜资产生成与使用

这版直接扩展原 MiniMax H3 分镜文本生成器。分镜保存角色、选项、有顺序的原文事件，以及独立的英文执行说明 execution；V3 根据它生成 H3 提示词。角色形象由连接的角色资产决定，场景资产可选。

## 在 ComfyUI 中

1. 添加「MiniMax H3 分镜资产生成器」，连接大语言模型服务。
2. 设 character_count，按编号填写 character_settings。每个角色一行，例如：

```text
角色1：男猫 | 画面内 | 左侧 | 低沉、简短的男声
角色2：女猫 | 画面内 | 右侧 | 清晰、自然的女声
```

角色状态支持「画面内」「画外」「稍后入场」。画外角色位置填「无」，稍后入场的角色必须有明确入场事件。角色名称不必与实际资产名称相同：角色编号决定接线。

3. 在 concept 中填写顺序事件，一行一件事：

```text
角色2动作：右手拿着洗好的红苹果，向角色1递出，但不松手。
角色2：吃苹果吗？
角色1：不吃。
角色1动作：双手仍插在口袋里，没有接苹果。
```

对白用「角色1：」，动作用「角色1动作：」。入场、离场分别用「角色1入场：」「角色1离场：」。英文台词可写「角色1[en]：No, thanks.」。未标角色的导演要求放在 constraints 中。

4. 选择 shot_count、target_duration_seconds 和 camera_framing / movement / perspective；look 是画面风格，setting 是环境与初始状态，constraints 是全片约束。默认 language=zh，locked 固定机位，observer 为旁观视角。two_shot 是双人构图，medium_close_up 是中近景。
5. 将 storyboard 输出接入 V3 的 storyboard；角色资产按 character_1、character_2 接入 V3。场景资产接 scene_1，可留空。V3 的 plan_json 接 Loop 计划节点；assets 经 Asset → Slots 接参考图输入。
6. 如果要保存，storyboard_json 接「保存分镜资产」的 json_text 输入，新建名称须与 JSON 中的 name 一致；下次通过「加载分镜资产」直接接 V3。提供的示例工作流让生成、保存共用一个「资产名称」字符串节点，只在该处改名即可。storyboard_text 只用于阅读预览。

仅交换角色资产即可复用角色动作与台词，但资产应适合所需动作，例如拿苹果需要能执行持握动作的角色。此版本从 H3 生成声音，不注入角色资产中的参考音频。

`shot_count` 是 Loop 生成片段数。同一场景里的短视频可以设为1，并在 constraints 中说明内部切镜，例如“前两句用中近景，在 event_004 前切到面部特写”。资产会保存 `shots[0].camera_cuts = [{"before_event":"event_004","framing":"close_up"}]`；V3 将它编译为同一条 H3 提示词中的 `[Shot 1]` 和 `[Shot 2]`，不会额外创建 Loop 片段。切镜位置必须属于当前片段、按事件顺序排列且不能重复；没有明确切镜要求时保持原机位。

角色参考图本身也是强视觉条件。参考图自带的杯子、书等物件可能被模型保留；编译器会要求这些附属物留在初始持有者手中，但要获得最稳定的动作控制，仍建议使用手部清楚、没有无关手持物的角色资产。

## 在其他大语言模型中

1. 把 [完整系统指令](../nodes/llm/prompts/h3_storyboard/system_asset.txt) 全文放入系统指令；不支持系统消息的聊天工具可把它作为首条指令。需要包含实战经验（时段光效锚点、环境文字语言、单手动作、节拍拆分、切镜景别匹配）时，用 [整合版系统指令](storyboard_asset_system_prompt.txt)（含一个通过校验器的完整示例资产）。
2. 提交 [吃苹果输入模板](examples/storyboard-generation/apple-input.txt) 或 [刺猬输入模板](examples/storyboard-generation/hedgehog-input.txt)，再替换为自己的角色和事件。
3. 模型应输出一个 version=2 JSON 对象。复制 JSON 到「保存分镜资产」，随后加载并接 V3。
4. 可先在仓库目录运行下面的纯 Python 检查，不需要启动 ComfyUI 或安装 torch：

```text
python scripts/validate_storyboard_asset.py my_storyboard.json --output my_storyboard_checked.json
```

它校验角色编号、台词与动作原文、事件归属与顺序、出入场和片段时长，并写出规范 JSON。错误会给出原因。只通过 JSON Schema 不足以验证事件顺序和动作归属；[Schema](../nodes/llm/prompts/h3_storyboard/asset.schema.json) 用于格式约束，同一 Python 校验器负责语义检查。

## 时长与可靠性边界

target_duration_seconds 是拼接成片目标时长。H3 原始片段必须落在 24fps 的 17*k+5 帧网格上，预览展示实际预计时长和每镜原始时长，因此可能与目标差约 0.35 秒。

只有一句短对白的独立镜头会被限制在“估算语音时长 + 两档帧网格余量”内；总时长的剩余额度优先分给包含多条事件的镜头。这样可避免一句短台词结束后，H3 用过长的空余音频继续补出无意义语音。若所有镜头都过短而目标时长过长，生成器会要求缩短目标时长、加入可见动作或合并短对白镜头。

continuation_overlap_frames 是每次续接在拼接时去掉的前导上下文帧数，默认22。它用于计算原始片段时长，需要与 Loop 的实际续接裁切一致；不是 video_blend_frames。独立片段无前导裁切时填0。更改此值不会自动改动 Loop 的策略节点。

模型负责分组、分配时长，并把非对白要求编译成 execution，不能改用户原文、角色、事件顺序或选项。每条英文动作按事件 ID 绑定，并以对应的 Character N 开头；中文对白保持原样。原文和执行说明都保存在资产中，可对照检查。校验失败最多纠正两次，仍失败就报错，不输出猜测的替代资产。服务额度、网络等错误直接报告。

用户输入的 look、setting、constraints 和事件原文可以是中文，并会原样保存在资产中；真正送给 H3 的动作、声音和导演执行说明一律使用英文，中文只在 `<d>[Chinese] ...</d>` 对白标签中出现。真实测试曾发现直接把中文导演约束拼进提示词会被念成旁白，因此不能把“JSON 结构正确”直接当作成片通过。

“不想生成”的内容可以写在用户原始 constraints 中，用来表达意图，但不会被逐字塞进画面提示。生成器必须把它改写成 `execution.visible_state` 中可拍到的正向状态。例如“男猫不接苹果”编译为“女猫是唯一苹果持有者，男猫双手可见且空着”；“不出现刺猬或回忆”编译为“两名角色留在当前沙发上对话，双手空着”。执行动作和可见状态禁止使用 no、not、never、without、avoid 等否定控制词，避免模型反而把被禁止的对象画出来。

结构校验通过代表指令契约完整，成片仍需检查角色动作、道具数量与归属、对白覆盖和口型。对白里可以提及不出镜的事物；只有正向写进可见状态或动作事件的对象才应进入画面。要表现回忆，必须把回忆设计成明确的出镜事件和镜头。

## 现成的资产系列

[分镜资产系列](storyboard-assets/) 存放可直接入库的 version 2 资产，是「千层套路」玩梗短剧系列：001（既有 v1）/ 002 中文逗号 / 003 凌晨与午饭（含各表演版本）/ 004 系统时区 / 005 隐藏撇号。每集都通过了契约校验，并已导入部署资产库，用「加载分镜资产」直接选用。批量导入用 `python scripts/import_storyboard_assets.py docs/storyboard-assets`，逐个校验用 `python scripts/validate_storyboard_asset.py <文件>`。
