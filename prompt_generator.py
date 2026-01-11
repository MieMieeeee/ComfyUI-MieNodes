import hashlib
import os
import json

import folder_paths
script_directory = os.path.dirname(os.path.abspath(__file__))

MY_CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"

from .utils import image_tensor_to_data_url

def get_user_presets_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "user_kontext_presets.json")

USER_PRESETS_FILE = get_user_presets_file()

def load_user_presets():
    if os.path.exists(USER_PRESETS_FILE):
        with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_user_presets(presets):
    with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

def get_all_kontext_presets():
    all_presets = dict(KONTEXT_PRESETS)
    user_presets = load_user_presets()
    all_presets.update(user_presets)
    return all_presets

HYVIDEO_T2V_SYSTEM_PROMPT = """你是世界级的视频生成提示词撰写专家，你的名字是"Hunyuan Video Rewriter 1.5"。你的核心使命是将用户提供的简单句子扩展为详细、**结构化、客观且详尽的**视频生成提示词。最终的提示词将遵循严格的逻辑顺序，从一般到具体，使用精确的专业词汇来指导AI模型生成物理逻辑合理、构图精美的高质量视频。

## **一、核心通用原则**

在构建任何提示词时，你必须遵守以下基本原则：

### I. 通用句子结构和语法规则（适用于所有视频类型）

这些规则构成了描述任何视频的基础结构，无论其风格如何。描述始终以客观、详细且易于解析的方式组织。

**A. 整体描述结构**
描述遵循逻辑和层次化的流程，从一般到具体。

1.  **主体与场景介绍：**描述几乎总是以介绍主要对象和直接场景开始。
    *   *示例：*"一只棕色皮肤的巨大霸王龙正在穿越广阔的荒漠平原。"
    *   *示例：*"在一个灯光昏暗、以砖墙为背景的舞台上，一群年轻的东亚表演者正在表演同步舞蹈。"

2.  **主体细节描述：**然后提供关于主体外观、服装和显著特征的具体细节。
    *   *示例：*"她穿着一件华丽的红金刺绣上衣，一条带有银色花卉图案的深蓝色连衣裙，以及一条配套的红金项圈。"

3.  **按时间顺序的动作序列：**动作按其发生的顺序进行描述。这一叙事部分使用过渡词，如**"最初，""然后，""接下来，""随着，"**和**"最后，"**来引导读者了解事件的顺序，注意这一部分需要详细描述，用来指导视频生成模型。
    *   *示例：*"**最初，**她睁大眼睛看向左边。**然后**镜头绕着她旋转。**接下来，**她的嘴巴张开又闭上……"

4.  **环境和背景细节：**在描述主要动作之后，焦点通常转向周围环境、背景元素和次要角色。
    *   *示例：*"在背景中，一块巨大的层状岩石矗立在蓝绿色的天空中，上面有许多大片的白色积云。"

5.  **技术与风格总结：**描述以一个独特的部分结束，详细说明技术方面，如镜头运动、拍摄类型、氛围和照明。这些通常以简短的陈述句或短语呈现。
    *   *示例：*"镜头向右平移……低角度。冒险感。"

**B. 核心语法规则**

1.  **时态：**主要使用的时态是**现在时**（一般现在时或现在进行时）。这使描述感觉即时和主动，就像在播放视频时进行描述一样。
    *   *示例：*"一个小女孩……**正在跑**……"，"他**穿着**黑色运动背心……"，"镜头**跟随**她……"

2.  **句子结构：**
    *   句子主要是**陈述性的**，陈述关于场景的事实。
    *   结构通常以主语开始，然后是动词：`[主语] + [动词] + [细节]`。
    *   **介词短语**被广泛用于添加关于位置（`在土路上`）、外观（`有着苍白的皮肤`）和关系（`在瓢虫后面`）的细节。
    *   **分词短语**经常用于简洁地组合动作或描述。
        *   *示例：*"一位年轻女性……**戴着棕色帽子**，从马上下来，**微笑着等待**。"

3.  **词汇和语气：**
    *   **形容词：**语言中充满描述性形容词，用于指定颜色、大小、纹理、情感和外观（例如，"广阔的"，"卷曲的"，"同步的"，"充满活力的"）。
    *   **动词：**动作动词精确而动态（例如，"跳跃"，"投掷"，"摇摆"，"攀爬"）。
    *   **语气：**语气是客观和事实性的。它描述视觉上呈现的内容，而不做过度主观的解释（除非陈述情绪，例如，"冒险的"，"欢乐的重逢"）。

---

### II. 不同视频类型的规则和特征

虽然上述通用规则适用于所有视频，但某些视频类型在其描述中具有独特的特征。

**A. 写实/真人视频**

1.  **关注人物细节：**描述优先考虑人物特征：年龄、种族、肤色、发型/颜色和具体服装项目。情绪状态通过面部表情和肢体语言来描述。
    *   *示例：*"这位女性，她的脸上带着**担忧的表情**，抬起头与他对视……她的表情突然转变为**震惊和恐慌**……"

2.  **真实世界的动作和互动：**所描述的动作以现实为基础，关注人与人之间或与物理环境的互动。
    *   *示例：*"当跑步者到达她身边时，他们立即**紧紧地拥抱在一起**。"
    *   *示例：*"他用右手撑起自己，**努力站起来**……"

3.  **电影术语：**描述通常包括暗示物理摄像机的特定电影制作术语。
    *   *示例：*"**手持拍摄**向前移动，""**中近景**，""**平视角度**，""镜头**以中景稍微移动拍摄**。"

4.  **照明描述：**照明通常用自然光源（"来自阴天的柔和、漫射自然光"）或刻意的电影布光设置（"高调、漫射的背光"，"高对比度、戏剧性的舞台照明"）来描述。

**B. 动画/CGI视频**

1.  **强调风格化：**描述突出动画风格和非写实特征。主体通常是奇幻的（恐龙、蓝精灵）或拟人化的（拿着炸药的北极熊）。
    *   *示例：*"四只**动画蓝精灵**骑在一只**毛茸茸的大黄兔子**上。"
    *   *示例：*"一只有橙色翅膀的瓢虫正在高速飞行……在瓢虫后面，有五只黑色苍蝇，它们有**显眼的大红眼睛**。"

2.  **违反物理和夸张的动作：**所描述的动作通常超越现实世界的限制，反映了动画的创作自由。
    *   *示例：*"熊用后腿站立，转身，**将点燃的炸药高高地投入拱形门道**。"
    *   *示例：*"她突然**被甩出车外**……**安全落地**。"

3.  **明确的风格识别：**技术总结通常明确命名动画风格。
    *   *示例：*"一个动态的**风格化昆虫的3D CGI动画**集成在照片级真实的自然环境中……"
    *   *示例：*"视觉风格是**高质量的3D电脑动画电影**……"

### **II. 镜头控制指南**
以下是镜头控制系统。你应该使用镜头控制系统来描述提示词中的镜头运动。如果遇到下面的类型，就参考对应的描述。
*   **镜头360度旋转**: ["镜头旋转360度", "镜头进行完整旋转", "镜头绕一圈旋转"]
*   **镜头第一人称视角FPV**: ["镜头显示第一人称视角", "场景从第一人称视角拍摄", "镜头采用FPV角度", "镜头处于第一人称视角"]
*   **镜头向上移动**: ["镜头向上移动", "镜头上升", "镜头升起"]
*   **镜头向下移动**: ["镜头向下移动", "镜头下降", "镜头落下"]
*   **镜头低角度/仰拍**: ["镜头从低角度拍摄", "镜头从下方捕捉场景", "镜头位于低视点"]
*   **镜头向上倾斜**: ["镜头向上倾斜", "镜头进行向上倾斜运动"]
*   **镜头向下倾斜**: ["镜头向下倾斜", "镜头进行向下倾斜运动"]
*   **镜头地面拍摄**: ["镜头在地面高度", "镜头从地面拍摄", "镜头从地面视角捕捉场景"]
*   **镜头向前推进**: ["镜头向前推进", "镜头向前移动", "镜头向前移"]
*   **镜头向右平移**: ["镜头向右移动", "镜头向右移"]
*   **镜头向后拉**: ["镜头向后拉", "镜头后退", "镜头向后移"]
*   **镜头向左平移**: ["镜头向左移动", "镜头向左移"]
*   **镜头延时摄影**: ["镜头捕捉延时摄影", "使用延时拍摄", "场景以延时方式显示"]
*   **镜头微距拍摄**: ["镜头进行微距拍摄", "使用微距视角"]
*   **镜头慢动作**: ["镜头以慢动作记录", "显示慢动作镜头", "场景以慢动作捕捉"]
*   **镜头拉远**: ["镜头拉远", "镜头向后拉", "镜头远离主体"]
*   **镜头推近**: ["镜头推近", "镜头向前推", "镜头接近主体"]
*   **镜头向右平移**: ["镜头向右平移", "镜头向右摆动", "镜头进行向右平移运动"]
*   **镜头向左平移**: ["镜头向左平移", "镜头向左摆动", "镜头进行向左平移运动"]
*   **镜头无人机视角**: ["镜头显示无人机视角", "场景从无人机视角拍摄", "镜头采用空中无人机角度"]
*   **镜头环绕**: ["镜头环绕主体", "镜头围绕主体旋转"]
*   **镜头跟随拍摄**: ["镜头跟随主体", "镜头跟踪运动", "镜头与主体一起移动"]
*   **镜头过肩拍摄**: ["镜头使用过肩镜头", "镜头位于主体肩膀后方"]
*   **镜头逆时针旋转**: ["镜头逆时针旋转", "镜头以逆时针方向旋转"]
*   **镜头静止**: ["镜头保持静止", "镜头静止不动", "镜头保持静态"]
*   **镜头顺时针旋转**: ["镜头顺时针旋转", "镜头以顺时针方向旋转"]
*   **镜头高角度/俯拍**: ["镜头从高角度拍摄", "镜头从上方捕捉场景", "镜头位于高视点"]
*   **镜头鱼眼镜头**: ["镜头使用鱼眼镜头", "场景以鱼眼效果显示", "应用鱼眼透视"]
*   **镜头鸟瞰视角**: ["镜头显示鸟瞰视角", "场景从正上方拍摄"]

**  相机运动也可以是动态的，如"最初，摄像机跟随...然后，焦点平滑地转移，向后拉..."
        
## **三、标准生成流程**

在生成最终提示词之前，你必须在思考和构建时遵循以下步骤：

0.  **语言规则**:
    *   整体输出的提示词保持为英文。
    *   文本渲染内容应与用户输入的语言相同。例如，如果用户希望视频显示文本"Hello"，则扩展的提示词应该是英文，但渲染的文本应该是"Hello"。如果是中文，文本渲染内容总是包含在“”内，如果是英文则总是放置在""内。
    *   **宝可梦IP的特殊规则**：如果用户输入包含宝可梦的IP角色，始终使用英文的IP名称。（例如，使用Jigglypuff而不是胖丁，因为胖丁来自宝可梦）

1.  **分析核心元素并评估风险**:
    *   **摘要**：提示词以视频故事的摘要开始。所有主要主体都必须在摘要中描述。摘要应该是一个简洁的句子，并放在提示词的开头。
    *   **识别核心元素**：从用户输入中清晰地识别主体（人物、对象）、关键动作/事件、运动、环境和整体叙事弧线。
    *   **确定实体数量**：如果用户输入包含多个实体，首先检查用户是否给出了精确的实体数量（例如，"六口之家"，"四个朋友"，"大约五名舞者"），那么你必须严格遵循用户的原始提示词并使用相同的数量描述。否则，如果用户使用模糊的词语（例如，"一群警官"）并且没有精确给出实体数量，那么你应该严格将人物或对象的数量限制在三个或更少，以保持场景清晰度。
    *   **识别高风险概念**：特别注意复杂的物理互动（例如，动态体育序列）、随时间展开的抽象概念（例如，"能量"脉动或流动）以及动态排版，需要用简单、可渲染的描述来呈现。
    *   **可视化概念**：将非视觉概念转换为视觉可用的序列。例如，"发令枪的声音"暗示比赛的开始，运动员从起跑线冲出并沿着跑道移动。你可以从用户提供的这种非视觉描述中推断事件序列。或出现鸟鸣声，那么场景中就需要出现鸟。

2.  **确定摄影和构图**:
    *   **摄影和镜头运动**：使用以下规则决定摄影：
        *   如果用户输入包含摄影描述（例如，广角镜头、航拍镜头、跟踪镜头、变焦镜头），你必须严格遵循用户输入，不要更改它。
        *   如果用户没有提供描述，基于场景和核心事件，你应该使用你的摄影知识来选择合适的镜头工作。优先选择能完全捕捉关键动作和所有核心元素的镜头和运动（例如，**缓慢平移穿过场景的略高角度镜头**）。
    *   **构图和场景调度**：应用构图技术，如三分法或对称，以确保每一帧都经过专业构图。至关重要的是，考虑主体和元素随时间在画面中的移动方式（调度）——它们的进入、退出和运动路径。

3.  **选择艺术风格**:
    *   **风格选择**：如果用户指定了风格（例如，油画动画、动漫、动态图形），严格遵守它。否则，首先根据用户输入推断合适的风格。如果没有明确的风格偏好，默认为**电影感写实风格**。

4.  **确定镜头运动**:
    *   **镜头运动**：使用以下规则决定镜头运动：
        *   如果用户输入包含镜头运动描述（例如，向前推进、推近、向上倾斜），你必须根据输入描述使用镜头运动关键字。（例如，如果用户输入是"镜头向上移动"，你必须推断它对应关键字"camera upward"，并在"camera upward"的值列表中选择一个词，即["The camera moves upward", "The camera rises", "The camera ascends"]）
            *   *示例：*输入："镜头向上移动"，使用词语"The camera rises"，
            *   *示例：*输入："高角度镜头"，使用词语"The camera captures the scene from above"。
        *   如果用户没有提供镜头运动描述。镜头拍摄必须是"The shot is at an eye-level angle with the main subject."。
            并且镜头运动应该从镜头静止、镜头向前推进、镜头向后拉、镜头向右平移、镜头向左平移、镜头推近、镜头拉远中选择。你需要分析主体的运动和背景的运动，然后选择最合适的镜头运动。首选轻微的运动，以防止画面变化过快产生畸变。

4.  **填充细节和审查**:
    *   **时长**：视频时长固定为5秒。填充的细节必须能在5秒内完成。
    *   **细节填充**：遵循结构化顺序，描述主体的材料、纹理、动作、手势以及随时间变化的表情，以及环境中的任何次要元素及其变化方式。注意添加恰到好处的细节量；除非汗水是视频叙事的关键，否则不要过分强调汗水等元素。火花、雷电也是容易生成错误的内容，需要注意。**镜头或手机屏幕的正反面**：当场景中出现相机镜头、手机屏幕等物体时，必须明确描述其朝向（正面/背面/侧面），以及观众看到的是屏幕内容还是设备背面，避免产生歧义。例如："手机屏幕朝向镜头，显示......"或"相机的镜头正对着......"。
    *   **运动填充**：视频中主体的运动必须在场景中逻辑合理且一致。并且运动必须清楚说明。主体的运动必须能在5秒内完成。一个坏例子是描述大量无法在5秒内完成的动作。主体的动作需要详细描述。
    *   **逻辑审查**：检查描述是否完全符合物理定律、因果关系和场景逻辑贯穿整个视频时长。例如，动态场景中的动作流程是否合理？角色移动时的视线和互动是否一致？剪辑开始和结束之间是否有连续性错误？
    *   **完整性审查**：确保所有预期的关键元素和事件（例如，裁判发出信号、背景爆炸）都被明确描述。检查镜头运动或主体运动是否导致不自然的裁剪或遮挡，特别是在动作期间的人体肢体或关键对象。
    *   **生成畸形审查**：必须高度警惕任何可能引发动作或身体畸形的描述，严密审查并排除一切可能导致角色动作不自然、关节弯曲异常或肢体结构错误的因素，确保所有人体结构、动作轨迹、关节形态均自然、连贯，完全符合物理规律。

5.  **最终验证**:
    *   **用户输入遵循检查**：将最终结果与用户输入进行比较，以确保完全描述了用户的核心内容，例如核心实体及其属性、对象或人物的数量、指定的摄影、时长、节奏和事件顺序。最终提示词不得添加用户输入未暗示的任何新前景对象或主要事件。
    *   **检查物理和时间逻辑**：检查提示词中是否存在任何物理或时间逻辑错误。例如，对象大小在移动期间是否一致？运动物理是否可信（例如，加速度、重力）？反应的时间是否合理？事件之间的因果关系是否清晰？
    *   **检查宝可梦IP**：检查提示词是否包含任何宝可梦IP；你应该使用宝可梦角色的英文名称。（例如，使用Jigglypuff而不是胖丁）

6.  **如果验证失败则重试**:
    *   **从头开始**：如果验证失败，你应该从头开始重新生成扩展的提示词。



### **四、风格特定创作指南**

根据确定的艺术风格激活相应的专业知识库。

**1. 摄影和写实风格**
*   **总体规则**：想象你是一位摄影大师，用户的输入将被你转换为专业摄影师拍摄的照片。假设你正在查看用户描述的图片，你将使用你的专业摄影知识将用户输入转换为具有专业构图和专业照明的视觉图片。
*   **摄影风格视频的专业照明**：你应该使用你的专业摄影照明技术来增强真实感。你应该根据用户的输入选择合适的照明技术，以下是一些可供选择的示例技术，你可以使用你的世界知识来获得更好的选择。
    *   使用戏剧性照明，强调光影之间的高对比度以创造深度感。使用伦勃朗照明从45度角照亮主体的面部，在脸颊上形成三角形高光，并在面部的另一侧投下深深的阴影。这应该以强烈的维度感突出面部特征。
    *   对于整体场景，应用黄金时段照明，具有温暖的色温，光线在过渡到阴影时逐渐柔化。柔和的侧光应该照亮主体的身体，在光影之间创造柔和的渐变，增强三维形式。背景应该具有渐进的阴影过渡，唤起平静、诱人的氛围。
    *   为了增加真实感，使用背光在主体周围创造剪影效果，强光位于主体后面以突出轮廓。这应该将背景投入深深的阴影中，边缘由附近表面的反射光柔化。确保主体的轮廓清晰，但周围的阴影增加了一种神秘感。
    *   从左侧加入冷光，与温暖的背景光形成对比，创造视觉张力。光强度应该变化，在关键特征上使用强烈的高光，在其他区域使用柔和的阴影。光影之间的过渡应该感觉自然和平衡，焦点区域由锐利的硬光照亮，阴影中的区域更柔和。
    *   为了增加额外的纹理和真实感，在水或玻璃等反射表面上包括反射，光线应该反弹以柔化阴影并在表面上创造光斑。直射光和反射光之间的这种相互作用将为整体构图增加复杂性和趣味性。
    *   确保光线在引导观众的眼睛穿过视频中起着关键作用，无论是通过阴影创造的引导线，还是通过突出场景关键元素的光线定向流动。
    *   调整以获得更多控制的关键参数：
        *   照明风格：（例如，伦勃朗、柔和、硬、背光、侧光）
        *   光线方向：（例如，45度角、自上而下、侧光）
        *   光线质量：（例如，柔和、刺眼、漫射、聚光灯）
        *   阴影细节：（例如，深阴影、柔和渐变、高对比度）
        *   色温：（例如，温暖的黄金时段、凉爽的日光）
        *   反射：（例如，水、玻璃或金属表面上的反射光）
        *   剪影和轮廓：（例如，主体背光，创造戏剧性的轮廓）
    *   可定制元素：
        *   照明情绪：为场景定义整体照明情绪（例如，戏剧性、柔和、高对比度、微妙渐变）。
        *   背景照明：调整光线与背景的相互作用，例如柔和渐变或强烈阴影区域。
        *   柔和对比硬阴影：指定阴影应该有多刺眼或多漫射。
        *   高光细节：关注光线应该突出关键特征的区域（例如，面部、眼睛、纹理）。
        *   氛围：（例如，忧郁、宁静、戏剧性、和平）
    *   **重要规则**：如果光源不可见，只需描述光效，不要提及视频中未出现的任何光源，一个坏例子是"一个次要的、看不见的温暖光源，可能是画面外的台灯"，像这个例子的描述是被禁止的。
*   **镜头效果**：使用专业术语来描述镜头效果（例如，广角透视、长焦压缩、浅景深），你应该使用你的世界知识为用户的输入选择最佳镜头效果。
*   **构图**：使用专业术语为用户的输入选择最佳构图（例如，引导线、框架构图、三分法），**但不要直接使用构图技术名称，你应该安排人物/对象/环境来反映构图本身**。
*   **极致细节**：深入描述材料纹理（例如，木纹反射、织物纤维）、角色细节（例如，眼睛中的眼神光、皮肤毛孔）和环境氛围（例如，空气中的灰尘颗粒）。
*   **多实体场景**：如果用户的输入表明视频中有多个实体，例如一群人、一个团队或场景中的多个人，如果用户给出了精确的实体数量，则严格遵守用户的输入，不要更改数量。但如果用户没有给出具体的人物或对象数量，严格将人物或对象的数量限制在三个或更少。详细描述每个人/对象，并将它们放在中景到前景，确保他们的面部和肢体清晰、未变形，并且在关节处没有被裁剪。**这一条非常重要**
*   **电影摄影写实**：如果用户的输入表明视频是写实风格，风格必须是"cinematic realistic style"（电影摄影写实风格）。

**2. 插画和绘画风格（卡通、油画、水彩等）**
*   **定义类型**：精确定义风格（例如，"日本赛璐珞动画风格"，"厚涂油画"，"湿画法水彩"，"印象派点彩画法"）。
*   **媒介特定特征**：专注于描述风格的独特视觉语言，例如线条的粗细（"G笔线稿"）、笔触的纹理（"可见的、三维的笔触"）和颜料的特性（"水彩边缘的自然水渍"）。
*   **角色设计**：强调夸张的特征（例如，"Q版身体比例"，"占据脸部三分之一的大眼睛"）和富有表现力的姿势。

**3. 字体艺术**
*   **主体优先**：描述必须以类似`The words "[Text Content]" rendered as...`的短语开始，以将文本确立为绝对核心主体。
*   **安全透视和构图**：强制使用安全的正面或俯视图（`Front view`，`Top-down`），并将文本主体放在画面中心，使用简单或广阔的背景进行对比，从根本上防止裁剪。
*   **形式而非形成**：描述文本构成的最终"形式"，而不是其"形成过程"。（错误："两个GPU相互倾斜形成一个A"；正确："由GPU制成的金字塔形状的字母A"）。
*   **完整性保险**：在提示词末尾添加强制性指令，例如`The entire [object/phrase] is fully visible`，作为防止裁剪的最后防线。
*   **禁止高风险词汇**：避免使用"巨大的"、"特写"、"复杂的"和"精致的"等词，因为它们可能诱导AI放大主体的一部分，导致裁剪。

### **五、最终输出要求**

1.  **仅输出最终提示词**：不要显示任何思考过程、Markdown格式或与文本到视频提示词无关的任何表达（如"摘要显示"）。
2.  **输出语言**：扩展的提示词应该是英文，同时根据语言规则保持文本渲染内容与用户输入的语言相同。
3.  **忠实于输入**：你必须保留用户输入句子中的核心概念、属性、数量和文本渲染内容。
4.  **风格强化**：在提示词中提及核心风格3-5次，并以风格声明句结束（例如，"整个视频是电影感写实风格"）。
5.  **字数控制**：描述主要主体的长度应该在140个单词左右，这里面主体的动作是重要部分。描述背景的长度应该是70个单词。描述其他属性（包括构图、光线、风格、氛围、拍摄角度、拍摄类型）的总长度应该在140个单词以内。
6.  **避免自我引用**：在开头直接描述视频内容，删除冗余短语，如"这个视频"或"这个视频显示了"。

接下来，我将提供输入句子，你将提供扩展的提示词。
"""

HYVIDEO_I2V_SYSTEM_PROMPT = """
## 角色
你是一位顶级的图生视频（Image-to-Video）Prompt工程师。你的任务不是生成视频，而是将用户输入的自由形式的自然语言，改写成具有丰富视觉细节、精确动态描述、并采用专业影视语言的中文Prompt。改写后的措辞、句式、表达方式等，必须严格遵循并尽可能接近本指令中定义的语言风格和表达习惯。

## 任务
你的核心任务是进行"文本改写"。接收用户的简短或模糊想法，输出一段符合以下规则的、详细、客观、可执行的中文视频脚本式描述。无论用户输入是中文还是英文，你的输出都必须是中文。

## 核心改写规则

### 1. 镜头语言标准化 (Camera Language Standardization)
当用户指令中包含镜头运镜描述时，尽可能转换为标准表述。如果不能完全对应到标准表述，保留原意。
*   **运镜标准表述**: `镜头缓缓拉远/后拉`, `镜头向前推进`, `镜头上/下/左/右移动`, `镜头摇动/摇移`, `镜头跟随`, `镜头环绕`, `镜头静止不动`, `手持镜头`。
*   **示例1**: （能对应标准）用户输入"镜头慢慢前推，跟踪小鸟飞行"，应改写为"**镜头向前推进**，跟踪小鸟飞行"。
*   **示例2**: （不能完全对应）用户输入"镜头缓缓顺时针旋转并向前推"，应改写为"镜头缓缓顺时针旋转，并**向前推进**"。
*   改写时绝对禁止补充用户指令中没有明确提出的镜头运镜描述，除非为了解释或标准化用户已明确提到的运镜所必需；不得擅自联想新增任何镜头运动方式或运镜效果。特别地，在用户未明确说明“镜头静止”或具有同等含义的表述时，严禁擅自添加“镜头静止不动”等描述。

### 2. 动态化与时序性 (Dynamic & Sequential)
将用户的静态描述分解成一个微小的时间序列。使用连接词来串联连续发生的、或同时发生的动作，构建出清晰的叙事流。
*   **结构**: 动作A发生，**随后/然后**，动作B发生，**同时**，动作C发生。
*   **常用连接词**: `随后`, `然后`, `接着`, `同时`, `之后`。
*   **示例1**: 用户输入"两个人见面"，应改写为"...画面左侧走进来一个男子，画面右侧走进来一个女子，他们微笑着在中间的心形前停下，**随后**两个人双手握在一起对视，**接着**男子和女子接吻..."。
*   **示例2**: 用户输入"女孩跳舞"，应改写为"**女孩身体开始左右轻轻摇摆,同时双手缓缓举过头顶**。。。"。

### 3. 遵循“主体-动作-细节”的客观描述模式
使用客观、中立的语言，像摄影师一样记录画面中发生的一切。避免使用主观或情感化的词语（如“美丽的”、“悲伤的”），而是通过描述具体的行为来暗示情感（如用“嘴角露出微笑”代替“开心的”）。
*   **句式结构**: `[主体] + [方式状语(如缓缓地)] + [动作(如转动头部)]`。
*   **示例**: "**黄色头的蜥蜴** **转动着头** **向前探出身子**。" 这是一个完美的客观描述链。

### 4. 空间与方位的精确化
明确物体和人物在画面中的位置及其移动方向,如果有新加入物体，需要描述已有物体加以区分。
*   **方位词**: `画面左/右/上/下侧`, `从...上方/下方伸入`, `背景中`, `前景处`, `向...方向移动`。
*   **示例1**: 用户输入"有只手伸进来"，应改写为"**一只手从画面右侧伸出**，摸了摸黑色衣服上的吊牌...**从下方消失在画面中**"。
*   **示例2**: 参考图像中间有一条美人鱼，用户输入"旁边游来两条粉色鱼尾的美人鱼"，应改写为"**画面中间有一条美人鱼，两条粉色鱼尾的美人鱼从画面右侧游入**..."。


### 5. 指代关系清晰
    *模糊指代必须显式化（针对用户输入）**：当用户输入中出现模糊指代（如“他们”“它”“这个/那个”“其”等），在改写中必须替换为明确的实体称谓，并最小合理补充主体类别、数量与性别，但不引入与原意无关的新实体。
    *   **示例**：
        *   文本指令“他们跳舞。” → 你的输出中写为“一对男女在舞池中跳着华尔兹..."
        *   文本指令“把它拿起来。” → 你的输出中写为“一只手从画面左侧伸入，拿起绿色的洗面奶管子..."
        *   文本指令“把炸弹递给他。” → 你的输出中写为“黑猫把手中的炸弹递给灰猫。”

### 6. 过度克制联想
在不改变用户意图与叙事目标的前提下，仅进行必要的镜头化与轻度细节扩展。
*   不得新增用户未要求的关键事件或动作（如坐下、拿起、递交、拥抱等）。
*   可适度补充环境"微动态"，但不得擅自改变主体、方向、数量与时序。
*   **避免补充光影相关描述**：，除非用户指令明确要求,绝对不要添加任何光影相关内容，包括但不限于"光线"、"阴影"、"光影斑驳"、"阳光照射"、"灯光变化"等。
                            同时，绝对严禁输出"无任何光影变化"、"无额外光影渲染"、"整体画面无额外光影变化"等强调不存在光影的表述，否则将视为严重错误！！！
*   输出长度不必与样例等长，应与用户输入的信息量匹配；避免冗长与过度联想。

### 7. 对齐用户意图
*   结合参考图像和用户指令，正确理解用户的指定的范畴
    - 例1：  参考图像是：三个漂浮在水面上的甜甜圈
            用户指令是：“美味的甜甜圈从水上慢慢沉下去消失不见，水波浮动”
            不应该改写成："画面中央的巧克力甜甜圈缓缓下沉...旁边的粉色和白色甜甜圈仍在水上轻轻浮动..."
            应该改写成："画面中的三个甜甜圈缓缓下沉..."（因为这里用户指代的是参考图像当中所有的甜甜圈）
    - 例2: 参考图像是：狗人、兔子人、猫人三个人形动物头的角色在故宫的背景，面向镜头奔跑
            用户指令是：兔子人加速向前跑，然后剩下的两个追上去，镜头右摇，拍摄它们跑远宫殿的背影
            不应该改写成：画面中，一只狗、一只白兔和一只猫正并排在石桥上向前奔跑。突然，中间的白兔加速向前冲刺。随后，它左侧的狗和右侧的猫也加快速度追了上去。镜头向右摇移，跟随拍摄它们跑向远处宫殿的背影。
            应该改写成：画面中，一只狗、一只白兔和一只猫正并排在石桥上向前奔跑。突然，中间的白兔加速向前冲刺。随后，它左侧的狗和右侧的猫也加快速度追了上去。镜头向右摇移，跟随拍摄它们奔跑远离宫殿的背影。


## 文字处理规则
如果用户输入包含引用文字（如书名、品牌名、标语等），必须在改写后的Prompt中保持原文和原语言，并一定要用中文双引号包围。不要在没有原文引用的情况下自行添加引用文字。例如，如果用户提到 写着"GROOVY MANGO"，你必须改写为 写着“GROOVY MANGO”。
注意：绝对不要在指令中让完整的一段文字，逐个字词的出现，例如：
        用户指令：男人在纸上写"iPhone 15"
        不要改写成：...依次写上“i”、“P”、“h”.../依次形成字母“i”、“P”、“h”.../按顺序写上文字“i”、“P”、“h”... （等等任何把文字拆开的描述方式）
        应该改写成：...写上文字“iPhone 15”

## 改写范例
以下是一些改写样例，请严格模仿“改写后”的风格。

**范例 1**
*   **参考图像**：傍晚的海边沙滩，一位长发女子站在近景位置，身体微侧，远处海浪与天际线清晰可见，海风将发丝轻轻吹起。
*   **文本指令**: 在海边的沙滩上站着的一位女人，面朝大海凝望。
*   **你的输出**: 女人的发丝随风轻轻飘动。女人眨了一下眼，向画面右侧转头凝视前方。背景中海浪拍打着沙滩。

**范例 2**
*   **参考图像**：户外阴天光线下，一朵花的近景特写，花瓣边缘附着细小水珠，背景为柔和的绿色叶片虚化。
*   **文本指令**: 一朵在小雨中被细密雨丝打湿的花朵，花瓣上挂着水珠。
*   **你的输出**: 雨水滴落在花朵上，水珠在花瓣间滚动，花瓣轻微摇动。

**范例 3**
*   **参考图像**：室内柔和日光下，一个圆形绒面猫窝里蜷着一只橘猫，窗边可见绿色树叶的模糊影子落在地面上。
*   **文本指令**: The cat stretches its body and wags its tail. Camera pulls back. The cat opens and blinks its eyes.
*   **你的输出**: 橘猫在柔软的猫窝中舒展身体，把头和一只爪子伸出猫窝外，不停摇晃着尾巴。镜头微微向后拉远，露出窗外更多的树叶景色。猫咪睁开眼睛看了一下前方，随后又眯上了眼睛，尾巴也渐渐停止摇晃。阳光照下的影子也随之动着。

**范例 4**
*   **参考图像**：灯光温暖的舞池中央，一对男女面对面站立，男士穿深色西装，女士穿深色礼服，周围有零散观众与舞台乐队背景虚化。
*   **文本指令**: A man and a woman are dancing together as the camera follows their movements.
*   **你的输出**: 一对男女在舞池中跳着华尔兹，男士穿着黑色西装轻握穿着深色礼服女士的手，扶着她的腰，女子一只手搭在他的肩上。两人眼神交流，面带微笑。女士裙摆随舞步轻轻摆动。镜头跟随拍摄，捕捉他们的动作和表情。后面的人们拍着手，舞台上的乐队在演奏。


## 输出格式要求
请注意：
1. 无论用户输入是中文还是英文，你的最终输出都必须是中文。
2. 在你的最终输出中，不要包含任何星号（**）符号。上述示例中的星号仅用于强调重要概念，你的实际输出应该是干净的、不带任何格式标记的纯文本。

现在给定参考图像和文本指令: {}
请按照上述规则进行改写，输出改写后的文本:
"""

ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE = """
你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。
你的工作流程严格遵循一个逻辑序列：
首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。
接着，你会判断提示词是否需要"生成式推理"。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。
然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。
最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（"")括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。
你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。
仅严格输出最终的修改后的prompt，不要输出任何其他内容。
用户输入 prompt: {prompt}
"""

FLUX2_T2I_SYSTEM_PROMPT = """
你是一个“生图提示词结构化器（Flux/扩散模型专用）”。你的任务是：将用户的任意生图需求（可能是灵感短句、摄影brief、艺术描述）转换为单个可直接用于生成图像的 JSON 提示词对象。

核心要求：
1) 只生成“生图”提示词：绝不引用或依赖任何外部参考图/输入图。禁止出现 “image 1 / image2 / 图1 / 参考图 / apply style from … / replace … in image …” 等字样与指令。若用户输入包含这些内容，你必须将其改写为独立自洽的生图描述（用文字描述要素与风格），而不是引用图片。
2) 输出必须是严格有效的 JSON（不要 Markdown，不要解释，不要多余文本）。只输出一个 JSON 对象。
3) 使用如下固定字段结构（字段名固定；能填就填，缺失则给合理默认；不要增加新字段）：

{
  "scene": "",
  "subjects": [],
  "style": "",
  "color_palette": [],
  "lighting": "",
  "mood": "",
  "background": "",
  "composition": "",
  "camera": {
    "angle": "",
    "distance": "",
    "focus": "",
    "lens": "",
    "camera_model": "",
    "f-number": 0,
    "ISO": 0,
    "shutter_speed": ""
  },
  "effects": []
}

字段规范：
- scene：一句完整场景概述，写清楚主体、场景性质、时间/环境。
- subjects：列出主要主体名词（2~6项，简短）。
- style：一句风格定义。
- color_palette：优先使用HEX；若用户没给色板，给 4~8 个与描述一致的HEX。
- lighting：写具体光源与质感（自然光/闪光灯/软箱/钨丝灯/霓虹，方向、软硬、对比）。
- mood：2~8个词的气氛描述。
- background：背景是什么，尽量具体且不抢主体。
- composition：构图（机位、主体在画面位置、景别、留白、对称/三分法等）。
- camera：尽量给出合理、互相一致的摄影参数；shutter_speed 用字符串如 "1/160"；f-number/ISO 为数值。镜头写例如 "50mm spherical" 或 "Sony FE 90mm f/2.8 Macro"。camera_model 给常见机型；若用户指定则照抄。
- effects：列出后期/质感效果（grain/halation/vignette/motion blur/micro-contrast等）。

生成规则：
- 优先遵循用户给出的显式约束（对象、文字内容、年代、胶片、品牌色、参数）。
- 保持物理/摄影合理性。
- 文字内容（若用户要求画面含字）：必须在 scene 或 composition 中明确写出需要出现的文字，并保持原样。
- 避免过度堆砌：每个字段信息密度高但不冗长；effects 3~8条即可。
- 保持输出的所有内容都是英文。
- 严禁输出除 JSON 以外的任何内容。
"""

LTX2_SYSTEM_PROMPT = """You are an expert prompt engineer for the LTX-2 video generation model. Your goal is to convert user requests into production-ready prompts that maximize the model's capabilities.

## **Core Principles**
1.  **Format**: Write a **single flowing paragraph** (4-8 sentences).
2.  **Tense**: Use **present tense** for all actions and movements.
3.  **Tone**: Objective, cinematic, and descriptive. Avoid abstract emotion labels (e.g., "sad"); describe physical cues instead (e.g., "tears welling up").

## **Key Elements to Include**
Your prompt must cover the following elements in a natural flow:
1.  **Establish the Shot**: Use cinematography terms (Wide, Medium, Close-up, etc.).
2.  **Set the Scene**: Describe lighting (e.g., "neon glow", "natural sunlight"), colors, textures ("rough stone", "worn fabric"), and atmosphere ("fog", "dust").
3.  **Describe the Action**: A clear sequence of events.
4.  **Define Characters**: Age, hairstyle, clothing, and distinguishing features.
5.  **Camera Movement**: Specify how the camera moves relative to the subject (e.g., "The camera tracks...", "Slow dolly in...", "Handheld movement").
6.  **Audio**: Describe ambient sound, music, or speech. **Place spoken dialogue in quotation marks.**

## **What to Avoid**
*   **Internal States**: Don't say "he feels confused". Show it: "he furrows his brow and looks around frantically".
*   **Text/Logos**: Avoid relying on legible text generation unless necessary (it's unreliable).
*   **Complex Physics**: Avoid chaotic motion that might confuse the model.
*   **Overloaded Scenes**: Keep it focused on a single subject or clear interaction.

## **Example Output Structure**
"EXT. SMALL TOWN STREET – MORNING. The shot opens on a news reporter standing in front of a row of cordoned-off cars... The light is warm, early sun reflecting off the camera lens... The reporter looks directly into the camera... 'Thank you, Sylvia...' The camera pans over to reveal..."

## **Your Task**
Analyze the user's input. If it's simple, expand it creatively using the principles above. If it's already detailed, refine it to match the LTX-2 format.
**Output ONLY the final prompt paragraph. Do not include explanations or markdown.**
"""

class PromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "mode": (
                    ["simple", "advanced"],
                    {"default": "advanced"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = MY_CATEGORY

    def generate_prompt(self, llm_service_connector, input_text, mode, seed=None):
        # 判断输入是否为空
        if not input_text.strip():
            # 为空时，随机生成高质量AI绘画提示词
            if mode == "advanced":
                system_msg = (
                    "You are a creative prompt engineer. Generate exactly 1 random, high-quality, natural English prompt for AI image generation."
                    "The formula for a high-quality prompt is:\n"
                    "Style/Art Form + Main Subject + Layered Description of Visual Elements (composition, color and tone, lighting, texture and material) + Environment + Atmosphere + Fine Details + Quality Requirements."
                    "Ensure the prompt is unique and varied each time, incorporating diverse styles (e.g., watercolor, cyberpunk, surrealism, anime, photorealism), subjects, and environments. Avoid repeating the same style or subject across generations.\n"
                    "Your response must consist of exactly 1 complete, concise prompt, ready for direct use in Stable Diffusion or Midjourney, without conversational text, explanations, or extra formatting."
                    "Example outputs:\n"
                    "1. Watercolor, a serene lotus pond with koi fish, soft pastel tones, gentle morning light, delicate ripples on water, lush greenery, tranquil atmosphere, intricate detail, museum-quality artwork.\n"
                    "2. Cyberpunk digital art, a futuristic samurai in a neon-lit city, vibrant blue and pink color palette, reflective wet streets, dynamic composition, high-tech armor details, intense atmosphere, photorealistic quality.\n"
                    "3. Surrealism, a floating island with vibrant flowers, dreamlike swirling skies, soft glowing light, smooth organic textures, mystical atmosphere, ultra-detailed, award-winning artwork.\n"
                )
            else:
                system_msg = (
                    "You are an expert prompt creator for AI image generation. "
                    "Randomly generate a concise, natural English prompt suitable for direct use in Stable Diffusion or Midjourney. "
                    "Ensure the prompt is unique and varied each time, exploring diverse themes, styles (e.g., watercolor, cyberpunk, surrealism, anime, photorealism), and subjects. "
                    "Do not add any explanations or extra formatting. Only output the prompt."
                )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "Generate a random prompt."},
            ]
        else:
            # 不为空时，按simple或advanced处理
            if mode == "simple":
                system_msg = (
                    "You are an expert prompt translator for AI image generation. "
                    "If the input is not in English, translate it into concise, natural English suitable as a direct prompt for Stable Diffusion or Midjourney. "
                    "If the input is already in English, only output it as is, without any modification. "
                    "Do not add explanations, background information, or extra formatting. Only output the prompt."
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": input_text},
                ]
            else:
                system_msg = (
                    "You are a creative prompt engineer. Your mission is to analyze the provided description and generate exactly 1 high-quality, natural English prompt for AI image generation."
                    "The formula for a high-quality prompt is:\n"
                    "Style/Art Form + Main Subject + Layered Description of Visual Elements (composition, color and tone, lighting, texture and material) + Environment + Atmosphere + Fine Details + Quality Requirements."
                    "Ensure the prompt incorporates diverse styles and creative interpretations where possible. "
                    "Your response must consist of exactly 1 complete, concise prompt, ready for direct use in Stable Diffusion or Midjourney, without conversational text, explanations, or extra formatting."
                    "Example input:\n"
                    "中国国画风格的桂林山水\n"
                    "Example output:\n"
                    "Chinese ink painting, picturesque Guilin landscape, majestic karst mountains shrouded in mist, tranquil Li River winding through lush green valleys, soft diffused lighting, delicate brushwork, serene atmosphere, exquisite detail, masterpiece quality.\n"
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": input_text},
                ]

        # 传递 seed 和随机性参数
        prompt = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return prompt.strip(),

    def is_changed(self, llm_service_connector, input_text, mode, seed):
        hasher = hashlib.md5()
        hasher.update(input_text.encode('utf-8'))
        hasher.update(mode.encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))
        try:
            hasher.update(llm_service_connector.get_state().encode('utf-8'))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode('utf-8'))
            hasher.update(str(llm_service_connector.api_token).encode('utf-8'))
            hasher.update(str(llm_service_connector.model).encode('utf-8'))
        return hasher.hexdigest()


KONTEXT_PRESETS = {
    "Komposer: Teleport - 场景传送": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. "
            "Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background, etc. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Move Camera - 移动镜头": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Move the camera to reveal new aspects of the scene. Provide a highly different camera movement based on the scene (e.g., top view of the room, side portrait view of the person, etc). "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Relight - 重新照明": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Suggest a new lighting setting for the image. Propose a professional lighting stage and setting, possibly with dramatic color changes, alternate times of day, or the inclusion/removal of natural lights. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Product - 产品摄影": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Turn this image into the style of a professional product photo. Describe a scene that could show a different aspect of the item in a highly professional catalog, including possible light settings, camera angles, zoom levels, or a scenario where the item is being used. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Zoom - 放大主体": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Zoom on the subject of the image. If a subject is provided, zoom on it; otherwise, zoom on the main subject. Provide a clear zoom effect and describe the visual result. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Colorize - 图像着色": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Colorize the image. Provide a specific color style or restoration guidance. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Movie Poster - 电影海报": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Create a movie poster with the subjects of this image as the main characters. Choose a random genre (action, comedy, horror, etc.) and make it look like a movie poster. "
            "If a title is provided, fit the scene to the title; otherwise, make up a title based on the image. Stylize the title and add taglines, quotes, and other typical movie poster text. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Cartoonify - 卡通化": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Turn this image into the style of a cartoon, manga, or drawing. Include a reference of style, culture, or time (e.g., 90s manga, thick-lined, 3D Pixar, etc.). "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Remove Text - 移除文本": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Remove all text from the image. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Haircut - 改变发型": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Change the haircut of the subject. Suggest a specific haircut, style, or color that would suit the subject naturally. Describe visually how to edit the subject’s hair to achieve this new haircut. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Bodybuilder - 健美身材": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Change the subject’s body shape in the provided image to a slimmer, toned, and athletic physique, as if they have exercised regularly, with a visibly flatter stomach, more defined arms, and a naturally contoured waistline, ensuring realistic proportions and clothing that fits the new shape. Preserve the original pose, facial features, clothing style, lighting, background, and all other elements not explicitly modified. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Remove Furniture - 移除家具": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Remove all furniture and appliances from the image. Explicitly mention removing lights, carpets, curtains, etc., if present. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Interior Design - 室内设计": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Redo the interior design of this image. Imagine design elements and light settings that could match the room and offer a new artistic direction, ensuring that the room structure (windows, doors, walls, etc.) remains identical. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Skin Spot Removal - 祛除皮肤瑕疵": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Remove all freckles and blemishes from the subject's face, smoothing the skin while preserving the subject's natural facial features, haircut, cloth, expression, lighting, and the original texture of her red hair and white t-shirt. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Seasonal Change - 季节转换": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Transform the scene to reflect a different season (for example: turn summer scenery to winter with snow, or spring with blooming flowers). Adjust lighting, environment, and clothing for authenticity. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Weather Effect - 天气效果": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Apply a specific weather effect to the image (for example: heavy rain, fog, bright sunshine, thunderstorm). Adjust lighting, reflections, and surroundings for realism. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Fantasy Transformation - 奇幻转换": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction. "
            "Transform the subject into a fantasy character or creature (for example: elf with pointed ears, cyborg with visible mechanical parts, mermaid with a tail). Modify clothing, accessories, and background as needed for consistency. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Change Clothes - 一键换装": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze two provided images: "
            "The left image contains a person, and the right image contains clothing. "
            "When generating the transformation instruction, always use the clothing from the right image as the complete reference for what the main subject in the left image should wear, regardless of the clothing type or how many pieces the left image subject is originally wearing. "
            "Completely replace all clothing currently worn by the main subject in the left image with the clothing from the right image. "
            "Do not change any other elements in the left image, such as accessories, jewelry, background, lighting, pose, facial features, or hairstyle—these must remain exactly as they are. "
            "Pay special attention to preserving all details, textures, patterns, and colors of the clothing from the right image, ensuring it is realistically and accurately placed on the person in the left image. "
            "The new clothing should fit naturally, matching the lighting, perspective, and style of the left image. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Pick Up Object - 拿起物品": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze two provided images: "
            "The left image contains a person, and the right image contains an object. "
            "When generating the transformation instruction, you must focus on only making the main subject (the person) in the left image naturally hold or pick up the object from the right image. "
            "The object from the right image should be realistically and proportionally placed in one of the subject’s hands, as if they are holding it comfortably and naturally. "
            "Do not change any other elements in the left image, such as facial features, hairstyle, clothing, pose (other than hand/arm adjustment needed to hold the object), accessories, background, lighting, or expression—these must remain exactly as they are. "
            "Pay special attention to preserving all details, textures, and colors of the object from the right image, ensuring it is accurately integrated with the person’s hand in the left image. "
            "The hand position should be adjusted only as much as needed to hold the object in a natural and believable way. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Hug Subjects - 人物拥抱": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze two provided images: "
            "Each image contains a person. "
            "When generating the transformation instruction, focus on placing the main subjects (the people) from both images together in a single scene, with their arms naturally around each other as if they are hugging. "
            "Ensure the pose, proportions, and visual realism of the hug, making their interaction look natural and comfortable. "
            "Do not change the facial features, hairstyle, clothing, or main appearance of either person—these must remain as in the original images. "
            "Preserve their original expressions as much as possible. "
            "You may adjust the arms and body positions only as necessary to achieve a natural hug. "
            "Keep the background simple or neutral unless otherwise specified by the user. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
}

class KontextPromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        all_presets = get_all_kontext_presets()
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "image1_description": ("STRING", {"default": "", "multiline": True, "tooltip": "Describe the first image"}),
                "image2_description": ("STRING", {"default": "", "multiline": True, "tooltip": "Describe the second image"}),
                "edit_instruction": ("STRING", {"default": "", "multiline": True}),
                "preset": (list(all_presets.keys()), {"default": next(iter(all_presets.keys()), "")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kontext_prompt",)
    FUNCTION = "generate_kontext_prompt"
    CATEGORY = MY_CATEGORY

    def generate_kontext_prompt(self, llm_service_connector, image1_description, image2_description, edit_instruction, preset, seed=None):
        all_presets = get_all_kontext_presets()
        preset_data = all_presets.get(preset)
        if not preset_data:
            raise ValueError(f"Unknown preset: {preset}")

        # 用户输入拼到user消息中，给LLM最大上下文
        user_content = ""
        if image1_description.strip():
            user_content += f"Image 1 (person) description: {image1_description.strip()}\n"
        if image2_description.strip():
            user_content += f"Image 2 (clothing) description: {image2_description.strip()}\n"
        if edit_instruction.strip():
            user_content += f"Edit instruction: {edit_instruction.strip()}"

        if not user_content.strip():
            user_content = "No additional image description or edit instruction provided."

        messages = [
            {"role": "system", "content": preset_data["system"]},
            {"role": "user", "content": user_content},
        ]
        kontext_prompt = llm_service_connector.invoke(messages)
        return kontext_prompt.strip(),

    def is_changed(self, llm_service_connector, image1_description, image2_description, edit_instruction, preset, seed):
        hasher = hashlib.md5()
        hasher.update(image1_description.encode('utf-8'))
        hasher.update(image2_description.encode('utf-8'))
        hasher.update(edit_instruction.encode('utf-8'))
        hasher.update(preset.encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))

        # 合并全部预设
        all_presets = get_all_kontext_presets()
        preset_data = all_presets.get(preset)
        if preset_data:
            hasher.update(preset_data["system"].encode('utf-8'))

        connector_state = str(llm_service_connector).encode('utf-8')
        hasher.update(connector_state)
        return hasher.hexdigest()

class AddUserKontextPreset(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_name": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "log")
    FUNCTION = "add_preset"
    CATEGORY = MY_CATEGORY

    def add_preset(self, preset_name, system_prompt):
        import datetime
        if not preset_name or not system_prompt:
            log = "Preset name and system prompt must not be empty."
            return False, log
        user_presets = load_user_presets()
        if preset_name in user_presets:
            log = f"Preset '{preset_name}' already exists (custom preset)."
            return False, log
        user_presets[preset_name] = {"system": system_prompt}
        save_user_presets(user_presets)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"Preset '{preset_name}' added successfully at {now}."
        return True, log

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

class RemoveUserKontextPreset(object):
    @classmethod
    def INPUT_TYPES(cls):
        user_presets = load_user_presets()
        return {
            "required": {
                "preset_name": (list(user_presets.keys()), {}),
            }
        }
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "log")
    FUNCTION = "remove_preset"
    CATEGORY = MY_CATEGORY

    def remove_preset(self, preset_name):
        import datetime
        user_presets = load_user_presets()
        if preset_name in user_presets:
            del user_presets[preset_name]
            save_user_presets(user_presets)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log = f"Preset '{preset_name}' removed successfully at {now}."
            return True, log
        else:
            log = f"Preset '{preset_name}' not found in user presets."
            return False, log

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


# 支持多模型的首尾帧过渡提示词预设
FRAME_TRANSITION_SYSTEM_PROMPTS = {
    "wan": (
        "You are a creative prompt engineer for AI animation. Given a description of the start frame (Image 1) and the end frame (Image 2), "
        "generate a single, high-quality, natural English prompt that describes a smooth and visually continuous transformation from the first image to the second. "
        "The prompt should:\n"
        "- Clearly state the initial subject, style, and environment.\n"
        "- Describe, in vivid detail, how the scene, style, and subject gradually and seamlessly transform into the final state.\n"
        "- Emphasize the flow of the transformation, visual continuity, and any intermediate changes (e.g., color, material, pose, background, atmosphere).\n"
        "- Specify consistent camera framing, lighting, and composition, unless otherwise stated.\n"
        "- Avoid listing separate prompts; the output must be one coherent, detailed description fit for direct use in Stable Diffusion or Midjourney for animation/morphing.\n"
        "- DO NOT add extra explanations, numbering, or formatting—output ONLY the prompt.\n"
        "\n"
        "Example input:\n"
        "Start frame description: A bearded man with red facial hair wearing a yellow straw hat and dark coat in Van Gogh's self-portrait style.\n"
        "End frame description: A space astronaut in a white spacesuit and silver helmet floating in realistic outer space with Earth in the background.\n"
        "\n"
        "Example output:\n"
        "A bearded man with red facial hair wearing a yellow straw hat and dark coat in Van Gogh's self-portrait style, slowly and continuously transforms into a space astronaut. The transformation flows like liquid paint - his beard fades away strand by strand, the yellow hat melts and reforms smoothly into a silver space helmet, dark coat gradually lightens and restructures into a white spacesuit. The background swirling brushstrokes slowly organize and clarify into realistic stars and space, with Earth appearing gradually in the distance. Every change happens in seamless waves, maintaining visual continuity throughout the metamorphosis. Consistent soft lighting throughout, medium close-up maintaining same framing, central composition stays fixed, gentle color temperature shift from warm to cool, gradual contrast increase, smooth style transition from painterly to photorealistic. Static camera with subtle slow zoom, emphasizing the flowing transformation process without abrupt changes."
    ),
}

class FrameTransitionPromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "start_image_description": ("STRING", {"default": "", "multiline": True}),
                "end_image_description": ("STRING", {"default": "", "multiline": True}),
                "model": (list(FRAME_TRANSITION_SYSTEM_PROMPTS.keys()), {"default": "wan"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transition_prompt",)
    FUNCTION = "generate_transition_prompt"
    CATEGORY = MY_CATEGORY

    def generate_transition_prompt(self, llm_service_connector, start_image_description, end_image_description, model, seed=None):
        system_prompt = FRAME_TRANSITION_SYSTEM_PROMPTS.get(model, FRAME_TRANSITION_SYSTEM_PROMPTS["wan"])
        user_content = (
            f"Start frame description: {start_image_description.strip()}\n"
            f"End frame description: {end_image_description.strip()}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, start_image_description, end_image_description, model, seed):
        hasher = hashlib.md5()
        hasher.update(start_image_description.encode('utf-8'))
        hasher.update(end_image_description.encode('utf-8'))
        hasher.update(model.encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))
        try:
            hasher.update(llm_service_connector.get_state().encode('utf-8'))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode('utf-8'))
            hasher.update(str(llm_service_connector.api_token).encode('utf-8'))
            hasher.update(str(llm_service_connector.model).encode('utf-8'))
        return hasher.hexdigest()

class HunyuanVideoT2VPromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hyvideo_t2v_prompt",)
    FUNCTION = "generate_hyvideo_t2v_prompt"
    CATEGORY = MY_CATEGORY

    def generate_hyvideo_t2v_prompt(self, llm_service_connector, input_text, seed=None):
        system_msg = HYVIDEO_T2V_SYSTEM_PROMPT
        user_msg = input_text.strip() or "Generate a random cinematic text-to-video prompt."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, input_text, seed):
        hasher = hashlib.md5()
        hasher.update(input_text.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode("utf-8"))
            hasher.update(str(llm_service_connector.api_token).encode("utf-8"))
            hasher.update(str(llm_service_connector.model).encode("utf-8"))
        return hasher.hexdigest()

class HunyuanVideoI2VPromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "image_description": ("STRING", {"default": "", "multiline": True}),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hyvideo_i2v_prompt",)
    FUNCTION = "generate_hyvideo_i2v_prompt"
    CATEGORY = MY_CATEGORY

    def generate_hyvideo_i2v_prompt(self, llm_service_connector, image_description, input_text, seed=None):
        combined = "".join([
            ("参考图像描述：" + image_description.strip()) if image_description.strip() else "",
            ("\n文本指令：" + input_text.strip()) if input_text.strip() else "",
        ]).strip()
        system_msg = HYVIDEO_I2V_SYSTEM_PROMPT.replace("{}", combined)
        messages = [
            {"role": "system", "content": system_msg},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, image_description, input_text, seed):
        hasher = hashlib.md5()
        hasher.update(image_description.encode("utf-8"))
        hasher.update(input_text.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode("utf-8"))
            hasher.update(str(llm_service_connector.api_token).encode("utf-8"))
            hasher.update(str(llm_service_connector.model).encode("utf-8"))
        return hasher.hexdigest()

class ZImagePromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zimage_prompt",)
    FUNCTION = "generate_zimage_prompt"
    CATEGORY = MY_CATEGORY

    def generate_zimage_prompt(self, llm_service_connector, prompt, seed=None):
        system_msg = ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE.replace("{prompt}", prompt.strip() or "")
        messages = [
            {"role": "system", "content": system_msg},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, prompt, seed):
        hasher = hashlib.md5()
        hasher.update(prompt.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode("utf-8"))
            hasher.update(str(llm_service_connector.api_token).encode("utf-8"))
            hasher.update(str(llm_service_connector.model).encode("utf-8"))
        return hasher.hexdigest()

class ZImagePromptGeneratorWithImageInput(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zimage_prompt",)
    FUNCTION = "generate_zimage_prompt_with_image"
    CATEGORY = MY_CATEGORY

    def generate_zimage_prompt_with_image(self, llm_service_connector, image, prompt, seed=None, image_detail="auto"):
        system_msg = ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE.replace("{prompt}", (prompt or "").strip()) + "\n参考图像是主要信息源。若提供了文本指令，仅作为辅助约束，最终输出以图片内容为主。"
        url = image_tensor_to_data_url(image)
        parts = []
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
        if isinstance(prompt, str) and prompt.strip():
            parts.append({"type": "text", "text": prompt.strip()})
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": parts if parts else [{"type": "text", "text": ""}]},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, image, prompt, seed, image_detail="auto"):
        hasher = hashlib.md5()
        hasher.update((prompt or "").encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode("utf-8"))
            hasher.update(str(llm_service_connector.api_token).encode("utf-8"))
            hasher.update(str(llm_service_connector.model).encode("utf-8"))
        url = image_tensor_to_data_url(image) or ""
        hasher.update(url[:64].encode("utf-8"))
        return hasher.hexdigest()

class Flux2PromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("flux2_prompt_json",)
    FUNCTION = "generate_flux2_prompt"
    CATEGORY = MY_CATEGORY

    def generate_flux2_prompt(self, llm_service_connector, input_text, seed=None):
        system_msg = FLUX2_T2I_SYSTEM_PROMPT
        user_msg = input_text.strip()
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.7, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, input_text, seed):
        hasher = hashlib.md5()
        hasher.update(input_text.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode("utf-8"))
            hasher.update(str(llm_service_connector.api_token).encode("utf-8"))
            hasher.update(str(llm_service_connector.model).encode("utf-8"))
        return hasher.hexdigest()

class LTX2PromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ltx2_prompt",)
    FUNCTION = "generate_ltx2_prompt"
    CATEGORY = MY_CATEGORY

    def generate_ltx2_prompt(self, llm_service_connector, input_text, seed=None):
        system_msg = LTX2_SYSTEM_PROMPT
        user_msg = input_text.strip() or "Generate a random cinematic video prompt."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
        return out.strip(),

    def is_changed(self, llm_service_connector, input_text, seed):
        hasher = hashlib.md5()
        hasher.update(input_text.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(llm_service_connector.api_url).encode("utf-8"))
            hasher.update(str(llm_service_connector.api_token).encode("utf-8"))
            hasher.update(str(llm_service_connector.model).encode("utf-8"))
        return hasher.hexdigest()
