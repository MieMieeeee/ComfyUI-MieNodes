"""ComfyUI compatibility shell; generation implementation lives in mie_craft."""
from __future__ import annotations
from mie_craft import minimax_h3_storyboard_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class MiniMaxH3StoryboardGenerator:
    """User role events -> validated portable asset -> V3 (no prompt rewriting)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "llm_service_connector": ("LLMServiceConnector",),
            "name": ("STRING", {"default": "新剧本", "tooltip": "剧本资产名，可直接保存入库。\nScript-asset name; can be saved into the library as-is."}),
            "concept": ("STRING", {"multiline": True, "default": "角色2动作：双手拿着苹果递向角色1。\n角色2：吃苹果吗？\n角色1：不吃。",
                "tooltip": "按输入模式填 · depends on mode:\n"
                           "concept 概念创作：随便写——一个点子、一段故事、几句台词草稿都行，AI 出完整剧本，你看了预览再说好不好；\n"
                           "script 锁定剧本：一行一件事（角色1：台词 / 角色1动作：动作 / 角色1入场：/ 角色1离场：，可加[zh]/[en]），台词逐字锁定。\n"
                           "In concept mode write anything (an idea, a story, loose lines) and the AI authors the full asset; "
                           "in script mode every line is a role-labelled event, locked verbatim."}),
            "character_count": ("INT", {"default": 2, "min": 1, "max": 8}),
            "character_settings": ("STRING", {"multiline": True, "default": "角色1：角色1 | 画面内 | 左侧 | adult male\n角色2：角色2 | 画面内 | 右侧 | adult female",
                "tooltip": "每角色一行：角色1：名字 | 画面内/画外/稍后入场 | 左侧/右侧/中间/无 | 声音要求（分隔符 | 或逗号都行）。锁定模式须与角色数一致；概念模式可整栏留空由 AI 定。外观以接入角色资产为准，未接则用默认设计或模型生成。\n"
                           "One line per role ( | or commas as separators). Locked mode: must cover every numbered role; concept mode: leave empty and the AI casts it."}),
            "shot_count": ("INT", {"default": 0, "min": 0, "max": 20,
                                  "tooltip": "生成片段数，0=自动（能装进约14秒就用最少分段；推荐）。锁定模式下事件会被逐字分组；单人独占台词的续接片段是已知崩角色形态。\n"
                                             "Generation segments, 0 = auto (fewest segments that fit; recommended)."}),
            "target_duration_seconds": ("FLOAT", {"default": 20, "min": 1, "max": 600, "step": 1,
                "tooltip": "目标交付总时长；概念模式只是建议（AI 按内容调整），锁定模式按 H3 帧网格分配。过短会报错，不删减台词。"
                           "怒斥/喊叫演绎实测约 2.5 字/秒，愤怒独白要把目标时长按慢速翻着加。\n"
                           "Target delivered duration; a hint in concept mode, enforced in locked mode. Too short errors loudly; "
                           "dialogue is never cut. Furious/shouting delivery measured ~2.5 chars/s - budget angry monologues accordingly."}),
            "camera_framing": (["medium_shot", "two_shot", "medium_close_up", "close_up", "wide_shot"],
                {"tooltip": "景别：two_shot 双人同框 / medium_shot 中景 / medium_close_up 中近景 / close_up 特写 / wide_shot 全景。概念模式是建议。\n"
                            "Framing (a hint in concept mode): two_shot / medium_shot / medium_close_up / close_up / wide_shot."}),
            "camera_movement": (["locked", "gentle_follow"],
                {"tooltip": "机位运动：locked 固定机位（推荐）/ gentle_follow 轻微跟随。\nCamera movement: locked (recommended) or gentle_follow."}),
            "camera_perspective": (["observer", "first_person"],
                {"tooltip": "视角：observer 旁观（独白对镜头也用它）/ first_person 第一人称（观众自己的手和身体会入画）。\n"
                            "Perspective: observer, or first_person (the viewer's own hands/body enter the frame)."}),
            "language": (["zh", "en"], {"default": "zh"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            # 输入模式（2026-10-29 裁定）：人只给概念/台词，我们出完整分镜，用户验收；
            # 资产是保存物，要能基于上一版持续微调。放 required 尾部保 widget 位序。
            "mode": (["script", "concept", "revise"], {"default": "script", "tooltip":
                "输入模式 · input mode:\n"
                "concept 概念创作：concept 里随便写（点子/故事/台词草稿），AI 出完整剧本，预览验收；\n"
                "revise 修订：把「加载剧本资产」的 storyboard 接到下方输入，concept 里写这一版要改什么"
                "（如「结尾改成她笑而不语，其余不动」），AI 在上一版基础上出新版，预览附逐条 diff；"
                "未提及的台词逐字保留。接线事实（audio_mode/续接重叠帧）继承上一版；\n"
                "script 锁定剧本（默认，旧行为）：逐行剧本，台词逐字锁定，AI 只分组和写英文执行说明。\n"
                "concept: write anything, the AI authors the whole asset. revise: wire the previous "
                "version's storyboard, describe THIS revision, get the next version with a diff. "
                "script (default): role-labelled lines locked verbatim."}),
        }, "optional": {
            # 修订模式的基线：上一版剧本资产。只在 mode=revise 时被读取。
            "storyboard": ("MIE_STORYBOARD", {"tooltip":
                "接「加载剧本资产」的 storyboard 输出 · wire the Load Script Asset output here.\n"
                "仅 revise 修订模式读取：作为基线版本，concept 里写的修改要求作用在它之上，"
                "产出带新名字的下一版（旧资产不动）。concept/script 模式忽略此输入。"}),
            "look": ("STRING", {"default": "Natural lighting, coherent visual style.", "multiline": True, "tooltip":
                "画风·look 只写艺术风格（媒介/画法/色彩），绝不写场景内容或时段（「都市夜景」「室内咖啡馆」都不行）——"
                "look 逐字进提示词且不被任何接线资产覆盖，写了环境就会压过接入的场景图（2026-11-01 真机实测）。\n"
                "Art style ONLY (medium/technique/palette); never scene content or time of day — the look line "
                "reaches the prompt verbatim and nothing overrides it, so an environment named here beats the "
                "wired scene image."}),
            "setting": ("STRING", {"default": "", "multiline": True, "tooltip": "人物位置与情境；具体环境可由V3的可选场景资产提供。"}),
            "constraints": ("STRING", {"default": "", "multiline": True, "tooltip": "导演约束，原样保留。例如：只拍两人对话，不把台词中的回忆拍出来。"}),
            "continuation_overlap_frames": ("INT", {"default": 22, "min": 0, "max": 120, "tooltip": "需与Loop续接重叠帧一致；独立片段不裁剪时设0。"}),
            "temperature": ("FLOAT", {"default": 0.2, "min": 0, "max": 2, "step": 0.05}),
            "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768}),
            "timeout": ([30, 60, 120, 300], {"default": 120}),
            "audio_mode": (["generated", "reference"], {"default": "generated", "tooltip": "generated：H3自选声音；reference：V3中每个角色都接带参考音频的人物资产，画外角色仅使用声音。最多3个角色；Asset→Slots音频需接到H3参考音频输入。"}),
            # 2026-10-29 裁定：场景/光锚是独立输入，不再塞进 setting。
            # 新 widget 一律放 optional 尾部（widget 顺序按位置序列化）。
            "scene": ("STRING", {"default": "", "multiline": True, "tooltip":
                "场景·scene 只写默认环境：地点与固定陈设（例：都市街头和路灯 / 诊室和问诊桌）——"
                "不写时段（归 light 预设）、不写画风（归 look）、不写站位（归 setting）。"
                "在 V3 里接了场景资产时，编译器会机械丢弃这里的文字（资产覆盖剧本场景），图与 ambience 生效；"
                "未接场景资产时使用默认环境，留空由模型设计。\n"
                "ENVIRONMENT ONLY - the default place and its fixed contents (a city street with lamps; a clinic "
                "consultation room). No time of day (light preset), no art style (look), no staging (setting). "
                "A wired scene asset drops this text entirely at compile time; it serves scene-less runs and "
                "story-required rooms only."}),
            "light": (["follow_image", "daylight", "golden_hour", "night_lamp", "night_screen",
                       "late_night", "morning", "custom"], {"default": "follow_image", "tooltip":
                "时段光锚（预设选择器）。follow_image=不写光文字，跟随场景图自带光线；夜戏不接场景图时"
                "必须选夜间预设，否则 H3 默认渲染成白天。night_lamp 是室内台灯场景，户外夜景选 late_night。"
                "custom=在下方 light_text 自定义。注意：接了场景资产时，光锚预设会在编译时被丢弃（场景图自带光线生效，用户的资产赢）。\n"
                "Time-of-day light anchor (preset selector). follow_image writes no light text and "
                "follows the wired scene image's own lighting; a night story without a visibly dark "
                "scene image MUST pick a night preset or H3 renders daylight. night_lamp is an INDOOR "
                "table-lamp scene; outdoor night picks late_night. custom = write your own below."}),
            "light_text": ("STRING", {"default": "", "multiline": True, "tooltip":
                "自定义光锚，仅在 light=custom 时填写（留空会响亮报错）；会被译成英文 execution.light。\n"
                "Custom light anchor, only with light=custom (empty here is a loud error); "
                "translated into the English execution.light."}),
            "character_designs": ("STRING", {"default": "", "multiline": True, "tooltip":
                "默认角色外观 JSON，例如 {\"character_1\":\"穿红外套的猫\"}。仅未接该角色资产时使用；"
                "留空由模型自行设计。不要把默认外观写进站位或动作。修订模式继承上一版。\n"
                "Optional default appearances by role ID. Wired character assets override each role's default."}),
            # 连续性审片（2026-11-05）：资产过了契约校验之后，再让一个"读者"
            # 跨镜头复核一遍（纯文本，不需要视觉模型）。发现自相矛盾或缺戏
            # 会自动回修重摇；3 次内修不好就交付通过契约校验的版本 + 警告。
            "critic": (["on", "off"], {"default": "on", "tooltip":
                "连续性审片 · continuity critic:\n"
                "on：资产通过契约校验后再做一次跨镜头复核（覆盖、连续性、单镜可演、节奏），"
                "发现观众看得见的矛盾自动回修重摇；每次生成多一次模型调用（约 30~60 秒），"
                "修不好则交付通过校验的版本并附警告。off：只做契约校验（快、零额外调用）。\n"
                "on: after the contract passes, a second reader checks the asset across shots "
                "(coverage, continuity, one beat per segment, pacing) and repairs what a viewer "
                "would notice. One extra model call per generation; an unrepaired asset is still "
                "delivered with the notes attached. off: contract only."}),
        }}

    RETURN_TYPES = ("MIE_STORYBOARD", "STRING", "STRING", "INT")
    RETURN_NAMES = ("storyboard", "storyboard_json", "storyboard_text", "shot_count")
    FUNCTION = "generate_storyboard"
    CATEGORY = MY_CATEGORY

    def generate_storyboard(self, llm_service_connector, concept, shot_count=2, seed=0,
                            temperature=0.2, max_tokens=8192, timeout=120, **options):
        planner = AssetStoryboardEnhancer(llm_service_connector, temperature=temperature,
                                           max_tokens=max_tokens, timeout=timeout)
        base_asset = None
        diff_lines = None
        if options.get("mode") == "revise":
            base_asset = options.pop("storyboard", None)
        asset = planner(concept, shot_count, seed=seed, base_asset=base_asset, **options)
        if base_asset:
            from .storyboard_asset_generator import asset_diff
            diff_lines = asset_diff(base_asset, asset)
        import json
        text = preview_asset(asset, diff_lines=diff_lines, warnings=planner.warnings)
        return asset, json.dumps(asset, ensure_ascii=False, indent=2), text, len(asset["shots"])

    @classmethod
    def IS_CHANGED(cls, llm_service_connector, **options):
        import json
        from .storyboard_asset_generator import PROMPT_PATH
        digest = hashlib.sha256(PROMPT_PATH.read_bytes())
        digest.update(json.dumps(options, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        state = getattr(llm_service_connector, "get_state", None)
        if callable(state):
            digest.update(str(state()).encode("utf-8"))
        else:
            for key in ("api_url", "api_token", "model"):
                digest.update(str(getattr(llm_service_connector, key, "")).encode("utf-8"))
        return digest.hexdigest()

    is_changed = IS_CHANGED
