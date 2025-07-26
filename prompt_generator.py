import hashlib
import os
import json

import folder_paths
script_directory = os.path.dirname(os.path.abspath(__file__))

MY_CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"


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