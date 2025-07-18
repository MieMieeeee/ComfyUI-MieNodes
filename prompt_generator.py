import hashlib

MY_CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"


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
    "Komposer: Teleport": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. "
            "Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background, etc. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Move Camera": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Move the camera to reveal new aspects of the scene. Provide a highly different camera movement based on the scene (e.g., top view of the room, side portrait view of the person, etc). "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Relight": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Suggest a new lighting setting for the image. Propose a professional lighting stage and setting, possibly with dramatic color changes, alternate times of day, or the inclusion/removal of natural lights. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Product": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Turn this image into the style of a professional product photo. Describe a scene that could show a different aspect of the item in a highly professional catalog, including possible light settings, camera angles, zoom levels, or a scenario where the item is being used. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Zoom": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Zoom on the subject of the image. If a subject is provided, zoom on it; otherwise, zoom on the main subject. Provide a clear zoom effect and describe the visual result. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Colorize": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Colorize the image. Provide a specific color style or restoration guidance. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Movie Poster": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Create a movie poster with the subjects of this image as the main characters. Choose a random genre (action, comedy, horror, etc.) and make it look like a movie poster. "
            "If a title is provided, fit the scene to the title; otherwise, make up a title based on the image. Stylize the title and add taglines, quotes, and other typical movie poster text. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Cartoonify": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Turn this image into the style of a cartoon, manga, or drawing. Include a reference of style, culture, or time (e.g., 90s manga, thick-lined, 3D Pixar, etc.). "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Remove Text": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Remove all text from the image. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Haircut": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Change the haircut of the subject. Suggest a specific haircut, style, or color that would suit the subject naturally. Describe visually how to edit the subject’s hair to achieve this new haircut. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Bodybuilder": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Largely increase the muscles of the subjects while keeping the same pose and context. Describe visually how to edit the subjects so they become bodybuilders with exaggerated large muscles, and change clothes if needed to reveal the new body. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Remove Furniture": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Remove all furniture and appliances from the image. Explicitly mention removing lights, carpets, curtains, etc., if present. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Interior Design": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Redo the interior design of this image. Imagine design elements and light settings that could match the room and offer a new artistic direction, ensuring that the room structure (windows, doors, walls, etc.) remains identical. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Skin Spot Removal": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Smooth and even out the subject’s skin tone by removing visible spots, freckles, blemishes, or dark marks, while keeping skin texture natural. Enhance clarity, radiance, and overall skin health without altering facial features or expressions. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Slim Belly": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Slim down the subject’s abdominal area to create a visibly flatter, more toned stomach. Adjust the waistline and contour naturally, ensuring proportions remain realistic and clothing fits the new shape. Do not modify other body areas. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Pose Reference": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Change the subject’s pose to a specific reference pose. Clearly describe the new body position, limb placement, and posture (for example: arms raised overhead, one leg bent, side profile). Adjust clothing and scene as needed for realism. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Expression Reference": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Change the subject’s facial expression to a specified emotion or action (for example: surprise, laughter, wink, pout). Maintain realism and accurate facial muscle movement. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Seasonal Change": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Transform the scene to reflect a different season (for example: turn summer scenery to winter with snow, or spring with blooming flowers). Adjust lighting, environment, and clothing for authenticity. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Age Progression": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Alter the subject’s age, making them visibly older or younger while maintaining recognizable features and natural skin texture. Adjust hair color if appropriate. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Weather Effect": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Apply a specific weather effect to the image (for example: heavy rain, fog, bright sunshine, thunderstorm). Adjust lighting, reflections, and surroundings for realism. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Fashion Update": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Change the subject's outfit to a specific fashion style or era (for example: futuristic streetwear, 1920s flapper dress, formal business suit). Ensure clothes fit naturally on the subject and match the setting. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Add Accessories": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Add visually fitting accessories to the subject (for example: sunglasses, hat, jewelry, headphones). Ensure accessories appear naturally integrated and suit the subject’s style. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Fantasy Transformation": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction."
            "Transform the subject into a fantasy character or creature (for example: elf with pointed ears, cyborg with visible mechanical parts, mermaid with a tail). Modify clothing, accessories, and background as needed for consistency. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    }
}


class KontextPromptGenerator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "image_description": ("STRING", {"default": "", "multiline": True}),
                "edit_instruction": ("STRING", {"default": "", "multiline": True}),
                "preset": (list(KONTEXT_PRESETS.keys()), {"default": "Komposer: Teleport"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kontext_prompt",)
    FUNCTION = "generate_kontext_prompt"
    CATEGORY = MY_CATEGORY

    def generate_kontext_prompt(self, llm_service_connector, image_description, edit_instruction, preset, seed=None):
        preset_data = KONTEXT_PRESETS.get(preset)
        if not preset_data:
            raise ValueError(f"Unknown preset: {preset}")

        # 用户输入拼到user消息中，给LLM最大上下文
        user_content = ""
        if image_description.strip():
            user_content += f"Image description: {image_description.strip()}\n"
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

    def is_changed(self, llm_service_connector, image_description, edit_instruction, preset, seed):
        # 创建一个哈希对象
        hasher = hashlib.md5()

        # 添加基本类型的输入到哈希
        hasher.update(image_description.encode('utf-8'))
        hasher.update(edit_instruction.encode('utf-8'))
        hasher.update(preset.encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))

        # 处理 KONTEXT_PRESETS 的内容
        preset_data = KONTEXT_PRESETS.get(preset)
        if preset_data:
            hasher.update(preset_data["system"].encode('utf-8'))

        # 处理 llm_service_connector
        connector_state = str(llm_service_connector).encode('utf-8')
        hasher.update(connector_state)

        # 返回哈希值
        return hasher.hexdigest()
