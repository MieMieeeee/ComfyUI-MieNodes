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
                    "You are a creative prompt engineer. Generate exactly 1 random, high-quality, natural English prompt for AI image generation.\n\n"
                    "The formula for a high-quality prompt is:\n"
                    "Style/Art Form + Main Subject + Layered Description of Visual Elements (composition, color and tone, lighting, texture and material) + Environment + Atmosphere + Fine Details + Quality Requirements.\n\n"
                    "Your response must consist of exactly 1 complete, concise prompt, ready for direct use in Stable Diffusion or Midjourney, without conversational text, explanations, or extra formatting.\n\n"
                    "Example output:\n"
                    "Oil painting, a majestic stag standing in a misty forest clearing, soft golden sunrise lighting, intricate antler details, dewdrops on tall grass, ethereal atmosphere, ultra-detailed, award-winning artwork.\n"
                )
            else:
                system_msg = (
                    "You are an expert prompt creator for AI image generation. "
                    "Randomly generate a concise, natural English prompt suitable for direct use in Stable Diffusion or Midjourney. "
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
                    "You are a creative prompt engineer. Your mission is to analyze the provided description and generate exactly 1 high-quality, natural English prompt for AI image generation.\n\n"
                    "The formula for a high-quality prompt is:\n"
                    "Style/Art Form + Main Subject + Layered Description of Visual Elements (composition, color and tone, lighting, texture and material) + Environment + Atmosphere + Fine Details + Quality Requirements.\n\n"
                    "Your response must consist of exactly 1 complete, concise prompt, ready for direct use in Stable Diffusion or Midjourney, without conversational text, explanations, or extra formatting.\n\n"
                    "Example input:\n"
                    "中国国画风格的桂林山水\n"
                    "Example output:\n"
                    "Chinese ink painting, picturesque Guilin landscape, majestic karst mountains shrouded in mist, tranquil Li River winding through lush green valleys, soft diffused lighting, delicate brushwork, serene atmosphere, exquisite detail, masterpiece quality.\n"
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": input_text},
                ]
        prompt = llm_service_connector.invoke(messages)
        return prompt,

    def is_changed(self, llm_service_connector, input_text, mode, seed):
        # 创建一个哈希对象
        hasher = hashlib.md5()

        # 添加基本类型的输入到哈希
        hasher.update(input_text.encode('utf-8'))
        hasher.update(mode.encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))

        # 处理 llm_service_connector
        connector_state = str(llm_service_connector).encode('utf-8')
        hasher.update(connector_state)

        # 返回哈希值
        return hasher.hexdigest()


KONTEXT_PRESETS = {
    "Komposer: Teleport": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. "
            "Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background, etc. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Move Camera": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Move the camera to reveal new aspects of the scene. Provide a highly different camera movement based on the scene (e.g., top view of the room, side portrait view of the person, etc). "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Relight": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Suggest a new lighting setting for the image. Propose a professional lighting stage and setting, possibly with dramatic color changes, alternate times of day, or the inclusion/removal of natural lights. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Product": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Turn this image into the style of a professional product photo. Describe a scene that could show a different aspect of the item in a highly professional catalog, including possible light settings, camera angles, zoom levels, or a scenario where the item is being used. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Zoom": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Zoom on the subject of the image. If a subject is provided, zoom on it; otherwise, zoom on the main subject. Provide a clear zoom effect and describe the visual result. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Colorize": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Colorize the image. Provide a specific color style or restoration guidance. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Movie Poster": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Create a movie poster with the subjects of this image as the main characters. Choose a random genre (action, comedy, horror, etc.) and make it look like a movie poster. "
            "If a title is provided, fit the scene to the title; otherwise, make up a title based on the image. Stylize the title and add taglines, quotes, and other typical movie poster text. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Cartoonify": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Turn this image into the style of a cartoon, manga, or drawing. Include a reference of style, culture, or time (e.g., 90s manga, thick-lined, 3D Pixar, etc.). "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Remove Text": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Remove all text from the image. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Haircut": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Change the haircut of the subject. Suggest a specific haircut, style, or color that would suit the subject naturally. Describe visually how to edit the subject’s hair to achieve this new haircut. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Bodybuilder": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Largely increase the muscles of the subjects while keeping the same pose and context. Describe visually how to edit the subjects so they become bodybuilders with exaggerated large muscles, and change clothes if needed to reveal the new body. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Remove Furniture": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Remove all furniture and appliances from the image. Explicitly mention removing lights, carpets, curtains, etc., if present. "
            "Output only the transformation instruction, without any explanations, numbering, or extra text."
        )
    },
    "Interior Design": {
        "system": (
            "You are a creative prompt engineer. Your mission is to analyze the provided image and generate a distinct image transformation instruction.\n\n"
            "Redo the interior design of this image. Imagine design elements and light settings that could match the room and offer a new artistic direction, ensuring that the room structure (windows, doors, walls, etc.) remains identical. "
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
        return kontext_prompt,

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
