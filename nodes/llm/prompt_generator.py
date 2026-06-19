import hashlib
import os
import json

import folder_paths
script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MY_CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"

try:
    from _mienodes_internal.core.utils import image_tensor_to_data_url
except ImportError:
    from ...core.utils import image_tensor_to_data_url

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_dict, load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_dict, load_prompt_text

def get_user_presets_file():
    base_dir = script_directory
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

HYVIDEO_T2V_SYSTEM_PROMPT = load_prompt_text("hunyuan/t2v")

HYVIDEO_I2V_SYSTEM_PROMPT = load_prompt_text("hunyuan/i2v")

ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE = load_prompt_text("zimage/t2i")

FLUX2_T2I_SYSTEM_PROMPT = load_prompt_text("flux2/t2i")

LTX2_SYSTEM_PROMPT = load_prompt_text("ltx2/system")

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
                system_msg = load_prompt_text("prompt_generator/random_advanced")
            else:
                system_msg = load_prompt_text("prompt_generator/random_simple")
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "Generate a random prompt."},
            ]
        else:
            # 不为空时，按simple或advanced处理
            if mode == "simple":
                system_msg = load_prompt_text("prompt_generator/translate_simple")
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": input_text},
                ]
            else:
                system_msg = load_prompt_text("prompt_generator/expand_advanced")
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


KONTEXT_PRESETS = load_prompt_dict("kontext/presets")

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
FRAME_TRANSITION_SYSTEM_PROMPTS = load_prompt_dict("frame_transition/system_prompts")

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
            {"role": "user", "content": combined or "请根据系统提示生成结果。"},
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
            {"role": "user", "content": prompt.strip() or "请根据系统提示生成结果。"},
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

FLUX_KLEIN_T2V_SYSTEM_PROMPT = load_prompt_text("flux_klein/t2v")

class FluxKleinT2VPromptGenerator(object):
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
    RETURN_NAMES = ("flux_klein_t2v_prompt",)
    FUNCTION = "generate_flux_klein_t2v_prompt"
    CATEGORY = MY_CATEGORY

    def generate_flux_klein_t2v_prompt(self, llm_service_connector, input_text, seed=None):
        system_msg = FLUX_KLEIN_T2V_SYSTEM_PROMPT
        user_msg = input_text.strip() or "Generate a random high-quality prompt for FLUX.2 [klein]."
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
