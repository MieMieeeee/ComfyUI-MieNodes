import requests

MY_CATEGORY = "🐑 MieNodes/🐑 Translator"


class LLMServiceConfig(object):
    def __init__(self, api_url, api_token, model):
        self.api_url = api_url
        self.api_token = api_token
        self.model = model


class SetLLMServiceConfig(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://api.siliconflow.cn/v1/chat/completions"}),
                "api_token": ("STRING", {"default": ""}),
                "model": ("STRING", {
                    "default": "deepseek-ai/DeepSeek-V3",
                }),
            },
        }

    RETURN_TYPES = ("LLMServiceConfig",)
    RETURN_NAMES = ("llm_service_config",)
    FUNCTION = "execute"

    CATEGORY = MY_CATEGORY

    def execute(self, api_url, api_token, model):
        return LLMServiceConfig(api_url, api_token, model),


class SetSiliconFlowLLMServiceConfig(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model": ("STRING", {
                    "default": "deepseek-ai/DeepSeek-V3",
                }),
            },
        }

    RETURN_TYPES = ("LLMServiceConfig",)
    RETURN_NAMES = ("llm_service_config",)
    FUNCTION = "execute"
    DESCRIPTION = """
    Only test with deepseek-ai/DeepSeek-V3.
    """

    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model):
        return LLMServiceConfig("https://api.siliconflow.cn/v1/chat/completions", api_token, model),


class TextTranslator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_config": ("LLMServiceConfig",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "target_language": (["zh", "en", "es", "fr", "de", "ja", "ko", "ru", "it", "pt"], {"default": "zh"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate_text"

    CATEGORY = MY_CATEGORY

    def translate_text(self, llm_service_config, text, target_language):

        url = llm_service_config.api_url
        model = llm_service_config.model
        api_token = llm_service_config.api_token

        # 构造请求的 payload
        payload = {
            "model": f"{model}",  # 使用的 LLM 模型
            "messages": [
                {
                    "role": "user",
                    "content": f"Translate the following text to {target_language}: {text}"
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
        }

        # 设置请求头
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

        # 发送 POST 请求
        response = requests.post(url, json=payload, headers=headers)

        # 检查响应状态
        if response.status_code == 200:
            response_data = response.json()
            # 提取翻译结果
            try:
                translated_text = response_data["choices"][0]["message"]["content"]
                return translated_text,
            except KeyError:
                raise ValueError("Unexpected response format: missing 'content'.")
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # @classmethod
    # def IS_CHANGED(cls, **kwargs):
    #     return float("nan")
