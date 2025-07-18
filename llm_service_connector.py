import requests

MY_CATEGORY = "🐑 MieNodes/🐑 LLM Service Config"


class GeneralLLMServiceConnector:
    def __init__(self, api_url, api_token, model):
        self.api_url = api_url
        self.api_token = api_token
        self.model = model

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "text"},
        }

    def invoke(self, messages, **kwargs):
        payload = self.generate_payload(messages, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                try:
                    return response_data["choices"][0]["message"]["content"]
                except KeyError:
                    raise ValueError("Unexpected response format: missing 'content'.")
            else:
                raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("Request timed out after 30 seconds.")

    def get_state(self):
        """返回用于比较状态的字符串表示"""
        return f"{self.api_url}|{self.api_token}|{self.model}"


# 适配SiliconFlow
class SiliconFlowConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://api.siliconflow.cn/v1/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1),
            "response_format": {"type": "text"},
        }

class GithubModelsConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://models.github.ai/inference/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)

class SetGeneralLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://api.siliconflow.cn/v1/chat/completions"}),
                "api_token": ("STRING", {"default": "token"}),
                "model_select": ("STRING", {"default": "deepseek-ai/DeepSeek-V3"}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_url, api_token, model_select):
        return GeneralLLMServiceConnector(api_url, api_token, model_select),


class SetGithubModelsLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "openai/gpt-4.1",
                        "Custom",
                    ],
                    {"default": "openai/gpt-4.1"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model=""):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "openai/gpt-4.1"  # 默认模型
        return GithubModelsConnectorGeneral(api_token, model),


class SetSiliconFlowLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "deepseek-ai/DeepSeek-V3",
                        "THUDM/GLM-Z1-9B-0414",
                        "THUDM/GLM-4-32B-0414",
                        "Qwen/Qwen3-8B",
                        "moonshotai/Kimi-K2-Instruct",
                        "Custom",
                    ],
                    {"default": "THUDM/GLM-4-32B-0414"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model=""):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "THUDM/GLM-4-32B-0414"  # 默认模型
        return SiliconFlowConnectorGeneral(api_token, model),
