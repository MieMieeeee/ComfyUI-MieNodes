import requests
from abc import ABC, abstractmethod

MY_CATEGORY = "🐑 MieNodes/🐑 LLM Service Config"


class LLMServiceConnectorBase(ABC):
    def __init__(self, api_url, api_token, model):
        self.api_url = api_url
        self.api_token = api_token
        self.model = model

    @abstractmethod
    def invoke(self, messages, **kwargs):
        pass

    def get_state(self):
        """返回用于比较状态的字符串表示"""
        return f"{self.api_url}|{self.api_token}|{self.model}"


# 适配SiliconFlow
class SiliconFlowConnector(LLMServiceConnectorBase):
    def invoke(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1),
            "response_format": {"type": "text"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            try:
                return response_data["choices"][0]["message"]["content"]
            except KeyError:
                raise ValueError("Unexpected response format: missing 'content'.")
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


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
    RETURN_NAMES = ("llm_service_config",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model=""):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "THUDM/GLM-4-32B-0414"  # 默认模型
        return SiliconFlowConnector("https://api.siliconflow.cn/v1/chat/completions", api_token, model),
