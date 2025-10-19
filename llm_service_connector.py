import requests
import time

from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 LLM Service Config"


# 引入 time 模块用于在重试间增加延迟
class GeneralLLMServiceConnector:
    def __init__(self, api_url, api_token, model, timeout=30, max_retries=3, retry_delay=5):
        self.api_url = api_url
        self.api_token = api_token
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries  # 最大重试次数
        self.retry_delay = retry_delay  # 重试间隔（秒）

    def generate_payload(self, messages, **kwargs):
        """
        生成标准的 OpenAI 兼容服务的 Payload。
        子类如果需要不同的默认参数或结构，可以重写此方法。
        """
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "text"},
        }

    def invoke(self, messages, **kwargs):
        """
        调用 LLM 服务，并实现针对瞬时错误的重试机制。
        重试包括：Timeout, ConnectionError, 和 5xx 状态码。
        """
        payload = self.generate_payload(messages, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # 增加重试循环
        for attempt in range(self.max_retries):
            is_last_attempt = (attempt == self.max_retries - 1)

            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)

                # 请求成功（状态码 200）
                if response.status_code == 200:
                    response_data = response.json()
                    try:
                        return response_data["choices"][0]["message"]["content"]
                    except KeyError:
                        raise ValueError(
                            f"Unexpected response format: missing 'content'. Response: {response.text[:200]}...")

                # 服务器错误（5xx）: 瞬时错误，尝试重试
                elif 500 <= response.status_code < 600:
                    error_message = f"API returned {response.status_code}. Server Side Error."
                    if is_last_attempt:
                        raise Exception(
                            f"{error_message} Max retries ({self.max_retries}) exceeded. Response: {response.text}")
                    else:
                        mie_log(
                            f"{error_message} Retrying in {self.retry_delay} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue  # 进入下一次循环重试

                # 其他非 200/5xx 错误（例如 4xx 客户端错误）: 不重试，直接抛出
                else:
                    raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

            # 处理 Timeout 和 ConnectionError 等瞬时请求异常
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                error_type = type(e).__name__
                if is_last_attempt:
                    raise Exception(f"Request failed: {error_type}. Max retries ({self.max_retries}) exceeded.")
                else:
                    mie_log(
                        f"Request failed with {error_type}. Retrying in {self.retry_delay} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    continue  # 进入下一次循环重试

            # 其他非瞬时 requests 错误 (例如 SSL/Proxy 错误)
            except requests.exceptions.RequestException as e:
                # 这类错误通常不是瞬时的，直接抛出
                raise Exception(f"A non-retryable request error occurred: {e}")

        # 理论上不会执行到这里，但以防万一
        raise Exception(f"LLM Service failed after {self.max_retries} attempts due to an unknown error.")

    def get_state(self):
        """返回用于比较状态的字符串表示"""
        # 恢复为原先的无分隔符形式，保证与历史行为一致（避免回归）
        return f"{self.api_url}{self.api_token}{self.model}"


class StandardOpenAICompatibleConnector(GeneralLLMServiceConnector):
    """
    针对 SiliconFlow, ZhiPu, Kimi 等具有标准 OpenAI 兼容参数的 API。
    """

    def generate_payload(self, messages, **kwargs):
        # 封装 OpenAI 兼容服务的通用参数
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


# 适配SiliconFlow
class SiliconFlowConnectorGeneral(StandardOpenAICompatibleConnector):
    api_url = "https://api.siliconflow.cn/v1/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)


class ZhiPuConnectorGeneral(StandardOpenAICompatibleConnector):
    api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)


class KimiConnectorGeneral(StandardOpenAICompatibleConnector):
    api_url = "https://api.moonshot.cn/v1/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)


class GithubModelsConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://models.github.ai/inference/chat/completions"

    def __init__(self, api_token, model):
        # 继承 GeneralLLMServiceConnector 的默认 Payload
        super().__init__(self.api_url, api_token, model)


class BailianLLMServiceConnector(GeneralLLMServiceConnector):
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)

    def generate_payload(self, messages, **kwargs):
        # 阿里百炼（通义千问）的 Payload 可能有所不同，这里保留其特殊性
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # 可以根据需要添加其他参数
        }


class DeepSeekConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://api.deepseek.com/chat/completions"

    def __init__(self, api_token, model):
        super().__init__(self.api_url, api_token, model)


class GeminiConnectorGeneral(GeneralLLMServiceConnector):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_token, model):
        self.model = model
        self.api_token = api_token  # 需要保留，因为 Token 是通过 URL 参数传递的
        api_url = f"{self.base_url}/{model}:generateContent"
        # 继承基类的 timeout, max_retries, retry_delay
        super().__init__(api_url, api_token, model)

    def generate_payload(self, messages, **kwargs):
        # 适配 Gemini 特有的 Payload 格式
        contents = []
        for msg in messages:
            # Gemini 要求 'role' 是 'user' 或 'model'
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        return {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 10240),
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9),
                "topK": kwargs.get("top_k", 50)
            }
        }

    def invoke(self, messages, **kwargs):
        """
        重写 invoke 方法以处理 Gemini 特有的认证方式 (URL 参数) 和响应解析，
        但重试循环逻辑将**尽可能依赖基类**。

        为了利用基类的重试循环，我们构造一个辅助方法或者在 invoke 中模拟基类逻辑
        或者**直接在 invoke 中实现自己的循环并专注于 Gemini的特殊性**。
        由于认证方式（URL参数）和返回解析（candidates 列表）不同，
        最清晰的办法是重写整个 invoke，但利用基类的重试参数。
        """
        payload = self.generate_payload(messages, **kwargs)
        headers = {"Content-Type": "application/json"}
        # Gemini 认证方式：Token 作为 URL 参数
        url = f"{self.api_url}?key={self.api_token}"

        for attempt in range(self.max_retries):
            is_last_attempt = (attempt == self.max_retries - 1)

            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

                if response.status_code == 200:
                    response_data = response.json()
                    # 适配 Gemini 响应解析: candidates -> content -> parts -> text
                    if not response_data.get("candidates"):
                        raise ValueError(f"No candidates in response. Response: {response.text}")
                    return response_data["candidates"][0]["content"]["parts"][0]["text"]

                # 5xx 瞬时错误处理 (利用基类逻辑)
                elif 500 <= response.status_code < 600:
                    error_message = f"API returned {response.status_code}. Server Side Error."
                    if is_last_attempt:
                        raise Exception(
                            f"{error_message} Max retries ({self.max_retries}) exceeded. Response: {response.text}")
                    else:
                        mie_log(
                            f"[{self.model}] {error_message} Retrying in {self.retry_delay} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue  # 进入下一次循环重试

                else:
                    raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

            # 处理 Timeout 和 ConnectionError 等瞬时请求异常 (利用基类逻辑)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                error_type = type(e).__name__
                if is_last_attempt:
                    raise Exception(f"Request failed: {error_type}. Max retries ({self.max_retries}) exceeded.")
                else:
                    mie_log(
                        f"[{self.model}] Request failed with {error_type}. Retrying in {self.retry_delay} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    continue  # 进入下一次循环重试

            # 其他 requests 错误
            except requests.exceptions.RequestException as e:
                raise Exception(f"[{self.model}] A non-retryable request error occurred: {e}")

            # 其他 Python 异常 (例如 JSON 解析错误)
            except Exception as e:
                # 捕获其他非网络错误，例如 ValueError（如 No candidates in response）
                raise Exception(f"[{self.model}] Unknown error during API call: {e}")

        # 理论上不会执行到这里
        raise Exception(f"[{self.model}] LLM Service failed after {self.max_retries} attempts due to an unknown error.")


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
                        "deepseek-ai/DeepSeek-V3.1-Terminus",
                        "deepseek-ai/DeepSeek-V3.2-Exp",
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


class SetZhiPuLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "GLM-4-Flash-250414",
                        "Custom",
                    ],
                    {"default": "GLM-4-Flash-250414"},
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
            model = "GLM-4-Flash-250414"  # 默认模型
        return ZhiPuConnectorGeneral(api_token, model),


class SetKimiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "kimi-k2-0711-preview",
                        "moonshot-v1-8k",
                        "moonshot-v1-32k",
                        "moonshot-v1-128k",
                        "moonshot-v1-auto",
                        "kimi-latest",
                        "moonshot-v1-8k-vision-preview",
                        "moonshot-v1-32k-vision-preview",
                        "moonshot-v1-128k-vision-preview",
                        "kimi-thinking-preview",
                        "Custom",
                    ],
                    {"default": "kimi-k2-0711-preview"},
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
            model = "kimi-k2-0711-preview"  # 默认模型
        return KimiConnectorGeneral(api_token, model),


class SetDeepSeekLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "deepseek-chat",
                        "deepseek-reasoner",
                        "Custom",
                    ],
                    {"default": "deepseek-chat"},
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
            model = "deepseek-chat"  # 默认模型
        return DeepSeekConnectorGeneral(api_token, model),


class SetGeminiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "Custom",
                    ],
                    {"default": "gemini-2.5-pro"},
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
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "gemini-2.5-pro"
        return GeminiConnectorGeneral(api_token, model),


class SetBailianLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "qwen-plus",
                        "qwen-max",
                        "qwen-flash",
                        "qwen-turbo",
                        "qwen-long",
                        "Custom",
                    ],
                    {"default": "qwen-flash"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "自定义模型名（当选择Custom时生效）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model=""):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "qwen-flash"
        return BailianLLMServiceConnector(api_token, model),


class CheckLLMServiceConnectivity(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, llm_service_connector):
        try:
            # 只发一个空消息（有些API需要messages至少有一条，给个简单的提示）
            test_messages = [{"role": "user", "content": "你是什么模型？"}]
            result = llm_service_connector.invoke(test_messages)
            # 只要没报错，说明服务可联通
            return mie_log(f"LLM服务接口可联通 (HTTP 200 + 正常响应), 返回内容: {result}"),
        except Exception as e:
            return mie_log(f"LLM服务检测失败: {str(e)}"),


# 通用调用节点：对接任意已创建的 LLMServiceConnector
class CallLLMService(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "max_tokens": ("INT", {"default": 512, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "call"
    CATEGORY = MY_CATEGORY

    def call(self, llm_service_connector, input_text, temperature=0.7, top_p=0.9, max_tokens=512, seed=None):
        """
        一个简单的通用节点，将纯文本包装为用户消息并调用任意 LLMServiceConnector 的 invoke 方法。
        该节点不会改变底层 connector 的行为或 state。
        """
        messages = [{"role": "user", "content": input_text}]
        # 将可选参数直接转发给 connector.invoke
        result = llm_service_connector.invoke(messages, seed=seed, temperature=temperature, top_p=top_p,
                                              max_tokens=max_tokens)
        return (result.strip(),)
