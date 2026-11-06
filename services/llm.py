"""The LLM service connector NODES.

The connectors themselves now live in ComfyUI-Mie-Core; this module is the
ComfyUI shell around them.
"""

from mie_llm.client import *  # noqa: F401,F403
from mie_llm.client import (
    BailianCodingPlanConnectorGeneral,
    BailianTokenPlanConnectorGeneral,
    BailianLLMServiceConnector,
    ClaudeConnectorGeneral,
    DeepSeekConnectorGeneral,
    DoubaoConnectorGeneral,
    GeminiConnectorGeneral,
    GeneralLLMServiceConnector,
    GrokConnectorGeneral,
    KimiConnectorGeneral,
    MiniMaxConnectorGeneral,
    MiniMaxTokenPlanConnectorGeneral,
    MiMoConnectorGeneral,
    MiMoTokenPlanConnectorGeneral,
    OllamaConnectorGeneral,
    OpenAIConnectorGeneral,
    OpenRouterConnectorGeneral,
    QianfanConnectorGeneral,
    SiliconFlowConnectorGeneral,
    SiliconFlowJevConnectorGeneral,
    SparkConnectorGeneral,
    StandardOpenAICompatibleConnector,
    ZhiPuCodeConnectorGeneral,
    ZhiPuConnectorGeneral,
)

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    from ..core.utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 LLM Service Config"


class SetGeneralLLMServiceConnector(object):
    """Generic OpenAI-compatible escape hatch (user supplies base URL + model).

    This is also the way in for **local model servers**: llama.cpp
    (`llama-server`, `http://127.0.0.1:8080/v1/chat/completions`), LM Studio
    (port 1234), vLLM (port 8000), or any other OpenAI-compatible runtime.
    Those servers ignore the `Authorization` header, so any token value
    works. The plugin deliberately does not load models itself — every
    connector in this module is an HTTP client for a chat endpoint, which
    is what keeps the plugin dependency-light; see the local-servers box in
    `docs/LLM_SERVICE_SELECTION_GUIDE.md` for start commands and VRAM notes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://api.siliconflow.cn/v1/chat/completions"}),
                "api_token": ("STRING", {"default": ""}),
                "model_select": ("STRING", {"default": "deepseek-ai/DeepSeek-V3"}),
            },
            "optional": {
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "openai_compatible"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_url, api_token, model_select, config_file="mie_llm_keys.json", config_key="openai_compatible", prefer_local_config=True):
        return (GeneralLLMServiceConnector(api_url, api_token, model_select, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetSiliconFlowLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "deepseek-ai/DeepSeek-V4-Pro",
                        "deepseek-ai/DeepSeek-V4-Flash",
                        "deepseek-ai/DeepSeek-V3.2",
                        "Pro/deepseek-ai/DeepSeek-V3.2",
                        "deepseek-ai/DeepSeek-V3.1-Terminus",
                        "zai-org/GLM-5.3",
                        "zai-org/GLM-5.2",
                        "Pro/zai-org/GLM-5.1",
                        "zai-org/GLM-4.5V",
                        "Pro/moonshotai/Kimi-K2.6",
                        "moonshotai/Kimi-K2.7-Code",
                        "Qwen/Qwen3.8-27B",
                        "Qwen/Qwen3.6-35B-A3B",
                        "Qwen/Qwen3.6-27B",
                        "Qwen/Qwen3-VL-32B-Thinking",
                        "Qwen/Qwen3-VL-32B-Instruct",
                        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        "Custom",
                    ],
                    {"default": "deepseek-ai/DeepSeek-V4-Flash"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "siliconflow"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="siliconflow", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "deepseek-ai/DeepSeek-V4-Flash"  # 默认模型
        return (SiliconFlowConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetZhiPuLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "glm-5.3",
                        "glm-5.3-flash",
                        "glm-5.3-flashx",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5-turbo",
                        "glm-5",
                        "glm-4.7",
                        "glm-4.6",
                        "glm-4.5",
                        "glm-4.5-air",
                        "Custom",
                    ],
                    {"default": "glm-5.3"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "zhipu"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="zhipu", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "glm-5.3"  # 默认模型
        return (ZhiPuConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetZhiPuCodeLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "glm-5.3",
                        "glm-5.3-flash",
                        "glm-5.3-flashx",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5-turbo",
                        "glm-5",
                        "glm-4.7",
                        "glm-4.6",
                        "glm-4.5",
                        "glm-4.5-air",
                        "Custom",
                    ],
                    {"default": "glm-5.3"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "zhipu_code"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="zhipu_code", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "glm-5.3"
        return (ZhiPuCodeConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetKimiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "kimi-k3",
                        "kimi-k2.7-code",
                        "kimi-k2.7-code-highspeed",
                        "kimi-k2.6",
                        "Custom",
                    ],
                    {"default": "kimi-k3"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "kimi"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="kimi", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "kimi-k3"  # 默认模型
        return (KimiConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetDeepSeekLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "deepseek-v4-pro",
                        "deepseek-flash",
                        "Custom",
                    ],
                    {"default": "deepseek-flash"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "deepseek"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="deepseek", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "deepseek-flash"  # 默认模型
        return (DeepSeekConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetMiniMaxLLMServiceConnector(object):
    """Standard MiniMax Open Platform connector.

    Use this node when you have a standard MiniMax Open Platform API key
    (`eyJ...` JWT format) issued from the MiniMax developer console. For
    the Token Plan / Coding Plan (`sk-cp-...` prefixed) keys, use
    `SetMiniMaxTokenPlanLLMServiceConnector` instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "MiniMax-M2.7",
                        "MiniMax-M2.7-highspeed",
                        "MiniMax-M2.5",
                        "MiniMax-M2.5-highspeed",
                        "MiniMax-M2.1",
                        "MiniMax-M2.1-highspeed",
                        "MiniMax-M2",
                        "Custom",
                    ],
                    {"default": "MiniMax-M2.7"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "minimax_open_platform"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="minimax_open_platform", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "MiniMax-M2.7"
        return (MiniMaxConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetMiniMaxTokenPlanLLMServiceConnector(object):
    """MiniMax Token Plan / Coding Plan connector.

    Use this node when you have a Token Plan / Coding Plan API key
    (`sk-cp-...` prefix). MiniMax-M3.1-Flash-Preview is the default;
    M3 and the older M2 lineup stay available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "MiniMax-M3.1-Flash-Preview",
                        "MiniMax-M3",
                        "MiniMax-M2.7",
                        "MiniMax-M2.7-highspeed",
                        "MiniMax-M2.5",
                        "MiniMax-M2.5-highspeed",
                        "MiniMax-M2.1",
                        "MiniMax-M2.1-highspeed",
                        "MiniMax-M2",
                        "Custom",
                    ],
                    {"default": "MiniMax-M3.1-Flash-Preview"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "minimax"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="minimax", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "MiniMax-M3.1-Flash-Preview"
        return (MiniMaxTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetMiMoLLMServiceConnector(object):
    """Standard Xiaomi MiMo Open Platform connector.

    Use this node when you have a standard MiMo API key (`sk-xxxxx`
    format) issued from the MiMo developer console. For the Token Plan
    / Coding Plan (`tp-xxxxx` prefixed) keys, use
    `SetMiMoTokenPlanLLMServiceConnector` instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "mimo-v2.6-pro",
                        "mimo-v2.6-pro-ultraspeed",
                        "mimo-v2.6-flash",
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "Custom",
                    ],
                    {"default": "mimo-v2.6-pro"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "mimo"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="mimo", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "mimo-v2.6-pro"
        return (MiMoConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetMiMoTokenPlanLLMServiceConnector(object):
    """Xiaomi MiMo Token Plan / Coding Plan connector.

    Use this node when you have a Token Plan / Coding Plan API key
    (`tp-xxxxx` prefix). The Token Plan is a fixed-fee subscription
    with its own base URL and billing; the model lineup is shared
    with the standard tier.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "mimo-v2.6-pro",
                        "mimo-v2.6-pro-ultraspeed",
                        "mimo-v2.6-flash",
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "Custom",
                    ],
                    {"default": "mimo-v2.6-pro"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "mimo_token_plan"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="mimo_token_plan", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "mimo-v2.6-pro"
        return (MiMoTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetDoubaoLLMServiceConnector(object):
    """ByteDance Volcano Ark (火山方舟) Doubao connector.

    Use this node with an Ark API Key from the Volcano console
    (https://console.volcengine.com/ark). The model may be a versioned
    doubao-seed id from the dropdown, or paste an inference Endpoint ID
    (`ep-xxxx`) into `custom_model` with model_select set to `Custom`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "doubao-seed-2-0-pro-260215",
                        "doubao-seed-2-0-lite-260215",
                        "doubao-seed-2-0-mini-260215",
                        "doubao-seed-1-8-251228",
                        "doubao-seed-1-6",
                        "Custom",
                    ],
                    {"default": "doubao-seed-2-0-pro-260215"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "doubao model id or Ark Endpoint ID (ep-xxxx)",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "doubao"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="doubao", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "doubao-seed-2-0-pro-260215"
        return (DoubaoConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetQianfanLLMServiceConnector(object):
    """Baidu Qianfan (百度千帆) ERNIE connector.

    Use this node with a Qianfan ModelBuilder API key
    (https://console.bce.baidu.com/qianfan). Serves the ERNIE family
    plus hosted third-party models (glm / deepseek).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "ernie-5.1",
                        "ernie-5.0",
                        "ernie-5.0-thinking-preview",
                        "ernie-4.5-turbo-128k",
                        "ernie-4.5-turbo-32k",
                        "ernie-4.5-turbo-vl",
                        "glm-5.3",
                        "deepseek-v4.1-flash",
                        "Custom",
                    ],
                    {"default": "ernie-5.1"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "qianfan"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="qianfan", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "ernie-5.1"
        return (QianfanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetSparkLLMServiceConnector(object):
    """iFLYTEK Spark (讯飞星火) X2 connector.

    Use this node with the APIPassword issued from the iFLYTEK console
    (https://console.xfyun.cn). Serves the current Spark-X2 generation
    via the `/x2/` endpoint (model value `spark-x`); the retired
    lite/generalv3 classics live under `/v1/` — reach them with
    `SetGeneralLLMServiceConnector` instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "spark-x",
                        "Custom",
                    ],
                    {"default": "spark-x"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "spark"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="spark", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "spark-x"
        return (SparkConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetOpenAILLMServiceConnector(object):
    """OpenAI connector (gpt-6 / gpt-5 families).

    Use this node with an OpenAI API key (https://platform.openai.com).
    The payload uses `max_completion_tokens` because the newer
    gpt-5+/o-series models reject the legacy `max_tokens` field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "gpt-6-astra",
                        "gpt-6-astra-pro",
                        "gpt-6-luna",
                        "gpt-6-luna-pro",
                        "gpt-5.5",
                        "gpt-5.5-pro",
                        "gpt-5.2-chat",
                        "gpt-5.1",
                        "Custom",
                    ],
                    {"default": "gpt-6-astra"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "openai"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="openai", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "gpt-6-astra"
        return (OpenAIConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetGrokLLMServiceConnector(object):
    """xAI Grok connector.

    Use this node with an xAI API key (https://console.x.ai). Serves the
    grok-4.x families via the OpenAI-compatible /v1/chat/completions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "grok-4.7",
                        "grok-4.6",
                        "grok-4.5",
                        "grok-4.3",
                        "grok-4.20",
                        "grok-4.20-multi-agent",
                        "Custom",
                    ],
                    {"default": "grok-4.7"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "grok"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="grok", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "grok-4.7"
        return (GrokConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetOpenRouterLLMServiceConnector(object):
    """OpenRouter aggregator connector.

    Use this node with an OpenRouter API key (https://openrouter.ai) —
    one key routes to 400+ vendor models using `vendor/model` slugs.
    The dropdown carries a cross-vendor selection verified against the
    live catalog; browse everything at https://openrouter.ai/models and
    paste any slug into `custom_model`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "openrouter/auto",
                        "openai/gpt-6-astra",
                        "anthropic/claude-opus-5.5",
                        "google/gemini-3.8-flash",
                        "x-ai/grok-4.7",
                        "deepseek/deepseek-v4-pro",
                        "z-ai/glm-5.3",
                        "moonshotai/kimi-k3",
                        "qwen/qwen3.5-397b-a17b",
                        "minimax/minimax-m3",
                        "mistralai/mistral-large-2512",
                        "Custom",
                    ],
                    {"default": "openrouter/auto"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Any OpenRouter slug, e.g. openai/gpt-5.2 or vendor/model:free",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "openrouter"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="openrouter", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "openrouter/auto"
        return (OpenRouterConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetClaudeLLMServiceConnector(object):
    """Anthropic Claude connector (official OpenAI-compat layer).

    Use this node with a Claude API key (https://console.claude.com).
    Goes through Anthropic's official OpenAI SDK compatibility layer —
    fine for plain chat / prompt rewriting; features the layer ignores
    (structured outputs, prompt caching, detailed thinking) need the
    native Anthropic API instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "claude-sonnet-5-5",
                        "claude-opus-5-5",
                        "claude-fable-5-1",
                        "claude-haiku-4-5",
                        "claude-opus-5",
                        "claude-sonnet-5",
                        "Custom",
                    ],
                    {"default": "claude-sonnet-5-5"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "anthropic"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="anthropic", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "claude-sonnet-5-5"
        return (ClaudeConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetSiliconFlowJevLLMServiceConnector(object):
    """SiliconFlow Jev / System One decision-model connector.

    Use this node to reach the Jev decision-model replicas hosted on
    SiliconFlow (Kev-4B / SemIf / diffusiongemma) via
    `https://api.siliconflow.cn/v1/systemone` — it reuses your existing
    `siliconflow` API key, and the replicas are free during the
    2026-10-08 promo window. Jev models are decision models, not chat
    models — pair this connector with the `CallJevDecision` node, not
    `CallLLMService`. (TypeSafe's own hosted `jev-latest` is out of
    scope here; its protocol is identical if you ever need it.)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "Kev-4B",
                        "SemIf",
                        "diffusiongemma",
                        "Custom",
                    ],
                    {"default": "Kev-4B"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom decision model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "siliconflow"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="siliconflow", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "Kev-4B"
        return (SiliconFlowJevConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetOllamaLLMServiceConnector(object):
    """Set Ollama (local) LLM service connector.

    Connects to a locally running Ollama instance via its OpenAI-compatible
    ``/v1/chat/completions`` endpoint (Ollama 0.3+). Ollama does NOT require
    an API key -- the ``Authorization: Bearer ...`` header is sent but ignored
    by the server, so an empty ``api_token`` is fine.

    Vision models (llava, llama3.2-vision, qwen2.5vl, gemma3, etc.) work
    through the OpenAI-compat layer: just connect an ``IMAGE`` input to the
    downstream ``CallLLMService`` node and it will be forwarded as an
    ``image_url`` content part. Reasoning models (deepseek-r1, qwen3) emit
    ``<think>...</think>`` blocks; the base class strips them automatically.

    Cold-start note: Ollama loads the model from disk into VRAM on first call,
    which can take 30-90s for 7B+ models. The base class retries on timeout,
    but bumping the ``timeout`` input above the 30s base default avoids the
    wasted first attempt for big models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "host": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {
                    "default": "",
                    "placeholder": "Enter local Ollama model name (e.g. qwen2.5, llama3.2, deepseek-r1, llava)",
                }),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "ollama"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 60, "min": 1, "max": 600, "step": 5,
                    "tooltip": "Per-request HTTP timeout in seconds. Default 60s to absorb Ollama's cold-start cost (30-90s for 7B+ models). The base class retries on timeout, but a longer single-attempt timeout avoids the wasted retry."}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, host, model, api_token=None, config_file="mie_llm_keys.json", config_key="ollama", prefer_local_config=True, timeout=60):
        if not model:
            # Sensible default if the user left the field blank. Users can pull
            # other models with `ollama pull <name>` and then edit this field.
            model = "qwen2.5"
        return (OllamaConnectorGeneral(
            host,
            model,
            api_token=api_token,
            config_file=config_file,
            config_key=config_key,
            prefer_local_config=prefer_local_config,
            timeout=timeout,
        ),)
class SetGeminiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "gemini-3.8-flash",
                        "gemini-3.7-flash",
                        "gemini-3.5-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.1-flash-lite",
                        "gemini-3.1-pro-preview",
                        "gemini-3-flash-preview",
                        "Custom",
                    ],
                    {"default": "gemini-3.8-flash"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "gemini"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="gemini", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "gemini-3.8-flash"
        return (GeminiConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetBailianLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "qwen3.8-max",
                        "qwen3.8-flash",
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.7-flash",
                        "qwen3.6-flash",
                        "qwen3.6-plus",
                        "qwen3.5-flash",
                        "qwen3.5-plus",
                        "qwen-plus",
                        "qwen-max",
                        "qwen-flash",
                        "qwen-turbo",
                        "qwen-long",
                        "glm-5.3",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5",
                        "kimi-k3",
                        "kimi-k2.6",
                        "deepseek-v4-pro",
                        "deepseek-v4.1-flash",
                        "Custom",
                    ],
                    {"default": "qwen3.8-max"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="bailian", prefer_local_config=True):
            model = model_select if model_select != "Custom" else custom_model
            if not model:
                model = "qwen3.8-max"
            return (BailianLLMServiceConnector(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetBailianTokenPlanLLMServiceConnector(object):
    """Alibaba Bailian Token Plan connector (multimodal subscription tier).

    Use this node when you have a Token Plan API key (`sk-sp-...` prefix
    issued from https://bailian.console.aliyun.com/). The Token Plan is a
    fixed-fee subscription that grants access to the full Bailian
    multimodal lineup (text / vision / image / audio) via a separate
    endpoint. NOT interchangeable with the PAYG `sk-` key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "auto",
                        "qwen3.8-max",
                        "qwen3.8-flash",
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.6-flash",
                        "deepseek-v4.1-flash",
                        "deepseek-v4-pro",
                        "glm-5.3",
                        "glm-5.2",
                        "Custom",
                    ],
                    {"default": "qwen3.8-max"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian_token_plan"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="bailian_token_plan", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "qwen3.8-max"
        return (BailianTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
class SetBailianCodingPlanLLMServiceConnector(object):
    """Alibaba Bailian Coding Plan connector (subscription tier).

    Use this node when you have a Coding Plan API key (`sk-sp-...` prefix
    issued from https://bailian.console.aliyun.com/ — the official docs
    use the `sk-sp-` prefix for Coding Plan keys too, same shape as Token
    Plan keys but scoped to the `coding.dashscope.aliyuncs.com` host).
    The Coding Plan model list covers both coder models (qwen3-coder-*)
    and general chat models (qwen3.7-plus, glm-5, MiniMax-M2.5, kimi-k2.5).
    NOT interchangeable with the PAYG `sk-` key or with the Token Plan
    `sk-sp-` key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "qwen3.7-plus",
                        "qwen3.6-plus",
                        "qwen3-coder-plus",
                        "qwen3-coder-next",
                        "glm-5",
                        "MiniMax-M2.5",
                        "kimi-k2.5",
                        "qwen3.5-plus",
                        "qwen3-max-2026-01-23",
                        "glm-4.7",
                        "Custom",
                    ],
                    {"default": "qwen3.7-plus"},
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
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian_coding"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="bailian_coding", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "qwen3.7-plus"
        return (BailianCodingPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)
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
class CallJevDecision(object):
    """Ask a Jev / System One decision model a set of structured questions.

    Jev decision models do not write prose — they answer structured
    questions about a piece of state text. `questions_json` holds the
    Jev question spec as a JSON object mapping question names to specs:

        {
          "urgency":  {"type": "noul",
                       "instructions": "Does this message express urgency?"},
          "topic":    {"type": "choice",
                       "instructions": "Classify the inquiry",
                       "criteria": {"billing": "Refunds, payments",
                                    "technical": "Defects, troubleshooting"}},
          "severity": {"type": "score",
                       "instructions": "How severe is the issue?",
                       "criteria": ["Low", "Medium", "High"]}
        }

    - `noul`   -> yes/no probability
    - `choice` -> picked option + full probability distribution
    - `score`  -> position on an ordered scale + per-level probabilities

    Returns the full response (`model` / `answers` / `usage`) as a JSON
    string; parse `answers.<name>` downstream for routing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "state": ("STRING", {"default": "", "multiline": True,
                                     "placeholder": "The text to judge, e.g. a customer email or a dialogue line"}),
                "questions_json": ("STRING", {"default": "", "multiline": True,
                                              "placeholder": '{"q1": {"type": "noul", "instructions": "Is this urgent?"}}'}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("decision_json",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, llm_service_connector, state, questions_json):
        if not hasattr(llm_service_connector, "ask"):
            raise Exception(
                "CallJevDecision needs a Jev connector from the "
                "SetSiliconFlowJevLLMServiceConnector node (got a chat connector)."
            )
        try:
            questions = json.loads(questions_json) if questions_json.strip() else {}
        except json.JSONDecodeError as e:
            raise Exception(
                f"questions_json is not valid JSON: {e}. Keep it as a JSON "
                "object mapping question names to noul/choice/score specs."
            )
        if not isinstance(questions, dict) or not questions:
            raise Exception(
                "questions_json must be a non-empty JSON object mapping "
                "question names to specs (see the node tooltip)."
            )
        data = llm_service_connector.ask(state, questions)
        return (json.dumps(data, ensure_ascii=False, indent=2),)
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
                "image": ("IMAGE",),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "call"
    CATEGORY = MY_CATEGORY

    @staticmethod
    def _single_image_data_url(image):
        """Backward-compat shim: delegate to the shared `core.utils` helper."""
        return image_tensor_to_data_url(image)

    # Old private name kept as an alias for any external caller.
    _image_to_data_url = _single_image_data_url

    def call(self, llm_service_connector, input_text, temperature=0.7, top_p=0.9, max_tokens=512, seed=None, image=None, image_detail="auto"):
        """
        一个简单的通用节点，将纯文本 / 单图包装为用户消息并调用任意 LLMServiceConnector 的 invoke 方法。
        该节点不会改变底层 connector 的行为或 state。

        文本-only 路径保留 `content: <str>` 形态以维持历史行为；只有带图时才
        切到 `content: [<part>, ...]` 多模态形态。
        """
        image_urls = []
        if image is not None:
            url = image_tensor_to_data_url(image)
            if url:
                image_urls.append(url)
        if image_urls:
            content = build_multimodal_user_content(input_text, image_urls, image_detail=image_detail)
            messages = [{"role": "user", "content": content}]
        else:
            # Text-only: keep content as a plain string for back-compat.
            messages = [{"role": "user", "content": input_text if input_text is not None else ""}]
        # 将可选参数直接转发给 connector.invoke
        result = llm_service_connector.invoke(messages, seed=seed, temperature=temperature, top_p=top_p,
                                              max_tokens=max_tokens)
        return (result.strip(),)
