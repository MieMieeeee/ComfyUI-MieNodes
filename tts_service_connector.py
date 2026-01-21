try:
    from .utils import load_plugin_config
except ImportError:
    from utils import load_plugin_config

MY_CATEGORY = "🐑 MieNodes/🐑 TTS Service Config"

class TTSConnector:
    def __init__(self, api_token):
        self.api_token = api_token

class SetBailianTTSConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
            },
            "optional": {
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TTSConnector",)
    RETURN_NAMES = ("tts_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, config_file="mie_llm_keys.json", config_key="bailian", prefer_local_config=True):
        token = _resolve_token(api_token, default_key="bailian", config_file=config_file, config_key=config_key, prefer_local=prefer_local_config)
        return (TTSConnector(token),)

def _resolve_token(api_token, default_key=None, config_file="mie_llm_keys.json", config_key=None, prefer_local=True):
    cfg = load_plugin_config(config_file or "mie_llm_keys.json")
    k = config_key or default_key
    cfg_token = (cfg.get(k) or "")
    api_token = (api_token or "")
    if prefer_local:
        return (cfg_token or api_token)
    return (api_token or cfg_token)
