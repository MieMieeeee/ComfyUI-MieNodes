try:
    from .utils import load_plugin_config, resolve_token
except ImportError:
    from utils import load_plugin_config, resolve_token

MY_CATEGORY = "🐑 MieNodes/🐑 TTS Service Config"

class TTSConnector:
    def __init__(self, manual_token, config_file="mie_llm_keys.json", config_key="bailian", prefer_local_config=True):
        self.manual_token = manual_token
        self.config_file = config_file
        self.config_key = config_key
        self.prefer_local_config = prefer_local_config

    @property
    def api_token(self):
        return resolve_token(
            self.manual_token, 
            default_key="bailian", 
            config_file=self.config_file, 
            config_key=self.config_key, 
            prefer_local=self.prefer_local_config
        )

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
        # Pass all parameters to the connector so it can resolve the token dynamically
        return (TTSConnector(api_token, config_file, config_key, prefer_local_config),)
