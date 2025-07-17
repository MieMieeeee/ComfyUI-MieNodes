MY_CATEGORY = "🐑 MieNodes/🐑 Translator"


class TextTranslator(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "target_language": (
                    ["zh", "en", "es", "fr", "de", "ja", "ko", "ru", "it", "pt"],
                    {"default": "zh"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate_text"
    CATEGORY = MY_CATEGORY

    target_language_map = {
        "zh": "Chinese",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "it": "Italian",
        "pt": "Portuguese",
    }

    def translate_text(self, llm_service_connector, text, target_language):
        language_name = self.target_language_map.get(target_language, target_language)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a translation engine. "
                    f"Translate any user input into {language_name}. "
                    "Only return the translated text, without explanation, notes, or extra formatting."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]
        translated_text = llm_service_connector.invoke(messages)
        return translated_text,


