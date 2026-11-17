"""ComfyUI shell - the translation text domain lives in mie_prompt.

``build_messages`` and the language table moved to
``mie_prompt.translator`` (batch 2c); the node stays here.
"""

from __future__ import annotations

from mie_prompt.translator import TARGET_LANGUAGE_MAP, build_messages  # noqa: F401

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

    target_language_map = TARGET_LANGUAGE_MAP

    def translate_text(self, llm_service_connector, text, target_language):
        messages = build_messages(text, target_language)
        translated_text = llm_service_connector.invoke(messages)
        return translated_text.strip(),
