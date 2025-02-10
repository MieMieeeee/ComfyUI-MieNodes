from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Common"

# Learned a lot from https://github.com/cubiq/ComfyUI_essentials

class ShowAnythingMie(object):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (("*", {})),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    CATEGORY = MY_CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    def execute(self, anything):
        """
        以字符形式打印输入的内容。

        参数：
        - input (*): 输入的内容

        返回：
        - 给UI的json格式
        """

        result = str(anything)
        mie_log(f"ShowAnythingMie: {result}")

        return {"ui": {"text": result}, "result": ()}
