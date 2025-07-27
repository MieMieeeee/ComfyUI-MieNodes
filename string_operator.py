MY_CATEGORY = "🐑 MieNodes/🐑 String Operator"


class StringConcat(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "str1": ("STRING", {"default": "", "multiline": True}),
                "str2": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ","}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, str1, str2, delimiter=","):
        # 支持可选分隔符
        if delimiter:
            result = f"{str1}{delimiter}{str2}"
        else:
            result = f"{str1}{str2}"
        return (result,)