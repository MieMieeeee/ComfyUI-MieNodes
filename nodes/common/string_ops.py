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

class IntToString(object):
    """Convert an INT (or numeric string) to its decimal string form.

    Use this when you have an int widget / link (e.g. MieLoopGetIndex,
    PrimitiveInt) that you need to splice into a STRING path or filename
    builder (e.g. StringConcat|Mie). The output is a plain STRING so the
    downstream StringConcat can statically resolve it.

    Replaces the third-party "CR Integer To String" node in a ComfyUI-CR-Node
    setup. Use it to splice a loop index into a cache-path StringConcat for
    LoadOrCompute|Mie or similar filename builders.

    Inputs:
    - value: an INT (linked or literal). Floats are truncated toward zero;
      numeric strings are parsed; non-numeric input falls back to "0".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, value):
        if isinstance(value, bool):
            return ("1" if value else "0",)
        if isinstance(value, int):
            return (str(value),)
        if isinstance(value, float):
            return (str(int(value)),)
        if isinstance(value, str):
            s = value.strip()
            try:
                return (str(int(s)),)
            except ValueError:
                try:
                    return (str(int(float(s))),)
                except ValueError:
                    return ("0",)
        try:
            return (str(int(value)),)
        except (TypeError, ValueError):
            return ("0",)
