MY_CATEGORY = "🐑 MieNodes/🐑 String Operator"


# StringFormat|Mie autogrow: the JS extension in js/stringFormatAutogrow.js
# reveals only the first N slots at node creation and grows the visible
# input list by one each time the last visible slot gets a connection,
# up to MAX_FORMAT_VALUES. Mirrors the Bernini Conditioning UX
# (reference_image_3 only appears once reference_image_2 is wired).
MAX_FORMAT_VALUES = 16
DEFAULT_FORMAT_VALUES = 2


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


class StringFormat(object):
    """Format a Python str.format-style template with up to N positional values.

    The template uses {0}, {1}, {2}, ... placeholders (PEP 3101). Unconnected
    values are treated as the empty string so a partially wired node still
    produces a usable result - e.g. a template that references {2} but only
    has value_0/value_1 connected renders as ``"foo + bar + "`` instead of
    crashing with IndexError. The number of value slots grows automatically
    as the user wires more connections: see js/stringFormatAutogrow.js for
    the frontend UX (Bernini Conditioning style).

    Inputs:
    - template: the format string, multiline STRING widget.
    - value_0..value_{N-1}: positional arguments, STRING typed.
      The first DEFAULT_FORMAT_VALUES slots are visible when the node
      is dropped; the JS extension appends one more slot each time the
      last visible slot gets a connection, up to MAX_FORMAT_VALUES.

    Output:
    - result: the formatted string. If str.format raises (mismatched
      braces, unresolvable spec, etc.) the error is logged and the raw
      template is returned so the user can still inspect it on the wire.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(MAX_FORMAT_VALUES):
            # ``forceInput`` keeps the slot as a real connection socket
            # even on an empty widget, which is what the JS extension
            # needs to addInput() the next slot when the previous one is
            # wired. Without it, ComfyUI would hide the slot as a widget.
            #
            # Type is "*" (AnyType wildcard) so the slot can accept any
            # upstream output -- INT (width), BOOLEAN (mode), STRING
            # (hash/prompt), etc. -- and str()-coerce it in format().
            # This avoids needing AnyToString wrapper nodes in the graph.
            optional["value_" + str(i)] = ("*", {"forceInput": True})
        return {
            "required": {
                "template": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "format"
    CATEGORY = MY_CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, template=None, **kwargs):
        # Mirror SaveAny/LoadAny: structurally always valid. The JS
        # extension owns the UX of "how many slots are visible"; the
        # backend just accepts whatever connected inputs it receives.
        return True

    def format(self, template, **kwargs):
        # Collect positional values in slot order. Unconnected slots
        # default to "" so a template referencing a higher placeholder
        # still renders cleanly.
        values = []
        for i in range(MAX_FORMAT_VALUES):
            v = kwargs.get("value_" + str(i))
            if v is None:
                v = ""
            values.append(v)
        tpl = template if isinstance(template, str) else ""
        if not tpl:
            return ("",)
        try:
            return (tpl.format(*values),)
        except (IndexError, KeyError, ValueError, TypeError) as e:
            try:
                from ....core.utils import mie_log
            except Exception:
                from core.utils import mie_log
            mie_log(
                "StringFormat|Mie: format failed: " + str(e) + " (template=" + repr(tpl) + ")"
            )
            return (tpl,)
