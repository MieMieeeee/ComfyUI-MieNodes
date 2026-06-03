"""Frontend-only annotation nodes.

`SimpleTextNode` and `RichTextNode` are pure canvas annotations: they have no
inputs, no outputs, and never execute on the backend. The Python class is a
no-op shell that exists only so ComfyUI can list them in the node menu and
serialize/deserialize them in workflow JSON. All visual behavior — drawing,
text editing, markdown rendering — lives in `js/textNodes.js`.
"""


class _MieTextAnnotationBase:
    """Shared no-op config for the SimpleText / RichText annotation nodes."""

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "\U0001F411 MieNodes/\U0001F411 Extra"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def noop(self):
        return {}


class SimpleTextNode(_MieTextAnnotationBase):
    """Plain-text floating annotation; rendered with Canvas in the frontend."""


class RichTextNode(_MieTextAnnotationBase):
    """Markdown floating annotation; rendered as HTML in the frontend."""


NODE_CLASS_MAPPINGS = {
    "SimpleTextNode": SimpleTextNode,
    "RichTextNode": RichTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTextNode": "Simple Text",
    "RichTextNode": "Rich Text",
}