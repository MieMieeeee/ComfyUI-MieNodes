"""Frontend-only annotation nodes.

`SimpleTextNode`, `RichTextNode`, and `AboutAuthorNode` are pure canvas
annotations: they have no inputs, no outputs, and never execute on the
backend. The Python class is a no-op shell that exists only so ComfyUI can
list them in the node menu and serialize/deserialize them in workflow JSON.
All visual behavior — drawing, text editing, markdown rendering, and the
About Author card — lives in `js/textNodes.js`.
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


class AboutAuthorNode(_MieTextAnnotationBase):
    """Read-only author card; rendered as a styled HTML card in the frontend.

    Content is sourced from `js/profiles/author.json` and the node's serialized
    `properties.author_*` fields (properties take precedence so the card renders
    correctly for users who don't have the profile file). All fields are
    read-only; the only per-instance state is the theme (Dark/Light/Minimal/
    Banner), selected via the right-click menu.
    """


NODE_CLASS_MAPPINGS = {
    "SimpleTextNode": SimpleTextNode,
    "RichTextNode": RichTextNode,
    "AboutAuthorNode": AboutAuthorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTextNode": "Simple Text",
    "RichTextNode": "Rich Text",
    "AboutAuthorNode": "About Author",
}