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
    """Read-only author card (Chinese); rendered as a styled HTML card in
    the frontend.

    Content is sourced from `js/profiles/author.json` and the node's serialized
    `properties.author_*` fields (properties take precedence so the card renders
    correctly for users who don't have the profile file). All fields are
    read-only; the only per-instance state is the theme (Dark/Light/Minimal/
    Leaf), selected via the right-click menu.
    """


class AboutAuthorNodeEn(_MieTextAnnotationBase):
    """Read-only author card (English); same shape as ``AboutAuthorNode`` but
    sourced from ``js/profiles/author_en.json`` so the front-end renders the
    English tagline + link labels.

    The JS dispatch installs the same card-rendering behavior with a
    different profile URL (see ``js/textNodes.js``). Theme picker and
    right-click menu are identical to the Chinese variant.
    """


NODE_CLASS_MAPPINGS = {
    "SimpleTextNode": SimpleTextNode,
    "RichTextNode": RichTextNode,
    "AboutAuthorNode": AboutAuthorNode,
    "AboutAuthorNodeEn": AboutAuthorNodeEn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTextNode": "Simple Text",
    "RichTextNode": "Rich Text",
    "AboutAuthorNode": "About Author",
    "AboutAuthorNodeEn": "About Author EN",
}