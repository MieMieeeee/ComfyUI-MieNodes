from .general import *
from .string_ops import *
from .aspect_ratio import *
from .number_ops import *

# SimpleText and RichText are frontend-only annotation nodes; their Python
# classes are no-op shells registered here for the menu and workflow save/load.
from .text_nodes import (
    SimpleTextNode,
    RichTextNode,
    NODE_CLASS_MAPPINGS as _TEXT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _TEXT_DISPLAY,
)

for _k, _v in _TEXT_MAPPINGS.items():
    globals()[_k] = _v
for _k, _v in _TEXT_DISPLAY.items():
    if _k not in globals():
        globals()[_k] = _v
