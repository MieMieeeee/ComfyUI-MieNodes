"""Compatibility shim - the implementation now lives in mie_prompt.

The moss-image prompt helpers (batch 2c): mode parsing, the layout
reply parser and the box arithmetic - connector-free and torch-free,
so hosts can use them directly. Patch ``mie_prompt.moss_image_prompts``
to change the implementation.
"""

from mie_prompt import moss_image_prompts as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
