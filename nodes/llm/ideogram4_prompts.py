"""Compatibility shim - the implementation now lives in mie_prompt.

The Ideogram 4 prompt family (batch 2a). Patch
``mie_prompt.ideogram4_prompts`` to change the implementation.
"""

from mie_prompt import ideogram4_prompts as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
