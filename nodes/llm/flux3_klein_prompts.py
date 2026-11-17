"""Compatibility shim - the implementation now lives in mie_prompt.

The flux3_klein prompt family (batch 2b). Patch
``mie_prompt.flux3_klein_prompts`` to change the implementation.
"""

from mie_prompt import flux3_klein_prompts as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
