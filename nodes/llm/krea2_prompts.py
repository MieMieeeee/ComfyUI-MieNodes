"""Compatibility shim - the implementation now lives in mie_prompt.

The krea2 prompt family (batch 2b). Patch
``mie_prompt.krea2_prompts`` to change the implementation.
"""

from mie_prompt import krea2_prompts as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
