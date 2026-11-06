"""Compatibility shim - the implementation now lives in ComfyUI-Mie-Core.

The H3 loop prompt and plan builder.
"""

from mie_core.llm import h3_loop_prompts as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
