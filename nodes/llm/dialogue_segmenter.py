"""Compatibility shim - the implementation now lives in ComfyUI-Mie-Core.

The dialogue segmenter and the pacing budget.
"""

from mie_core.script import dialogue as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
