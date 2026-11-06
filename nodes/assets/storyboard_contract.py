"""Compatibility shim - the implementation now lives in ComfyUI-Mie-Core.

The script contract: the shape of a script asset.

Every attribute - including the private helpers the tests reach into -
resolves through the core module, through a module-level ``__getattr__``.
A star import would drop every underscore-prefixed name.

Note for tests: a monkeypatch on THIS module lands on the shim, not on the
implementation, so a test that needs to change the implementation's state
must patch the core module directly.
"""

from mie_core.script import contract as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
