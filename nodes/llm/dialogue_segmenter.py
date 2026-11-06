"""Compatibility shim - the implementation now lives in mie_craft.script.

The dialogue segmenter and the pacing budget.

Every attribute - including the private helpers the tests reach into -
resolves through the package module, through a module-level ``__getattr__``.
A star import would drop every underscore-prefixed name.

Note for tests: a monkeypatch on THIS module lands on the shim, not on the
implementation, so a test that needs to change the implementation's state
must patch the package module directly.
"""

from mie_craft.script import dialogue as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
