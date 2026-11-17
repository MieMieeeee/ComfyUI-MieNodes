"""Compatibility shim - the loader now lives in mie_prompt.

One loader, in the package (batch 2c): the logical-name resolution, the
verbatim read and the "never .format()" rule are defined ONCE, so a
prompt is found the same way whether the caller is a node, a test or a
host app. Patch ``mie_prompt.prompts.loader`` to change it.
"""

from mie_prompt.prompts import loader as _impl
from mie_prompt.prompts.loader import (  # noqa: F401  (re-exported surface)
    list_builtin_prompts,
    list_usable_builtin_prompts,
    load_prompt_dict,
    load_prompt_text,
    reload_prompt,
)


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
