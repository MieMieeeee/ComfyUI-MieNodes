"""Compatibility shim - the implementation now lives in mie_prompt.

The MiniMax-H3-Image prompt family: the four modes, the official-text
loaders, the message builders and the reply normalizer live in
``mie_prompt.h3_image_prompts``. Nothing ComfyUI-specific belongs here --
the family has no cache directory and no ComfyUI dependency -- so this
shim only re-exports.

Patch ``mie_prompt.h3_image_prompts`` to change the implementation.
"""

from mie_prompt import h3_image_prompts as _impl
from mie_prompt.h3_image_prompts import (  # noqa: F401  (re-exported)
    ADAPTIVE,
    ALLOWED_T2I_RATIOS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    GUIDE_MODES,
    IMAGE_MODES,
    JSON_MODES,
    MAX_REFERENCE_IMAGES,
    MODE_CODES,
    MODES,
    Expansion,
    H3ImageError,
    H3ImageReplyError,
    build_messages,
    build_system_prompt,
    build_user_content,
    build_user_message,
    image_ratio,
    is_image_mode,
    load_request_params,
    load_system_prompt,
    load_task_prompt,
    missing_literals,
    normalize_ratio,
    parse_expansion_reply,
    parse_guide_reply,
    parse_mode,
    reference_media_line,
)


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
