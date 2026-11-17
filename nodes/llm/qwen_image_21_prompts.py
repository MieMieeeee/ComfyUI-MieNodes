"""Compatibility shim - the implementation now lives in mie_prompt.

The Qwen-Image-2.1 prompt-enhancer helpers (batch 2a): the mode map,
block extractors, caption plumbing and the disk-cache helpers live in
``mie_prompt.qwen_image_21_prompts``. The ONE thing defined here is
``caption_cache_disk_root``: the package cannot import ComfyUI, so the
ComfyUI output-directory resolution stays in this shell - the generator
imports the name from here and passes the root to the read/write
helpers explicitly.

Patch ``mie_prompt.qwen_image_21_prompts`` to change the implementation.
"""

from __future__ import annotations

import os

from mie_prompt import qwen_image_21_prompts as _impl
from mie_prompt.qwen_image_21_prompts import (  # noqa: F401  (re-exported)
    CAPTION_MODES,
    MODES,
    _BLOCK_RE,
    _MODE_TO_PROMPT,
    _THINK_BLOCK_RE,
    caption_cache_key,
    caption_image_prompt,
    caption_mode_code,
    extract_enhanced_prompt_block,
    read_caption_cache_disk,
    scrub_caption_sheet_language,
    split_enhancer_reply,
    write_caption_cache_disk,
)


def caption_cache_disk_root() -> str:
    """ComfyUI's output tree, else the repo-relative fallback (the
    resolution the implementation used before the move)."""
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.join(
                os.path.abspath(str(output_dir)),
                "mien_nodes",
                "caption_cache",
            )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "output", "mien_nodes", "caption_cache")


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(dir(_impl)) | set(globals()))
