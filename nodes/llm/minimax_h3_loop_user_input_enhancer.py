"""Compatibility shim + ComfyUI shell - the text domain lives in mie_craft.

The implementation (batch 1 of PLAN_LLM_PROMPT_MIGRATION_TO_CORE) moved to
``mie_craft.loop_enhancer``: draft classification, speech extraction,
lock/compose verification, scene-count advice and ``run_enhancer`` with its
transport/caching. What stays HERE is the ComfyUI node: ``INPUT_TYPES``,
the |Mie category, and the cache-root that resolves against ComfyUI's own
output directory (down in the package, ComfyUI is not importable, so the
shell passes ``cache_root`` explicitly).

Every attribute - including the private helpers the tests reach into -
resolves through the package module, through a module-level ``__getattr__``.
Note for tests: monkeypatch THIS module and you patch the shim; patch
``mie_craft.loop_enhancer`` to change the implementation.
"""

from __future__ import annotations

import hashlib
import os

from mie_craft.h3_loop_prompts import REFERENCE_MODES  # noqa: F401
from mie_craft.loop_enhancer import (  # noqa: F401  (re-exported surface)
    _CACHE_ROOT_OVERRIDE,
    _DEFAULT_TEMPERATURE,
    _DEFAULT_TIMEOUT,
    _LOCK_TEMPERATURE,
    _MAX_MAX_TOKENS,
    _MAX_TOKENS_DEFAULT,
    _MIN_MAX_TOKENS,
    _PROMPT_COMPOSE,
    _PROMPT_LOCK,
    _PROMPT_MODES,
    _PROMPT_NAME,
    _TRANSPORT_ATTEMPTS,
    _UserInputEnhancer,
    _assemble_system_prompt,
    _build_advice,
    _cache_get,
    _cache_key,
    _cache_put,
    _connector_cache_identity,
    _enhancer_cache_root,
    _mode_contract,
    _parse_classification,
    clear_enhancer_cache,
    collect_draft_speech,
    draft_dialogue_lines,
    draft_dialogue_turns,
    draft_is_locked,
    extract_user_input_block,
    extra_dialogue_lines,
    lock_source_turns,
    missing_dialogue_lines,
    pin_dialogue_lines,
    run_enhancer,
    split_enhancer_reply,
    suggest_scene_count,
    verify_block_content,
)
try:
    from _mienodes_internal.nodes.llm.minimax_h3_loop_prompt_generator import (
        LOOP_CATEGORIES,
        _PACING_LABELS,
    )
except ImportError:
    from .minimax_h3_loop_prompt_generator import (
        LOOP_CATEGORIES,
        _PACING_LABELS,
    )

_impl = None


def __getattr__(name):
    global _impl
    if _impl is None:
        import mie_craft.loop_enhancer as impl
        _impl = impl
    return getattr(_impl, name)


def __dir__():
    import mie_craft.loop_enhancer as impl
    return sorted(set(dir(impl)) | set(globals()))


def _comfyui_cache_root() -> str:
    """The shell's cache root: ComfyUI's output tree when there is one,
    the old repo-relative path otherwise (same resolution the
    implementation used before the move)."""
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.join(
                os.path.abspath(str(output_dir)),
                "mien_nodes",
                "h3_loop_enhancer_cache",
            )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "output", "mien_nodes", "h3_loop_enhancer_cache")


MY_CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class MiniMaxH3LoopUserInputEnhancer:
    """ComfyUI node: rough draft -> node-ready ``user_input`` STRING.

    Sits next to ``MiniMaxH3LoopPromptGenerator`` in the same Prompt
    Generator category. Wire its ``user_input`` output socket straight
    into the loop node's ``user_input`` widget — the names match so
    no rewiring is needed.

    Raises ``RuntimeError`` if the LLM reply does not contain a
    ``--- BEGIN user_input ---`` ... ``--- END user_input ---`` block;
    the error includes the raw reply head so the user can diagnose.
    Silent fallback would let a mis-classified reply flow into H3 loop
    and waste the whole pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "draft": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Your rough idea in any shape — a sentence, "
                            "an outline, a few lines, a free paragraph. "
                            "The preprocessor rewrites it into the format "
                            "the H3 Loop Plan Generator node consumes."
                        ),
                    },
                ),
                "category": (
                    list(LOOP_CATEGORIES),
                    {
                        "default": LOOP_CATEGORIES[0],
                        "tooltip": (
                            "Mirrors the H3 Loop node's category widget. "
                            "Sets the cinematography advice the preprocessor "
                            "will respect (none / dialogue / action)."
                        ),
                    },
                ),
                "reference_mode": (
                    list(REFERENCE_MODES),
                    {
                        "default": REFERENCE_MODES[0],
                        "tooltip": (
                            "Mirrors the H3 Loop node's reference_mode widget. "
                            "CRITICAL: if you have reference images, set this "
                            "to fl2va or ref2va here AND on the loop node — "
                            "t2va ignores the images socket."
                        ),
                    },
                ),
                "pacing": (
                    list(_PACING_LABELS),
                    {
                        "default": _PACING_LABELS[1],
                        "tooltip": (
                            "Mirrors the H3 Loop node's pacing widget. "
                            "Existing speaker：line utterances are never "
                            "rewritten to hit a tempo; pacing only changes "
                            "how long the setting paragraph is. On a "
                            "free-form draft, fast packs more, shorter "
                            "beats and slow uses fewer, longer ones."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Seed forwarded to the LLM call. 0 lets the "
                            "connector pick a fresh seed."
                        ),
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _MAX_TOKENS_DEFAULT,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                    },
                ),
                "timeout": (
                    [60, 120, 300, 600],
                    {"default": _DEFAULT_TIMEOUT},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_input", "advice")
    FUNCTION = "enhance"
    OUTPUT_NODE = True
    CATEGORY = MY_CATEGORY

    def enhance(
        self,
        llm_service_connector,
        draft,
        category,
        reference_mode,
        pacing=_PACING_LABELS[1],
        seed=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        block, advice = run_enhancer(
            llm_service_connector,
            draft,
            category=category,
            reference_mode=reference_mode,
            pacing=pacing,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            cache_root=_comfyui_cache_root(),
        )
        shown = advice or ""
        if block:
            shown = f"{shown}\n\n--- user_input ---\n{block}"
        return {"ui": {"text": shown}, "result": (block, advice)}

    def is_changed(
        self,
        llm_service_connector,
        draft,
        category,
        reference_mode,
        pacing=_PACING_LABELS[1],
        seed=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            draft or "",
            category or "",
            reference_mode or "",
            pacing or "",
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update(part.encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
