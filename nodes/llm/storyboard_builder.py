"""The storyboard builder NODES.

The builder itself - the zero-LLM assembler that turns a script asset into a
plan - now lives in ComfyUI-Mie-Core, so the H3 engine can use exactly the
same assembler. What stays here is the ComfyUI shell around it.

The asset roster socket type is defined in the core as well, so a plan
produced by this pack and a plan produced by the engine are the same shape.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from mie_core.script.builder import *  # noqa: F401,F403
from mie_core.script.builder import (
    _storyboard_optional_slots,
    _storyboard_required_inputs,
)

class BuildPlanFromStoryboard(object):
    """Zero-LLM plan assembler (docs/PLAN_STORYBOARD_ASSET.md §4.2).

    Same inputs/outputs as ``MiniMaxH3LoopPromptGeneratorV2`` minus the
    LLM connector + every enhance widget: a hand-written storyboard
    asset + wired character/scene asset bundles → byte-stable
    ``plan_json``. Sits between ``LoadStoryboardAsset`` and the existing
    ``MiniMaxH3ChainPlanModern``; the wiring is identical to V2 except
    no LLM socket is required.

    The ``assets`` output is the same ``MIE_ASSET_ROSTER`` shape V2
    emits (characters in slot order, then scenes) — ``Asset → Slots``
    splits it into ``picture_i`` / ``audio_i`` for the H3 chain.
    """

    MAX_SLOTS = MAX_CHARACTER_SLOTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _storyboard_required_inputs(),
            "optional": _storyboard_optional_slots(cls.MAX_SLOTS),
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", ASSET_ROSTER_TYPE)
    RETURN_NAMES = ("plan_json", "summary", "images", "assets")

    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, storyboard, **kwargs):
        out = assemble_from_sockets(
            storyboard=storyboard,
            kwargs=kwargs,
            max_slots=self.MAX_SLOTS,
        )
        return (
            out["plan_json"],
            out["summary"],
            out["images"],
            out["assets"],
        )


class MiniMaxH3LoopPromptGeneratorV3(object):
    """V3 generator facade over the zero-LLM storyboard track
    (docs/PLAN_STORYBOARD_FIRST_ASSET_V3.md §4).

    Shape-compatible with ``MiniMaxH3LoopPromptGeneratorV2``: same five
    outputs in the same order and types (``plan_json``, ``summary``,
    ``board_kind``, ``images``, ``assets``), same CATEGORY, same
    self-growing ``character_N`` sockets. The difference is that V3 has
    NO LLM connector and no prose — the storyboard asset IS the script,
    so swapping a V2 node for a V3 node (rewiring only the ``storyboard``
    line) moves a workflow onto the storyboard track without touching
    anything downstream of ``plan_json``.

    ``board_kind`` is the stable ``"dialogue"`` literal (storyboard v1 is
    dialogue-only). It exists so downstream wiring written against V2's
    five-output shape keeps working unchanged.
    """

    MAX_SLOTS = MAX_CHARACTER_SLOTS
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _storyboard_required_inputs(),
            "optional": _storyboard_optional_slots(cls.MAX_SLOTS),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", ASSET_ROSTER_TYPE)
    RETURN_NAMES = ("plan_json", "summary", "board_kind", "images", "assets")

    FUNCTION = "generate_v3"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Storyboard content hash + every wired asset bundle's content.

        A Load-storyboard node already hashes its file, so re-queueing
        after an edit re-runs that side; this side additionally covers a
        user re-picking a DIFFERENT asset into an already-wired slot
        (the asset's own Load node hashes its files too, but the roster
        it feeds is a bundle, so hashing the bundle here keeps the
        generator honest when someone rewires from the canvas).
        """
        # seed and continuation_overlap_frames are DIGESTED TOO (2026-11-01):
        # both change the compiled plan (per-shot seeds / re-budgeted
        # durations) without touching any input link — leaving them out made
        # a seed-only re-roll silently reuse the cached plan, so the re-roll
        # lever looked dead until a restart cleared the cache.
        storyboard = kwargs.get("storyboard")
        parts = [storyboard_cache_key(storyboard)]
        for i in range(1, cls.MAX_SLOTS + 1):
            parts.append(_bundle_fingerprint(kwargs.get("character_%d" % i)))
        parts.append(_bundle_fingerprint(kwargs.get("scene_1")))
        return "|".join(parts)

    def generate_v3(self, storyboard, **kwargs):
        out = assemble_from_sockets(
            storyboard=storyboard,
            kwargs=kwargs,
            max_slots=self.MAX_SLOTS,
        )
        return (
            out["plan_json"],
            out["summary"],
            BOARD_KIND_DIALOGUE,
            out["images"],
            out["assets"],
        )


def _bundle_fingerprint(bundle) -> str:
    """Stable identity of one wired asset bundle for IS_CHANGED.

    The text fields (name / description / voice / ambience) decide what
    lands in the plan; the tensors are deliberately excluded — a re-render
    of the same portrait changes pixels without changing a single word of
    ``plan_json``, and busting the cache on pixels would re-run the
    assembler for nothing.
    """
    if bundle is None:
        return "-"
    if not isinstance(bundle, dict):
        return _library.MISSING_CACHE_MARKER
    return "\x1f".join([
        str(bundle.get("kind") or ""),
        str(bundle.get("name") or ""),
        str(bundle.get("description") or ""),
        str(bundle.get("voice") or ""),
        str(bundle.get("ambience") or ""),
        "audio" if bundle.get("audio") is not None else "-",
    ])


def _bundle_fingerprint(bundle) -> str:
    """Stable identity of one wired asset bundle for IS_CHANGED.

    The text fields (name / description / voice / ambience) decide what
    lands in the plan; the tensors are deliberately excluded — a re-render
    of the same portrait changes pixels without changing a single word of
    ``plan_json``, and busting the cache on pixels would re-run the
    assembler for nothing.
    """
    if bundle is None:
        return "-"
    if not isinstance(bundle, dict):
        return _library.MISSING_CACHE_MARKER
    return "\x1f".join([
        str(bundle.get("kind") or ""),
        str(bundle.get("name") or ""),
        str(bundle.get("description") or ""),
        str(bundle.get("voice") or ""),
        str(bundle.get("ambience") or ""),
        "audio" if bundle.get("audio") is not None else "-",
    ])
