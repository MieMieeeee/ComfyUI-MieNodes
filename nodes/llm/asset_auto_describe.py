# -*- coding: utf-8 -*-
"""LLM-assisted description filler for the asset library (2026-10-10).

The description (image description) and voice (audio description) fields
are authored English text the user fills by hand — that contract is
unchanged. This node DRAFTS both fields from the LLM so a save can start
from something better than a blank box:

* description — the SAME multimodal captioner the H3 prompt generator
  uses for reference images (60-100 word English identity description,
  the exact style the library already documents);
* voice — a text call from the name + the drafted appearance. The LLM
  cannot hear the voice sample (wav/mp3 make no difference — the file is
  never listened to); the draft infers from name/appearance, and the
  user edits after hearing their sample.

It never writes to the library: outputs are plain strings to wire into
Save Character/Scene Asset's description/voice inputs (convert the widget
to an input, or preview with any text-display node).
"""

from __future__ import annotations

import json

try:
    from _mienodes_internal.core.utils import image_tensor_batch_to_data_urls
except ImportError:
    from ...core.utils import image_tensor_batch_to_data_urls

try:
    from _mienodes_internal.nodes.llm.h3_prompts import caption_reference_prompt
except ImportError:
    from .h3_prompts import caption_reference_prompt

try:
    from _mienodes_internal.nodes.assets.nodes import (
        CHARACTER_ASSET_TYPE,
        SCENE_ASSET_TYPE,
    )
except ImportError:
    from ..assets.nodes import CHARACTER_ASSET_TYPE, SCENE_ASSET_TYPE

ASSETS_CATEGORY = "🐑 MieNodes/🐑 Assets"

_VOICE_SYSTEM = (
    "You write one-line English voice (timbre) descriptors for video "
    "characters, in the exact style: 'adult female, clear and slightly "
    "warm timbre, low-mid register, unhurried pace'. Given a character "
    "name and their appearance description, infer the most plausible "
    "voice (age, gender, register, pace, texture). You CANNOT hear any "
    "audio — this is a draft from the name and appearance only, and the "
    "user will edit it after hearing their sample. Output ONLY the one "
    "descriptor sentence, at most 200 characters, no quotes, no "
    "explanation."
)

_SCENE_DESCRIPTION_HINT = (
    "Describe the PLACE in one paragraph (60-100 words, English): the "
    "space, its key furniture/objects, the lighting, and the atmosphere. "
    "State spatial facts an image reference must preserve (what stands "
    "where). Output only the description, no preamble."
)


class AssetAutoDescribe(object):
    """Draft description (+ voice for characters) from a portrait via the LLM.

    Wire a Load Character/Scene Asset's ``asset`` output in (or a raw
    ``image`` + ``name`` for a brand-new asset), plus an
    ``llm_service_connector``. Out come ``name`` / ``description`` /
    ``voice`` / ``status`` — wire ``description`` (and ``voice``) into the
    Save node's corresponding inputs, or read them in a preview node. The
    node never touches the library.
    """

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
            },
            "optional": {
                "asset": (
                    # Combo-of-types inputs validate as membership; two
                    # separate sockets accept each bundle type cleanly.
                    CHARACTER_ASSET_TYPE,
                    {"tooltip": "Load Character Asset 的 asset 输出。"},
                ),
                "scene_asset": (
                    SCENE_ASSET_TYPE,
                    {"tooltip": "Load Scene Asset 的 asset 输出（接了以场景描述起草，无音色）。"},
                ),
                "image": ("IMAGE", {"tooltip": "裸图（新建资产且未打包时用）。"}),
                "name": ("STRING", {"default": "", "tooltip": "角色/场景名（裸图路径时填）。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("name", "description", "voice", "status")
    FUNCTION = "describe"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # One caption + one text call; re-run on every queue so an edit to
        # the portrait or a re-pick of the model always re-drafts.
        return float("nan")

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _messages(system, user_text, image_urls=None):
        content = []
        for url in image_urls or []:
            content.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
            )
        content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _clean(text, limit):
        body = " ".join(str(text or "").split())
        return body[:limit]

    # ---- execution ----------------------------------------------------------
    def describe(self, llm_service_connector, asset=None, scene_asset=None,
                 image=None, name=""):
        if not hasattr(llm_service_connector, "invoke"):
            raise RuntimeError(
                "auto-describe requires an LLM connector with invoke(); "
                "connect a full LLMServiceConnector"
            )
        kind = "character"
        if isinstance(scene_asset, dict):
            asset = scene_asset
            kind = "scene"
        if isinstance(asset, dict):
            if kind == "character":
                kind = str(asset.get("kind") or "character")
            if asset.get("image") is not None:
                image = asset["image"]
            if not (name or "").strip():
                name = str(asset.get("name") or "")
        name = (name or "").strip() or "Unnamed"

        if image is None:
            raise RuntimeError(
                "auto-describe needs a picture: wire an asset bundle or an "
                "image"
            )

        import torch as _torch

        tensor = _torch.as_tensor(image)
        if tensor.ndim == 3:
            tensor = tensor[None, ...]
        if tensor.ndim != 4 or tensor.shape[0] < 1:
            raise RuntimeError("auto-describe needs a non-empty IMAGE")
        urls = image_tensor_batch_to_data_urls(tensor[:1])
        if not urls:
            raise RuntimeError("failed to encode the portrait to a data URL")

        # 1) Appearance / scene description — the generator's own captioner.
        caption_system = caption_reference_prompt()
        if kind == "scene":
            caption_messages = self._messages(
                _SCENE_DESCRIPTION_HINT,
                "Describe this place for a scene reference image.",
                urls,
            )
        else:
            caption_messages = self._messages(
                caption_system,
                "Describe this character for the identity reference image.",
                urls,
            )
        reply = llm_service_connector.invoke(
            caption_messages, max_tokens=16384
        )
        if isinstance(reply, dict):
            reply = json.dumps(reply, ensure_ascii=False)
        description = self._clean(reply, 800)
        if not description:
            raise RuntimeError("the LLM returned an empty description draft")

        # 2) Voice descriptor — characters only; the LLM cannot hear the
        #    sample (wav or mp3, the file is never listened to).
        voice = ""
        if kind == "character":
            voice_messages = [
                {"role": "system", "content": _VOICE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Name: %s\nAppearance: %s\nDraft the one-line voice "
                        "descriptor." % (name, description[:400])
                    ),
                },
            ]
            reply = llm_service_connector.invoke(voice_messages, max_tokens=512)
            if isinstance(reply, dict):
                reply = json.dumps(reply, ensure_ascii=False)
            voice = self._clean(reply, 200)

        status = (
            "drafted %s description (%d chars)%s — review before saving; the "
            "voice draft is inferred from name+appearance (the sample audio "
            "is never listened to)"
            % (
                kind,
                len(description),
                "; voice drafted (%d chars)" % len(voice) if voice else "",
            )
        )
        return (name, description, voice, status)
