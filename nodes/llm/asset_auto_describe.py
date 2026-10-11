# -*- coding: utf-8 -*-
"""Save-based LLM-assisted asset nodes (2026-10-10, merged 2026-10-11).

The Auto nodes ARE the Save nodes plus one ``llm_service_connector`` input
and optional author-hint prompts — wire them exactly like Save Character /
Save Scene Asset, leave ``description`` (and ``voice``) EMPTY, and the LLM
drafts those fields during the same queue run that saves:

* description — the SAME multimodal captioner the H3 prompt generator
  uses for reference images (60-100 word English identity description);
* voice — a text call from the name + the drafted appearance. The LLM
  cannot hear the voice sample (wav/mp3 make no difference — the file is
  never listened to); the ``voice_prompt`` hint is the ONLY channel for
  ground-truth timbre (the author has heard the sample; the LLM has not).

Fields the user FILLED are saved verbatim — the LLM only fills blanks,
it never rewrites authored text. Media priority is unchanged from the
Save nodes: a wired socket beats the upload picker; both empty on an
overwrite keeps the disk media.
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
        SaveCharacterAssetV2,
        SaveSceneAssetV2,
        resolve_save_name,
        upload_audio_to_audio,
        upload_image_to_tensor,
    )
except ImportError:
    from ..assets.nodes import (
        SaveCharacterAssetV2,
        SaveSceneAssetV2,
        resolve_save_name,
        upload_audio_to_audio,
        upload_image_to_tensor,
    )

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


# ---- shared draft helpers -------------------------------------------------
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


def _clean(text, limit):
    body = " ".join(str(text or "").split())
    return body[:limit]


def _reply_text(reply):
    if isinstance(reply, dict):
        reply = json.dumps(reply, ensure_ascii=False)
    return reply


def _check_connector(llm_service_connector):
    if not hasattr(llm_service_connector, "invoke"):
        raise RuntimeError(
            "the Auto node requires an LLM connector with invoke(); "
            "connect a full LLMServiceConnector"
        )


def draft_description(connector, tensor, kind, hint=""):
    """One multimodal call: the captioner for characters, a place hint for
    scenes. Author hints ride in the USER turn; the system prompt keeps
    its style contract."""
    import torch as _torch

    tensor = _torch.as_tensor(tensor)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim != 4 or tensor.shape[0] < 1:
        raise RuntimeError("auto-draft needs a non-empty IMAGE")
    urls = image_tensor_batch_to_data_urls(tensor[:1])
    if not urls:
        raise RuntimeError("failed to encode the portrait to a data URL")

    if kind == "scene":
        system, user_text = (
            _SCENE_DESCRIPTION_HINT,
            "Describe this place for a scene reference image.",
        )
    else:
        system, user_text = (
            caption_reference_prompt(),
            "Describe this character for the identity reference image.",
        )
    hint = (hint or "").strip()
    if hint:
        user_text += (
            "\n\nAuthor notes (facts you must respect; keep the output "
            "style the system prompt defines): " + hint
        )
    reply = _reply_text(connector.invoke(_messages(system, user_text, urls),
                                         max_tokens=16384))
    description = _clean(reply, 800)
    if not description:
        raise RuntimeError("the LLM returned an empty description draft")
    return description


def draft_voice(connector, name, description, hint=""):
    """One text call. The hint is the only ground-truth timbre channel."""
    user_text = "Name: %s\nAppearance: %s" % (name, description[:400])
    hint = (hint or "").strip()
    if hint:
        user_text += (
            "\nAuthor voice notes (ground truth from the author, who "
            "has heard the sample; follow them over your inference): "
            + hint
        )
    user_text += "\nDraft the one-line voice descriptor."
    reply = _reply_text(
        connector.invoke(
            [{"role": "system", "content": _VOICE_SYSTEM},
             {"role": "user", "content": user_text}],
            max_tokens=512,
        )
    )
    voice = _clean(reply, 200)
    if not voice:
        raise RuntimeError("the LLM returned an empty voice draft")
    return voice


class SaveCharacterAssetAutoV2(SaveCharacterAssetV2):
    """Save Character Asset + LLM: empty description/voice get drafted.

    Everything else — name/new_name flow, upload pickers, media priority,
    overwrite semantics — is inherited unchanged from Save Character
    Asset. Filled text fields are saved verbatim; the LLM only fills
    blanks.
    """

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"]["llm_service_connector"] = (
            "LLMServiceConnector",
            {"tooltip": "起草空白 description/voice 的 LLM 连接器。LLM connector that drafts the empty fields."},
        )
        base["optional"]["image_prompt"] = (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "placeholder": "图像描述提示词（可选）· image prompt (optional)",
                "tooltip": "给描述 LLM 的作者先验（可选）：气质/着装细节、"
                "画风、必须写进描述的事实等，任意语言。空 = 纯看图起草。\n"
                "Author prior for the description call (optional): mood, wardrobe "
                "details, facts that must appear — any language. Empty = draft "
                "from the picture alone.",
            },
        )
        base["optional"]["voice_prompt"] = (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "placeholder": "音色提示词（可选）· voice prompt (optional)",
                "tooltip": "给音色 LLM 的作者先验（可选）：它听不到音频，"
                "真实音色只能在这里告诉它，如「老年男声，沙哑，语速慢」。"
                "空 = 从名字+外观推断。\n"
                "Author prior for the voice call (optional): the LLM cannot hear "
                "the sample — ground-truth timbre goes ONLY here, e.g. 'elderly "
                "male, hoarse, slow pace'. Empty = inferred from name+looks.",
            },
        )
        return base

    FUNCTION = "save_auto"

    def save_auto(self, llm_service_connector, library_dir="", name="",
                  description="", voice="", new_name="", image=None,
                  audio=None, image_upload="", audio_upload="",
                  image_prompt="", voice_prompt=""):
        _check_connector(llm_service_connector)
        # Media priority mirrors the Save nodes exactly (socket > upload);
        # resolving once here also feeds the captioner when drafting.
        eff_image = image
        if eff_image is None and (image_upload or "").strip():
            eff_image = upload_image_to_tensor((image_upload or "").strip())
        eff_audio = audio
        if eff_audio is None and (audio_upload or "").strip():
            eff_audio = upload_audio_to_audio((audio_upload or "").strip())

        drafted = []
        if not (description or "").strip():
            if eff_image is None:
                raise RuntimeError(
                    "auto-draft needs a picture to write the description: "
                    "wire an image (or pick one in the media card), or "
                    "type the description yourself"
                )
            description = draft_description(
                llm_service_connector, eff_image, "character", image_prompt
            )
            drafted.append("description")
        if not (voice or "").strip():
            voice = draft_voice(
                llm_service_connector,
                resolve_save_name(name, new_name),
                description,
                voice_prompt,
            )
            drafted.append("voice")

        out = super().save_character(
            library_dir=library_dir, name=name, description=description,
            voice=voice, new_name=new_name, image=eff_image,
            audio=eff_audio,
        )
        if drafted:
            note = (
                "; LLM drafted %s (sample audio is never listened to; "
                "review and re-save to correct)" % " and ".join(drafted)
            )
            out = out[:4] + (out[4] + note,) + out[5:]
        return out


class SaveSceneAssetAutoV2(SaveSceneAssetV2):
    """Save Scene Asset + LLM: an empty description gets drafted."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"]["llm_service_connector"] = (
            "LLMServiceConnector",
            {"tooltip": "起草空白 description 的 LLM 连接器。LLM connector that drafts the empty description."},
        )
        base["optional"]["image_prompt"] = (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "placeholder": "场景描述提示词（可选）· scene prompt (optional)",
                "tooltip": "给描述 LLM 的作者先验（可选）：地点/时段/氛围、"
                "必须写进描述的陈设事实等，任意语言。空 = 纯看图起草。\n"
                "Author prior for the description call (optional): place, time of "
                "day, mood, furnishing facts that must appear — any language. "
                "Empty = draft from the picture alone.",
            },
        )
        return base

    FUNCTION = "save_auto"

    def save_auto(self, llm_service_connector, library_dir="", name="",
                  description="", new_name="", image=None, image_upload="",
                  image_prompt=""):
        _check_connector(llm_service_connector)
        eff_image = image
        if eff_image is None and (image_upload or "").strip():
            eff_image = upload_image_to_tensor((image_upload or "").strip())

        drafted = []
        if not (description or "").strip():
            if eff_image is None:
                raise RuntimeError(
                    "auto-draft needs a picture to write the description: "
                    "wire an image (or pick one in the media card), or "
                    "type the description yourself"
                )
            description = draft_description(
                llm_service_connector, eff_image, "scene", image_prompt
            )
            drafted.append("description")

        out = super().save_scene(
            library_dir=library_dir, name=name, description=description,
            new_name=new_name, image=eff_image,
        )
        if drafted:
            note = (
                "; LLM drafted description (review and re-save to correct)"
            )
            out = out[:3] + (out[3] + note,) + out[4:]
        return out
