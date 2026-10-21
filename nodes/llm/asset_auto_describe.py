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
        SaveAssetV2,
        SaveCharacterAssetV2,
        SaveSceneAssetV2,
        resolve_save_name,
        upload_audio_to_audio,
        upload_image_to_tensor,
    )
except ImportError:
    from ..assets.nodes import (
        SaveAssetV2,
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
    if len(body) <= limit:
        return body
    # Never ship a mid-word stump (2026-10-14 live: 黑白/银白's saved
    # descriptions end "multi-pane windo" / "out-of-focus circul" — an
    # 800-char hard cut). Trim to the last sentence end inside the cap;
    # fall back to the last space.
    window = body[:limit]
    cut = max(window.rfind(". "), window.rfind(".\u2026"), window.rfind("。"))
    if cut < int(limit * 0.5):
        cut = window.rfind(" ")
    return (window[:cut + 1] if cut >= 0 else window).rstrip()


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
    # 800 -> 2000: the captioner's 60-100-word contract is ~500-700
    # chars, but detailed wardrobe runs longer; a hard 800 cut shipped
    # truncated stumps into the library (see _clean).
    description = _clean(reply, 2000)
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


class SaveAssetAutoV2(SaveAssetV2):
    """Unified Save Asset + LLM: kind dropdown; empty fields get drafted.

    Everything else — Existing/New name toggle, upload pickers, media
    priority, overwrite semantics — is inherited unchanged from Save
    Asset. Filled text fields are saved verbatim; the LLM only fills
    blanks. Scene mode skips the voice draft and ignores voice_prompt.
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
                "placeholder": "描述提示词（可选） · Description prompt (optional)",
                "tooltip": "给描述 LLM 的作者先验（可选）：角色 = 气质/着装/必须写进"
                    "的事实；场景 = 地点/时段/氛围/陈设事实；任意语言。空 = 纯看图"
                    "起草。\nAuthor prior for the description call (optional): "
                    "character mood/wardrobe facts or scene place/time/furnishing "
                    "facts — any language. Empty = draft from the picture alone.",
            },
        )
        base["optional"]["voice_prompt"] = (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "placeholder": "音色提示词（可选，仅角色） · Voice prompt (optional, character only)",
                "tooltip": "给音色 LLM 的作者先验（可选，仅角色类型）：它听不到音频，"
                    "真实音色只能在这里告诉它，如「老年男声，沙哑，语速慢」。空 = 从"
                    "名字+外观推断；场景类型忽略本字段。\nAuthor prior for the voice "
                    "call (optional, characters only): the LLM cannot hear the "
                    "sample — ground-truth timbre goes ONLY here, e.g. 'elderly "
                    "male, hoarse, slow pace'. Empty = inferred from name+looks; "
                    "ignored for scenes.",
            },
        )
        return base

    FUNCTION = "save_auto"

    def save_auto(self, llm_service_connector, kind="character",
                  library_dir="", name="", description="", voice="",
                  new_name="", image=None, audio=None, image_upload="",
                  audio_upload="", image_prompt="", voice_prompt="",
                  ambience=""):
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
                llm_service_connector, eff_image, kind, image_prompt
            )
            drafted.append("description")
        if kind != "scene" and not (voice or "").strip():
            voice = draft_voice(
                llm_service_connector,
                resolve_save_name(name, new_name),
                description,
                voice_prompt,
            )
            drafted.append("voice")

        out = super().save(
            kind=kind, library_dir=library_dir, name=name,
            description=description, voice=voice, new_name=new_name,
            image=eff_image, audio=eff_audio,
            ambience=ambience,
        )
        if drafted:
            note = (
                "; LLM drafted %s (sample audio is never listened to; "
                "review and re-save to correct)" % " and ".join(drafted)
            )
            out = out[:4] + (out[4] + note,) + out[5:]
        return out


class SaveCharacterAssetAutoV2(SaveAssetAutoV2):
    """Legacy per-kind Auto node — hidden, loads old workflows."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        return base

    RETURN_TYPES = SaveAssetV2.RETURN_TYPES  # single asset line, same shape
    RETURN_NAMES = ("name", "description", "voice", "image", "status", "asset")
    FUNCTION = "save_auto"

    def save_auto(self, llm_service_connector, library_dir="", name="",
                  description="", voice="", new_name="", image=None,
                  audio=None, image_upload="", audio_upload="",
                  image_prompt="", voice_prompt="", ambience=""):
        return super().save_auto(
            llm_service_connector, "character", library_dir=library_dir,
            name=name, description=description, voice=voice,
            new_name=new_name, image=image, audio=audio,
            image_upload=image_upload, audio_upload=audio_upload,
            image_prompt=image_prompt, voice_prompt=voice_prompt,
            ambience=ambience,
        )


class SaveSceneAssetAutoV2(SaveAssetAutoV2):
    """Legacy per-kind Auto node — hidden, loads old workflows.

    Pops the character-only widgets (voice / audio / audio_upload /
    voice_prompt) so the widget order matches the pre-merge scene node."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        base["required"].pop("voice")
        base["optional"].pop("audio")
        base["optional"].pop("audio_upload")
        base["optional"].pop("voice_prompt")
        return base

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", SaveAssetV2.RETURN_TYPES[5])
    RETURN_NAMES = ("name", "description", "image", "status", "asset")
    FUNCTION = "save_auto"

    def save_auto(self, llm_service_connector, library_dir="", name="",
                  description="", new_name="", image=None, image_upload="",
                  image_prompt="", ambience=""):
        out = super().save_auto(
            llm_service_connector, "scene", library_dir=library_dir,
            name=name, description=description, new_name=new_name,
            image=image, image_upload=image_upload, image_prompt=image_prompt,
            ambience=ambience,
        )
        return (out[0], out[1], out[3], out[4], out[5])
