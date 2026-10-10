# -*- coding: utf-8 -*-
"""Six asset-library nodes (docs/PLAN_ASSET_LIBRARY.md section 4).

nodes.py owns tensor <-> PNG conversion; everything on disk goes through
nodes/assets/library.py. No ComfyUI imports, so tests can load this module
without a running ComfyUI.
"""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import torch
from PIL import Image

try:
    from _mienodes_internal.nodes.assets import library
except ImportError:
    from . import library

ASSETS_CATEGORY = "🐑 MieNodes/🐑 Assets"

# Combo placeholders + save-mode sentinel. The real option lists are filled
# per node INSTANCE by js/assetLibraryUI.js from POST /mie_nodes/assets/list;
# the class-level list is only what an un-connected node shows. Combos whose
# real value came from there (or from an older workflow's typed name) would
# fail the "value in class options" check, so every node below carries a
# VALIDATE_INPUTS that lists the name inputs — execution.py:1019 skips the
# combo check for inputs named in that signature, and the executor's own
# not-found error is the message users actually need.
NAME_PLACEHOLDER = ["（下拉选择资产）"]
NEW_NAME_SENTINEL = "（✏️ 新名字…）"


def _validate_exempt(name, **_kwargs):
    """Exempt the combo-fed name inputs from class-level membership checks.

    Pure exemption: returns True (or None) no matter what — a wrong name is
    reported by the load/save executor with the asset's kind and the
    absolute path, which beats a generic "value not in list" here.
    """
    return True


def resolve_save_name(name, new_name):
    """The save dropdown lists existing names plus a new-name sentinel.

    Picking the sentinel means "I am creating a new asset" — the real name
    comes from the new_name text field the frontend reveals for that
    choice. Picking an existing name is an explicit overwrite intent.
    """
    name = (name or "").strip()
    new_name = (new_name or "").strip()
    if name == NEW_NAME_SENTINEL or (NEW_NAME_SENTINEL in (name or "")):
        if not new_name:
            raise ValueError(
                "name is %r but new_name is empty: pick an existing name from "
                "the dropdown to overwrite, or type the new asset's name in "
                "new_name" % NEW_NAME_SENTINEL
            )
        return new_name
    return name


def image_batch_to_png_bytes_list(images, max_frames=library.MAX_IMAGES):
    """float32 RGB [N,H,W,3] in 0..1 -> list of per-frame PNG bytes.

    One IMAGE socket cannot hold two resolutions; a tensor batch is
    same-size by construction, so the storage layer's same-size rule is
    satisfied here for free.
    """
    tensor = torch.as_tensor(images)
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(
            "expected an IMAGE tensor [N,H,W,3], got shape %s" % (tuple(tensor.shape),)
        )
    if tensor.shape[0] < 1:
        raise ValueError("the IMAGE batch is empty; an asset needs at least one image")
    if tensor.shape[0] > max_frames:
        raise ValueError(
            "%d images given; an asset supports at most %d" % (tensor.shape[0], max_frames)
        )
    out = []
    for i in range(int(tensor.shape[0])):
        arr = tensor[i].detach().cpu().numpy()
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def png_bytes_list_to_image_tensor(png_list):
    """List of PNG bytes -> float32 RGB [N,H,W,3] in 0..1."""
    frames = []
    for blob in png_list:
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        frames.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0))
    return torch.stack(frames)


def image_tensor_to_png_bytes(images):
    """float32 RGB [B,H,W,3] in 0..1 -> PNG bytes. B must be exactly 1."""
    tensor = torch.as_tensor(images)
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(
            "expected an IMAGE tensor [B,H,W,3], got shape %s" % (tuple(tensor.shape),)
        )
    if tensor.shape[0] != 1:
        raise ValueError(
            "the portrait must be exactly one frame, got a batch of %d; pick one "
            "frame first" % tensor.shape[0]
        )
    arr = tensor[0].detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_image_tensor(png_bytes):
    """PNG bytes -> float32 RGB [1,H,W,3] in 0..1 (the ComfyUI IMAGE convention)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()).unsqueeze(0)


def audio_to_wav_bytes(audio):
    """ComfyUI AUDIO dict -> 16-bit PCM WAV bytes (stdlib only).

    AUDIO convention in this repo: {"waveform": [1, C, N] float in [-1, 1],
    "sample_rate": int} — the shape QwenTTSNode produces and WavConcat consumes.
    """
    if (
        not isinstance(audio, dict)
        or "waveform" not in audio
        or "sample_rate" not in audio
    ):
        raise ValueError(
            "voice_sample must be a ComfyUI AUDIO connection "
            "(dict with 'waveform' and 'sample_rate')"
        )
    tensor = torch.as_tensor(audio["waveform"]).detach().cpu().float()
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] < 1:
        raise ValueError(
            "voice_sample waveform must be [1, channels, frames] from exactly one "
            "clip, got shape %s" % (tuple(tensor.shape),)
        )
    if tensor.shape[2] < 1:
        raise ValueError("voice_sample is empty; a character needs a non-empty sample")
    channels, frames = tensor.shape[1], tensor.shape[2]
    pcm = torch.clamp(tensor[0], -1.0, 1.0)
    pcm = (pcm * 32767.0).round().to(torch.int16).numpy().T  # [C,N] -> [N,C] interleaved
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(audio["sample_rate"]))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_audio(data):
    """16-bit PCM WAV bytes -> ComfyUI AUDIO dict (stdlib only)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        if w.getsampwidth() != 2:
            raise ValueError(
                "only 16-bit PCM voice samples are supported, got %d-bit"
                % (w.getsampwidth() * 8)
            )
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    arr = arr.reshape(-1, channels).T  # [N,C] interleaved -> [C,N]
    return {
        "waveform": torch.from_numpy(arr.copy()).unsqueeze(0),
        "sample_rate": sample_rate,
    }


def voice_sample_bytes_to_audio(data, ext):
    """Voice sample bytes from the library -> ComfyUI AUDIO dict.

    .wav decodes with the stdlib; .mp3 (hand-dropped) goes through soundfile,
    the same loader QwenTTSNode uses, imported lazily so wav-only setups never
    need it.
    """
    if ext == ".wav":
        return wav_bytes_to_audio(data)
    if ext == ".mp3":
        import soundfile as sf  # lazy: only hand-dropped mp3 samples need it

        arr, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return {
            "waveform": torch.from_numpy(arr.T.copy()).unsqueeze(0),  # [1,C,N]
            "sample_rate": sample_rate,
        }
    raise ValueError("unsupported voice sample extension %r" % ext)


# One wire carries the whole asset: the bundle types the generator slots
# consume. Named per kind so a scene cannot be wired into a character slot.
CHARACTER_ASSET_TYPE = "MIE_CHARACTER_ASSET"
SCENE_ASSET_TYPE = "MIE_SCENE_ASSET"


class SaveCharacterAssetV2(object):
    """Simple model: ONE image (may be a three-view sheet), ONE description,
    ONE audio clip, ONE voice line (2026-10-10 ruling)."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "voice": ("STRING", {"default": "", "multiline": True}),
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字"}),
            },
            "optional": {
                # Exactly one frame / one clip. Multi-image and multi-audio
                # stay on the (V1) nodes.
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "STRING", CHARACTER_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "voice", "image", "status", "asset")
    FUNCTION = "save_character"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name="", **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save_character(self, library_dir="", name="", description="", voice="",
                       new_name="", image=None, audio=None):
        name = resolve_save_name(name, new_name)
        png = None if image is None else image_tensor_to_png_bytes(image)
        wav = None if audio is None else audio_to_wav_bytes(audio)
        result = library.save_character("", name, description, voice,
                                        None if png is None else [png],
                                        None if wav is None else [wav])
        picture = png_bytes_list_to_image_tensor(result["png_bytes_list"][:1])
        audio_out = (
            voice_sample_bytes_to_audio(result["audio_bytes_list"][0],
                                        result["audio_exts"][0])
            if result["audio_bytes_list"] else None
        )
        bundle = {
            "kind": "character", "name": result["name"],
            "description": result["description"], "voice": result["voice"],
            "image": picture, "audio": audio_out,
        }
        return (result["name"], result["description"], result["voice"],
                picture, result["status"], bundle)


class LoadCharacterAssetV2(object):
    """Simple model read: name, description, ONE image, ONE audio, voice."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "AUDIO", CHARACTER_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "voice", "image", "audio", "asset")
    FUNCTION = "load_character"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return library.character_cache_key("", name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load_character(self, library_dir="", name=""):
        loaded = library.load_character("", name)
        if len(loaded["png_bytes_list"]) != 1 or len(loaded["audio_bytes_list"]) != 1:
            raise ValueError(
                "character %r does not fit the simple model (one image + one "
                "audio): found %d image(s) and %d audio clip(s). Use the (V1) "
                "Load Character Asset node, or trim the asset under "
                "characters/%s." % (loaded["name"], len(loaded["png_bytes_list"]),
                                    len(loaded["audio_bytes_list"]), loaded["name"])
            )
        image = png_bytes_list_to_image_tensor(loaded["png_bytes_list"])
        audio = voice_sample_bytes_to_audio(
            loaded["audio_bytes_list"][0], loaded["audio_exts"][0]
        )
        bundle = {
            "kind": "character", "name": loaded["name"],
            "description": loaded["description"], "voice": loaded["voice"],
            "image": image, "audio": audio,
        }
        return (loaded["name"], loaded["description"], loaded["voice"],
                image, audio, bundle)


class SaveSceneAssetV2(object):
    """Simple model: ONE image + ONE description."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", SCENE_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "image", "status", "asset")
    FUNCTION = "save_scene"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name="", **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save_scene(self, library_dir="", name="", description="", new_name="", image=None):
        name = resolve_save_name(name, new_name)
        png = None if image is None else image_tensor_to_png_bytes(image)
        result = library.save_scene("", name, description,
                                    None if png is None else [png])
        picture = png_bytes_list_to_image_tensor(result["png_bytes_list"][:1])
        bundle = {
            "kind": "scene", "name": result["name"],
            "description": result["description"], "image": picture,
        }
        return (result["name"], result["description"], picture,
                result["status"], bundle)


class LoadSceneAssetV2(object):
    """Simple model read: name, description, ONE image."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", SCENE_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "image", "asset")
    FUNCTION = "load_scene"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return library.scene_cache_key("", name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load_scene(self, library_dir="", name=""):
        loaded = library.load_scene("", name)
        if len(loaded["png_bytes_list"]) != 1:
            raise ValueError(
                "scene %r does not fit the simple model (one image): found %d "
                "image(s). Use the (V1) Load Scene Asset node, or trim the "
                "asset under scenes/%s."
                % (loaded["name"], len(loaded["png_bytes_list"]), loaded["name"])
            )
        image = png_bytes_list_to_image_tensor(loaded["png_bytes_list"])
        bundle = {
            "kind": "scene", "name": loaded["name"],
            "description": loaded["description"], "image": image,
        }
        return (loaded["name"], loaded["description"], image, bundle)
