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
NAME_PLACEHOLDER = ["（先填 library_dir，下拉自动出现）"]
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


class SaveCharacterAsset(object):
    """Write name + appearance + voice + one portrait PNG + one voice sample."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
                "appearance": ("STRING", {"default": "", "multiline": True}),
                "voice": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "voice_sample": ("AUDIO",),
                # Appended LAST on purpose: older workflows' widgets_values
                # arrays are one entry short and load positionally-safe.
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("name", "voice", "image", "status", "appearance")
    FUNCTION = "save_character"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name, **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save_character(
        self, library_dir, name, appearance, voice, image, voice_sample,
        new_name="",
    ):
        name = resolve_save_name(name, new_name)
        png_bytes = image_tensor_to_png_bytes(image)
        wav_bytes = audio_to_wav_bytes(voice_sample)
        result = library.save_character(
            library_dir, name, appearance, voice, png_bytes, wav_bytes
        )
        portrait = png_bytes_to_image_tensor(result["png_bytes"])
        return (
            result["name"],
            result["voice"],
            portrait,
            result["status"],
            result["appearance"],
        )


class LoadCharacterAsset(object):
    """Read name + appearance + voice + portrait PNG + voice sample."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("name", "voice", "image", "voice_sample", "appearance")
    FUNCTION = "load_character"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return library.character_cache_key(library_dir, name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load_character(self, library_dir, name):
        result = library.load_character(library_dir, name)
        portrait = png_bytes_to_image_tensor(result["png_bytes"])
        voice_sample = voice_sample_bytes_to_audio(
            result["voice_sample_bytes"], result["voice_sample_ext"]
        )
        return (
            result["name"],
            result["voice"],
            portrait,
            voice_sample,
            result["appearance"],
        )


class LoadAssetBatch(object):
    """Load characters AND scenes in one go, batched and numbered for H3.

    The IMAGE batch order IS the slot contract: picture N of this node's
    output is <Picture N> in the prompt, and roster_json records which
    asset owns which slot and of which kind. Wire both outputs into the
    prompt generator together, or the binding is a guess.
    """

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "names": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "names", "roster_json", "status")
    FUNCTION = "load_asset_batch"

    @classmethod
    def IS_CHANGED(cls, library_dir, names):
        return library.asset_batch_cache_key(library_dir, names)

    def load_asset_batch(self, library_dir, names):
        result = library.load_character_batch(library_dir, names)
        assets = result["assets"]
        pictures = [
            png_bytes_to_image_tensor(entry["png_bytes"]) for entry in assets
        ]
        # One IMAGE batch needs one shape. Resizing here would silently
        # rewrite the user's picture, so name the offending assets and the
        # sizes instead — torch.cat's own message does not.
        shapes = {}
        for entry, picture in zip(assets, pictures):
            shapes.setdefault(tuple(picture.shape[1:]), []).append(entry["name"])
        if len(shapes) > 1:
            detail = "; ".join(
                "%dx%d: %s" % (w, h, ", ".join(names))
                for (h, w, _c), names in sorted(shapes.items())
            )
            raise ValueError(
                "cannot batch pictures of different sizes (%s). Export the "
                "asset pictures at one resolution — H3 resizes its reference "
                "images to a 2048 short edge anyway, so a consistent size "
                "costs nothing and keeps <Picture N> predictable" % detail
            )
        images = torch.cat(pictures, dim=0)
        roster = []
        for entry in assets:
            item = {
                "name": entry["name"],
                "kind": entry["kind"],
                "slot": entry["slot"],
            }
            if entry["kind"] == "character":
                item["appearance"] = entry["appearance"]
                item["voice"] = entry["voice"]
            else:
                item["description"] = entry["description"]
            roster.append(item)
        kinds = [e["kind"] for e in assets]
        return (
            images,
            "\n".join(entry["name"] for entry in assets),
            json.dumps(roster, ensure_ascii=False, indent=2),
            "loaded %d character(s), %d scene(s)"
            % (kinds.count("character"), kinds.count("scene")),
        )


class ListCharacterAssets(object):
    """List readable character names, one per line, sorted."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("names", "status")
    FUNCTION = "list_characters"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def list_characters(self, library_dir):
        result = library.list_characters(library_dir)
        return (result["names"], result["status"])


class SaveSceneAsset(object):
    """Write name + one description + one picture into the library."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                # Appended LAST on purpose (see SaveCharacterAsset).
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("name", "description", "image", "status")
    FUNCTION = "save_scene"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name, **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save_scene(self, library_dir, name, description, image, new_name=""):
        name = resolve_save_name(name, new_name)
        png_bytes = image_tensor_to_png_bytes(image)
        result = library.save_scene(library_dir, name, description, png_bytes)
        picture = png_bytes_to_image_tensor(result["png_bytes"])
        return (result["name"], result["description"], picture, result["status"])


class LoadSceneAsset(object):
    """Read name + description + picture for one scene."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("name", "description", "image")
    FUNCTION = "load_scene"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return library.scene_cache_key(library_dir, name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load_scene(self, library_dir, name):
        result = library.load_scene(library_dir, name)
        picture = png_bytes_to_image_tensor(result["png_bytes"])
        return (result["name"], result["description"], picture)


class ListSceneAssets(object):
    """List readable scene names, one per line, sorted."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("names", "status")
    FUNCTION = "list_scenes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def list_scenes(self, library_dir):
        result = library.list_scenes(library_dir)
        return (result["names"], result["status"])
