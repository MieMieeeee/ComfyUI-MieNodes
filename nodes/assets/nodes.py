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

# Missing audio_i outputs become blockers (plan V2 section 5): a consumer
# wired to a clip the asset does not have fails with a message that says how
# many exist; an unconnected empty socket harms nothing. The class lives in
# the ComfyUI server process; dev/pytest get a local stand-in.
try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:  # pragma: no cover - outside a ComfyUI process

    class ExecutionBlocker(object):
        def __init__(self, message):
            self.message = message

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


class SaveCharacterAsset(object):
    """Write name + description + voice + an image set + an audio set.

    The audio sockets grow one at a time (js/assetLibraryUI.js mirrors the
    stringFormat autogrow): audio_1 is always there, audio_2 appears once
    audio_1 is wired, up to 8. None connected = keep the disk clips;
    ANY connected = replace the whole set with those clips. Same rule for
    the image batch. A new character needs at least one image.
    """

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "images": ("IMAGE",),
        }
        for i in range(1, library.MAX_AUDIO + 1):
            optional["audio_%d" % i] = ("AUDIO",)
        return {
            "required": {
                # Hidden and IGNORED since plan V2: the library location is
                # fixed per machine. Stays first so old workflows'
                # widgets_values arrays stay positionally aligned.
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "voice": ("STRING", {"default": "", "multiline": True}),
                # Appended LAST on purpose: older workflows' widgets_values
                # arrays are one entry short and load positionally-safe.
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("name", "description", "voice", "image", "status")
    FUNCTION = "save_character"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name="", **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save_character(
        self, library_dir="", name="", description="", voice="", new_name="",
        images=None, **audio_kwargs,
    ):
        name = resolve_save_name(name, new_name)
        png_list = None if images is None else image_batch_to_png_bytes_list(images)
        wav_list = [audio_to_wav_bytes(a) for a in audio_kwargs.values() if a is not None]
        result = library.save_character(
            "", name, description, voice, png_list, wav_list or None
        )
        picture = png_bytes_list_to_image_tensor(result["png_bytes_list"][:1])
        return (
            result["name"],
            result["description"],
            result["voice"],
            picture,
            result["status"],
        )


class LoadCharacterAsset(object):
    """Read one character: name, description, the image batch, the voice
    line, and every audio clip on its own socket."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING") + ("AUDIO",) * library.MAX_AUDIO
    RETURN_NAMES = ("name", "description", "images", "voice") + tuple(
        "audio_%d" % i for i in range(1, library.MAX_AUDIO + 1)
    )
    FUNCTION = "load_character"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return library.character_cache_key("", name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load_character(self, library_dir="", name=""):
        result = library.load_character("", name)
        images = png_bytes_list_to_image_tensor(result["png_bytes_list"])
        clips = [
            voice_sample_bytes_to_audio(blob, ext)
            for blob, ext in zip(result["audio_bytes_list"], result["audio_exts"])
        ]
        blocker_note = (
            "%s has %d audio clip(s); %%s does not exist. Wire a lower "
            "audio socket, or drop more clips under characters/%s/audio/."
            % (result["name"], len(clips), result["name"])
        )
        outs = [
            result["name"],
            result["description"],
            images,
            result["voice"],
        ]
        for i in range(1, library.MAX_AUDIO + 1):
            if i <= len(clips):
                outs.append(clips[i - 1])
            else:
                outs.append(ExecutionBlocker(blocker_note % ("audio_%d" % i)))
        return tuple(outs)


class LoadAssetBatch(object):
    """Load characters AND scenes in one go, batched and numbered for H3.

    The IMAGE batch order IS the slot contract: picture N of this node's
    output is <Picture N> in the prompt. Each asset contributes its FIRST
    image only (plan V2 section 6) — expanding extra images would shift
    <Picture N> behind the user's back; the full sets come from the Load
    nodes, and roster_json carries image_count / audio_count.
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
        return library.asset_batch_cache_key("", names)

    def load_asset_batch(self, library_dir="", names=""):
        result = library.load_character_batch("", names)
        assets = result["assets"]
        pictures = [png_bytes_to_image_tensor(entry["png_bytes"]) for entry in assets]
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
                "description": entry["description"],
                "image_count": entry["image_count"],
            }
            if entry["kind"] == "character":
                # appearance mirrors description so the already-shipped
                # roster parser needs no change in this plan (V2 section 6).
                item["appearance"] = entry["description"]
                item["voice"] = entry["voice"]
                item["audio_count"] = entry["audio_count"]
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

    def list_characters(self, library_dir=""):
        result = library.list_characters("")
        return (result["names"], result["status"])


class SaveSceneAsset(object):
    """Write name + description + an image set (scenes have no audio)."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
                "description": ("STRING", {"default": "", "multiline": True}),
                # Appended LAST on purpose (see SaveCharacterAsset).
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字"}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("name", "description", "image", "status")
    FUNCTION = "save_scene"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name="", **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save_scene(self, library_dir="", name="", description="", new_name="", images=None):
        name = resolve_save_name(name, new_name)
        png_list = None if images is None else image_batch_to_png_bytes_list(images)
        result = library.save_scene("", name, description, png_list)
        picture = png_bytes_list_to_image_tensor(result["png_bytes_list"][:1])
        return (result["name"], result["description"], picture, result["status"])


class LoadSceneAsset(object):
    """Read name + description + the image batch for one scene."""

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
    RETURN_NAMES = ("name", "description", "images")
    FUNCTION = "load_scene"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return library.scene_cache_key("", name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load_scene(self, library_dir="", name=""):
        result = library.load_scene("", name)
        images = png_bytes_list_to_image_tensor(result["png_bytes_list"])
        return (result["name"], result["description"], images)


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

    def list_scenes(self, library_dir=""):
        result = library.list_scenes("")
        return (result["names"], result["status"])


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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("name", "description", "voice", "image", "status")
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
        return (result["name"], result["description"], result["voice"],
                picture, result["status"])


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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("name", "description", "voice", "image", "audio")
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
        return (loaded["name"], loaded["description"], loaded["voice"], image, audio)


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

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("name", "description", "image", "status")
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
        return (result["name"], result["description"], picture, result["status"])


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

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("name", "description", "image")
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
        return (loaded["name"], loaded["description"], image)
