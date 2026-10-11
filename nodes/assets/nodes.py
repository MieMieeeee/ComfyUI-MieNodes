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

from pathlib import Path

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

# 2026-10-11 merge ruling: ONE node per operation, kind picked in a dropdown.
# The per-kind V2 classes stay registered as CATEGORY "hidden" compat aliases
# so old workflows (typed widgets_values, no kind slot) keep loading unchanged.
KIND_COMBO = ["character", "scene"]
KIND_TIP = (
    "资产类型 kind：character = 角色（voice 音色 + 音频样本）；scene = 场景"
    "（只有描述 + 图，音色/音频被忽略）。\n"
    "Asset kind: character (voice + audio sample) or scene (description + "
    "image only; voice/audio are ignored)."
)


def _validate_exempt(name, **_kwargs):
    """Exempt the combo-fed name inputs from class-level membership checks.

    Pure exemption: returns True (or None) no matter what — a wrong name is
    reported by the load/save executor with the asset's kind and the
    absolute path, which beats a generic "value not in list" here.
    """
    return True


def resolve_save_name(name, new_name):
    """The save node's visible UI is a mode toggle (Existing / New).

    The hidden name combo carries a new-name sentinel while "New" mode is
    active — the real name comes from the new_name input. An existing name
    (Existing mode) is an explicit overwrite intent.
    """
    name = (name or "").strip()
    new_name = (new_name or "").strip()
    if name == NEW_NAME_SENTINEL or (NEW_NAME_SENTINEL in (name or "")):
        if not new_name:
            raise ValueError(
                "「✏️ 新建 New」模式已选中，但还没有输入新资产的名字：请在 ✏️ 新建 "
                "New 的输入框里填名字，或切回 📋 已有资产 Existing 选择要覆盖的资产"
                "。(New-asset mode is on but new_name is empty: type the new "
                "asset's name in the ✏️ 新建 New input, or switch back to 📋 已有"
                "资产 Existing and pick the asset to overwrite.)"
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


# Upload pickers built the LoadImage / LoadAudio way: the widget hints
# below make the FRONTEND render its native upload button next to the combo
# (image files land in the input dir, audio files in the input "audio"
# category), and the combo lists what folder_paths already indexes. No
# ComfyUI in the dev env -> empty option lists, guarded imports throughout.
IMAGE_UPLOAD_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_UPLOAD_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")


def _upload_image_options():
    try:
        import folder_paths

        files = folder_paths.get_filename_list("input")
    except Exception:
        return [""]
    return [""] + [
        f for f in files if Path(f).suffix.lower() in IMAGE_UPLOAD_EXTENSIONS
    ]


def _upload_audio_options():
    try:
        import folder_paths

        files = folder_paths.get_filename_list("audio")
    except Exception:
        return [""]
    return [""] + [
        f for f in files if Path(f).suffix.lower() in AUDIO_UPLOAD_EXTENSIONS
    ]


def _read_upload_file(name, category="input"):
    """Resolve a picker file name to bytes.

    ``get_annotated_filepath`` resolves against the INPUT root, but the
    audio picker lists names relative to the 'audio' category dir
    (input/audio — where our upload endpoint writes). Try both; a
    hand-placed file in the input root still works."""
    candidates = []
    try:
        import folder_paths

        path = folder_paths.get_annotated_filepath(name)
        if isinstance(path, tuple):
            path = path[0]
        candidates.append(path)
        if category == "audio":
            candidates.append(
                Path(folder_paths.get_input_directory()) / "audio" / name
            )
    except Exception:
        pass
    for path in candidates:
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    raise FileNotFoundError(
        "%r is not readable from the ComfyUI %s directory (drop the "
        "file via the upload button or a LoadImage box, then press R to "
        "refresh)" % (name, category)
    )


def upload_image_to_tensor(name):
    """A picked input-dir image -> IMAGE tensor [1,H,W,3]."""
    blob = _read_upload_file(name, "input")
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    import torch as _torch

    return _torch.from_numpy(arr.copy()).unsqueeze(0)


def upload_audio_to_audio(name):
    """A picked audio clip -> ComfyUI AUDIO dict (wav or mp3)."""
    blob = _read_upload_file(name, "audio")
    return voice_sample_bytes_to_audio(blob, Path(name).suffix.lower())


# One wire carries the whole asset: the bundle types the generator slots
# consume. Named per kind so a scene cannot be wired into a character slot.
CHARACTER_ASSET_TYPE = "MIE_CHARACTER_ASSET"
SCENE_ASSET_TYPE = "MIE_SCENE_ASSET"
# The generator's aggregated output: every wired bundle in slot order
# (characters then scenes) — what Asset → Slots splits back apart.
ASSET_ROSTER_TYPE = "MIE_ASSET_ROSTER"


class SaveAssetV2(object):
    """Unified save node (2026-10-11): kind dropdown, one node for both.

    Scene mode ignores the voice text and the whole audio path (no scene
    asset ever carries them) and returns the character-side outputs as
    blanks/None; character mode vice versa. Both typed asset bundles are
    emitted so downstream typed sockets (character_N / scene_N) keep their
    type safety. Everything else matches the per-kind nodes: Existing/New
    name toggle, upload pickers, socket>picker priority, overwrite
    semantics."""

    CATEGORY = ASSETS_CATEGORY

    _NAME_TIP = (
        "资产名：📋 已有资产 Existing 模式下拉选名字 = 覆盖该资产；✏️ 新建 New 模式输入名字 = 新建（两者互斥二选一）。\n"
        "Asset name: 📋 Existing mode picks an asset to overwrite; ✏️ New mode types a name to create — the two are mutually exclusive."
    )
    _DESC_PLACEHOLDER = (
        "description · 资产描述 Asset description（英文 EN 60–100 词 words）\n"
        "角色例 character e.g.: A kind elderly Chinese grandfather, 68 years old, short silver-white hair…\n"
        "场景例 scene e.g.: A cozy family living room at night, beige sofa against the left wall, warm floor lamp…"
    )
    _DESC_TIP = (
        "资产的英文描述（60–100 词）。角色 = 外观事实（年龄/发型/脸型/着装），场景 = 空间/陈设/光线/氛围。\n"
        "English description (60-100 words). Character = looks only; scene = space, furniture, lighting, atmosphere."
    )
    _VOICE_PLACEHOLDER = (
        "voice · 音色描述 Voice timbre（英文一行 EN one line，≤200 字符 chars，仅角色 · character only）\n"
        "例 e.g.: adult male, low thick steady voice, warm cheerful undertone, unhurried pace"
    )
    _VOICE_TIP = (
        "角色音色的英文一行描述（年龄/性别/音域/语速/质感，≤200 字符）；场景类型忽略本字段。\n"
        "One-line English timbre descriptor for characters (age/gender/register/pace/texture, <=200 chars); ignored for scenes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kind": (list(KIND_COMBO), {"default": "character", "tooltip": KIND_TIP}),
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {
                    "default": NAME_PLACEHOLDER[0],
                    "tooltip": cls._NAME_TIP,
                }),
                "description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": cls._DESC_PLACEHOLDER,
                    "tooltip": cls._DESC_TIP,
                }),
                "voice": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": cls._VOICE_PLACEHOLDER,
                    "tooltip": cls._VOICE_TIP,
                }),
                "new_name": ("STRING", {"default": "", "placeholder": "新资产的名字 new asset name"}),
            },
            "optional": {
                "image": ("IMAGE",),
                # Character only; a scene asset never carries audio.
                "audio": ("AUDIO",),
                # Upload pickers: files in ComfyUI's input directory. A
                # WIRED socket wins over a picked file; both empty on an
                # overwrite = keep the disk media.
                # Hidden STRING state; the visible picker + upload button
                # is one DOM widget built by js/assetLibraryUI.js.
                "image_upload": ("STRING", {"default": ""}),
                "audio_upload": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "STRING",
                    CHARACTER_ASSET_TYPE, SCENE_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "voice", "image", "status",
                    "character_asset", "scene_asset")
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, new_name="", **kwargs):
        return _validate_exempt(name, new_name=new_name, **kwargs)

    def save(self, kind="character", library_dir="", name="", description="",
             voice="", new_name="", image=None, audio=None,
             image_upload="", audio_upload=""):
        name = resolve_save_name(name, new_name)
        if image is None and (image_upload or "").strip():
            image = upload_image_to_tensor((image_upload or "").strip())
        if kind == "scene":
            png = None if image is None else image_tensor_to_png_bytes(image)
            result = library.save_scene("", name, description,
                                        None if png is None else [png])
            picture = png_bytes_list_to_image_tensor(result["png_bytes_list"][:1])
            scene_bundle = {
                "kind": "scene", "name": result["name"],
                "description": result["description"], "image": picture,
            }
            return (result["name"], result["description"], "", picture,
                    result["status"], None, scene_bundle)
        if audio is None and (audio_upload or "").strip():
            audio = upload_audio_to_audio((audio_upload or "").strip())
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
                picture, result["status"], bundle, None)


class SaveCharacterAssetV2(SaveAssetV2):
    """Legacy per-kind save node — hidden from the menu, loads old workflows.

    widgets_values shape is byte-compatible with the pre-merge node: no
    kind slot, the character widget set only."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        return base

    RETURN_TYPES = SaveAssetV2.RETURN_TYPES[:5] + (CHARACTER_ASSET_TYPE,)
    RETURN_NAMES = ("name", "description", "voice", "image", "status", "asset")
    FUNCTION = "save_character"

    def save_character(self, library_dir="", name="", description="", voice="",
                       new_name="", image=None, audio=None,
                       image_upload="", audio_upload=""):
        out = self.save("character", library_dir=library_dir, name=name,
                        description=description, voice=voice, new_name=new_name,
                        image=image, audio=audio, image_upload=image_upload,
                        audio_upload=audio_upload)
        return out[:5] + (out[5],)


class LoadAssetV2(object):
    """Unified load node: kind dropdown; both typed asset bundles emitted
    (the inactive kind's bundle is None), so typed downstream sockets keep
    working without dynamic RETURN_TYPES hacks."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kind": (list(KIND_COMBO), {"default": "character", "tooltip": KIND_TIP}),
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {"default": NAME_PLACEHOLDER[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "AUDIO",
                    CHARACTER_ASSET_TYPE, SCENE_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "voice", "image", "audio",
                    "character_asset", "scene_asset")
    FUNCTION = "load"

    @classmethod
    def IS_CHANGED(cls, kind, library_dir, name):
        if kind == "scene":
            return library.scene_cache_key("", name)
        return library.character_cache_key("", name)

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    def load(self, kind="character", library_dir="", name=""):
        if kind == "scene":
            loaded = library.load_scene("", name)
            if len(loaded["png_bytes_list"]) != 1:
                raise ValueError(
                    "scene %r does not fit the simple model (one image): "
                    "found %d image(s). Trim the asset under scenes/%s."
                    % (loaded["name"], len(loaded["png_bytes_list"]),
                       loaded["name"])
                )
            image = png_bytes_list_to_image_tensor(loaded["png_bytes_list"])
            scene_bundle = {
                "kind": "scene", "name": loaded["name"],
                "description": loaded["description"], "image": image,
            }
            return (loaded["name"], loaded["description"], "", image, None,
                    None, scene_bundle)
        loaded = library.load_character("", name)
        if len(loaded["png_bytes_list"]) != 1 or len(loaded["audio_bytes_list"]) != 1:
            raise ValueError(
                "character %r does not fit the simple model (one image + one "
                "audio): found %d image(s) and %d audio clip(s). Trim the "
                "asset under characters/%s."
                % (loaded["name"], len(loaded["png_bytes_list"]),
                   len(loaded["audio_bytes_list"]), loaded["name"])
            )
        image = png_bytes_list_to_image_tensor(loaded["png_bytes_list"])
        audio_out = voice_sample_bytes_to_audio(
            loaded["audio_bytes_list"][0], loaded["audio_exts"][0])
        bundle = {
            "kind": "character", "name": loaded["name"],
            "description": loaded["description"], "voice": loaded["voice"],
            "image": image, "audio": audio_out,
        }
        return (loaded["name"], loaded["description"], loaded["voice"],
                image, audio_out, bundle, None)


class LoadCharacterAssetV2(LoadAssetV2):
    """Legacy per-kind load node — hidden, loads old workflows."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        return base

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "AUDIO", CHARACTER_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "voice", "image", "audio", "asset")
    FUNCTION = "load_character"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return super().IS_CHANGED("character", library_dir, name)

    def load_character(self, library_dir="", name=""):
        out = self.load("character", library_dir=library_dir, name=name)
        return out[:5] + (out[5],)


class SaveSceneAssetV2(SaveAssetV2):
    """Legacy per-kind save node — hidden, loads old workflows.

    Pops the character-only widgets too (voice / audio / audio_upload) so
    the widget order matches the pre-merge scene node exactly."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        base["required"].pop("voice")
        base["optional"].pop("audio")
        base["optional"].pop("audio_upload")
        return base

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", SCENE_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "image", "status", "asset")
    FUNCTION = "save_scene"

    def save_scene(self, library_dir="", name="", description="", new_name="",
                   image=None, image_upload=""):
        out = self.save("scene", library_dir=library_dir, name=name,
                        description=description, new_name=new_name,
                        image=image, image_upload=image_upload)
        return (out[0], out[1], out[3], out[4], out[6])


class LoadSceneAssetV2(LoadAssetV2):
    """Legacy per-kind load node — hidden, loads old workflows."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        return base

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", SCENE_ASSET_TYPE)
    RETURN_NAMES = ("name", "description", "image", "asset")
    FUNCTION = "load_scene"

    @classmethod
    def IS_CHANGED(cls, library_dir, name):
        return super().IS_CHANGED("scene", library_dir, name)

    def load_scene(self, library_dir="", name=""):
        out = self.load("scene", library_dir=library_dir, name=name)
        return (out[0], out[1], out[3], out[6])


class _DeleteAssetBase(object):
    """Shared delete-node plumbing: the confirm gate + status text."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return _validate_exempt(name, **kwargs)

    _KIND = "character"

    def _delete(self, name, confirm, delete_fn, kind=None):
        kind = kind or self._KIND
        if not confirm:
            raise RuntimeError(
                "%s delete is ARMED BY A SECOND WIDGET: flip 「confirm 确认删除」"
                "to True on this node, then queue again. Nothing was deleted."
                % kind
            )
        result = delete_fn("", name)
        status = (
            "deleted %s %r (%d file(s) under %s) — Load nodes referencing it "
            "will fail on their next run until re-picked"
            % (result["kind"], result["name"], result["deleted_files"],
               result["deleted_dir"])
        )
        return (result["name"], status)


class DeleteAsset(_DeleteAssetBase):
    """Unified delete node: kind dropdown picks character or scene.

    The name dropdown lists only the chosen kind; the preview panel shows
    what is about to be deleted; confirm must be flipped to True
    (deliberate two-step for a destructive action)."""

    CATEGORY = ASSETS_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kind": (list(KIND_COMBO), {"default": "character", "tooltip": KIND_TIP}),
                "library_dir": ("STRING", {"default": ""}),
                "name": (list(NAME_PLACEHOLDER), {
                    "default": NAME_PLACEHOLDER[0],
                    "tooltip": "要删除的资产：下拉只列所选类型里已有的资产。\n"
                               "Asset to delete: the dropdown lists existing assets of the chosen kind only.",
                }),
                "confirm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "删除是破坏性操作：先选名字，再把本开关翻成 True，"
                               "排队执行才真正删除；False 时排队只会报错提醒，"
                               "什么都不删。\n"
                               "Destructive: pick the name, flip this to True, then "
                               "queue. False = loud error, nothing is deleted.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("name", "status")
    FUNCTION = "delete"
    OUTPUT_NODE = True

    def delete(self, kind="character", library_dir="", name="", confirm=False):
        delete_fn = (library.delete_scene if kind == "scene"
                     else library.delete_character)
        return self._delete(name, confirm, delete_fn, kind)


class DeleteCharacterAsset(DeleteAsset):
    """Legacy per-kind delete node — hidden, loads old workflows."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        return base

    FUNCTION = "delete_character"

    def delete_character(self, library_dir="", name="", confirm=False):
        return self.delete("character", library_dir=library_dir,
                           name=name, confirm=confirm)


class DeleteSceneAsset(DeleteAsset):
    """Legacy per-kind delete node — hidden, loads old workflows."""

    CATEGORY = "hidden"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"].pop("kind")
        return base

    FUNCTION = "delete_scene"

    def delete_scene(self, library_dir="", name="", confirm=False):
        return self.delete("scene", library_dir=library_dir,
                           name=name, confirm=confirm)


class AssetToSlots(object):
    """Split the generator's aggregated assets output into numbered slots.

    Wire `MiniMaxH3LoopPromptGeneratorV2.assets` into the single input
    (2026-10-11 ruling: the ONLY input — the old character_N/scene_N
    direct sockets are gone); out come, in the SAME slot order the plan
    used (characters then scenes = <Picture N>), ALL declared slots at
    once:

    * ``picture_1..9`` (IMAGE) — picture_i is the i-th asset's image;
      picture_1 -> ref_image_0 by construction;
    * ``audio_1..3`` (AUDIO) — the CHARACTERS' voice samples in character
      order (scenes carry none; H3 caps standalone audios at 3);
    * ``count`` / ``audio_count`` (INT) — how many are live.

    Slots beyond the roster carry None, which the H3 ref2v node skips
    WITHOUT occupying a reference number — wire any subset, leave the
    rest unconnected. The slot list is deliberately STATIC: JS trimming
    of outputs shifts link indices and silently retypes existing wires
    (live-caught as ref_audio receiving IMAGE), so extra outputs stay
    visible instead.
    """

    CATEGORY = ASSETS_CATEGORY

    MAX_SLOTS = 9
    MAX_AUDIOS = 3  # H3 standalone ref_audios cap

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "assets": (
                    ASSET_ROSTER_TYPE,
                    {"tooltip": "接 MiniMax H3 Loop Plan Generator V2 的 assets 输出（唯一入口）。\nWire the generator V2's assets output here (the only input)."},
                ),
            },
        }

    RETURN_TYPES = ("INT",) + ("IMAGE",) * 9 + ("INT",) + ("AUDIO",) * 3
    RETURN_NAMES = (
        ("count",)
        + tuple("picture_%d" % i for i in range(1, 10))
        + ("audio_count",)
        + tuple("audio_%d" % i for i in range(1, 4))
    )
    FUNCTION = "split"

    def split(self, assets=None):
        if assets is None:
            ordered = []
        elif isinstance(assets, (list, tuple)):
            ordered = list(assets)
            for i, bundle in enumerate(ordered, 1):
                if not isinstance(bundle, dict) or bundle.get("kind") not in (
                    "character",
                    "scene",
                ):
                    raise ValueError(
                        "assets item %d is not an asset bundle (got %s); wire "
                        "the 'assets' output of the generator"
                        % (i, type(bundle).__name__)
                    )
        else:
            raise ValueError(
                "assets must be the generator's MIE_ASSET_ROSTER output, got %s"
                % type(assets).__name__
            )
        names = [b["name"] for b in ordered]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(
                "duplicate asset in the roster: %s" % ", ".join(sorted(dupes))
            )
        if len(ordered) > self.MAX_SLOTS:
            raise ValueError(
                "%d assets in the roster; at most %d pictures are supported"
                % (len(ordered), self.MAX_SLOTS)
            )

        outs = [len(ordered)]
        for i in range(1, self.MAX_SLOTS + 1):
            outs.append(ordered[i - 1]["image"] if i <= len(ordered) else None)

        characters = [b for b in ordered if b.get("kind") == "character"]
        audios_out = [b.get("audio") for b in characters[: self.MAX_AUDIOS]]
        outs.append(sum(1 for a in audios_out if a is not None))
        for i in range(self.MAX_AUDIOS):
            outs.append(audios_out[i] if i < len(audios_out) else None)
        return tuple(outs)
