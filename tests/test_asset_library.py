# -*- coding: utf-8 -*-
"""Tests for nodes/assets — library.py plus the pure tensor/PNG/audio conversion.

Covers the required cases in docs/PLAN_ASSET_LIBRARY.md section 7 (including
the voice-sample revision). No ComfyUI process, no models; pytest tmp_path only.
"""

import io
import json
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nodes.assets import library
from nodes.assets.nodes import (
    audio_to_wav_bytes,
    image_tensor_to_png_bytes,
    png_bytes_to_image_tensor,
    voice_sample_bytes_to_audio,
    wav_bytes_to_audio,
)


def make_png(color=(255, 0, 0), width=6, height=4):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_wav(freq=440.0, seconds=0.05, sample_rate=24000, channels=1):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = (np.sin(2 * np.pi * freq * t) * 0.6 * 32767.0).astype("<i2")
    data = np.repeat(mono[:, None], channels, axis=1)  # [N, C]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())
    return buf.getvalue()


APPEARANCE = "A gray tabby cat with green eyes and a red collar."


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture()
def lib_dir(tmp_path):
    d = tmp_path / "asset_lib"
    d.mkdir()
    return d


# 1. save then read back: name, voice, PNG bytes and voice sample identical.
def test_save_character_then_load_roundtrip(lib_dir):
    png = make_png((10, 200, 30))
    wav = make_wav()
    result = library.save_character(str(lib_dir), "虎斑",APPEARANCE,  "成年男性，低音，平稳", png, wav)
    assert result["status"] == "created"
    assert result["name"] == "虎斑"

    loaded = library.load_character(str(lib_dir), "虎斑")
    assert loaded["name"] == "虎斑"
    assert loaded["appearance"] == APPEARANCE
    assert loaded["voice"] == "成年男性，低音，平稳"
    assert loaded["png_bytes"] == png
    assert loaded["voice_sample_bytes"] == wav
    assert loaded["voice_sample_ext"] == ".wav"

    char_dir = lib_dir / "characters" / "虎斑"
    assert sorted(p.name for p in char_dir.iterdir()) == [
        "character.json",
        "portrait.png",
        "voice_sample.wav",
    ]
    text = (char_dir / "character.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "虎斑" in text  # ensure_ascii=false keeps CJK verbatim
    assert read_json(char_dir / "character.json") == {
        "name": "虎斑",
        "appearance": APPEARANCE,
        "voice": "成年男性，低音，平稳",
    }
    assert (char_dir / "portrait.png").read_bytes() == png
    assert (char_dir / "voice_sample.wav").read_bytes() == wav


# 2. same-name save overwrites: status overwritten, new voice/PNG/sample on disk.
def test_save_character_overwrite(lib_dir):
    old_png = make_png((0, 0, 255))
    new_png = make_png((255, 0, 0))
    old_wav = make_wav(freq=440.0)
    new_wav = make_wav(freq=880.0)
    library.save_character(str(lib_dir), "Tiger",APPEARANCE,  "voice A", old_png, old_wav)
    result = library.save_character(str(lib_dir), "Tiger",APPEARANCE,  "voice B", new_png, new_wav)
    assert result["status"] == "overwritten"

    loaded = library.load_character(str(lib_dir), "Tiger")
    assert loaded["voice"] == "voice B"
    assert loaded["png_bytes"] == new_png
    assert loaded["voice_sample_bytes"] == new_wav


# 3. scene save / load / overwrite mirrors the character behaviour.
def test_scene_roundtrip_and_overwrite(lib_dir):
    setting = "深夜便利店打烊后，货架灯白光，冷柜嗡嗡作响。"
    png = make_png((10, 20, 30))
    result = library.save_scene(str(lib_dir), "便利店", setting, png)
    assert result == {
        "name": "便利店",
        "description": setting,
        "status": "created",
        "png_bytes": png,
    }
    assert read_json(lib_dir / "scenes" / "便利店" / "scene.json") == {
        "name": "便利店",
        "description": setting,
    }
    assert (lib_dir / "scenes" / "便利店" / "scene.png").read_bytes() == png

    loaded = library.load_scene(str(lib_dir), "便利店")
    assert loaded == {
        "name": "便利店",
        "description": setting,
        "png_bytes": png,
    }

    result = library.save_scene(
        str(lib_dir), "便利店", "白天 crowded", make_png((30, 20, 10))
    )
    assert result["status"] == "overwritten"
    assert library.load_scene(str(lib_dir), "便利店")["description"] == "白天 crowded"


# 4. listing is sorted; empty library (and missing characters/) returns "" + ok.
def test_list_characters_sorted_and_empty(lib_dir):
    assert library.list_characters(str(lib_dir)) == {"names": "", "status": "ok"}

    (lib_dir / "characters").mkdir(parents=True)
    assert library.list_characters(str(lib_dir)) == {"names": "", "status": "ok"}

    png = make_png()
    wav = make_wav()
    for name in ("虎斑", "beta", "Alpha"):
        library.save_character(str(lib_dir), name,APPEARANCE,  "v", png, wav)
    result = library.list_characters(str(lib_dir))
    assert result == {"names": "Alpha\nbeta\n虎斑", "status": "ok"}

    library.save_scene(str(lib_dir), "s1", "setting", make_png())
    assert library.list_scenes(str(lib_dir)) == {"names": "s1", "status": "ok"}
    assert library.list_scenes(str(lib_dir / "nope")) == {"names": "", "status": "ok"}


# 5. broken character directories are skipped and reported in status.
def test_list_characters_skips_broken(lib_dir):
    png = make_png()
    wav = make_wav()
    library.save_character(str(lib_dir), "good",APPEARANCE,  "v", png, wav)

    chars = lib_dir / "characters"

    no_portrait = chars / "no_portrait"
    no_portrait.mkdir()
    (no_portrait / "character.json").write_text(
        json.dumps({"name": "no_portrait", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )

    bad_json = chars / "bad_json"
    bad_json.mkdir()
    (bad_json / "character.json").write_text("{ not json", encoding="utf-8")
    (bad_json / "portrait.png").write_bytes(png)

    wrong_name = chars / "wrong_name"
    wrong_name.mkdir()
    (wrong_name / "character.json").write_text(
        json.dumps({"name": "别的", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (wrong_name / "portrait.png").write_bytes(png)

    no_sample = chars / "no_sample"
    no_sample.mkdir()
    (no_sample / "character.json").write_text(
        json.dumps({"name": "no_sample", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (no_sample / "portrait.png").write_bytes(png)

    ambiguous = chars / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / "character.json").write_text(
        json.dumps({"name": "ambiguous", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (ambiguous / "portrait.png").write_bytes(png)
    (ambiguous / "voice_sample.wav").write_bytes(wav)
    (ambiguous / "voice_sample.mp3").write_bytes(b"fake mp3")

    invalid_dir = chars / "has space"
    invalid_dir.mkdir()

    result = library.list_characters(str(lib_dir))
    assert result["names"] == "good"
    status = result["status"]
    assert "no_portrait" in status and "portrait.png" in status
    assert "bad_json" in status
    assert "wrong_name" in status and "别的" in status
    assert "no_sample" in status and "voice_sample" in status
    assert "ambiguous" in status
    assert "has space" in status


# 6. invalid names are rejected on save and nothing is created on disk.
@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "   ",
        "a" * 41,
        "..",
        "a/b",
        "a\\b",
        "CON",
        "con",
        "LPT3",
        ".hidden",
        "a b",
        "a.b",
        "a:b",
        "a*b",
        "emoji😀",
    ],
)
def test_invalid_names_rejected(lib_dir, bad_name):
    with pytest.raises(ValueError):
        library.save_character(str(lib_dir), bad_name,APPEARANCE,  "v", make_png(), make_wav())
    with pytest.raises(ValueError):
        library.save_scene(str(lib_dir), bad_name, "s", make_png())
    assert not (lib_dir / "characters").exists()
    assert not (lib_dir / "scenes").exists()


def test_save_character_requires_voice_sample(lib_dir):
    with pytest.raises(ValueError, match="voice_sample"):
        library.save_character(str(lib_dir), "x",APPEARANCE,  "v", make_png(), b"")
    assert not (lib_dir / "characters").exists()


# 7. library_dir must be absolute, non-empty and outside the plugin root.
def test_library_dir_validation(tmp_path):
    plugin_root = tmp_path / "plugin_root" / "ComfyUI_MieNodes"
    plugin_root.mkdir(parents=True)
    outside = tmp_path / "asset_lib"
    outside.mkdir()

    # sanity: outside the fake plugin root works
    library.save_scene(str(outside), "ok", "s", make_png(), plugin_root=plugin_root)

    inside = plugin_root / "library"
    with pytest.raises(ValueError, match="inside the plugin folder"):
        library.save_character(
            str(inside), "x",APPEARANCE,  "v", make_png(), make_wav(), plugin_root=plugin_root
        )
    with pytest.raises(ValueError, match="inside the plugin folder"):
        library.save_scene(
            str(plugin_root), "x", "s", make_png(), plugin_root=plugin_root
        )
    with pytest.raises(ValueError, match="inside the plugin folder"):
        library.load_character(str(inside), "x", plugin_root=plugin_root)
    assert not inside.exists()

    for bad in ("assets", "./assets"):
        with pytest.raises(ValueError):
            library.validate_library_dir(bad, plugin_root=plugin_root)


# The library location is fixed per machine (sixth revision): blank
# library_dir is the "use the default" marker, not an error.
def test_resolve_library_dir_default_env_and_passthrough(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(library.os.path, "expanduser", lambda p: str(home))
    monkeypatch.delenv(library.LIBRARY_DIR_ENV, raising=False)
    # Blank -> default under the (patched) home.
    assert library.resolve_library_dir("").endswith("mie_asset_library")
    assert library.resolve_library_dir(None).endswith("mie_asset_library")
    assert library.validate_library_dir("").is_absolute()
    # Env var overrides the home default.
    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(tmp_path / "custom_lib"))
    assert library.resolve_library_dir("") == str(tmp_path / "custom_lib")
    # Non-blank always wins (the older-workflow escape hatch).
    assert library.resolve_library_dir(str(tmp_path / "explicit")) == str(tmp_path / "explicit")
    # The env default flows through a real save -> load round-trip.
    env_lib = tmp_path / "env_lib"
    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(env_lib))
    library.save_character("", "虎斑", "a cat", "v", make_png(), make_wav())
    assert (env_lib / "characters" / "虎斑" / "character.json").is_file()
    assert library.load_character("", "虎斑")["name"] == "虎斑"


def test_payload_echoes_resolved_default_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(tmp_path))
    payload = library.list_assets_payload("")
    assert payload["ok"] is True
    assert payload["library_dir"] == str(tmp_path.resolve())


# 8. voice / setting length and emptiness rules.
def test_voice_and_setting_validation(lib_dir):
    png = make_png()
    wav = make_wav()
    with pytest.raises(ValueError):
        library.save_character(str(lib_dir), "x",APPEARANCE,  "", png, wav)
    with pytest.raises(ValueError):
        library.save_character(str(lib_dir), "x",APPEARANCE,  "   ", png, wav)
    with pytest.raises(ValueError):
        library.save_character(str(lib_dir), "x",APPEARANCE,  "a" * 201, png, wav)
    library.save_character(str(lib_dir), "x",APPEARANCE,  "a" * 200, png, wav)  # boundary ok

    with pytest.raises(ValueError):
        library.save_scene(str(lib_dir), "x", "", make_png())
    with pytest.raises(ValueError):
        library.save_scene(str(lib_dir), "x", " ", make_png())
    with pytest.raises(ValueError):
        library.save_scene(str(lib_dir), "x", "a" * 801, make_png())
    library.save_scene(str(lib_dir), "x", "a" * 800, make_png())  # boundary ok


# 9. three-file replace failure points.
def _patch_replace(monkeypatch, fail_when_name):
    real_replace = os.replace

    def fake_replace(src, dst, *args, **kwargs):
        if Path(dst).name == fail_when_name:
            raise OSError("injected failure for %s" % dst)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fake_replace)


def test_replace_failure_at_portrait_leaves_all_files(lib_dir, monkeypatch):
    old = library.save_character(
        str(lib_dir), "虎斑",APPEARANCE,  "old voice", make_png((0, 0, 255)), make_wav(freq=440.0)
    )
    char_dir = lib_dir / "characters" / "虎斑"
    old_json = (char_dir / "character.json").read_bytes()

    _patch_replace(monkeypatch, "portrait.png")
    with pytest.raises(OSError, match="unchanged"):
        library.save_character(
            str(lib_dir), "虎斑",APPEARANCE,  "new voice", make_png((255, 0, 0)), make_wav(freq=880.0)
        )

    assert (char_dir / "character.json").read_bytes() == old_json
    assert (char_dir / "portrait.png").read_bytes() == old["png_bytes"]
    assert (char_dir / "voice_sample.wav").read_bytes() == old["voice_sample_bytes"]
    assert sorted(p.name for p in char_dir.iterdir()) == [
        "character.json",
        "portrait.png",
        "voice_sample.wav",
    ]  # temp files cleaned up


def test_replace_failure_at_voice_sample_keeps_json(lib_dir, monkeypatch):
    old = library.save_character(
        str(lib_dir), "虎斑",APPEARANCE,  "old voice", make_png((0, 0, 255)), make_wav(freq=440.0)
    )
    char_dir = lib_dir / "characters" / "虎斑"
    old_json = (char_dir / "character.json").read_bytes()

    _patch_replace(monkeypatch, "voice_sample.wav")
    with pytest.raises(OSError, match="character.json and the previous voice sample are unchanged"):
        library.save_character(
            str(lib_dir),
            "虎斑",APPEARANCE, 
            "new voice",
            make_png((255, 0, 0)),
            make_wav(freq=880.0),
        )

    assert (char_dir / "portrait.png").read_bytes() != old["png_bytes"]  # new image
    assert (char_dir / "voice_sample.wav").read_bytes() == old["voice_sample_bytes"]
    assert (char_dir / "character.json").read_bytes() == old_json


def test_replace_failure_at_json_leaves_new_media_old_json(lib_dir, monkeypatch):
    old = library.save_character(
        str(lib_dir), "虎斑",APPEARANCE,  "old voice", make_png((0, 0, 255)), make_wav(freq=440.0)
    )
    char_dir = lib_dir / "characters" / "虎斑"
    old_json = (char_dir / "character.json").read_bytes()
    new_wav = make_wav(freq=880.0)

    _patch_replace(monkeypatch, "character.json")
    with pytest.raises(OSError, match="NEW portrait and NEW voice sample"):
        library.save_character(
            str(lib_dir), "虎斑",APPEARANCE,  "new voice", make_png((255, 0, 0)), new_wav
        )

    assert (char_dir / "portrait.png").read_bytes() != old["png_bytes"]
    assert (char_dir / "voice_sample.wav").read_bytes() == new_wav
    assert (char_dir / "character.json").read_bytes() == old_json
    assert sorted(p.name for p in char_dir.iterdir()) == [
        "character.json",
        "portrait.png",
        "voice_sample.wav",
    ]


# 10. cache keys track content (including the voice sample); missing -> marker.
def test_cache_keys_track_content(lib_dir):
    png = make_png()
    library.save_character(str(lib_dir), "虎斑",APPEARANCE,  "voice A", png, make_wav(freq=440.0))
    key1 = library.character_cache_key(str(lib_dir), "虎斑")
    assert key1 == library.character_cache_key(str(lib_dir), "虎斑")

    library.save_character(
        str(lib_dir), "虎斑",APPEARANCE,  "voice B", make_png((9, 9, 9)), make_wav(freq=880.0)
    )
    assert library.character_cache_key(str(lib_dir), "虎斑") != key1

    # changing only the voice sample must also change the key
    library.save_character(str(lib_dir), "虎斑",APPEARANCE,  "voice B", make_png((9, 9, 9)), make_wav(freq=660.0))
    key3 = library.character_cache_key(str(lib_dir), "虎斑")
    assert key3 != key1

    assert (
        library.character_cache_key(str(lib_dir), "ghost")
        == library.MISSING_CACHE_MARKER
    )
    assert library.character_cache_key(str(lib_dir), "ghost") != key1

    library.save_scene(str(lib_dir), "便利店", "s1", make_png()),
    skey1 = library.scene_cache_key(str(lib_dir), "便利店")
    assert skey1 == library.scene_cache_key(str(lib_dir), "便利店")
    library.save_scene(str(lib_dir), "便利店", "s2", make_png()),
    assert library.scene_cache_key(str(lib_dir), "便利店") != skey1
    assert (
        library.scene_cache_key(str(lib_dir), "ghost")
        == library.MISSING_CACHE_MARKER
    )


# 11. case-only name variants merge onto the first-created disk spelling.
def test_casefold_merge(lib_dir):
    png_a, png_b = make_png((0, 0, 255)), make_png((255, 0, 0))
    wav_a, wav_b = make_wav(freq=440.0), make_wav(freq=880.0)
    first = library.save_character(str(lib_dir), "Tiger",APPEARANCE,  "voice A", png_a, wav_a)
    assert first["name"] == "Tiger" and first["status"] == "created"

    second = library.save_character(str(lib_dir), "tiger",APPEARANCE,  "voice B", png_b, wav_b)
    assert second["name"] == "Tiger" and second["status"] == "overwritten"

    chars = lib_dir / "characters"
    assert [p.name for p in chars.iterdir()] == ["Tiger"]  # exactly one entry

    for spelling in ("tiger", "Tiger", "TIGER"):
        loaded = library.load_character(str(lib_dir), spelling)
        assert loaded["name"] == "Tiger"
        assert loaded["voice"] == "voice B"
        assert loaded["png_bytes"] == png_b
        assert loaded["voice_sample_bytes"] == wav_b
    assert read_json(chars / "Tiger" / "character.json")["name"] == "Tiger"

    # scenes merge the same way
    library.save_scene(str(lib_dir), "Store", "s1", make_png((0, 0, 255)))
    merged = library.save_scene(
        str(lib_dir), "store", "s2", make_png((255, 0, 0))
    )
    assert merged == {
        "name": "Store",
        "description": "s2",
        "status": "overwritten",
        "png_bytes": make_png((255, 0, 0)),
    }
    assert [p.name for p in (lib_dir / "scenes").iterdir()] == ["Store"]


# 12. load failures raise with the absolute path (and both names).
def test_load_character_failures(lib_dir):
    chars = lib_dir / "characters"

    with pytest.raises(FileNotFoundError) as e:
        library.load_character(str(lib_dir), "ghost")
    assert str(chars.resolve()) in str(e.value)

    corrupt = chars / "corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "character.json").write_text("{ nope", encoding="utf-8")
    (corrupt / "portrait.png").write_bytes(make_png())
    (corrupt / "voice_sample.wav").write_bytes(make_wav())
    with pytest.raises(ValueError) as e:
        library.load_character(str(lib_dir), "corrupt")
    assert str((corrupt / "character.json").resolve()) in str(e.value)

    no_png = chars / "no_png"
    no_png.mkdir()
    (no_png / "character.json").write_text(
        json.dumps({"name": "no_png", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (no_png / "voice_sample.wav").write_bytes(make_wav())
    with pytest.raises(FileNotFoundError) as e:
        library.load_character(str(lib_dir), "no_png")
    assert str((no_png / "portrait.png").resolve()) in str(e.value)

    no_sample = chars / "no_sample"
    no_sample.mkdir()
    (no_sample / "character.json").write_text(
        json.dumps({"name": "no_sample", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (no_sample / "portrait.png").write_bytes(make_png())
    with pytest.raises(FileNotFoundError) as e:
        library.load_character(str(lib_dir), "no_sample")
    msg = str(e.value)
    assert str(no_sample.resolve()) in msg
    assert "voice_sample.wav" in msg and "voice_sample.mp3" in msg

    mismatch = chars / "虎斑"
    mismatch.mkdir()
    (mismatch / "character.json").write_text(
        json.dumps({"name": "别的", "appearance": "a", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (mismatch / "portrait.png").write_bytes(make_png())
    (mismatch / "voice_sample.wav").write_bytes(make_wav())
    with pytest.raises(ValueError) as e:
        library.load_character(str(lib_dir), "虎斑")
    msg = str(e.value)
    assert str((mismatch / "character.json").resolve()) in msg
    assert "别的" in msg and "虎斑" in msg

    with pytest.raises(FileNotFoundError):
        library.load_scene(str(lib_dir), "ghost")
    bad_scene = lib_dir / "scenes" / "bad"
    bad_scene.mkdir(parents=True)
    (bad_scene / "scene.json").write_text("{ nope", encoding="utf-8")
    (bad_scene / "scene.png").write_bytes(make_png())
    with pytest.raises(ValueError):
        library.load_scene(str(lib_dir), "bad")

    no_pic = lib_dir / "scenes" / "no_pic"
    no_pic.mkdir()
    (no_pic / "scene.json").write_text(
        json.dumps({"name": "no_pic", "description": "d"}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="scene picture"):
        library.load_scene(str(lib_dir), "no_pic")


# 13. tensor -> PNG -> tensor stays within 1/255 per channel; batch of 1 only.
def test_tensor_png_roundtrip():
    h, w = 8, 6
    # Seeded: the tolerance below is exactly the quantisation step, so an
    # unlucky off-grid value can land a float epsilon past it. Unseeded this
    # failed roughly 1 run in 8, which trains everyone to ignore a red suite.
    torch.manual_seed(20261009)
    base = torch.linspace(0.0, 1.0, h * w * 3).reshape(1, h, w, 3).float()
    # Keep the probe inside the IMAGE domain. linspace lands exactly on 1.0,
    # so un-clamped noise pushes the endpoint past the domain, the converter
    # clips it, and the error is then the CLIP distance (not the
    # quantisation step) — an in-range bound cannot hold an out-of-range input.
    noisy = (base + (torch.rand_like(base) - 0.5) * 0.01).clamp(0.0, 1.0)
    png = image_tensor_to_png_bytes(noisy)
    back = png_bytes_to_image_tensor(png)

    assert back.shape == (1, h, w, 3)
    assert back.dtype == torch.float32
    # Truncation to uint8 puts the true error strictly below one quantisation
    # step, but float32 in the comparison itself can overshoot the step by
    # ~1e-7 relative. 1e-6 is still three orders of magnitude tighter than
    # any difference this contract cares about.
    assert torch.all((back - noisy).abs() <= 1.0 / 255.0 + 1e-6)
    assert not torch.equal(back, noisy)  # 8-bit quantization happened

    batch2 = torch.rand(2, h, w, 3)
    with pytest.raises(ValueError, match="one frame"):
        image_tensor_to_png_bytes(batch2)

    # Out-of-domain input is CLIPPED, not rejected — the 1/255 bound above
    # only covers [0, 1]; this covers what happens outside it.
    out_of_range = torch.tensor([[[[-0.5, 1.5, 0.5]]]])
    clipped = png_bytes_to_image_tensor(image_tensor_to_png_bytes(out_of_range))
    assert float(clipped[0, 0, 0, 0]) == 0.0
    assert float(clipped[0, 0, 0, 1]) == 1.0
    assert float(clipped[0, 0, 0, 2]) == pytest.approx(0.5, abs=1 / 255)


# 14. hand-dropped voice_sample.mp3: readable as-is; ambiguity is an error;
#     a node save supersedes the mp3 with the fresh wav.
def test_hand_dropped_mp3_voice_sample(lib_dir):
    sf = pytest.importorskip("soundfile")

    buf = io.BytesIO()
    sf.write(buf, np.zeros((2400, 1), dtype="float32"), 24000, format="MP3")
    mp3_bytes = buf.getvalue()
    assert mp3_bytes

    png = make_png()
    library.save_character(str(lib_dir), "x",APPEARANCE,  "v", png, make_wav())
    char_dir = lib_dir / "characters" / "x"

    # replace the wav with a hand-dropped mp3
    (char_dir / "voice_sample.wav").unlink()
    (char_dir / "voice_sample.mp3").write_bytes(mp3_bytes)

    loaded = library.load_character(str(lib_dir), "x")
    assert loaded["voice_sample_ext"] == ".mp3"
    assert loaded["voice_sample_bytes"] == mp3_bytes

    audio = voice_sample_bytes_to_audio(loaded["voice_sample_bytes"], ".mp3")
    assert audio["sample_rate"] == 24000
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[1] == 1

    # both files present -> ambiguous, load fails loudly, list skips
    (char_dir / "voice_sample.wav").write_bytes(make_wav())
    with pytest.raises(ValueError, match="ambiguous voice sample"):
        library.load_character(str(lib_dir), "x")
    listing = library.list_characters(str(lib_dir))
    assert listing["names"] == ""
    assert "ambiguous" in listing["status"]

    # a node-path save removes the superseded mp3
    library.save_character(str(lib_dir), "x",APPEARANCE,  "v2", png, make_wav(freq=880.0))
    assert not (char_dir / "voice_sample.mp3").exists()
    loaded = library.load_character(str(lib_dir), "x")
    assert loaded["voice_sample_ext"] == ".wav"

    with pytest.raises(ValueError, match="unsupported voice sample extension"):
        voice_sample_bytes_to_audio(b"xx", ".flac")


# 15. AUDIO dict <-> 16-bit PCM WAV roundtrip within 1/32767 per sample.
def test_audio_wav_roundtrip():
    n, channels, sr = 500, 2, 32000
    t = torch.linspace(-1.0, 1.0, n * channels).reshape(1, channels, n)
    audio = {"waveform": t, "sample_rate": sr}

    wav = audio_to_wav_bytes(audio)
    back = wav_bytes_to_audio(wav)

    assert back["sample_rate"] == sr
    assert back["waveform"].shape == (1, channels, n)
    assert back["waveform"].dtype == torch.float32
    assert torch.all((back["waveform"] - t).abs() <= 1.0 / 32767.0 + 1e-9)
    assert not torch.equal(back["waveform"], t)  # 16-bit quantization happened

    # mono roundtrip too
    mono = {"waveform": torch.zeros(1, 1, 10), "sample_rate": 16000}
    back_mono = wav_bytes_to_audio(audio_to_wav_bytes(mono))
    assert back_mono["waveform"].shape == (1, 1, 10)


def test_audio_to_wav_bytes_rejects_bad_shapes():
    with pytest.raises(ValueError, match="AUDIO connection"):
        audio_to_wav_bytes("not a dict")
    with pytest.raises(ValueError, match="AUDIO connection"):
        audio_to_wav_bytes({"waveform": torch.zeros(1, 1, 10)})
    with pytest.raises(ValueError, match="exactly one"):
        audio_to_wav_bytes({"waveform": torch.zeros(2, 1, 10), "sample_rate": 16000})
    with pytest.raises(ValueError, match="non-empty"):
        audio_to_wav_bytes({"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000})


def test_wav_bytes_to_audio_rejects_non_pcm16():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit
        w.setframerate(16000)
        w.writeframes(b"\x00\x80\xff\x7f")
    with pytest.raises(ValueError, match="16-bit PCM"):
        wav_bytes_to_audio(buf.getvalue())




# --------------------------------------------------------------------------- #
# Mixed character/scene batch loading + the Load Asset Batch node
# --------------------------------------------------------------------------- #
def test_load_asset_batch_mixes_characters_and_scenes_in_order(lib_dir):
    png, wav = make_png(), make_wav()
    library.save_character(str(lib_dir), "Alice", "red hair", "voice A", png, wav)
    library.save_scene(str(lib_dir), "便利店", "cold light, humming cooler", png)
    library.save_character(str(lib_dir), "Bob", "blue coat", "voice B", png, wav)

    result = library.load_character_batch(str(lib_dir), "Alice\n便利店\nBob")
    entries = result["assets"]
    assert [e["name"] for e in entries] == ["Alice", "便利店", "Bob"]
    assert [e["kind"] for e in entries] == ["character", "scene", "character"]
    assert [e["slot"] for e in entries] == [1, 2, 3]
    # Each entry carries its kind's fields and nothing else.
    assert entries[0]["appearance"] == "red hair"
    assert "description" not in entries[0]
    assert entries[1]["description"] == "cold light, humming cooler"
    assert "appearance" not in entries[1] and "voice" not in entries[1]


def test_load_asset_batch_rejects_a_name_under_both_kinds(lib_dir):
    png, wav = make_png(), make_wav()
    library.save_character(str(lib_dir), "Ghost", "red hair", "voice A", png, wav)
    library.save_scene(str(lib_dir), "Ghost", "an empty room", png)
    with pytest.raises(ValueError, match="exists both as a character"):
        library.load_character_batch(str(lib_dir), "Ghost")


def test_load_asset_batch_fails_loudly_rather_than_skipping(lib_dir):
    png, wav = make_png(), make_wav()
    library.save_character(str(lib_dir), "Alice", "red hair", "voice A", png, wav)
    with pytest.raises(FileNotFoundError) as e:
        library.load_character_batch(str(lib_dir), "Alice\nGhost")
    assert "Ghost" in str(e.value)
    with pytest.raises(ValueError, match="listed twice"):
        library.load_character_batch(str(lib_dir), "Alice\nAlice")
    with pytest.raises(ValueError, match="names is empty"):
        library.load_character_batch(str(lib_dir), "  \n \n")


def test_asset_batch_cache_key_tracks_every_file_and_the_order(lib_dir):
    png, wav = make_png(), make_wav()
    library.save_character(str(lib_dir), "Alice", "red hair", "voice A", png, wav)
    library.save_scene(str(lib_dir), "便利店", "cold light", png)

    key = library.asset_batch_cache_key(str(lib_dir), "Alice\n便利店")
    assert key == library.asset_batch_cache_key(str(lib_dir), "Alice\n便利店")
    # Reordering changes which picture sits behind which name.
    assert key != library.asset_batch_cache_key(str(lib_dir), "便利店\nAlice")
    # Changing the scene's picture changes the key too.
    library.save_scene(
        str(lib_dir), "便利店", "cold light", make_png((1, 2, 3))
    )
    assert key != library.asset_batch_cache_key(str(lib_dir), "Alice\n便利店")
    # A missing asset is a fixed marker, not a cached success.
    assert (
        library.asset_batch_cache_key(str(lib_dir), "Alice\nGhost")
        == library.MISSING_CACHE_MARKER
    )


def test_load_asset_batch_node_writes_kind_aware_roster(lib_dir):
    from nodes.assets.nodes import LoadAssetBatch

    library.save_character(
        str(lib_dir), "Alice", "A woman in a red coat.", "voice A",
        make_png((255, 0, 0), width=8, height=6), make_wav(freq=440.0),
    )
    library.save_scene(
        str(lib_dir), "Store",
        "A convenience store at midnight, cold fluorescent tubes.",
        make_png((0, 0, 255), width=8, height=6),
    )

    node = LoadAssetBatch()
    images, names, roster_json, status = node.load_asset_batch(
        str(lib_dir), "Alice\nStore"
    )
    assert images.shape == (2, 6, 8, 3)
    assert names.splitlines() == ["Alice", "Store"]
    assert status == "loaded 1 character(s), 1 scene(s)"

    roster = json.loads(roster_json)
    assert [r["slot"] for r in roster] == [1, 2]
    assert roster[0] == {
        "name": "Alice",
        "kind": "character",
        "slot": 1,
        "appearance": "A woman in a red coat.",
        "voice": "voice A",
    }
    assert roster[1] == {
        "name": "Store",
        "kind": "scene",
        "slot": 2,
        "description": "A convenience store at midnight, cold fluorescent tubes.",
    }
    # Batch order IS the slot order: Alice's red portrait is image 1.
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, 3).expand(6, 8, 3)
    blue = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).expand(6, 8, 3)
    assert torch.allclose(images[0], red, atol=1 / 255)
    assert torch.allclose(images[1], blue, atol=1 / 255)


def test_load_asset_batch_node_names_mismatched_picture_sizes(lib_dir):
    """torch.cat's message for a shape mismatch names tensors, not assets;
    the node's must name the assets and the sizes."""
    from nodes.assets.nodes import LoadAssetBatch

    library.save_character(
        str(lib_dir), "Alice", "A woman in a red coat.", "voice A",
        make_png((255, 0, 0), width=8, height=6), make_wav(),
    )
    library.save_scene(
        str(lib_dir), "Store", "a store", make_png((0, 0, 255), width=4, height=2)
    )
    node = LoadAssetBatch()
    with pytest.raises(ValueError) as e:
        node.load_asset_batch(str(lib_dir), "Alice" + chr(10) + "Store")
    message = str(e.value)
    assert "Alice" in message and "Store" in message
    assert "8x6" in message and "4x2" in message


# --------------------------------------------------------------------------- #
# Frontend assist: dropdown payload, save-name resolution, validation exempt
# --------------------------------------------------------------------------- #
def test_list_assets_payload_merges_kinds_and_skip_notes(lib_dir):
    png, wav = make_png(), make_wav()
    library.save_character(str(lib_dir), "Alice", "red hair", "voice A", png, wav)
    library.save_scene(str(lib_dir), "便利店", "cold light", png)
    payload = library.list_assets_payload(str(lib_dir))
    assert payload == {
        "ok": True,
        "library_dir": str(lib_dir.resolve()),
        "characters": ["Alice"],
        "scenes": ["便利店"],
        "error": None,
    }


def test_list_assets_payload_empty_and_missing_library(lib_dir, tmp_path):
    assert library.list_assets_payload(str(lib_dir))["ok"] is True
    assert library.list_assets_payload(str(tmp_path / "nope"))["ok"] is True


def test_list_assets_payload_never_raises_on_bad_dir():
    bad = library.list_assets_payload("relative/path")
    assert bad["ok"] is False
    assert bad["characters"] == [] and bad["scenes"] == []
    assert "absolute" in bad["error"]
    inside = library.list_assets_payload(str(Path(library.__file__).parent))
    assert inside["ok"] is False
    assert "inside the plugin folder" in inside["error"]


def test_list_assets_payload_carries_skip_notes(lib_dir):
    chars = lib_dir / "characters"
    chars.mkdir(parents=True)
    broken = chars / "broken"
    broken.mkdir()
    (broken / "character.json").write_text("{ nope", encoding="utf-8")
    (broken / "portrait.png").write_bytes(make_png())
    (broken / "voice_sample.wav").write_bytes(make_wav())
    payload = library.list_assets_payload(str(lib_dir))
    assert payload["ok"] is True
    assert payload["characters"] == []
    assert payload["error"] and "broken" in payload["error"]


def test_resolve_save_name_sentinel_and_overwrite():
    from nodes.assets.nodes import resolve_save_name, NEW_NAME_SENTINEL

    # Sentinel + empty new_name -> actionable error, not a weird filename.
    with pytest.raises(ValueError, match="new_name is empty"):
        resolve_save_name(NEW_NAME_SENTINEL, "  ")
    with pytest.raises(ValueError, match="new_name is empty"):
        resolve_save_name(NEW_NAME_SENTINEL, "")
    # Sentinel + new name -> the new name wins.
    assert resolve_save_name(NEW_NAME_SENTINEL, " 虎斑 ") == "虎斑"
    # Existing name picked from the dropdown -> explicit overwrite intent.
    assert resolve_save_name("[角色] 夏栀", "") == "[角色] 夏栀"
    assert resolve_save_name("夏栀", "whatever") == "夏栀"


def test_save_character_node_resolves_new_name(lib_dir):
    from nodes.assets.nodes import SaveCharacterAsset, NEW_NAME_SENTINEL

    node = SaveCharacterAsset()
    with pytest.raises(ValueError, match="new_name is empty"):
        node.save_character(
            str(lib_dir), NEW_NAME_SENTINEL, "looks", "voice",
            png_bytes_to_image_tensor(make_png()),
            wav_bytes_to_audio(make_wav()),
            new_name="",
        )
    out = node.save_character(
        str(lib_dir), NEW_NAME_SENTINEL, "looks", "voice",
        png_bytes_to_image_tensor(make_png()),
        wav_bytes_to_audio(make_wav()),
        new_name="虎斑",
    )
    assert out[0] == "虎斑" and out[3] == "created"
    assert (lib_dir / "characters" / "虎斑").is_dir()


def test_validate_inputs_exempts_combo_membership():
    """The anchor the whole dropdown scheme rests on: a name typed in an
    older workflow (or fed by the frontend scan) is NOT in the class-level
    placeholder list, and VALIDATE_INPUTS must wave it through — the
    executor's own not-found error is the message users need."""
    from nodes.assets import nodes as asset_nodes

    for cls in (
        asset_nodes.LoadCharacterAsset,
        asset_nodes.LoadSceneAsset,
        asset_nodes.SaveCharacterAsset,
        asset_nodes.SaveSceneAsset,
    ):
        assert cls.VALIDATE_INPUTS("夏栀", new_name="") is True
        assert cls.VALIDATE_INPUTS("（先填 library_dir，下拉自动出现）", new_name="") is True


def test_name_widgets_are_combos_with_placeholder():
    from nodes.assets import nodes as asset_nodes

    for cls in (
        asset_nodes.LoadCharacterAsset,
        asset_nodes.LoadSceneAsset,
        asset_nodes.SaveCharacterAsset,
        asset_nodes.SaveSceneAsset,
    ):
        spec = cls.INPUT_TYPES()["required"]["name"]
        assert isinstance(spec[0], list) and "library_dir" in spec[0][0]


def test_save_nodes_append_new_name_last():
    """Old workflows' widgets_values arrays are positional; new widgets must
    come AFTER every widget that existed before."""
    from nodes.assets import nodes as asset_nodes

    order = list(asset_nodes.SaveCharacterAsset.INPUT_TYPES()["required"])
    assert order[-1] == "new_name"
    assert order[:4] == ["library_dir", "name", "appearance", "voice"]
    order = list(asset_nodes.SaveSceneAsset.INPUT_TYPES()["required"])
    assert order[-1] == "new_name"
    assert order[:3] == ["library_dir", "name", "description"]
