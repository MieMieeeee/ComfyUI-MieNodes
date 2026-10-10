# -*- coding: utf-8 -*-
"""Tests for nodes/assets — library.py (V2 layout) + node conversions.

Covers docs/PLAN_ASSET_LIBRARY_V2.md section 9 acceptance plus the storage
edge cases. No ComfyUI process, no models; pytest tmp_path only.
"""

import io
import json
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
    image_batch_to_png_bytes_list,
    png_bytes_list_to_image_tensor,
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture()
def lib_dir(tmp_path):
    d = tmp_path / "asset_lib"
    d.mkdir()
    return d


# --------------------------------------------------------------------------- #
# V2 disk layout: save / load round-trips
# --------------------------------------------------------------------------- #
def test_v2_character_roundtrip_multi_image_multi_audio(lib_dir):
    png_a, png_b = make_png((1, 2, 3), 8, 6), make_png((4, 5, 6), 8, 6)
    wav_a, wav_b = make_wav(freq=440.0), make_wav(freq=880.0)
    result = library.save_character(
        str(lib_dir), "A", "desc a", "voice a", [png_a, png_b], [wav_a, wav_b]
    )
    assert result["status"] == "created"
    assert result["png_bytes_list"] == [png_a, png_b]
    assert result["audio_bytes_list"] == [wav_a, wav_b]
    char_dir = lib_dir / "characters" / "A"
    assert sorted(p.name for p in (char_dir / "images").iterdir()) == ["01.png", "02.png"]
    assert sorted(p.name for p in (char_dir / "audio").iterdir()) == ["01.wav", "02.wav"]
    assert read_json(char_dir / "character.json") == {
        "name": "A", "description": "desc a", "voice": "voice a",
    }

    loaded = library.load_character(str(lib_dir), "A")
    assert loaded["description"] == "desc a"
    assert loaded["voice"] == "voice a"
    assert loaded["png_bytes_list"] == [png_a, png_b]
    assert loaded["audio_bytes_list"] == [wav_a, wav_b]
    assert loaded["audio_exts"] == [".wav", ".wav"]


def test_v2_zero_audio_character_roundtrip(lib_dir):
    library.save_character(str(lib_dir), "Silent", "desc", "voice", [make_png()], [])
    loaded = library.load_character(str(lib_dir), "Silent")
    assert loaded["audio_bytes_list"] == []
    assert (lib_dir / "characters" / "Silent" / "audio").is_dir()
    assert list((lib_dir / "characters" / "Silent" / "audio").iterdir()) == []


def test_v2_scene_roundtrip_multi_image(lib_dir):
    png_a, png_b = make_png((1, 1, 1), 6, 6), make_png((2, 2, 2), 6, 6)
    library.save_scene(str(lib_dir), "便利店", "cold light", [png_a, png_b])
    loaded = library.load_scene(str(lib_dir), "便利店")
    assert loaded["description"] == "cold light"
    assert loaded["png_bytes_list"] == [png_a, png_b]
    assert read_json(lib_dir / "scenes" / "便利店" / "scene.json")["description"] == "cold light"


# --------------------------------------------------------------------------- #
# Group replace / keep semantics (acceptance 6)
# --------------------------------------------------------------------------- #
def test_keep_media_on_text_only_save(lib_dir):
    png_a, wav_a = make_png((9, 9, 9)), make_wav(freq=100.0)
    library.save_character(str(lib_dir), "X", "old", "old voice", [png_a], [wav_a])
    result = library.save_character(str(lib_dir), "X", "new text", "old voice", None, None)
    assert result["status"] == "overwritten"
    assert result["png_bytes_list"] == [png_a]
    assert result["audio_bytes_list"] == [wav_a]
    char_dir = lib_dir / "characters" / "X"
    assert (char_dir / "images" / "01.png").read_bytes() == png_a
    assert (char_dir / "audio" / "01.wav").read_bytes() == wav_a
    assert read_json(char_dir / "character.json")["description"] == "new text"


def test_group_replace_shrinks_and_grows_sets(lib_dir):
    pngs3 = [make_png((i, i, i), 6, 6) for i in range(3)]
    pngs1 = [make_png((7, 7, 7), 6, 6)]
    wavs2 = [make_wav(freq=440.0), make_wav(freq=880.0)]
    library.save_character(str(lib_dir), "Y", "d", "v", pngs3, wavs2)
    library.save_character(str(lib_dir), "Y", "d", "v", pngs1, None)
    assert sorted(p.name for p in (lib_dir / "characters" / "Y" / "images").iterdir()) == ["01.png"]
    assert len(list((lib_dir / "characters" / "Y" / "audio").iterdir())) == 2
    library.save_character(str(lib_dir), "Y", "d", "v", None, [make_wav(freq=10.0)])
    assert sorted(p.name for p in (lib_dir / "characters" / "Y" / "audio").iterdir()) == ["01.wav"]
    assert len(list((lib_dir / "characters" / "Y" / "images").iterdir())) == 1


def test_new_asset_requires_image(lib_dir):
    with pytest.raises(ValueError, match="NEW character needs at least one image"):
        library.save_character(str(lib_dir), "N", "d", "v", None, None)
    assert not (lib_dir / "characters" / "N").exists()
    with pytest.raises(ValueError, match="NEW scene needs at least one image"):
        library.save_scene(str(lib_dir), "N", "d", None)
    assert not (lib_dir / "scenes" / "N").exists()


def test_empty_payload_lists_rejected_before_dirs(lib_dir):
    with pytest.raises(ValueError, match="non-empty list"):
        library.save_character(str(lib_dir), "E", "d", "v", [], None)
    with pytest.raises(ValueError):
        library.save_character(str(lib_dir), "E", "d", "v", None, [])
    assert not (lib_dir / "characters" / "E").exists()


# --------------------------------------------------------------------------- #
# Same-size rule (acceptance 8)
# --------------------------------------------------------------------------- #
def test_mismatched_sizes_fail_with_names_and_dims(lib_dir):
    png_big, png_small = make_png((0, 0, 0), 8, 6), make_png((0, 0, 0), 4, 3)
    with pytest.raises(ValueError, match="share one resolution"):
        library.save_character(str(lib_dir), "M", "d", "v", [png_big, png_small], None)
    library.save_character(str(lib_dir), "M", "d", "v", [png_big], None)
    (lib_dir / "characters" / "M" / "images" / "02.png").write_bytes(png_small)
    with pytest.raises(ValueError) as e:
        library.load_character(str(lib_dir), "M")
    msg = str(e.value)
    assert "01.png (8x6)" in msg and "02.png (4x3)" in msg


# --------------------------------------------------------------------------- #
# Pre-V2 migration (acceptance 7 + plan section 4)
# --------------------------------------------------------------------------- #
def _make_legacy_character(lib_dir, name, *, with_mp3=False):
    d = lib_dir / "characters" / name
    d.mkdir(parents=True)
    (d / "character.json").write_text(
        json.dumps({"name": name, "appearance": "old look", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    portrait = make_png((9, 9, 9))
    (d / "portrait.png").write_bytes(portrait)
    if with_mp3:
        (d / "voice_sample.mp3").write_bytes(b"fake mp3")
    else:
        (d / "voice_sample.wav").write_bytes(make_wav(freq=55.0))
    return portrait


def test_legacy_character_reads_as_one_and_one(lib_dir):
    _make_legacy_character(lib_dir, "旧角")
    loaded = library.load_character(str(lib_dir), "旧角")
    assert loaded["description"] == "old look"  # appearance fallback
    assert len(loaded["png_bytes_list"]) == 1
    assert len(loaded["audio_bytes_list"]) == 1


def test_legacy_assets_survive_text_only_save_and_migrate(lib_dir):
    portrait = _make_legacy_character(lib_dir, "旧角", with_mp3=True)
    library.save_character(str(lib_dir), "旧角", "new look", "v", None, None)
    d = lib_dir / "characters" / "旧角"
    assert sorted(p.name for p in d.iterdir()) == ["audio", "character.json", "images"]
    assert (d / "images" / "01.png").read_bytes() == portrait
    assert (d / "audio" / "01.mp3").read_bytes() == b"fake mp3"
    loaded = library.load_character(str(lib_dir), "旧角")
    assert loaded["description"] == "new look"
    assert loaded["png_bytes_list"] == [portrait]
    assert loaded["audio_exts"] == [".mp3"]


def test_legacy_scene_image_used_until_migrated(lib_dir):
    d = lib_dir / "scenes" / "旧景"
    d.mkdir(parents=True)
    (d / "scene.json").write_text(
        json.dumps({"name": "旧景", "description": "old setting"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (d / "scene.png").write_bytes(make_png((3, 3, 3)))
    loaded = library.load_scene(str(lib_dir), "旧景")
    assert len(loaded["png_bytes_list"]) == 1
    library.save_scene(str(lib_dir), "旧景", "new setting", None)
    assert sorted(p.name for p in d.iterdir()) == ["images", "scene.json"]
    assert (d / "images" / "01.png").is_file()


# --------------------------------------------------------------------------- #
# Natural order + caps + ambiguous audio
# --------------------------------------------------------------------------- #
def test_natural_order_images(lib_dir):
    d = lib_dir / "characters" / "N"
    d.mkdir(parents=True)
    (d / "character.json").write_text(
        json.dumps({"name": "N", "description": "d", "voice": "v"}), encoding="utf-8"
    )
    (d / "images").mkdir()
    (d / "images" / "10.png").write_bytes(make_png((1, 1, 1)))
    (d / "images" / "2.png").write_bytes(make_png((2, 2, 2)))
    (d / "images" / "1.png").write_bytes(make_png((3, 3, 3)))
    loaded = library.load_character(str(lib_dir), "N")
    assert loaded["png_bytes_list"] == [
        make_png((3, 3, 3)), make_png((2, 2, 2)), make_png((1, 1, 1)),
    ]


def test_over_cap_and_ambiguous_audio_rejected(lib_dir):
    d = lib_dir / "characters" / "Cap"
    d.mkdir(parents=True)
    (d / "character.json").write_text(
        json.dumps({"name": "Cap", "description": "d", "voice": "v"}), encoding="utf-8"
    )
    (d / "images").mkdir()
    (d / "images" / "01.png").write_bytes(make_png())
    (d / "audio").mkdir()
    for i in range(1, library.MAX_AUDIO + 2):
        (d / "audio" / ("%02d.wav" % i)).write_bytes(make_wav(freq=float(i)))
    with pytest.raises(ValueError, match="at most %d" % library.MAX_AUDIO):
        library.load_character(str(lib_dir), "Cap")
    (d / "audio" / "01.mp3").write_bytes(b"x")
    (d / "audio" / "02.wav").unlink()
    with pytest.raises(ValueError, match="ambiguous audio"):
        library.load_character(str(lib_dir), "Cap")


# --------------------------------------------------------------------------- #
# Batch: first image only + counts + roster compat (plan section 6)
# --------------------------------------------------------------------------- #
def _load_h3_prompts_module():
    import importlib.util
    import types

    LLM_DIR = PROJECT_DIR / "nodes" / "llm"
    PROMPTS_DIR = LLM_DIR / "prompts"
    for fqn, sub in [
        ("_mienodes_internal", None),
        ("_mienodes_internal.core", PROJECT_DIR / "core"),
        ("_mienodes_internal.nodes", PROJECT_DIR / "nodes"),
        ("_mienodes_internal.nodes.llm", LLM_DIR),
        ("_mienodes_internal.nodes.llm.prompts", PROMPTS_DIR),
    ]:
        if fqn in sys.modules:
            continue
        m = types.ModuleType(fqn)
        if sub:
            m.__path__ = [str(sub)]
        m.__package__ = fqn
        sys.modules[fqn] = m
    for fqn, path in [
        ("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py"),
        ("_mienodes_internal.nodes.llm.prompts.loader", PROMPTS_DIR / "loader.py"),
        ("_mienodes_internal.nodes.llm.h3_prompts", LLM_DIR / "h3_prompts.py"),
        (
            "_mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts",
            LLM_DIR / "minimax_h3_storyboard_prompts.py",
        ),
        (
            "_mienodes_internal.nodes.llm.minimax_h3_loop_prompts",
            LLM_DIR / "minimax_h3_loop_prompts.py",
        ),
    ]:
        spec = importlib.util.spec_from_file_location(fqn, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
    return sys.modules["_mienodes_internal.nodes.llm.minimax_h3_loop_prompts"]


def test_batch_first_image_counts(lib_dir):
    library.save_character(
        str(lib_dir), "Alice", "looks", "voice A",
        [make_png((1, 2, 3), 8, 6), make_png((4, 5, 6), 8, 6)],
        [make_wav(freq=1.0), make_wav(freq=2.0), make_wav(freq=3.0)],
    )
    library.save_scene(str(lib_dir), "夜市", "a market", [make_png((9, 9, 9), 8, 6)])
    entries = library.load_character_batch(str(lib_dir), "Alice\n夜市")["assets"]
    assert [e["name"] for e in entries] == ["Alice", "夜市"]
    assert entries[0]["image_count"] == 2 and entries[0]["audio_count"] == 3
    assert entries[0]["png_bytes"] == entries[0]["png_bytes_list"][0]
    assert entries[1]["image_count"] == 1
    assert "audio_count" not in entries[1]


def test_roster_shape_parses_with_shipped_h3_parser(lib_dir):
    lp = _load_h3_prompts_module()
    entries = lp.parse_asset_roster(json.dumps([
        {"name": "Alice", "kind": "character", "slot": 1,
         "description": "looks", "appearance": "looks", "voice": "voice A",
         "image_count": 1, "audio_count": 1},
        {"name": "S", "kind": "scene", "slot": 2,
         "description": "a scene", "image_count": 1},
    ], ensure_ascii=False))
    assert [e["name"] for e in entries] == ["Alice", "S"]
    assert lp.roster_bindings(entries) == [("Alice", 1)]
    assert lp.roster_voices(entries) == {"Alice": "voice A"}
    assert entries[0]["image_count"] == 1 and entries[1]["image_count"] == 1



def test_save_node_ignores_library_dir_and_bundles(lib_dir, monkeypatch):
    from nodes.assets.nodes import SaveCharacterAssetV2, NEW_NAME_SENTINEL

    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(lib_dir))
    node = SaveCharacterAssetV2()
    name, description, voice, image, status, bundle = node.save_character(
        # An old workflow's stored dir must be IGNORED (fixed library).
        library_dir="C:/definitely/not/a/real/library",
        name=NEW_NAME_SENTINEL, description="d", voice="v", new_name="Z",
        image=png_bytes_list_to_image_tensor([make_png((5, 5, 5))]),
        audio=wav_bytes_to_audio(make_wav(freq=440.0)),
    )
    assert (name, status) == ("Z", "created")
    assert bundle["kind"] == "character" and bundle["name"] == "Z"
    assert bundle["description"] == "d" and "waveform" in bundle["audio"]

    # Text-only overwrite via the node keeps disk media; bundle echoes it.
    _, _, _, _, _, bundle2 = node.save_character(name="Z", description="d2", voice="v2")
    loaded = library.load_character("", "Z")
    assert loaded["description"] == "d2"
    assert len(loaded["png_bytes_list"]) == 1
    assert len(loaded["audio_bytes_list"]) == 1
    assert bundle2["image"].shape[0] == 1


def test_widget_order_and_output_shapes():
    from nodes.assets.nodes import (
        CHARACTER_ASSET_TYPE,
        SCENE_ASSET_TYPE,
        LoadCharacterAssetV2,
        LoadSceneAssetV2,
        SaveCharacterAssetV2,
        SaveSceneAssetV2,
    )

    req = list(SaveCharacterAssetV2.INPUT_TYPES()["required"])
    opt = SaveCharacterAssetV2.INPUT_TYPES()["optional"]
    assert req[0] == "library_dir"  # hidden, positional anchor for old workflows
    assert req[-1] == "new_name"
    assert "image" in opt and "audio" in opt
    assert LoadCharacterAssetV2.RETURN_TYPES[-1] == CHARACTER_ASSET_TYPE
    assert LoadSceneAssetV2.RETURN_TYPES[-1] == SCENE_ASSET_TYPE
    assert SaveCharacterAssetV2.RETURN_NAMES == (
        "name", "description", "voice", "image", "status", "asset"
    )
    assert SaveSceneAssetV2.RETURN_NAMES == (
        "name", "description", "image", "status", "asset"
    )


# --------------------------------------------------------------------------- #
# Fixed library dir (plan V2 section 3) — resolution + payload echo
# --------------------------------------------------------------------------- #
def test_resolve_library_dir_default_is_plugin_subfolder(monkeypatch, tmp_path):
    # Patch BOTH escape hatches so a dev machine's real library (or legacy
    # home dir) can never be touched by this test.
    monkeypatch.delenv(library.LIBRARY_DIR_ENV, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(library.os.path, "expanduser", lambda p: str(home))
    resolved = library.resolve_library_dir("", plugin_root=tmp_path / "plugin")
    assert resolved == str(tmp_path / "plugin" / "asset_library")


def test_legacy_home_library_migrates_once(monkeypatch, tmp_path):
    monkeypatch.delenv(library.LIBRARY_DIR_ENV, raising=False)
    home = tmp_path / "home"
    legacy = home / "mie_asset_library" / "characters" / "夏栀"
    legacy.mkdir(parents=True)
    (legacy / "character.json").write_text(
        json.dumps({"name": "夏栀", "description": "d", "voice": "v"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(library.os.path, "expanduser", lambda p: str(home))
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    resolved = library.validate_library_dir("", plugin_root=plugin_root)
    assert resolved == (plugin_root / "asset_library").resolve()
    assert (plugin_root / "asset_library" / "characters" / "夏栀").is_dir()
    assert not (home / "mie_asset_library").exists()  # moved, not copied
    # Second resolution does not need (and must not disturb) the legacy dir.
    (home / "mie_asset_library").mkdir()
    assert library.validate_library_dir("", plugin_root=plugin_root) == resolved
    assert (home / "mie_asset_library").is_dir()  # untouched once target exists


def test_env_override_beats_plugin_subfolder(monkeypatch, tmp_path):
    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(tmp_path / "custom"))
    assert library.resolve_library_dir("", plugin_root=tmp_path / "plugin") == str(tmp_path / "custom")
    assert library.resolve_library_dir("explicit/path") == "explicit/path"


def test_list_payload_counts_and_dir_echo(lib_dir):
    library.save_character(
        str(lib_dir), "A", "d", "v",
        [make_png((0, 0, 0), 8, 6), make_png((1, 1, 1), 8, 6)],
        [make_wav(freq=440.0)],
    )
    library.save_scene(str(lib_dir), "S", "d", [make_png((2, 2, 2))])
    payload = library.list_assets_payload(str(lib_dir))
    assert payload["ok"] is True
    assert payload["library_dir"] == str(lib_dir.resolve())
    assert payload["characters"] == [
        {"name": "A", "image_count": 2, "audio_count": 1}
    ]
    assert payload["scenes"] == [{"name": "S", "image_count": 1}]


def test_read_asset_meta_v2(lib_dir):
    library.save_character(
        str(lib_dir), "A", "desc text", "voice text",
        [make_png((7, 7, 7))], [make_wav(freq=100.0), make_wav(freq=200.0)],
    )
    meta = library.read_asset_meta(str(lib_dir), "A", "character")
    assert meta["description"] == "desc text"
    assert meta["voice"] == "voice text"
    assert meta["image_count"] == 1
    assert meta["audio_count"] == 2
    assert meta["picture_size"] > 0 and meta["picture_mtime_ns"] > 0
    _make_legacy_character(lib_dir, "旧角")
    meta = library.read_asset_meta(str(lib_dir), "旧角", "character")
    assert meta["description"] == "old look"
    assert meta["image_count"] == 1 and meta["audio_count"] == 1


# --------------------------------------------------------------------------- #
# Cache keys: every image and every clip counts
# --------------------------------------------------------------------------- #
def test_cache_key_covers_every_file(lib_dir):
    library.save_character(
        str(lib_dir), "A", "d", "v",
        [make_png((1, 1, 1), 8, 6), make_png((2, 2, 2), 8, 6)],
        [make_wav(freq=440.0)],
    )
    key = library.character_cache_key(str(lib_dir), "A")
    assert key == library.character_cache_key(str(lib_dir), "A")
    (lib_dir / "characters" / "A" / "images" / "02.png").write_bytes(make_png((9, 9, 9), 8, 6))
    assert key != library.character_cache_key(str(lib_dir), "A")
    key = library.character_cache_key(str(lib_dir), "A")
    (lib_dir / "characters" / "A" / "audio" / "02.wav").write_bytes(make_wav(freq=2.0))
    assert key != library.character_cache_key(str(lib_dir), "A")
    assert library.character_cache_key(str(lib_dir), "Ghost") == library.MISSING_CACHE_MARKER
    assert library.scene_cache_key(str(lib_dir), "Ghost") == library.MISSING_CACHE_MARKER


def test_asset_batch_cache_key_tracks_every_file_and_order(lib_dir):
    library.save_character(str(lib_dir), "A", "d", "v", [make_png()], [make_wav()])
    library.save_scene(str(lib_dir), "S", "d", [make_png()])
    key = library.asset_batch_cache_key(str(lib_dir), "A\nS")
    assert key == library.asset_batch_cache_key(str(lib_dir), "A\nS")
    assert key != library.asset_batch_cache_key(str(lib_dir), "S\nA")
    (lib_dir / "characters" / "A" / "images" / "01.png").write_bytes(make_png((8, 8, 8)))
    assert key != library.asset_batch_cache_key(str(lib_dir), "A\nS")


# --------------------------------------------------------------------------- #
# Listing + skip reasons + casefold (unchanged behaviours)
# --------------------------------------------------------------------------- #
def test_list_characters_sorted_and_skip_reasons(lib_dir):
    library.save_character(str(lib_dir), "Good", "d", "v", [make_png()], [make_wav()])
    broken = lib_dir / "characters" / "broken"
    broken.mkdir(parents=True)
    (broken / "character.json").write_text("{ nope", encoding="utf-8")
    result = library.list_characters(str(lib_dir))
    assert result["names"] == "Good"
    assert "broken" in result["status"]
    empty = lib_dir / "no_lib"
    assert library.list_characters(str(empty)) == {"names": "", "status": "ok"}


def test_casefold_merge_still_applies(lib_dir):
    library.save_character(str(lib_dir), "Tiger", "d", "v", [make_png()], [make_wav()])
    merged = library.save_character(str(lib_dir), "tiger", "d2", "v2", None, None)
    assert merged["name"] == "Tiger" and merged["status"] == "overwritten"
    assert [p.name for p in (lib_dir / "characters").iterdir()] == ["Tiger"]


def test_cross_kind_same_name_rejected_in_batch(lib_dir):
    library.save_character(str(lib_dir), "Ghost", "d", "v", [make_png()], [make_wav()])
    library.save_scene(str(lib_dir), "Ghost", "d", [make_png()])
    with pytest.raises(ValueError, match="exists both as a character"):
        library.load_character_batch(str(lib_dir), "Ghost")


# --------------------------------------------------------------------------- #
# Name / text validation (unchanged shapes)
# --------------------------------------------------------------------------- #
def test_invalid_names_rejected(lib_dir):
    for bad in ("", "   ", "..", "a/b", "a\\b", "a" * 41, "CON"):
        with pytest.raises(ValueError):
            library.save_character(str(lib_dir), bad, "d", "v", [make_png()], [make_wav()])
    assert not (lib_dir / "characters").exists()


def test_text_lengths_validated(lib_dir):
    with pytest.raises(ValueError, match="description"):
        library.save_character(str(lib_dir), "x", "", "v", [make_png()], [make_wav()])
    with pytest.raises(ValueError, match="800"):
        library.save_character(str(lib_dir), "x", "a" * 801, "v", [make_png()], [make_wav()])
    with pytest.raises(ValueError, match="voice"):
        library.save_character(str(lib_dir), "x", "d", "", [make_png()], [make_wav()])
    with pytest.raises(ValueError, match="200"):
        library.save_character(str(lib_dir), "x", "d", "a" * 201, [make_png()], [make_wav()])


def test_validate_inputs_exempts_combo_membership():
    from nodes.assets.nodes import (
        LoadCharacterAssetV2,
        LoadSceneAssetV2,
        SaveCharacterAssetV2,
        SaveSceneAssetV2,
    )

    for cls in (LoadCharacterAssetV2, LoadSceneAssetV2,
                SaveCharacterAssetV2, SaveSceneAssetV2):
        assert cls.VALIDATE_INPUTS("夏栀", new_name="") is True


def test_library_dir_guards(tmp_path):
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    outside = tmp_path / "lib"
    outside.mkdir()
    library.validate_library_dir(str(outside), plugin_root=plugin_root)
    with pytest.raises(ValueError, match="absolute"):
        library.validate_library_dir("relative", plugin_root=plugin_root)
    # Since the 2026-10-10 ruling the library LIVES inside the plugin
    # folder — an explicit inside-plugin path is no longer rejected (the
    # mirror-sync script ignores asset_library/ instead).
    inside = plugin_root / "asset_library"
    library.validate_library_dir(str(inside), plugin_root=plugin_root)


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def test_image_batch_roundtrip():
    torch.manual_seed(20261010)
    batch = torch.rand(3, 8, 6, 3)
    pngs = image_batch_to_png_bytes_list(batch)
    assert len(pngs) == 3
    back = png_bytes_list_to_image_tensor(pngs)
    assert back.shape == (3, 8, 6, 3)
    for i in range(3):
        assert torch.all((back[i] - batch[i]).abs() <= 1 / 255 + 1e-6)


def test_image_batch_caps():
    with pytest.raises(ValueError, match="empty"):
        image_batch_to_png_bytes_list(torch.rand(0, 4, 4, 3))
    with pytest.raises(ValueError, match="at most"):
        image_batch_to_png_bytes_list(torch.rand(library.MAX_IMAGES + 1, 4, 4, 3))


def test_audio_wav_roundtrip():
    audio = wav_bytes_to_audio(make_wav(freq=440.0))
    back = wav_bytes_to_audio(audio_to_wav_bytes(audio))
    assert back["waveform"].shape == audio["waveform"].shape
    diff = (back["waveform"] - audio["waveform"]).abs().max()
    assert float(diff) <= 1 / 32767 + 1e-9


# --------------------------------------------------------------------------- #
# V2-simple nodes (2026-10-10 ruling): ONE image, ONE audio, texts.
# --------------------------------------------------------------------------- #
def test_v2_simple_character_node_roundtrip(lib_dir, monkeypatch):
    from nodes.assets.nodes import LoadCharacterAssetV2, SaveCharacterAssetV2

    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(lib_dir))
    save = SaveCharacterAssetV2()
    out = save.save_character(
        name="Solo", description="one look", voice="one voice",
        image=png_bytes_list_to_image_tensor([make_png((6, 6, 6))]),
        audio=wav_bytes_to_audio(make_wav(freq=440.0)),
    )
    assert out[0] == "Solo" and out[4] == "created"
    name, description, voice, image, audio, bundle = LoadCharacterAssetV2().load_character("", "Solo")
    assert (name, description, voice) == ("Solo", "one look", "one voice")
    assert image.shape[0] == 1 and "waveform" in audio


def test_v2_simple_nodes_reject_multi_assets(lib_dir, monkeypatch):
    from nodes.assets.nodes import (
        LoadCharacterAssetV2,
        LoadSceneAssetV2,
        SaveCharacterAssetV2,
    )

    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(lib_dir))
    library.save_character(
        str(lib_dir), "Multi", "d", "v",
        [make_png((1, 1, 1), 8, 6), make_png((2, 2, 2), 8, 6)],
        [make_wav(freq=440.0), make_wav(freq=880.0)],
    )
    with pytest.raises(ValueError, match="does not fit the simple model.*2 image.*2 audio"):
        LoadCharacterAssetV2().load_character("", "Multi")
    library.save_scene(
        str(lib_dir), "ManyScene", "d",
        [make_png((1, 1, 1), 6, 6), make_png((2, 2, 2), 6, 6)],
    )
    with pytest.raises(ValueError, match="does not fit the simple model.*2 image"):
        LoadSceneAssetV2().load_scene("", "ManyScene")
    # A multi-frame batch on the simple SAVE node dies at the one-frame gate.
    with pytest.raises(ValueError, match="exactly one frame"):
        SaveCharacterAssetV2().save_character(
            name="Bad", description="d", voice="v",
            image=png_bytes_list_to_image_tensor(
                [make_png((1, 1, 1)), make_png((2, 2, 2))]
            ),
            audio=wav_bytes_to_audio(make_wav()),
        )


def test_v2_simple_scene_node_roundtrip(lib_dir, monkeypatch):
    from nodes.assets.nodes import LoadSceneAssetV2, SaveSceneAssetV2

    monkeypatch.setenv(library.LIBRARY_DIR_ENV, str(lib_dir))
    SaveSceneAssetV2().save_scene(
        name="Alley", description="a rainy alley",
        image=png_bytes_list_to_image_tensor([make_png((3, 3, 3))]),
    )
    name, description, image, bundle = LoadSceneAssetV2().load_scene("", "Alley")
    assert (name, description) == ("Alley", "a rainy alley")
    assert image.shape[0] == 1


# --------------------------------------------------------------------------- #
# AssetPicturesToSlots: the ordering bridge between V2.images and ref_image_N
# --------------------------------------------------------------------------- #
def test_asset_pictures_to_slots_split_and_none_pad():
    import torch

    from nodes.assets.nodes import AssetPicturesToSlots

    node = AssetPicturesToSlots()
    batch = torch.rand(3, 8, 6, 3)
    outs = node.split(batch, 3)
    assert outs[0] == 3
    for i in (1, 2, 3):
        assert outs[i] is not None and outs[i].shape == (1, 8, 6, 3)
        assert torch.allclose(outs[i][0], batch[i - 1])
    for i in range(4, 10):
        assert outs[i] is None  # beyond the batch: H3 skips None without numbering


def test_asset_pictures_to_slots_count_bounds():
    import torch

    from nodes.assets.nodes import AssetPicturesToSlots

    node = AssetPicturesToSlots()
    batch = torch.rand(5, 4, 4, 3)
    outs = node.split(batch, 2)  # count < batch: extra frames unused
    assert outs[0] == 2 and outs[3] is None
    outs = node.split(batch, 9)  # count > batch: padded with None
    assert outs[0] == 5 and outs[6] is None and outs[5] is not None
    outs = node.split(None, 3)  # unwired images: all picture outputs None
    assert outs[0] == 0 and all(o is None for o in outs[1:])


def test_asset_pictures_to_slots_spec():
    from nodes.assets.nodes import AssetPicturesToSlots

    assert AssetPicturesToSlots.RETURN_NAMES == (
        ("count",) + tuple("picture_%d" % i for i in range(1, 10))
    )
    spec = AssetPicturesToSlots.INPUT_TYPES()["required"]
    assert spec["count"][1]["max"] == 9
