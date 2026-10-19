# -*- coding: utf-8 -*-
"""On-disk character & scene asset library (docs/PLAN_ASSET_LIBRARY_V2.md).

Pure filesystem layer: validation, atomic writes, reads, listing and content
hashing. No ComfyUI, no torch, no nodes.llm — images travel as PNG bytes,
audio as file bytes, tensor conversion stays in nodes.py.

Library layout (V2):

    <library_dir>/characters/<name>/character.json   name + description + voice
    <library_dir>/characters/<name>/images/01.png    1-8 PNGs, same W x H,
    <library_dir>/characters/<name>/images/02.png    natural filename order
    <library_dir>/characters/<name>/audio/01.wav     0-8 clips; hand-dropped
    <library_dir>/characters/<name>/audio/02.mp3     .mp3 read as-is (never
                                                     both extensions on one
                                                     index)
    <library_dir>/scenes/<name>/scene.json           name + description + ambience (optional)
    <library_dir>/scenes/<name>/images/01.png        1-8 PNGs

Pre-V2 layouts (portrait.png / voice_sample.* / scene.png / JSON
``appearance``) are read transparently as a one-image / one-clip asset; the
next successful save rewrites the new layout and deletes the superseded
files, so nothing is ever counted twice. Read-only access never deletes.

``ambience`` is the optional one-line English mood sentence used by the
zero-LLM storyboard assembler (PLAN_STORYBOARD_ASSET.md §3.2). Old scenes
without the field load transparently with ``ambience = ''``.

``description`` and ``voice`` are authored English text inserted into the H3
prompt verbatim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

# Plugin package root: nodes/assets/library.py -> nodes/assets -> nodes -> root.
PLUGIN_ROOT = Path(__file__).resolve().parents[2]

NAME_MAX_CHARS = 40
APPEARANCE_MAX_CHARS = 800
VOICE_MAX_CHARS = 200
DESCRIPTION_MAX_CHARS = 800
AMBIENCE_MAX_CHARS = 400

CHARACTERS_DIRNAME = "characters"
SCENES_DIRNAME = "scenes"
CHARACTER_JSON_NAME = "character.json"
SCENE_JSON_NAME = "scene.json"
IMAGES_DIRNAME = "images"
AUDIO_DIRNAME = "audio"
# Socket-count caps (plan V2 section 2): these are OUR node limits, not H3's.
MAX_IMAGES = 8
MAX_AUDIO = 8
# Pre-V2 single files, still readable, deleted after a new-format save.
LEGACY_PORTRAIT_NAME = "portrait.png"
LEGACY_SCENE_IMAGE_NAME = "scene.png"
LEGACY_VOICE_SAMPLE_NAMES = ("voice_sample.wav", "voice_sample.mp3")

# Fixed marker returned by the cache-key helpers when the asset files are
# missing or unreadable, so a failed read is never cached as a success.
MISSING_CACHE_MARKER = "mie-asset-missing"

_WINDOWS_DEVICE_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {"COM%d" % i for i in range(1, 10)}
    | {"LPT%d" % i for i in range(1, 10)}
)

# Names allow CJK ideographs, ASCII letters, digits, '-' and '_' only. The
# whitelist already excludes separators, whitespace, control chars, dots and
# every other character the plan rejects.
_NAME_PATTERN = re.compile("[A-Za-z0-9_\\-\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF]+")

# The library location is FIXED per machine, not hand-typed per node (V2
# plan §3 + the 2026-10-10 ruling): the fixed root is the ``asset_library``
# SUBFOLDER OF THE PLUGIN itself (<plugin>/asset_library), overridable with
# $MIE_ASSET_LIBRARY. The pre-2026-10-10 default ~/mie_asset_library is
# migrated into the new location automatically, exactly once. The old
# "must live outside the plugin folder" rule is RETIRED: the mirror-sync
# script ignores this directory (never copied, never mirror-deleted), so
# library data inside the plugin is machine-local and safe.
LIBRARY_DIR_ENV = "MIE_ASSET_LIBRARY"
LIBRARY_DIRNAME = "asset_library"
LEGACY_HOME_DIRNAME = "mie_asset_library"


def fixed_library_root(plugin_root=None):
    """The machine's fixed library root: <plugin>/asset_library."""
    root = Path(plugin_root).resolve() if plugin_root is not None else PLUGIN_ROOT
    return root / LIBRARY_DIRNAME


def _migrate_legacy_home_library_once(target, plugin_root):
    """Move a pre-2026-10-10 ~/mie_asset_library into the plugin folder.

    Runs only when the new location does not exist yet and the legacy home
    library does. Any failure is swallowed — the fallback is a fresh empty
    library at the new location (assets are never destroyed by this move:
    shutil.move relocates, it does not merge or delete).
    """
    if target.exists():
        return
    legacy = Path(os.path.expanduser("~")) / LEGACY_HOME_DIRNAME
    if not legacy.is_dir():
        return
    if (Path(plugin_root).resolve() if plugin_root is not None else PLUGIN_ROOT) in legacy.parents:
        return  # paranoia: legacy somehow inside the plugin — impossible, skip
    try:
        import shutil

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(target))
    except Exception:
        pass


def resolve_library_dir(library_dir="", plugin_root=None):
    """Return the effective library dir string: the given one when non-blank
    (the API/tests escape hatch), else $MIE_ASSET_LIBRARY when set, else
    <plugin>/asset_library — migrating a legacy ~/mie_asset_library into
    place on first use. Pure string resolution — validation happens in
    validate_library_dir."""
    raw = "" if library_dir is None else str(library_dir).strip()
    if raw:
        return raw
    env = os.environ.get(LIBRARY_DIR_ENV, "").strip()
    if env:
        return env
    target = fixed_library_root(plugin_root)
    _migrate_legacy_home_library_once(target, plugin_root)
    return str(target)


def validate_library_dir(library_dir, plugin_root=None):
    """Return the resolved library root, or raise for relative paths.

    Accepts ``str`` or ``os.PathLike`` so an already-resolved ``Path`` can be
    re-validated (the batch loader validates once and then loops). A BLANK
    value is the "use the machine default" marker and resolves through
    resolve_library_dir first (inside the plugin folder, by design).
    """
    if isinstance(library_dir, os.PathLike):
        library_dir = os.fspath(library_dir)
    if not isinstance(library_dir, str) or not library_dir.strip():
        library_dir = resolve_library_dir(library_dir, plugin_root)
    raw = library_dir.strip()
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError(
            "library_dir must be an absolute path, got relative %r" % raw
        )
    return candidate.resolve()


def validate_name(name):
    """Return the stripped, validated name, or raise ValueError."""
    if not isinstance(name, str):
        raise ValueError("asset name must be a string, got %s" % type(name).__name__)
    n = name.strip()
    if not n:
        raise ValueError("asset name is empty after strip")
    if len(n) > NAME_MAX_CHARS:
        raise ValueError(
            "asset name longer than %d characters (%d given)"
            % (NAME_MAX_CHARS, len(n))
        )
    if _NAME_PATTERN.fullmatch(n) is None:
        raise ValueError(
            "invalid asset name %r: allowed are CJK characters, letters, digits, "
            "'-' and '_'" % n
        )
    if n.upper() in _WINDOWS_DEVICE_NAMES:
        raise ValueError("invalid asset name %r: reserved Windows device name" % n)
    return n


def validate_appearance(appearance):
    """Return the stripped appearance description (1-800 chars), or raise ValueError."""
    if not isinstance(appearance, str):
        raise ValueError(
            "appearance must be a string, got %s" % type(appearance).__name__
        )
    a = appearance.strip()
    if not a:
        raise ValueError("appearance is empty after strip")
    if len(a) > APPEARANCE_MAX_CHARS:
        raise ValueError(
            "appearance longer than %d characters (%d given)"
            % (APPEARANCE_MAX_CHARS, len(a))
        )
    return a


def validate_voice(voice):
    """Return the stripped voice line (1-200 chars), or raise ValueError."""
    if not isinstance(voice, str):
        raise ValueError("voice must be a string, got %s" % type(voice).__name__)
    v = voice.strip()
    if not v:
        raise ValueError("voice is empty after strip")
    if len(v) > VOICE_MAX_CHARS:
        raise ValueError(
            "voice longer than %d characters (%d given)" % (VOICE_MAX_CHARS, len(v))
        )
    return v


def validate_description(description):
    """Return the stripped scene description (1-800 chars), or raise ValueError.

    Same length class as ``appearance``: both replace a vision-model caption,
    so both are held to the captioner's original 60-100 word budget.
    """
    if not isinstance(description, str):
        raise ValueError(
            "description must be a string, got %s" % type(description).__name__
        )
    d = description.strip()
    if not d:
        raise ValueError("description is empty after strip")
    if len(d) > DESCRIPTION_MAX_CHARS:
        raise ValueError(
            "description longer than %d characters (%d given)"
            % (DESCRIPTION_MAX_CHARS, len(d))
        )
    return d


def validate_ambience(ambience):
    """Return the stripped scene ambience (0..400 chars), or raise ValueError.

    Empty / None is legal (the storyboard assembler falls back to its own
    three-sentence palette). Reject CJK so the zero-LLM assembler can rely
    on the field being English (it is inserted into the prompt verbatim).
    """
    if ambience is None:
        return ""
    if not isinstance(ambience, str):
        raise ValueError(
            "ambience must be a string, got %s" % type(ambience).__name__
        )
    a = ambience.strip()
    if len(a) > AMBIENCE_MAX_CHARS:
        raise ValueError(
            "ambience longer than %d characters (%d given)"
            % (AMBIENCE_MAX_CHARS, len(a))
        )
    if re.search(r"[\u3400-\u9FFF]", a):
        raise ValueError(
            "ambience must be English (no CJK characters); got %r" % a
        )
    return a


def _find_casefold_sibling(parent, wanted):
    """Return the entry in parent whose name casefolds equal to wanted, or None.

    Never uses exists(): on case-sensitive filesystems exists('tiger') misses an
    existing 'Tiger', so saving would create a second entry instead of
    overwriting.
    """
    if not parent.is_dir():
        return None
    wanted_folded = wanted.casefold()
    for entry in parent.iterdir():
        if entry.name.casefold() == wanted_folded:
            return entry
    return None


def _write_temp_file(data, directory):
    """Write data to a temp file in directory (same filesystem) and return its path."""
    fd, tmp_name = tempfile.mkstemp(prefix=".mie_asset_", dir=str(directory))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _read_asset_json(json_path, text_fields):
    """Read and shape-check an asset JSON file. Raises with the absolute path.

    ``text_fields`` is a tuple of required non-name string fields.
    """
    if isinstance(text_fields, str):
        text_fields = (text_fields,)
    if not json_path.is_file():
        raise FileNotFoundError("missing asset metadata file: %s" % json_path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError("%s is not valid JSON: %s" % (json_path, e)) from e
    if not isinstance(payload, dict):
        raise ValueError("%s must contain a JSON object" % json_path)
    missing = [f for f in text_fields if not isinstance(payload.get(f), str)]
    if not isinstance(payload.get("name"), str) or missing:
        raise ValueError(
            "%s must have string fields 'name' and %s"
            % (json_path, ", ".join(repr(f) for f in missing))
        )
    return payload


def _require_description(payload, json_path):
    """Return the description text, with the pre-V2 ``appearance`` field as
    the documented fallback (plan V2 section 4). Both spellings mean the
    same authored text; a save rewrites it as ``description``."""
    for field in ("description", "appearance"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(
        "%s has neither 'description' nor 'appearance'. Write one authored "
        "identity description into the JSON, or re-save the asset through "
        "the Save node." % json_path
    )


def _natural_key(name: str):
    """Sort key putting 2.png before 10.png (natural filename order)."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _png_dimensions(data: bytes):
    """(width, height) from the IHDR chunk — stdlib only, no Pillow needed
    to enforce the same-size rule at the storage layer."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _image_files(asset_dir: Path):
    """Ordered image list: images/*.png (V2), else the single legacy file.

    Returns list[(path, legacy: bool)] or raises a descriptive error for
    over-cap sets. Missing entirely -> ValueError (callers turn it into the
    right message)."""
    images_dir = asset_dir / IMAGES_DIRNAME
    if images_dir.is_dir():
        pngs = sorted(
            (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
            key=lambda p: _natural_key(p.name),
        )
        if not pngs:
            raise ValueError(
                "%s exists but holds no .png files" % images_dir
            )
        if len(pngs) > MAX_IMAGES:
            raise ValueError(
                "%s holds %d images; an asset supports at most %d"
                % (images_dir, len(pngs), MAX_IMAGES)
            )
        return [(p, False) for p in pngs]
    for legacy_name in (LEGACY_PORTRAIT_NAME, LEGACY_SCENE_IMAGE_NAME):
        legacy = asset_dir / legacy_name
        if legacy.is_file():
            return [(legacy, True)]
    raise ValueError("no images under %s (expected %s/*.png)" % (asset_dir, IMAGES_DIRNAME))


def _audio_files(char_dir: Path):
    """Ordered audio list: audio/*.{wav,mp3} (V2), else the legacy sample.

    Returns list[path]. Same index in both extensions is an ambiguous error.
    Zero clips is legal for a character."""
    audio_dir = char_dir / AUDIO_DIRNAME
    if audio_dir.is_dir():
        by_stem: dict = {}
        for p in sorted(
            (p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".mp3")),
            key=lambda p: _natural_key(p.name),
        ):
            by_stem.setdefault(p.stem, []).append(p)
        out = []
        problems = []
        for stem, paths in by_stem.items():
            if len(paths) > 1:
                problems.append(
                    "audio index %r has both %s" % (stem, " and ".join(p.name for p in paths))
                )
                continue
            out.append(paths[0])
        if problems:
            raise ValueError(
                "ambiguous audio under %s: %s" % (audio_dir, "; ".join(problems))
            )
        if len(out) > MAX_AUDIO:
            raise ValueError(
                "%s holds %d clips; an asset supports at most %d"
                % (audio_dir, len(out), MAX_AUDIO)
            )
        return out
    for legacy_name in LEGACY_VOICE_SAMPLE_NAMES:
        legacy = char_dir / legacy_name
        if legacy.is_file():
            return [legacy]
    return []


def _validate_same_size(entries):
    """All PNGs in one asset must share W x H — one IMAGE socket cannot hold
    two resolutions. The error names every file and its size."""
    sizes = {}
    for path, _legacy in entries:
        width, height = _png_dimensions(path.read_bytes())
        sizes.setdefault((width, height), []).append("%s (%dx%d)" % (path.name, width, height))
    if len(sizes) > 1:
        detail = "; ".join(
            "%dx%d: %s" % (w, h, ", ".join(names))
            for (w, h), names in sorted(sizes.items())
        )
        raise ValueError(
            "images of one asset must share one resolution, got %s. Export "
            "them at the same size — H3 resizes references to a 2048 short "
            "edge anyway" % detail
        )
    return next(iter(sizes))


def save_character(
    library_dir, name, description, voice,
    png_bytes_list=None, wav_bytes_list=None, plugin_root=None,
):
    """Write one character in the V2 layout (plan section 4).

    ``png_bytes_list`` / ``wav_bytes_list``: ``None`` = sockets left
    unconnected = KEEP the disk set byte-for-byte (editing one sentence
    never forces re-wiring). A list = replace the WHOLE set with these
    frames/clips (images 1..MAX_IMAGES, audio 0..MAX_AUDIO legal). A new
    character needs at least one image; audio may be zero.

    Commit shape: every payload that changes lands in temp files, files are
    replaced index by index, ``character.json`` is replaced LAST as the
    commit point, and only after a successful commit are the superseded
    files deleted (a replaced image set drops indices above the new count;
    the first new-format save also removes legacy portrait.png /
    voice_sample.*). Read-only access never deletes.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    d = validate_description(description)
    v = validate_voice(voice)

    if png_bytes_list is not None:
        if not isinstance(png_bytes_list, (list, tuple)) or not png_bytes_list:
            raise ValueError(
                "png_bytes_list must be a non-empty list of PNG bytes (None = "
                "keep the images on disk)"
            )
        if len(png_bytes_list) > MAX_IMAGES:
            raise ValueError(
                "%d images given; an asset supports at most %d"
                % (len(png_bytes_list), MAX_IMAGES)
            )
        png_bytes_list = [bytes(b) for b in png_bytes_list]
        dims = {_png_dimensions(b) for b in png_bytes_list}
        if len(dims) > 1:
            raise ValueError(
                "images of one asset must share one resolution, got %s"
                % ", ".join("%dx%d" % dim for dim in sorted(dims))
            )
    if wav_bytes_list is not None:
        if not isinstance(wav_bytes_list, (list, tuple)):
            raise ValueError("wav_bytes_list must be a list of WAV bytes (None = keep)")
        if len(wav_bytes_list) > MAX_AUDIO:
            raise ValueError(
                "%d audio clips given; an asset supports at most %d"
                % (len(wav_bytes_list), MAX_AUDIO)
            )
        wav_bytes_list = [bytes(b) for b in wav_bytes_list]

    characters_dir = root / CHARACTERS_DIRNAME
    characters_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_casefold_sibling(characters_dir, n)
    if existing is not None and not existing.is_dir():
        raise ValueError(
            "%s exists but is not a directory; cannot store character %r there"
            % (existing, n)
        )
    if existing is None:
        target_dir = characters_dir / n
        status = "created"
    else:
        target_dir = existing
        status = "overwritten"

    # Media resolution BEFORE mkdir of the asset dir — a failed validation
    # must not leave an empty asset directory behind.
    if png_bytes_list is None:
        if status == "created":
            raise ValueError(
                "a NEW character needs at least one image: connect an IMAGE "
                "batch to save %r for the first time (overwriting later "
                "without reconnecting keeps the images on disk)" % n
            )
        try:
            current_images = _image_files(target_dir)
        except ValueError as e:
            raise ValueError(
                "overwrite of %r without images, but the disk set is "
                "unreadable (%s); connect a new image batch" % (n, e)
            ) from e
        if not (target_dir / IMAGES_DIRNAME).is_dir():
            # Keep-on-disk over a PRE-V2 asset: promote the legacy file's
            # bytes to images/01.png so the post-commit retirement of
            # portrait.png cannot delete the only image. Byte-identical
            # content, new name.
            png_bytes_list = [current_images[0][0].read_bytes()]
    if wav_bytes_list is None and status == "created":
        pass  # zero audio is legal for a new character
    legacy_audio_promotion = None
    if wav_bytes_list is None and not (target_dir / AUDIO_DIRNAME).is_dir():
        for legacy_name in LEGACY_VOICE_SAMPLE_NAMES:
            legacy = target_dir / legacy_name
            if legacy.is_file():
                # Keep-on-disk over a PRE-V2 asset: copy the legacy sample
                # (its ORIGINAL extension — an .mp3 stays an .mp3) to
                # audio/01.<ext> so retiring voice_sample.* keeps the clip.
                legacy_audio_promotion = (legacy, "%02d%s" % (1, legacy.suffix))
                break

    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / IMAGES_DIRNAME
    audio_dir = target_dir / AUDIO_DIRNAME
    json_path = target_dir / CHARACTER_JSON_NAME
    json_payload = {"name": target_dir.name, "description": d, "voice": v}

    tmp_files = []  # (tmp_path, final_path)
    try:
        if png_bytes_list is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            for i, blob in enumerate(png_bytes_list, 1):
                tmp = _write_temp_file(blob, images_dir)
                tmp_files.append((tmp, images_dir / ("%02d.png" % i)))
        if wav_bytes_list is not None:
            audio_dir.mkdir(parents=True, exist_ok=True)
            for i, blob in enumerate(wav_bytes_list, 1):
                tmp = _write_temp_file(blob, audio_dir)
                tmp_files.append((tmp, audio_dir / ("%02d.wav" % i)))
        if legacy_audio_promotion is not None:
            src, dst_name = legacy_audio_promotion
            audio_dir.mkdir(parents=True, exist_ok=True)
            tmp = _write_temp_file(src.read_bytes(), audio_dir)
            tmp_files.append((tmp, audio_dir / dst_name))
        tmp_json = _write_temp_file(_json_bytes(json_payload), target_dir)
        replaced = []
        try:
            for tmp, final in tmp_files:
                os.replace(tmp, final)
                replaced.append(final)
            os.replace(tmp_json, json_path)
        except OSError as e:
            raise OSError(
                "character.json/files replacement failed (%s). Replaced so "
                "far: %s; saving %r again successfully rewrites everything."
                % (e, ", ".join(p.name for p in replaced) or "none", target_dir.name)
            ) from e
    finally:
        for tmp, _final in tmp_files:
            tmp.unlink(missing_ok=True)

    # Post-commit cleanup: superseded files only ever die after the JSON
    # landed, so a failed save cannot destroy the previous asset.
    if png_bytes_list is not None:
        keep = {"%02d.png" % i for i in range(1, len(png_bytes_list) + 1)}
        if images_dir.is_dir():
            for p in images_dir.iterdir():
                if p.is_file() and p.name not in keep:
                    p.unlink()
    if wav_bytes_list is not None or legacy_audio_promotion is not None:
        keep = {"%02d.wav" % i for i in range(1, (len(wav_bytes_list) if wav_bytes_list else 0) + 1)}
        if legacy_audio_promotion is not None:
            keep.add(legacy_audio_promotion[1])
        if audio_dir.is_dir():
            for p in audio_dir.iterdir():
                if p.is_file() and p.name not in keep:
                    p.unlink()
    # The first new-format save retires the legacy single files (their
    # content was promoted above whenever it should survive).
    for legacy in (LEGACY_PORTRAIT_NAME,) + LEGACY_VOICE_SAMPLE_NAMES:
        (target_dir / legacy).unlink(missing_ok=True)

    images = _image_files(target_dir)
    audios = _audio_files(target_dir)
    return {
        "name": target_dir.name,
        "description": d,
        "voice": v,
        "status": status,
        "png_bytes_list": [p.read_bytes() for p, _l in images],
        "audio_bytes_list": [p.read_bytes() for p in audios],
        "audio_exts": [p.suffix for p in audios],
    }



def load_character(library_dir, name, plugin_root=None):
    """Read one character (V2 multi-image / multi-audio, legacy-transparent).

    Every failure raises with the absolute path involved; mismatched image
    resolutions fail with every filename and its size named.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    characters_dir = root / CHARACTERS_DIRNAME
    target_dir = _find_casefold_sibling(characters_dir, n)
    if target_dir is None or not target_dir.is_dir():
        raise FileNotFoundError("character %r not found under %s" % (n, characters_dir))
    json_path = target_dir / CHARACTER_JSON_NAME
    payload = _read_asset_json(json_path, ("voice",))
    description = _require_description(payload, json_path)
    if payload["name"] != target_dir.name:
        raise ValueError(
            "%s declares name %r but the directory is %r; rename the directory or "
            "fix the JSON so both match" % (json_path, payload["name"], target_dir.name)
        )
    try:
        images = _image_files(target_dir)
        audios = _audio_files(target_dir)
        _validate_same_size(images)
    except ValueError as e:
        raise ValueError("character %r under %s: %s" % (target_dir.name, target_dir, e)) from e
    return {
        "name": payload["name"],
        "description": description,
        "voice": payload["voice"],
        "png_bytes_list": [p.read_bytes() for p, _legacy in images],
        "audio_bytes_list": [p.read_bytes() for p in audios],
        "audio_exts": [p.suffix for p in audios],
    }


def load_character_batch(library_dir, names, plugin_root=None):
    """Load several assets (characters AND scenes) in the GIVEN ORDER.

    The slot is each asset's 1-based position in the returned list, which is
    also its position in the IMAGE batch the caller will build — that
    ordering is the only contract between the asset library and the H3
    prompt generator, so a partial load is never tolerated: skipping one
    asset would shift every later slot and silently bind the wrong picture
    to the wrong name.
    """
    root = validate_library_dir(library_dir, plugin_root)
    if isinstance(names, str):
        candidates = [n.strip() for n in names.replace(",", "\n").splitlines()]
    else:
        candidates = [str(n).strip() for n in (names or [])]
    wanted = [n for n in candidates if n]
    if not wanted:
        raise ValueError(
            "names is empty; list one asset name per line, characters and "
            "scenes mixed (connect the names output of List Character Assets "
            "or List Scene Assets to fill it)"
        )
    entries = []
    seen: dict = {}
    for slot, name in enumerate(wanted, 1):
        if name in seen:
            raise ValueError(
                "asset %r is listed twice (slots %d and %d); every name must "
                "occupy exactly one picture slot"
                % (name, seen[name], slot)
            )
        seen[name] = slot
        # A name existing under BOTH characters/ and scenes/ is ambiguous:
        # picking one would let a scene silently consume a character's slot
        # (or the reverse), which is exactly the wrong-binding failure the
        # slot contract exists to prevent.
        char_dir = _find_casefold_sibling(root / CHARACTERS_DIRNAME, name)
        scene_dir = _find_casefold_sibling(root / SCENES_DIRNAME, name)
        if char_dir is not None and scene_dir is not None:
            raise ValueError(
                "asset name %r exists both as a character (%s) and as a scene "
                "(%s); rename one of them so every name means exactly one "
                "asset" % (name, char_dir, scene_dir)
            )
        # Re-validate the SAME plugin root, not the library root: passing
        # root as plugin_root would make every path "inside the plugin".
        if char_dir is not None:
            loaded = load_character(root, name, plugin_root=plugin_root)
            loaded["kind"] = "character"
            loaded["image_count"] = len(loaded["png_bytes_list"])
            loaded["audio_count"] = len(loaded["audio_bytes_list"])
            # The batch uses the FIRST image only (plan V2 section 6);
            # counts tell the roster consumer the rest exist.
            loaded["png_bytes"] = loaded["png_bytes_list"][0]
        else:
            loaded = load_scene(root, name, plugin_root=plugin_root)
            loaded["kind"] = "scene"
            loaded["image_count"] = len(loaded["png_bytes_list"])
            loaded["png_bytes"] = loaded["png_bytes_list"][0]
        loaded["slot"] = slot
        entries.append(loaded)
    return {"assets": entries}


def list_characters(library_dir, plugin_root=None):
    """List readable character names (one per line, sorted) plus a skip report."""
    root = validate_library_dir(library_dir, plugin_root)
    characters_dir = root / CHARACTERS_DIRNAME
    names = []
    skipped = []
    if not characters_dir.is_dir():
        return {"names": "", "status": "ok"}
    for entry in characters_dir.iterdir():
        if not entry.is_dir():
            continue
        reason = _character_entry_problem(entry)
        if reason is None:
            names.append(entry.name)
        else:
            skipped.append("skipped %s: %s" % (entry.name, reason))
    return {
        "names": "\n".join(sorted(names)),
        "status": "ok" if not skipped else "\n".join(skipped),
    }


def _asset_counts(asset_dir):
    """(image_count, audio_count) for the dropdown badges; raises on an
    unreadable set so the caller can report the reason."""
    images = _image_files(asset_dir)
    _validate_same_size(images)
    audio = _audio_files(asset_dir) if (asset_dir / AUDIO_DIRNAME).is_dir() or any(
        (asset_dir / name).is_file() for name in LEGACY_VOICE_SAMPLE_NAMES
    ) else []
    return len(images), len(audio)


def list_assets_payload(library_dir, plugin_root=None):
    """One scan for the frontend dropdowns: names + per-asset counts, plus
    skip notes. Never raises — an unusable library becomes ``{"ok": False,
    "error": ...}`` so the dropdown shows the reason instead of guessing.

    Names and counts only (plan V2 section 8); no file contents leave the
    machine, images are not decoded.
    """
    try:
        root = validate_library_dir(library_dir, plugin_root=plugin_root)
        chars = list_characters(str(root), plugin_root=plugin_root)
        scenes = list_scenes(str(root), plugin_root=plugin_root)
    except (ValueError, OSError) as e:
        return {"ok": False, "characters": [], "scenes": [], "error": str(e)}
    char_items = []
    for name in [n for n in chars["names"].splitlines() if n]:
        asset_dir = _find_casefold_sibling(root / CHARACTERS_DIRNAME, name)
        try:
            image_count, audio_count = _asset_counts(asset_dir)
        except (ValueError, OSError) as e:
            char_items.append({"name": name, "image_count": 0, "audio_count": 0,
                               "error": str(e)})
            continue
        char_items.append({"name": name, "image_count": image_count,
                           "audio_count": audio_count})
    scene_items = []
    for name in [n for n in scenes["names"].splitlines() if n]:
        asset_dir = _find_casefold_sibling(root / SCENES_DIRNAME, name)
        try:
            image_count, _audio = _asset_counts(asset_dir)
        except (ValueError, OSError) as e:
            scene_items.append({"name": name, "image_count": 0, "error": str(e)})
            continue
        scene_items.append({"name": name, "image_count": image_count})
    return {
        "ok": True,
        # Echoed so the frontend can show WHERE the fixed library resolved
        # to (blank library_dir = machine default).
        "library_dir": str(root),
        "characters": char_items,
        "scenes": scene_items,
        "error": "; ".join(
            part for part in (chars["status"] if chars["status"] != "ok" else "",
                              scenes["status"] if scenes["status"] != "ok" else "")
            if part
        ) or None,
    }


def read_picture(library_dir, name, kind, plugin_root=None):
    """Return the PNG bytes of an asset's FIRST image (the thumbnail source).

    Both the library dir and the name go through their validators first —
    the name whitelist (CJK/letters/digits/-/_) structurally excludes every
    path separator, so there is no traversal case to chase.
    """
    root = validate_library_dir(library_dir, plugin_root=plugin_root)
    n = validate_name(name)
    if kind == "character":
        target = _find_casefold_sibling(root / CHARACTERS_DIRNAME, n)
    elif kind == "scene":
        target = _find_casefold_sibling(root / SCENES_DIRNAME, n)
    else:
        raise ValueError("kind must be 'character' or 'scene', got %r" % kind)
    if target is None or not target.is_dir():
        raise FileNotFoundError("%s %r not found under %s" % (kind, n, root))
    try:
        images = _image_files(target)
    except ValueError as e:
        raise FileNotFoundError("%s %r: %s" % (kind, n, e)) from e
    return images[0][0].read_bytes()


def read_asset_meta(library_dir, name, kind, plugin_root=None):
    """One asset's text + counts + first-image stamp for the save autofill.

    The save dropdown loads this when an EXISTING name is picked, so the
    widgets show what is actually about to be overwritten. The picture stamp
    (size, mtime_ns) doubles as the thumbnail cache key.
    """
    root = validate_library_dir(library_dir, plugin_root=plugin_root)
    n = validate_name(name)
    if kind == "character":
        target = _find_casefold_sibling(root / CHARACTERS_DIRNAME, n)
        if target is None or not target.is_dir():
            raise FileNotFoundError(
                "character %r not found under %s" % (n, root / CHARACTERS_DIRNAME)
            )
        payload = _read_asset_json(target / CHARACTER_JSON_NAME, ("voice",))
        description = _require_description(payload, target / CHARACTER_JSON_NAME)
        meta = {"kind": "character", "name": target.name,
                "description": description, "voice": payload["voice"]}
    elif kind == "scene":
        target = _find_casefold_sibling(root / SCENES_DIRNAME, n)
        if target is None or not target.is_dir():
            raise FileNotFoundError(
                "scene %r not found under %s" % (n, root / SCENES_DIRNAME)
            )
        payload = _read_asset_json(target / SCENE_JSON_NAME, ("description",))
        raw_ambience = payload.get("ambience")
        ambience = validate_ambience(raw_ambience) if isinstance(raw_ambience, str) else ""
        meta = {"kind": "scene", "name": target.name,
                "description": payload["description"], "ambience": ambience}
    else:
        raise ValueError("kind must be 'character' or 'scene', got %r" % kind)
    try:
        images = _image_files(target)
        _validate_same_size(images)
    except ValueError as e:
        raise ValueError("%s %r under %s: %s" % (kind, target.name, target, e)) from e
    first_image = images[0][0]
    stat = first_image.stat()
    meta["image_count"] = len(images)
    meta["audio_count"] = len(_audio_files(target))
    meta["picture_size"] = stat.st_size
    meta["picture_mtime_ns"] = stat.st_mtime_ns
    return meta


def _character_entry_problem(entry):
    try:
        validate_name(entry.name)
    except ValueError as e:
        return str(e)
    json_path = entry / CHARACTER_JSON_NAME
    if not json_path.is_file():
        return "missing %s" % CHARACTER_JSON_NAME
    try:
        payload = _read_asset_json(json_path, ("voice",))
        _require_description(payload, json_path)
    except (ValueError, OSError) as e:
        return str(e)
    if payload["name"] != entry.name:
        return "JSON name %r does not match directory name %r" % (
            payload["name"],
            entry.name,
        )
    try:
        images = _image_files(entry)
        _validate_same_size(images)
        _audio_files(entry)
    except (ValueError, OSError) as e:
        return str(e)
    return None


def save_scene(library_dir, name, description, png_bytes_list=None,
              ambience=None, plugin_root=None):
    """Write one scene in the V2 layout. Same keep/replace commit shape as
    save_character, minus audio (scenes have none).

    ``ambience`` is the optional one-line English mood sentence used by the
    zero-LLM storyboard assembler (PLAN_STORYBOARD_ASSET.md §3.2). Pass
    ``None`` to PRESERVE the on-disk value; pass ``""`` (empty) to clear it.
    On a brand-new scene, ``None`` or ``""`` writes no ambience field; a
    non-empty string writes it.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    d = validate_description(description)

    if png_bytes_list is not None:
        if not isinstance(png_bytes_list, (list, tuple)) or not png_bytes_list:
            raise ValueError(
                "png_bytes_list must be a non-empty list of PNG bytes (None = "
                "keep the images on disk)"
            )
        if len(png_bytes_list) > MAX_IMAGES:
            raise ValueError(
                "%d images given; an asset supports at most %d"
                % (len(png_bytes_list), MAX_IMAGES)
            )
        png_bytes_list = [bytes(b) for b in png_bytes_list]
        dims = {_png_dimensions(b) for b in png_bytes_list}
        if len(dims) > 1:
            raise ValueError(
                "images of one asset must share one resolution, got %s"
                % ", ".join("%dx%d" % dim for dim in sorted(dims))
            )

    scenes_dir = root / SCENES_DIRNAME
    scenes_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_casefold_sibling(scenes_dir, n)
    if existing is not None and not existing.is_dir():
        raise ValueError(
            "%s exists but is not a directory; cannot store scene %r there"
            % (existing, n)
        )
    if existing is None:
        target_dir = scenes_dir / n
        status = "created"
    else:
        target_dir = existing
        status = "overwritten"

    if png_bytes_list is None:
        if status == "created":
            raise ValueError(
                "a NEW scene needs at least one image: connect an IMAGE batch "
                "to save %r for the first time (overwriting later without "
                "reconnecting keeps the images on disk)" % n
            )
        try:
            current_images = _image_files(target_dir)
        except ValueError as e:
            raise ValueError(
                "overwrite of %r without images, but the disk set is "
                "unreadable (%s); connect a new image batch" % (n, e)
            ) from e
        if not (target_dir / IMAGES_DIRNAME).is_dir():
            # Keep-on-disk over a PRE-V2 scene: promote the legacy picture's
            # bytes to images/01.png so retiring scene.png cannot delete the
            # only image (same promotion as a character's portrait).
            png_bytes_list = [current_images[0][0].read_bytes()]

    # Resolve the on-disk ambience when None is passed (keep the disc value),
    # so the save roundtrip never silently drops an authored field.
    resolved_ambience = None
    if ambience is None and status == "overwritten":
        existing_json = target_dir / SCENE_JSON_NAME
        if existing_json.is_file():
            try:
                existing_payload = json.loads(
                    existing_json.read_text(encoding="utf-8")
                )
                if isinstance(existing_payload, dict):
                    prev = existing_payload.get("ambience")
                    if isinstance(prev, str):
                        resolved_ambience = prev
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    elif ambience is not None:
        resolved_ambience = validate_ambience(ambience)

    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / IMAGES_DIRNAME
    json_path = target_dir / SCENE_JSON_NAME
    json_payload = {"name": target_dir.name, "description": d}
    if resolved_ambience:
        json_payload["ambience"] = resolved_ambience

    tmp_files = []
    try:
        if png_bytes_list is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            for i, blob in enumerate(png_bytes_list, 1):
                tmp = _write_temp_file(blob, images_dir)
                tmp_files.append((tmp, images_dir / ("%02d.png" % i)))
        tmp_json = _write_temp_file(_json_bytes(json_payload), target_dir)
        try:
            for tmp, final in tmp_files:
                os.replace(tmp, final)
            os.replace(tmp_json, json_path)
        except OSError as e:
            raise OSError(
                "scene.json/files replacement failed (%s); saving %r again "
                "successfully rewrites everything." % (e, target_dir.name)
            ) from e
    finally:
        for tmp, _final in tmp_files:
            tmp.unlink(missing_ok=True)

    if png_bytes_list is not None:
        keep = {"%02d.png" % i for i in range(1, len(png_bytes_list) + 1)}
        if images_dir.is_dir():
            for p in images_dir.iterdir():
                if p.is_file() and p.name not in keep:
                    p.unlink()
    (target_dir / LEGACY_SCENE_IMAGE_NAME).unlink(missing_ok=True)

    images = _image_files(target_dir)
    return {
        "name": target_dir.name,
        "description": d,
        "ambience": resolved_ambience or "",
        "status": status,
        "png_bytes_list": [p.read_bytes() for p, _l in images],
    }



def load_scene(library_dir, name, plugin_root=None):
    """Read one scene (multi-image, legacy scene.png = image 1).

    Returns ``ambience``: the optional one-line English mood sentence
    authored for the zero-LLM storyboard assembler. Old scenes without the
    key load transparently with ``ambience = ""``.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    scenes_dir = root / SCENES_DIRNAME
    target_dir = _find_casefold_sibling(scenes_dir, n)
    if target_dir is None or not target_dir.is_dir():
        raise FileNotFoundError("scene %r not found under %s" % (n, scenes_dir))
    json_path = target_dir / SCENE_JSON_NAME
    payload = _read_asset_json(json_path, ("description",))
    if payload["name"] != target_dir.name:
        raise ValueError(
            "%s declares name %r but the directory is %r; rename the directory "
            "or fix the JSON so both match"
            % (json_path, payload["name"], target_dir.name)
        )
    raw_ambience = payload.get("ambience")
    ambience = validate_ambience(raw_ambience) if isinstance(raw_ambience, str) else ""
    try:
        images = _image_files(target_dir)
        _validate_same_size(images)
    except ValueError as e:
        raise ValueError("scene %r under %s: %s" % (target_dir.name, target_dir, e)) from e
    return {
        "name": payload["name"],
        "description": payload["description"],
        "ambience": ambience,
        "png_bytes_list": [p.read_bytes() for p, _legacy in images],
    }


def list_scenes(library_dir, plugin_root=None):
    """List readable scene names (one per line, sorted) plus a skip report."""
    root = validate_library_dir(library_dir, plugin_root)
    scenes_dir = root / SCENES_DIRNAME
    names = []
    skipped = []
    if not scenes_dir.is_dir():
        return {"names": "", "status": "ok"}
    for entry in scenes_dir.iterdir():
        if not entry.is_dir():
            continue
        reason = _scene_entry_problem(entry)
        if reason is None:
            names.append(entry.name)
        else:
            skipped.append("skipped %s: %s" % (entry.name, reason))
    return {
        "names": "\n".join(sorted(names)),
        "status": "ok" if not skipped else "\n".join(skipped),
    }


def _scene_entry_problem(entry):
    try:
        validate_name(entry.name)
    except ValueError as e:
        return str(e)
    json_path = entry / SCENE_JSON_NAME
    if not json_path.is_file():
        return "missing %s" % SCENE_JSON_NAME
    try:
        payload = _read_asset_json(json_path, "description")
    except (ValueError, OSError) as e:
        return str(e)
    if payload["name"] != entry.name:
        return "JSON name %r does not match directory name %r" % (
            payload["name"],
            entry.name,
        )
    try:
        images = _image_files(entry)
        _validate_same_size(images)
    except (ValueError, OSError) as e:
        return str(e)
    return None


def _hash_asset_files(target_dir):
    """sha256 over every file of an asset in natural order: JSON, all
    images, all audio (plan V2 section 5)."""
    digest = hashlib.sha256()
    for json_name in (CHARACTER_JSON_NAME, SCENE_JSON_NAME):
        json_path = target_dir / json_name
        if json_path.is_file():
            digest.update(json_path.name.encode("utf-8"))
            digest.update(json_path.read_bytes())
            break
    for path, _legacy in _image_files(target_dir):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    for path in _audio_files(target_dir):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def character_cache_key(library_dir, name, plugin_root=None):
    """Content hash of the whole character (JSON + every image + every
    audio clip) for IS_CHANGED. Never raises: any problem (missing,
    unreadable, invalid) returns the fixed MISSING_CACHE_MARKER so the
    failure is not cached as a success.
    """
    try:
        root = validate_library_dir(library_dir, plugin_root)
        n = validate_name(name)
        target_dir = _find_casefold_sibling(root / CHARACTERS_DIRNAME, n)
        if target_dir is None or not target_dir.is_dir():
            return MISSING_CACHE_MARKER
        return _hash_asset_files(target_dir)
    except Exception:
        return MISSING_CACHE_MARKER


def asset_batch_cache_key(library_dir, names, plugin_root=None):
    """Combined content hash for a mixed batch load, in slot order.

    Any file of any listed asset changing changes the key, and so does
    reordering the names. Never raises: a failure returns the fixed
    MISSING_CACHE_MARKER, and a re-run repeats the same failure loudly
    instead of reusing a stale batch.
    """
    try:
        if isinstance(names, str):
            candidates = [n.strip() for n in names.replace(",", "\n").splitlines()]
        else:
            candidates = [str(n).strip() for n in (names or [])]
        wanted = [n for n in candidates if n]
        if not wanted:
            return MISSING_CACHE_MARKER
        digest = hashlib.sha256()
        for name in wanted:
            if (
                _find_casefold_sibling(
                    validate_library_dir(library_dir, plugin_root) / CHARACTERS_DIRNAME,
                    name,
                )
                is not None
            ):
                key = character_cache_key(library_dir, name, plugin_root=plugin_root)
            else:
                key = scene_cache_key(library_dir, name, plugin_root=plugin_root)
            if key == MISSING_CACHE_MARKER:
                return MISSING_CACHE_MARKER
            digest.update(name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(key.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()
    except Exception:
        return MISSING_CACHE_MARKER


def scene_cache_key(library_dir, name, plugin_root=None):
    """Content hash of the whole scene (JSON + every image). Never raises."""
    try:
        root = validate_library_dir(library_dir, plugin_root)
        n = validate_name(name)
        target_dir = _find_casefold_sibling(root / SCENES_DIRNAME, n)
        if target_dir is None or not target_dir.is_dir():
            return MISSING_CACHE_MARKER
        return _hash_asset_files(target_dir)
    except Exception:
        return MISSING_CACHE_MARKER


def delete_character(library_dir, name, plugin_root=None):
    """Delete one character asset (the whole characters/<name> directory).

    Raises FileNotFoundError when the character does not exist. Returns a
    payload describing what was removed.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    characters_dir = root / CHARACTERS_DIRNAME
    target_dir = _find_casefold_sibling(characters_dir, n)
    if target_dir is None or not target_dir.is_dir():
        raise FileNotFoundError(
            "character %r not found under %s (nothing deleted)" % (n, characters_dir)
        )
    files = [p for p in target_dir.rglob("*") if p.is_file()]
    shutil.rmtree(target_dir)
    return {
        "ok": True,
        "kind": "character",
        "name": target_dir.name,
        "deleted_files": len(files),
        "deleted_dir": str(target_dir),
    }


def delete_scene(library_dir, name, plugin_root=None):
    """Delete one scene asset (the whole scenes/<name> directory)."""
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    scenes_dir = root / SCENES_DIRNAME
    target_dir = _find_casefold_sibling(scenes_dir, n)
    if target_dir is None or not target_dir.is_dir():
        raise FileNotFoundError(
            "scene %r not found under %s (nothing deleted)" % (n, scenes_dir)
        )
    files = [p for p in target_dir.rglob("*") if p.is_file()]
    shutil.rmtree(target_dir)
    return {
        "ok": True,
        "kind": "scene",
        "name": target_dir.name,
        "deleted_files": len(files),
        "deleted_dir": str(target_dir),
    }
