# -*- coding: utf-8 -*-
"""On-disk character & scene asset library (docs/PLAN_ASSET_LIBRARY.md).

Pure filesystem layer: validation, atomic writes, reads, listing and content
hashing. No ComfyUI, no torch, no nodes.llm — images travel as PNG bytes and
tensor conversion stays in nodes.py.

Library layout:

    <library_dir>/characters/<name>/character.json   name + appearance + voice
    <library_dir>/characters/<name>/portrait.png
    <library_dir>/characters/<name>/voice_sample.wav   (or a hand-dropped .mp3)
    <library_dir>/scenes/<name>/scene.json          name + description
    <library_dir>/scenes/<name>/scene.png

``appearance``, ``voice`` and ``description`` are authored English text
inserted into the H3 prompt verbatim — they replace the vision-model caption
a pipeline would otherwise spend a multimodal call to produce. A character is
a picture plus two descriptions; a scene is the same minus the voice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

# Plugin package root: nodes/assets/library.py -> nodes/assets -> nodes -> root.
PLUGIN_ROOT = Path(__file__).resolve().parents[2]

NAME_MAX_CHARS = 40
APPEARANCE_MAX_CHARS = 800
VOICE_MAX_CHARS = 200
DESCRIPTION_MAX_CHARS = 800

CHARACTERS_DIRNAME = "characters"
SCENES_DIRNAME = "scenes"
CHARACTER_JSON_NAME = "character.json"
PORTRAIT_NAME = "portrait.png"
SCENE_JSON_NAME = "scene.json"
SCENE_IMAGE_NAME = "scene.png"
# The node writes voice_sample.wav (PCM16). A hand-dropped voice_sample.mp3 is
# read as-is; if both exist the load fails loudly instead of guessing.
VOICE_SAMPLE_NAMES = ("voice_sample.wav", "voice_sample.mp3")

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

# The library location is FIXED per machine, not hand-typed per node (sixth
# revision): blank library_dir resolves to $MIE_ASSET_LIBRARY when set, else
# ~/mie_asset_library. A non-blank value still wins — it is the escape hatch
# older workflows already depend on.
LIBRARY_DIR_ENV = "MIE_ASSET_LIBRARY"


def resolve_library_dir(library_dir=""):
    """Return the effective library dir string: the given one when non-blank,
    else the machine's fixed default ($MIE_ASSET_LIBRARY, then
    ``~/mie_asset_library``). Pure string resolution — validation happens
    in validate_library_dir as before."""
    raw = "" if library_dir is None else str(library_dir).strip()
    if raw:
        return raw
    env = os.environ.get(LIBRARY_DIR_ENV, "").strip()
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), "mie_asset_library")


def validate_library_dir(library_dir, plugin_root=None):
    """Return the resolved library root, or raise for empty/relative/inside-plugin paths.

    Accepts ``str`` or ``os.PathLike`` so an already-resolved ``Path`` can be
    re-validated (the batch loader validates once and then loops). A BLANK
    value is the "use the machine default" marker and resolves through
    resolve_library_dir first.
    """
    if isinstance(library_dir, os.PathLike):
        library_dir = os.fspath(library_dir)
    if not isinstance(library_dir, str) or not library_dir.strip():
        library_dir = resolve_library_dir(library_dir)
    raw = library_dir.strip()
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError(
            "library_dir must be an absolute path, got relative %r" % raw
        )
    resolved = candidate.resolve()
    root = Path(plugin_root).resolve() if plugin_root is not None else PLUGIN_ROOT
    resolved_str = os.path.normcase(str(resolved))
    root_str = os.path.normcase(str(root))
    if resolved_str == root_str or resolved_str.startswith(root_str + os.sep):
        raise ValueError(
            "library_dir %s is inside the plugin folder %s; the plugin folder is "
            "mirror-synced by scripts/sync_to_comfyui.py and library files inside "
            "it would be deleted on the next sync — put the library outside the "
            "plugin folder" % (resolved, root)
        )
    return resolved


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


def _require_appearance(payload, json_path):
    """Return the appearance text, or raise a migration-specific error.

    A character written before the appearance field existed cannot fall back
    to a machine caption: mixing an authored identity and a generated one in
    one plan is exactly the two-contradictory-identities failure the identity
    pin exists to prevent. So this fails loudly and says how to fix it.
    """
    appearance = payload.get("appearance")
    if isinstance(appearance, str) and appearance.strip():
        return appearance
    raise ValueError(
        "%s has no 'appearance' field. Write one English identity description "
        "for this character into the JSON (60-100 words: who they are, what "
        "they wear, what a reference image must preserve) or re-save the "
        "character through Save Character Asset." % json_path
    )


def _find_voice_sample(char_dir):
    """Return the single voice_sample file present, or None if missing.

    Raises when both .wav and .mp3 exist — that state is ambiguous, and the
    reader must not guess.
    """
    found = [char_dir / fname for fname in VOICE_SAMPLE_NAMES if (char_dir / fname).is_file()]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            "ambiguous voice sample: both %s exist; keep exactly one"
            % " and ".join(str(p) for p in found)
        )
    return found[0]


def save_character(
    library_dir, name, appearance, voice, png_bytes, voice_sample_wav, plugin_root=None
):
    """Write one character (name + appearance + voice + portrait PNG + sample).

    ``png_bytes`` / ``voice_sample_wav`` may be ``None`` — that means "the
    socket was left unconnected", and is only legal when OVERWRITING: the
    file already on disk is kept byte-for-byte, so editing one sentence
    never forces re-wiring a portrait and a voice sample. A new character
    still requires both.

    Fixed order (plan section 3): write the temp files for every payload
    that changes, replace portrait.png, replace voice_sample.wav (removing
    a superseded hand-dropped .mp3), then replace character.json as the
    commit point. A failure leaves the documented partial state and raises
    — the save never reports created/overwritten unless everything landed.
    Kept files are not rewritten at all; their failure windows vanish.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    a = validate_appearance(appearance)
    v = validate_voice(voice)
    # Payload TYPE validation before any mkdir, so a bad call leaves no
    # half-created directories behind.
    for field, payload in (("png_bytes", png_bytes), ("voice_sample_wav", voice_sample_wav)):
        if payload is not None and not (isinstance(payload, (bytes, bytearray)) and payload):
            raise ValueError(
                "%s must be non-empty bytes or None (None = keep the file on "
                "disk when overwriting)" % field
            )

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

    # Resolve the media payloads BEFORE creating anything: a failed
    # validation must not leave an empty asset directory behind (a later
    # attempt would then look like an overwrite).
    portrait_path = target_dir / PORTRAIT_NAME
    if png_bytes is None:
        if status == "created":
            raise ValueError(
                "a NEW character needs a portrait: connect an image to save "
                "%r for the first time (overwriting later without reconnecting "
                "keeps the picture on disk)" % n
            )
        if not portrait_path.is_file():
            raise FileNotFoundError(
                "overwrite of %r without an image, but %s is missing; connect "
                "a new portrait" % (n, portrait_path)
            )
    else:
        png_bytes = bytes(png_bytes)

    sample_path = target_dir / VOICE_SAMPLE_NAMES[0]
    existing_sample = None
    if voice_sample_wav is None:
        if status == "created":
            raise ValueError(
                "a NEW character needs a voice sample: connect AUDIO to save "
                "%r for the first time (overwriting later without reconnecting "
                "keeps the sample on disk)" % n
            )
        existing_sample = _find_voice_sample(target_dir)
        if existing_sample is None:
            raise FileNotFoundError(
                "overwrite of %r without AUDIO, but no voice sample exists "
                "under %s; connect a new sample" % (n, target_dir)
            )
    else:
        voice_sample_wav = bytes(voice_sample_wav)

    target_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {"name": target_dir.name, "appearance": a, "voice": v}
    json_path = target_dir / CHARACTER_JSON_NAME

    tmp_png = None
    tmp_sample = None
    tmp_json = None
    tmp_png_was_written = False
    tmp_sample_was_written = False
    try:
        if png_bytes is not None:
            tmp_png = _write_temp_file(png_bytes, target_dir)
        if voice_sample_wav is not None:
            tmp_sample = _write_temp_file(voice_sample_wav, target_dir)
        tmp_json = _write_temp_file(_json_bytes(json_payload), target_dir)
        if tmp_png is not None:
            try:
                os.replace(tmp_png, portrait_path)
            except OSError as e:
                raise OSError(
                    "failed to replace %s (%s); all target files are unchanged"
                    % (portrait_path, e)
                ) from e
            tmp_png = None
            tmp_png_was_written = True
        if tmp_sample is not None:
            try:
                os.replace(tmp_sample, sample_path)
                stale_mp3 = target_dir / VOICE_SAMPLE_NAMES[1]
                if stale_mp3.is_file():
                    # superseded by the freshly written .wav; keeping both
                    # would make every later load ambiguous
                    stale_mp3.unlink()
            except OSError as e:
                raise OSError(
                    "portrait.png was replaced with the new image, but updating "
                    "the voice sample failed (%s). character.json and the "
                    "previous voice sample are unchanged." % e
                ) from e
            tmp_sample = None
            tmp_sample_was_written = True
        try:
            os.replace(tmp_json, json_path)
        except OSError as e:
            media_state = " and ".join(
                part
                for flag, part in (
                    (tmp_png_was_written, "the NEW portrait"),
                    (tmp_sample_was_written, "the NEW voice sample"),
                )
                if flag
            ) or "the files on disk (nothing was rewritten)"
            raise OSError(
                "character.json replacement failed (%s). The library now holds "
                "%s beside the OLD texts; saving %r again successfully will "
                "rewrite everything." % (e, media_state, target_dir.name)
            ) from e
        tmp_json = None
    finally:
        if tmp_png is not None:
            tmp_png.unlink(missing_ok=True)
        if tmp_sample is not None:
            tmp_sample.unlink(missing_ok=True)
        if tmp_json is not None:
            tmp_json.unlink(missing_ok=True)

    kept_sample = existing_sample if existing_sample is not None else sample_path
    return {
        "name": target_dir.name,
        "appearance": a,
        "voice": v,
        "status": status,
        "png_bytes": portrait_path.read_bytes(),
        "voice_sample_bytes": kept_sample.read_bytes(),
        "voice_sample_ext": kept_sample.suffix,
    }


def load_character(library_dir, name, plugin_root=None):
    """Read one character. Every failure raises with the absolute path involved."""
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    characters_dir = root / CHARACTERS_DIRNAME
    target_dir = _find_casefold_sibling(characters_dir, n)
    if target_dir is None or not target_dir.is_dir():
        raise FileNotFoundError("character %r not found under %s" % (n, characters_dir))
    json_path = target_dir / CHARACTER_JSON_NAME
    portrait_path = target_dir / PORTRAIT_NAME
    payload = _read_asset_json(json_path, ("appearance", "voice"))
    appearance = _require_appearance(payload, json_path)
    if payload["name"] != target_dir.name:
        raise ValueError(
            "%s declares name %r but the directory is %r; rename the directory or "
            "fix the JSON so both match" % (json_path, payload["name"], target_dir.name)
        )
    if not portrait_path.is_file():
        raise FileNotFoundError(
            "missing portrait for character %r: %s" % (target_dir.name, portrait_path)
        )
    sample = _find_voice_sample(target_dir)
    if sample is None:
        raise FileNotFoundError(
            "missing voice sample for character %r: expected one of %s under %s"
            % (
                target_dir.name,
                " or ".join(VOICE_SAMPLE_NAMES),
                target_dir,
            )
        )
    return {
        "name": payload["name"],
        "appearance": appearance,
        "voice": payload["voice"],
        "png_bytes": portrait_path.read_bytes(),
        "voice_sample_bytes": sample.read_bytes(),
        "voice_sample_ext": sample.suffix,
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
        else:
            loaded = load_scene(root, name, plugin_root=plugin_root)
            loaded["kind"] = "scene"
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


def list_assets_payload(library_dir, plugin_root=None):
    """One scan for the frontend dropdowns: both kinds, plus per-kind skip
    notes. Never raises — an unusable library_dir becomes ``{"ok": False,
    "error": ...}`` so the dropdown can show the reason instead of guessing.

    Returns only names and kinds; no file contents leave the machine.
    """
    try:
        root = validate_library_dir(library_dir, plugin_root=plugin_root)
        chars = list_characters(str(root), plugin_root=plugin_root)
        scenes = list_scenes(str(root), plugin_root=plugin_root)
    except (ValueError, OSError) as e:
        return {"ok": False, "characters": [], "scenes": [], "error": str(e)}
    error_parts = []
    if chars["status"] != "ok":
        error_parts.append(chars["status"])
    if scenes["status"] != "ok":
        error_parts.append(scenes["status"])
    return {
        "ok": True,
        # Echoed so the frontend can show WHERE the fixed library resolved
        # to (blank library_dir = machine default).
        "library_dir": str(root),
        "characters": [n for n in chars["names"].splitlines() if n],
        "scenes": [n for n in scenes["names"].splitlines() if n],
        "error": "; ".join(error_parts) or None,
    }


def read_picture(library_dir, name, kind, plugin_root=None):
    """Return the PNG bytes of one asset's picture for the picker cards.

    ``kind`` is ``character`` (portrait.png) or ``scene`` (scene.png). Both
    the library dir and the name go through their validators first — the
    name whitelist (CJK/letters/digits/-/_) structurally excludes every
    path separator, so there is no traversal case to chase.
    """
    root = validate_library_dir(library_dir, plugin_root=plugin_root)
    n = validate_name(name)
    if kind == "character":
        target = _find_casefold_sibling(root / CHARACTERS_DIRNAME, n)
        path = target / PORTRAIT_NAME if target else None
    elif kind == "scene":
        target = _find_casefold_sibling(root / SCENES_DIRNAME, n)
        path = target / SCENE_IMAGE_NAME if target else None
    else:
        raise ValueError("kind must be 'character' or 'scene', got %r" % kind)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "%s %r has no picture under %s" % (kind, n, root)
        )
    return path.read_bytes()


def read_asset_meta(library_dir, name, kind, plugin_root=None):
    """Return one asset's text + picture stamp for the save-node autofill.

    The save dropdown loads this when an EXISTING name is picked, so the
    widgets show what is actually about to be overwritten — never blanks,
    never the previous character's text. The picture stamp (size, mtime_ns)
    doubles as the thumbnail cache key: a URL carrying it goes stale exactly
    when the picture changes.
    """
    root = validate_library_dir(library_dir, plugin_root=plugin_root)
    n = validate_name(name)
    if kind == "character":
        target = _find_casefold_sibling(root / CHARACTERS_DIRNAME, n)
        if target is None or not target.is_dir():
            raise FileNotFoundError(
                "character %r not found under %s" % (n, root / CHARACTERS_DIRNAME)
            )
        payload = _read_asset_json(target / CHARACTER_JSON_NAME, ("appearance", "voice"))
        _require_appearance(payload, target / CHARACTER_JSON_NAME)
        picture = target / PORTRAIT_NAME
        meta = {"kind": "character", "name": target.name,
                "appearance": payload["appearance"], "voice": payload["voice"]}
    elif kind == "scene":
        target = _find_casefold_sibling(root / SCENES_DIRNAME, n)
        if target is None or not target.is_dir():
            raise FileNotFoundError(
                "scene %r not found under %s" % (n, root / SCENES_DIRNAME)
            )
        payload = _read_asset_json(target / SCENE_JSON_NAME, ("description",))
        picture = target / SCENE_IMAGE_NAME
        meta = {"kind": "scene", "name": target.name,
                "description": payload["description"]}
    else:
        raise ValueError("kind must be 'character' or 'scene', got %r" % kind)
    if not picture.is_file():
        raise FileNotFoundError("missing picture for %r: %s" % (meta["name"], picture))
    stat = picture.stat()
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
        payload = _read_asset_json(json_path, ("appearance", "voice"))
        _require_appearance(payload, json_path)
    except (ValueError, OSError) as e:
        return str(e)
    if payload["name"] != entry.name:
        return "JSON name %r does not match directory name %r" % (
            payload["name"],
            entry.name,
        )
    if not (entry / PORTRAIT_NAME).is_file():
        return "missing %s" % PORTRAIT_NAME
    try:
        if _find_voice_sample(entry) is None:
            return "missing %s" % " or ".join(VOICE_SAMPLE_NAMES)
    except ValueError as e:
        return str(e)
    return None


def save_scene(library_dir, name, description, png_bytes, plugin_root=None):
    """Write one scene (name + description + picture) into the library.

    ``png_bytes`` may be ``None`` when OVERWRITING (socket left
    unconnected): the picture on disk is kept, so editing the description
    never forces re-wiring an image. A new scene still requires one.

    Two-file commit shape as a character: the payloads that change land in
    temp files, then scene.png, then scene.json as the commit point.
    """
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    d = validate_description(description)
    if png_bytes is not None and not (isinstance(png_bytes, (bytes, bytearray)) and png_bytes):
        raise ValueError(
            "png_bytes must be non-empty bytes or None (None = keep the file "
            "on disk when overwriting)"
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

    # Media resolution BEFORE mkdir — same no-empty-dir guarantee as a
    # character save.
    image_path = target_dir / SCENE_IMAGE_NAME
    if png_bytes is None:
        if status == "created":
            raise ValueError(
                "a NEW scene needs a picture: connect an image to save %r for "
                "the first time (overwriting later without reconnecting keeps "
                "the picture on disk)" % n
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                "overwrite of %r without an image, but %s is missing; connect "
                "a new picture" % (n, image_path)
            )
    else:
        png_bytes = bytes(png_bytes)

    target_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {"name": target_dir.name, "description": d}
    json_path = target_dir / SCENE_JSON_NAME

    tmp_png = None
    tmp_json = None
    try:
        if png_bytes is not None:
            tmp_png = _write_temp_file(png_bytes, target_dir)
        tmp_json = _write_temp_file(_json_bytes(json_payload), target_dir)
        if tmp_png is not None:
            try:
                os.replace(tmp_png, image_path)
            except OSError as e:
                raise OSError(
                    "failed to replace %s (%s); all target files are unchanged"
                    % (image_path, e)
                ) from e
            tmp_png = None
        try:
            os.replace(tmp_json, json_path)
        except OSError as e:
            raise OSError(
                "scene.json replacement failed (%s); the picture on disk is "
                "unchanged. Saving %r again successfully rewrites everything."
                % (e, target_dir.name)
            ) from e
        tmp_json = None
    finally:
        if tmp_png is not None:
            tmp_png.unlink(missing_ok=True)
        if tmp_json is not None:
            tmp_json.unlink(missing_ok=True)

    return {
        "name": target_dir.name,
        "description": d,
        "status": status,
        "png_bytes": image_path.read_bytes(),
    }


def load_scene(library_dir, name, plugin_root=None):
    """Read one scene. Every failure raises with the absolute path involved."""
    root = validate_library_dir(library_dir, plugin_root)
    n = validate_name(name)
    scenes_dir = root / SCENES_DIRNAME
    target_dir = _find_casefold_sibling(scenes_dir, n)
    if target_dir is None or not target_dir.is_dir():
        raise FileNotFoundError("scene %r not found under %s" % (n, scenes_dir))
    json_path = target_dir / SCENE_JSON_NAME
    image_path = target_dir / SCENE_IMAGE_NAME
    payload = _read_asset_json(json_path, "description")
    if payload["name"] != target_dir.name:
        raise ValueError(
            "%s declares name %r but the directory is %r; rename the directory "
            "or fix the JSON so both match"
            % (json_path, payload["name"], target_dir.name)
        )
    if not image_path.is_file():
        raise FileNotFoundError(
            "missing scene picture for %r: %s" % (target_dir.name, image_path)
        )
    return {
        "name": payload["name"],
        "description": payload["description"],
        "png_bytes": image_path.read_bytes(),
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
    if not (entry / SCENE_IMAGE_NAME).is_file():
        return "missing %s" % SCENE_IMAGE_NAME
    return None


def character_cache_key(library_dir, name, plugin_root=None):
    """Content hash of character.json + portrait.png for IS_CHANGED.

    Never raises: any problem (missing, unreadable, invalid) returns the fixed
    MISSING_CACHE_MARKER so the failure is not cached as a success.
    """
    try:
        root = validate_library_dir(library_dir, plugin_root)
        n = validate_name(name)
        target_dir = _find_casefold_sibling(root / CHARACTERS_DIRNAME, n)
        if target_dir is None or not target_dir.is_dir():
            return MISSING_CACHE_MARKER
        digest = hashlib.sha256()
        for fname in (CHARACTER_JSON_NAME, PORTRAIT_NAME):
            digest.update(fname.encode("utf-8"))
            digest.update(b"\x00")
            digest.update((target_dir / fname).read_bytes())
            digest.update(b"\x00")
        sample = _find_voice_sample(target_dir)
        if sample is None:
            return MISSING_CACHE_MARKER
        digest.update(sample.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sample.read_bytes())
        return digest.hexdigest()
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
    """Content hash of scene.json + scene.png for IS_CHANGED. Never raises."""
    try:
        root = validate_library_dir(library_dir, plugin_root)
        n = validate_name(name)
        target_dir = _find_casefold_sibling(root / SCENES_DIRNAME, n)
        if target_dir is None or not target_dir.is_dir():
            return MISSING_CACHE_MARKER
        digest = hashlib.sha256()
        for fname in (SCENE_JSON_NAME, SCENE_IMAGE_NAME):
            path = target_dir / fname
            if not path.is_file():
                return MISSING_CACHE_MARKER
            digest.update(fname.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\x00")
        return digest.hexdigest()
    except Exception:
        return MISSING_CACHE_MARKER
