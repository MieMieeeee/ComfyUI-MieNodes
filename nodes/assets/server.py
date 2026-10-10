# -*- coding: utf-8 -*-
"""HTTP endpoints for the asset UI (docs/PLAN_ASSET_LIBRARY_V2.md section 3/7/8).

POST /mie_nodes/assets/list     -> names + per-asset counts of the FIXED library
GET  /mie_nodes/assets/meta     -> one asset's texts + counts + first-image stamp
GET  /mie_nodes/assets/picture  -> first image PNG bytes
GET  /mie_nodes/assets/thumb    -> 256px JPEG thumbnail of the first image
POST /mie_nodes/assets/open     -> open the library folder in the OS file manager

The library location is FIXED per machine ($MIE_ASSET_LIBRARY, then
~/mie_asset_library) — none of these endpoints honor a client-supplied
path, and /open specifically opens ONLY the resolved folder (created when
missing). The whole scan/thumb logic lives in nodes/assets/library.py
(pure stdlib, unit-tested); this module only wraps aiohttp + Pillow.
Imported from the plugin root inside a try/except because the dev/pytest
environment has no ComfyUI server.
"""

from __future__ import annotations

try:
    from _mienodes_internal.nodes.assets import library
except ImportError:
    from . import library

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:  # dev / pytest: no ComfyUI process
    PromptServer = None


def _open_folder_in_file_manager(path):
    """Open `path` in the OS file manager. Windows-first (os.startfile is
    the plan's prescribed mechanism); the fallbacks cover the LAN box if it
    ever runs something else."""
    import os
    import subprocess
    import sys

    path.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 - the documented Windows route
        return True
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)])  # noqa: S603,S607
    return True


def _register() -> None:
    if PromptServer is None:
        return
    routes = PromptServer.instance.routes

    @routes.post("/mie_nodes/assets/list")
    async def list_assets(request):  # noqa: ANN001 - aiohttp handler
        # The location is fixed per machine; the body is ignored entirely.
        payload = library.list_assets_payload("")
        return web.json_response(payload)

    @routes.post("/mie_nodes/assets/upload_audio")
    async def upload_audio(request):  # noqa: ANN001 - aiohttp handler
        """Upload one audio clip into the folder_paths 'audio' category
        (input/audio) — the directory our audio_upload picker lists and
        where uploaded clips are expected to land. Core has /upload/image
        only; audio has no core endpoint, so it lives here."""
        import os
        import tempfile
        from pathlib import Path as _P

        try:
            import folder_paths
        except Exception as e:  # pragma: no cover - outside ComfyUI
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        try:
            post = await request.post()
            upload = post.get("audio")
            if upload is None or not getattr(upload, "filename", ""):
                return web.json_response(
                    {"ok": False, "error": "multipart field 'audio' is required"},
                    status=400,
                )
            name = os.path.basename(upload.filename)
            if not name or name.startswith("."):
                return web.json_response(
                    {"ok": False, "error": "invalid file name"}, status=400
                )
            root = _P(folder_paths.get_input_directory()) / "audio"
            root.mkdir(parents=True, exist_ok=True)
            data = upload.file.read()
            if not data:
                return web.json_response(
                    {"ok": False, "error": "empty upload"}, status=400
                )
            fd, tmp = tempfile.mkstemp(prefix=".mie_audio_", dir=str(root))
            tmp_path = _P(tmp)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                os.replace(tmp_path, root / name)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
            folder_paths.filename_list_cache.pop("audio", None)
            return web.json_response({"ok": True, "name": name})
        except Exception as e:  # noqa: BLE001 - surfaced verbatim
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @routes.post("/mie_nodes/assets/open")
    async def open_library(request):  # noqa: ANN001 - aiohttp handler
        import traceback

        try:
            root = library.validate_library_dir("")
            _open_folder_in_file_manager(root)
            return web.json_response({"ok": True, "library_dir": str(root)})
        except Exception as e:  # noqa: BLE001 - surfaced to the UI verbatim
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(e)}, status=400)

    @routes.get("/mie_nodes/assets/picture")
    async def asset_picture(request):  # noqa: ANN001 - aiohttp handler
        q = request.rel_url.query
        try:
            data = library.read_picture("", q.get("name", ""), q.get("kind", ""))
        except (ValueError, FileNotFoundError) as e:
            return web.json_response({"error": str(e)}, status=404)
        return web.Response(body=data, content_type="image/png")

    @routes.get("/mie_nodes/assets/meta")
    async def asset_meta(request):  # noqa: ANN001 - aiohttp handler
        q = request.rel_url.query
        try:
            meta = library.read_asset_meta("", q.get("name", ""), q.get("kind", ""))
        except (ValueError, FileNotFoundError) as e:
            return web.json_response({"ok": False, "error": str(e)}, status=404)
        meta = dict(meta)
        meta["ok"] = True
        return web.json_response(meta)

    # Thumbnails: the picker decodes dozens of portraits per open, so it
    # must never decode originals. Keyed by (path, size, mtime_ns) — the
    # same key travels in the URL, so a changed picture is a NEW URL and
    # the browser cache can be long.
    _thumb_cache: dict = {}

    @routes.get("/mie_nodes/assets/thumb")
    async def asset_thumb(request):  # noqa: ANN001 - aiohttp handler
        q = request.rel_url.query
        try:
            meta = library.read_asset_meta("", q.get("name", ""), q.get("kind", ""))
            original = library.read_picture("", q.get("name", ""), q.get("kind", ""))
        except (ValueError, FileNotFoundError) as e:
            return web.json_response({"error": str(e)}, status=404)
        stamp = "%d-%d" % (meta["picture_size"], meta["picture_mtime_ns"])
        if q.get("stamp") and q.get("stamp") != stamp:
            # URL carries an outdated stamp: answer with the fresh thumb but
            # a no-cache header so the caller re-keys.
            cache_control = "no-cache"
        else:
            cache_control = "max-age=86400"
        import io as _io

        from PIL import Image

        cache_key = (meta["name"], meta["kind"], stamp)
        thumb = _thumb_cache.get(cache_key)
        if thumb is None:
            img = Image.open(_io.BytesIO(original)).convert("RGB")
            img.thumbnail((256, 256))
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            thumb = buf.getvalue()
            if len(_thumb_cache) > 256:  # bound it; a save bumps stamps anyway
                _thumb_cache.clear()
            _thumb_cache[cache_key] = thumb
        return web.Response(
            body=thumb,
            content_type="image/jpeg",
            headers={"Cache-Control": cache_control},
        )


_register()
