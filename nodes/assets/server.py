# -*- coding: utf-8 -*-
"""One HTTP endpoint feeding the asset dropdowns (frontend assist).

POST /mie_nodes/assets/list   {"library_dir": "..."} ->
    {"ok": true, "characters": [...], "scenes": [...], "error": null|str}

The whole scan lives in library.list_assets_payload (pure stdlib, tested);
this module only wraps it in aiohttp. Imported from the plugin root inside
a try/except because the dev/pytest environment has no ComfyUI server.
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


def _register() -> None:
    if PromptServer is None:
        return
    routes = PromptServer.instance.routes

    @routes.post("/mie_nodes/assets/list")
    async def list_assets(request):  # noqa: ANN001 - aiohttp handler
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "characters": [], "scenes": [],
                 "error": "request body must be JSON"}, status=400
            )
        library_dir = ""
        if isinstance(body, dict):
            library_dir = str(body.get("library_dir") or "")
        payload = library.list_assets_payload(library_dir)
        return web.json_response(payload)

    @routes.get("/mie_nodes/assets/picture")
    async def asset_picture(request):  # noqa: ANN001 - aiohttp handler
        q = request.rel_url.query
        try:
            data = library.read_picture(
                q.get("dir", ""), q.get("name", ""), q.get("kind", "")
            )
        except (ValueError, FileNotFoundError) as e:
            return web.json_response({"error": str(e)}, status=404)
        return web.Response(body=data, content_type="image/png")

    @routes.get("/mie_nodes/assets/meta")
    async def asset_meta(request):  # noqa: ANN001 - aiohttp handler
        q = request.rel_url.query
        try:
            meta = library.read_asset_meta(
                q.get("dir", ""), q.get("name", ""), q.get("kind", "")
            )
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
            meta = library.read_asset_meta(
                q.get("dir", ""), q.get("name", ""), q.get("kind", "")
            )
            original = library.read_picture(
                q.get("dir", ""), q.get("name", ""), q.get("kind", "")
            )
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
