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


_register()
