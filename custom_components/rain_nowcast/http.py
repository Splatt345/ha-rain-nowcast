"""Proxy the add-on's radar frames and state through Home Assistant.

The Lovelace card runs in a browser that generally CANNOT reach the add-on: on
Home Assistant OS the add-on has no published port and lives on the Supervisor's
internal network, reachable only from Home Assistant itself. So the card asks
Home Assistant, and Home Assistant asks the add-on.

That also means the card inherits Home Assistant's authentication instead of
needing an unauthenticated hole punched through to the add-on.

The proxy is a WHITELIST, not a path join. It forwards exactly the state
endpoint and the radar frames, and nothing else - it is an authenticated route
into a service on an internal network, and "proxy whatever path you're given"
is how that becomes a way to reach the rest of it.
"""
from __future__ import annotations

import logging
import re

import aiohttp
from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

#: p0..p23 and f1..f24 - the frame names server.refresh emits. Anchored, so
#: nothing with a slash or a dot can pass.
_FRAME = re.compile(r"^[pf]\d{1,2}$")

URL = f"/api/{DOMAIN}/{{entry_id}}/{{kind}}/{{name}}"


class RainNowcastProxyView(HomeAssistantView):
    """GET /api/rain_nowcast/<entry_id>/<kind>/<name>."""

    url = "/api/" + DOMAIN + "/{entry_id}/{kind}/{name}"
    name = f"api:{DOMAIN}:proxy"
    requires_auth = True

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    def _addon(self, entry_id: str) -> tuple[str, int] | None:
        entry = self.hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN:
            return None
        return entry.data["host"], entry.data["port"]

    async def get(self, request: web.Request, entry_id: str, kind: str,
                  name: str) -> web.Response:
        target = self._addon(entry_id)
        if target is None:
            return web.Response(status=404, text="unknown Rain Nowcast entry")
        host, port = target

        if kind == "api" and name == "state":
            path, ctype = "api/state", "application/json"
        elif kind == "frames" and _FRAME.match(name):
            path, ctype = f"frames/{name}.png", "image/png"
        elif kind == "api" and name == "point":
            # Click-to-forecast-elsewhere. Parsed and re-encoded rather than
            # forwarded, so nothing from the query string reaches the add-on
            # except two floats in range.
            try:
                lat = float(request.query["lat"])
                lon = float(request.query["lon"])
            except (KeyError, TypeError, ValueError):
                return web.Response(status=400, text="lat and lon required")
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return web.Response(status=400, text="lat/lon out of range")
            path = f"api/point?lat={lat:.4f}&lon={lon:.4f}"
            ctype = "application/json"
        else:
            return web.Response(status=404, text="not a proxied resource")

        session = async_get_clientsession(self.hass)
        try:
            async with session.get(f"http://{host}:{port}/{path}",
                                   timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    return web.Response(status=resp.status,
                                        text="add-on returned %s" % resp.status)
                body = await resp.read()
        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.debug("proxy to %s:%s/%s failed: %s", host, port, path, err)
            return web.Response(status=502, text="cannot reach the Rain Nowcast add-on")

        # Frames are immutable for a given radar timestamp and the card appends
        # ?v=<updated>, so they cache well; the state must not.
        cache = "max-age=300" if ctype == "image/png" else "no-store"
        return web.Response(body=body, content_type=ctype,
                            headers={"Cache-Control": cache})
