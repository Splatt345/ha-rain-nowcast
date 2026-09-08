"""Rain Nowcast: Home Assistant entities from the Rain Nowcast add-on.

Talks to the add-on over HTTP, so no MQTT broker is needed. The add-on's own
MQTT path still works and remains the default; this is the alternative for
people who do not want to run a broker, and it produces the same eight entities
from the same numbers (the add-on serves them at /api/metrics, derived once in
its publish.payload).
"""
from __future__ import annotations

from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant

from homeassistant.components.http import StaticPathConfig

from .const import DOMAIN
from .coordinator import RainNowcastCoordinator
from .http import RainNowcastProxyView

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]
STATIC_URL = f"/{DOMAIN}_static"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator = RainNowcastCoordinator(
        hass, entry.data[CONF_HOST], entry.data[CONF_PORT])
    # Fail setup loudly if the add-on is unreachable, so Home Assistant retries
    # rather than creating eight permanently-unavailable entities.
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    # Registered once, however many entries exist. The proxy carries the radar
    # frames (the browser cannot reach the add-on directly); the static path
    # carries the Lovelace card and its copy of Leaflet, which are public assets
    # and need no authentication.
    #
    # cache_headers=False, deliberately. True sets a max-age of a month, and a
    # Lovelace resource URL has no version in it - so after updating the
    # integration the browser keeps serving the OLD card for weeks, with no
    # refresh strong enough to dislodge it short of editing the resource URL.
    # Without the header the file is revalidated and answered with a 304 when
    # unchanged, which for one 40 KB script is a fair trade for updates that
    # actually arrive. Leaflet is behind the same path and never changes, so it
    # pays the same revalidation for nothing; that is one conditional GET.
    if not hass.data[DOMAIN].get("_http"):
        hass.http.register_view(RainNowcastProxyView(hass))
        await hass.http.async_register_static_paths([
            StaticPathConfig(STATIC_URL, str(Path(__file__).parent / "www"), False)
        ])
        hass.data[DOMAIN]["_http"] = True

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.data[DOMAIN].pop(entry.entry_id)
    return unloaded
