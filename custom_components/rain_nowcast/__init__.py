"""Rain Nowcast: Home Assistant entities from the Rain Nowcast add-on.

Talks to the add-on over HTTP, so no MQTT broker is needed. The add-on's own
MQTT path still works and remains the default; this is the alternative for
people who do not want to run a broker, and it produces the same eight entities
from the same numbers (the add-on serves them at /api/metrics, derived once in
its publish.payload).
"""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import RainNowcastCoordinator

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator = RainNowcastCoordinator(
        hass, entry.data[CONF_HOST], entry.data[CONF_PORT])
    # Fail setup loudly if the add-on is unreachable, so Home Assistant retries
    # rather than creating eight permanently-unavailable entities.
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.data[DOMAIN].pop(entry.entry_id)
    return unloaded
