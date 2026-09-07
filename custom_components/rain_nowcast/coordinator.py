"""Polls the add-on's /api/metrics and hands the result to the entities."""
from __future__ import annotations

import logging
from datetime import timedelta

import aiohttp
import async_timeout
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DEFAULT_SCAN_INTERVAL, DOMAIN

_LOGGER = logging.getLogger(__name__)


class RainNowcastCoordinator(DataUpdateCoordinator):
    """One poll of the add-on, shared by all eight entities.

    Polling is cheap and deliberately more frequent than the forecast changes:
    the add-on only rebuilds when a new radar frame lands (every 5 minutes), so
    most polls return the same payload. A minute is short enough that a new
    frame shows up promptly and long enough to cost nothing.
    """

    def __init__(self, hass: HomeAssistant, host: str, port: int) -> None:
        self.url = f"http://{host}:{port}/api/metrics"
        self._session = async_get_clientsession(hass)
        super().__init__(
            hass, _LOGGER, name=DOMAIN,
            update_interval=timedelta(seconds=DEFAULT_SCAN_INTERVAL),
        )

    async def _async_update_data(self) -> dict:
        try:
            async with async_timeout.timeout(20):
                async with self._session.get(self.url) as resp:
                    if resp.status == 503:
                        # The add-on is up but has no forecast yet - it spends
                        # the first minute or two downloading radar. That is a
                        # normal startup state, not a failure, but there is
                        # nothing to show until it passes.
                        raise UpdateFailed("add-on has not produced a forecast yet")
                    resp.raise_for_status()
                    return await resp.json()
        except aiohttp.ClientError as err:
            raise UpdateFailed(f"cannot reach the Rain Nowcast add-on: {err}") from err
        except TimeoutError as err:
            raise UpdateFailed("timed out talking to the Rain Nowcast add-on") from err
