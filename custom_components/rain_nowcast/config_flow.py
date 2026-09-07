"""UI setup: ask where the add-on is, prove it answers, then create the entry."""
from __future__ import annotations

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DEFAULT_HOST, DEFAULT_PORT, DOMAIN


class RainNowcastConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        errors = {}
        if user_input is not None:
            host, port = user_input[CONF_HOST], user_input[CONF_PORT]
            # One instance per add-on, and re-adding the same one should say so
            # rather than duplicating all eight entities.
            await self.async_set_unique_id(f"{host}:{port}")
            self._abort_if_unique_id_configured()

            error = await self._probe(host, port)
            if error is None:
                return self.async_create_entry(title="Rain Nowcast",
                                               data={CONF_HOST: host, CONF_PORT: port})
            errors["base"] = error

        return self.async_show_form(
            step_id="user", errors=errors,
            data_schema=vol.Schema({
                vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }),
        )

    async def _probe(self, host: str, port: int) -> str | None:
        """Check the add-on answers before accepting the entry.

        A 503 means it is running but has not finished its first forecast -
        which is normal for the first minute or two and NOT a reason to reject
        the configuration, so it is accepted.
        """
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(f"http://{host}:{port}/api/metrics",
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status in (200, 503):
                    return None
                return "cannot_connect"
        except (aiohttp.ClientError, TimeoutError):
            return "cannot_connect"
