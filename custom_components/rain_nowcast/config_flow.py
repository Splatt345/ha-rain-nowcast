"""UI setup: ask where the add-on is, prove it answers, then create the entry."""
from __future__ import annotations

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow)
from homeassistant.core import callback
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DEFAULT_HOST, DEFAULT_PORT, DOMAIN


class RainNowcastConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(entry: ConfigEntry) -> OptionsFlow:
        return RainNowcastOptionsFlow()

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


class RainNowcastOptionsFlow(OptionsFlow):
    """Change where the add-on lives without deleting the entry.

    Needed because host and port are set once at setup and then frozen: moving
    the add-on, or changing its port, otherwise means removing the integration
    and re-adding it, which loses every entity's history and any renames.
    """

    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        errors = {}
        current = {**self.config_entry.data, **self.config_entry.options}

        if user_input is not None:
            session = async_get_clientsession(self.hass)
            host, port = user_input[CONF_HOST], user_input[CONF_PORT]
            try:
                async with session.get(f"http://{host}:{port}/api/metrics",
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    ok = resp.status in (200, 503)
            except (aiohttp.ClientError, TimeoutError):
                ok = False
            if ok:
                # Written to data, not options, so the coordinator reads one
                # place; reloading picks the new address up immediately.
                self.hass.config_entries.async_update_entry(
                    self.config_entry, data=user_input)
                self.hass.async_create_task(
                    self.hass.config_entries.async_reload(self.config_entry.entry_id))
                return self.async_create_entry(title="", data={})
            errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="init", errors=errors,
            data_schema=vol.Schema({
                vol.Required(CONF_HOST, default=current.get(CONF_HOST, DEFAULT_HOST)): str,
                vol.Required(CONF_PORT, default=current.get(CONF_PORT, DEFAULT_PORT)): int,
            }),
        )
