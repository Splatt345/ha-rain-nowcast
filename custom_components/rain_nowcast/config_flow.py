"""UI setup: ask where the add-on is, prove it answers, then create the entry."""
from __future__ import annotations

import logging

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow)
from homeassistant.core import callback
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DEFAULT_HOST, DEFAULT_PORT, DOMAIN, HOST_CANDIDATES

_LOGGER = logging.getLogger(__name__)


async def probe(hass, host: str, port: int) -> bool:
    """Does the add-on answer at this address?

    A 503 counts as yes: the add-on is up but spends its first minute or two
    downloading radar before it can forecast, and refusing the configuration for
    that would be wrong.
    """
    session = async_get_clientsession(hass)
    try:
        async with session.get(f"http://{host}:{port}/api/metrics",
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return resp.status in (200, 503)
    except (aiohttp.ClientError, TimeoutError):
        return False


async def find_addon(hass, typed: str, port: int) -> str | None:
    """The address the add-on actually answers on, or None.

    Tries what the user typed first, then the known Supervisor naming variants.
    The Supervisor has used more than one over time and the right one is not
    discoverable from inside Home Assistant, so probing beats guessing: the
    failure mode of a wrong guess is an integration that cannot connect and
    cannot say why.
    """
    for host in (typed, *(h for h in HOST_CANDIDATES if h != typed)):
        if await probe(hass, host, port):
            if host != typed:
                _LOGGER.info("Rain Nowcast add-on found at %s, not %s", host, typed)
            return host
    return None


class RainNowcastConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(entry: ConfigEntry) -> OptionsFlow:
        return RainNowcastOptionsFlow()

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        errors = {}
        if user_input is not None:
            port = user_input[CONF_PORT]
            host = await find_addon(self.hass, user_input[CONF_HOST], port)
            if host is None:
                errors["base"] = "cannot_connect"
            else:
                # One instance per add-on, and re-adding the same one should say
                # so rather than duplicating all eight entities.
                await self.async_set_unique_id(f"{host}:{port}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(title="Rain Nowcast",
                                               data={CONF_HOST: host, CONF_PORT: port})

        return self.async_show_form(
            step_id="user", errors=errors,
            data_schema=vol.Schema({
                vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }),
        )




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
            port = user_input[CONF_PORT]
            host = await find_addon(self.hass, user_input[CONF_HOST], port)
            if host is not None:
                user_input = {CONF_HOST: host, CONF_PORT: port}
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
