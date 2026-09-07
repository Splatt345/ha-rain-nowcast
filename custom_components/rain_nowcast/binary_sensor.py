"""Is it raining on you right now."""
from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import BINARY_SENSORS, DOMAIN
from .entity import RainNowcastEntity


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry,
                            async_add_entities: AddEntitiesCallback) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        RainNowcastBinarySensor(coordinator, entry.entry_id, key)
        for key in BINARY_SENSORS)


class RainNowcastBinarySensor(RainNowcastEntity, BinarySensorEntity):
    def __init__(self, coordinator, entry_id: str, key: str) -> None:
        name, device_class = BINARY_SENSORS[key]
        super().__init__(coordinator, entry_id, key, name)
        self._attr_device_class = device_class

    @property
    def is_on(self) -> bool | None:
        value = self.value()
        return None if value is None else bool(value)
