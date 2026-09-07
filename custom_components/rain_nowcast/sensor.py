"""The seven sensors."""
from __future__ import annotations

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, FORECAST_ATTRS, SENSORS
from .entity import RainNowcastEntity


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry,
                            async_add_entities: AddEntitiesCallback) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        RainNowcastSensor(coordinator, entry.entry_id, key) for key in SENSORS)


class RainNowcastSensor(RainNowcastEntity, SensorEntity):
    def __init__(self, coordinator, entry_id: str, key: str) -> None:
        name, unit, device_class, state_class, icon = SENSORS[key]
        super().__init__(coordinator, entry_id, key, name)
        self._attr_native_unit_of_measurement = unit
        self._attr_device_class = device_class
        self._attr_state_class = state_class
        self._attr_icon = icon

    @property
    def native_value(self):
        """None becomes 'unknown' in Home Assistant, which is the honest answer
        to "when does rain start" when none is forecast. A sentinel number would
        be silently compared against by an automation."""
        return self.value()

    @property
    def extra_state_attributes(self) -> dict | None:
        """The whole per-minute curve rides on the summary entity, matching what
        the MQTT path puts there, so a template sensor or custom card written
        against one works against the other."""
        if self._key != "summary":
            return None
        attrs = (self.coordinator.data or {}).get("attributes") or {}
        return {k: attrs[k] for k in FORECAST_ATTRS if k in attrs}
