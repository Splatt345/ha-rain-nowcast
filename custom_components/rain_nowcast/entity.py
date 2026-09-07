"""The device every entity attaches to, and the value lookup they share."""
from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import RainNowcastCoordinator


class RainNowcastEntity(CoordinatorEntity[RainNowcastCoordinator]):
    """Base for both platforms.

    `_attr_has_entity_name` plus a device name gives "Rain Nowcast Rain rate"
    without repeating the prefix in every entity name, which is the modern
    Home Assistant convention.

    The unique_id is derived from the config entry, so the entities are proper
    registry entries: they survive a restart, can be renamed in the UI, and
    group under one device. That is the whole reason this integration exists
    rather than leaving people on the add-on's REST fallback, whose states
    vanish on every restart.
    """

    _attr_has_entity_name = True

    def __init__(self, coordinator: RainNowcastCoordinator, entry_id: str,
                 key: str, name: str) -> None:
        super().__init__(coordinator)
        self._key = key
        self._attr_name = name
        self._attr_unique_id = f"{entry_id}_{key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="Rain Nowcast",
            manufacturer="pysteps",
            model="STEPS radar extrapolation + SmaAt-UNet",
        )

    @property
    def available(self) -> bool:
        return super().available and self.coordinator.data is not None

    def value(self):
        return (self.coordinator.data or {}).get(self._key)
