"""Shared constants, and the one table that defines every entity.

The add-on derives the numbers (`nowcast/publish.py: payload`) and serves them
at /api/metrics. This integration only maps them onto Home Assistant entities,
so the two can never disagree about what "minutes until rain" means - which is
exactly what would happen if the integration recomputed them from the raw
series.
"""
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import PERCENTAGE, UnitOfLength, UnitOfTime

DOMAIN = "rain_nowcast"
#: The Supervisor names an add-on's container from its slug, and the exact form
#: has varied: dashes or underscores, with or without the repository prefix, and
#: the "local" prefix only for a locally-installed add-on. Guessing wrong means
#: the integration simply cannot connect, with nothing to tell the user which
#: name to try - so instead of guessing once, the config flow probes this list
#: and keeps whichever answers.
DEFAULT_HOST = "local-rain-nowcast"
HOST_CANDIDATES = (
    "local-rain-nowcast",
    "local_rain_nowcast",
    "addon_local_rain_nowcast",
    "rain-nowcast",
    "rain_nowcast",
    "local-rain-nowcast.local.hass.io",
)
DEFAULT_PORT = 8061
DEFAULT_SCAN_INTERVAL = 60            # the add-on itself only rebuilds every 5 min

# key -> (name, unit, device_class, state_class, icon)
# Deliberately mirrors publish.ENTITIES: same keys, same units, same device
# classes, so an automation written against the MQTT sensors keeps working
# unchanged if you switch to this integration.
SENSORS = {
    "rain_rate": ("Rain rate", "mm/h", SensorDeviceClass.PRECIPITATION_INTENSITY,
                  SensorStateClass.MEASUREMENT, None),
    "minutes_until_rain": ("Rain starts in", UnitOfTime.MINUTES, None, None,
                           "mdi:weather-rainy"),
    "minutes_until_dry": ("Rain stops in", UnitOfTime.MINUTES, None, None,
                          "mdi:weather-partly-cloudy"),
    "peak_rate": ("Peak rain rate", "mm/h",
                  SensorDeviceClass.PRECIPITATION_INTENSITY,
                  SensorStateClass.MEASUREMENT, None),
    "rain_chance": ("Rain chance", PERCENTAGE, None,
                    SensorStateClass.MEASUREMENT, "mdi:weather-pouring"),
    "expected_rain": ("Rain expected", UnitOfLength.MILLIMETERS,
                      SensorDeviceClass.PRECIPITATION,
                      SensorStateClass.MEASUREMENT, None),
    "summary": ("Rain summary", None, None, None, "mdi:text-short"),
}

BINARY_SENSORS = {"raining": ("Raining", "moisture")}

#: Carried as attributes on the summary entity, exactly as the MQTT path does.
FORECAST_ATTRS = ("forecast_minutes", "forecast_values", "forecast_probability",
                  "forecast_low", "forecast_high", "latitude", "longitude")
