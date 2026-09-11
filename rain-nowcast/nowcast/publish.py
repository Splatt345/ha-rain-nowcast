"""Publish the nowcast to Home Assistant as entities, so it can drive
automations and not just be looked at.

Two routes, in order of preference:

1. MQTT discovery, when a broker is available. Gives real registry entities -
   they survive restarts, group under one device, and can be renamed in the UI.
2. The Supervisor's core REST API, when there's no broker. Zero setup, but the
   states it creates are not registry entities and vanish on a Home Assistant
   restart (this republishes them on the next radar frame, so at most a few
   minutes of absence).
"""
import datetime
import json
import os

import numpy as np
import requests

DISCOVERY_PREFIX = "homeassistant"
NODE = "rain_nowcast"
STATE_TOPIC = f"{NODE}/state"
AVAILABILITY_TOPIC = f"{NODE}/status"
RAIN_THRESHOLD = 0.1  # mm/h

DEVICE = {
    "identifiers": [NODE],
    "name": "Rain Nowcast",
    "manufacturer": "pysteps",
    "model": "STEPS radar extrapolation",
}

# key -> (friendly name, unit, device_class, state_class, icon)
ENTITIES = {
    "rain_rate": ("Rain rate", "mm/h", "precipitation_intensity", "measurement", None),
    "minutes_until_rain": ("Rain starts in", "min", None, None, "mdi:weather-rainy"),
    "minutes_until_dry": ("Rain stops in", "min", None, None, "mdi:weather-partly-cloudy"),
    "peak_rate": ("Peak rain rate", "mm/h", "precipitation_intensity", "measurement", None),
    "rain_chance": ("Rain chance", "%", None, "measurement", "mdi:weather-pouring"),
    "expected_rain": ("Rain expected", "mm", "precipitation", "measurement", None),
    "summary": ("Rain summary", None, None, None, "mdi:text-short"),
}
BINARY = {"raining": ("Raining", "moisture")}


def frame_age(state):
    """Minutes since the radar frame this forecast was built from.

    Minute 0 of the series is that frame, NOT the present moment: OPERA
    composites come on a 5-minute cadence and then take a few minutes to
    publish, so a forecast is typically 5-10 minutes old before anyone reads
    it. Counting "minutes until rain" from minute 0 is therefore that much
    late, which is the difference between watering the garden and not.

    Computed per call rather than stored, because this endpoint is polled: the
    answer keeps ageing until the next radar frame lands.
    """
    try:
        updated = datetime.datetime.fromisoformat(state["updated"])
    except (KeyError, TypeError, ValueError):
        return 0
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=datetime.timezone.utc)
    delta = datetime.datetime.now(datetime.timezone.utc) - updated
    # Capped: a wrong clock either side must not silently throw the whole
    # forecast away, and beyond an hour there is nothing left to report on.
    return int(min(60, max(0, round(delta.total_seconds() / 60))))


def metrics(state):
    """Derive the automation-useful numbers from a finished nowcast state."""
    minutes = np.array(state["series"]["minutes"])
    values = np.array(state["series"]["values"])
    age = frame_age(state)
    now = int(np.searchsorted(minutes, age))
    # Everything below is phrased as "in N minutes", so the axis is shifted to
    # start from now rather than from the radar timestamp.
    ahead, future = minutes[now:] - age, values[now:]

    raining = bool(future[0] >= RAIN_THRESHOLD)
    wet = np.flatnonzero(future >= RAIN_THRESHOLD)
    dry = np.flatnonzero(future < RAIN_THRESHOLD)

    return {
        "rain_rate": round(float(future[0]), 2),
        "raining": raining,
        # None becomes "unknown" in HA, which is the honest answer for "when
        # does rain start" when none is forecast - better than a sentinel
        # number that an automation would compare against by accident.
        "minutes_until_rain": 0 if raining else (int(ahead[wet[0]]) if wet.size else None),
        "minutes_until_dry": (int(ahead[dry[0]]) if raining and dry.size else
                              (0 if not raining else None)),
        "peak_rate": round(float(future.max()), 2),
        # Share of the STEPS ensemble that rains on this point at its wettest
        # minute. An automation that wants "don't water the garden if it might
        # rain" needs this, not a smoothed mean that reads 0.0 while 4 of 12
        # members are wet.
        "rain_chance": (None if state.get("rain_chance") is None
                        else int(round(100 * state["rain_chance"]))),
        # each per-minute value is a rate in mm/h, so a minute contributes v/60 mm
        "expected_rain": round(float(future.sum() / 60.0), 2),
        "summary": state["summary"],
        "horizon_minutes": int(ahead[-1]),
        "updated": state["updated"],
    }


def payload(state):
    """(metrics, attributes) - everything a Home Assistant entity needs.

    Split out of Publisher.publish so the MQTT path, the REST path and the
    /api/metrics endpoint the custom integration polls all derive their values
    HERE, exactly once. Two consumers computing "minutes until rain" from the
    same series independently is how they end up disagreeing.
    """
    return metrics(state), {
        # The whole per-minute curve rides along as attributes, so a template
        # sensor or custom card can use it without re-deriving anything.
        "forecast_minutes": state["series"]["minutes"],
        "forecast_values": state["series"]["values"],
        "forecast_probability": state["series"].get("probability"),
        "forecast_low": state["series"].get("p10"),
        "forecast_high": state["series"].get("p90"),
        "forecast_min": state["series"].get("pmin"),
        "forecast_max": state["series"].get("pmax"),
        "latitude": state["location"][0],
        "longitude": state["location"][1],
    }


class Publisher:
    """Publishes to MQTT if a broker is reachable, else to the core REST API."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.client = None
        self.mode = "off"
        self.token = os.environ.get("SUPERVISOR_TOKEN")
        if enabled:
            self.mode = "mqtt" if self._connect_mqtt() else (
                "rest" if self.token else "off")
            if self.mode == "off":
                print("Sensors: no MQTT broker and no Supervisor token - not publishing")
            else:
                print(f"Sensors: publishing over {self.mode}")

    # -- MQTT ---------------------------------------------------------------
    def _broker(self):
        """Broker settings: the Supervisor first, then explicit configuration.

        The Supervisor path (services: mqtt:want) is preferred and unchanged -
        on Home Assistant OS or Supervised there is nothing to fill in. But it
        was the ONLY path, and it does not exist on Home Assistant Container or
        a plain docker-compose install: no Supervisor means no token, which
        meant no MQTT and no REST fallback either, so the add-on published
        nothing at all and reported only "not publishing". The explicit
        fallback makes those installs work, and makes this path testable
        against a real broker outside a full HA OS.
        """
        if self.token:
            try:
                r = requests.get(
                    "http://supervisor/services/mqtt",
                    headers={"Authorization": f"Bearer {self.token}"}, timeout=10)
                if r.status_code == 200:
                    return r.json()["data"]
            except (requests.RequestException, KeyError, ValueError):
                pass

        host = (os.environ.get("MQTT_HOST") or "").strip()
        if not host:
            return None
        return {"host": host,
                "port": int(os.environ.get("MQTT_PORT") or 1883),
                "username": os.environ.get("MQTT_USERNAME") or "",
                "password": os.environ.get("MQTT_PASSWORD") or "",
                "ssl": str(os.environ.get("MQTT_SSL", "")).lower()
                       in ("1", "true", "yes", "on")}

    def _connect_mqtt(self):
        cfg = self._broker()
        if not cfg:
            return False
        try:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=NODE)
            if cfg.get("username"):
                client.username_pw_set(cfg["username"], cfg.get("password") or None)
            if cfg.get("ssl"):
                client.tls_set()
            client.will_set(AVAILABILITY_TOPIC, "offline", retain=True)
            client.connect(cfg["host"], int(cfg["port"]), keepalive=60)
            client.loop_start()
            self.client = client
            self._announce()
            client.publish(AVAILABILITY_TOPIC, "online", retain=True)
            return True
        except Exception as exc:  # broker down, bad creds, paho missing
            print(f"Sensors: MQTT unavailable ({exc}); falling back")
            return False

    def _announce(self):
        """Retained discovery configs. One shared state topic, one template each."""
        def common(key, name):
            return {
                "unique_id": f"{NODE}_{key}",
                "object_id": f"{NODE}_{key}",
                "name": name,
                "state_topic": STATE_TOPIC,
                "availability_topic": AVAILABILITY_TOPIC,
                "json_attributes_topic": STATE_TOPIC,
                "device": DEVICE,
            }

        for key, (name, unit, dev_class, state_class, icon) in ENTITIES.items():
            cfg = common(key, name)
            cfg["value_template"] = (
                "{% set v = value_json." + key + " %}"
                "{{ 'unknown' if v is none else v }}"
            )
            if unit:
                cfg["unit_of_measurement"] = unit
            if dev_class:
                cfg["device_class"] = dev_class
            if state_class:
                cfg["state_class"] = state_class
            if icon:
                cfg["icon"] = icon
            self.client.publish(f"{DISCOVERY_PREFIX}/sensor/{NODE}/{key}/config",
                                json.dumps(cfg), retain=True)

        for key, (name, dev_class) in BINARY.items():
            cfg = common(key, name)
            cfg["device_class"] = dev_class
            cfg["payload_on"], cfg["payload_off"] = "True", "False"
            cfg["value_template"] = "{{ value_json." + key + " }}"
            self.client.publish(f"{DISCOVERY_PREFIX}/binary_sensor/{NODE}/{key}/config",
                                json.dumps(cfg), retain=True)

    # -- REST fallback ------------------------------------------------------
    def _publish_rest(self, data, attrs):
        headers = {"Authorization": f"Bearer {self.token}"}
        for key in list(ENTITIES) + list(BINARY):
            value = data[key]
            domain = "binary_sensor" if key in BINARY else "sensor"
            if key in BINARY:
                value = "on" if value else "off"
            body = {
                "state": "unknown" if value is None else value,
                "attributes": {"friendly_name":
                               (BINARY[key][0] if key in BINARY else ENTITIES[key][0])},
            }
            if key in ENTITIES:
                _, unit, dev_class, state_class, icon = ENTITIES[key]
                if unit:
                    body["attributes"]["unit_of_measurement"] = unit
                if dev_class:
                    body["attributes"]["device_class"] = dev_class
                if state_class:
                    body["attributes"]["state_class"] = state_class
                if icon:
                    body["attributes"]["icon"] = icon
            elif key in BINARY:
                body["attributes"]["device_class"] = BINARY[key][1]
            if key == "summary":
                body["attributes"].update(attrs)
            requests.post(f"http://supervisor/core/api/states/{domain}.{NODE}_{key}",
                          headers=headers, json=body, timeout=10)

    # -- entry point --------------------------------------------------------
    def publish(self, state):
        if self.mode == "off":
            return
        data, attrs = payload(state)
        if self.mode == "mqtt":
            self.client.publish(STATE_TOPIC, json.dumps({**data, **attrs}), retain=True)
        else:
            self._publish_rest(data, attrs)


def _demo():
    """Self-check for the derived numbers - no broker, no Home Assistant."""
    def state(values, minutes=None, chance=None):
        minutes = minutes or list(range(-55, 61))
        return {"series": {"minutes": minutes, "values": values},
                "rain_chance": chance,
                "summary": "x", "updated": "t", "location": [50.0, 4.0]}

    n_past, n_fut = 55, 61
    dry = metrics(state([0.0] * (n_past + n_fut)))
    assert dry["raining"] is False and dry["minutes_until_rain"] is None
    assert dry["minutes_until_dry"] == 0 and dry["peak_rate"] == 0.0

    # dry now, 4 mm/h starting 20 minutes out
    vals = [0.0] * (n_past + 20) + [4.0] * (n_fut - 20)
    m = metrics(state(vals))
    assert m["raining"] is False, m
    assert m["minutes_until_rain"] == 20, m
    # The same forecast read six minutes later: the rain is six minutes nearer.
    # Without this the sensors are always as stale as the radar's publishing
    # lag, which is what made the widget disagree with the wall clock.
    stale = state(vals)
    stale["updated"] = (datetime.datetime.now(datetime.timezone.utc)
                        - datetime.timedelta(minutes=6)).isoformat()
    assert frame_age(stale) == 6, frame_age(stale)
    assert metrics(stale)["minutes_until_rain"] == 14, metrics(stale)
    assert metrics(stale)["horizon_minutes"] == 54
    assert m["peak_rate"] == 4.0
    # 41 minutes of 4 mm/h -> 41/60 * 4 mm
    assert abs(m["expected_rain"] - 41 * 4 / 60) < 0.02, m

    # raining now, stopping in 10
    vals = [2.0] * (n_past + 10) + [0.0] * (n_fut - 10)
    m = metrics(state(vals))
    assert m["raining"] is True and m["minutes_until_rain"] == 0
    assert m["minutes_until_dry"] == 10, m
    assert m["rain_rate"] == 2.0

    # the case the ensemble exists for: mean says dry, a third of members disagree
    m = metrics(state([0.0] * (n_past + n_fut), chance=0.33))
    assert m["raining"] is False and m["peak_rate"] == 0.0
    assert m["rain_chance"] == 33, m
    # a state from before probabilities existed must still publish
    assert metrics(state([0.0] * (n_past + n_fut)))["rain_chance"] is None
    print("publish self-check ok")


if __name__ == "__main__":
    _demo()
