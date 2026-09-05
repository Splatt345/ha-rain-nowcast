# Rain Nowcast

Live rain radar over Belgium plus a 60-minute forecast, as a Home Assistant
panel. The forecast is real radar extrapolation (pysteps STEPS), not a
re-serving of somebody else's API.

## Installation

1. Settings → Add-ons → Add-on store → ⋮ → **Repositories**, add this repo.
2. Install **Rain Nowcast**, set your latitude/longitude and API key, start it.
3. Open it from the sidebar (the panel is added automatically).

First start takes a minute or two: it downloads an hour of radar (~35 MB)
before it can forecast anything.

## Getting an API key

The radar comes from EUMETNET's OPERA composite via the MeteoGate Open Radar
Data API. Anonymous access is rate-limited to the point of being unusable for
a 5-minute poll, so register a free key at <https://meteogate.eu/> (sign in
with GitHub/Google/Azure AD, no approval wait) and paste it into `api_key`.

## Options

| Option | Default | What it does |
|---|---|---|
| `latitude` / `longitude` | Brussels | The point the minute-by-minute graph is for. Must be inside the radar crop (roughly Belgium plus a margin: 2.0–6.8°E, 49.0–51.8°N). |
| `api_key` | – | MeteoGate ORD API key. |
| `history_minutes` | 60 | How much radar history to show and feed the nowcast. |
| `forecast_minutes` | 60 | How far ahead to forecast. Skill is largely gone past ~35 min. |
| `ensemble_members` | 12 | STEPS ensemble size. Lower it to 4–6 on a Raspberry Pi; a full cycle takes ~12 s on a desktop CPU. |
| `publish_sensors` | true | Publish the forecast as Home Assistant entities. |

## Sensors

Eight entities update on every radar frame, so automations can act on rain
before it arrives rather than after.

| Entity | Value |
|---|---|
| `binary_sensor.rain_nowcast_raining` | Is it raining at your location right now |
| `sensor.rain_nowcast_rain_rate` | Current rain rate, mm/h |
| `sensor.rain_nowcast_minutes_until_rain` | Minutes until rain starts — `0` if it already is, `unknown` if none is forecast within the hour |
| `sensor.rain_nowcast_minutes_until_dry` | Minutes until it stops |
| `sensor.rain_nowcast_peak_rate` | Heaviest rain expected in the forecast window, mm/h |
| `sensor.rain_nowcast_rain_chance` | Chance of rain at its likeliest minute in the window, % |
| `sensor.rain_nowcast_expected_rain` | Total rain expected over the window, mm |
| `sensor.rain_nowcast_summary` | The one-line summary, plus the whole per-minute curve as attributes |

### Why there is a chance and not just a rate

The forecast is an ensemble: twelve runs of the same model, each with a
different draw of plausible small-scale detail. Every other sensor here reports
their *average*, and an average of twelve rain fields is smoother than any rain
that actually falls. It flattens showers, and near the edge of a band it can
read 0.0 mm/h while a third of the runs put rain on your roof.

`rain_chance` is the share of runs that are wet at the minute they most agree
on. Use it for "should I do this outside"; `rain_rate` and `peak_rate` are still
the right ones for "how hard". The shaded band along the bottom of the chart is
the same number, minute by minute — full height means all twelve runs agree it
rains then.

They are published over MQTT discovery when a broker is configured — the
add-on asks the Supervisor for it, so there is nothing to fill in — which
gives proper registry entities you can rename and group. Without a broker it
falls back to the core REST API: the same entities appear, but they are not
registry entries and are lost on a Home Assistant restart until the next radar
frame republishes them (a few minutes at most).

```yaml
automation:
  - alias: Close the awning before the rain
    trigger:
      - platform: numeric_state
        entity_id: sensor.rain_nowcast_minutes_until_rain
        below: 15
    condition:
      - condition: numeric_state
        entity_id: sensor.rain_nowcast_peak_rate
        above: 1
    action:
      - service: cover.close_cover
        target: { entity_id: cover.awning }
```

## How good is it?

Backtested against real RMI radar (see the repository README for the full
method): CSI@1mm/h of 0.74 at 5 min, 0.40 at 30 min, 0.18 at 60 min, with a
held-out mean of 0.35. In plain terms — **the first half hour is genuinely
useful, the second half hour is a hint**. This is a known limit of pure
extrapolation: it moves existing rain around correctly but cannot predict
showers forming or dying.

The radar updates every 5 minutes, but the per-minute line is not a linear
fade between frames: values in between are found by tracing along the motion
field to where that rain was at the previous frame and where it lands at the
next. A shower therefore arrives at the minute it arrives, and a peak passing
between two frames is not flattened — on a synthetic test, tracing recovers
9.7 of a 10 mm/h peak where linear interpolation reports 7.3.

Point values are averaged over about 3 km around your location, because a
single 1 km pixel is mostly speckle.

## Data use

One 2.9 MB composite every 5 minutes, about 25 GB/month if left running
continuously. Downloads are cached in `/data/cache` and pruned after 3 hours.

## Troubleshooting

**"Radar update failed: ORD API rate limit hit"** — the key is missing or
wrong. Check the add-on log; anonymous access will not sustain a 5-min poll.

**Map is empty but the add-on is running** — it may simply not be raining.
Check the graph: if it reads "Dry for the next 60 minutes", that is the answer.

**Slow or high CPU** — lower `ensemble_members`.
