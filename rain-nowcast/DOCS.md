# Rain Nowcast

Live rain radar over Belgium plus a 60-minute forecast, as a Home Assistant
panel. The forecast is real radar extrapolation (pysteps STEPS), not a
re-serving of somebody else's API.

## Installation

**This add-on is amd64 only.** A Raspberry Pi cannot run it — `aarch64` is not
built, so the add-on will not appear in your store at all on one. That is
deliberate: the architecture used to be advertised and then fail at install
with nothing explaining why.

Either copy the `rain-nowcast` folder to the `/addons` share (via the *Samba
share* or *Advanced SSH & Web Terminal* add-on) and use
**Settings → Add-ons → Add-on Store → ⋮ → Check for updates**, where it appears
under **Local add-ons** — or add the repository under
**⋮ → Repositories** if you have pushed it somewhere reachable.

Then install **Rain Nowcast**, set your latitude/longitude and API key, and
start it. The panel is added to the sidebar automatically.

The first build compiles pysteps from source and takes several minutes. The
first start then downloads an hour of radar (~35 MB) before it can forecast
anything. Budget ~1.1 GB of disk and 0.5-0.7 GB of RAM.

## Getting an API key

The radar comes from EUMETNET's OPERA composite via the MeteoGate Open Radar
Data API. Anonymous access is rate-limited to the point of being unusable for
a 5-minute poll, so register a free key at <https://meteogate.eu/> (sign in
with GitHub/Google/Azure AD, no approval wait) and paste it into `api_key`.

## Options

Set these in the add-on's **Configuration** tab, where each one is labelled and
explained. The two that most often go wrong: `latitude`/`longitude` must be
inside the radar area, and `mqtt_host` must be left **empty** on Home Assistant
OS, where the Supervisor supplies the broker.

| Option | Default | What it does |
|---|---|---|
| `latitude` / `longitude` | Brussels | The point the minute-by-minute graph is for. Must be inside the radar crop (roughly Belgium plus a margin: 2.0–6.8°E, 49.0–51.8°N). |
| `api_key` | – | MeteoGate ORD API key. |
| `history_minutes` | 60 | How much radar history to show and feed the nowcast. |
| `forecast_minutes` | 60 | How far ahead to forecast. Skill is largely gone past ~35 min. |
| `ensemble_members` | 12 | STEPS ensemble size. A full cycle takes ~12 s on a desktop CPU; lower it to 4–6 on a slow machine, at the cost of a coarser `rain_chance` (it can only report in steps of 1/N). |
| `publish_sensors` | true | Publish the forecast as Home Assistant entities. |
| `use_model` | true | Combine the STEPS ensemble with the bundled SmaAt-UNet neural network. Adds ~0.25 s per cycle. Turn it off for plain STEPS. |
| `mqtt_host` | – | **Leave empty on Home Assistant OS/Supervised** — the Supervisor supplies the broker. Only needed on Home Assistant Container or docker-compose, where there is no Supervisor and sensors would otherwise not publish at all. |
| `mqtt_port` | 1883 | With `mqtt_host`. |
| `mqtt_username` / `mqtt_password` | – | With `mqtt_host`, if your broker needs them. |

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

### A dashboard card instead of this panel

The repository's integration ships a Lovelace card with everything this panel
has - animated radar, pan and zoom, colour schemes, opacity, the per-minute
chart and tap-to-forecast-elsewhere - so the radar can live on a dashboard.
Add `/rain_nowcast_static/rain-nowcast-card.js` as a JavaScript-module resource
and use `type: custom:rain-nowcast-card`. See the repository README.

### No MQTT broker? Use the integration instead

The repository ships a custom integration (`custom_components/rain_nowcast`)
that polls this add-on over HTTP and creates the same eight entities with no
broker at all. Copy it to `/config/custom_components/`, restart Home Assistant,
add **Rain Nowcast** under Devices & Services, and set `publish_sensors: false`
here so the entities are not created twice.

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

Measured on **20 independent Belgian events** (2021–2024, KNMI 1 km 5-minute
composites), scored only on the 10 the model was never tuned against:

| lead | 5 min | 15 min | 30 min | 45 min | 60 min |
|---|---|---|---|---|---|
| CSI@1mm/h | 0.72 | 0.55 | 0.41 | 0.31 | 0.25 |

Held-out mean 0.404. In plain terms — **the first half hour is genuinely
useful, the second half hour is a hint**. This is a known limit of pure
extrapolation: it moves existing rain around correctly but cannot predict
showers forming or dying.

It does not invent rain. On 10 near-dry events it predicted wet pixels 23×
*less* often than they actually occurred: the failure mode is being too
cautious, not crying wolf.

### What `rain_chance` is worth, and where it isn't

The probability was scored, not assumed (Brier skill score 0.44 at 1 mm/h).
Two honest limits worth knowing before you automate on it:

- **Above about 40% it over-promises.** When it says 75% it rains 68% of the
  time; at 92% it rains 82%. Below 40% it is accurate or slightly pessimistic.
- **It has essentially no skill about *heavy* rain.** At the 5 mm/h level the
  measured resolution collapses to almost zero. The system can tell you whether
  it will rain, not whether it will rain hard — so `peak_rate` is fine for
  "bring the washing in" and should not carry a flood-response automation.

The shaded band on the chart is the ensemble's own disagreement about the rate,
and it is a real signal: wider band genuinely means larger error (correlation
0.57). It is, however, about 21% narrower than the errors actually are, so treat
it as the optimistic edge of the uncertainty rather than the full extent.

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
