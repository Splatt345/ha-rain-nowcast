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

## How good is it?

Backtested against real RMI radar (see the repository README for the full
method): CSI@1mm/h of 0.74 at 5 min, 0.40 at 30 min, 0.18 at 60 min, with a
held-out mean of 0.35. In plain terms — **the first half hour is genuinely
useful, the second half hour is a hint**. This is a known limit of pure
extrapolation: it moves existing rain around correctly but cannot predict
showers forming or dying.

The radar updates every 5 minutes. The per-minute line between those updates
is interpolated, not extra information, and point values are averaged over
about 3 km around your location because a single 1 km pixel is mostly speckle.

## Data use

One 2.9 MB composite every 5 minutes, about 25 GB/month if left running
continuously. Downloads are cached in `/data/cache` and pruned after 3 hours.

## Troubleshooting

**"Radar update failed: ORD API rate limit hit"** — the key is missing or
wrong. Check the add-on log; anonymous access will not sustain a 5-min poll.

**Map is empty but the add-on is running** — it may simply not be raining.
Check the graph: if it reads "Dry for the next 60 minutes", that is the answer.

**Slow or high CPU** — lower `ensemble_members`.
