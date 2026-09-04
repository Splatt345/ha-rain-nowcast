# ha-rain-nowcast

A Home Assistant add-on that shows live rain radar over Belgium and forecasts
the next hour, minute by minute.

Forked from `local_weather_model`, which is the research side of this:
figuring out *which* nowcasting method actually works and measuring how well.
This repo is that method packaged as something you can click on.

- **Radar view** — the last hour of real radar, animated, on a real map, plus
  the next hour extrapolated forward. Scrub or let it loop.
- **Rain graph** — rain rate at your exact location, per minute, past hour
  through next hour, with a plain-language summary ("Rain starting in 25 min,
  up to 3.4 mm/h").

Install and configuration: [`rain-nowcast/DOCS.md`](rain-nowcast/DOCS.md).

## What's behind it

`pysteps` STEPS: optical-flow radar extrapolation with a stochastic ensemble.
No AI, no GPU, no cloud service — it runs on the Home Assistant box in about
12 seconds per cycle. Global AI weather models (GraphCast, Pangu-Weather,
Google WeatherNext) are the wrong tool here: 6h+ timesteps on 25–100 km grids.
DeepMind's DGMR is the right *class* of model, but the only public checkpoint
produces physically impossible output (rain rates over 3000 mm/h) and lost
badly to pysteps in a head-to-head backtest — 0.055 CSI against 0.52.

Radar comes from EUMETNET's OPERA composite via the MeteoGate API, at 5-minute
cadence — the same cadence the accuracy numbers were validated at.

## Accuracy

| lead | CSI@1mm/h | RMSE (mm/h) |
|--------|-----------|-------------|
| 5 min | 0.74 | 0.65 |
| 30 min | 0.40 | 0.89 |
| 60 min | 0.18 | 0.87 |
| mean (held-out) | 0.35 | 0.74 |

Useful skill runs to roughly 30–35 minutes, then degrades gracefully. That
matches published STEPS validation for European radar (Pulkkinen et al. 2019),
so it is representative rather than a lucky event. Extrapolation cannot
predict storm genesis or decay — only advect what already exists.

Two attempts to push past this were made and honestly failed to clear the bar:
a guarded hyperparameter search (0.397 → 0.403 CSI, essentially noise) and
NWP blending via `pysteps.blending.steps` (+0.002 CSI, 13% worse RMSE).

## Layout

```
repository.yaml            Home Assistant add-on repository manifest
rain-nowcast/
  config.yaml  build.yaml  Dockerfile   add-on packaging
  DOCS.md                               user-facing docs
  web/index.html                        the UI (Leaflet map + canvas chart)
  nowcast/
    server.py       background refresh loop + HTTP server
    render.py       radar field -> Web-Mercator PNG overlay, point series
    live.py         fetch OPERA composites, crop to Belgium
    backtest.py     the shared STEPS core, plus offline scoring
    optimize.py     the guarded hyperparameter search
```

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r rain-nowcast/requirements.txt
cd rain-nowcast
ORD_API_KEY=... LATITUDE=50.85 LONGITUDE=4.35 ../.venv/bin/python -m nowcast.server
# http://localhost:8099
```

`python -m nowcast.render` runs the rendering self-check.
