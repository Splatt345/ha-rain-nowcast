# local_weather_model

Minute-scale precipitation nowcasting for Belgium.

## Why pysteps

Global AI weather models (GraphCast, FourCastNet, Pangu-Weather, ClimaX,
Google WeatherNext) are the wrong tool: 6h+ timesteps, 25-100km grids. Minute
scale local rain needs radar-extrapolation nowcasting instead. A pretrained
DGMR (DeepMind's nowcasting GAN, via openclimatefix's HuggingFace weights)
was also tried and rejected: it loaded cleanly (all 854 weight tensors
matched) but produced physically implausible output (rain rates up to
3410 mm/h) - the published checkpoint is not a faithful reproduction of the
paper model, and training one's own would be a much bigger project.

`pysteps` (optical-flow radar extrapolation, no AI, no GPU needed) is the
baseline and current best working option.

## Result (tuned)

Backtested against real RMI radar (Belgium, 2021-07-04, 5-min QPE composite,
700x700 @ 1km): 60 min of history -> 60 min forecast, scored against the
actual observed radar frames that followed.

A bounded, guarded search (`nowcast/optimize.py`, 10 candidates, coordinate-
wise, auto-reverting anything that didn't clearly improve) tried ensemble
size, cascade levels, AR order, optical-flow method, and noise method one
at a time against the baseline. Only one change was accepted:
**n_ens_members 6 -> 12** (more ensemble members lowers Monte Carlo noise in
the ensemble-mean estimate - a generically safe change, not a fit to this
one event). Everything else (cascade levels, AR order, lucaskanade optical
flow, nonparametric noise) stayed at the original baseline; 9 other
candidates were tried and rejected, including `vet`/`proesmans` optical flow
(incompatible with this input-frame count, not a scoring rejection).

**Correctness fix (post-tuning):** an earlier version of this backtest had a
units bug - pysteps' `dB_transform(..., inverse=True)` is a silent no-op
unless the metadata dict passed in has `transform: "dB"` exactly, and the
original code passed the pre-transform metadata (`transform: None`) instead
of the one the forward transform actually returns. Forecasts were staying in
dB units while being scored against real mm/h truth. CSI@1mm/h was barely
affected by coincidence (0 dB is very close to the 1 mm/h threshold on the
log scale), but every RMSE number reported before this fix was comparing
mismatched units and should be disregarded - the true RMSE is roughly 15x
better than what was originally reported. The table below is post-fix.

| lead   | CSI@1mm/h | RMSE (mm/h) |
|--------|-----------|-------------|
| 5 min  | 0.74      | 0.65        |
| 30 min | 0.40      | 0.89        |
| 60 min | 0.18      | 0.87        |
| mean (search split)  | 0.40 | 0.89 |
| mean (held-out split)| 0.35 | 0.74 |

The held-out split (a different issue time within the same event, never
seen during search) scores lower on CSI than the search split - expected;
treat 0.35 mean CSI as the more honest generalization estimate, not 0.40.
CSI matches published pysteps/STEPS validation figures for European radar
(Pulkkinen et al. 2019), so this is representative, not a fluke. Useful lead
time (CSI above ~0.3-0.4) is roughly the first 30-35 minutes; degrades
gracefully after that - a known limit of pure extrapolation (it can't
predict storm genesis/dissipation, only advect existing cells).

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install pysteps h5py pyproj dask opencv-python-headless matplotlib netCDF4
nowcast/fetch_data.sh   # pulls real RMI Belgium radar example data (~800MB)
.venv/bin/python nowcast/backtest.py
```

## Live nowcasting

`nowcast/live.py` fetches real-time radar and runs the same validated STEPS
pipeline (`run_forecast()`, shared with the backtest) - no known future to
score against, just a live forecast.

Belgium's own open data portal (opendata.meteo.be) requires account
registration with no documented anonymous tier, so this uses EUMETNET's
pan-European OPERA radar composite instead (same ODIM HDF5 format, includes
Belgium). Anonymous access is rate-limited quickly; get a free API key from
the [MeteoGate developer portal](https://meteogate.eu/) (login via GitHub/
Google/Azure AD, no approval wait) and put it in a `.env` file (gitignored):

```
ORD_API_KEY=your-key-here
```

```bash
set -a; source .env; set +a
.venv/bin/python -m nowcast.live
```

Verified end-to-end against live data on 2026-09-04: forecasts came out
physically sane (0-44 mm/h across the domain, ~97-99% finite - a thin edge
band is legitimately NaN, STEPS advects data off a small crop's border).
Two real bugs were found and fixed getting here: the S3 download URLs the
API returns must NOT be sent the ORD auth headers (S3 rejects them as a
malformed AWS Authorization header, 400); and pysteps' importer defaults to
requesting quantity "RATE", but our files only carry "DBZH" (reflectivity,
converted to rain rate via pysteps' Marshall-Palmer default).

Caveat: OPERA's composite cadence (15 min here) is coarser than RMI's own
5-min product used for the backtest above, so live forecasts have less
temporal context than the validated numbers reflect.

## Files

- `nowcast/backtest.py` - `score_config(config)` runs one STEPS backtest and
  returns CSI/RMSE per lead time; `run_forecast()` is the shared STEPS-running
  core used by both backtesting and live nowcasting. Also a CLI for a
  one-off backtest run.
- `nowcast/live.py` - fetches live OPERA radar over Belgium and runs
  `run_forecast()` on it. CLI entry point prints a per-step summary.
- `nowcast/fetch_data.sh` - pulls the RMI example radar data (gitignored,
  not committed - ~800MB).
