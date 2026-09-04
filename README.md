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

## Baseline result

Backtested against real RMI radar (Belgium, 2021-07-04, 5-min QPE composite,
700x700 @ 1km): 60 min of history -> 60 min forecast, scored against the
actual observed radar frames that followed.

| lead   | CSI@1mm/h | RMSE (mm/h) |
|--------|-----------|-------------|
| 5 min  | 0.71      | 13.8        |
| 30 min | 0.39      | 13.8        |
| 60 min | 0.21      | 13.7        |
| mean   | 0.40      | 13.8        |

Matches published pysteps/STEPS validation figures for European radar
(Pulkkinen et al. 2019), so this single-event result is representative, not
a fluke. Useful lead time (CSI above ~0.3-0.4) is roughly the first 30-35
minutes; degrades gracefully after that - a known limit of pure extrapolation
(it can't predict storm genesis/dissipation, only advect existing cells).

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install pysteps h5py pyproj dask opencv-python-headless matplotlib netCDF4
nowcast/fetch_data.sh   # pulls real RMI Belgium radar example data (~800MB)
.venv/bin/python nowcast/backtest.py
```

## Files

- `nowcast/backtest.py` - `score_config(config)` runs one STEPS backtest and
  returns CSI/RMSE per lead time. Also a CLI for a one-off run.
- `nowcast/fetch_data.sh` - pulls the RMI example radar data (gitignored,
  not committed - ~800MB).
