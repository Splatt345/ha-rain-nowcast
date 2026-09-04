"""Backtest a pysteps STEPS nowcast config against real RMI (Belgium) radar data.

Reusable as a library (score_config) so an optimizer can try many configs, and
as a CLI for a one-off run against the baseline config.
"""
import argparse
import glob
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pysteps-data",
                         "radar", "rmi", "radqpe", "20210704")
CACHE_NPY = os.path.join(os.path.dirname(__file__), "..", "data", "rmi_stack.npy")


def load_stack():
    if os.path.exists(CACHE_NPY):
        return np.load(CACHE_NPY)
    from pysteps import io
    files = sorted(glob.glob(f"{DATA_DIR}/*.hdf"))
    if not files:
        raise FileNotFoundError(
            f"No RMI radar frames in {DATA_DIR}. Run ./fetch_data.sh first."
        )
    fields, metadata = [], None
    for f in files:
        field, _, meta = io.import_odim_hdf5(f)
        fields.append(field)
        metadata = meta
    stack = np.stack(fields, axis=0)
    os.makedirs(os.path.dirname(CACHE_NPY), exist_ok=True)
    np.save(CACHE_NPY, stack)
    return stack


def csi_at_threshold(pred, obs, thr=1.0):
    p, o = pred > thr, obs > thr
    hits = np.sum(p & o)
    misses = np.sum(~p & o)
    false_alarms = np.sum(p & ~o)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


def score_config(config, R=None, n_obs=12, n_lead=12):
    """Run one STEPS backtest for `config` and return per-lead CSI/RMSE plus means.

    `config` keys (all optional, fall back to the validated baseline):
      n_ens_members, n_cascade_levels, ar_order, precip_thr, oflow_method,
      noise_method, ar_context_frames (how many trailing train frames feed STEPS)
    """
    from pysteps import motion, nowcasts, utils

    if R is None:
        R = load_stack()

    cfg = {
        "n_ens_members": 6,
        "n_cascade_levels": 6,
        "ar_order": 2,
        "precip_thr_db": -10.0,
        "oflow_method": "lucaskanade",
        "noise_method": "nonparametric",
        "ar_context_frames": 3,  # must be ar_order + 1
        **config,
    }

    train = R[:n_obs]
    truth = R[n_obs:n_obs + n_lead]

    # metadata is fixed for this dataset (RMI ODIM composite); avoid re-reading HDF5
    # just to get it when we already have a cached stack.
    metadata = {"unit": "mm/h", "transform": None, "zerovalue": 0.0,
                "threshold": 0.04034535586833954, "accutime": 5.0}

    train_db, _ = utils.transformation.dB_transform(
        train, metadata, threshold=0.1, zerovalue=-15.0
    )
    train_db[~np.isfinite(train_db)] = -15.0

    oflow = motion.get_method(cfg["oflow_method"])
    motion_field = oflow(train_db)

    nowcast_method = nowcasts.get_method("steps")
    forecast_db = nowcast_method(
        train_db[-cfg["ar_context_frames"]:],
        motion_field,
        n_lead,
        n_ens_members=cfg["n_ens_members"],
        n_cascade_levels=cfg["n_cascade_levels"],
        ar_order=cfg["ar_order"],
        precip_thr=cfg["precip_thr_db"],
        noise_method=cfg["noise_method"],
        kmperpixel=1.0,
        timestep=5,
        seed=42,
    )
    forecast = utils.transformation.dB_transform(
        forecast_db, metadata, threshold=cfg["precip_thr_db"], inverse=True
    )[0]
    forecast_mean = np.nanmean(forecast, axis=0)

    csi_scores, rmse_scores = [], []
    for t in range(n_lead):
        pred, obs = forecast_mean[t], truth[t]
        valid = np.isfinite(pred) & np.isfinite(obs)
        csi_scores.append(csi_at_threshold(pred[valid], obs[valid], thr=1.0))
        rmse_scores.append(np.sqrt(np.nanmean((pred[valid] - obs[valid]) ** 2)))

    return {
        "config": cfg,
        "csi_per_lead": csi_scores,
        "rmse_per_lead": rmse_scores,
        "mean_csi": float(np.nanmean(csi_scores)),
        "mean_rmse": float(np.nanmean(rmse_scores)),
    }


def run_forecast(train, metadata, config, n_lead):
    """Run STEPS on a raw mm/h `train` stack (most-recent-last) and return the
    ensemble-mean forecast in mm/h, shape (n_lead, y, x). No verification -
    used for live nowcasting where there's no future truth yet to score
    against (see score_config, which wraps this for backtesting).
    """
    from pysteps import motion, nowcasts, utils

    cfg = {**BASELINE_CONFIG, **config}

    train_db, _ = utils.transformation.dB_transform(
        train, metadata, threshold=0.1, zerovalue=-15.0
    )
    train_db[~np.isfinite(train_db)] = -15.0

    oflow = motion.get_method(cfg["oflow_method"])
    motion_field = oflow(train_db)

    nowcast_method = nowcasts.get_method("steps")
    forecast_db = nowcast_method(
        train_db[-cfg["ar_context_frames"]:],
        motion_field,
        n_lead,
        n_ens_members=cfg["n_ens_members"],
        n_cascade_levels=cfg["n_cascade_levels"],
        ar_order=cfg["ar_order"],
        precip_thr=cfg["precip_thr_db"],
        noise_method=cfg["noise_method"],
        kmperpixel=metadata.get("xpixelsize", 1000.0) / 1000.0,
        timestep=metadata.get("accutime", 5.0),
        seed=42,
    )
    forecast = utils.transformation.dB_transform(
        forecast_db, metadata, threshold=cfg["precip_thr_db"], inverse=True
    )[0]
    return np.nanmean(forecast, axis=0)


BASELINE_CONFIG = {
    # n_ens_members=12 per nowcast/optimize.py's guarded search (accepted:
    # more ensemble members lowers Monte Carlo noise in the ensemble mean).
    # Everything else here is the original, still-best validated baseline.
    "n_ens_members": 12,
    "n_cascade_levels": 6,
    "ar_order": 2,
    "precip_thr_db": -10.0,
    "oflow_method": "lucaskanade",
    "noise_method": "nonparametric",
    "ar_context_frames": 3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                         help="path to a JSON config file; defaults to the validated baseline")
    parser.add_argument("--n-obs", type=int, default=12)
    parser.add_argument("--n-lead", type=int, default=12)
    args = parser.parse_args()

    cfg = BASELINE_CONFIG
    if args.config:
        with open(args.config) as f:
            cfg = {**BASELINE_CONFIG, **json.load(f)}

    result = score_config(cfg, n_obs=args.n_obs, n_lead=args.n_lead)
    print(json.dumps(
        {k: v for k, v in result.items() if k != "config"}, indent=2
    ))
    print(f"\nmean_csi={result['mean_csi']:.4f}  mean_rmse={result['mean_rmse']:.4f}")
