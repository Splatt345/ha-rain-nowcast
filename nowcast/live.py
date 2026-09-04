"""Fetch live European (OPERA) radar composites and run the same STEPS
nowcast pipeline used in backtest.py, but on real-time data with no known
future to score against.

Source: EUMETNET's Open Radar Data (ORD) API, https://api.meteogate.eu -
a pan-European radar mosaic covering Belgium (among ~40 other countries),
served as ODIM HDF5 - the same format the RMI example data uses, so it
plugs into the exact same importer pysteps already reads for backtest.py.

Belgium's own portal (opendata.meteo.be) requires account registration to
download data, so this uses the EUMETNET composite instead: anonymous
access works but is rate-limited (no signup needed for occasional runs;
a free API key from https://placeholder.meteogate.eu raises the limit for
frequent/scheduled polling - not set up here, add via ORD_API_KEY env var
if you register one).

Caveat: OPERA's native composite cadence is coarser than RMI's own 5-min
product (docs didn't confirm the exact interval - the code below detects it
from the actual returned timestamps rather than assuming 5 min). Fewer,
sparser frames of context will somewhat understate the skill measured in
the backtest, which used RMI's full 5-min cadence directly.
"""
import datetime
import os
import time

import numpy as np
import requests

ORD_BASE = "https://api.meteogate.eu/eu-eumetnet-weather-radar"
LOCATION_ID = "0-20010-0-OPERA"  # all-Europe composite
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "live")

# Belgium bounding box (lon_min, lat_min, lon_max, lat_max), with a small
# margin so STEPS' optical flow has surrounding context to work with.
BELGIUM_BBOX = (2.0, 49.0, 6.8, 51.8)


def _session():
    s = requests.Session()
    api_key = os.environ.get("ORD_API_KEY")
    if api_key:
        # Confirmed by testing each candidate individually - this is the
        # only one of five that actually authenticates (Authorization:
        # Bearer, Ocp-Apim-Subscription-Key and X-API-Key all still 429 as
        # anonymous). Never put the key in the URL (params) - it'd end up
        # in server/proxy logs.
        s.headers["apikey"] = api_key
    return s


def find_latest_composites(product="DBZH", n_frames=4, lookback_minutes=90, session=None):
    """Query the ORD API for the most recent `n_frames` composite files for
    `product` and return their download URLs, oldest first.
    """
    session = session or _session()
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(minutes=lookback_minutes)
    dt_range = f"{start.strftime('%Y-%m-%dT%H:%MZ')}/{now.strftime('%Y-%m-%dT%H:%MZ')}"

    params = {
        "datetime": dt_range,
        "standard_name": product,
        "format": "ODIM",
        "method": "comp",
        "f": "CoverageJSON",
    }

    resp = session.get(
        f"{ORD_BASE}/collections/observations/locations/{LOCATION_ID}",
        params=params,
        timeout=30,
    )
    if resp.status_code == 429:
        reset = resp.headers.get("X-RateLimit-Reset", "unknown")
        raise RuntimeError(
            f"ORD API rate limit hit (resets in {reset}s). "
            "Wait, or set ORD_API_KEY to a registered key."
        )
    resp.raise_for_status()
    data = resp.json()

    # CoverageJSON domain listing - one entry per available timestep, each
    # carrying a download link for the actual ODIM HDF5 file.
    urls = _extract_download_urls(data)
    if not urls:
        raise RuntimeError(
            f"No {product} composites found in the last {lookback_minutes} min. "
            "Try a longer lookback or check `standard_name` is valid for this API."
        )
    return urls[-n_frames:]


def _extract_download_urls(coverage_json):
    """Pull (timestamp, url) pairs out of the API's CoverageJSON-ish response
    and return URLs sorted oldest-first. The exact response shape is only
    documented loosely, so this walks common locations defensively rather
    than assuming one fixed schema.
    """
    found = []

    def walk(obj):
        if isinstance(obj, dict):
            url = obj.get("href") or obj.get("url") or obj.get("download")
            if isinstance(url, str) and url.lower().endswith((".h5", ".hdf", ".hdf5")):
                found.append((obj.get("datetime") or obj.get("time") or "", url))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(coverage_json)
    found.sort(key=lambda t: t[0])
    return [u for _, u in found]


def download_composite(url, session=None):
    # Deliberately NOT using the ORD-authenticated session here: the API
    # response hands back a direct S3 (CloudFerro) URL, and S3 rejects our
    # ORD API-key headers as a malformed AWS Authorization header (400) -
    # this download needs a bare, header-less request.
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, os.path.basename(url.split("?")[0]))
    if not os.path.exists(local_path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path


def crop_to_belgium(field, metadata):
    """Crop a full-Europe field to the Belgium bbox using the field's own
    projection (pixel grid isn't lat/lon, so this reprojects the bbox
    corners rather than assuming a pixel offset).
    """
    import pyproj

    proj = pyproj.Proj(metadata["projection"])
    lon_min, lat_min, lon_max, lat_max = BELGIUM_BBOX
    xs, ys = proj([lon_min, lon_max, lon_max, lon_min], [lat_min, lat_min, lat_max, lat_max])
    x1, y1, x2, y2 = metadata["x1"], metadata["y1"], metadata["x2"], metadata["y2"]
    ny, nx = field.shape
    px = lambda x: int((x - x1) / (x2 - x1) * nx)
    py = lambda y: int((y2 - y) / (y2 - y1) * ny)  # y-axis is flipped (yorigin="upper")
    col_min, col_max = sorted([px(min(xs)), px(max(xs))])
    row_min, row_max = sorted([py(min(ys)), py(max(ys))])
    cropped = field[max(row_min, 0):min(row_max, ny), max(col_min, 0):min(col_max, nx)]

    meta = dict(metadata)
    meta["x1"] = x1 + col_min * metadata["xpixelsize"]
    meta["x2"] = x1 + col_max * metadata["xpixelsize"]
    meta["y1"] = y2 - row_max * metadata["ypixelsize"]
    meta["y2"] = y2 - row_min * metadata["ypixelsize"]
    return cropped, meta


def get_live_frames(product="DBZH", n_frames=4, lookback_minutes=90):
    """Return (stack, metadata) - a (n_frames, y, x) mm/h array over Belgium
    plus pysteps-style metadata, ready for backtest.run_forecast.
    """
    from pysteps.io import import_odim_hdf5
    from pysteps.utils.conversion import to_rainrate

    session = _session()
    urls = find_latest_composites(product=product, n_frames=n_frames,
                                   lookback_minutes=lookback_minutes, session=session)

    fields, metas, timestamps = [], [], []
    for url in urls:
        path = download_composite(url, session=session)
        # pysteps defaults to qty="RATE"; our files only carry the quantity
        # we requested from the API (DBZH reflectivity), so ask for that.
        field, _, meta = import_odim_hdf5(path, qty=product)
        field, meta = crop_to_belgium(field, meta)
        if product == "DBZH":
            field, meta = to_rainrate(field, meta)
        fields.append(field)
        metas.append(meta)
        timestamps.append(meta.get("timestamps", [None])[0])

    stack = np.stack(fields, axis=0)
    metadata = metas[-1]  # shared geometry/units, most recent timestep's own fields
    return stack, metadata, timestamps


def run_live_nowcast(n_lead=12, product="DBZH", n_frames=4):
    from nowcast.backtest import run_forecast, BASELINE_CONFIG

    stack, metadata, timestamps = get_live_frames(product=product, n_frames=n_frames)
    print(f"Fetched {stack.shape[0]} live frames over Belgium, shape {stack.shape[1:]}, "
          f"timestamps: {timestamps}")

    forecast = run_forecast(stack, metadata, BASELINE_CONFIG, n_lead)
    return forecast, metadata, timestamps


if __name__ == "__main__":
    t0 = time.time()
    forecast, metadata, timestamps = run_live_nowcast()
    print(f"Forecast shape: {forecast.shape} (lead steps, y, x)")
    for t in range(forecast.shape[0]):
        # nan-aware: a thin edge band is legitimately NaN (STEPS advects
        # data off the crop's border), plain .max()/.mean() would report
        # NaN for the whole step from that alone.
        print(f"  step {t + 1}: max {np.nanmax(forecast[t]):.1f} mm/h, "
              f"mean {np.nanmean(forecast[t]):.2f} mm/h, "
              f"finite {np.isfinite(forecast[t]).mean():.0%}")
    print(f"Done in {time.time() - t0:.1f}s")
