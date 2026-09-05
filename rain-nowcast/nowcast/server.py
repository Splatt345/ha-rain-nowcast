"""Home Assistant add-on: live rain radar + minute-by-minute rain forecast.

Serves a small web UI on the add-on's ingress port. Everything expensive
happens on a background thread and requests only ever read the last finished
result, so a slow STEPS run never blocks the page.
"""
import datetime
import glob
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from nowcast import interpolate, live, render
from nowcast.backtest import BASELINE_CONFIG, run_forecast
from nowcast.publish import Publisher

WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
PORT = int(os.environ.get("PORT", "8099"))
POLL_SECONDS = 120  # how often we check for a new frame (cheap: one list call)
RAIN_THRESHOLD = 0.1  # mm/h; below this, call it dry
CACHE_KEEP_MINUTES = 180


def load_options():
    """Add-on options arrive as /data/options.json; env vars cover local runs."""
    opts = {}
    try:
        with open("/data/options.json") as f:
            opts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    def opt(key, default, cast):
        value = opts.get(key, os.environ.get(key.upper()))
        if value in (None, ""):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    api_key = opt("api_key", "", str) or os.environ.get("ORD_API_KEY", "")
    if api_key:
        os.environ["ORD_API_KEY"] = api_key  # live._session() picks it up here

    # options.json gives a real bool; an env var gives a string, and bool("false")
    # is True, so parse rather than cast.
    def yes_no(v):
        return str(v).strip().lower() not in ("0", "false", "no", "off")

    return {
        "latitude": opt("latitude", 50.85, float),
        "longitude": opt("longitude", 4.35, float),
        "history_minutes": opt("history_minutes", 60, int),
        "forecast_minutes": opt("forecast_minutes", 60, int),
        "ensemble_members": opt("ensemble_members", 12, int),
        "publish_sensors": opt("publish_sensors", True, yes_no),
    }


def prune_cache():
    cutoff = time.time() - CACHE_KEEP_MINUTES * 60
    for path in glob.glob(os.path.join(live.CACHE_DIR, "*.h5")):
        if os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
            except OSError:
                pass


class Nowcaster:
    def __init__(self, opts):
        self.opts = opts
        self.lock = threading.Lock()
        self.images = {}
        self.state = {"status": "starting"}
        self._last_url = None
        self.publisher = Publisher(opts["publish_sensors"])

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def image(self, name):
        with self.lock:
            return self.images.get(name)

    def refresh(self):
        """Rebuild everything if a new radar frame has landed. Returns whether
        it did any work."""
        opts = self.opts
        n_past = max(3, round(opts["history_minutes"] / 5))
        urls = live.find_latest_composites(
            n_frames=n_past, lookback_minutes=opts["history_minutes"] + 40
        )
        if urls[-1] == self._last_url and self.state.get("status") == "ok":
            return False

        stack, metadata, timestamps = live.get_live_frames(n_frames=n_past, urls=urls)
        interval = metadata["accutime"]
        n_lead = max(1, round(opts["forecast_minutes"] / interval))
        config = {**BASELINE_CONFIG, "n_ens_members": opts["ensemble_members"]}
        forecast, motion, members = run_forecast(stack, metadata, config, n_lead,
                                                 return_extras=True)

        last_time = timestamps[-1] or datetime.datetime.now(datetime.timezone.utc)
        rows, cols, _ = render.mercator_sampler(metadata, live.BELGIUM_BBOX)

        images, frames = {}, []

        def add(name, field, minutes, kind):
            images[name] = render.png_bytes(
                render.colorize(render.resample(field, rows, cols))
            )
            frames.append({
                "name": name,
                "kind": kind,
                "offset": int(round(minutes)),
                "time": (last_time + datetime.timedelta(minutes=minutes)).isoformat(),
            })

        for i, field in enumerate(stack):
            add(f"p{i}", field, -interval * (len(stack) - 1 - i), "past")
        for k, field in enumerate(forecast, start=1):
            add(f"f{k}", field, interval * k, "forecast")

        # Point series at one-minute resolution. Radar still only arrives every
        # `interval` minutes, but the values in between are obtained by tracing
        # along the motion field rather than fading linearly between frames -
        # so a shower arrives at the right minute instead of ramping up over
        # five, and a peak that passes between two frames isn't clipped.
        row, col = render.point_rowcol(metadata, opts["latitude"], opts["longitude"])
        ny, nx = stack.shape[1:]
        in_coverage = 0 <= row < ny and 0 <= col < nx
        row, col = int(np.clip(row, 0, ny - 1)), int(np.clip(col, 0, nx - 1))
        offsets = [f["offset"] for f in frames]
        minutes = np.arange(offsets[0], offsets[-1] + 1)

        sub_steps = int(round(interval))

        def point_curve(sequence):
            if sub_steps >= 1 and abs(interval - sub_steps) < 1e-6:
                candidate = interpolate.point_series(sequence, motion, row, col,
                                                     sub_steps=sub_steps)
                if len(candidate) == len(minutes):
                    return candidate
            # odd cadence: fall back to the plain time blend
            return np.interp(minutes, offsets,
                             render.point_series(sequence, row, col))

        series = point_curve(np.concatenate([stack, forecast]))

        # The same curve for each ensemble member. STEPS runs 12 of them and we
        # were averaging them away: the mean is a smoothed field that measurably
        # under-calls peaks, so thresholding it answers a question nobody asked.
        # How many members are wet at a given minute *is* a probability, needs
        # nothing fitted, and is what the ensemble was run for in the first
        # place. Cheap - each curve samples a 3x3 box, not the whole grid.
        member_curves = np.array([point_curve(np.concatenate([stack, m]))
                                  for m in members])
        probability = (member_curves >= RAIN_THRESHOLD).mean(axis=0)
        del members, member_curves

        state = {
            "status": "ok",
            "updated": last_time.isoformat(),
            "bounds": [
                [live.BELGIUM_BBOX[1], live.BELGIUM_BBOX[0]],
                [live.BELGIUM_BBOX[3], live.BELGIUM_BBOX[2]],
            ],
            "location": [opts["latitude"], opts["longitude"]],
            "in_coverage": in_coverage,
            "interval_minutes": interval,
            "n_members": config["n_ens_members"],
            "frames": frames,
            "now_frame": len(stack) - 1,
            "series": {
                "minutes": minutes.tolist(),
                "values": [round(float(v), 3) for v in series],
                "probability": [round(float(p), 2) for p in probability],
            },
            "rain_chance": round(float(probability[int(np.searchsorted(minutes, 0)):].max()), 2),
            "summary": self._summary(minutes, series, probability),
        }

        with self.lock:
            self.images, self.state = images, state
        self._last_url = urls[-1]

        # A broker hiccup must not cost us the forecast we just computed.
        try:
            self.publisher.publish(state)
        except Exception:
            traceback.print_exc()

        prune_cache()
        return True

    @staticmethod
    def _summary(minutes, series, probability):
        now = int(np.searchsorted(minutes, 0))
        future, ahead = series[now:], minutes[now:]
        chance = int(round(100 * probability[now:].max()))
        if future[0] >= RAIN_THRESHOLD:
            dry = np.flatnonzero(future < RAIN_THRESHOLD)
            stops = f", stopping in {int(ahead[dry[0]])} min" if dry.size else ""
            return f"Raining now, {future[0]:.1f} mm/h{stops}"
        wet = np.flatnonzero(future >= RAIN_THRESHOLD)
        if wet.size:
            return (f"Rain starting in {int(ahead[wet[0]])} min ({chance}% chance), "
                    f"up to {future[wet[0]:].max():.1f} mm/h")
        # The mean can read dry while a third of the members rain on you - that
        # is exactly the case the old wording got confidently wrong.
        if chance >= 20:
            return (f"Probably dry, but a {chance}% chance of rain "
                    f"in the next {int(ahead[-1])} min")
        return f"Dry for the next {int(ahead[-1])} minutes"

    def loop(self):
        while True:
            try:
                self.refresh()
            except Exception as exc:  # keep serving the last good result
                traceback.print_exc()
                with self.lock:
                    self.state = {**self.state, "error": str(exc)}
                    if self.state.get("status") != "ok":
                        self.state["status"] = "error"
            time.sleep(POLL_SECONDS)


class Handler(BaseHTTPRequestHandler):
    nowcaster = None
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Ingress strips its own path prefix before proxying, and the page uses
        # relative URLs, so these paths arrive clean either way.
        path = self.path.split("?")[0].strip("/")
        if path in ("", "index.html"):
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "api/state":
            body = json.dumps(self.nowcaster.snapshot()).encode()
            return self._send(200, body, "application/json")
        if path.startswith("frames/") and path.endswith(".png"):
            image = self.nowcaster.image(path[len("frames/"):-len(".png")])
            if image is not None:
                return self._send(200, image, "image/png", cache="max-age=300")
        self._send(404, b"not found", "text/plain")

    def log_message(self, *args):
        pass


def main():
    opts = load_options()
    os.makedirs(live.CACHE_DIR, exist_ok=True)
    print(f"Rain Nowcast starting on :{PORT} for {opts['latitude']},{opts['longitude']}")

    nowcaster = Nowcaster(opts)
    threading.Thread(target=nowcaster.loop, daemon=True).start()
    Handler.nowcaster = nowcaster
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def _demo():
    """Self-check for the parts that need no radar, broker or network."""
    minutes = np.arange(-55, 61)
    n = len(minutes)
    flat = lambda v: np.full(n, v, dtype=float)

    s = Nowcaster._summary(minutes, flat(0.0), flat(0.0))
    assert s.startswith("Dry for the next"), s
    # the case the ensemble exists for: the mean reads dry, a third of the
    # members rain on you anyway, and the old wording called that "Dry".
    s = Nowcaster._summary(minutes, flat(0.0), flat(1 / 3))
    assert s.startswith("Probably dry") and "33% chance" in s, s
    s = Nowcaster._summary(minutes, flat(2.0), flat(1.0))
    assert s.startswith("Raining now, 2.0 mm/h") and "stopping" not in s, s
    s = Nowcaster._summary(minutes, np.where(minutes >= 20, 4.0, 0.0),
                           np.where(minutes >= 20, 0.75, 0.0))
    assert "starting in 20 min (75% chance)" in s and "up to 4.0" in s, s

    # The contract refresh() leans on: the extras really are the members the
    # mean was taken over, so the probability band and the rate line above it
    # describe the same forecast.
    rng = np.random.default_rng(0)
    train = np.abs(rng.gamma(1.0, 2.0, size=(4, 64, 64)))
    meta = {"unit": "mm/h", "transform": None, "zerovalue": 0.0,
            "threshold": 0.01, "accutime": 5.0,
            "xpixelsize": 1000.0, "ypixelsize": 1000.0}
    mean, motion, members = run_forecast(
        train, meta, {**BASELINE_CONFIG, "n_ens_members": 4}, 3, return_extras=True)
    assert members.shape == (4, 3, 64, 64), members.shape
    assert motion.shape == (2, 64, 64), motion.shape
    assert np.allclose(np.nanmean(members, axis=0), mean, equal_nan=True)
    print("server self-check ok")


if __name__ == "__main__":
    import sys

    _demo() if "--selfcheck" in sys.argv else main()
