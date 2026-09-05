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

from nowcast import live, render
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
        forecast = run_forecast(stack, metadata, config, n_lead)

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

        # Point series, resampled to the minute. The radar itself only updates
        # every `interval` minutes, so the between-frame values are linear
        # interpolation, not extra information.
        row, col = render.point_rowcol(metadata, opts["latitude"], opts["longitude"])
        ny, nx = stack.shape[1:]
        in_coverage = 0 <= row < ny and 0 <= col < nx
        row, col = int(np.clip(row, 0, ny - 1)), int(np.clip(col, 0, nx - 1))
        values = np.concatenate([
            render.point_series(stack, row, col),
            render.point_series(forecast, row, col),
        ])
        offsets = [f["offset"] for f in frames]
        minutes = np.arange(offsets[0], offsets[-1] + 1)
        series = np.interp(minutes, offsets, values)

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
            "frames": frames,
            "now_frame": len(stack) - 1,
            "series": {
                "minutes": minutes.tolist(),
                "values": [round(float(v), 3) for v in series],
            },
            "summary": self._summary(minutes, series),
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
    def _summary(minutes, series):
        now = int(np.searchsorted(minutes, 0))
        future, ahead = series[now:], minutes[now:]
        if future[0] >= RAIN_THRESHOLD:
            dry = np.flatnonzero(future < RAIN_THRESHOLD)
            stops = f", stopping in {int(ahead[dry[0]])} min" if dry.size else ""
            return f"Raining now, {future[0]:.1f} mm/h{stops}"
        wet = np.flatnonzero(future >= RAIN_THRESHOLD)
        if wet.size:
            peak = future[wet[0]:].max()
            return f"Rain starting in {int(ahead[wet[0]])} min, up to {peak:.1f} mm/h"
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


if __name__ == "__main__":
    main()
