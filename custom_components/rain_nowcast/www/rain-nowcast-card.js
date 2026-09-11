/*
 * Rain Nowcast Lovelace card.
 *
 * Everything the add-on's own web panel shows, on a dashboard: the animated
 * radar, the per-minute rain graph with its uncertainty fan, and the summary.
 *
 * Two data paths, for two different reasons:
 *
 *   - the CHART comes from the summary entity's attributes, which the
 *     integration already keeps up to date. No extra requests, and it keeps
 *     working from anywhere Home Assistant works.
 *   - the RADAR FRAMES come through the integration's proxy, because the
 *     browser generally cannot reach the add-on: on Home Assistant OS it has no
 *     published port and sits on the Supervisor's internal network.
 *
 * PNG frames carry an 8-bit rain LEVEL and no colour at all - the palette and
 * alpha tables arrive in the state and are applied here on a canvas, so
 * changing scheme is a repaint of frames already fetched.
 *
 * The timeline runs a MINUTE at a time even though radar arrives every five.
 * The in-between minutes are advected here, in the browser, from the two
 * neighbouring frames along the motion field the state ships - the same trick
 * the add-on uses for the point curve, and for the same reason: rain does not
 * fade in, it moves, so cross-fading two frames makes a shower arrive over five
 * minutes instead of at the minute it arrives. Doing it server-side would mean
 * 121 PNGs per cycle instead of 25, measured at 17 s of CPU and 8 MB per
 * refresh; warping one frame here costs 9 ms and no bandwidth.
 */

const CARD = "rain-nowcast-card";
const HOME_ZOOM = 9;          // roughly 100 km across: the useful nowcast range
const STEP_MS = 80;           // one minute of forecast per frame of animation

const rel = (m) => (m === 0 ? "now" : m < 0 ? `${-m} min ago` : `in ${m} min`);

// Leaflet is served by the integration, not a CDN, and is loaded exactly once
// no matter how many cards are on the dashboard. A promise rather than a
// "script tag already there?" check: two cards built in the same frame, or one
// rebuilt while the first load was still in flight, both used to find the tag
// present, skip it, and then wait forever for an onload that was another
// card's - which is why the map sometimes simply never appeared.
let _leaflet = null;
function loadLeaflet() {
  if (window.L) return Promise.resolve();
  if (!_leaflet) {
    _leaflet = new Promise((ok, fail) => {
      const js = document.createElement("script");
      js.id = "rain-nowcast-leaflet";
      js.src = "/rain_nowcast_static/leaflet.js";
      js.onload = ok;
      js.onerror = () => { _leaflet = null; fail(new Error("cannot load Leaflet")); };
      document.head.appendChild(js);
    });
  }
  return _leaflet;
}

class RainNowcastCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._minute = 0;           // offset from now, in minutes: the whole timeline
    this._playing = true;
    this._raw = new Map();      // name -> decoded level PNG
    this._tinted = new Map();   // name -> data URL under the current palette
    this._levels = new Map();   // name -> the level plane, for advecting
    this._canvas = document.createElement("canvas");
    this._built = false;
  }

  static getConfigElement() { return document.createElement(CARD + "-editor"); }
  static getStubConfig() { return { type: `custom:${CARD}` }; }
  getCardSize() { return 8; }

  setConfig(config) {
    this._config = { height: 320, show_chart: true, ...config };
    // `height` is interpolated into a <style> block below, and config comes
    // from the dashboard YAML, so it is coerced to a number rather than
    // trusted - a string could otherwise close the style element.
    const h = Number.parseInt(this._config.height, 10);
    this._config.height = Number.isFinite(h) ? Math.min(Math.max(h, 120), 1200) : 320;
    if (this._config.entity && typeof this._config.entity !== "string") {
      throw new Error("entity must be an entity id");
    }
  }

  set hass(hass) {
    const first = !this._hass;
    this._hass = hass;
    if (!this._built) this._build();
    // The summary entity carries the forecast arrays. Find it once; the user
    // can pin a specific one if they somehow have two add-ons.
    this._entity = this._config.entity ||
      Object.keys(hass.states).find((e) => e.endsWith("_rain_summary") &&
                                           e.startsWith("sensor.rain_nowcast"));
    this._render();
    if (first) this._loadState();
  }

  // --- plumbing ------------------------------------------------------------

  async _entryId() {
    // The proxy is per config entry, and the card should not need to be told
    // which. Derive it from the summary entity's device.
    if (this._entry) return this._entry;
    const res = await this._hass.callWS({ type: "config_entries/get",
                                          domain: "rain_nowcast" });
    this._entry = res && res.length ? res[0].entry_id : null;
    return this._entry;
  }

  async _fetch(path) {
    const id = await this._entryId();
    if (!id) throw new Error("Rain Nowcast integration is not set up");
    // <img src> cannot carry an Authorization header, so everything is fetched
    // as bytes and handed to the canvas as a blob instead.
    const r = await fetch(`/api/rain_nowcast/${id}/${path}`, {
      headers: { Authorization: `Bearer ${this._hass.auth.data.access_token}` },
    });
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r;
  }

  async _loadState() {
    try {
      const state = await (await this._fetch("api/state")).json();
      if (state.status !== "ok") { this._note("Waiting for the first radar frame…"); return; }
      const changed = !this._state || this._state.updated !== state.updated;
      this._state = state;
      if (changed) { this._raw.clear(); this._tinted.clear(); this._levels.clear(); }
      this._palette = this._config.palette && state.palettes[this._config.palette]
        ? this._config.palette : state.default_palette;
      this._lut = Uint8ClampedArray.from(state.palettes[this._palette]);
      this._alpha = Uint8ClampedArray.from(state.alpha_lut);
      try { await loadLeaflet(); } catch (e) { this._note(e.message); }
      this._setupMap();
      state.frames.forEach((f) => this._load(f.name));
      const sc = this.shadowRoot.getElementById("scrub");
      sc.min = state.frames[0].offset;
      sc.max = state.frames[state.frames.length - 1].offset;
      sc.step = 1;
      // Not 0: that is the last radar frame, several minutes ago. The present
      // minute is a short extrapolation of it, which is exactly what the
      // forecast frames are for.
      if (changed) this._minute = this._lag();
      this._showMinute(this._minute);
      this._play(this._playing);
      this._note("");
    } catch (err) {
      this._note(`Cannot reach the Rain Nowcast add-on (${err.message})`);
    }
    clearTimeout(this._poll);
    this._poll = setTimeout(() => this._loadState(), 60000);
  }

  _load(name) {
    if (this._raw.has(name)) return Promise.resolve();
    return this._fetch(`frames/${name}`).then((r) => r.blob()).then((b) => {
      const url = URL.createObjectURL(b);
      return new Promise((done) => {
        const img = new Image();
        img.onload = () => { this._raw.set(name, img); URL.revokeObjectURL(url); done(); };
        img.onerror = () => { URL.revokeObjectURL(url); done(); };
        img.src = url;
      });
    }).catch(() => {});
  }

  _tint(name) {
    if (this._tinted.has(name)) return this._tinted.get(name);
    const img = this._raw.get(name);
    if (!img || !this._lut) return null;
    const c = this._canvas;
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.clearRect(0, 0, c.width, c.height);
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height), px = d.data;
    // greyscale decodes to R=G=B=level, so one channel read gives the level
    for (let i = 0; i < px.length; i += 4) {
      const l = px[i], j = l * 3;
      px[i] = this._lut[j]; px[i + 1] = this._lut[j + 1]; px[i + 2] = this._lut[j + 2];
      px[i + 3] = this._alpha[l];
    }
    g.putImageData(d, 0, 0);
    const url = c.toDataURL("image/png");
    this._tinted.set(name, url);
    return url;
  }

  _store(k, v) { try { localStorage.setItem("raincard." + k, JSON.stringify(v)); } catch (e) {} }
  _recall(k, d) {
    try { const v = localStorage.getItem("raincard." + k); return v === null ? d : JSON.parse(v); }
    catch (e) { return d; }
  }

  _usePalette(name) {
    if (!this._state || !this._state.palettes[name]) return;
    this._palette = name;
    this._lut = Uint8ClampedArray.from(this._state.palettes[name]);
    this._alpha = Uint8ClampedArray.from(this._state.alpha_lut);
    this._tinted.clear();                 // every cached frame is now stale
    this._store("palette", name);
    const sel = this.shadowRoot.getElementById("palette");
    if (sel) sel.value = name;
    this._showMinute(this._minute);
  }

  // --- forecast somewhere else ---------------------------------------------

  async _setPoint(latlng) {
    this._custom = latlng;
    this._store("point", latlng && { home: this._state.location, at: latlng });
    if (!latlng) {
      this._point = null;
      if (this._marker && this._state) this._marker.setLatLng(this._state.location);
    } else {
      if (this._marker) this._marker.setLatLng(latlng);
      try {
        const r = await this._fetch(`api/point?lat=${latlng[0]}&lon=${latlng[1]}`);
        this._point = await r.json();
      } catch (e) { /* keep the previous curve rather than blanking the chart */ }
    }
    this._renderPoint();
  }

  _renderPoint() {
    const label = this.shadowRoot.getElementById("loclabel");
    const reset = this.shadowRoot.getElementById("resetloc");
    if (label) label.textContent = this._custom
      ? `${this._custom[0].toFixed(3)}, ${this._custom[1].toFixed(3)}`
      : "Tap the map to forecast elsewhere";
    if (reset) reset.hidden = !this._custom;
    const sum = this.shadowRoot.getElementById("summary");
    if (sum && this._point) sum.textContent = this._point.summary;
    this._drawChart();
  }

  // --- map -----------------------------------------------------------------

  _setupMap() {
    if (this._map || !window.L) return;
    const el = this.shadowRoot.getElementById("map");
    this._map = L.map(el, { zoomControl: true, attributionControl: true });
    // Plain OSM tiles, darkened in CSS below. The ready-made dark basemaps
    // (CARTO, Stadia) all want an API key now - CARTO still answers 200, but
    // the tile it returns is a "API KEY REQUIRED" watermark.
    // maxZoom 12 because the radar is 1 km data: past that you magnify pixels
    // rather than reveal anything.
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 12, minZoom: 5,
      attribution: '&copy; OpenStreetMap contributors | radar: EUMETNET OPERA',
    }).addTo(this._map);
    const b = L.latLngBounds(this._state.bounds);
    this._overlay = L.imageOverlay("", b, { opacity: 0.78, interactive: false }).addTo(this._map);
    this._marker = L.circleMarker(this._state.location, {
      radius: 6, color: "#fff", weight: 2.5, fillColor: "#111", fillOpacity: 0.9,
    }).addTo(this._map);
    this._map.on("click", (e) => this._setPoint([+e.latlng.lat.toFixed(4), +e.latlng.lng.toFixed(4)]));

    const sel = this.shadowRoot.getElementById("palette");
    sel.innerHTML = "";
    Object.keys(this._state.palettes).forEach((n) => {
      const o = document.createElement("option");
      o.value = n; o.textContent = n.replace("-", " ");
      sel.appendChild(o);
    });
    this._usePalette(this._recall("palette", this._palette));
    const op = this._recall("opacity", 78);
    this.shadowRoot.getElementById("opacity").value = op;
    this._overlay.setOpacity(op / 100);
    // Open on the location configured in the add-on, not on the whole radar
    // domain - the forecast is about one spot, so that is what the map shows.
    // A point clicked earlier is remembered, but only while the configured
    // location it was chosen against is still the current one.
    const home = this._state.location;
    const saved = this._recall("point", null);
    const fresh = saved && saved.home && saved.home[0] === home[0]
                                      && saved.home[1] === home[1];
    this._map.setView(fresh ? saved.at : home, HOME_ZOOM);
    if (fresh) this._setPoint(saved.at);
    else this._store("point", null);
    // Leaflet measures the container on creation; inside a dashboard the card
    // is often still being laid out at that point.
    setTimeout(() => this._map.invalidateSize(), 200);
  }

  _levelPlane(name) {
    // Advecting needs random access to the pixels, which an <img> cannot give,
    // so the level byte is read out of each frame once and kept.
    if (this._levels.has(name)) return this._levels.get(name);
    const img = this._raw.get(name);
    if (!img) return null;
    const c = this._canvas;
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.clearRect(0, 0, c.width, c.height);
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const lv = new Uint8Array(c.width * c.height);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) lv[j] = d[i];
    const plane = { w: c.width, h: c.height, data: lv };
    this._levels.set(name, plane);
    return plane;
  }

  _warp(k, d) {
    // The overlay a fraction `d` of the way from frame k to frame k+1: trace
    // each pixel back along the flow to where that air was at k, and forward to
    // where it lands at k+1, then blend. Same scheme as the add-on's
    // interpolate.point_series, applied to the whole raster.
    //
    // Blended in LEVEL rather than in mm/h, because the level is all the
    // browser has - a geometric rather than arithmetic mean of the two rates.
    // The two samples are the same echo seen from either side, so they only
    // differ where rain is growing or decaying, and one level step is a few
    // percent of a rate.
    const F = this._state.frames, mo = this._state.motion;
    const A = this._levelPlane(F[k].name), B = this._levelPlane(F[k + 1].name);
    if (!mo || !A || !B || A.w !== B.w || A.h !== B.h) return null;
    if (!(mo.width > 1 && mo.height > 1)) return null;
    const w = A.w, h = A.h, a = A.data, b = B.data;
    if (!this._lut || !this._alpha) return null;

    const samp = (src, x, y) => {
      if (!(x > 0)) x = 0; else if (x > w - 1) x = w - 1;
      if (!(y > 0)) y = 0; else if (y > h - 1) y = h - 1;
      const x0 = x | 0, y0 = y | 0, tx = x - x0, ty = y - y0;
      const x1 = x0 < w - 1 ? x0 + 1 : x0, y1 = y0 < h - 1 ? y0 + 1 : y0;
      const r0 = y0 * w, r1 = y1 * w;
      return src[r0 + x0] * (1 - tx) * (1 - ty) + src[r0 + x1] * tx * (1 - ty)
           + src[r1 + x0] * (1 - tx) * ty + src[r1 + x1] * tx * ty;
    };

    const out = new Uint8Array(w * h);
    const MW = mo.width, MH = mo.height, mdx = mo.dx, mdy = mo.dy;
    for (let y = 0; y < h; y++) {
      const fy = y / (h - 1) * (MH - 1);
      let y0 = fy | 0; if (y0 > MH - 2) y0 = MH - 2;
      const ty = fy - y0, r0 = y0 * MW, r1 = r0 + MW;
      for (let x = 0; x < w; x++) {
        const fx = x / (w - 1) * (MW - 1);
        let x0 = fx | 0; if (x0 > MW - 2) x0 = MW - 2;
        const tx = fx - x0;
        const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty);
        const w01 = (1 - tx) * ty, w11 = tx * ty;
        const vx = mdx[r0 + x0] * w00 + mdx[r0 + x0 + 1] * w10
                 + mdx[r1 + x0] * w01 + mdx[r1 + x0 + 1] * w11;
        const vy = mdy[r0 + x0] * w00 + mdy[r0 + x0 + 1] * w10
                 + mdy[r1 + x0] * w01 + mdy[r1 + x0 + 1] * w11;
        const back = samp(a, x - vx * d, y - vy * d);
        const fwd = samp(b, x + vx * (1 - d), y + vy * (1 - d));
        out[y * w + x] = ((1 - d) * back + d * fwd + 0.5) | 0;
      }
    }
    return { w, h, data: out };
  }

  _paint(plane) {
    // Level plane -> a tinted PNG, the same colouring _tint does for a whole
    // frame. Kept apart from _warp so the advection can be checked as
    // arithmetic, without a palette and a PNG round-trip in the way.
    if (!plane) return null;
    const { w, h, data } = plane, lut = this._lut, alpha = this._alpha;
    const c = this._canvas;
    c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    const out = g.createImageData(w, h), px = out.data;
    for (let i = 0, o = 0; i < data.length; i++, o += 4) {
      const l = data[i], j = l * 3;
      px[o] = lut[j]; px[o + 1] = lut[j + 1]; px[o + 2] = lut[j + 2];
      px[o + 3] = alpha[l];
    }
    g.putImageData(out, 0, 0);
    return c.toDataURL("image/png");
  }

  _between(k, d) { return this._paint(this._warp(k, d)); }

  // How old the newest radar frame already is, in minutes.
  //
  // Minute 0 of the data is that FRAME, not the present moment: OPERA
  // composites arrive every five minutes and then take a few more to publish,
  // so by the time anyone looks at this the newest picture is typically 5-10
  // minutes old. Reading minute 0 as "now" is what put the clock on this card
  // about five minutes behind the one on the wall.
  //
  // Live from the browser's own clock rather than a number from the server,
  // because it keeps ageing between polls.
  _lag() {
    const u = this._state && Date.parse(this._state.updated);
    if (!u) return 0;
    // Capped both ways: a browser clock that is wrong, or radar that has
    // stopped arriving, must not drag the whole timeline somewhere absurd.
    return Math.max(0, Math.min(60, Math.floor((Date.now() - u) / 60000)));
  }

  _showMinute(m) {
    if (!this._state) return;
    const F = this._state.frames;
    m = Math.round(Math.max(F[0].offset, Math.min(F[F.length - 1].offset, m)));
    this._minute = m;

    let k = 0;
    while (k + 1 < F.length && F[k + 1].offset <= m) k++;
    const span = k + 1 < F.length ? F[k + 1].offset - F[k].offset : 0;
    const d = span > 0 ? (m - F[k].offset) / span : 0;

    let url = d < 1e-6 ? this._tint(F[k].name) : this._between(k, d);
    // No motion estimate, or a neighbour not fetched yet: the nearer real frame
    // is still the truth, it just repeats for a few minutes.
    if (!url) url = this._tint(F[d < 0.5 ? k : Math.min(k + 1, F.length - 1)].name);
    if (url && this._overlay) this._overlay.setUrl(url);
    else if (!this._raw.has(F[k].name)) this._load(F[k].name).then(() => {
      if (this._minute === m) this._showMinute(m);
    });

    const at = new Date(new Date(F[0].time).getTime() + (m - F[0].offset) * 60000);
    const when = at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    this.shadowRoot.getElementById("stamp").textContent = `${when} · ${rel(m - this._lag())}`;
    this.shadowRoot.getElementById("scrub").value = m;
    this._drawChart();
  }

  _play(on) {
    this._playing = on;
    const btn = this.shadowRoot.getElementById("play");
    if (btn) btn.textContent = on ? "❚❚" : "▶";
    clearInterval(this._timer);
    if (on && this._state) {
      this._timer = setInterval(() => {
        const F = this._state.frames;
        const last = F[F.length - 1].offset;
        this._showMinute(this._minute >= last ? F[0].offset : this._minute + 1);
      }, STEP_MS);
    }
  }

  // --- chart ---------------------------------------------------------------

  _attrs() {
    // A clicked point overrides the entity, which is always the configured
    // location. Same shape either way: the add-on's /api/point returns the
    // series under the same keys publish.payload puts in the attributes.
    if (this._point && this._point.series) {
      const s = this._point.series;
      return { forecast_minutes: s.minutes, forecast_values: s.values,
               forecast_probability: s.probability,
               forecast_low: s.p10, forecast_high: s.p90,
               forecast_min: s.pmin, forecast_max: s.pmax };
    }
    const st = this._hass && this._entity && this._hass.states[this._entity];
    return st ? st.attributes : null;
  }

  _drawChart() {
    const cv = this.shadowRoot.getElementById("chart");
    const a = this._attrs();
    if (!cv || !a || !a.forecast_minutes) return;
    const minutes = a.forecast_minutes, values = a.forecast_values;
    const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    // Numbers on the axis when there is room for them; "heavy" alone tells you
    // nothing you can compare a forecast against.
    const wide = w > 340;
    const padL = wide ? 52 : 30, padR = 8, padT = 14, padB = 18;
    // Kept so a pointer position can be turned back into a minute; the chart is
    // the only thing that knows where the axis actually is.
    this._geom = { padL, padR, m0: minutes[0], m1: minutes.at(-1) };
    const LO = 0.05, HI = 50;
    const norm = (v) => (v < LO ? 0 : Math.log10(v / LO) / Math.log10(HI / LO));
    const X = (m) => padL + (m - minutes[0]) / (minutes.at(-1) - minutes[0]) * (w - padL - padR);
    const Y = (v) => h - padB - norm(v) * (h - padT - padB);
    const css = (n) => getComputedStyle(this).getPropertyValue(n) || "#2f7fe8";
    const accent = css("--primary-color").trim() || "#2f7fe8";
    const ink = css("--primary-text-color").trim() || "#222";

    g.strokeStyle = css("--divider-color").trim() || "#ddd";
    g.font = "10px system-ui, sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
    const muted = css("--secondary-text-color").trim() || "#888";
    for (const [v, label] of [[0.1, "light"], [1, "mod"], [5, "heavy"], [20, "violent"]]) {
      g.beginPath(); g.moveTo(padL, Y(v) + 0.5); g.lineTo(w - padR, Y(v) + 0.5); g.stroke();
      g.fillStyle = muted;
      g.fillText(wide ? `${v} ${label}` : `${v}`, padL - 4, Y(v));
    }
    g.textBaseline = "top";
    g.fillText("mm/h", padL - 4, 1);
    g.textBaseline = "middle";

    // The spread and the probability only exist ahead of "now": behind it every
    // ensemble member carries the same observed radar, so the fan collapses to
    // the line and the probability is a bare 0/1 step - which drew as a solid
    // block over the whole past hour and read as a broken chart.
    const n0 = minutes.findIndex((m) => m >= 0);
    const fut = n0 < 0 ? [] : minutes.slice(n0);

    // uncertainty fan first, so the mean line always reads on top
    const fan = (lo, hi, alpha) => {
      if (!lo || !hi || fut.length < 2) return;
      g.beginPath();
      fut.forEach((m, i) => (i ? g.lineTo(X(m), Y(hi[n0 + i])) : g.moveTo(X(m), Y(hi[n0 + i]))));
      for (let i = fut.length - 1; i >= 0; i--) g.lineTo(X(fut[i]), Y(lo[n0 + i]));
      g.closePath(); g.fillStyle = accent; g.globalAlpha = alpha; g.fill(); g.globalAlpha = 1;
    };
    fan(a.forecast_low, a.forecast_high, 0.16);

    // The outright envelope: no member of the ensemble went above the upper
    // line or below the lower one. Lines rather than another shaded band -
    // these are the noisiest thing on the chart, one member each out of twelve,
    // and should not look as solid as the percentiles they bracket.
    const edge = (arr, label) => {
      if (!arr || fut.length < 2) return;
      g.beginPath();
      fut.forEach((m, i) => (i ? g.lineTo(X(m), Y(arr[n0 + i])) : g.moveTo(X(m), Y(arr[n0 + i]))));
      g.strokeStyle = accent; g.globalAlpha = 0.75; g.lineWidth = 1.1;
      g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
      // Above its own line when the line runs along the axis, or the label
      // lands on top of the "+60" tick.
      const ly = Y(arr[minutes.length - 1]);
      const above = label === "max" || ly > h - padB - 14;
      g.fillStyle = ink; g.textAlign = "right";
      g.textBaseline = above ? "bottom" : "top";
      g.fillText(label, w - padR - 2, ly + (above ? -3 : 3));
      g.globalAlpha = 1;
    };
    edge(a.forecast_max, "max"); edge(a.forecast_min, "min");

    g.beginPath();
    minutes.forEach((m, i) => (i ? g.lineTo(X(m), Y(values[i])) : g.moveTo(X(m), Y(values[i]))));
    g.strokeStyle = accent; g.lineWidth = 1.8; g.stroke();

    // ensemble agreement along the bottom
    if (a.forecast_probability && fut.length > 1) {
      const band = (h - padT - padB) * 0.18, base = h - padB;
      g.beginPath(); g.moveTo(X(fut[0]), base);
      fut.forEach((m, i) => g.lineTo(X(m), base - a.forecast_probability[n0 + i] * band));
      g.lineTo(X(fut.at(-1)), base); g.closePath();
      g.fillStyle = accent; g.globalAlpha = 0.3; g.fill(); g.globalAlpha = 1;
    }

    const lag = this._lag();
    g.strokeStyle = ink; g.lineWidth = 1.2; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(lag), padT); g.lineTo(X(lag), h - padB); g.stroke();
    g.setLineDash([]);
    if (this._state) {
      g.strokeStyle = accent; g.lineWidth = 2;
      g.beginPath();
      g.moveTo(X(this._minute), padT); g.lineTo(X(this._minute), h - padB);
      g.stroke();
    }
    g.fillStyle = css("--secondary-text-color").trim() || "#888";
    g.textBaseline = "top";
    // Ticks on round minutes FROM NOW, not from the radar frame, so the axis
    // agrees with the labels everywhere else on the card.
    const first = Math.ceil((minutes[0] - lag) / 30) * 30;
    for (let d = first; d <= minutes.at(-1) - lag; d += 30) {
      const at = d + lag;
      g.textAlign = at <= minutes[0] + 2 ? "left"
                  : (at >= minutes.at(-1) - 2 ? "right" : "center");
      g.fillText(d === 0 ? "now" : `${d > 0 ? "+" : ""}${d}`, X(at), h - padB + 4);
    }

    // Name the worst minute outright: reading a number off a log axis by eye is
    // hopeless, and "how hard could it get" is the question being asked.
    const hi = a.forecast_max, lo = a.forecast_min;
    if (hi && lo && fut.length > 1) {
      let k = n0;
      for (let i = n0; i < minutes.length; i++) if (hi[i] > hi[k]) k = i;
      if (hi[k] >= 0.05) {
        const fmt = (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1));
        const px = Math.min(w - padR - 20, Math.max(padL, X(minutes[k])));
        g.textAlign = px < padL + 40 ? "left" : "center";
        g.fillStyle = accent; g.textBaseline = "bottom";
        g.fillText(`up to ${fmt(hi[k])}`, px, Math.max(Y(hi[k]) - 3, padT + 9));
        g.fillStyle = muted; g.textBaseline = "top";
        g.fillText(lo[k] < 0.05 ? "or nothing at all" : `at least ${fmt(lo[k])}`,
                   px, Math.min(Y(lo[k]) + 3, h - padB - 11));
      }
    }
  }

  // --- reading the graph ----------------------------------------------------

  _minuteAt(e) {
    const cv = this.shadowRoot.getElementById("chart"), gm = this._geom;
    if (!cv || !gm) return null;
    const r = cv.getBoundingClientRect();
    const span = r.width - gm.padL - gm.padR;
    if (span <= 0) return null;
    const f = Math.max(0, Math.min(1, (e.clientX - r.left - gm.padL) / span));
    return Math.round(gm.m0 + f * (gm.m1 - gm.m0));
  }

  _tip(e) {
    const tip = this.shadowRoot.getElementById("tip");
    if (!tip) return;
    const a = e && this._attrs();
    const m = e && this._minuteAt(e);
    if (!a || m === null || !a.forecast_minutes) { tip.style.opacity = 0; return; }
    const mins = a.forecast_minutes;
    const stepM = mins.length > 1 ? mins[1] - mins[0] : 1;
    const i = Math.round((m - mins[0]) / stepM);
    if (!(i >= 0 && i < mins.length)) { tip.style.opacity = 0; return; }
    const v = a.forecast_values[i];
    const lo = a.forecast_low && a.forecast_low[i];
    const hi = a.forecast_high && a.forecast_high[i];
    // The spread is only meaningful ahead of now - behind it every member
    // carries the same observed radar - so it is shown only there.
    const band = (m > 0 && lo != null && hi != null && hi - lo > 0.05)
      ? ` (${lo.toFixed(1)}\u2013${hi.toFixed(1)})` : "";
    const p = a.forecast_probability;
    tip.textContent = `${v.toFixed(2)}${band} mm/h \u00b7 ${rel(m - this._lag())}`
      + (p && m > 0 ? ` \u00b7 ${Math.round(p[i] * 100)}% of runs wet` : "");
    const box = this.shadowRoot.querySelector(".chartbox").getBoundingClientRect();
    tip.style.left = `${e.clientX - box.left}px`;
    tip.style.opacity = 1;
  }

  // --- DOM -----------------------------------------------------------------

  _note(text) {
    const n = this.shadowRoot.getElementById("note");
    if (n) { n.textContent = text; n.style.display = text ? "block" : "none"; }
  }

  _build() {
    this._built = true;
    const css = document.createElement("link");
    css.rel = "stylesheet"; css.href = "/rain_nowcast_static/leaflet.css";
    const root = document.createElement("ha-card");
    root.innerHTML = `
      <style>
        .head { display:flex; align-items:baseline; gap:10px; padding:12px 16px 6px; }
        .head .sum { font-weight:600; color:var(--primary-color); }
        #map { height:${this._config.height}px; background:#12161c; }
        /* Dark basemap from light tiles: invert, then hue-rotate back so the
           few surviving hues are the right way round. Tile pane only, so the
           radar overlay and the marker keep their real colours. */
        #map .leaflet-tile-pane { filter:grayscale(.9) invert(.92) hue-rotate(180deg) brightness(.95) contrast(.92); }
        .row { display:flex; align-items:center; gap:10px; padding:8px 16px; }
        #play { flex:none; width:30px; height:30px; border-radius:50%; border:0; cursor:pointer;
                background:var(--primary-color); color:#fff; }
        #scrub { flex:1; accent-color:var(--primary-color); min-width:0; }
        #stamp { flex:none; font-size:.85rem; font-variant-numeric:tabular-nums; }
        #chart { display:block; width:100%; height:150px; }
        .chartbox { position:relative; }
        #tip {
          position:absolute; top:8px; transform:translateX(-50%); opacity:0;
          pointer-events:none; transition:opacity .12s; white-space:nowrap;
          background:var(--card-background-color); color:var(--primary-text-color);
          border:1px solid var(--divider-color); border-radius:6px;
          padding:2px 7px; font-size:.78rem; font-variant-numeric:tabular-nums;
        }
        #note { display:none; padding:8px 16px; color:var(--error-color,#b00); font-size:.85rem; }
        .chartbox { padding:0 8px 10px; }
        .opts { flex-wrap:wrap; font-size:.8rem; color:var(--secondary-text-color); padding-top:0; }
        .opts label { display:flex; align-items:center; gap:6px; }
        .opts select, .opts button {
          font:inherit; color:var(--primary-text-color); background:var(--card-background-color);
          border:1px solid var(--divider-color); border-radius:6px; padding:2px 6px; cursor:pointer;
        }
        .opts input[type=range] { width:80px; accent-color:var(--primary-color); }
        .opts .spacer { margin-left:auto; }
      </style>
      <div class="head"><span class="sum" id="summary"></span></div>
      <div id="note"></div>
      <div id="map"></div>
      <div class="row">
        <button id="play" aria-label="Play or pause">❚❚</button>
        <input id="scrub" type="range" min="-60" max="60" step="1" value="0"
               aria-label="Radar time, in minutes from now">
        <span id="stamp">--:--</span>
      </div>
      <div class="row opts">
        <label>Colours <select id="palette" aria-label="Radar colour scheme"></select></label>
        <label>Opacity <input id="opacity" type="range" min="20" max="100" value="78"
               aria-label="Radar overlay opacity"></label>
        <span class="spacer" id="loclabel">Tap the map to forecast elsewhere</span>
        <button id="resetloc" hidden>My location</button>
      </div>
      <div class="chartbox" ${this._config.show_chart ? "" : "hidden"}>
        <div id="tip"></div>
        <canvas id="chart"></canvas>
      </div>`;
    this.shadowRoot.append(css, root);

    root.querySelector("#play").onclick = () => this._play(!this._playing);
    root.querySelector("#scrub").oninput = (e) => { this._play(false); this._showMinute(+e.target.value); };
    root.querySelector("#palette").onchange = (e) => this._usePalette(e.target.value);
    root.querySelector("#opacity").oninput = (e) => {
      if (this._overlay) this._overlay.setOpacity(e.target.value / 100);
      this._store("opacity", +e.target.value);
    };
    root.querySelector("#resetloc").onclick = () => this._setPoint(null);

    // Reading the graph: hover for the exact numbers, tap or drag to take the
    // radar to that minute. Hover deliberately does NOT move the map - having
    // it jump every time the pointer crosses the card is worse than useless on
    // a dashboard you are only passing over.
    const box = root.querySelector(".chartbox");
    const seek = (e) => {
      const m = this._minuteAt(e);
      if (m !== null) { this._play(false); this._showMinute(m); }
    };
    box.addEventListener("pointerdown", (e) => {
      seek(e);
      // Drag-to-scrub off the edge of the card. Not every pointer id can be
      // captured, and failing to is not a reason to lose the tap.
      try { box.setPointerCapture(e.pointerId); } catch (err) { /* fine */ }
    });
    box.addEventListener("pointermove", (e) => { this._tip(e); if (e.buttons & 1) seek(e); });
    box.addEventListener("pointerleave", () => this._tip(null));
    box.addEventListener("pointercancel", () => this._tip(null));
    new ResizeObserver(() => { this._drawChart(); if (this._map) this._map.invalidateSize(); })
      .observe(root);
  }

  _render() {
    if (!this._built) return;
    const s = this._hass.states[this._entity];
    const el = this.shadowRoot.getElementById("summary");
    if (el) el.textContent = s ? s.state : "Rain Nowcast not set up";
    this._drawChart();
  }

  disconnectedCallback() { clearInterval(this._timer); clearTimeout(this._poll); }
}

customElements.define(CARD, RainNowcastCard);
window.customCards = window.customCards || [];
window.customCards.push({
  type: CARD,
  name: "Rain Nowcast",
  description: "Animated rain radar and a minute-by-minute forecast.",
  preview: false,
});
console.info("%c RAIN-NOWCAST-CARD ", "background:#2f7fe8;color:#fff", "loaded");
