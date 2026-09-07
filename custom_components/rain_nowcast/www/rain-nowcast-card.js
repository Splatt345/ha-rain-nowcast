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
 */

const CARD = "rain-nowcast-card";
const HOME_ZOOM = 9;          // roughly 100 km across: the useful nowcast range

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
    this._frame = 0;
    this._playing = true;
    this._raw = new Map();      // name -> decoded level PNG
    this._tinted = new Map();   // name -> data URL under the current palette
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
      if (changed) { this._raw.clear(); this._tinted.clear(); }
      this._palette = this._config.palette && state.palettes[this._config.palette]
        ? this._config.palette : state.default_palette;
      this._lut = Uint8ClampedArray.from(state.palettes[this._palette]);
      this._alpha = Uint8ClampedArray.from(state.alpha_lut);
      try { await loadLeaflet(); } catch (e) { this._note(e.message); }
      this._setupMap();
      state.frames.forEach((f) => this._load(f.name));
      if (changed || this._frame >= state.frames.length) this._frame = state.now_frame;
      this._show(this._frame);
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
    this._show(this._frame);
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

  _show(i) {
    if (!this._state) return;
    this._frame = i;
    const f = this._state.frames[i];
    const url = this._tint(f.name);
    if (url && this._overlay) this._overlay.setUrl(url);
    else this._load(f.name).then(() => {
      if (this._frame === i && this._overlay) {
        const u = this._tint(f.name); if (u) this._overlay.setUrl(u);
      }
    });
    const when = new Date(f.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const rel = f.offset === 0 ? "now" : (f.offset < 0 ? `${-f.offset} min ago` : `in ${f.offset} min`);
    this.shadowRoot.getElementById("stamp").textContent = `${when} · ${rel}`;
    this.shadowRoot.getElementById("scrub").value = i;
    this._drawChart();
  }

  _play(on) {
    this._playing = on;
    const btn = this.shadowRoot.getElementById("play");
    if (btn) btn.textContent = on ? "❚❚" : "▶";
    clearInterval(this._timer);
    if (on && this._state) {
      this._timer = setInterval(() => {
        const last = this._state.frames.length - 1;
        this._show(this._frame >= last ? 0 : this._frame + 1);
      }, 300);
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
               forecast_low: s.p10, forecast_high: s.p90 };
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

    const padL = 30, padR = 8, padT = 8, padB = 18;
    const LO = 0.05, HI = 50;
    const norm = (v) => (v < LO ? 0 : Math.log10(v / LO) / Math.log10(HI / LO));
    const X = (m) => padL + (m - minutes[0]) / (minutes.at(-1) - minutes[0]) * (w - padL - padR);
    const Y = (v) => h - padB - norm(v) * (h - padT - padB);
    const css = (n) => getComputedStyle(this).getPropertyValue(n) || "#2f7fe8";
    const accent = css("--primary-color").trim() || "#2f7fe8";
    const ink = css("--primary-text-color").trim() || "#222";

    g.strokeStyle = css("--divider-color").trim() || "#ddd";
    g.font = "10px system-ui, sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
    for (const [v, label] of [[0.1, "light"], [1, "mod"], [5, "heavy"]]) {
      g.beginPath(); g.moveTo(padL, Y(v) + 0.5); g.lineTo(w - padR, Y(v) + 0.5); g.stroke();
      g.fillStyle = css("--secondary-text-color").trim() || "#888";
      g.fillText(label, padL - 4, Y(v));
    }

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

    g.strokeStyle = ink; g.lineWidth = 1.2; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(0), padT); g.lineTo(X(0), h - padB); g.stroke();
    g.setLineDash([]);
    if (this._state) {
      const cur = this._state.frames[this._frame].offset;
      g.strokeStyle = accent; g.lineWidth = 2;
      g.beginPath(); g.moveTo(X(cur), padT); g.lineTo(X(cur), h - padB); g.stroke();
    }
    g.fillStyle = css("--secondary-text-color").trim() || "#888";
    g.textBaseline = "top";
    for (const m of minutes.filter((m) => m % 30 === 0)) {
      g.textAlign = m === minutes[0] ? "left" : (m === minutes.at(-1) ? "right" : "center");
      g.fillText(m === 0 ? "now" : `${m > 0 ? "+" : ""}${m}`, X(m), h - padB + 4);
    }
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
        <input id="scrub" type="range" min="0" max="0" value="0" aria-label="Radar time">
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
        <canvas id="chart"></canvas>
      </div>`;
    this.shadowRoot.append(css, root);

    root.querySelector("#play").onclick = () => this._play(!this._playing);
    root.querySelector("#scrub").oninput = (e) => { this._play(false); this._show(+e.target.value); };
    root.querySelector("#palette").onchange = (e) => this._usePalette(e.target.value);
    root.querySelector("#opacity").oninput = (e) => {
      if (this._overlay) this._overlay.setOpacity(e.target.value / 100);
      this._store("opacity", +e.target.value);
    };
    root.querySelector("#resetloc").onclick = () => this._setPoint(null);
    new ResizeObserver(() => { this._drawChart(); if (this._map) this._map.invalidateSize(); })
      .observe(root);
  }

  _render() {
    if (!this._built) return;
    const s = this._hass.states[this._entity];
    const el = this.shadowRoot.getElementById("summary");
    if (el) el.textContent = s ? s.state : "Rain Nowcast not set up";
    if (this._state) this.shadowRoot.getElementById("scrub").max = this._state.frames.length - 1;
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
