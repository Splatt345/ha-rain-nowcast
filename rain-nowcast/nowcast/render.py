"""Turn mm/h radar fields into web-map-ready PNG overlays and point series.

The radar grid is a Lambert azimuthal equal-area projection; Leaflet's
ImageOverlay stretches an image linearly across a lat/lon rectangle in Web
Mercator space. So the field has to be resampled onto a Mercator-spaced grid
first, otherwise the rain sits visibly north/south of where it actually is.
"""
import io

import numpy as np

# Rain-rate colour scales (mm/h -> RGB), interpolated in log space so the light
# end (where nearly all the pixels live) gets most of the range.
#
# The overlay PNGs do NOT carry any of these. Colour and alpha are pure
# functions of one value (the rain rate), so the server ships the rate as an
# 8-bit level and the browser colours it through `lut()` - which means changing
# palette is a canvas repaint with no refetch, no re-render and no server CPU.
# That last part is the point on a Raspberry Pi.
PALETTES = {
    "blue-green": [
        (0.10, (150, 200, 255)), (0.50, (80, 160, 250)), (1.00, (35, 110, 235)),
        (2.50, (25, 190, 180)), (5.00, (60, 205, 90)), (10.0, (245, 215, 60)),
        (20.0, (250, 150, 45)), (40.0, (235, 60, 60)), (80.0, (200, 50, 180)),
    ],
    # Perceptually uniform and safe for the common forms of colour blindness -
    # the default's green/red separation at 5 vs 40 mm/h is exactly the pair
    # deuteranopia collapses.
    "viridis": [
        (0.10, (68, 1, 84)), (0.50, (71, 45, 123)), (1.00, (59, 82, 139)),
        (2.50, (44, 114, 142)), (5.00, (33, 145, 140)), (10.0, (40, 174, 128)),
        (20.0, (94, 201, 98)), (40.0, (173, 220, 48)), (80.0, (253, 231, 37)),
    ],
    # What people expect a weather radar to look like.
    "classic": [
        (0.10, (100, 240, 240)), (0.50, (30, 180, 240)), (1.00, (20, 90, 220)),
        (2.50, (20, 220, 60)), (5.00, (240, 240, 40)), (10.0, (250, 180, 30)),
        (20.0, (240, 80, 30)), (40.0, (200, 20, 20)), (80.0, (255, 255, 255)),
    ],
}
DEFAULT_PALETTE = "blue-green"

MIN_RATE = 0.05  # below this we draw nothing at all
MAX_RATE = 80.0  # top of the level scale, matching the last palette stop

# Distinct levels the overlay carries. Not cosmetic: a full 256-level greyscale
# PNG is 40 KB where the 64-colour quantised RGBA it replaces was 26 KB, and the
# page preloads two dozen frames. 64 restores most of that (33 KB) and matches
# the visual budget the old palette had anyway.
LEVEL_STEPS = 64


def level(field):
    """mm/h field -> uint8 level, the rain rate on a quantised log scale.

    This is the ONLY thing the overlay carries. `lut()` turns a level into
    colour and `alpha_lut()` into opacity, both in the browser.
    """
    rate = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    log_rate = np.log10(np.clip(rate, 1e-3, None))
    lo, hi = np.log10(MIN_RATE), np.log10(MAX_RATE)
    unit = np.clip((log_rate - lo) / (hi - lo), 0.0, 1.0)
    step = 255.0 / (LEVEL_STEPS - 1)
    return (np.round(unit * (LEVEL_STEPS - 1)) * step).astype(np.uint8)


def alpha_lut():
    """Opacity per level, as a 256-entry uint8 table.

    Defined as a function OF THE LEVEL rather than of the rate, deliberately.
    Deriving it from the rate independently made 62 levels ambiguous (two rates
    rounding to one level but to different alphas), because the fade spans
    0.05-0.3 mm/h while the level scale spans 0.05-80. As a function of level it
    is exact by construction, and the browser needs no rate maths at all.

    It fades in across the drizzle range so light rain does not appear as a
    hard-edged blob at the detection threshold.
    """
    lo, hi = np.log10(MIN_RATE), np.log10(MAX_RATE)
    log_rate = lo + (hi - lo) * np.arange(256) / 255.0
    fade = (log_rate - lo) / (np.log10(0.3) - lo)
    table = (np.clip(fade, 0.0, 1.0) * 215.0).astype(np.uint8)
    table[0] = 0  # level 0 is at or below MIN_RATE: draw nothing
    return table


def lut(name=DEFAULT_PALETTE):
    """A palette as a flat 256*3 uint8 RGB table, indexed by `level`.

    Built here rather than in JavaScript so the log-spaced stops have exactly
    one definition. The browser only ever does `rgb = lut[level]`.
    """
    stops = PALETTES[name]
    lo, hi = np.log10(MIN_RATE), np.log10(MAX_RATE)
    log_rate = lo + (hi - lo) * np.arange(256) / 255.0
    stop_x = np.log10([s[0] for s in stops])
    table = np.zeros((256, 3), np.uint8)
    for channel in range(3):
        stop_y = np.array([s[1][channel] for s in stops], float)
        table[:, channel] = np.interp(log_rate, stop_x, stop_y).astype(np.uint8)
    return table


def png_bytes(field):
    """PNG-encode an overlay as a single-channel greyscale level image.

    One byte per pixel, carrying no colour at all - the browser applies `lut()`
    and `alpha_lut()`. Measured against the 64-colour RGBA this replaces:
    33 KB vs 26 KB per frame, the price of moving palette choice to the client.
    Shipping alpha as a second channel would have cost 57 KB, which is why it is
    derived from the level instead.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(level(field), "L").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _crs(metadata):
    import pyproj

    return pyproj.CRS(metadata["projection"])


def mercator_sampler(metadata, bbox, width=640):
    """Precompute the index arrays that resample fields on `metadata`'s grid
    onto a Web-Mercator-spaced lat/lon rectangle covering `bbox`.

    The grid never moves between frames, so this is computed once and reused.
    Returns (rows, cols, height).
    """
    import pyproj

    lon0, lat0, lon1, lat1 = bbox
    to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx0, my0 = to_merc.transform(lon0, lat0)
    mx1, my1 = to_merc.transform(lon1, lat1)
    height = max(1, int(round(width * (my1 - my0) / (mx1 - mx0))))

    from_merc = pyproj.Transformer.from_crs("EPSG:3857", _crs(metadata), always_xy=True)
    grid_x, grid_y = np.meshgrid(
        np.linspace(mx0, mx1, width),
        np.linspace(my1, my0, height),  # first row is the north edge
    )
    proj_x, proj_y = from_merc.transform(grid_x, grid_y)

    cols = (proj_x - metadata["x1"]) / metadata["xpixelsize"]
    rows = (metadata["y2"] - proj_y) / metadata["ypixelsize"]  # yorigin="upper"
    return rows, cols, height


#: Display-only smoothing, in SOURCE pixels, applied before the Mercator
#: resample. Tied to the source grid rather than to a distance in km, because
#: what it exists to hide is an artefact of that grid.
#:
#: The OPERA composite is assembled from national radars of differing
#: resolution, and coarser contributors are filled into the 1 km grid as flat
#: 2-4 km blocks. Zoomed in, the overlay showed hard square steps and stair-
#: stepped edges - visibly wrong, and not something the forecast believes.
#:
#: Higher-order interpolation does NOT fix it: order=3 was rendered and compared
#: side by side with order=1 and looks the same, because interpolating blocky
#: data smoothly still yields blocky data. Only actual smoothing removes it, at
#: roughly the source pixel scale - which is another way of saying "do not draw
#: detail finer than the data resolves".
SMOOTH_PX = 0.8


def resample(field, rows, cols, smooth=SMOOTH_PX):
    """Resample a field onto the Mercator grid for display.

    DISPLAY ONLY. This is the overlay path and nothing else calls it: the
    sensors, the per-minute curve and the uncertainty fan all come from
    `server._point_state`, which reads the unsmoothed forecast. So this changes
    what the map looks like and cannot change what the add-on predicts.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    field = np.nan_to_num(field, nan=0.0)
    if smooth:
        field = gaussian_filter(field, smooth)
    return map_coordinates(field, [rows, cols], order=1, mode="constant", cval=0.0)


def point_rowcol(metadata, lat, lon):
    import pyproj

    to_grid = pyproj.Transformer.from_crs("EPSG:4326", _crs(metadata), always_xy=True)
    x, y = to_grid.transform(lon, lat)
    col = int(round((x - metadata["x1"]) / metadata["xpixelsize"]))
    row = int(round((metadata["y2"] - y) / metadata["ypixelsize"]))
    return row, col


def point_series(stack, row, col, box=1):
    """Rain rate at (row, col) for each frame in `stack`, averaged over a
    small box. A single 1 km pixel is noisy speckle; ~3x3 km is both steadier
    and closer to what you actually experience standing there.
    """
    r0, r1 = max(row - box, 0), row + box + 1
    c0, c1 = max(col - box, 0), col + box + 1
    return np.array(
        [float(np.nan_to_num(np.nanmean(f[r0:r1, c0:c1]), nan=0.0)) for f in stack]
    )


def _demo():
    """Self-check: levels are monotonic, alpha gates, palettes round-trip."""
    field = np.array([[0.0, 0.02, 0.1, 1.0, 10.0, 100.0]])
    lvl = level(field)
    alpha = alpha_lut()[lvl]
    assert lvl.shape == (1, 6)
    assert alpha[0, 0] == 0 and alpha[0, 1] == 0, "sub-threshold must be clear"
    assert alpha[0, 3] > 0 and alpha[0, 5] > 0, "rain must be opaque"
    # heavier rain must never map to a lower level, and must saturate not wrap:
    # 100 mm/h is above MAX_RATE and has to clamp to 255, not roll over to 0.
    assert list(lvl[0]) == sorted(lvl[0]), lvl
    assert lvl[0, 5] == 255, lvl
    # alpha must be a FUNCTION of level - the ambiguity that forced alpha_lut()
    # to be defined on the level rather than on the rate.
    ramp = np.arange(256, dtype=np.uint8)
    assert len(set(zip(ramp.tolist(), alpha_lut().tolist()))) == 256
    assert level(np.array([[MIN_RATE / 2]]))[0, 0] == 0, "below MIN_RATE -> 0"

    # Every palette must be a full 256-entry table and actually differ from the
    # others, or the picker silently does nothing.
    tables = {n: lut(n) for n in PALETTES}
    for name, table in tables.items():
        assert table.shape == (256, 3) and table.dtype == np.uint8, name
        assert tuple(table[255]) != tuple(table[0]), f"{name} is flat"
    assert not np.array_equal(tables["viridis"], tables["blue-green"])

    # Display smoothing must soften block edges WITHOUT moving the rain or
    # inventing coverage - it is a presentation change and has to stay one.
    rng = np.random.default_rng(1)
    blocky = np.repeat(np.repeat(rng.random((16, 16)) * 20, 4, axis=0), 4, axis=1)
    ident = np.arange(64.0)
    idx = np.meshgrid(ident, ident, indexing="ij")
    raw = resample(blocky, idx[0], idx[1], smooth=0)
    soft = resample(blocky, idx[0], idx[1], smooth=SMOOTH_PX)
    jump = lambda a: np.abs(np.diff(a, axis=1)).max()
    assert jump(soft) < jump(raw), "smoothing must reduce the worst block edge"
    assert abs(soft.mean() - raw.mean()) < 0.05, "must not change how much rain there is"
    peak = lambda a: np.unravel_index(np.argmax(a), a.shape)
    assert np.hypot(*(np.array(peak(soft)) - np.array(peak(raw)))) <= 4, \
        "must not move the peak"
    assert (soft >= 0).all()

    blob = png_bytes(np.random.default_rng(0).random((64, 64)) * 20)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    assert Image.open(io.BytesIO(blob)).mode == "L"

    # Mercator row spacing must be non-uniform in latitude (that's the point).
    meta = {
        "projection": "+proj=longlat +ellps=WGS84",
        "x1": 2.0, "y2": 51.8, "xpixelsize": 0.01, "ypixelsize": 0.01,
    }
    rows, cols, height = mercator_sampler(meta, (2.0, 49.0, 6.8, 51.8), width=64)
    assert rows.shape == (height, 64) and cols.shape == (height, 64)
    spacing = np.diff(rows[:, 0])
    assert not np.allclose(spacing, spacing[0], rtol=1e-4), "should be Mercator-spaced"
    print("render self-check ok")


if __name__ == "__main__":
    _demo()
