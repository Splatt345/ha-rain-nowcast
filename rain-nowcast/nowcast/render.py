"""Turn mm/h radar fields into web-map-ready PNG overlays and point series.

The radar grid is a Lambert azimuthal equal-area projection; Leaflet's
ImageOverlay stretches an image linearly across a lat/lon rectangle in Web
Mercator space. So the field has to be resampled onto a Mercator-spaced grid
first, otherwise the rain sits visibly north/south of where it actually is.
"""
import io

import numpy as np

# Rain-rate colour scale (mm/h -> RGB), interpolated in log space so the
# light end (where nearly all the pixels live) gets most of the range.
_STOPS = [
    (0.10, (150, 200, 255)),
    (0.50, (80, 160, 250)),
    (1.00, (35, 110, 235)),
    (2.50, (25, 190, 180)),
    (5.00, (60, 205, 90)),
    (10.0, (245, 215, 60)),
    (20.0, (250, 150, 45)),
    (40.0, (235, 60, 60)),
    (80.0, (200, 50, 180)),
]

MIN_RATE = 0.05  # below this we draw nothing at all


def colorize(field):
    """mm/h field -> (h, w, 4) uint8 RGBA, transparent below MIN_RATE."""
    rate = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    log_rate = np.log10(np.clip(rate, 1e-3, None))
    stop_x = np.log10([s[0] for s in _STOPS])

    rgba = np.zeros(rate.shape + (4,), np.uint8)
    for channel in range(3):
        stop_y = np.array([s[1][channel] for s in _STOPS], float)
        rgba[..., channel] = np.interp(log_rate, stop_x, stop_y).astype(np.uint8)

    # Fade alpha in across the drizzle range so light rain doesn't show up as
    # a hard-edged blob at the detection threshold.
    fade = (log_rate - np.log10(MIN_RATE)) / (np.log10(0.3) - np.log10(MIN_RATE))
    rgba[..., 3] = (np.clip(fade, 0.0, 1.0) * 215).astype(np.uint8)
    rgba[rate < MIN_RATE, 3] = 0
    return rgba


def png_bytes(rgba, colors=64):
    """PNG-encode an RGBA overlay, palette-quantized.

    Colour and alpha are both pure functions of one value (the rain rate), so
    a 64-entry palette is visually lossless here - and it takes a frame from
    ~370 KB of true-colour noise down to a few tens of KB, which matters when
    the page preloads two dozen of them.
    """
    from PIL import Image

    img = Image.fromarray(rgba, "RGBA").quantize(
        colors=colors, method=Image.Quantize.FASTOCTREE
    )
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
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


def resample(field, rows, cols):
    from scipy.ndimage import map_coordinates

    return map_coordinates(
        np.nan_to_num(field, nan=0.0), [rows, cols], order=1, mode="constant", cval=0.0
    )


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
    """Self-check: colour scale is monotonic-ish and alpha gates correctly."""
    field = np.array([[0.0, 0.02, 0.1, 1.0, 10.0, 100.0]])
    rgba = colorize(field)
    assert rgba.shape == (1, 6, 4)
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 0, "sub-threshold must be clear"
    assert rgba[0, 3, 3] > 0 and rgba[0, 5, 3] > 0, "rain must be opaque"
    assert tuple(rgba[0, 5, :3]) != tuple(rgba[0, 3, :3]), "heavy != light colour"

    blob = png_bytes(colorize(np.random.default_rng(0).random((64, 64)) * 20))
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"

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
