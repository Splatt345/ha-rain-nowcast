"""Sub-frame temporal interpolation by advection.

Radar arrives every 5 minutes, but people want to know which *minute* the rain
reaches them. Interpolating linearly between two frames at a fixed pixel is
wrong in a specific way: it makes a shower fade in over five minutes instead of
arriving, and it clips the peak whenever the peak passes between two frames.

Rain doesn't fade in, it moves. So to get the value at a point part-way between
two frames, trace back along the motion field to where that air was at the
previous frame, and forward to where it will be at the next one, then blend the
two estimates. Same idea as the semi-Lagrangian scheme pysteps already uses to
extrapolate - just applied at fractional timesteps.
"""
import numpy as np


def _sample(field, rows, cols):
    """Bilinear sample at fractional (rows, cols), edges clamped."""
    from scipy.ndimage import map_coordinates

    return map_coordinates(
        np.nan_to_num(field, nan=0.0), [np.atleast_1d(rows), np.atleast_1d(cols)],
        order=1, mode="nearest",
    )


def _box_offsets(box):
    span = np.arange(-box, box + 1)
    dr, dc = np.meshgrid(span, span, indexing="ij")
    return dr.ravel(), dc.ravel()


def point_series(frames, motion, row, col, sub_steps, box=1):
    """Values at (row, col) for every sub-step across `frames`.

    frames    (n, y, x) mm/h, consecutive, one motion-timestep apart
    motion    (2, y, x) pixels per timestep, pysteps convention (x, then y)
    sub_steps how many sub-intervals per frame gap (5 -> one per minute
              when frames are 5 minutes apart)
    box       half-width of the averaging box, in pixels

    Returns (n - 1) * sub_steps + 1 values, first aligned with frames[0].
    """
    frames = np.asarray(frames)
    dr, dc = _box_offsets(box)
    rows, cols = row + dr, col + dc

    if motion is None:  # no motion estimate: fall back to a plain time blend
        vx = vy = np.zeros_like(rows, dtype=float)
    else:
        vx = _sample(motion[0], rows, cols)
        vy = _sample(motion[1], rows, cols)

    out = []
    for k in range(len(frames) - 1):
        for s in range(sub_steps):
            d = s / sub_steps
            # where this air sat at frame k, and where it lands at frame k+1
            back = _sample(frames[k], rows - vy * d, cols - vx * d)
            fwd = _sample(frames[k + 1], rows + vy * (1 - d), cols + vx * (1 - d))
            out.append(float(np.mean((1 - d) * back + d * fwd)))
    out.append(float(np.mean(_sample(frames[-1], rows, cols))))
    return np.array(out)


def _demo():
    """Self-check against a case with a known analytic answer: a Gaussian blob
    crossing the domain at a fixed velocity, sampled at a fixed point.

    Written before the wiring, and deliberately checks the property that
    matters - recovering the peak as it passes between two frames - rather
    than whatever the implementation happens to produce.
    """
    ny = nx = 64
    yy, xx = np.mgrid[0:ny, 0:nx]
    vx, vy, sigma = 4.0, 0.0, 2.5  # 4 px per frame, straight across
    peak = 10.0

    def field(t):
        """Blob centre at x = 10 + vx*t, y = 32."""
        return peak * np.exp(-(((xx - (10 + vx * t)) ** 2 + (yy - 32) ** 2)
                               / (2 * sigma ** 2)))

    # col 28 puts the blob's closest approach at t = 4.5 frames - deliberately
    # *between* two frames, which is the case linear interpolation gets wrong
    row, col, sub = 32, 28, 5
    frames = np.stack([field(t) for t in range(6)])
    motion = np.stack([np.full((ny, nx), vx), np.full((ny, nx), vy)])

    got = point_series(frames, motion, row, col, sub_steps=sub, box=0)
    truth = np.array([field(t / sub)[row, col] for t in range(len(got))])
    linear = np.interp(np.arange(len(got)) / sub, np.arange(6), frames[:, row, col])

    assert len(got) == 5 * sub + 1, len(got)
    # frame times must reproduce the frames exactly
    for k in range(6):
        assert abs(got[k * sub] - frames[k, row, col]) < 1e-6, k

    # peak passes the point at t = 4.5 frames - between samples, so linear
    # interpolation necessarily clips it
    adv_err = np.abs(got - truth).max()
    lin_err = np.abs(linear - truth).max()
    assert adv_err < 0.35, f"advected max error {adv_err:.3f}"
    assert lin_err > 2 * adv_err, f"linear {lin_err:.3f} vs advected {adv_err:.3f}"
    assert abs(got.max() - peak) < 0.35, f"peak recovered as {got.max():.2f}"
    assert linear.max() < peak - 1.0, "linear should clip the peak"

    # a still field must stay still whatever the motion
    still = np.stack([np.full((ny, nx), 3.0)] * 3)
    flat = point_series(still, motion, row, col, sub_steps=sub, box=1)
    assert np.allclose(flat, 3.0), flat

    # no motion estimate -> degrade to the linear blend, not an exception
    none_motion = point_series(frames, None, row, col, sub_steps=sub, box=0)
    assert np.allclose(none_motion, linear, atol=1e-6)

    print(f"interpolate self-check ok "
          f"(peak {got.max():.2f}/{peak:.0f} advected vs {linear.max():.2f} linear; "
          f"max err {adv_err:.3f} vs {lin_err:.3f})")


if __name__ == "__main__":
    _demo()
