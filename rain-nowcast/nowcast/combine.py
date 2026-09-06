"""Combine the STEPS ensemble with the SmaAt-UNet second opinion.

Ported from the research repo, trimmed to what the add-on needs (the research
copy also carries the CLI and the ANVIL demo path, which depend on its
evaluation harness). Keep the two in step if the maths changes.

WHY A SHIFT AND NOT A BLEND. The obvious combiner, out_i = (1-w)*steps_i + w*ml,
scales every member's deviation from the mean by (1-w) and so throws away the
ensemble spread in proportion to the weight - half of it at w = 0.5. Spread is
the scarce resource here (measured dispersion 0.747, worsening with lead) and
the ensemble is the single most valuable object the add-on has: it is what
`rain_chance` and the uncertainty fan are made of. So the mean is moved and the
deviations are left alone:

    out_i = steps_i + w * (ml - mean(steps))

The mean lands exactly where convex blending would put it, while
out_i - mean(out) == steps_i - mean(steps) for every w.

WHAT IT IS WORTH, measured on 30 CIRRUS validation windows never trained on,
against the pre-registered CRPS gate (PREREGISTRATION.md):

    steps_ens 0.38568   combined 0.36894   dCRPS +0.01674   wins 25/30

That is 3.3x the 0.005 floor, and the weight was not tuned to get it - see
`weights_from_skill`.
"""
import numpy as np

#: Per-lead weight toward the model, from `weights_from_skill` on the measured
#: per-lead errors of both forecasts (STEPS mean 0.351..0.823 mm/h, SmaAt-UNet
#: 0.290..0.556 over 5..60 min). Recomputed, not hand-picked, and rising with
#: lead because that is where the network's advantage grows.
WEIGHTS = np.array([0.59, 0.59, 0.62, 0.62, 0.63, 0.65,
                    0.66, 0.66, 0.67, 0.68, 0.68, 0.69])


def weights_from_skill(err_steps, err_other):
    """Closed-form inverse-error weight per lead: w = e_s^2 / (e_s^2 + e_o^2).

    Optimal under INDEPENDENT errors, which these are not - both forecasts
    extrapolate the same radar, so their errors correlate and this over-weights
    the second opinion somewhat. It is still the right way to set w: the
    alternative is searching for the value that maximises a score, and that has
    failed three times in this project's history.
    """
    s = np.asarray(err_steps, dtype=float) ** 2
    o = np.asarray(err_other, dtype=float) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where((s + o) > 0, s / (s + o), 0.0)
    return np.clip(np.nan_to_num(w), 0.0, 1.0)


def combine(members, other, w):
    """Shift `members` toward `other` by `w`, largely preserving spread.

    members  (m, n_lead, H, W) ensemble, or (n_lead, H, W) for a single field
    other    (n_lead, H, W) deterministic second opinion
    w        scalar, or per-lead array of length n_lead
    """
    members = np.asarray(members, dtype=float)
    other = np.asarray(other, dtype=float)
    single = members.ndim == other.ndim
    if single:
        members = members[None]

    w = np.asarray(w, dtype=float)
    if w.ndim == 1:                       # per-lead -> broadcast over (lead,y,x)
        w = w[:, None, None]

    mean = np.nanmean(members, axis=0)
    shift = w * (other - mean)
    # The clip is why spread is "largely" and not "exactly" preserved. The shift
    # is exact for every w, but rain cannot be negative and clipping is
    # nonlinear, so it compresses the low tail wherever a member is pushed under
    # zero. Measured at these weights: 15.6% of member values clipped, spread
    # returned at 0.9435 of the original. Still far better than the 50% convex
    # blending would have cost at w = 0.5.
    out = np.clip(members + shift[None], 0.0, None)
    return out[0] if single else out


def _demo():
    rng = np.random.default_rng(0)
    ens = np.abs(rng.normal(2.0, 0.5, size=(12, 4, 8, 8)))
    ml = np.abs(rng.normal(3.0, 0.5, size=(4, 8, 8)))

    out = combine(ens, ml, 0.5)
    # the deviations survive exactly when nothing clips
    assert np.allclose(out - out.mean(axis=0), ens - ens.mean(axis=0), atol=1e-9)
    assert np.allclose(out.mean(axis=0), 0.5 * ens.mean(axis=0) + 0.5 * ml, atol=1e-9)

    # and the naive blend would not: half the spread gone at w = 0.5
    naive = 0.5 * ens + 0.5 * ml[None]
    assert np.allclose(naive.std(axis=0), 0.5 * ens.std(axis=0), atol=1e-9)

    # clipping is the documented exception, not a surprise
    low = combine(ens, np.zeros_like(ml) - 50.0, 1.0)
    assert (low >= 0).all() and low.std(axis=0).sum() < ens.std(axis=0).sum()

    w = weights_from_skill([1.0, 2.0], [1.0, 1.0])
    assert np.isclose(w[0], 0.5) and w[1] > w[0], w
    assert len(WEIGHTS) == 12 and (WEIGHTS > 0).all() and (WEIGHTS < 1).all()
    print("combine self-check ok")


if __name__ == "__main__":
    _demo()
