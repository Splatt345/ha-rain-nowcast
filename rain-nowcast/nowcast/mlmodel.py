"""SmaAt-UNet inference for the add-on, via ONNX Runtime.

The research repo's `nowcast/smaat.py` builds, trains, evaluates and runs this
network and needs PyTorch to do it. Serving needs only the forward pass, so the
add-on carries this instead and PyTorch never enters the image.

WHY. Measured inside the image: torch is 711 MB of site-packages and drags in
sympy (73 MB) and networkx (18 MB) that nothing else uses - about 800 MB to
evaluate a 4.02 M-parameter convnet. onnxruntime is 66 MB and runs the same
graph. It is also faster, because torch's eager mode re-dispatches every op
while ONNX Runtime plans the graph once:

    input          torch      onnxruntime
    256x256        0.37 s        0.12 s
    352x368        0.53 s        0.25 s     <- the live crop, padded
    384x384        0.60 s        0.29 s

The exported graph is numerically the same model, not an approximation:
agreement is 1e-7 in model units, 0.00001 mm/h after the inverse transform, at
every size tested. Regenerate models/smaat.onnx from the .pt checkpoint with
`python -m nowcast.smaat --export-onnx` in the research repo; do not hand-edit
it, and do not let it drift from the checkpoint it came from.
"""
import os

import numpy as np

#: 12 frames in, 12 out. On the add-on's 5-minute OPERA feed that is one hour
#: in and one hour out. The same frame counts mean 3 h -> 3 h on the 15-minute
#: archive product, which is why the research repo labours the point.
N_INPUT, N_OUTPUT = 12, 12
LOG_SCALE = float(np.log1p(100.0))  # 100 mm/h maps to 1.0

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "smaat.onnx")

_SESSION = {}


def to_model(rain_mm_h):
    """mm/h -> roughly [0, 1]. Rain is heavy-tailed; a network trained on raw
    mm/h spends all its capacity on the rare extremes."""
    x = np.nan_to_num(np.asarray(rain_mm_h, dtype=np.float32), nan=0.0)
    return np.log1p(np.clip(x, 0.0, None)) / LOG_SCALE


def from_model(y):
    """Inverse of to_model, clamped to the range the transform is defined over.

    The upper clip is not cosmetic. The output layer is an unbounded Conv2d, so
    nothing stops the network emitting y > 1, and expm1 turns that into nonsense
    fast: y = 1.67 becomes 2163 mm/h, which is what the first trained checkpoint
    actually did.
    """
    return np.expm1(np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0) * LOG_SCALE)


def _session(weights):
    if weights not in _SESSION:
        import onnxruntime as ort

        if not os.path.exists(weights):
            raise FileNotFoundError(f"No model at {weights}")
        opts = ort.SessionOptions()
        # One radar cycle every 5 minutes on a box that is also running Home
        # Assistant: capping threads keeps the forecast from monopolising a Pi.
        # Measured on the research desktop, 12 threads was SLOWER than 4 for
        # this graph anyway (824 ms against 603 ms) - oversubscription.
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        # The single biggest memory win in the add-on, measured on the live
        # crop padded to 352x368:
        #
        #   arena on   RSS +792 MB   0.22 s/run
        #   arena off  RSS + 59 MB   0.23 s/run
        #
        # ONNX Runtime's arena allocator pre-allocates for the graph's peak
        # intermediate activations and never gives it back, which is a sensible
        # default for a server answering thousands of requests a second and a
        # terrible one for a forecast that runs once every five minutes. Turning
        # it off costs nothing measurable in time and 733 MB less resident.
        opts.enable_cpu_mem_arena = False
        _SESSION[weights] = ort.InferenceSession(
            weights, opts, providers=["CPUExecutionProvider"])
    return _SESSION[weights]


def forecast(train_stack, n_lead, weights=WEIGHTS):
    """(n_lead, H, W) rain rates in mm/h from the last N_INPUT observed frames.

    Raises rather than returning something plausible if the model is missing or
    its output is unphysical - a silent fallback is how a broken model gets
    served as if it were real. The caller decides what to do about it.
    """
    x = to_model(np.asarray(train_stack, dtype=np.float32)[-N_INPUT:])

    # Pad to a multiple of 16 so the four downsamplings divide evenly. Not
    # optional: the live Belgium crop is 338x364 and the U-Net's skip
    # connections fail to concatenate at any size that is not.
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)))

    y = _session(weights).run(None, {"x": x[None]})[0][0]
    if ph or pw:
        y = y[:, :h, :w]

    out = from_model(y)[:n_lead]
    if not np.all(np.isfinite(out)):
        raise ValueError("model produced non-finite rain rates")
    # The DGMR lesson: check the physical range before anything uses this.
    if np.nanmax(out) > 500.0:
        raise ValueError(f"model produced {np.nanmax(out):.0f} mm/h - "
                         "physically impossible; refusing to serve it")
    return out


def _demo():
    """Self-check: transforms round-trip, and the model runs at the live size."""
    probe = np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    assert np.allclose(from_model(to_model(probe)), probe, atol=1e-3), "round trip"
    assert from_model(np.array([2.0]))[0] <= 100.0 + 1e-3, "must clamp above 1.0"
    assert to_model(np.array([np.nan]))[0] == 0.0, "NaN must map to dry"

    if not os.path.exists(WEIGHTS):
        print("mlmodel self-check ok (transforms only; no model file present)")
        return

    rng = np.random.default_rng(0)
    # 338x364 is the live crop and is NOT a multiple of 16 - the padding path is
    # the one that actually runs in production, so it is the one tested.
    stack = np.abs(rng.gamma(1.0, 2.0, size=(N_INPUT, 338, 364))).astype(np.float32)
    out = forecast(stack, 12)
    assert out.shape == (12, 338, 364), out.shape
    assert np.isfinite(out).all() and out.min() >= 0.0
    assert out.max() <= 100.0 + 1e-3, out.max()
    dry = forecast(np.zeros((N_INPUT, 64, 64), dtype=np.float32), 12)
    assert dry.max() < 1.0, f"dry input must stay dry, got {dry.max()}"
    print(f"mlmodel self-check ok (338x364 -> {out.shape}, max {out.max():.2f} mm/h)")


if __name__ == "__main__":
    _demo()
