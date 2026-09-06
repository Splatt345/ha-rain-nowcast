"""SmaAt-UNet scaffolding: model, data pipeline, training entry, inference.

STATUS. Trained once as a shakedown (2026-09-05) on 30 archive days / 2190
windows: it beats persistence at every lead (0.300 vs 0.440 mm/h MAE overall)
but that is a pipeline validation, not a competitive model - the paper used
~420k maps. `forecast()` still refuses to run without weights rather than
inventing any.

WHY THIS ARCHITECTURE. The techniques report ranks the ML options by what
actually fits one 12 GB consumer GPU (SS2). SmaAt-UNet is the only one that fits
comfortably: 4.1 M parameters against a plain U-Net's 17.4 M, trained by its
authors on four years of KNMI radar (~420k maps) on a single GPU. DGMR needed
16 TPU cores for a week and is explicitly out of budget by three orders of
magnitude. The match here is unusually good - the frozen event set is KNMI
5-minute 1 km composites, the same instrument class the paper trained on.

  Trebing, Stanczyk, Mehrkanoon (2021) - https://arxiv.org/abs/2007.04417
  U-Net with every convolution made depthwise-separable, plus CBAM attention
  on each skip connection. That is where the 4x parameter reduction comes from.

THE BLOCKER, STATED PLAINLY. Data, still. `nowcast/archive.py` now harvests
`openradar-archive` (anonymous, keyless, 13 years deep) at ~37 s/day, which is
what unblocked training at all - but the counts remain small next to the
paper's ~420k maps, and `train()` guards on it.

TWO CHOICES MADE FROM THIS PROJECT'S OWN MEASUREMENTS, not from the paper:

1. **The loss is MAE, not MSE.** `nowcast/anatomy.py` measured that MSE is
   minimised by the conditional mean and that the resulting blur is precisely
   what makes the current baseline under-dispersed (sd_p/sd_o = 0.747, growing
   with lead time). Training a network on MSE would reproduce that failure
   deliberately. RainNet's published behaviour is the same story: it wins CSI
   at drizzle thresholds by blurring, with effective resolution degrading to
   32 km at 60 min. MAE is less blur-prone. See `LOSSES` for the alternatives,
   including the exceedance-segmentation target that Ayzel & Heistermann (2025)
   report gaining ~0.06 CSI over regression.

2. **The rain transform is explicitly invertible and round-trip asserted.**
   Landmine #1 in HANDOVER.md is a silent dB transform that invalidated every
   RMSE number in the project once. `to_model`/`from_model` are checked to
   round-trip in `_selfcheck` rather than trusted.

MEASURED ON THE TARGET HARDWARE (RTX 4070, 11.6 GiB usable). Full speed table
and the channels_last warning are in `train()`; the short version is bf16 +
TF32 + torch.compile, 1.91x faster and 2.8x less memory than fp32, and
channels_last is 2.4x SLOWER here despite being the standard advice.

    python -m nowcast.smaat --selfcheck      # architecture + shapes + transform
    python -m nowcast.smaat --summary        # parameter count vs the paper
    python -m nowcast.smaat --overfit        # can the training loop learn at all
    python -m nowcast.smaat --train          # train on the harvested archive
"""
import argparse
import os

import numpy as np

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "data", "smaat.pt")

# 12 frames in, 12 out. The WALL-CLOCK span depends entirely on the source
# cadence, and the two sources this repo uses do not agree:
#   * OPERA archive (what the model is trained on): 15 min -> 3 h in, 3 h out
#   * KNMI event set / the live add-on:              5 min -> 1 h in, 1 h out
# So a model trained on the archive is a THREE-HOUR, 15-minute-resolution, 2 km
# forecaster. It is not a drop-in for the add-on, which serves one hour at
# 5-minute resolution on a 1 km grid. See HANDOVER.md 7b.
N_INPUT, N_OUTPUT = 12, 12
LOG_SCALE = float(np.log1p(100.0))  # 100 mm/h maps to 1.0


# --- rain <-> network units -------------------------------------------------

def to_model(rain_mm_h):
    """mm/h -> roughly [0, 1]. Rain is heavy-tailed; a network trained on raw
    mm/h spends all its capacity on the rare extremes."""
    x = np.nan_to_num(np.asarray(rain_mm_h, dtype=np.float32), nan=0.0)
    return np.log1p(np.clip(x, 0.0, None)) / LOG_SCALE


def from_model(y):
    """Inverse of to_model, clamped to the range the transform is defined over.

    Clipped at BOTH ends, and the upper clip is not cosmetic. The output layer
    is an unbounded Conv2d, so nothing stops the network emitting y > 1, and
    expm1 turns that into nonsense fast: y = 1.67 becomes 2163 mm/h. Measured on
    the first trained checkpoint - predicted max 2163.3 mm/h against a truth max
    of 99.9. That is the same >3000 mm/h failure that killed the openclimatefix
    DGMR checkpoint (HANDOVER.md SS6), reproduced in our own model.

    y = 1.0 is 100 mm/h by construction (LOG_SCALE), and the training data is
    capped at 99.9 mm/h by archive.MAX_DBZ, so the network can never legitimately
    need more. Clamping here makes the transform invertible over exactly the
    range it was fitted on rather than extrapolating off a cliff.
    """
    return np.expm1(np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)
                    * LOG_SCALE)


# --- model ------------------------------------------------------------------

def build_model(n_input=N_INPUT, n_output=N_OUTPUT, bilinear=True,
                reduction=16, kernel_size=7, kernels_per_layer=2):
    """SmaAt-UNet. Imports torch lazily so the rest of this module - the data
    pipeline, the transforms, the self-check - works without it installed.

    `kernels_per_layer` is the depthwise depth multiplier and matters more than
    it looks: at 1 the model comes out at 2.05M parameters, half the paper's
    4.1M. The reference implementation uses 2, and the parameter count is the
    only cheap way to catch an architecture that is subtly not the published
    one - which is why _selfcheck asserts it rather than trusting the shapes.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DepthwiseSeparableConv(nn.Module):
        """The parameter saving: a per-channel spatial conv followed by a 1x1
        mixing conv, instead of one dense conv over all channels. The depthwise
        stage emits cin*kernels_per_layer channels before the mix."""

        def __init__(self, cin, cout, kernel_size=3, padding=1, kpl=1):
            super().__init__()
            self.depthwise = nn.Conv2d(cin, cin * kpl, kernel_size,
                                       padding=padding, groups=cin, bias=False)
            self.pointwise = nn.Conv2d(cin * kpl, cout, 1, bias=False)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    class DoubleConvDS(nn.Module):
        def __init__(self, cin, cout, mid=None):
            super().__init__()
            mid = mid or cout
            kpl = kernels_per_layer
            self.block = nn.Sequential(
                DepthwiseSeparableConv(cin, mid, kpl=kpl), nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(mid, cout, kpl=kpl), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True))

        def forward(self, x):
            return self.block(x)

    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction):
            super().__init__()
            hidden = max(1, channels // reduction)
            self.mlp = nn.Sequential(nn.Linear(channels, hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden, channels))

        def forward(self, x):
            b, c, _, _ = x.shape
            avg = self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c))
            mx = self.mlp(F.adaptive_max_pool2d(x, 1).view(b, c))
            return x * torch.sigmoid(avg + mx).view(b, c, 1, 1)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size,
                                  padding=kernel_size // 2, bias=False)

        def forward(self, x):
            avg = x.mean(dim=1, keepdim=True)
            mx = x.max(dim=1, keepdim=True)[0]
            return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

    class CBAM(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.ca = ChannelAttention(channels, reduction)
            self.sa = SpatialAttention(kernel_size)

        def forward(self, x):
            return self.sa(self.ca(x))

    class Down(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConvDS(cin, cout))

        def forward(self, x):
            return self.block(x)

    class Up(nn.Module):
        def __init__(self, cin, cout, bilinear):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                      align_corners=True)
                self.conv = DoubleConvDS(cin, cout, mid=cin // 2)
            else:
                self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
                self.conv = DoubleConvDS(cin, cout)

        def forward(self, x, skip):
            x = self.up(x)
            # odd input sizes leave a one-pixel mismatch against the skip
            dy = skip.shape[2] - x.shape[2]
            dx = skip.shape[3] - x.shape[3]
            if dy or dx:
                x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            return self.conv(torch.cat([skip, x], dim=1))

    class SmaAtUNet(nn.Module):
        def __init__(self):
            super().__init__()
            factor = 2 if bilinear else 1
            self.inc = DoubleConvDS(n_input, 64)
            self.cbam1 = CBAM(64)
            self.down1, self.cbam2 = Down(64, 128), CBAM(128)
            self.down2, self.cbam3 = Down(128, 256), CBAM(256)
            self.down3, self.cbam4 = Down(256, 512), CBAM(512)
            self.down4 = Down(512, 1024 // factor)
            self.cbam5 = CBAM(1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = nn.Conv2d(64, n_output, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.cbam5(self.down4(x4))
            x = self.up1(x5, self.cbam4(x4))
            x = self.up2(x, self.cbam3(x3))
            x = self.up3(x, self.cbam2(x2))
            x = self.up4(x, self.cbam1(x1))
            return self.outc(x)

    return SmaAtUNet()


# --- data -------------------------------------------------------------------

class RadarSequences:
    """(input, target) windows cut from radar stacks.

    Deliberately source-agnostic: hand it any iterable of (n_frames, H, W) mm/h
    arrays. Today that is `archive.load_stacks`, a generator; the 20 frozen KNMI
    events also fit and are useful only for shape-checking the pipeline.

    It CONSUMES the iterable one stack at a time and keeps only the cropped
    float16 model units, because the raw float32 does not fit: 149 CIRRUS days
    is 21.6 GB and a second transformed copy made it 43 GB against 18 GB of RAM.
    Three things follow, all of them saving memory at no cost to the training:

      * the raw stack is never retained - it was stored as `self.stacks` and
        read by nothing;
      * `crop` is applied HERE rather than in `__getitem__`. It is a fixed
        centre crop, so the result is identical, and the per-window slicing goes
        away;
      * storage is float16. `to_model` output lies in [0, 1] and the forward
        pass runs under bf16 autocast, which has FEWER mantissa bits, so this
        cannot be the limiting precision. `__getitem__` hands back float32.
    """

    def __init__(self, stacks, n_input=N_INPUT, n_output=N_OUTPUT, stride=1,
                 crop=None, min_wet=0.0):
        # Transform ONCE per stack, not once per window. Consecutive windows
        # share 23 of 24 frames, so doing it in __getitem__ recomputes log1p
        # about 24x more than necessary and makes the CPU the bottleneck.
        self.model_units = []
        self.n_input, self.n_output = n_input, n_output
        self.crop, self.min_wet = crop, min_wet
        self.index = []
        span = n_input + n_output
        for s, stack in enumerate(stacks):
            for start in range(0, len(stack) - span + 1, stride):
                if min_wet > 0:
                    # thresholded on RAW mm/h, before the transform below
                    wet = np.mean(np.nan_to_num(stack[start + n_input - 1]) > 1.0)
                    if wet < min_wet:
                        continue
                self.index.append((s, start))
            units = to_model(stack)
            if crop:
                h, w = crop
                y0, x0 = (units.shape[1] - h) // 2, (units.shape[2] - w) // 2
                units = units[:, y0:y0 + h, x0:x0 + w]
            self.model_units.append(units.astype(np.float16))
            del stack, units

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, start = self.index[i]
        window = self.model_units[s][start:start + self.n_input + self.n_output]
        window = window.astype(np.float32)
        return (np.ascontiguousarray(window[:self.n_input]),
                np.ascontiguousarray(window[self.n_input:]))


#: name -> (torch.nn attribute, note). Looked up by getattr, never eval'd.
#: "mae" is the default: MSE would reproduce, on purpose, the blur that
#: anatomy.py measured as the baseline's central weakness. "exceedance" is
#: Ayzel & Heistermann (2025), who report ~0.06 CSI over regression by
#: predicting threshold exceedance instead of rate - it needs a binary target,
#: so switching to it means changing RadarSequences' target too.
LOSSES = {
    "mae": ("L1Loss", "default; less blur-prone than MSE"),
    "mse": ("MSELoss", "blur-prone; see the module docstring"),
    "exceedance": ("BCEWithLogitsLoss", "needs a binary target"),
}


def _criterion(name):
    import torch.nn as nn

    return getattr(nn, LOSSES[name][0])()


def train(stacks, val_stacks=None, epochs=50, batch_size=16, lr=1e-3,
          loss="mae", crop=(256, 256), device=None, out_path=WEIGHTS,
          min_samples=10000, compile_model=True, amp=True, stride=1):
    """Train SmaAt-UNet. Refuses to run on a dataset that cannot support it.

    The guard is the point: a network trained on 480 radar maps produces weights
    that look like a model and behave like noise, and this project has already
    been burned once by a checkpoint emitting >3000 mm/h (the openclimatefix
    DGMR reproduction, HANDOVER.md SS6).

    SPEED SETTINGS ARE MEASURED, not assumed (RTX 4070, crop 256):

        fp32                        169.1 ms/step   9.08 GiB   3.10 Mpx/s
        + TF32                      152.5 ms        9.08 GiB   3.44
        + channels_last             360.0 ms        6.36 GiB   1.46  <- HARMFUL
        + bf16 autocast             112.5 ms        3.26 GiB   4.66
        + bf16 + torch.compile      (bs16)          5.27 GiB   5.93

    1.91x end-to-end and 2.8x less memory. channels_last is DISABLED
    deliberately: it is the standard advice and it is 2.4x slower here, because
    cuDNN lacks good NHWC kernels for the depthwise (grouped) convolutions this
    architecture is built from. Do not "fix" that without re-measuring.

    bf16 rather than fp16: no GradScaler, no loss-scale tuning, and Ada handles
    it natively. torch.compile costs ~39 s once.
    """
    import torch
    from torch.utils.data import DataLoader

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True          # fixed shapes, so this pays

    ds = RadarSequences(stacks, crop=crop, stride=stride)
    print(f"train {len(ds.model_units)} days / {len(ds)} windows", flush=True)
    if len(ds) < min_samples:
        raise RuntimeError(
            f"{len(ds)} training windows available, need >= {min_samples}. "
            "SmaAt-UNet's authors used ~420k radar maps over four years. "
            "Harvest openradar-archive first (HANDOVER.md 10.1), or pass "
            "min_samples explicitly to override this on purpose.")
    val = RadarSequences(val_stacks, crop=crop, stride=stride) if val_stacks else None
    if val:
        print(f"val   {len(val.model_units)} days / {len(val)} windows", flush=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    if compile_model and device == "cuda":
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4)
    criterion = _criterion(loss)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (
        amp and device == "cuda") else torch.enable_grad

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = (DataLoader(val, batch_size=batch_size, num_workers=2,
                             pin_memory=True) if val else None)

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                batch_loss = criterion(model(x), y)
            batch_loss.backward()
            opt.step()
            total += float(batch_loss)
        mean = total / max(1, len(loader))

        v = float("nan")
        if val_loader:
            model.eval()
            vt = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast():
                        vt += float(criterion(model(x), y))
            v = vt / max(1, len(val_loader))
            sched.step(v)
        else:
            sched.step(mean)

        print(f"epoch {epoch + 1}/{epochs}  train {mean:.5f}  val {v:.5f}"
              f"  lr {opt.param_groups[0]['lr']:.2e}", flush=True)

        # Checkpoint on VALIDATION, never on training loss - the whole point of
        # the day-level split is that it is the only honest signal here.
        score = v if val_loader else mean
        if score < best:
            best = score
            raw = getattr(model, "_orig_mod", model)     # unwrap torch.compile
            torch.save({"state_dict": raw.state_dict(), "n_input": N_INPUT,
                        "n_output": N_OUTPUT, "loss": loss, "epoch": epoch + 1,
                        "val": v, "train": mean}, out_path)
    return out_path


# --- inference --------------------------------------------------------------

def evaluate_checkpoint(val_stacks, weights=WEIGHTS, batch_size=16,
                        crop=(256, 256), stride=1, device=None):
    """Score a trained checkpoint in mm/h against persistence and zeros.

    The validation loss a run prints is in model units - log1p(mm/h)/LOG_SCALE -
    and is comparable to nothing: not to another run with a different transform,
    not to the KNMI numbers in HANDOVER.md SS3. This converts back to mm/h and
    reports MAE per lead, which is the only form anyone can act on.

    Two baselines, and both are needed. PERSISTENCE (hold the last observed
    frame) is the one a nowcast must beat to justify existing. ZEROS (predict no
    rain ever) is the one that catches a model which has quietly learned that
    the domain is mostly dry - on this data that is a genuinely hard baseline,
    and the ODYSSEY run beat it only 0.352 to 0.428.

    Scored at the training crop rather than the full domain: the network is
    fully convolutional and runs at any size, but its training distribution was
    the 256x256 centre, and evaluating elsewhere measures generalisation to a
    different field of view rather than forecast skill.
    """
    import torch
    from torch.utils.data import DataLoader

    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_model(ckpt.get("n_input", N_INPUT), ckpt.get("n_output", N_OUTPUT))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds = RadarSequences(val_stacks, crop=crop, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    def mm(t):
        """from_model() in torch, so the whole sweep stays on the GPU."""
        return torch.expm1(t.clamp(0.0, 1.0) * LOG_SCALE)

    # The duplicated transform above is the kind of thing that silently drifts,
    # so it is checked against the audited numpy one rather than trusted.
    probe = np.linspace(0.0, 1.0, 51, dtype=np.float32)
    assert np.allclose(mm(torch.from_numpy(probe)).numpy(), from_model(probe),
                       atol=1e-4), "torch/numpy rain transforms disagree"

    n_out = ckpt.get("n_output", N_OUTPUT)
    err = {k: torch.zeros(n_out, dtype=torch.float64, device=device)
           for k in ("model", "persistence", "zeros")}
    wet = {k: torch.zeros(n_out, dtype=torch.float64, device=device)
           for k in ("model", "persistence", "obs")}
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(x)
            obs = mm(y.float())
            got = {"model": mm(pred.float()),
                   # persistence: hold the last OBSERVED frame at every lead
                   "persistence": mm(x[:, -1:].float()).expand_as(obs),
                   "zeros": torch.zeros_like(obs)}
            for k, v in got.items():
                err[k] += (v - obs).abs().sum(dim=(0, 2, 3)).double()
            for k in ("model", "persistence"):
                wet[k] += (got[k] > 1.0).sum(dim=(0, 2, 3)).double()
            wet["obs"] += (obs > 1.0).sum(dim=(0, 2, 3)).double()
            n += obs.shape[0] * obs.shape[2] * obs.shape[3]

    return {"n_windows": len(ds), "n_samples": int(n), "epoch": ckpt.get("epoch"),
            "val": ckpt.get("val"),
            "mae": {k: (v / n).cpu().numpy() for k, v in err.items()},
            "wet": {k: (v / n).cpu().numpy() for k, v in wet.items()}}


def report_eval(res, cadence_min=5.0):
    mae, wet = res["mae"], res["wet"]
    n_out = len(mae["model"])
    print(f"\ncheckpoint epoch {res['epoch']}  val {res['val']:.5f} (model units)")
    print(f"{res['n_windows']} validation windows, {res['n_samples']:,} pixel-samples\n")
    print("mm/h MAE overall   zeros %.4f   persistence %.4f   SmaAt-UNet %.4f"
          % (mae["zeros"].mean(), mae["persistence"].mean(), mae["model"].mean()))

    heads = "".join(f"{int((i + 1) * cadence_min):>7d}" for i in range(n_out))
    print(f"\nlead (min)   {heads}")
    # zeros is printed per lead, not just as an overall figure, because on a
    # domain this dry it can BEAT persistence: MAE is minimised by the median,
    # and the median pixel is dry. Reporting only "beats persistence" against a
    # baseline that is itself beaten by predicting nothing would flatter.
    for name, key in (("zeros", "zeros"), ("persistence", "persistence"),
                      ("SmaAt-UNet", "model")):
        print(f"{name:<12} " + "".join(f"{v:>7.3f}" for v in mae[key]))
    gain = 100 * (1 - mae["model"] / np.maximum(mae["persistence"], 1e-12))
    print("improvement  " + "".join(f"{v:>6.0f}%" for v in gain))

    print("\nwet area at 1 mm/h (fraction of pixels)")
    print(f"  observed    {wet['obs'].mean():.4f}")
    for name, key in (("SmaAt-UNet", "model"), ("persistence", "persistence")):
        print(f"  {name:<11} {wet[key].mean():.4f}   "
              f"bias {wet[key].mean() / max(wet['obs'].mean(), 1e-12):.2f}x")


_CACHED = {}


def forecast(train_stack, n_lead, weights=WEIGHTS, device=None):
    """Forecast function in the project's interface: (n_lead, H, W) in mm/h.

    Raises if no weights exist. It does NOT fall back to something plausible -
    a silent fallback is how a broken model gets scored as if it were real.
    """
    import torch

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"No trained weights at {weights}. This model is scaffolding only; "
            "see the module docstring for why training is blocked on data.")

    if weights not in _CACHED:
        # weights_only=True: the checkpoint holds tensors and plain
        # values only, and unpickling arbitrary objects from a
        # weights file is a remote-code-execution path.
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        model = build_model(ckpt.get("n_input", N_INPUT),
                            ckpt.get("n_output", N_OUTPUT))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _CACHED[weights] = model
    model = _CACHED[weights]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = to_model(np.asarray(train_stack, dtype=np.float32)[-N_INPUT:])
    # pad to a multiple of 16 so the four downsamplings divide evenly
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
    with torch.no_grad():
        y = model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
    if ph or pw:
        y = y[:, :h, :w]

    out = from_model(y)[:n_lead]
    if not np.all(np.isfinite(out)):
        raise ValueError("model produced non-finite rain rates")
    # The DGMR lesson: check the physical range before anything scores this.
    if np.nanmax(out) > 500.0:
        raise ValueError(
            f"model produced {np.nanmax(out):.0f} mm/h - physically impossible; "
            "the openclimatefix DGMR checkpoint failed exactly this way")
    return out


def _selfcheck():
    # --- transform round-trip: HANDOVER.md landmine #1 is a silent transform ---
    rain = np.array([0.0, 0.01, 0.1, 1.0, 5.0, 20.0, 100.0], dtype=np.float32)
    back = from_model(to_model(rain))
    assert np.allclose(back, rain, rtol=1e-4, atol=1e-4), (rain, back)
    assert abs(to_model(np.array([100.0]))[0] - 1.0) < 1e-6
    assert to_model(np.array([0.0]))[0] == 0.0
    assert from_model(np.array([-5.0]))[0] == 0.0          # no negative rain
    # and the upper clamp: an unbounded output layer CAN emit y > 1, and expm1
    # turns 1.67 into 2163 mm/h. Measured on the first trained checkpoint.
    assert from_model(np.array([1.67]))[0] <= 100.01, from_model(np.array([1.67]))
    assert from_model(np.array([50.0]))[0] <= 100.01
    assert to_model(np.array([np.nan]))[0] == 0.0          # NaN -> dry
    # monotone, so ordering of rain rates survives the transform
    t = to_model(rain)
    assert np.all(np.diff(t) > 0)

    # --- dataset windowing ---
    stacks = [np.abs(np.random.default_rng(0).gamma(1, 2, size=(30, 64, 64)))]
    ds = RadarSequences(stacks, crop=(32, 32))
    assert len(ds) == 30 - (N_INPUT + N_OUTPUT) + 1 == 7, len(ds)
    x, y = ds[0]
    assert x.shape == (N_INPUT, 32, 32) and y.shape == (N_OUTPUT, 32, 32)
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() < 2.0
    # windows must be consecutive: the target starts at exactly the frame after
    # the last input frame. Checked uncropped, so the crop cannot mask an
    # off-by-one in the time axis.
    full = RadarSequences(stacks)
    xf, yf = full[0]
    # atol=1e-3: storage is float16 (~3 decimal digits over [0, 1]), which is
    # the whole point of the memory fix. Tighter than this would be asserting
    # precision the store deliberately does not keep.
    assert np.allclose(xf[-1], to_model(stacks[0][N_INPUT - 1]), atol=1e-3)
    assert np.allclose(yf[0], to_model(stacks[0][N_INPUT]), atol=1e-3)
    assert np.allclose(full[3][0][0], to_model(stacks[0][3]), atol=1e-3)
    # a too-short stack yields nothing rather than raising
    assert len(RadarSequences([np.zeros((5, 32, 32))])) == 0
    # the memory fix itself: half-precision cropped store, float32 out, and no
    # retained copy of the raw mm/h stack (43 GB -> 5.6 GB on the CIRRUS set).
    assert full.model_units[0].dtype == np.float16
    assert ds.model_units[0].shape[1:] == (32, 32)      # cropped at load
    assert not hasattr(ds, "stacks")
    # stride drops near-duplicate windows proportionally
    assert len(RadarSequences(stacks, stride=3)) == 3, len(RadarSequences(stacks, stride=3))
    # a generator source is consumed exactly once and still indexes correctly
    gen = RadarSequences(iter(stacks))
    assert len(gen) == len(full) and np.allclose(gen[0][0], xf, atol=1e-3)

    # --- inference refuses to invent weights ---
    try:
        forecast(np.zeros((12, 64, 64)), 12, weights="/nonexistent.pt")
        raise AssertionError("should have refused")
    except (FileNotFoundError, ImportError):
        pass

    try:
        import torch
    except ImportError:
        print("smaat selfcheck OK (transforms + data; torch not installed, "
              "architecture unverified)")
        return

    # --- architecture ---
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(2, N_INPUT, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, N_OUTPUT, 64, 64), y.shape
    # non-square and larger inputs must also survive the skip connections
    with torch.no_grad():
        assert model(torch.zeros(1, N_INPUT, 96, 64)).shape == (1, N_OUTPUT, 96, 64)
    # the paper's headline: ~4.1M against a plain U-Net's 17.4M
    assert 3.5e6 < n_params < 5.5e6, n_params
    print(f"smaat selfcheck OK ({n_params / 1e6:.2f}M parameters, "
          f"paper reports 4.1M for the 12->1 configuration)")


def overfit_check(steps=200, lr=3e-3, device=None):
    """Can the training loop memorise a trivially learnable case?

    The standard pre-training sanity test, and the cheapest bug-catcher there
    is: if the loop cannot overfit ONE drifting blob, gradients are not
    reaching the weights and no amount of real data will help. Five seconds on
    a GPU against hours wasted on a training run that was never going to work.
    """
    import torch

    ny = nx = 64
    yy, xx = np.mgrid[0:ny, 0:nx]
    stack = [np.stack([8.0 * np.exp(-(((xx - (8 + 2.0 * t)) ** 2
                                       + (yy - 32) ** 2) / (2 * 5.0 ** 2)))
                       for t in range(40)])]
    ds = RadarSequences(stack, crop=(64, 64))
    x = torch.from_numpy(np.stack([ds[i][0] for i in range(len(ds))]))
    y = torch.from_numpy(np.stack([ds[i][1] for i in range(len(ds))]))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = _criterion("mae")
    x, y = x.to(device), y.to(device)

    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    final = float(loss)
    assert final < first / 10, (
        f"loop cannot overfit a trivial case ({first:.4f} -> {final:.4f}); "
        "gradients are not reaching the weights")
    print(f"overfit check PASSED: loss {first:.5f} -> {final:.5f} "
          f"({first / final:.0f}x) over {len(ds)} windows")
    return first, final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--overfit", action="store_true",
                    help="sanity-test the training loop on a learnable case")
    ap.add_argument("--train", action="store_true",
                    help="train on the harvested OPERA archive")
    ap.add_argument("--evaluate", action="store_true",
                    help="score the saved checkpoint in mm/h on the val days")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--loss", default="mae", choices=sorted(LOSSES))
    ap.add_argument("--stride", type=int, default=1,
                    help="step between window starts; >1 drops near-duplicate "
                         "windows and shortens the epoch proportionally")
    ap.add_argument("--min-samples", type=int, default=10000,
                    help="refuse to train below this many windows; lower it "
                         "only deliberately")
    args = ap.parse_args()

    if args.overfit:
        overfit_check()
        raise SystemExit

    if args.train:
        from nowcast.archive import load_stacks

        from nowcast.archive import split_days

        from nowcast.archive import cfg

        tr_days, va_days = split_days()
        # Generators: load_stacks yields one day at a time so the raw float32
        # never accumulates. Do NOT count windows here - that would consume them.
        stacks = load_stacks(days=set(tr_days))
        val_stacks = load_stacks(days=set(va_days))
        print("Split is BY DAY: consecutive windows share 23 of 24 frames, so a "
              "random\nwindow split would leak validation into training almost "
              "entirely.")
        c = cfg()
        span = int(N_INPUT * c["cadence"])
        print(f"NOTE: training on {c['cadence']}-min {c['km']:g} km OPERA "
              f"({N_INPUT} in / {N_OUTPUT} out = {span} min -> {span} min).\n"
              "The frozen KNMI event set is a different product; numbers from "
              "this run\nare not comparable to it without saying so.")
        path = train(stacks, val_stacks=val_stacks, epochs=args.epochs,
                     batch_size=args.batch_size, loss=args.loss,
                     min_samples=args.min_samples, stride=args.stride)
        print(f"weights -> {path}")
        raise SystemExit

    if args.evaluate:
        from nowcast.archive import cfg, load_stacks, split_days

        _, va_days = split_days()
        c = cfg()
        print(f"scoring {WEIGHTS} on {len(va_days)} validation days "
              f"({c['cadence']:g}-min {c['km']:g} km {c['era'] if 'era' in c else ''})".rstrip())
        print("These days were never trained on: split_days hashes the day "
              "itself, so\nthe assignment is fixed the moment a day exists.")
        res = evaluate_checkpoint(load_stacks(days=set(va_days)),
                                  batch_size=args.batch_size, stride=args.stride)
        report_eval(res, cadence_min=c["cadence"])
        raise SystemExit

    if args.selfcheck:
        _selfcheck()
        raise SystemExit

    if args.summary:
        import torch

        model = build_model()
        total = sum(p.numel() for p in model.parameters())
        print(f"SmaAt-UNet  {N_INPUT} frames in -> {N_OUTPUT} frames out")
        print(f"parameters: {total:,} ({total / 1e6:.2f}M); "
              f"paper reports 4.1M vs plain U-Net 17.4M")
        print(f"fp32 weights: {total * 4 / 1e6:.0f} MB")
        x = torch.zeros(1, N_INPUT, 384, 384)
        with torch.no_grad():
            print(f"384x384 forward pass -> {tuple(model(x).shape)}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device: {torch.cuda.get_device_name(0)}")
