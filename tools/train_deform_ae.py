"""Stage 2 — self-supervised training of the tactile deformation encoder.

Reproduces the "pre-trained within a self-supervised convolutional autoencoder
framework and subsequently frozen" step from the paper, using the architecture
already shipped in `qwen_vla/DeformAE.py` (ResNet-18 stem modified to one input
channel, first three residual stages, 3x3 re-projections to 128 channels).

Input scale is [0, 1] — this matters.  Both VLA data paths feed the encoder
[0, 1] tensors (`scripts/train.py:277` divides by 255; `lerobot_dataset.py:217`
takes channel 0 of a float [0,1] video tensor) and `modeling_vla.py:508` passes
them to the encoder unscaled.  The `__main__` demo inside `DeformAE.py` feeds
raw 0-255 instead; do not copy it.

The saved state_dict uses `encoder.*` / `decoder.*` keys, which is exactly what
`Qwen3VLVLAModel.load_deform_encoder_weights` strips and loads.  Only `encoder.*`
is ever read back, so the decoder is ours to shape freely — it exists solely to
provide the reconstruction gradient and is discarded afterwards.

Three guards, all earned from a run that collapsed:

* The stock decoder's plain ReLU died on the first epoch (100% of its inputs
  went negative), leaving the final conv with only its bias — a constant output,
  zero gradient to the encoder, and a loss frozen to six decimals for 29 epochs.
  `--decoder_act leaky` removes that failure mode outright.
* Warmup + gradient clipping stop the first-epoch gradient spike that killed it.
* R^2 against the data variance, and an explicit collapse check on the output's
  spatial std, make a constant predictor impossible to mistake for convergence.
  PSNR alone hid it: a constant image still scored 24.3 dB.

Validation holds out whole shards, not random frames: neighbouring frames of a
30 fps video are near-duplicates, so a random split would leak and report a
meaninglessly low val loss.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_vla.DeformAE import DeformAEInfer  # noqa: E402


class ShardFrames(Dataset):
    """Frames drawn from a list of uint8 [n,240,240] .npy shards."""

    def __init__(self, shards):
        self.shards = shards
        self._arrs = None
        lens = [len(np.load(p, mmap_mode="r")) for p in shards]
        self.index = np.array(
            [(s, i) for s, n in enumerate(lens) for i in range(n)], dtype=np.int64
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        if self._arrs is None:  # lazy per-worker mmap
            self._arrs = [np.load(p, mmap_mode="r") for p in self.shards]
        s, i = self.index[k]
        img = np.asarray(self._arrs[s][i], dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)  # [1,240,240] in [0,1]


def split_shards(shards, val_frac):
    """Hold out whole files, spread across all 10 fingers."""
    by_key = {}
    for p in shards:
        by_key.setdefault(os.path.basename(p).split("__")[0], []).append(p)
    train, val = [], []
    for key in sorted(by_key):
        files = sorted(by_key[key])
        n_val = max(1, int(round(len(files) * val_frac)))
        val += files[-n_val:]
        train += files[:-n_val]
    return train, val


def set_decoder_act(model, kind):
    """Swap the decoder's activation. Discarded at load time, so this is free."""
    if kind == "relu":
        return
    act = {"leaky": nn.LeakyReLU(0.01, inplace=True), "gelu": nn.GELU()}[kind]
    head = model.decoder.deformation_head
    for i, m in enumerate(head):
        if isinstance(m, nn.ReLU):
            head[i] = act
            return
    raise RuntimeError("no ReLU found in decoder.deformation_head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--out", required=True, help="path for the .pth to write")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--decoder_act", choices=["leaky", "gelu", "relu"], default="leaky",
                    help="'relu' is the stock DeformAE decoder, which died on the "
                         "first epoch; leaky/gelu cannot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--contact_thresh", type=float, default=0.12,
                    help="pixels above this count as contact (data mean ~0.031, std ~0.069)")
    ap.add_argument("--mag_alpha", type=float, default=10.0,
                    help="magnitude-weighted MSE: w = 1 + alpha*relu(x - baseline). "
                         "0 = plain MSE; contact pixels are a tiny fraction of the "
                         "frame, so plain MSE rewards predicting the background")
    ap.add_argument("--mag_baseline", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("WARNING: no GPU visible — this will be very slow", flush=True)

    shards = sorted(
        os.path.join(args.shard_dir, f)
        for f in os.listdir(args.shard_dir)
        if f.endswith(".npy") and not f.endswith(".tmp.npy")
    )
    if not shards:
        raise SystemExit(f"no shards in {args.shard_dir} — run extract_deform_frames.py first")
    tr_sh, va_sh = split_shards(shards, args.val_frac)
    if not tr_sh or not va_sh:
        raise SystemExit(f"split left train={len(tr_sh)} val={len(va_sh)} files; "
                         f"need >=2 shards per finger (lower --val_frac or re-extract)")
    tr, va = ShardFrames(tr_sh), ShardFrames(va_sh)
    print(f"{len(shards)} shards -> train {len(tr)} frames / {len(tr_sh)} files, "
          f"val {len(va)} frames / {len(va_sh)} files", flush=True)

    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(va, batch_size=args.batch_size, shuffle=False, **dl_kw)

    model = DeformAEInfer().to(dev)
    set_decoder_act(model, args.decoder_act)
    model.to(dev)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"encoder {n_enc/1e6:.2f}M params, decoder {n_dec/1e6:.2f}M params "
          f"(decoder act={args.decoder_act})", flush=True)

    with torch.no_grad():  # shape contract: encoder must emit [B,128,15,15] = 28800
        feat = model.encoder(torch.zeros(1, 1, 240, 240, device=dev))
    assert tuple(feat.shape[1:]) == (128, 15, 15), \
        f"encoder emits {tuple(feat.shape[1:])}, but modeling_vla.py expects 128x15x15"
    print(f"encoder output {tuple(feat.shape)} -> flattened {feat[0].numel()} "
          f"(deform_proj expects 28800) OK", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(tr_dl))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")

    best, flat_epochs = math.inf, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x in tr_dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast(**amp):
                y = model(x)
            y = y.float()
            err = (y - x) ** 2
            if args.mag_alpha > 0:
                w = 1.0 + args.mag_alpha * (x - args.mag_baseline).clamp_min(0.0)
                loss = (err * w).sum() / w.sum()
            else:
                loss = err.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            run += loss.item() * x.size(0)
            seen += x.size(0)

        # ── validation: MSE, R^2 vs the data variance, contact-only MSE, output std ──
        model.eval()
        n = sx = sxx = se = 0.0
        c_se = c_n = 0.0
        out_std = 0.0
        with torch.no_grad():
            for x in va_dl:
                x = x.to(dev, non_blocking=True)
                with torch.autocast(**amp):
                    y = model(x)
                y = y.float()
                e = (y - x) ** 2
                b = x.numel()
                n += b
                sx += x.sum().item()
                sxx += (x * x).sum().item()
                se += e.sum().item()
                m = x > args.contact_thresh
                if m.any():
                    c_se += e[m].sum().item()
                    c_n += m.sum().item()
                out_std += y.std(dim=(2, 3)).mean().item() * x.size(0)

        mse = se / n
        var = sxx / n - (sx / n) ** 2
        r2 = 1.0 - mse / max(var, 1e-12)
        c_mse = c_se / max(c_n, 1.0)
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        c_psnr = 10.0 * math.log10(1.0 / max(c_mse, 1e-12))
        out_std /= len(va)

        print(f"ep {ep:3d}/{args.epochs}  train {run/seen:.6f}  val_mse {mse:.6f}  "
              f"R2 {r2:6.3f}  PSNR {psnr:5.2f}  contact-PSNR {c_psnr:5.2f}  "
              f"out_std {out_std:.5f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.0f}s", flush=True)

        # collapse guard: a constant image has zero spatial std and R^2 <= 0
        if out_std < 1e-4:
            flat_epochs += 1
            if flat_epochs >= 2:
                raise SystemExit(
                    "\nCOLLAPSED: the decoder emits a constant image (out_std ~ 0), so no "
                    "gradient reaches the encoder. Retry with --decoder_act leaky, a lower "
                    "--lr, or a longer --warmup_steps.")
        else:
            flat_epochs = 0

        if mse < best:
            best = mse
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)  # encoder.* / decoder.* keys
            print(f"    saved {args.out} (val_mse {best:.6f}, R2 {r2:.3f})", flush=True)

    print(f"\nbest val_mse {best:.6f} -> {args.out}")
    print("verify with:  python tools/verify_deform_encoder.py --ckpt " + args.out)


if __name__ == "__main__":
    main()
