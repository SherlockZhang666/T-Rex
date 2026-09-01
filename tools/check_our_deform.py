"""Diagnose our own Sharpa-hand deform maps before trusting T-Rex's frozen encoder.

The encoder in `qwen_vla/DeformAE.py` carries 17 BatchNorm layers whose
running_mean/var were estimated on T-Rex data (4,226,018 batches).  Those
buffers are distribution statistics, so they only stay valid if our deform maps
look like T-Rex's.  This script measures whether they do, in the order the
answers actually depend on each other:

  1. FORMAT   — 240x240 uint8, deform_valid, no all-placeholder rows.
                Nothing below means anything if this fails.
  2. BASELINE — T-Rex's background is *exactly 0* (82% of pixels).  Ours sits on
                a constant floor (2 on ep_0000).  Detected as the modal pixel
                value rather than hard-coded, because it is a property of the
                capture pipeline and may change.  Everything downstream is
                measured after subtracting it.
  3. ACTIVITY — a finger that never touched anything contributes only constant
                frames.  Feeding those to a BN recalibration teaches it "mostly
                no contact", so they must be counted, not silently averaged in.
  4. HOLD     — deform can be held between refreshes while f6/seq keep ticking
                at 30 Hz.  Consecutive-identical frames measure the *effective*
                refresh rate, which is what bounds the independent-sample count.
  5. FEATURES — the actual question: run the frozen encoder and compare its
                dead-unit fraction and activation std against the T-Rex
                reference.  Close => use it frozen as-is.  Far => recalibrate BN.

The verdict is deliberately conservative: it reports "recalibrate" only on the
feature-level check, since pixel-level differences that the encoder absorbs are
not worth acting on.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Measured on T-Rex deform frames with the released encoder (eval mode).
# Regenerate with --trex_shards if the reference set changes.
TREX_REF = {"pixel_zero_frac": 0.822, "pixel_hot_frac": 0.0142, "pixel_std": 16.4,
            "feat_dead": 0.745, "feat_std": 0.309}

HOT = 30          # pixel value above the floor that counts as real contact
FEAT_TOL = 0.08   # |dead - ref| tolerance before we call BN stats stale
STD_RATIO = 2.0   # feat_std ratio outside [1/r, r] is a scale mismatch
MIN_CONTACT = 30  # fewer contact frames than this and the encoder pass mostly sees
                  # constant input, so its feature stats say nothing about BN drift


def load_episode(path):
    with h5py.File(path, "r", locking=False) as f:
        a = dict(f.attrs)
        order = json.loads(a.get("tactile_finger_order", '["f0","f1","f2","f3","f4"]'))
        return (f["tactile/deform"][:], f["tactile/deform_valid"][:],
                f["tactile/f6"][:], f["tactile/t_capture"][:], order, a)


def detect_floor(d):
    """Modal pixel value — the background the sensor idles at."""
    return int(np.bincount(d[::max(1, len(d) // 50)].ravel(), minlength=256).argmax())


def trex_reference(shard_dir, encoder, n_per_finger=20):
    """Recompute the T-Rex reference from extracted shards (optional)."""
    by = {}
    for p in sorted(glob.glob(os.path.join(shard_dir, "*.npy"))):
        if p.endswith(".tmp.npy"):
            continue
        by.setdefault(os.path.basename(p).split("__")[0], []).append(p)
    out = []
    for k in sorted(by):
        a = np.load(sorted(by[k])[-1], mmap_mode="r")
        out.append(np.asarray(a[np.linspace(0, len(a) - 1, n_per_finger).astype(int)]))
    T = np.concatenate(out)
    ref = {"pixel_zero_frac": float(np.mean(T == 0)),
           "pixel_hot_frac": float(np.mean(T > HOT)),
           "pixel_std": float(T.astype(np.float32).std())}
    if encoder is not None:
        z = encode(encoder, T[::2])
        ref["feat_dead"], ref["feat_std"] = z["dead"], z["std"]
    return ref


def build_encoder(ckpt):
    import torch
    from qwen_vla.DeformAE import DeformEncoder
    m = DeformEncoder()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k[len("encoder."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} encoder tensors missing from {ckpt}")
    return m.eval()


def encode(model, arr, bs=8):
    """Feature stats of the frozen encoder on uint8 frames (scaled to [0,1])."""
    import torch
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(1)
    with torch.no_grad():
        z = torch.cat([model(x[i:i + bs]).flatten(1) for i in range(0, len(x), bs)])
    return {"dead": float((z == 0).float().mean()), "std": float(z.std()),
            "mean": float(z.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, nargs="+",
                    help="ep_*.hdf5 files (or dirs holding one)")
    ap.add_argument("--encoder", default="", help="deform encoder .pth; enables step 5")
    ap.add_argument("--trex_shards", default="",
                    help="deform_frames dir; recomputes the T-Rex reference instead "
                         "of using the stored constants")
    ap.add_argument("--floor", type=int, default=-1,
                    help="background pixel value to subtract (-1 = auto-detect)")
    ap.add_argument("--feat_stride", type=int, default=20,
                    help="subsample frames before the (slow) encoder pass")
    args = ap.parse_args()

    paths = []
    for p in args.episodes:
        paths += sorted(glob.glob(os.path.join(p, "*.hdf5"))) if os.path.isdir(p) else [p]

    enc = build_encoder(args.encoder) if args.encoder else None
    ref = dict(TREX_REF)
    if args.trex_shards:
        print("recomputing T-Rex reference from shards ...", flush=True)
        ref.update(trex_reference(args.trex_shards, enc))
    print(f"T-Rex reference: {ref}\n")

    verdicts = []
    for path in paths:
        print("=" * 78)
        print(os.path.basename(path))
        print("=" * 78)
        d, dv, f6, tc, order, attrs = load_episode(path)
        n, nf = d.shape[0], d.shape[1]

        # ── 1. format ───────────────────────────────────────────────────────
        ok_shape = d.shape[2:] == (240, 240) and d.dtype == np.uint8
        print(f"[1] FORMAT   deform {d.shape} {d.dtype}  "
              f"{'OK' if ok_shape else 'FAIL — encoder wants 240x240 uint8'}")
        print(f"             deform_valid {dv.mean():.3f}   task={attrs.get('task','?')!r}  "
              f"frames={n}  fingers={order}")
        if not ok_shape:
            verdicts.append((path, "FAIL: bad shape")); continue

        # ── 2. baseline ─────────────────────────────────────────────────────
        floor = detect_floor(d) if args.floor < 0 else args.floor
        sub = np.clip(d.astype(np.int16) - floor, 0, 255).astype(np.uint8)
        zf_raw, zf_sub = float(np.mean(d == 0)), float(np.mean(sub == 0))
        print(f"\n[2] BASELINE floor={floor} ({'auto' if args.floor < 0 else 'given'})   "
              f"frac==0  raw {zf_raw:.4f} -> minus-floor {zf_sub:.4f}   "
              f"(T-Rex {ref['pixel_zero_frac']:.3f})")
        if floor != 0:
            print(f"             ACTION: subtract {floor} in your dataloader before /255 — "
                  f"T-Rex's background is exactly 0")

        # ── 3. activity ─────────────────────────────────────────────────────
        print(f"\n[3] ACTIVITY (after floor removal, contact = px > {HOT})")
        print(f"    {'finger':8s} {'uniq':>5s} {'max':>4s} {'contact frames':>15s} "
              f"{'active px p99':>14s} {'f6 |F| max':>11s}")
        live, ncontact = [], []
        for i in range(nf):
            x = sub[:, i].reshape(n, -1)
            act = (x > HOT).sum(1)
            fmax = float(np.linalg.norm(f6[:, i, :3], axis=1).max())
            ncon = int((act > 0).sum())
            live.append(ncon > 0)
            ncontact.append(ncon)
            print(f"    {order[i]:8s} {len(np.unique(sub[:, i])):5d} {int(sub[:, i].max()):4d} "
                  f"{ncon:8d}/{n:<6d} {np.percentile(act, 99):14.0f} {fmax:10.3f}N")
        print(f"    -> {sum(live)}/{nf} fingers carry any contact this episode")

        # ── 4. hold / effective refresh rate ────────────────────────────────
        print(f"\n[4] HOLD     f6 sampling dt median = "
              f"{np.median(np.diff(tc[:, 0])) * 1000:.1f} ms")
        for i in range(nf):
            x = sub[:, i].reshape(n, -1)
            dup = float((x[1:] == x[:-1]).all(1).mean())
            act = (x > HOT).sum(1) > 0
            xc = x[act]
            dupc = float((xc[1:] == xc[:-1]).all(1).mean()) if len(xc) > 2 else float("nan")
            eff = (1 - dup) * (1000 / max(np.median(np.diff(tc[:, i])) * 1000, 1e-9))
            note = "constant finger" if not live[i] else f"~{eff:.1f} Hz effective"
            print(f"    {order[i]:8s} identical-consecutive {dup:.3f}  "
                  f"(contact frames only {dupc:.3f})   {note}")

        # ── 5. encoder features ─────────────────────────────────────────────
        if enc is None:
            print("\n[5] FEATURES skipped (pass --encoder to run it)")
            verdicts.append((path, "pixel checks only")); continue
        print(f"\n[5] FEATURES frozen encoder, every {args.feat_stride}th frame, "
              f"floor removed  (T-Rex dead={ref['feat_dead']:.3f} std={ref['feat_std']:.3f})")
        judged, flagged = 0, []
        for i in range(nf):
            if not live[i]:
                print(f"    {order[i]:8s} skipped — constant input, feature is a constant")
                continue
            z = encode(enc, sub[::args.feat_stride, i])
            dd = z["dead"] - ref["feat_dead"]
            rr = z["std"] / max(ref["feat_std"], 1e-9)
            # A finger with almost no contact is mostly constant frames; its low
            # feature std reflects the sampling, not BN drift, so it cannot vote.
            thin = ncontact[i] < MIN_CONTACT
            bad = abs(dd) > FEAT_TOL or not (1 / STD_RATIO < rr < STD_RATIO)
            if not thin:
                judged += 1
                if bad:
                    flagged.append(order[i])
            tag = (f"inconclusive — only {ncontact[i]} contact frames "
                   f"(<{MIN_CONTACT}), excluded from the verdict" if thin
                   else ("<-- MISMATCH" if bad else "ok"))
            print(f"    {order[i]:8s} dead={z['dead']:.3f} (d={dd:+.3f})  "
                  f"std={z['std']:.4f} (x{rr:.2f})  mean={z['mean']:.3f}   {tag}")
        if judged == 0:
            v = "INCONCLUSIVE — no finger had enough contact to judge"
        elif flagged:
            v = f"BN stats look stale — recalibrate (flagged: {', '.join(flagged)})"
        else:
            v = f"BN stats transfer — use the encoder frozen as-is ({judged} finger(s) judged)"
        print(f"\n    VERDICT: {v}")
        verdicts.append((path, v))

    print("\n" + "=" * 78)
    for p, v in verdicts:
        print(f"{os.path.basename(p):30s} {v}")


if __name__ == "__main__":
    main()
