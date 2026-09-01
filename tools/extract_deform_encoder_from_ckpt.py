"""Pull the authors' deform encoder out of a released checkpoint's model.pt.

The released midtrain checkpoint carries all 104 `deform_encoder.*` tensors
inside `model.pt`, so the weights the paper used were shipped after all — just
not as the standalone `sharpa_wave_deform_encoder.pth` that `scripts/train.sh`
points at.

Post-train does not strictly need this file: `train.py:801` loads
`--deform_encoder_ckpt` but `train.py:842` then loads the resume checkpoint over
it, so the embedded copy wins either way.  Writing it out anyway keeps
`DEFORM_ENCODER_PATH` pointing at something real, so the run does not print
"Warning: DeformEncoder checkpoint not found" — a warning worth keeping
meaningful, since it is the only signal that a from-scratch run is about to
train against a frozen, randomly-initialised encoder.

Inference needs nothing from this script: `scripts/test.py` has no deform
encoder argument at all and reads the weights straight out of `model.pt`.
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir holding model.pt, or the model.pt itself")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.ckpt
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)

    prefix = "deform_encoder."
    enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not enc:
        raise SystemExit(f"no {prefix}* tensors in {path} — was it trained with "
                         f"use_tactile_deform=1?")

    # store under `encoder.*` so load_deform_encoder_weights strips it the same
    # way it strips a DeformAEInfer checkpoint (modeling_vla.py:293-295)
    out_sd = {f"encoder.{k}": v for k, v in enc.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(out_sd, args.out)

    tracked = enc.get("stem.1.num_batches_tracked")
    print(f"extracted {len(enc)} tensors -> {args.out}")
    if tracked is not None:
        print(f"stem BN num_batches_tracked = {int(tracked)} "
              f"({'trained' if int(tracked) > 0 else 'NOT trained — suspicious'})")
    print("verify with:  python tools/verify_deform_encoder.py --ckpt " + args.out)


if __name__ == "__main__":
    main()
