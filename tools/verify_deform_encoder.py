"""Stage 3 — check a trained checkpoint actually loads into the VLA.

`Qwen3VLVLAModel.load_deform_encoder_weights` fails silently: a missing file or
mismatched keys only prints a warning, and `scripts/train.py:859-862` then
freezes the encoder at whatever it holds.  A randomly-initialised frozen encoder
trains without error and without complaint, so verify before launching SFT.

This replays that exact loading logic (modeling_vla.py:290-298) and fails loudly
if any encoder parameter would be left uninitialised.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_vla.DeformAE import DeformEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"FAIL: {args.ckpt} does not exist — this is exactly the case "
                         f"modeling_vla.py:286-288 swallows with a warning")

    enc = DeformEncoder()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # verbatim from modeling_vla.py:292-298
    encoder_sd = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_sd[k[len("encoder."):]] = v
        elif k in enc.state_dict():
            encoder_sd[k] = v
    missing, unexpected = enc.load_state_dict(encoder_sd, strict=False)

    print(f"checkpoint keys      : {len(state_dict)}")
    print(f"matched into encoder : {len(encoder_sd)}")
    print(f"missing              : {len(missing)}")
    print(f"unexpected           : {len(unexpected)}")
    if missing:
        print("  e.g. " + ", ".join(list(missing)[:5]))

    with torch.no_grad():
        feat = enc(torch.zeros(1, 1, 240, 240))
    print(f"encoder output       : {tuple(feat.shape)} -> {feat[0].numel()} flat "
          f"(deform_proj expects 28800)")

    ok = not missing and feat[0].numel() == 28800
    print("\n" + ("PASS — set DEFORM_ENCODER_PATH to this file" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
