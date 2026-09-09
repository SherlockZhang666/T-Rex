"""Would `--action_dim N` resume cleanly from this checkpoint, or silently discard
the pretrained action head?

`scripts/train.py:834-841` filters the resume state dict by shape and prints one
line:

    Skipped 123 keys with shape mismatch (e.g. x_embedder.mlp.fc1.weight)

and then carries on. Every skipped key keeps whatever `initialize_vla_weights()`
put there, so a wrong `--action_dim` throws away the midtrain action head while
the run looks completely healthy. This script answers the question before a GPU
is allocated, by reading the checkpoint's own tensor shapes.

    python tools/check_ckpt_action_dims.py \
        --ckpt ckpt/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6 --action_dim 62

Exit status is the number of parameters that would be skipped.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# Modules whose shapes are tied to action_dim, and which side of the weight the
# action dimension lands on. See Qwen3VLVLAModel.__init__ (modeling_vla.py:101-104,
# :147): ActionEmbedder projects action_dim -> hidden, FinalLayer hidden -> action_dim.
ACTION_DIM_KEYS = {
    "x_embedder": "in",           # ActionEmbedder(action_dim, H)
    "state_embedder": "in",       # ActionEmbedder(action_dim, H), only if use_robot_state
    "final_layer": "out",         # FinalLayer(H, action_dim)
    "final_layer_tactile": "out",
}
# Tactile buffers are fixed at 10 fingers x 6 channels regardless of action_dim
# (encode_tactile_f6_history does n_hands = NF // 5).
TACTILE_BUFFERS = {
    "tacf6_vqvae_min": 60,
    "tacf6_vqvae_max": 60,
    "tacf6_vqvae_mask": 60,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint dir or model.pt")
    p.add_argument("--action_dim", type=int, default=62)
    args = p.parse_args()

    path = args.ckpt
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    if not os.path.exists(path):
        sys.exit(f"missing {path}")

    print(f"reading {path} ({os.path.getsize(path) / 1e9:.1f} GB) ...")
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"  {len(sd)} tensors\n")

    print(f"parameters tied to action_dim (checking against --action_dim {args.action_dim}):")
    mismatched = []
    seen = 0
    for k, v in sd.items():
        root = k.split(".")[0]
        if root not in ACTION_DIM_KEYS or v.ndim < 1:
            continue
        side = ACTION_DIM_KEYS[root]
        # Both modules wrap a two-layer Mlp (diffusion.py:46-76): ActionEmbedder is
        # fc1(action_dim -> hidden) + fc2(hidden -> hidden), FinalLayer is
        # fc1(hidden -> hidden) + fc2(hidden -> action_dim). Only one layer in
        # each touches the action dim; the other is hidden x hidden, and
        # hidden_size is 2048 on Qwen3-VL-2B, so the width tells them apart.
        dim = v.shape[-1] if side == "in" else v.shape[0]
        if dim > 512:
            continue
        seen += 1
        ok = dim == args.action_dim
        if not ok:
            mismatched.append((k, tuple(v.shape), dim))
        print(f"  [{'ok  ' if ok else 'SKIP'}] {k:44s} {str(tuple(v.shape)):18s} action_dim={dim}")

    print(f"\ntactile buffers (must be 60 = 10 fingers x 6, independent of action_dim):")
    tac_bad = []
    for k, want in TACTILE_BUFFERS.items():
        if k not in sd:
            print(f"  [miss] {k} not in checkpoint")
            continue
        got = int(sd[k].shape[0])
        ok = got == want
        if not ok:
            tac_bad.append(k)
        print(f"  [{'ok  ' if ok else 'BAD '}] {k:24s} {got} (want {want})")

    print()
    if seen == 0:
        print("no action-dim-tied parameters found -- is this the right checkpoint?")
        return 1
    if mismatched:
        print(f"{len(mismatched)} parameter(s) WOULD BE SKIPPED at --action_dim {args.action_dim}.")
        print("Each one keeps its freshly initialised weights instead of the checkpoint's.")
        print("Fix the --action_dim (or the data layout) before training.")
    else:
        print(f"All {seen} action-dim-tied parameters match --action_dim {args.action_dim}.")
        print("train.py will print 'Skipped 0 keys with shape mismatch' for these.")
    if tac_bad:
        print(f"Tactile buffers are the wrong width: {tac_bad}")
    return len(mismatched) + len(tac_bad)


if __name__ == "__main__":
    sys.exit(main())
