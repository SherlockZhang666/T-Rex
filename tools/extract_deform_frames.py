"""Stage 1 — sample grayscale deform frames from the released T-Rex videos.

The 10 deform streams hold ~55 M frames total (11 files x ~503 k frames x 10
streams), far more than a ~3 M-parameter autoencoder needs.  Sequential decode
runs at ~8 k fps while seeking runs at ~44 fps, so we decode straight through
and keep every `--stride`-th frame.

Each input file becomes one uint8 shard of shape [n, 240, 240]; stage 2 mmaps
the shards, so no merge step is needed.

Frames are taken as the R channel of the decoded BGR frame, matching how the
VLA reads them (`lerobot_dataset.py:217` takes channel 0 of the [3,H,W] RGB
tensor).  The videos are true grayscale (B == G == R), so the choice is moot.
"""

import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np


def find_videos(dataset_root: str) -> list[str]:
    vid_root = os.path.join(dataset_root, "videos")
    out = []
    for key in sorted(os.listdir(vid_root)):
        if "_deform_" not in key:
            continue
        for chunk in sorted(os.listdir(os.path.join(vid_root, key))):
            d = os.path.join(vid_root, key, chunk)
            if not os.path.isdir(d):
                continue
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".mp4")]
    return sorted(out)


def shard_name(path: str) -> str:
    # .../videos/observation.images.tactile_left_deform_index/chunk-000/file-000.mp4
    parts = path.split(os.sep)
    key = parts[-3].replace("observation.images.", "")
    return f"{key}__{parts[-2]}__{os.path.splitext(parts[-1])[0]}.npy"


def extract_one(job):
    path, stride, out_dir = job
    out_path = os.path.join(out_dir, shard_name(path))
    if os.path.exists(out_path):
        return out_path, len(np.load(out_path, mmap_mode="r")), 0

    cap = cv2.VideoCapture(path)
    kept, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            kept.append(frame[:, :, 2])
        i += 1
    cap.release()

    arr = np.stack(kept) if kept else np.zeros((0, 240, 240), np.uint8)
    tmp = out_path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, out_path)
    return out_path, len(arr), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="trex_dataset root (the dir holding videos/ meta/ data/)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=500,
                    help="keep 1 frame every STRIDE (500 -> ~110 k frames total)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = find_videos(args.dataset_root)
    if not videos:
        raise SystemExit(f"no *_deform_* videos under {args.dataset_root}/videos")
    print(f"{len(videos)} deform videos, stride={args.stride}, workers={args.workers}", flush=True)

    jobs = [(v, args.stride, args.out_dir) for v in videos]
    total_kept = total_seen = 0
    with Pool(args.workers) as pool:
        for n, (out_path, kept, seen) in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            total_kept += kept
            total_seen += seen
            print(f"[{n}/{len(videos)}] {os.path.basename(out_path)}  kept={kept}", flush=True)

    print(f"\ndecoded {total_seen} frames, kept {total_kept} "
          f"({total_kept * 240 * 240 / 1e9:.1f} GB uint8) in {args.out_dir}")


if __name__ == "__main__":
    main()
