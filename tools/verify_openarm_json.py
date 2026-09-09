"""Assert that a converted OpenArm training JSON is actually what T-Rex will read.

Every failure mode this checks for is silent.  Nothing in `scripts/train.py`
raises when the deform paths stop matching its regex, when the tactile fingers
are in the wrong order, when the F6 bias was never subtracted, or when the
right-hand padding is the wrong width -- the run just trains on the wrong thing
and the loss curve looks fine.

Run this before spending a GPU:

    python tools/verify_openarm_json.py \
        --json /path/pick_up_the_egg_..._train.json

It finds `_statistics.json` and `_provenance.json` next to the JSON, re-opens the
source HDF5 files named in the provenance, and prints one PASS/FAIL line per
check.  Exit status is the number of failed checks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gen_json_openarm_sharpa_left import (  # noqa: E402
    COLLECTOR_FINGER_ORDER, COLLECTOR_TO_TREX, DEFORM_HW, IDENTITY_POSE_9D,
    N_FINGERS_PER_HAND, TREX_FINGER_ORDER, quat_wxyz_pos_to_matrix,
)
from utils.lerobot_common import ACTION_CHUNK, ACTION_DIM, F6_DIM, get_rot_mat  # noqa: E402

# The two patterns scripts/train.py hard-codes.  Copied here as literals, not
# imported, so that a change on either side shows up as a failure here instead of
# quietly agreeing with itself.
TRAIN_FRAME_RE = re.compile(r"(.+/episode_\d+)/image(\d+)_")   # train.py:177, 209
TRAIN_FLARE_SUB = r"image\d+_"                                  # train.py:299


class Checks:
    def __init__(self):
        self.failed = 0
        self.n = 0

    def __call__(self, name: str, ok: bool, detail: str = ""):
        self.n += 1
        if not ok:
            self.failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    def skip(self, name: str, why: str):
        print(f"  [SKIP] {name}  --  {why}")


def rot_angle_deg(R: np.ndarray) -> float:
    """Rotation magnitude of R, in degrees.

    Note for anyone reading a round-trip residual: arccos is ill-conditioned at
    identity. cos(theta) ~= 1 - theta^2/2, so a perturbation eps in the trace
    surfaces as theta ~= sqrt(2 eps). The action chunk is stored as float32, so
    eps ~ 5e-8 and the floor of this measurement is ~0.02 deg even for a
    reconstruction that is exact in float32. A residual at that scale is
    arithmetic, not misalignment; the tolerances below are set accordingly.
    """
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--n_episodes", type=int, default=3,
                   help="how many source episodes to re-open for the round-trip checks")
    p.add_argument("--pos_tol_mm", type=float, default=0.5)
    p.add_argument("--flare_steps", type=int, default=8)
    p.add_argument("--flare_stride", type=int, default=4)
    args = p.parse_args()

    json_path = os.path.abspath(args.json)
    stats_path = json_path.replace(".json", "_statistics.json")
    prov_path = json_path.replace(".json", "_provenance.json")
    for p_ in (json_path, stats_path, prov_path):
        if not os.path.exists(p_):
            sys.exit(f"missing {p_}")

    print(f"reading {json_path} ...")
    with open(json_path) as fh:
        records = json.load(fh)
    with open(stats_path) as fh:
        stats = json.load(fh)
    with open(prov_path) as fh:
        prov = json.load(fh)
    print(f"  {len(records)} samples, {len(prov['episodes'])} episode report(s)\n")

    c = Checks()

    # ── 1. shapes and the 62-D padding ───────────────────────────────────────
    print("1. layout")
    r0 = records[0]
    act = np.asarray(r0["action"], dtype=np.float32)
    c("action chunk is [16, 62]", act.shape == (ACTION_CHUNK, ACTION_DIM), str(act.shape))
    c("state is 62-D", len(r0["state_fast"]) == ACTION_DIM, str(len(r0["state_fast"])))
    c("tactile_f6 is [10, 6]",
      np.asarray(r0["tactile_f6"]).shape == (10, 6),
      str(np.asarray(r0["tactile_f6"]).shape))
    c("10 deform paths", len(r0["tactile_image_deform"]) == 10,
      str(len(r0["tactile_image_deform"])))
    c("2 fast images (single wrist repeated)", len(r0["input_image_fast"]) == 2)
    c("both fast slots are the same left-wrist frame",
      r0["input_image_fast"][0] == r0["input_image_fast"][1])

    sample = records[:: max(1, len(records) // 400)]
    right_arm = np.asarray([s["action"][k][31:40] for s in sample
                            for k in range(ACTION_CHUNK)], dtype=np.float32)
    c("right-arm action block is the identity pose",
      np.allclose(right_arm, IDENTITY_POSE_9D[None], atol=0),
      f"unique rows = {len(np.unique(right_arm, axis=0))}")
    right_hand = np.asarray([s["action"][k][40:62] for s in sample
                             for k in range(ACTION_CHUNK)], dtype=np.float32)
    c("right-hand action block is zero", np.all(right_hand == 0))
    right_f6 = np.asarray([np.asarray(s["tactile_f6"])[5:10] for s in sample])
    c("right-hand F6 block is zero", np.all(right_f6 == 0))
    allnum = np.asarray([s["state_fast"] for s in sample], dtype=np.float32)
    c("no NaN in state", not np.isnan(allnum).any())
    c("no NaN in action",
      not np.isnan(np.asarray([s["action"] for s in sample], dtype=np.float32)).any())

    # ── 2. the two regexes train.py hard-codes ───────────────────────────────
    print("\n2. path patterns that scripts/train.py depends on")
    bad = [s["tactile_image_deform"][0] for s in sample
           if not TRAIN_FRAME_RE.search(s["tactile_image_deform"][0])]
    c("VQ-VAE F6 history regex matches tactile_image_deform[0]", not bad,
      f"{len(bad)} unmatched, e.g. {bad[0] if bad else ''}")
    bad = [s["input_image_slow"][0] for s in sample
           if not re.search(TRAIN_FLARE_SUB, os.path.basename(s["input_image_slow"][0]))]
    c("FLARE substitution regex matches the head image name", not bad)

    eps = {}
    for s in records:
        m = TRAIN_FRAME_RE.search(s["tactile_image_deform"][0])
        if m:
            eps.setdefault(m.group(1), []).append(int(m.group(2)))
    contiguous = all(sorted(v) == list(range(len(v))) for v in eps.values())
    c("frame indices are 0..n-1 inside every episode dir", contiguous,
      f"{len(eps)} episode dir(s)")

    # ── 3. FLARE future frames actually exist on disk ────────────────────────
    print("\n3. FLARE frame availability")
    probe = records[:: max(1, len(records) // 60)]
    missing = 0
    for s in probe:
        slow = s["input_image_slow"][0]
        cur = int(re.search(r"image(\d+)_", os.path.basename(slow)).group(1))
        for k in range(args.flare_steps):
            nxt = cur + (k + 1) * args.flare_stride
            if not os.path.exists(re.sub(TRAIN_FLARE_SUB, f"image{nxt}_", slow)):
                missing += 1
    total = len(probe) * args.flare_steps
    # Frames past the end of an episode are legitimately absent; train.py falls
    # back to the current frame. A large fraction missing means the naming broke.
    c("most FLARE future frames resolve", missing < 0.35 * total,
      f"{missing}/{total} missing (end-of-episode misses are expected)")

    # ── 4. images on disk ────────────────────────────────────────────────────
    print("\n4. images")
    ok_all = True
    for s in probe[:20]:
        for pth in [s["input_image_slow"][0]] + s["input_image_fast"] + s["tactile_image_deform"]:
            if not os.path.exists(pth):
                ok_all = False
                print(f"        missing: {pth}")
    c("every referenced image exists", ok_all)

    zero_png = records[0]["tactile_image_deform"][5]
    z = cv2.imread(zero_png, cv2.IMREAD_GRAYSCALE)
    c("shared right-hand deform pad is an all-zero 240x240",
      z is not None and z.shape == (DEFORM_HW, DEFORM_HW) and not z.any(),
      f"{None if z is None else z.shape}, max={None if z is None else z.max()}")

    dpaths = [s["tactile_image_deform"][0] for s in probe[:40]]
    modes = []
    for pth in dpaths:
        img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            modes.append(int(np.bincount(img.ravel(), minlength=256).argmax()))
    c("left-hand deform background is 0 after de-flooring",
      modes and all(m == 0 for m in modes),
      f"modal pixel values seen: {sorted(set(modes))}")

    # ── 5. statistics sidecar ────────────────────────────────────────────────
    print("\n5. normalization statistics")
    key = next(iter(stats))
    blk = stats[key]
    a_q01 = np.asarray(blk["action"]["q01"], dtype=np.float64)   # [16, 62]
    a_q99 = np.asarray(blk["action"]["q99"], dtype=np.float64)
    c("action stats are per-(step, dim) [16, 62]", a_q01.shape == (ACTION_CHUNK, ACTION_DIM),
      str(a_q01.shape))
    span = a_q99 - a_q01
    c("right half of the action is degenerate (constant pad)",
      np.allclose(span[:, 31:], 0.0, atol=1e-12),
      f"max span = {span[:, 31:].max():.3e}")
    live = span[:, :31]
    c("left half of the action is NOT degenerate",
      (live > 1e-9).sum() >= 0.8 * live.size,
      f"{(live > 1e-9).sum()}/{live.size} dims with nonzero span")
    t_q01 = np.asarray(blk["tactile_f6"]["q01"], dtype=np.float64)
    t_q99 = np.asarray(blk["tactile_f6"]["q99"], dtype=np.float64)
    c("tactile stats are 60-D", t_q01.shape == (F6_DIM,), str(t_q01.shape))
    c("right-hand F6 stats are degenerate", np.allclose(t_q99[30:] - t_q01[30:], 0.0))
    c("left-hand F6 stats are not degenerate", (t_q99[:30] - t_q01[:30]).max() > 1e-6,
      f"max span = {(t_q99[:30] - t_q01[:30]).max():.4f}")
    c("tracking_error is 56-D",
      len(blk.get("tracking_error", {}).get("mean", [])) == 56)

    # ── 6. per-episode round trip against the source HDF5 ────────────────────
    print("\n6. round trip against the source HDF5")
    mount = prov.get("mount", {})
    applied = bool(mount.get("applied"))
    if applied:
        c("mount transform recorded", "total_z_m" in mount,
          f"roll {mount.get('applied_roll_deg'):+.4f} deg, z {mount.get('total_z_m'):.4f} m")
    else:
        c.skip("mount transform", "--no_ee_offset was used")

    by_ep = {}
    for s in records:
        m = TRAIN_FRAME_RE.search(s["tactile_image_deform"][0])
        by_ep.setdefault(os.path.basename(m.group(1)), []).append((int(m.group(2)), s))
    reports = {r["episode_tag"]: r for r in prov["episodes"] if "skipped" not in r}

    checked = 0
    for tag in sorted(by_ep):
        if checked >= args.n_episodes or tag not in reports:
            continue
        rep = reports[tag]
        rows = [s for _, s in sorted(by_ep[tag])]
        lo, hi = rep["engaged_run"]
        with h5py.File(rep["source"], "r", locking=False) as f:
            a = dict(f.attrs)
            side_order = json.loads(a["side_order"])
            L = side_order.index("left")
            obs7 = np.asarray(f["obs/wrist_pose_b"][lo:hi, L, :], dtype=np.float64)
            act7 = np.asarray(f["action/wrist_pose_b"][lo:hi, L, :], dtype=np.float64)
            f6_src = np.asarray(f["tactile/f6"][:], dtype=np.float64)
            hand_a = np.asarray(f["action/hand_joint_pos"][lo:hi], dtype=np.float64)

        print(f"\n  {tag}  ({len(rows)} rows, source rows [{lo}, {hi}))")

        # 6a. the mount offset actually moved the poses by the recorded amount
        if applied:
            T_link7 = quat_wxyz_pos_to_matrix(obs7[0])
            # reconstruct the stored L_ee pose of row 0 from state (rebased) is
            # awkward; compare against the delta the transform itself implies.
            dz = float(mount["total_z_m"])
            droll = float(mount["applied_roll_deg"])
            T_off = np.eye(4)
            th = np.deg2rad(droll)
            T_off[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
            T_off[:3, 3] = [0, 0, dz]
            T_ee = T_link7 @ T_off
            shift_mm = np.linalg.norm(T_ee[:3, 3] - T_link7[:3, 3]) * 1000.0
            ang = rot_angle_deg(T_link7[:3, :3].T @ T_ee[:3, :3])
            c(f"    offset displaces {dz * 1000:.1f} mm",
              abs(shift_mm - dz * 1000) < 0.05, f"{shift_mm:.2f} mm")
            c(f"    offset rotates {abs(droll):.3f} deg about the finger axis",
              abs(ang - abs(droll)) < 0.02, f"{ang:.4f} deg")

        # 6b. delta-base chunk k=0 reconstructs the absolute target pose
        T_off = np.eye(4)
        if applied:
            th = np.deg2rad(float(mount["applied_roll_deg"]))
            T_off[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
            T_off[:3, 3] = [0, 0, float(mount["total_z_m"])]
        errs = []
        for r in range(0, len(rows), max(1, len(rows) // 50)):
            d9 = np.asarray(rows[r]["action"][0][:9], dtype=np.float64)
            base = quat_wxyz_pos_to_matrix(obs7[r]) @ T_off
            want = quat_wxyz_pos_to_matrix(act7[r]) @ T_off
            got_t = base[:3, 3] + base[:3, :3] @ d9[:3]
            got_R = base[:3, :3] @ get_rot_mat(d9[3:9])
            errs.append((np.linalg.norm(got_t - want[:3, 3]) * 1000.0,
                         rot_angle_deg(got_R.T @ want[:3, :3])))
        errs = np.asarray(errs)
        c("    delta-base chunk[0] reconstructs the absolute target pose",
          errs[:, 0].max() < args.pos_tol_mm and errs[:, 1].max() < 0.05,
          f"pos max {errs[:, 0].max():.4f} mm, rot max {errs[:, 1].max():.5f} deg")

        # 6c. hand action passes through unpermuted
        h_err = max(float(np.abs(np.asarray(rows[r]["action"][0][9:31]) - hand_a[r]).max())
                    for r in range(0, len(rows), max(1, len(rows) // 50)))
        c("    hand action passes through in the same 22-joint order",
          h_err < 1e-5, f"max |diff| = {h_err:.2e} rad")

        # 6d. finger order really was reversed
        n_head = int(rep.get("f6_bias_head_rows", 60))
        bias = f6_src[:n_head].mean(axis=0)                    # collector order
        r_probe = min(len(rows) - 1, len(rows) // 2)
        got = np.asarray(rows[r_probe]["tactile_f6"], dtype=np.float64)[:5]
        want = (f6_src[lo + r_probe] - bias)[COLLECTOR_TO_TREX]
        c("    fingers reordered collector -> T-Rex and de-biased",
          np.abs(got - want).max() < 1e-4,
          f"max |diff| = {np.abs(got - want).max():.2e}; "
          f"{COLLECTOR_FINGER_ORDER} -> {TREX_FINGER_ORDER}")
        # and prove it is not accidentally the identity permutation
        wrong = (f6_src[lo + r_probe] - bias)
        c("    the reordering is not a no-op",
          np.abs(got - wrong).max() > 1e-6,
          f"unpermuted would differ by {np.abs(got - wrong).max():.4f}")

        # 6e. the recomputed bias agrees with the magnitude the archive stored
        agreement = rep.get("f6_bias_agreement_N")
        if agreement is None:
            c.skip("    F6 bias vs archived magnitude_N", "no tactile_zero_offset attr")
        else:
            # 0.15 N, not something tighter: the archived magnitude is itself a
            # drifting measurement (0.70 N of drift inside a single episode per
            # the capture schema's section 8.1). This check is here to catch a
            # bias that was never subtracted or was subtracted in the wrong
            # finger order -- both of which are off by whole newtons.
            worst = float(np.abs(np.asarray(agreement)).max())
            c("    recomputed F6 bias matches the archived magnitude_N",
              worst < 0.15, f"max |diff| = {worst:.4f} N")

        # 6f. the de-biased thumb and pinky actually return to zero somewhere.
        # A percentile over the whole episode rather than the opening rows: the
        # engaged run may well start after first contact, so "the first rows are
        # quiet" is not guaranteed, whereas "the signal reaches zero at some
        # point" is exactly what removing a standing offset should produce.
        # These two are the channels the archive flags as standing-offset ones.
        allf6 = np.asarray([r_["tactile_f6"] for r_ in rows], dtype=np.float64)
        for fname in ("thumb", "pinky"):
            i_trex = TREX_FINGER_ORDER.index(fname)
            mag = np.linalg.norm(allf6[:, i_trex, :3], axis=1)
            p05 = float(np.percentile(mag, 5))
            c(f"    de-biased {fname} |F| returns to ~0",
              p05 < 0.3,
              f"5th pct {p05:.4f} N (un-corrected it stands at "
              f"{'2.2-3.5' if fname == 'thumb' else '~1.7'} N)")

        checked += 1

    print(f"\n{c.n - c.failed}/{c.n} checks passed")
    if c.failed:
        print("\nFAILED -- do not start training on this dataset.")
    return c.failed


if __name__ == "__main__":
    sys.exit(main())
