"""OpenArm `rt-v1` episodes (single LEFT Sharpa Wave hand) -> T-Rex post-train JSON.

This is the OpenArm counterpart of `gen_json_tac_deltabase_eef_bimanual_parallel.py`.
It reproduces that script's action semantics byte-for-byte -- delta-base arm +
absolute target hand, FRAME_STRIDE=1, ACTION_CHUNK=16, 62-D bimanual layout --
while reading a completely different capture schema and a rig that has only one
hand.

Read `docs/superpowers/specs/2026-09-09-openarm-sharpa-left-finetune-design.md`
for why each decision below is what it is.  The four that matter most:

1. The 62-D layout is KEPT and the right half is constant-padded.  `train.py:834`
   silently drops shape-mismatched keys when resuming, so shrinking action_dim to
   31 would re-initialise the midtrain action head with nothing in the log but a
   single "Skipped N keys" line.

2. Poses are pushed from `openarm_{side}_link7` (what the collector archives) out
   to the Sharpa `hand_wrist` link (what T-Rex calls `L_ee`).  The arm's *base*
   frame cancels in the delta-base action, but the *EEF* frame does not -- it
   appears conjugated in R_delta -- so this transform is mandatory, not cosmetic.

3. Tactile fingers are re-ordered.  The collector measured [pinky, ring, middle,
   index, thumb]; T-Rex uses [thumb, index, middle, ring, pinky].

4. The F6 zero offset is subtracted.  The embedded VQ-VAE normalises with the
   min/max buffers baked into the checkpoint (`modeling_vla.py:137-145`), NOT with
   the stats this script writes, so a standing bias lands in a different region of
   the codebook.

Output layout (the filenames are load-bearing -- see `_FRAME_RE` below):

    <img_save_root>/<task_name>/episode_<NNNN>/image<f>_head.png
                                              /image<f>_wrist_left.png
                                              /image<f>_tactile_left_deform_<0..4>.png
    <img_save_root>/<task_name>/_zero_deform_240.png     (shared right-hand pad)
    <json_save_root>/<json_name_base>.json
    <json_save_root>/<json_name_base>_statistics.json
    <json_save_root>/<json_name_base>_provenance.json

Usage:
  python utils/gen_json_openarm_sharpa_left.py \
      --data_root /n/netscratch/.../tactile-steering-data/pick_up_the_egg \
      --img_save_root /n/netscratch/.../training_data/egg_images \
      --json_save_root /n/netscratch/.../training_data/egg_json \
      --task_name pick_up_the_egg \
      --json_name_base pick_up_the_egg_deltabase_eef_left_stride1_train \
      --instruction "Pick up the egg." \
      --mount_geom /n/netscratch/.../openarm-sharpa-sim/openarm_sharpa/assets/mount_geom.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lerobot_common import (  # noqa: E402
    ACTION_CHUNK, ACTION_DIM, FRAME_STRIDE, F6_DIM, TRACKING_ERROR_DIM,
    compute_bimanual_tracking_error, compute_chunk_delta_pose, pose_matrix_to_9d,
)

# ── constants ────────────────────────────────────────────────────────────────

# `scripts/train.py` hard-codes these two patterns.  If the emitted paths stop
# matching, the VQ-VAE history window silently degrades to "repeat the current
# frame 16 times" and FLARE silently loses every future frame.  Neither raises,
# so `tools/verify_openarm_json.py` asserts against these exact objects.
_FRAME_RE = re.compile(r"(.+/episode_\d+)/image(\d+)_")      # train.py:177, 209
_FLARE_SUB_RE = re.compile(r"image\d+_")                     # train.py:299

N_FINGERS_PER_HAND = 5
F6_PER_FINGER = 6
DEFORM_HW = 240

# The collector measured this order by per-fingertip poke (2026-08-20); T-Rex's
# is thumb-first.  `COLLECTOR_TO_TREX[i]` is the collector index that supplies
# T-Rex finger `i`.
COLLECTOR_FINGER_ORDER = ["pinky", "ring", "middle", "index", "thumb"]
TREX_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
COLLECTOR_TO_TREX = [COLLECTOR_FINGER_ORDER.index(n) for n in TREX_FINGER_ORDER]

# Right half of the 62-D vector.  The 9-D arm block is the IDENTITY pose, not
# zeros: all-zero is not a valid rotation and makes every replay/visualisation
# tool draw garbage.  Both normalise to a constant -1 anyway (q99 == q01).
IDENTITY_POSE_9D = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.], dtype=np.float32)
RIGHT_PAD_31D = np.concatenate([IDENTITY_POSE_9D, np.zeros(22, dtype=np.float32)])

# base +X=LEFT, +Y=FORWARD, +Z=DOWN  ->  REP-103 +X=FORWARD, +Y=LEFT, +Z=UP.
# Affects `state` ONLY (the absolute pose); the delta-base action is invariant
# under any change of base frame, see the design doc's proof.  det = +1.
REP103_FROM_CAPTURE = np.array([[0., 1., 0.],
                                [1., 0., 0.],
                                [0., 0., -1.]], dtype=np.float64)


# ── small helpers ────────────────────────────────────────────────────────────

def quat_wxyz_pos_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz] -> 4x4.  The collector writes W FIRST."""
    x, y, z, qw, qx, qy, qz = [float(v) for v in pose7]
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T[:3, 3] = [x, y, z]
    return T


def load_mount_transform(mount_geom_path: str, side: str) -> tuple[np.ndarray, dict]:
    """T_link7 -> Sharpa `hand_wrist` (== T-Rex's `L_ee`), from mount_geom.py.

    Deliberately has no identity fallback: a silent identity would emit link7-frame
    data that looks completely normal downstream while every trajectory carries a
    54.5 mm offset and a 144.175 deg roll.  Callers who really want that must pass
    --no_ee_offset.
    """
    if not os.path.exists(mount_geom_path):
        raise FileNotFoundError(
            f"mount_geom.py not found at {mount_geom_path}.\n"
            "This module is the only source of truth for the link7 -> hand_wrist "
            "transform. Point --mount_geom at it, or pass --no_ee_offset if you "
            "deliberately want link7-frame poses.")
    spec = importlib.util.spec_from_file_location("_mount_geom", mount_geom_path)
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)

    # + for LEFT, - for RIGHT; a roll about the finger axis (flange +Z).
    sign = 1.0 if side == "left" else -1.0
    roll_deg = sign * float(mg.PALM_ROLL_DEG)
    # HAND_MOUNT_Z reaches the adapter face; T-Rex's L_ee is one more fixed joint
    # out, at hand_wrist (left_sharpa_wave_with_flange.urdf:35-41).
    z = float(mg.HAND_MOUNT_Z) + float(mg.FLANGE_TO_WRIST_Z)

    c, s = np.cos(np.deg2rad(roll_deg)), np.sin(np.deg2rad(roll_deg))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    T[:3, 3] = [0., 0., z]

    prov = {
        "mount_geom_path": os.path.abspath(mount_geom_path),
        "side": side,
        "PALM_ROLL_DEG": float(mg.PALM_ROLL_DEG),
        "applied_roll_deg": roll_deg,
        "LINK7_TOOL_FACE_Z": float(mg.LINK7_TOOL_FACE_Z),
        "ARM_PLATE_THICKNESS": float(mg.ARM_PLATE_THICKNESS),
        "HAND_MOUNT_Z": float(mg.HAND_MOUNT_Z),
        "FLANGE_TO_WRIST_Z": float(mg.FLANGE_TO_WRIST_Z),
        "total_z_m": z,
        "note": "T_link7->L_ee, where T-Rex's L_ee is the Sharpa hand_wrist link "
                "(robot_descriptions.py:66-69 loads the with_wrist MJCF and "
                "appends it at L_ee with an identity SE3).",
    }
    return T, prov


def hold_forward(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid rows with the most recent valid one (leading rows use the
    first valid row).  Returns a copy; `arr` is untouched."""
    out = np.array(arr, copy=True)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return out
    if not valid.any():
        raise ValueError("no valid rows at all")
    idx = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), 0))
    first = int(np.argmax(valid))
    idx[:first] = first
    return out[idx]


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """[lo, hi) of the longest contiguous True run; (0, 0) if there is none.

    Samples are emitted only from a single contiguous engaged run so that no
    action chunk straddles a disengagement (during which the operator is walking
    the arm into place and the motion says nothing about the task).
    """
    mask = np.asarray(mask, dtype=bool)
    best = (0, 0)
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) > (best[1] - best[0]):
            best = (i, j)
        i = j
    return best


def deform_floor(frames: np.ndarray) -> int:
    """Modal pixel value = the capture pipeline's constant background floor.

    T-Rex's deform background is exactly 0 (82 % of pixels); ours sits on a
    constant floor -- 2 on the episodes measured by tools/check_our_deform.py.
    Detected rather than hard-coded because it is a property of the pipeline and
    may change between capture sessions.
    """
    sample = frames[:: max(1, len(frames) // 40)]
    counts = np.bincount(np.asarray(sample, dtype=np.uint8).ravel(), minlength=256)
    return int(counts.argmax())


def f6_zero_offset(f6: np.ndarray, attrs: dict, n_head_default: int = 60):
    """Per-(finger, channel) resting bias, measured from this episode's own head.

    The archive stores only the resting force *magnitude* per finger
    (`tactile_zero_offset["magnitude_N"]`), not the 3-vector, so the vector is
    recomputed here over the same rows the collector used -- the frames before
    first contact.  The stored magnitude then serves as an independent check
    (returned as `agreement_N`, asserted by tools/verify_openarm_json.py).

    A session-wide constant would be wrong (2.77 -> 2.93 N of drift across one
    session) and so would a first/last average (0.70 N of drift inside one
    episode), which is why this is per-episode and head-only.
    """
    raw = attrs.get("tactile_zero_offset")
    meta = {}
    n_head = n_head_default
    if raw:
        try:
            meta = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
            n_head = int(meta.get("first_contact_row", meta.get("n_frames", n_head_default)))
        except (ValueError, TypeError):
            meta = {}
    n_head = int(np.clip(n_head, 1, len(f6)))
    bias = f6[:n_head].mean(axis=0)                       # [5, 6], collector order

    stored = meta.get("magnitude_N")
    agreement = None
    if stored is not None and len(stored) == f6.shape[1]:
        measured = np.linalg.norm(bias[:, :3], axis=1)
        agreement = (measured - np.asarray(stored, dtype=np.float64)).tolist()
    return bias.astype(np.float32), n_head, agreement


def write_shared_zero_deform(path: str) -> str:
    """One all-zero 240x240 PNG shared by every right-hand deform slot.

    Only `tactile_image_deform[0]` is regex-matched by train.py; the other nine
    entries are merely opened.  The right hand does not exist on this rig, so all
    five of its maps are the same constant image -- pointing them at one file
    instead of writing ~79 k identical PNGs.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, np.zeros((DEFORM_HW, DEFORM_HW), dtype=np.uint8))
    return path


def stream_video_frames(video_path: str, wanted: dict, out_dir: str, suffix: str,
                        size: tuple | None, crop_box: tuple | None) -> int:
    """Decode `video_path` once, writing every frame the rows asked for.

    `wanted` maps video-frame-index -> list of row indices that want that frame,
    so one sequential pass covers both the head camera (identity mapping) and the
    free-running wrist camera (many rows may share a frame).  Sequential decode
    runs ~200x faster than seeking, and holding nothing but the current frame
    keeps a 1739-frame 960x540 wrist stream at O(1) memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"cannot open {video_path}")
    written = 0
    fi = 0
    remaining = dict(wanted)
    while remaining:
        ok, frame = cap.read()
        if not ok:
            break
        rows = remaining.pop(fi, None)
        if rows is not None:
            if crop_box is not None:
                y0, y1, x0, x1 = crop_box
                frame = frame[y0:y1, x0:x1]
            if size is not None:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            for r in rows:
                cv2.imwrite(os.path.join(out_dir, f"image{r}_{suffix}.png"), frame)
                written += 1
        fi += 1
    cap.release()
    if remaining:
        raise IOError(f"{video_path}: ran out of frames, {len(remaining)} row(s) unfilled "
                      f"(first missing video frame {min(remaining)})")
    return written


# ── per-episode work ─────────────────────────────────────────────────────────

def process_episode(job: dict) -> tuple[str, list, dict]:
    """Convert one episode.  Returns (episode_tag, records, report)."""
    h5_path = job["h5_path"]
    ep_tag = job["episode_tag"]                      # 'episode_0007'
    out_dir = os.path.join(job["img_save_root"], job["task_name"], ep_tag)
    os.makedirs(out_dir, exist_ok=True)

    report = {"episode_tag": ep_tag, "source": h5_path}

    with h5py.File(h5_path, "r", locking=False) as f:
        a = dict(f.attrs)

        if bool(a.get("discarded", False)):
            return ep_tag, [], {**report, "skipped": "discarded"}
        if not bool(a.get("complete", True)):
            return ep_tag, [], {**report, "skipped": "incomplete"}
        if "tactile/f6" not in f:
            return ep_tag, [], {**report, "skipped": "no tactile/f6"}
        if bool(a.get("tactile_synthetic", False)):
            return ep_tag, [], {**report, "skipped": "tactile_synthetic"}

        side_order = json.loads(a["side_order"]) if isinstance(a.get("side_order"), (str, bytes)) \
            else list(a.get("side_order", ["right", "left"]))
        side_order = [s.decode() if isinstance(s, bytes) else str(s) for s in side_order]
        L = side_order.index("left")

        engaged = np.asarray(f["teleop/engaged"][:, L], dtype=bool)
        lo, hi = longest_true_run(engaged)
        if (hi - lo) < job["min_frames"]:
            return ep_tag, [], {**report, "skipped": f"engaged run {hi - lo} < {job['min_frames']}"}

        obs_pose = np.asarray(f["obs/wrist_pose_b"][:, L, :], dtype=np.float64)
        act_pose = np.asarray(f["action/wrist_pose_b"][:, L, :], dtype=np.float64)
        obs_pose = hold_forward(obs_pose, f["obs/valid"][:])
        act_pose = hold_forward(act_pose, np.asarray(f["action/valid"][:, L], dtype=bool))

        hand_state = hold_forward(np.asarray(f["hand/joint_pos"][:], dtype=np.float64),
                                  f["hand/valid"][:])
        hand_act = hold_forward(np.asarray(f["action/hand_joint_pos"][:], dtype=np.float64),
                                f["action/hand_valid"][:])

        f6_raw = np.asarray(f["tactile/f6"][:], dtype=np.float64)          # [N, 5, 6]
        deform_valid = np.asarray(f["tactile/deform_valid"][:], dtype=bool)  # [N, 5]
        wrist_fi = np.asarray(f["cam_wrist/frame_index"][:], dtype=np.int64)
        # Row i is built from scene frame time/frame_index[i]; read it rather
        # than assuming the identity mapping the schema currently guarantees.
        head_fi = np.asarray(f["time/frame_index"][:], dtype=np.int64)

        bias, n_head, agreement = f6_zero_offset(f6_raw, a)
        report.update({"f6_bias_head_rows": n_head, "f6_bias_agreement_N": agreement})

        # deform: hold invalid (frame, finger) maps forward, then de-floor.
        deform = np.asarray(f["tactile/deform"][lo:hi], dtype=np.uint8)     # [n, 5, H, W]
        dv = deform_valid[lo:hi]
        for fg in range(N_FINGERS_PER_HAND):
            if not dv[:, fg].all():
                deform[:, fg] = hold_forward(deform[:, fg], dv[:, fg])
        floor = deform_floor(deform)
        deform = np.clip(deform.astype(np.int16) - floor, 0, 255).astype(np.uint8)
        report["deform_floor"] = floor

    n = hi - lo
    # NaN must never reach the output.  The right arm's columns are all-NaN by
    # design; the left columns of everything the policy actually regresses must
    # not be, so a NaN there kills the episode rather than being papered over.
    for name, arr in (("obs_pose", obs_pose[lo:hi]), ("act_pose", act_pose[lo:hi]),
                      ("hand_act", hand_act[lo:hi]), ("f6", f6_raw[lo:hi])):
        if np.isnan(arr).any():
            return ep_tag, [], {**report, "skipped": f"NaN in {name} on the engaged run"}

    # `hand/joint_pos` is the exception: archives repaired for the one-index
    # shift (schema section 8.2) carry a permanent NaN in column 0, which is not
    # recoverable. It only feeds `state`, and `state` does not enter the network
    # while --use_robot_state is 0, so zero-fill it and record that we did.
    hand_state_nans = int(np.isnan(hand_state[lo:hi]).sum())
    if hand_state_nans:
        hand_state = np.nan_to_num(hand_state, nan=0.0)
        report["hand_state_nans_zeroed"] = hand_state_nans
        report["hand_state_offset_fix"] = bool(a.get("hand_state_offset_fix", False))

    T_off = job["ee_offset"]                       # 4x4, or None
    s_pose, a_pose = [], []
    for i in range(lo, hi):
        Ts = quat_wxyz_pos_to_matrix(obs_pose[i])
        Ta = quat_wxyz_pos_to_matrix(act_pose[i])
        if T_off is not None:
            Ts = Ts @ T_off
            Ta = Ta @ T_off
        s_pose.append(Ts)
        a_pose.append(Ta)
    s_pose = np.stack(s_pose)                      # [n,4,4] base -> L_ee
    a_pose = np.stack(a_pose)

    # The absolute 9-D poses (`state` and `absolute_target_action`) are the only
    # quantities a base-frame change touches, so they are the only ones rebased
    # to REP-103. The delta-base action is invariant -- see the design doc's
    # proof. Both must use the SAME frame: `compute_bimanual_tracking_error`
    # is frame-invariant only when its two arguments agree.
    if job["rep103_state"]:
        M = np.eye(4)
        M[:3, :3] = REP103_FROM_CAPTURE
        s_pose_abs, a_pose_abs = M[None] @ s_pose, M[None] @ a_pose
    else:
        s_pose_abs, a_pose_abs = s_pose, a_pose
    s_arm_9d = pose_matrix_to_9d(s_pose_abs)       # [n,9]
    a_arm_9d = pose_matrix_to_9d(a_pose_abs)

    # tactile, reordered to T-Rex's thumb-first convention and de-biased
    f6 = (f6_raw[lo:hi] - bias[None])[:, COLLECTOR_TO_TREX, :]     # [n,5,6]
    deform = deform[:, COLLECTOR_TO_TREX]                          # [n,5,H,W]
    hand_s = hand_state[lo:hi]
    hand_a = hand_act[lo:hi]

    # ── images ──
    head_wanted: dict[int, list] = {}
    for i in range(lo, hi):
        head_wanted.setdefault(int(head_fi[i]), []).append(i - lo)
    stream_video_frames(job["head_video"], head_wanted, out_dir, "head",
                        job["head_size"], job["crop_box"])

    # The wrist camera free-runs at 60 fps off the row clock, so a row names its
    # frame through cam_wrist/frame_index. -1 means no wrist frame had arrived
    # yet, which can only happen at the very start; those rows borrow the first
    # frame that did arrive.
    run_wf = wrist_fi[lo:hi]
    if not (run_wf >= 0).any():
        return ep_tag, [], {**report, "skipped": "no wrist frame on the engaged run"}
    fallback_wf = int(run_wf[run_wf >= 0][0])
    wrist_wanted: dict[int, list] = {}
    for i in range(lo, hi):
        wf = int(wrist_fi[i])
        wrist_wanted.setdefault(fallback_wf if wf < 0 else wf, []).append(i - lo)
    stream_video_frames(job["wrist_video"], wrist_wanted, out_dir, "wrist_left",
                        job["wrist_size"], None)

    zero_png = job["zero_deform_path"]
    for r in range(n):
        for fg in range(N_FINGERS_PER_HAND):
            cv2.imwrite(os.path.join(out_dir, f"image{r}_tactile_left_deform_{fg}.png"),
                        deform[r, fg])

    # ── records ──
    chunk = int(job["action_chunk"])
    stride = int(job["frame_stride"])
    records = []
    for r in range(n):
        head_p = os.path.join(out_dir, f"image{r}_head.png")
        wrist_p = os.path.join(out_dir, f"image{r}_wrist_left.png")

        deform_paths = [os.path.join(out_dir, f"image{r}_tactile_left_deform_{fg}.png")
                        for fg in range(N_FINGERS_PER_HAND)]
        deform_paths += [zero_png] * N_FINGERS_PER_HAND

        base = s_pose[r]
        action_chunk = []
        for k in range(chunk):
            fut = min(min(r + k * stride, n - 1) + stride - 1, n - 1)
            d9 = compute_chunk_delta_pose(base, a_pose[fut])
            action_chunk.append(np.concatenate(
                [d9, hand_a[fut], RIGHT_PAD_31D]).astype(np.float32).tolist())

        state62 = np.concatenate([s_arm_9d[r], hand_s[r], RIGHT_PAD_31D]).astype(np.float32)
        tgt = min(r + stride - 1, n - 1)
        abs62 = np.concatenate([a_arm_9d[tgt], hand_a[tgt], RIGHT_PAD_31D]).astype(np.float32)

        f6_60 = np.concatenate(
            [f6[r], np.zeros((N_FINGERS_PER_HAND, F6_PER_FINGER))], axis=0)  # [10,6]

        records.append({
            "episode_id": ep_tag,
            "image_old_slow": head_p,
            # Both fast slots carry the same left-wrist frame.  This rig has one
            # wrist camera; repeating the path preserves midtrain's two-image
            # layout (and the "second fast image is the left wrist" positional
            # association) at the cost of one extra string, not a duplicated video.
            "image_old_fast": [wrist_p, wrist_p],
            "action": action_chunk,
            "absolute_target_action": abs62.tolist(),
            "state": state62.tolist(),
            "tactile_f6": f6_60.tolist(),
            "tactile_image_deform": deform_paths,
            "language_instruction": job["instruction"],
        })

    report.update({"frames": n, "engaged_run": [int(lo), int(hi)],
                   "source_frames": int(len(engaged))})
    return ep_tag, records, report


# ── driver ───────────────────────────────────────────────────────────────────

def list_episodes(data_root: str) -> list[dict]:
    """Every non-discarded episode under <data_root>/<session>/ep_*/."""
    out = []
    for sess in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(sess):
            continue
        index_path = os.path.join(sess, "index.jsonl")
        keep = None
        if os.path.exists(index_path):
            keep = {}
            with open(index_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    keep[rec["path"]] = rec
        for ep_dir in sorted(glob.glob(os.path.join(sess, "ep_*"))):
            if not os.path.isdir(ep_dir):
                continue
            name = os.path.basename(ep_dir)
            h5 = os.path.join(ep_dir, f"{name}.hdf5")
            if not os.path.exists(h5):
                continue
            rel = f"{name}/{name}.hdf5"
            rec = (keep or {}).get(rel)
            if rec is not None and (rec.get("discarded") or not rec.get("success", True)):
                continue
            out.append({
                "h5_path": h5,
                "session": os.path.basename(sess),
                "ep_name": name,
                "head_video": os.path.join(ep_dir, f"{name}_scene.mkv"),
                "wrist_video": os.path.join(ep_dir, f"{name}_wrist.mkv"),
                "index_record": rec,
            })
    return out


def calculate_stats(data: np.ndarray, mask) -> dict:
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
        "mask": mask,
    }


def write_stats(all_records: list, out_path: str, dataset_name: str) -> dict:
    """Mirror `gen_json_tac_deltabase_eef_bimanual_parallel.cal_stats` exactly.

    Action stats stay per-(step, dim): np.*(axis=0) over [M,16,62] -> [16,62],
    which SftDataset._normalize broadcasts against the [B,16,62] chunk.
    """
    actions, states, tactiles, track = [], [], [], []
    by_ep: dict[str, list] = {}
    for rec in all_records:
        actions.append(rec["action"])
        states.append(rec["state"])
        tactiles.append(rec["tactile_f6"])
        by_ep.setdefault(rec["episode_id"], []).append(rec)
    for ep, rows in by_ep.items():
        for t in range(1, len(rows)):
            track.append(compute_bimanual_tracking_error(
                np.array(rows[t]["state"]), np.array(rows[t - 1]["absolute_target_action"])))

    actions = np.asarray(actions, dtype=np.float32)            # [M,16,62]
    states = np.asarray(states, dtype=np.float32)              # [M,62]
    tac = np.asarray(tactiles, dtype=np.float32).reshape(len(tactiles), -1)   # [M,60]

    block = {
        "action": calculate_stats(actions, [True] * ACTION_DIM),
        "state": calculate_stats(states, [True] * ACTION_DIM),
        "tactile_f6": calculate_stats(tac, [True] * F6_DIM),
        "num_transitions": int(len(actions)),
        "num_trajectories": int(len(by_ep)),
    }
    if track:
        tr = np.asarray(track, dtype=np.float32)
        block["tracking_error"] = {
            "mean": np.mean(tr, axis=0).tolist(),
            "std": np.std(tr, axis=0).tolist(),
            "mean_abs": np.mean(np.abs(tr), axis=0).tolist(),
            "mask": [True] * TRACKING_ERROR_DIM,
        }
    else:
        block["tracking_error"] = {}

    result = {dataset_name: block}
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                   help="dir holding <session>/ep_*/ep_*.hdf5")
    p.add_argument("--img_save_root", required=True)
    p.add_argument("--json_save_root", required=True)
    p.add_argument("--task_name", required=True)
    p.add_argument("--json_name_base", required=True)
    p.add_argument("--instruction", default="Pick up the egg.")
    p.add_argument("--dataset_name", default="rlbench",
                   help="top-level key in _statistics.json; SftDataset reads the first key")

    p.add_argument("--mount_geom", default="",
                   help="path to openarm-sharpa-sim/openarm_sharpa/assets/mount_geom.py "
                        "(required unless --no_ee_offset)")
    p.add_argument("--no_ee_offset", action="store_true",
                   help="emit link7-frame poses. Almost never what you want -- see the "
                        "docstring of load_mount_transform.")
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--no_rep103_state", action="store_true",
                   help="keep `state` in the capture base frame instead of REP-103. "
                        "Inert while --use_robot_state is 0.")

    p.add_argument("--action_chunk", type=int, default=ACTION_CHUNK)
    p.add_argument("--frame_stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--min_frames", type=int, default=48,
                   help="drop episodes whose longest engaged run is shorter than this")
    p.add_argument("--crop_box", type=int, nargs=4, default=None,
                   metavar=("Y0", "Y1", "X0", "X1"),
                   help="head-camera crop. Off by default: T-Rex's CROP_BOX_SLOW was "
                        "measured for the Dexmate head mount and means nothing here.")
    p.add_argument("--head_size", type=int, nargs=2, default=[640, 400], metavar=("W", "H"))
    p.add_argument("--wrist_size", type=int, nargs=2, default=[640, 360], metavar=("W", "H"))
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="convert only the first N episodes")
    args = p.parse_args()

    if not args.no_ee_offset and not args.mount_geom:
        p.error("--mount_geom is required (or pass --no_ee_offset deliberately)")

    ee_offset, mount_prov = (None, {"applied": False, "reason": "--no_ee_offset"})
    if not args.no_ee_offset:
        ee_offset, mount_prov = load_mount_transform(args.mount_geom, args.side)
        mount_prov["applied"] = True
        print("[mount] T_link7->L_ee:")
        print(f"        roll {mount_prov['applied_roll_deg']:+.4f} deg about +Z, "
              f"z {mount_prov['total_z_m']:.4f} m "
              f"({mount_prov['HAND_MOUNT_Z']:.4f} + {mount_prov['FLANGE_TO_WRIST_Z']:.4f})")
        print(f"        from {mount_prov['mount_geom_path']}")

    episodes = list_episodes(args.data_root)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"[scan] {len(episodes)} episode(s) under {args.data_root}")
    if not episodes:
        sys.exit("nothing to convert")

    os.makedirs(args.json_save_root, exist_ok=True)
    zero_png = write_shared_zero_deform(
        os.path.join(args.img_save_root, args.task_name, "_zero_deform_240.png"))

    jobs = []
    for gi, ep in enumerate(episodes):
        jobs.append({
            **ep,
            "episode_tag": f"episode_{gi:04d}",
            "img_save_root": args.img_save_root,
            "task_name": args.task_name,
            "instruction": args.instruction,
            "ee_offset": ee_offset,
            "rep103_state": not args.no_rep103_state,
            "action_chunk": args.action_chunk,
            "frame_stride": args.frame_stride,
            "min_frames": args.min_frames,
            "crop_box": tuple(args.crop_box) if args.crop_box else None,
            "head_size": tuple(args.head_size),
            "wrist_size": tuple(args.wrist_size),
            "zero_deform_path": zero_png,
        })

    all_records, reports = [], []
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futs = {pool.submit(process_episode, j): j for j in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                tag, recs, rep = fut.result()
            except Exception as exc:                          # noqa: BLE001
                rep = {"episode_tag": j["episode_tag"], "source": j["h5_path"],
                       "skipped": f"{type(exc).__name__}: {exc}"}
                tag, recs = j["episode_tag"], []
            reports.append(rep)
            all_records.extend(recs)
            status = rep.get("skipped", f"{rep.get('frames', 0)} frames")
            print(f"[{done}/{len(jobs)}] {j['session']}/{j['ep_name']} -> {tag}: {status}",
                  flush=True)

    if not all_records:
        sys.exit("every episode was skipped -- nothing written")

    # episode_id order must be stable so the F6 history index groups correctly.
    all_records.sort(key=lambda r: (r["episode_id"],
                                    int(_FRAME_RE.search(r["tactile_image_deform"][0]).group(2))))

    json_path = os.path.join(args.json_save_root, f"{args.json_name_base}.json")
    train_records = [{
        "input_prompt": r["language_instruction"],
        "input_image_slow": [r["image_old_slow"]],
        "input_image_fast": r["image_old_fast"],
        "input_image_resolution": [384, 384],
        "action": r["action"],
        "state_slow": r["state"],
        "state_fast": r["state"],
        "tactile_f6": r["tactile_f6"],
        "tactile_image_deform": r["tactile_image_deform"],
    } for r in all_records]
    with open(json_path, "w") as fh:
        json.dump(train_records, fh)

    stats_path = json_path.replace(".json", "_statistics.json")
    write_stats(all_records, stats_path, args.dataset_name)

    prov_path = json_path.replace(".json", "_provenance.json")
    with open(prov_path, "w") as fh:
        json.dump({
            "generator": os.path.basename(__file__),
            "data_root": os.path.abspath(args.data_root),
            "mount": mount_prov,
            "rep103_state": not args.no_rep103_state,
            "collector_finger_order": COLLECTOR_FINGER_ORDER,
            "trex_finger_order": TREX_FINGER_ORDER,
            "collector_to_trex": COLLECTOR_TO_TREX,
            "action_chunk": args.action_chunk,
            "frame_stride": args.frame_stride,
            "crop_box": args.crop_box,
            "head_size": args.head_size,
            "wrist_size": args.wrist_size,
            "instruction": args.instruction,
            "episodes": sorted(reports, key=lambda r: r["episode_tag"]),
        }, fh, indent=2)

    kept = sum(1 for r in reports if "skipped" not in r)
    print(f"\n[done] {kept}/{len(reports)} episodes, {len(all_records)} samples")
    print(f"  json  {json_path}")
    print(f"  stats {stats_path}")
    print(f"  prov  {prov_path}")


if __name__ == "__main__":
    main()
