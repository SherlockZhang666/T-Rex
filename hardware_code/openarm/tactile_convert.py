"""Live Sharpa tactile readings -> the conventions the OpenArm post-train data was written in.

Every transform here mirrors one line of `utils/gen_json_openarm_sharpa_left.py`, which is
what produced the training JSON. The policy never saw raw device values: it saw fingers in
T-Rex order, forces with the resting bias removed, deform maps with the background floor
removed, and a right hand made of zeros. A client that skips any one of these does not get
an error -- it gets a policy that thinks the thumb is always touching something.

NUMPY ONLY, ON PURPOSE. This module is imported by the rig's rollout client, whose venv has
no cv2 and no h5py, so it cannot import the converter itself. `tactile_convert_test.py`
(run in the T-Rex venv) pins every constant and function here against the converter's, so
the two cannot drift apart silently.

Finger order
------------
The collector measured its channel order by poking each fingertip (2026-08-20):
channels 5..9 = [pinky, ring, middle, index, thumb]. T-Rex's is thumb-first. Both F6 and
deform are reordered with the same index list.

F6 bias
-------
The converter subtracts, per episode, the MEAN of all six channels over the frames before
first contact (`f6_zero_offset`). Not just the three force axes -- all six. The rollout
client measures the same thing at the start of a run, hand in the air, and passes it in.

Deform floor
------------
The converter subtracts the modal pixel value of the episode (`deform_floor`) so the
background is exactly 0, which is what T-Rex's own deform maps look like. On this rig the
floor has been 2 on every session measured; it is detected, not assumed.
"""

from __future__ import annotations

import numpy as np

N_FINGERS_PER_HAND = 5
F6_PER_FINGER = 6
DEFORM_HW = 240
F6_WINDOW = 16          # the embedded VQ-VAE's history window, `training_args.vqvae_config.window`

COLLECTOR_FINGER_ORDER = ("pinky", "ring", "middle", "index", "thumb")
TREX_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
# COLLECTOR_TO_TREX[i] is the collector index that supplies T-Rex finger i.
COLLECTOR_TO_TREX = [COLLECTOR_FINGER_ORDER.index(n) for n in TREX_FINGER_ORDER]


def f6_bias(head_frames) -> np.ndarray:
    """(N, 5, 6) raw collector-order frames captured with nothing in contact -> (5, 6) bias.

    Mean over all six channels, exactly as `f6_zero_offset` computes it over an episode's
    pre-contact head. Refuses fewer than 10 frames: a bias from a handful of samples is
    noise, and the failure it produces (a constant offset on every reading of the run) is
    invisible downstream.
    """
    f = np.asarray(head_frames, dtype=np.float64)
    if f.ndim != 3 or f.shape[1:] != (N_FINGERS_PER_HAND, F6_PER_FINGER):
        raise ValueError(f"expected (N, 5, 6) frames, got {f.shape}")
    if f.shape[0] < 10:
        raise ValueError(f"need at least 10 frames for a bias, got {f.shape[0]}")
    if not np.isfinite(f).all():
        raise ValueError("non-finite values in the bias frames -- a finger was not live")
    return f.mean(axis=0).astype(np.float32)


def deform_floor(frames) -> int:
    """Modal pixel value over a sample of frames. Verbatim from the converter."""
    frames = np.asarray(frames, dtype=np.uint8)
    sample = frames[:: max(1, len(frames) // 40)]
    counts = np.bincount(sample.ravel(), minlength=256)
    return int(counts.argmax())


def convert_f6_window(window, bias) -> np.ndarray:
    """(T, 5, 6) raw collector order -> (T, 5, 6) T-Rex order, bias removed, float32.

    `T` is whatever the caller has; the server takes the last `F6_WINDOW` and front-pads
    with the first frame when there are fewer, the same rule `test.py` applies to a dense
    client window.
    """
    w = np.asarray(window, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    if w.ndim != 3 or w.shape[1:] != (N_FINGERS_PER_HAND, F6_PER_FINGER):
        raise ValueError(f"expected (T, 5, 6) window, got {w.shape}")
    if b.shape != (N_FINGERS_PER_HAND, F6_PER_FINGER):
        raise ValueError(f"expected a (5, 6) bias, got {b.shape}")
    if not np.isfinite(w).all():
        raise ValueError("non-finite F6 in the window -- a finger dropped off the tap")
    return (w - b[None])[:, COLLECTOR_TO_TREX, :]


def convert_deform(deform, floor: int) -> np.ndarray:
    """(5, 240, 240) uint8 collector order -> T-Rex order with the floor subtracted."""
    d = np.asarray(deform)
    if d.shape != (N_FINGERS_PER_HAND, DEFORM_HW, DEFORM_HW):
        raise ValueError(f"expected (5, {DEFORM_HW}, {DEFORM_HW}) deform, got {d.shape}")
    if d.dtype != np.uint8:
        raise ValueError(f"deform must be uint8 as decoded off the wire, got {d.dtype}")
    out = np.clip(d.astype(np.int16) - int(floor), 0, 255).astype(np.uint8)
    return out[COLLECTOR_TO_TREX]


def pad_right_hand_f6(left) -> np.ndarray:
    """(T, 5, 6) -> (T, 10, 6): the right hand is all zeros, as in training."""
    left = np.asarray(left, dtype=np.float32)
    if left.ndim != 3 or left.shape[1:] != (N_FINGERS_PER_HAND, F6_PER_FINGER):
        raise ValueError(f"expected (T, 5, 6), got {left.shape}")
    return np.concatenate([left, np.zeros_like(left)], axis=1)


def pad_right_hand_deform(left) -> np.ndarray:
    """(5, H, W) uint8 -> (10, H, W): five all-zero maps for the right hand."""
    left = np.asarray(left, dtype=np.uint8)
    if left.ndim != 3 or left.shape[0] != N_FINGERS_PER_HAND:
        raise ValueError(f"expected (5, H, W), got {left.shape}")
    return np.concatenate([left, np.zeros_like(left)], axis=0)
