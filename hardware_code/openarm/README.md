# Serving T-Rex to the OpenArm + Sharpa Wave (left) rig

Code for running a checkpoint from
[`zhx-tactile-steering/T-Rex-openarm-sharpa-left-egg`](https://huggingface.co/zhx-tactile-steering/T-Rex-openarm-sharpa-left-egg)
on the real robot. The operator's checklist is [`RUNBOOK.md`](RUNBOOK.md) (Chinese, step by
step, with a pass/fail criterion per step). This file is the *why*: the contracts between
the pieces and the conventions that make the live inputs match the training data. The
design record is `docs/superpowers/specs/2026-09-11-openarm-sharpa-left-rollout-design.md`.

```
 T-Rex checkout (.venv, GPU)                    rig checkout  ~/openarm/openarm_track  (rig venv + ROS)
 ┌──────────────────────────────┐   websocket   ┌──────────────────────────────────────────────┐
 │ scripts/serve_openarm.sh     │◄─ msgpack ───►│ rollout/cli.py --backend trex                │
 │  └ hardware_code/openarm/    │               │  ├ rollout/policy_trex.py   payload, decoder │
 │     serve.py                 │               │  ├ rollout/observe.py       + tactile window │
 │      └ scripts/test.py       │               │  ├ rollout/robot_openarm.py absolute targets │
 │         CascadedServer       │               │  └ collect/tactile.py       SniffTactile     │
 └──────────────────────────────┘               └──────────────────────────────────────────────┘
        hardware_code/openarm/tactile_convert.py  ──── imported by the rig via --trex-root ────┘
```

The rig side is on the rig's branch `trex-backend`; the seam it plugs into is documented in
`rollout/policy_base.py` there. The pi0.5 path in that client is untouched.

## Files here

| file | role |
|---|---|
| `serve.py` | The policy server. Wraps `scripts/test.py`'s `CascadedServer` (unchanged) in openpi's websocket + msgpack transport so the rig's existing client class talks to it. Validates every payload key. `--random_weights 1` builds the graph without `model.pt` for a latency measurement. |
| `msgpack_numpy.py` | 40 lines vendored from openpi-client so the T-Rex venv needs no openpi checkout. |
| `tactile_convert.py` | Collector conventions → dataset conventions: finger order, F6 resting bias, deform floor, right-hand padding. Pure numpy, imported by the rig client. Pinned against the converter by `tactile_convert_test.py`. |
| `ping_trex.py` | Synthetic observations against a running server: latency (sets `--infer-lead`), chunk shape, finiteness, hand targets vs `stats_data.json`. |
| `../../scripts/serve_openarm.sh` | Launches `serve.py` with the checkpoint's settings; refuses a directory missing `model.pt` or `stats_data.json`. |
| `../../scripts/setup_rollout_env.sh` | The inference venv. The rollout laptop is an RTX 5090 (sm_120); `pyproject.toml`'s torch 2.6/cu124 has no kernels for it, so this installs torch 2.7.1+cu128. |

## The payload contract

Client → server, one msgpack-numpy dict per request:

| key | shape / type | notes |
|---|---|---|
| `prompt` | str | **verbatim** `Pick up the egg with your left hand and place it down.` (the converter's `--instruction`; the model card repeats it). |
| `head` | `(400, 640, 3)` uint8 RGB | Scene camera, full frame, no crop. Training: 1280×800 → `INTER_AREA` 640×400 → PNG → LANCZOS 384×288 in `test.py`. The rig grabber is asked for short side 400 so the server-side LANCZOS sees the same input size. |
| `wrist` | `(360, 640, 3)` uint8 RGB | Left wrist camera, 960×540 → 640×360. Fed to **both** fast slots: the converter wrote the same left-wrist path twice to keep the two-wrist layout the midtrain used. |
| `tactile_f6` | `(T, 5, 6)` float32 | T-Rex finger order **[thumb, index, middle, ring, pinky]**, resting bias already subtracted. Consecutive frames at the 30 Hz loop rate, newest last; the server uses the last 16 (front-pads if fewer). |
| `tactile_deform` | `(5, 240, 240)` uint8 | T-Rex finger order, background floor already subtracted so the modal pixel is 0. |
| `state` | `(62,)` float32, optional | Ignored by these checkpoints (`use_robot_state 0`). |

Server → client: `actions` `(16, 62)` float32 denormalised, `chunk_id`, `server_timing`.
The right hand's five fingers are zero-padded on the server; the client never sees them.

On connect the server sends metadata: `{"model": "trex", "horizon": 16, "action_dim": 62,
"f6_window": 16, "prompt_hint": ..., "random_weights": bool, ...}`. The client refuses to
command hardware when `random_weights` is true.

## The action contract and its decoder

`actions[k]` for `k = 0..15`:

| slice | meaning |
|---|---|
| `[0:3]` | Δxyz of the target at step *k*, in the **chunk-start `hand_wrist` frame**, metres |
| `[3:9]` | first two columns of `R_delta = R_measᵀ R_target` |
| `[9:31]` | 22 hand joint targets, absolute radians, Sharpa order (`thumb_CMC_FE` first) |
| `[31:62]` | right side, constant padding — discard |

All 16 steps are relative to the **same** pose: the `hand_wrist` pose at the observation,
computed from the **measured** joints (the converter used `obs/wrist_pose_b` = FK(q_meas)
as the base and `action/wrist_pose_b` = the teleop node's Cartesian target as the label,
`gen_json_openarm_sharpa_left.py:465-475`). The decoder in `rollout/policy_trex.py`:

```
T_ee_meas  = FK_link7(q_meas) · T_mount           T_mount = Rz(+144.175°) · Trans(0, 0, 0.0545)
T_ee_k     = T_ee_meas · [R_delta_k | Δxyz_k]      R_delta_k rebuilt by Gram-Schmidt from the 6-D
T_link7_k  = T_ee_k · T_mount⁻¹
```

`T_mount` is imported from `openarm-sharpa-sim/openarm_sharpa/assets/mount_geom.py`
(`HAND_MOUNT_Z + FLANGE_TO_WRIST_Z`, `PALM_ROLL_DEG`), the same file the converter used;
the decoder refuses to run without it. The base frame cancels in the delta (design doc
§1.1), so the rig's pinocchio base is used throughout. The EEF frame does *not* cancel: a
decoder that skipped `T_mount` would be wrong by a 144° roll and 54.5 mm.

Each `T_link7_k` becomes a `RobotCommand.wrist_pose7` — an **absolute** link7 target. The
robot layer (`rollout/robot_openarm.py`) converts it to the palm-frame increment from its
own commanded pose at the moment of application and then runs the unchanged per-step and
cumulative wrist envelopes. That is deliberately not done in the backend: when the chunk is
decoded, `--infer-lead` steps before it executes, the backend cannot know where the chain
will be.

Why absolute and not "treat the delta as an increment like pi0.5": pi0.5's labels are
differences between consecutive *commands*; T-Rex's are *command minus measurement*, and the
difference between the two is the servo's tracking error under the 1.44 kg hand (8 mm
median). Integrating T-Rex deltas as increments would add that error once per chunk.

`loop.py` consumes every chunk after the first from index `--infer-lead`, because
`actions[j]` is the target *j* steps after the observation. For absolute targets that is
exactly right and has no double-integration hazard; repeating a target is a hold.

## Tactile

```
sudo collect/sharpa_tap.py --serve /tmp/sharpa_tap.sock      # AF_PACKET copy of :50011
        → collect/tactile.py SniffTactile(with_state=False)  # (5,6) f6 + (5,240,240) deform, channel order
        → rollout/observe.py  16-frame F6 deque at 30 Hz     # Observation.tactile, collector conventions
        → rollout/policy_trex.py + tactile_convert.py        # reorder, − bias, − floor
        → serve.py                                           # right hand zero-padded
```

- **Order.** Channels 5..9 are `[pinky, ring, middle, index, thumb]` on this rig (measured
  by poking, `finger_map.py`); T-Rex is thumb-first. `COLLECTOR_TO_TREX = [4, 3, 2, 1, 0]`.
- **Bias.** The converter subtracted, per episode, the mean of **all six** channels over the
  pre-contact head. `policy_trex.calibrate` samples 2 s at the start of each run, hand in
  the air, and does the same. This is not optional: the embedded VQ-VAE normalises with
  min/max buffers baked into the checkpoint, and the thumb's standing 2–3.5 N lands in a
  different code if left in.
- **Floor.** The converter subtracted the modal deform pixel (2 on every session measured)
  so the background is 0, as in T-Rex's own data. Same calibration frames.
- **Window density.** The VQ-VAE's 16-frame history is 16 consecutive 30 Hz rows. The
  client keeps the deque and sends the whole window; a server-side rolling buffer fed once
  per chunk would space frames 8 steps apart.
- **Pilot.** The device only computes deform maps while a client asks for them, so Sharpa
  Pilot must be open with the tactile view, exactly as during collection. The pi0.5
  runbook's "Pilot not needed" does not apply.

## Latency

`horizon = 16` and `loop.py` requires `infer_lead < chunk_steps` and
`infer_lead + chunk_steps ≤ 16`, so the largest lead is **7 steps = 233 ms** at 30 Hz, and
the whole `slow_and_fast` pass plus transport has to fit in it. Measure with
`ping_trex.py` (works with `--random_weights 1` before the weights are downloaded).

If it does not fit, in order of preference:

1. lower `--cascaded_total_steps` (and `--cascaded_split_step` proportionally) on the server;
2. add an action stride to the loop: execute `actions[0, 2, 4, …]` at 15 Hz. Exact for
   absolute targets (action 2k is the target 2k/30 s after the observation) and doubles the
   budget. Not implemented; do it only if 1 is not enough.

## What is not done

- No mid-chunk `fast` refinement (the reference eval refines at offsets 4/8/12). The loop
  re-infers every `chunk_steps` (≥ 8) steps with fresh tactile, which is half that rate.
- `state` is sent as nothing; the 62-d absolute-pose layout is in `gen_json_openarm_sharpa_left.py`
  (`state62`, REP-103 base) if a future checkpoint enables `use_robot_state`.
- Deform maps are not recorded in the flight file (8.6 MB/s raw); F6 is.

## Tests

```
# T-Rex side, in .venv: constants and functions pinned against the converter
.venv/bin/python -m pytest hardware_code/openarm/tactile_convert_test.py -q
# rig side, in openpi's venv as the other rollout tests
/home/yiming/Public/openpi/.venv/bin/python -m pytest ~/openarm/openarm_track/rollout/{policy_trex,robot_openarm,observe}_test.py -q
```
