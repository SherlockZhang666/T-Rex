#!/usr/bin/env python3
"""Exercise hardware_code/openarm/serve.py end to end without the robot.

    .venv/bin/python hardware_code/openarm/ping_trex.py --n 30
    .venv/bin/python hardware_code/openarm/ping_trex.py --stats <ckpt>/stats_data.json

Answers the three questions you would otherwise find out with the arm engaged:

  1. Does the serving path -- checkpoint, processor, resize, cascaded flow, VQ-VAE window,
     denormalisation -- produce a (16, 62) chunk at all?
  2. How long does slow_and_fast take on THIS machine? That sets --infer-lead for the
     rollout client: the budget is infer_lead / fps seconds, and with a 16-step horizon
     the loop's constraints (lead < chunk_steps, lead + chunk_steps <= 16) cap the lead
     at 7, i.e. 233 ms at 30 Hz.
  3. With --stats: are the hand targets inside the q01/q99 the policy was trained in?
     Synthetic images make the answer meaningless as behaviour, but a chunk of NaN, the
     wrong width, or fingers commanded to +-pi/2 is a wiring fault and shows up here.

Client-only: needs `websockets` and `msgpack` (both in the serving venv) and nothing else.
Talks the same protocol as openpi_client.WebsocketClientPolicy, reimplemented in 20 lines
so the T-Rex venv does not need the openpi checkout.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import msgpack_numpy  # noqa: E402

HEAD_HW = (400, 640)
WRIST_HW = (360, 640)
N_LEFT = 5
DEFORM_HW = 240
HORIZON_MAX = 16       # loop.py: chunk_steps + infer_lead <= horizon, infer_lead < chunk_steps


class Client:
    def __init__(self, host: str, port: int):
        import websockets.sync.client

        self.uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None)
        self.metadata = msgpack_numpy.unpackb(self._ws.recv())

    def infer(self, payload: dict) -> dict:
        self._ws.send(self._packer.pack(payload))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"server error:\n{reply}")
        return msgpack_numpy.unpackb(reply)

    def close(self):
        self._ws.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", default=None,
                   help="default: the server's advertised prompt_hint")
    p.add_argument("--n", type=int, default=20, help="inference calls to time")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--infer-lead", type=int, default=7,
                   help="the value you intend to pass to cli.py; budget is lead/fps")
    p.add_argument("--stats", default=None, metavar="stats_data.json",
                   help="check the hand targets against the checkpoint's q01/q99")
    p.add_argument("--window", type=int, default=16)
    return p.parse_args(argv)


def synthetic_payload(rng, prompt: str, window: int) -> dict:
    """Shapes and dtypes exactly as the rollout client sends them."""
    return {
        "prompt": prompt,
        "head": rng.integers(0, 256, size=(*HEAD_HW, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 256, size=(*WRIST_HW, 3), dtype=np.uint8),
        # Resting hand after bias removal: forces near zero, as in the training head.
        "tactile_f6": rng.normal(scale=0.02, size=(window, N_LEFT, 6)).astype(np.float32),
        "tactile_deform": np.zeros((N_LEFT, DEFORM_HW, DEFORM_HW), np.uint8),
    }


def load_hand_bounds(path):
    raw = json.loads(pathlib.Path(path).read_text())
    ds = raw[next(iter(raw))]["action"]
    q01 = np.asarray(ds["q01"], dtype=np.float64)
    q99 = np.asarray(ds["q99"], dtype=np.float64)
    if q01.ndim == 2:           # [16, 62] per step -> the envelope over all steps
        q01, q99 = q01.min(axis=0), q99.max(axis=0)
    return q01[9:31], q99[9:31]


def check_chunk(chunk: np.ndarray, horizon: int, action_dim: int, bounds) -> list[str]:
    problems = []
    if chunk.shape != (horizon, action_dim):
        return [f"chunk is {chunk.shape}, expected ({horizon}, {action_dim})"]
    if not np.isfinite(chunk).all():
        problems.append(f"{int((~np.isfinite(chunk)).sum())} non-finite values")
        return problems
    # The rotation block must be two near-orthonormal columns: that is a model-side
    # invariant (it is what the data looked like), and a random-weights server breaks it,
    # which is a useful reminder that --random_weights is not a policy.
    c1, c2 = chunk[:, 3:6], chunk[:, 6:9]
    n1, n2 = np.linalg.norm(c1, axis=1), np.linalg.norm(c2, axis=1)
    dot = np.abs((c1 * c2).sum(axis=1))
    if (np.abs(n1 - 1) > 0.2).any() or (np.abs(n2 - 1) > 0.2).any() or (dot > 0.2).any():
        problems.append(f"rotation columns are not orthonormal (|c1| {n1.min():.2f}..{n1.max():.2f}, "
                        f"|c2| {n2.min():.2f}..{n2.max():.2f}, |c1.c2| max {dot.max():.2f})")
    if bounds is not None:
        lo, hi = bounds
        hand = chunk[:, 9:31]
        margin = 0.15
        n_out = int(((hand < lo - margin) | (hand > hi + margin)).sum())
        if n_out:
            problems.append(f"{n_out} hand targets more than {margin} rad outside q01/q99")
    return problems


def main(argv=None) -> int:
    a = parse_args(argv)
    print(f"connecting to ws://{a.host}:{a.port} ...")
    client = Client(a.host, a.port)
    meta = client.metadata
    print("server metadata:", meta)
    horizon = int(meta.get("horizon", 16))
    action_dim = int(meta.get("action_dim", 62))
    prompt = a.prompt or meta.get("prompt_hint") or "Pick up the egg with your left hand and place it down."
    if meta.get("random_weights"):
        print("\n!!! the server runs with RANDOM WEIGHTS: timing is real, everything else is noise\n")
    bounds = load_hand_bounds(a.stats) if a.stats else None

    rng = np.random.default_rng(0)
    latencies, server_ms, problems = [], [], []
    slow_ms, fast_ms = [], []
    hand_steps, wrist_off = [], []
    for i in range(a.n):
        payload = synthetic_payload(rng, prompt, a.window)
        t0 = time.monotonic()
        out = client.infer(payload)
        latencies.append(time.monotonic() - t0)
        server_ms.append(float(out["server_timing"]["infer_ms"]))
        slow_ms.append(float(out["server_timing"].get("slow_ms", float("nan"))))
        fast_ms.append(float(out["server_timing"].get("fast_ms", float("nan"))))
        chunk = np.asarray(out["actions"], dtype=np.float64)
        problems.extend(f"call {i}: {m}" for m in check_chunk(chunk, horizon, action_dim, bounds))
        if chunk.shape == (horizon, action_dim):
            wrist_off.append(np.linalg.norm(chunk[:, 0:3], axis=1))
            hand_steps.append(np.abs(np.diff(chunk[:, 9:31], axis=0)).max())
    client.close()

    # Drop the first call by POSITION: it may still carry lazy CUDA initialisation.
    steady = sorted(latencies[1:]) or sorted(latencies)
    p50 = statistics.median(steady)
    p95 = steady[min(len(steady) - 1, int(0.95 * len(steady)))]
    budget = a.infer_lead / a.fps

    print(f"\nchunk shape        ({horizon}, {action_dim})")
    print(f"first call         {latencies[0] * 1e3:7.0f} ms")
    print(f"steady   p50       {p50 * 1e3:7.0f} ms   round trip, over {len(steady)} calls")
    print(f"         p95       {p95 * 1e3:7.0f} ms")
    print(f"         max       {steady[-1] * 1e3:7.0f} ms")
    print(f"server   p50       {statistics.median(server_ms):7.0f} ms   slow_and_fast only "
          f"(slow {statistics.median(slow_ms):.0f} ms: images + prefill + {meta.get('cascaded_steps', ['?', '?'])[1]} "
          f"flow steps; fast {statistics.median(fast_ms):.0f} ms: tactile expert, the rest)")
    print(f"budget at --infer-lead {a.infer_lead} @ {a.fps:g} Hz: {budget * 1e3:.0f} ms")
    if wrist_off:
        w = np.concatenate(wrist_off)
        print(f"\nwrist |delta_xyz|  median {np.median(w) * 1e3:5.1f} mm  max {w.max() * 1e3:5.1f} mm "
              f"(chunk-start relative; training q99 is a few cm at step 15)")
        print(f"hand max |step|    {max(hand_steps):.3f} rad between consecutive actions")

    if problems:
        print("\nCHUNK PROBLEMS (wiring faults, not policy quality):")
        for msg in problems[:10]:
            print("  " + msg)
        if not meta.get("random_weights"):
            return 1

    need = int(np.ceil(p95 * a.fps)) + 1
    if p95 > budget:
        max_lead = (HORIZON_MAX - 1) // 2
        if need > max_lead:
            print(f"\nlatency does not fit ANY lead: p95 needs --infer-lead {need}, and with a "
                  f"{HORIZON_MAX}-step horizon the largest legal lead is {max_lead} "
                  f"(lead < chunk_steps, lead + chunk_steps <= {HORIZON_MAX}). Options: serve "
                  f"with --cascaded_total_steps 5 --cascaded_split_step 3 (same tau_split as "
                  f"training), or run the client at --fps 15 (slow motion). See "
                  f"hardware_code/openarm/README.md, 'Latency'.")
        else:
            print(f"\nlatency does not fit the budget. Use --infer-lead {need} "
                  f"--chunk-steps {HORIZON_MAX - need}.")
        return 2
    print(f"\nOK -- p95 fits in {budget * 1e3:.0f} ms with {(budget - p95) * 1e3:.0f} ms to spare. "
          f"Pass --infer-lead {a.infer_lead} --chunk-steps {HORIZON_MAX - a.infer_lead}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
