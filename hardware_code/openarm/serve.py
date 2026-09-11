#!/usr/bin/env python3
"""Serve a T-Rex checkpoint to the OpenArm + Sharpa rollout client over websocket + msgpack.

    scripts/serve_openarm.sh checkpoints/openarm-sharpa-left-egg/checkpoint-29-49290
    # or, by hand, in the venv from scripts/setup_rollout_env.sh:
    .venv/bin/python hardware_code/openarm/serve.py --checkpoint_path <ckpt> --image_size 384 288

WHAT THIS IS
------------
`scripts/test.py` already holds everything model-side: loading, the LANCZOS resize, the
cascaded slow/fast flow, the F6 window, denormalisation (`CascadedServer`). It exposes it
over ZMQ + pickle. The rollout client on this rig already speaks openpi's websocket +
msgpack-numpy protocol (that is how it talked to pi0.5), and pickle across two Python /
numpy versions is a coin flip. So this file is a transport adapter and nothing else:
one `CascadedServer`, openpi's wire format, a stricter payload contract.

THE PAYLOAD CONTRACT (client -> server), all keys required unless noted
-----------------------------------------------------------------------
    prompt          str, VERBATIM the training instruction
    head            (H, W, 3) uint8 RGB, the scene camera, full frame
    wrist           (H, W, 3) uint8 RGB, the LEFT wrist camera; it is fed to BOTH fast
                    slots because training repeated it (one wrist camera, two-slot model)
    tactile_f6      (T, 5, 6) float32, T-Rex finger order, bias already removed; the
                    last 16 are used, front-padded if T < 16. 30 Hz consecutive frames.
    tactile_deform  (5, 240, 240) uint8, T-Rex finger order, floor already removed
    state           (62,) float32, optional; ignored unless the checkpoint used robot state

The right hand (fingers 5..9) is padded with zeros HERE, so the client only ever deals
with the five fingers it has. Finger reordering, bias and floor are the client's job
(hardware_code/openarm/tactile_convert.py) because they are properties of the collection
rig, not of the model.

RESPONSE
--------
    actions         (16, 62) float32, denormalised. [0:9] left arm delta-base EEF 9-D in
                    the chunk-start hand_wrist frame, [9:31] left hand absolute radians,
                    [31:62] padding -- discard.
    chunk_id        int
    server_timing   {"infer_ms": float, "slow_ms": float, "fast_ms": float}

`--random_weights` builds the network from config.json without model.pt. The outputs are
garbage; the timing is real. It exists so --infer-lead can be sized before 8.5 GB finishes
downloading.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import importlib.util
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import msgpack_numpy  # noqa: E402  (vendored, next to this file)

logger = logging.getLogger("trex-serve")

N_LEFT = 5
F6_DIM = 6
DEFORM_HW = 240


def _load_trex_test():
    """scripts/test.py as a module. Not `import test`: that name is the stdlib's."""
    path = ROOT / "scripts" / "test.py"
    spec = importlib.util.spec_from_file_location("trex_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- arguments
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cuda", default="0")
    p.add_argument("--prompt_hint", default="Pick up the egg with your left hand and place it down.",
                   help="advertised in the server metadata; the client still sends its own")
    p.add_argument("--random_weights", type=int, default=0,
                   help="1: do not load model.pt (latency probe only, outputs are garbage)")
    p.add_argument("--fast_processor", type=int, default=1,
                   help="1: Qwen2VLImageProcessorFast (torch) instead of the PIL one. Same "
                        "pixel_values to 6e-8 on this checkpoint's 384x288 inputs, 3-4x "
                        "faster on the CPU side. 0 restores test.py's default.")
    # Everything below mirrors scripts/test.py's parser so model_load / CascadedServer see
    # the Namespace they expect. Defaults are THIS checkpoint family's, not test.py's
    # (which defaults to action_dim 31 / chunk 8 and would build the wrong head).
    p.add_argument("--base_model_path", default="")
    p.add_argument("--stats_path", default="")
    p.add_argument("--dataset_name", default="")
    p.add_argument("--action_dim", type=int, default=62)
    p.add_argument("--action_chunk", type=int, default=16)
    p.add_argument("--use_robot_state", type=int, default=0)
    p.add_argument("--use_tactile_deform", type=int, default=1)
    p.add_argument("--use_tactile_vec", type=int, default=1)
    p.add_argument("--tactile_intermediate_size", type=int, default=0)
    p.add_argument("--n_flare_tokens_per_frame", type=int, default=0)
    p.add_argument("--n_flare_steps", type=int, default=0)
    p.add_argument("--image_size", type=int, nargs=2, default=[384, 288], metavar=("W", "H"))
    p.add_argument("--cascaded_total_steps", type=int, default=10)
    p.add_argument("--cascaded_split_step", type=int, default=6)
    p.add_argument("--disable_tactile", type=int, default=0)
    p.add_argument("--use_tactile_code", type=int, default=0)
    p.add_argument("--vqvae_codebook_size", type=int, default=64)
    p.add_argument("--vqvae_ckpt", default="")
    p.add_argument("--vqvae_fp32", type=int, default=0)
    return p.parse_args(argv)


def check_checkpoint(ckpt: pathlib.Path, random_weights: bool) -> None:
    """Refuse the two silent failures: a policy with no weights, or no normalisation."""
    missing = [n for n in ("config.json", "training_args.json", "stats_data.json", "processor")
               if not (ckpt / n).exists()]
    if not random_weights and not (ckpt / "model.pt").exists():
        missing.append("model.pt")
    if missing:
        raise SystemExit(f"{ckpt}: missing {missing}. A checkpoint directory from the Hub "
                         f"has all of config.json, model.pt, processor/, stats_data.json, "
                         f"training_args.json. Count the files -- `hf download --include` "
                         f"that matches nothing exits 0.")
    ta = json.loads((ckpt / "training_args.json").read_text())
    for key in ("action_dim", "action_chunk", "use_tactile_deform", "use_tactile_vec",
                "use_robot_state"):
        if key not in ta:
            raise SystemExit(f"training_args.json has no '{key}'")


def reconcile_with_training_args(args, ckpt: pathlib.Path) -> None:
    """The flags test.py does NOT restore from training_args.json, restored here.

    test.py auto-detects the tactile/flare/cascaded settings but leaves action_dim,
    action_chunk, use_robot_state and the tactile switches to the CLI, and its own defaults
    are for a different checkpoint family. Take them from the checkpoint and refuse a CLI
    value that contradicts it -- a 62-d head served as 31-d loads with "shape mismatch"
    on the action head and then runs.
    """
    ta = json.loads((ckpt / "training_args.json").read_text())
    for key in ("action_dim", "action_chunk", "use_robot_state",
                "use_tactile_deform", "use_tactile_vec"):
        saved = int(ta[key])
        got = int(getattr(args, key))
        if got != saved:
            raise SystemExit(f"--{key} {got} contradicts training_args.json ({saved}). "
                             f"Drop the flag; it is read from the checkpoint.")


# ---------------------------------------------------------------- inference
class TrexPolicy:
    """One `CascadedServer`, driven with in-memory arrays instead of PNG bytes."""

    def __init__(self, args, trex_test):
        self.args = args
        self.trex = trex_test
        model, processor, statistic = trex_test.model_load(args)
        if getattr(args, "fast_processor", 1):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                os.path.join(args.checkpoint_path, "processor"), trust_remote_code=True,
                use_fast=True)
            logger.info("image processor: %s", type(processor.image_processor).__name__)
        self.server = trex_test.CascadedServer(args, model, processor, statistic)
        self.device = self.server.device
        self.server.model = self.server.model.to(self.device).eval()
        self.horizon = int(args.action_chunk)
        self.action_dim = int(args.action_dim)
        self.window = int(self.server.vqvae_window)
        self.n_infer = 0

    # -- payload validation: every rejection names the key and what was expected --
    def _image(self, payload, key):
        from PIL import Image

        if key not in payload:
            raise ValueError(f"payload has no '{key}'")
        img = np.asarray(payload[key])
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError(f"'{key}' must be (H, W, 3) uint8 RGB, got {img.shape} {img.dtype}")
        return Image.fromarray(img, mode="RGB")

    def _f6(self, payload):
        if "tactile_f6" not in payload:
            raise ValueError("payload has no 'tactile_f6' (T, 5, 6)")
        f6 = np.asarray(payload["tactile_f6"], dtype=np.float32)
        if f6.ndim != 3 or f6.shape[1:] != (N_LEFT, F6_DIM):
            raise ValueError(f"'tactile_f6' must be (T, 5, 6), got {f6.shape}")
        if not np.isfinite(f6).all():
            raise ValueError("'tactile_f6' has non-finite values")
        return np.concatenate([f6, np.zeros_like(f6)], axis=1)          # (T, 10, 6)

    def _deform(self, payload):
        if "tactile_deform" not in payload:
            raise ValueError("payload has no 'tactile_deform' (5, 240, 240) uint8")
        d = np.asarray(payload["tactile_deform"])
        if d.shape != (N_LEFT, DEFORM_HW, DEFORM_HW) or d.dtype != np.uint8:
            raise ValueError(f"'tactile_deform' must be (5, {DEFORM_HW}, {DEFORM_HW}) uint8, "
                             f"got {d.shape} {d.dtype}")
        return np.concatenate([d, np.zeros_like(d)], axis=0)             # (10, 240, 240)

    def infer(self, payload: dict) -> dict:
        import torch

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("payload has no 'prompt'")
        head = self._image(payload, "head")
        wrist = self._image(payload, "wrist")
        f6 = self._f6(payload)
        deform = self._deform(payload)
        state = payload.get("state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if state.shape[0] != self.action_dim:
                raise ValueError(f"'state' must be ({self.action_dim},), got {state.shape}")

        srv = self.server
        t0 = time.monotonic()
        with srv.lock, torch.inference_mode():
            # One wrist camera on this rig, two fast slots in the model: training wrote the
            # same left-wrist frame into both (gen_json_openarm_sharpa_left.py, image_old_fast).
            srv._run_slow(prompt, [head], [wrist, wrist], f6, deform, state)
            torch.cuda.synchronize()
            t_slow = time.monotonic()
            actions, chunk_id = srv._run_fast(f6, deform)
            torch.cuda.synchronize()
        t1 = time.monotonic()
        infer_ms = (t1 - t0) * 1e3
        slow_ms, fast_ms = (t_slow - t0) * 1e3, (t1 - t_slow) * 1e3
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.shape != (self.horizon, self.action_dim):
            raise RuntimeError(f"model returned {chunk.shape}, expected "
                               f"({self.horizon}, {self.action_dim})")
        self.n_infer += 1
        return {"actions": chunk, "chunk_id": int(chunk_id),
                "server_timing": {"infer_ms": infer_ms, "slow_ms": slow_ms, "fast_ms": fast_ms}}

    def warm_up(self) -> float:
        rng = np.random.default_rng(0)
        payload = {
            "prompt": self.args.prompt_hint,
            "head": rng.integers(0, 256, size=(400, 640, 3), dtype=np.uint8),
            "wrist": rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8),
            "tactile_f6": np.zeros((self.window, N_LEFT, F6_DIM), np.float32),
            "tactile_deform": np.zeros((N_LEFT, DEFORM_HW, DEFORM_HW), np.uint8),
        }
        out = self.infer(payload)
        return out["server_timing"]["infer_ms"]

    def metadata(self) -> dict:
        return {
            "model": "trex",
            "checkpoint": str(pathlib.Path(self.args.checkpoint_path).resolve()),
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "f6_window": self.window,
            "image_size": list(self.args.image_size),
            "prompt_hint": self.args.prompt_hint,
            "random_weights": bool(self.args.random_weights),
            "cascaded_steps": [int(self.args.cascaded_total_steps),
                               int(self.args.cascaded_split_step)],
        }


# ---------------------------------------------------------------- transport
class WebsocketServer:
    """openpi's WebsocketPolicyServer, re-typed here so this file has no openpi dependency.

    Same handshake (metadata first), same framing (one msgpack-numpy dict per direction),
    same error path (traceback as a text frame, then close). The forward pass runs on a
    worker thread so /healthz and a second client's handshake are not blocked behind it;
    `CascadedServer.lock` serialises the GPU.
    """

    def __init__(self, policy: TrexPolicy, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = port
        self._lock = threading.Lock()

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as ws_server

        async with ws_server.serve(self._handler, self.host, self.port, compression=None,
                                   max_size=None, process_request=_health) as server:
            logger.info("serving on ws://%s:%d  (metadata: %s)", self.host, self.port,
                        self.policy.metadata())
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets

        logger.info("client %s connected", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata()))
        while True:
            try:
                payload = msgpack_numpy.unpackb(await websocket.recv())
                t0 = time.monotonic()
                result = await asyncio.to_thread(self.policy.infer, payload)
                result["server_timing"]["total_ms"] = (time.monotonic() - t0) * 1e3
                await websocket.send(packer.pack(result))
                if self.policy.n_infer % 10 == 0:
                    logger.info("%d chunks served, last infer %.0f ms",
                                self.policy.n_infer, result["server_timing"]["infer_ms"])
            except websockets.ConnectionClosed:
                logger.info("client %s disconnected", websocket.remote_address)
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="see the previous frame")
                logger.exception("request failed")
                return


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ckpt = pathlib.Path(args.checkpoint_path).expanduser().resolve()
    args.checkpoint_path = str(ckpt)
    check_checkpoint(ckpt, bool(args.random_weights))
    reconcile_with_training_args(args, ckpt)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda)
    args.cuda = "0"     # model_load builds f"cuda:{args.cuda}" after the env var masks the rest

    trex_test = _load_trex_test()
    if args.random_weights:
        logger.warning("--random_weights 1: model.pt is NOT loaded. Outputs are garbage; "
                       "only the timing means anything.")
    logger.info("loading %s", ckpt)
    policy = TrexPolicy(args, trex_test)
    ms = policy.warm_up()
    logger.info("warm-up slow_and_fast: %.0f ms (the first call compiles; run "
                "hardware_code/openarm/ping_trex.py for a steady-state number)", ms)
    ms = policy.warm_up()
    logger.info("second warm-up: %.0f ms", ms)
    WebsocketServer(policy, args.host, args.port).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
