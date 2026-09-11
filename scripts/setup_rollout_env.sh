#!/usr/bin/env bash
# Build the inference venv for serving T-Rex on the rollout laptop.
#
#   scripts/setup_rollout_env.sh            # creates <repo>/.venv
#   VENV=/elsewhere scripts/setup_rollout_env.sh
#
# Why not `uv sync` against pyproject.toml: that manifest pins torch 2.6.0+cu124, which has
# no sm_120 kernels. The rollout machine is an RTX 5090 Laptop (Blackwell, sm_120), so
# torch must come from the cu128 index -- 2.7.x is the first release that ships them. The
# model code only needs torch/torchvision/transformers/timm at inference time; DeepSpeed,
# datasets, wandb and the rest of the training stack are deliberately left out.
set -euo pipefail
# The rollout laptop's uplink is slow; uv's default 30 s per-request timeout trips on the
# 300 MB CUDA wheels. Resumable, so re-running after a timeout picks up where it stopped.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$ROOT/.venv}
PY=${PYTHON_VERSION:-3.10}

command -v uv >/dev/null || { echo "uv not found; install from https://docs.astral.sh/uv/" >&2; exit 1; }

uv venv --allow-existing --python "$PY" "$VENV"
# shellcheck disable=SC1091
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1" "torchvision==0.22.1"
uv pip install --python "$VENV/bin/python" \
    "transformers==4.57.3" "tokenizers==0.22.2" "accelerate==1.8.1" \
    "safetensors==0.5.3" "numpy==2.2.6" "pillow==12.1.1" \
    "opencv-python-headless==4.12.0.88" "timm==1.0.17" "h5py==3.16.0" \
    "matplotlib==3.10.3" "pyzmq==27.1.0" "tqdm==4.67.1" \
    "websockets>=12.0" "msgpack>=1.0.5" "pytest"

"$VENV/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "archs", torch.cuda.get_arch_list()[-2:] if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("device", torch.cuda.get_device_name(0), "sm_%d%d" % cap)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok:", bool(torch.isfinite(x @ x).all()))
PY
echo "venv ready: $VENV"
