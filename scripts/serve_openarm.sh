#!/usr/bin/env bash
# Serve an OpenArm + Sharpa-left T-Rex checkpoint to the rollout client.
#
#   scripts/serve_openarm.sh <ckpt-dir> [extra serve.py args...]
#
#   hf download zhx-tactile-steering/T-Rex-openarm-sharpa-left-egg \
#       --include "checkpoint-29-49290/*" --local-dir checkpoints/openarm-sharpa-left-egg
#   scripts/serve_openarm.sh checkpoints/openarm-sharpa-left-egg/checkpoint-29-49290
#   scripts/serve_openarm.sh checkpoints/.../checkpoint-19-32860 --port 8001
#   scripts/serve_openarm.sh checkpoints/.../checkpoint-29-49290 --random_weights 1   # timing only
#
# action_dim / action_chunk / tactile switches / cascaded schedule / VQ-VAE config are all
# read from the checkpoint's training_args.json by serve.py + test.py. The one thing that
# is NOT in training_args.json is the image size, so it is pinned here to what the
# post-train used (design doc §1.4, model card). Passing a different --image_size does not
# fail; it feeds the network a resolution it was not trained at.
set -euo pipefail

CKPT=${1:?usage: serve_openarm.sh <ckpt-dir> [serve.py args...]}
shift

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$ROOT/.venv}
PY=$VENV/bin/python
[[ -x $PY ]] || { echo "no venv at $VENV -- run scripts/setup_rollout_env.sh first" >&2; exit 1; }

CKPT=$(readlink -f "$CKPT")
for f in config.json training_args.json stats_data.json processor model.pt; do
    if [[ ! -e $CKPT/$f ]]; then
        if [[ $f == model.pt && " $* " == *" --random_weights 1 "* ]]; then
            continue
        fi
        echo "missing $CKPT/$f" >&2
        echo "A Hub checkpoint directory has config.json, model.pt (8.5 GB), processor/," >&2
        echo "stats_data.json and training_args.json. Count the files: an --include pattern" >&2
        echo "that matches nothing makes 'hf download' exit 0 without downloading anything." >&2
        exit 1
    fi
done

export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
cd "$ROOT"
exec "$PY" hardware_code/openarm/serve.py \
    --checkpoint_path "$CKPT" \
    --image_size 384 288 \
    "$@"
