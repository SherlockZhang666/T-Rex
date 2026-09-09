#!/bin/bash
# OpenArm rt-v1 episodes -> T-Rex post-train JSON.  CPU only; the cost is video
# decode, so cores matter and GPUs do not.
#
#   sbatch utils/gen_json_openarm_sharpa_left.sbatch
#   sbatch utils/gen_json_openarm_sharpa_left.sbatch --limit 2   # quick trial
#
#SBATCH -J oa_genjson
#SBATCH -A hankyang_lab
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH -t 0-06:00:00
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/oa_genjson_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/oa_genjson_%j.err
set -uo pipefail

R=/n/netscratch/ydu_lab/Lab/hangxing
PROJECT_ROOT=$R/code/Tactile/T-Rex
PY=$R/Downloads/envs/trex/bin/python
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

DATA_ROOT=$R/data/tactile-steering-data/pick_up_the_egg
MOUNT_GEOM=$R/code/Tactile/openarm-sharpa-sim/openarm_sharpa/assets/mount_geom.py

OUT=$R/data/trex_posttrain/pick_up_the_egg
IMG_ROOT=$OUT/images
JSON_ROOT=$OUT/json
TASK_NAME=pick_up_the_egg
JSON_BASE=pick_up_the_egg_deltabase_eef_left_stride1_train
INSTRUCTION="Pick up the egg with your left hand and place it down."

mkdir -p "$IMG_ROOT" "$JSON_ROOT" "$R/logs"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "start: $(date)"

$PY ${PROJECT_ROOT}/utils/gen_json_openarm_sharpa_left.py \
    --data_root      "$DATA_ROOT" \
    --img_save_root  "$IMG_ROOT" \
    --json_save_root "$JSON_ROOT" \
    --task_name      "$TASK_NAME" \
    --json_name_base "$JSON_BASE" \
    --instruction    "$INSTRUCTION" \
    --mount_geom     "$MOUNT_GEOM" \
    --side left \
    --action_chunk 16 \
    --frame_stride 1 \
    --num_workers 16 \
    "$@"

rc=$?
echo "exit: $rc"
echo "end: $(date)"
[ $rc -eq 0 ] && echo GENJSON_DONE || echo GENJSON_FAILED
exit $rc
