#!/bin/bash
# Verify a converted OpenArm dataset before spending a GPU on it.
#
#   sbatch tools/verify_openarm_json.sbatch
#
#SBATCH -J oa_verify
#SBATCH -A hankyang_lab
#SBATCH -p test
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 0-00:30:00
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/oa_verify_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/oa_verify_%j.err
set -uo pipefail

R=/n/netscratch/ydu_lab/Lab/hangxing
PROJECT_ROOT=$R/code/Tactile/T-Rex
PY=$R/Downloads/envs/trex/bin/python
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

JSON=$R/data/trex_posttrain/pick_up_the_egg/json/pick_up_the_egg_deltabase_eef_left_stride1_train.json

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="

$PY ${PROJECT_ROOT}/tools/verify_openarm_json.py --json "$JSON" "$@"
rc=$?

echo "--- deform distribution vs the T-Rex reference ---"
# check_our_deform.py reads raw episodes, not the converted PNGs, so it answers a
# different question than verify_openarm_json.py: whether the frozen encoder's
# BatchNorm statistics still transfer at all.
$PY ${PROJECT_ROOT}/tools/check_our_deform.py --help >/dev/null 2>&1 \
    && echo "(run tools/check_our_deform.py separately against the raw episodes)"

echo "verify exit: $rc"
[ $rc -eq 0 ] && echo VERIFY_PASS || echo VERIFY_FAIL
exit $rc
