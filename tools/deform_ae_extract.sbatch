#!/bin/bash
#SBATCH --job-name=deform_extract
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/hangxing/logs/deform_extract_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/hangxing/logs/deform_extract_%j.err

set -euo pipefail

# ---------------- 可编辑区 ----------------
REPO="/n/netscratch/ydu_lab/Lab/hangxing/code/Tactile/T-Rex"
DATASET="/n/netscratch/ydu_lab/Lab/hangxing/data/trex_dataset"
SHARDS="/n/netscratch/ydu_lab/Lab/hangxing/data/deform_frames"
STRIDE=500          # 每 500 帧留 1 帧 -> 约 95k 帧 / 5.5 GB
# -----------------------------------------

module load python
source activate trex

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
echo "start: $(date)"

python "$REPO/tools/extract_deform_frames.py" \
    --dataset_root "$DATASET" \
    --out_dir      "$SHARDS" \
    --stride       "$STRIDE" \
    --workers      "$SLURM_CPUS_PER_TASK"

echo "end: $(date)"
du -sh "$SHARDS"
