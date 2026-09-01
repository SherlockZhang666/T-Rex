#!/bin/bash
#SBATCH --job-name=deform_ae
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/hangxing/logs/deform_ae_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/hangxing/logs/deform_ae_%j.err

set -euo pipefail

# ---------------- 可编辑区 ----------------
REPO="/n/netscratch/ydu_lab/Lab/hangxing/code/Tactile/T-Rex"
SHARDS="/n/netscratch/ydu_lab/Lab/hangxing/data/deform_frames"
OUT="/n/netscratch/ydu_lab/Lab/hangxing/code/Tactile/ckpt/my_sharpa_wave_deform_encoder.pth"
EPOCHS=30
BATCH=256
LR=3e-4             # 1e-3 会在第 1 轮把 decoder 的 ReLU 打死（job 39302558）
WARMUP=500
DECODER_ACT=leaky   # decoder 训完就扔，换激活不影响 encoder 兼容性
MAG_ALPHA=10        # 接触像素只占 ~1%，朴素 MSE 会奖励"只预测背景"
# -----------------------------------------

module load python
source activate trex

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "start: $(date)"

python "$REPO/tools/train_deform_ae.py" \
    --shard_dir  "$SHARDS" \
    --out        "$OUT" \
    --epochs     "$EPOCHS" \
    --batch_size   "$BATCH" \
    --lr           "$LR" \
    --warmup_steps "$WARMUP" \
    --decoder_act  "$DECODER_ACT" \
    --workers      "$SLURM_CPUS_PER_TASK" \
    --mag_alpha    "$MAG_ALPHA"

echo "=== 验证能否被 VLA 加载 ==="
python "$REPO/tools/verify_deform_encoder.py" --ckpt "$OUT"

echo "end: $(date)"
