#!/bin/bash
# Post-train T-Rex on the OpenArm + single-left-Sharpa egg data.
#
# Derived from scripts/train.sh, with the NVIDIA-cluster paths replaced by this
# cluster's and the data pointed at utils/gen_json_openarm_sharpa_left.py's output.
#
#   SMOKE=1 sbatch scripts/train_openarm.sbatch    # 1 GPU, 1 epoch, no checkpoints
#   sbatch scripts/train_openarm.sbatch            # the real run
#
# In SMOKE mode the ONE line that must be read in the log is
#
#     Skipped 0 keys with shape mismatch
#
# printed by scripts/train.py when it resumes. Any non-zero count means the 62-D
# padding in the converter is wrong and the midtrain action head is being thrown
# away -- see docs/superpowers/specs/2026-09-09-openarm-sharpa-left-finetune-design.md
# section 2.1. Training will otherwise proceed and look completely normal.
#
#SBATCH -J trex_openarm
#SBATCH -A hankyang_lab
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH -c 32
#SBATCH --mem 256G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/trex_openarm_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/trex_openarm_%j.err
set -uo pipefail

SMOKE=${SMOKE:-0}

R=/n/netscratch/ydu_lab/Lab/hangxing
PROJECT_ROOT=$R/code/Tactile/T-Rex
PY_BIN=$R/Downloads/envs/trex/bin
export PATH=$PY_BIN:$PATH
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export HF_HOME=$R/hf_home
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

CKPT=$R/code/Tactile/ckpt
ORIGIN_MODEL_PATH=$CKPT/Qwen3-VL-2B-Instruct
RESUME_CHECKPOINT=$CKPT/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6
RESUME_SOURCE=midtrain

# The midtrain checkpoint carries both the VQ-VAE and the deform encoder, and
# resume runs AFTER load_deform_encoder_weights (train.py:800 then :823), so
# these two are belt-and-braces: they seed the modules before the checkpoint
# overwrites them. Keeping them means a missing key in the checkpoint surfaces
# as a mismatch rather than as a frozen randomly-initialised encoder.
DEFORM_ENCODER_PATH=$CKPT/sharpa_wave_deform_encoder.pth
VQVAE_CKPT=$CKPT/trex_tactile_vqvae_f6_w16_k64_finger.pt

DATA_JSON=$R/data/trex_posttrain/pick_up_the_egg/json/pick_up_the_egg_deltabase_eef_left_stride1_train.json
OUTPUT_DIR=$R/logs/trex_openarm

EXPERIMENT_NAME=t-rex_posttrain_openarm_egg
RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M)"

# 62, NOT 31. The rig has one hand; the right half of every vector is a constant
# pad. train.py:834 silently drops shape-mismatched keys on resume, so shrinking
# this re-initialises x_embedder / final_layer / final_layer_tactile from the
# midtrain checkpoint while the log still looks clean.
ACTION_DIM=62

# NOTE --save_freq is counted in EPOCHS, not steps (train.py:1222), and the last
# epoch always saves regardless. A large value therefore means "only the final
# checkpoint", which is the opposite of what the name suggests.
#
# NOTE --val_ratio splits by SAMPLE, not by episode (train.py:238-243). At 30 Hz
# a held-out frame's neighbours are in the training set, so this val loss is
# leaked and will track the training loss no matter how hard the model overfits.
# Read it as "is it fitting at all", never as a generalisation signal. Getting a
# real one needs an episode-level split, which the JSON path does not have.
if [ "$SMOKE" = "1" ]; then
    NUM_PROCESSES=1
    N_EPOCHS=1
    SAVE_FREQ=1000000          # last epoch still saves; nothing else does
    TRAIN_BSZ=2
    GRAD_ACCUM=1
    VAL_ARGS="--val_ratio 0 --val_freq 0"
else
    NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l)
    NUM_PROCESSES=${NUM_PROCESSES:-1}
    N_EPOCHS=${N_EPOCHS:-100}
    SAVE_FREQ=${SAVE_FREQ:-5}          # epochs
    # 8 x 4 GPUs x 4 accum = 128 global, which is exactly the effective batch
    # scripts/train.sh post-trains at (16 x 8). Matching it lets --learning_rate
    # stay at the reference 1e-4 instead of being a second thing we changed.
    TRAIN_BSZ=${TRAIN_BSZ:-8}
    GRAD_ACCUM=${GRAD_ACCUM:-4}
    VAL_ARGS="--val_ratio 0.05 --val_freq 200 --max_val_batches 30"
fi
LR=${LR:-1e-4}
# train.sh ships 0. Resuming a midtrain checkpoint means step 0 takes a full-LR
# AdamW step with empty moment estimates, which is the cheapest way to damage
# weights that took six epochs of midtraining to get.
WARMUP=${WARMUP:-0.03}

mkdir -p "$OUTPUT_DIR" "$R/logs"
cd "${PROJECT_ROOT}/scripts" || exit 1

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "SMOKE=$SMOKE  procs=$NUM_PROCESSES  epochs=$N_EPOCHS  bsz=$TRAIN_BSZ x accum=$GRAD_ACCUM"
echo "global batch = $((TRAIN_BSZ * NUM_PROCESSES * GRAD_ACCUM))  lr=$LR  warmup=$WARMUP  save every $SAVE_FREQ epoch(s)"
echo "data:   $DATA_JSON"
echo "resume: $RESUME_CHECKPOINT"
nvidia-smi -L

for f in "$DATA_JSON" "${DATA_JSON%.json}_statistics.json"; do
    [ -f "$f" ] || { echo "MISSING $f -- run utils/gen_json_openarm_sharpa_left.sbatch first"; exit 1; }
done

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port ${MASTER_PORT:-29500} \
    train.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_format json \
    --data_path ${DATA_JSON} \
    --n_epochs ${N_EPOCHS} \
    --save_freq ${SAVE_FREQ} \
    --action_dim ${ACTION_DIM} \
    --action_chunk 16 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --warmup_rates ${WARMUP} \
    --weight_decay 0 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${RUN_NAME} \
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 1 \
    --use_tactile_vqvae 1 \
    --vqvae_ckpt ${VQVAE_CKPT} \
    --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --cascaded_total_steps 10 \
    --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
    --resume_checkpoint ${RESUME_CHECKPOINT} \
    --resume_source ${RESUME_SOURCE} \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    ${VAL_ARGS}

rc=$?
echo "exit: $rc"
[ $rc -eq 0 ] && echo TRAIN_DONE || echo TRAIN_FAILED
exit $rc
