#!/bin/bash
# Post-train T-Rex on the OpenArm + single-left-Sharpa egg data.
#
# Derived from scripts/train.sh, with the NVIDIA-cluster paths replaced by this
# cluster's and the data pointed at utils/gen_json_openarm_sharpa_left.py's output.
#
#   SMOKE=1 sbatch scripts/train_openarm.sbatch    # 1 epoch, no checkpoints
#                                                  (still needs >= 2 GPUs, see below)
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
# The OOM traceback reported 14.6 GB reserved-but-unallocated; AdamW's lazy
# state init asks for three ~14 GB blocks in a row, which fragments badly.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# DeepSpeed probes the CUDA version by shelling out to $CUDA_HOME/bin/nvcc while
# `import deepspeed` is still running (op_builder/builder.py:53). CUDA_HOME is
# unset on these nodes and there is no /usr/local/cuda, so the import raises
# FileNotFoundError before accelerate has launched anything.
#
# Set CUDA_HOME rather than `module load cuda`: the probe only needs the nvcc
# binary to exist, and the module also prepends its lib64 to LD_LIBRARY_PATH,
# which would shadow the cu124 libraries torch ships in site-packages/nvidia/*.
# 12.4 to match torch 2.6.0+cu124.
export CUDA_HOME=${CUDA_HOME:-/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda}
export PATH=$PATH:$CUDA_HOME/bin
[ -x "$CUDA_HOME/bin/nvcc" ] || {
    echo "no nvcc at $CUDA_HOME/bin/nvcc -- deepspeed will fail on import."
    echo "pick one from: module avail cuda"
    exit 1
}

# Keeps DeepSpeed's Triton autotune cache off NFS (it warns, and can hang on exit).
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_${SLURM_JOB_ID:-$$}}
mkdir -p "$TRITON_CACHE_DIR"

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
NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NUM_PROCESSES:-0}" -ge 1 ] || { echo "no GPUs visible"; exit 1; }

# This model is NOT 2B. The MoT architecture triples the language layers into
# latent / action / tactile experts, so model.pt is 8.5 GB of bf16 = 4.25 B
# parameters, ~3.84 B of them trainable. AdamW keeps 12 bytes per trainable
# parameter (fp32 exp_avg + exp_avg_sq + master weight) = ~43 GB of optimizer
# state, and ZeRO-2's only way to fit that is to shard it across ranks.
#
# So a single-GPU run cannot work at ANY batch size: 43 GB of optimizer plus
# 8.5 GB of weights plus gradients overruns an 80 GB card before the first
# forward pass. It OOMs inside AdamW's lazy state init on step 1, which reads
# like a batch-size problem and is not one. Smoke runs get every GPU too.
if [ "$NUM_PROCESSES" -lt 2 ]; then
    echo "ERROR: $NUM_PROCESSES GPU(s). ZeRO-2 shards the ~43 GB of AdamW state"
    echo "across ranks; with one rank there is no sharding and it cannot fit."
    echo "Allocate at least 2 (4 is the tested configuration)."
    exit 1
fi

if [ "$SMOKE" = "1" ]; then
    N_EPOCHS=1
    SAVE_FREQ=1000000          # last epoch still saves; nothing else does
    TRAIN_BSZ=${TRAIN_BSZ:-2}
    GLOBAL_BATCH=$((TRAIN_BSZ * NUM_PROCESSES))    # accum 1, get to a step fast
    VAL_ARGS="--val_ratio 0 --val_freq 0"
else
    # 30, not train.sh's 100. Measured on 4 x A100: 32 min per epoch, so 100
    # epochs is ~53 h. The smoke run showed why that is not worth it -- ONE
    # annealed epoch already takes the action loss from 1.131 to 0.0077, on 38
    # demonstrations of a single task from a midtrain checkpoint. train.sh's 100
    # was set for a much larger post-train corpus.
    #
    # 30 epochs is ~16 h, fits the -t below with margin, and --save_freq 5 leaves
    # six checkpoints. Which of them is best is a rollout question, not a loss
    # question: on data this small the training loss will keep falling long after
    # the policy has stopped improving, and the val loss cannot arbitrate (see
    # the --val_ratio note above).
    N_EPOCHS=${N_EPOCHS:-30}
    SAVE_FREQ=${SAVE_FREQ:-5}          # epochs
    # scripts/train.sh post-trains at 16 x 8 GPUs = 128 global. Hitting the same
    # effective batch by accumulation lets --learning_rate stay at the reference
    # 1e-4 instead of being a second thing changed at once.
    #
    # 2 per GPU is what the smoke run actually fit. ZeRO-2 replicates the bf16
    # weights on every rank (8.5 GB) and shards optimizer + gradients, so a rank
    # holds ~22 GB before any activation, and the forward carries 3 images at
    # 384x288, 8 FLARE frames and 10 deform maps through a ResNet.
    #
    # Accumulation costs no wall time here -- the same number of forward and
    # backward passes either way -- so raising this only buys ~15 % from better
    # GPU utilisation, which is not worth an OOM 10 minutes into a 16 h run.
    # TRAIN_BSZ=4 probably fits; test it with SMOKE=1 before trusting it.
    TRAIN_BSZ=${TRAIN_BSZ:-2}
    GLOBAL_BATCH=${GLOBAL_BATCH:-128}
    VAL_ARGS="--val_ratio 0.05 --val_freq 200 --max_val_batches 30"
fi

PER_STEP=$((TRAIN_BSZ * NUM_PROCESSES))
GRAD_ACCUM=$((GLOBAL_BATCH / PER_STEP))
[ "$GRAD_ACCUM" -ge 1 ] || GRAD_ACCUM=1
if [ $((GRAD_ACCUM * PER_STEP)) -ne "$GLOBAL_BATCH" ]; then
    echo "WARNING: global batch $GLOBAL_BATCH is not $TRAIN_BSZ x $NUM_PROCESSES x an integer;"
    echo "         using accum $GRAD_ACCUM -> effective $((GRAD_ACCUM * PER_STEP))"
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
