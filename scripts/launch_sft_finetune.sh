#!/bin/bash
# Launch RL fine-tuning from SFT checkpoint
#
# This script loads the SFT-trained policy and fine-tunes it with PPO + online pool.
# Target: push mean affinity from ~-0.5 to 0.0

set -e

# Configuration
SFT_CHECKPOINT="${1:-output/sft_training/checkpoint_best.pt}"
OUTPUT_DIR="${2:-output/sft_finetune}"
GPU="${3:-0}"

# Hyperparameters (conservative for fine-tuning)
TOTAL_STEPS=1000000
N_ENVS=8
LR=3e-5  # Lower than SFT (1e-4)
HIDDEN_DIM=512

# Online pool settings
ONLINE_POOL_MIN_AFFINITY=-0.5  # Only keep good TCRs
ONLINE_POOL_MAX_SIZE=1000

# Reward weights (same as trace73)
AFFINITY_WEIGHT=1.0
DECOY_WEIGHT=0.0
NATURALNESS_WEIGHT=0.0
DIVERSITY_WEIGHT=0.0

# Target peptides (use high-quality subset)
PEPTIDES="GILGFVFTL,NLVPMVATV,GLCTLVAML,LLWNGPMAV,YLQPRTFLL"

echo "=== RL Fine-tuning from SFT Checkpoint ==="
echo "SFT checkpoint: $SFT_CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "Total steps: $TOTAL_STEPS"
echo "Learning rate: $LR (conservative)"
echo "Online pool min affinity: $ONLINE_POOL_MIN_AFFINITY"
echo ""

# Check SFT checkpoint exists
if [ ! -f "$SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found: $SFT_CHECKPOINT"
    echo "Please run train_sft.py first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
CUDA_VISIBLE_DEVICES=$GPU python -u tcrppo_v2/ppo_trainer.py \
    --run_name sft_finetune \
    --output_dir "$OUTPUT_DIR" \
    --resume_from "$SFT_CHECKPOINT" \
    --total_timesteps $TOTAL_STEPS \
    --n_envs $N_ENVS \
    --learning_rate $LR \
    --hidden_dim $HIDDEN_DIM \
    --max_steps 8 \
    --reward_mode terminal \
    --affinity_scorer tfold \
    --encoder esm2 \
    --use_delta_reward \
    --use_znorm \
    --affinity_weight $AFFINITY_WEIGHT \
    --decoy_weight $DECOY_WEIGHT \
    --naturalness_weight $NATURALNESS_WEIGHT \
    --diversity_weight $DIVERSITY_WEIGHT \
    --use_online_pool \
    --online_pool_min_affinity $ONLINE_POOL_MIN_AFFINITY \
    --online_pool_max_size $ONLINE_POOL_MAX_SIZE \
    --curriculum_mode L2 \
    --target_peptides "$PEPTIDES" \
    --eval_interval 50000 \
    --save_interval 100000 \
    --log_interval 100 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=== Training Complete ==="
echo "Logs: $OUTPUT_DIR/train.log"
echo "Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR/logs"
