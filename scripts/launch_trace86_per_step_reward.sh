#!/bin/bash
# trace86: Test per-step reward + longer episodes
# Hypothesis: Terminal reward only causes credit assignment problem
# Key config: terminal_reward_only=false, max_steps=12

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Start tFold server first
echo "Starting tFold server for trace86..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace86.sock \
    --gpu 0 \
    --use-amp-wrapper \
    > logs/tfold_server_trace86.log 2>&1 &

TFOLD_PID=$!
echo "tFold server started (PID: $TFOLD_PID)"
sleep 5

# Launch training
echo "Launching trace86 training..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace86_per_step_reward.yaml \
    --run_name trace86_per_step_reward \
    --seed 86 \
    > logs/trace86_per_step_reward_train.log 2>&1 &

TRAIN_PID=$!
echo "Training started (PID: $TRAIN_PID)"
echo ""
echo "=== trace86 Launch Summary ==="
echo "Hypothesis: Per-step reward solves credit assignment problem"
echo "Key changes:"
echo "  - terminal_reward_only: false (per-step delta reward)"
echo "  - max_steps: 12 (vs 8)"
echo "  - curriculum: 50% L0 → 10% L0 (start from good TCRs)"
echo ""
echo "Expected outcome:"
echo "  - Visible improvement in 5K-10K episodes"
echo "  - Positive rate >5% within 10K episodes"
echo "  - Mean A reaches -1.0 within 50K episodes"
echo ""
echo "Monitor:"
echo "  tail -f logs/trace86_per_step_reward_train.log"
echo ""
echo "PIDs:"
echo "  tFold server: $TFOLD_PID"
echo "  Training: $TRAIN_PID"
