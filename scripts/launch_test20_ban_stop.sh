#!/bin/bash
# Test20: v2_full reward mode with STOP action banned
# Agent must use all max_steps (8 steps) per episode
# Using lightweight encoder (CPU-friendly BiLSTM) to avoid ESM dependency

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=0 nohup python -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test20_ban_stop \
    --seed 42 \
    --reward_mode v2_full \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder lightweight \
    --encoder_dim 256 \
    --ban_stop \
    --device cuda \
    > logs/test20_ban_stop_train.log 2>&1 &

echo "Launched test20_ban_stop on GPU 0"
echo "PID: $!"
echo "Monitor: tail -f logs/test20_ban_stop_train.log"
