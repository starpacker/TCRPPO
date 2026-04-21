#!/bin/bash
# test24: Large batch (n_envs=32) with seed=123
# Hypothesis: larger batch reduces gradient variance, stabilizes failed seed=123
# Key config: seed=123, n_envs=32, ESM-2, ban_stop, v1_ergo_only

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test24_large_batch \
    --seed 123 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --total_timesteps 2000000 \
    --n_envs 32 \
    --device cuda \
    > logs/test24_large_batch_train.log 2>&1 &

echo "test24_large_batch launched on GPU 1"
echo "Monitor: tail -f logs/test24_large_batch_train.log"
echo "Hypothesis: n_envs=32 stabilizes seed=123 (previously AUROC=0.5462)"
