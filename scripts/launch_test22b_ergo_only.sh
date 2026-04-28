#!/bin/bash
# test22b: Pure ERGO baseline (no cascade) for comparison
# Same config as test22 but without tFold cascade overhead

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test22b_ergo_only \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > logs/test22b_ergo_only_train.log 2>&1 &

echo "test22b_ergo_only launched on GPU 3"
echo "Monitor: tail -f logs/test22b_ergo_only_train.log"
