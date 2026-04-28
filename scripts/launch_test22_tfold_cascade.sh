#!/bin/bash
# test22: ERGO+tFold cascade scorer with uncertainty-gated arbitration
# Expected: Better reward signal quality than pure ERGO, faster than pure tFold
# Cascade threshold: 0.15 (ERGO std > 0.15 triggers tFold)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test22_tfold_cascade \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer tfold_cascade \
    --cascade_threshold 0.15 \
    --tfold_cache_only \
    --tfold_cache_miss_score 0.0 \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > logs/test22_tfold_cascade_train.log 2>&1 &

echo "test22_tfold_cascade launched on GPU 2"
echo "Monitor: tail -f logs/test22_tfold_cascade_train.log"
echo "Check cascade telemetry in log for tFold call percentage"
