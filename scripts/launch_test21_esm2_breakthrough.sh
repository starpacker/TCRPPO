#!/bin/bash
# test21: ESM-2 breakthrough experiment
# - ESM-2 encoder (1280d) with pMHC warmup cache
# - v1_ergo_shaped reward (terminal + intermediate shaped)
# - Enhanced curriculum (L0→L1→L2 gradual)
# - ban_stop (force 8 steps)
# Goal: Break through 0.80 AUROC ceiling with stable, seed-independent performance

cd /share/liuyutian/tcrppo_v2

# Use tcrppo_v2 conda env which has ESM-2 installed
PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test21_esm2_breakthrough \
    --seed 42 \
    --reward_mode v1_ergo_shaped \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > logs/test21_esm2_breakthrough_train.log 2>&1 &

echo "Launched test21_esm2_breakthrough on GPU 3"
echo "PID: $!"
echo "Monitor: tail -f logs/test21_esm2_breakthrough_train.log"
echo ""
echo "Key features:"
echo "  - ESM-2 1280d encoder (vs lightweight 256d)"
echo "  - v1_ergo_shaped reward (terminal + 0.1*delta intermediate)"
echo "  - Enhanced curriculum: 0-500K pure L0, 500K-1M L0+L1, 1M-2M L0+L1+L2"
echo "  - ban_stop: forced 8 steps per episode"
echo "  - pMHC warmup: 163 targets pre-cached"
