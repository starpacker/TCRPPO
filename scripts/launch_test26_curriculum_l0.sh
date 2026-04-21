#!/bin/bash
# test26: L0-heavy curriculum (L0=50%, L1=20%, L2=30%)
# Hypothesis: starting from known binder variants reduces wasted exploration
# Key config: curriculum_l0=0.5, ESM-2, ban_stop, v1_ergo_only, seed=42

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test26_curriculum_l0 \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3 \
    --device cuda \
    > logs/test26_curriculum_l0_train.log 2>&1 &

echo "test26_curriculum_l0 launched on GPU 4"
echo "Monitor: tail -f logs/test26_curriculum_l0_train.log"
echo "Hypothesis: L0=50% curriculum improves early training vs pure L2 random"
