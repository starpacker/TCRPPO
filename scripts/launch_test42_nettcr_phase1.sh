#!/bin/bash
# test42 Phase 1: Pure NetTCR warm-start (equivalent to test22b but with NetTCR)
# Hypothesis: NetTCR can provide comparable warm-start quality to ERGO
# Key config: affinity_scorer=nettcr, reward_mode=v1_ergo_only (pure affinity)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test42_nettcr_phase1 \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3 \
    --device cuda \
    > logs/test42_nettcr_phase1_train.log 2>&1 &

echo "test42_nettcr_phase1 launched on GPU 2"
echo "Monitor: tail -f logs/test42_nettcr_phase1_train.log"
echo "Hypothesis: NetTCR warm-start for 2M steps to match test22b's R≈2.05"
