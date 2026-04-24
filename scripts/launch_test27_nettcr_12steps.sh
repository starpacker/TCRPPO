#!/bin/bash
# test27: NetTCR with 12 Steps
# Hypothesis: NetTCR scorer + longer trajectories (12 steps) will improve performance
# Key config: affinity_scorer=nettcr, max_steps=12, ESM-2, curriculum, ban_stop

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test27_nettcr_12steps \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 12 \
    --ban_stop \
    --l0_prob 0.5 \
    --l1_prob 0.2 \
    --l2_prob 0.3 \
    > logs/test27_nettcr_12steps_train.log 2>&1 &

echo "test27_nettcr_12steps launched on GPU 1"
echo "Monitor: tail -f logs/test27_nettcr_12steps_train.log"
echo "Hypothesis: NetTCR + 12 steps will improve over test26 (AUROC 0.5027)"
