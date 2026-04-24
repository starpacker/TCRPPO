#!/bin/bash
# test28: ERGO with 12 Steps
# Hypothesis: Longer trajectories (12 steps) will improve performance
# Key config: affinity_scorer=ergo, max_steps=12, ESM-2, curriculum, ban_stop

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test28_ergo_12steps \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 12 \
    --ban_stop \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3 \
    > logs/test28_ergo_12steps_train.log 2>&1 &

echo "test28_ergo_12steps launched on GPU 1"
echo "Monitor: tail -f logs/test28_ergo_12steps_train.log"
echo "Hypothesis: max_steps=12 will improve over test26 (max_steps=8, AUROC 0.5027)"
