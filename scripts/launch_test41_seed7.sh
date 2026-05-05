#!/bin/bash
# test41_seed7: Reproduce test41 with seed=7
# Hypothesis: Third seed validation for test41 robustness
# Key config: Same as test41 but seed=7

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=6 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test41_seed7 \
    --seed 7 \
    --resume_from output/test33_twophase_strong_contrastive/checkpoints/milestone_1000000.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt \
    > logs/test41_seed7_train.log 2>&1 &

echo "test41_seed7 launched on GPU 6"
echo "Monitor: tail -f logs/test41_seed7_train.log"
echo "Hypothesis: Third seed validation for test41 (0.6243)"
