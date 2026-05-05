#!/bin/bash
# test41_seed123: Reproduce test41 with seed=123
# Hypothesis: Validate test41's 0.6243 AUROC is seed-stable
# Key config: Same as test41 but seed=123

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=5 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test41_seed123 \
    --seed 123 \
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
    > logs/test41_seed123_train.log 2>&1 &

echo "test41_seed123 launched on GPU 5"
echo "Monitor: tail -f logs/test41_seed123_train.log"
echo "Hypothesis: Reproduce test41's 0.6243 AUROC with seed=123"
