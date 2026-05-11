#!/bin/bash
# test51c: No Decoy + Longer Episodes + V2 Full Reward
# Hypothesis: Removing decoy variance and extending episodes will stabilize training
# Key config: max_steps=8, n_contrast_decoys=0, reward_mode=v2_no_decoy

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/test51c.yaml \
    --run_name test51c_no_decoy_long_ep \
    --seed 42 \
    --reward_mode v2_no_decoy \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 0 \
    --w_affinity 1.0 \
    --w_naturalness 0.5 \
    --w_diversity 0.2 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db \
    > logs/test51c_no_decoy_long_ep_train.log 2>&1 &

echo "test51c_no_decoy_long_ep launched on GPU 1"
echo "Monitor: tail -f logs/test51c_no_decoy_long_ep_train.log"
echo "Hypothesis: No decoy variance + longer episodes = stable reward growth"
