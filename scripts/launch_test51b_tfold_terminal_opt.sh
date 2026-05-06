#!/bin/bash
# test51b: Pure tFold with Terminal-Only Reward (Optimized)
# Key changes from test51: n_decoys=1, n_steps=32, n_envs=4, checkpoint=100K
# Hypothesis: Terminal reward + single decoy enables practical tFold training in ~4 days
# Estimated: 2M steps, ~3.5 days at 85% cache hit rate

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/test51b.yaml \
    --run_name test51b_tfold_terminal_opt \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 4 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 4 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 1 \
    --contrastive_agg mean \
    --w_naturalness 0.1 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db \
    > logs/test51b_tfold_terminal_opt_train.log 2>&1 &

echo "PID: $!"
echo "test51b launched on GPU 4"
echo "Monitor: tail -f logs/test51b_tfold_terminal_opt_train.log"
