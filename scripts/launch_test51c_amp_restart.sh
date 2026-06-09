#!/bin/bash
# test51c_amp: AMP-accelerated tFold with 20K checkpoints
# Restart after test51c stopped at 20,936 steps
# Key changes: affinity_scorer=tfold_amp, checkpoint_interval=20000

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/test51c.yaml \
    --run_name test51c_amp_restart \
    --seed 42 \
    --reward_mode v2_no_decoy \
    --affinity_scorer tfold_amp \
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
    > logs/test51c_amp_restart_train.log 2>&1 &

echo "test51c_amp_restart launched on GPU 1"
echo "Monitor: tail -f logs/test51c_amp_restart_train.log"
echo "Key features:"
echo "  - AMP-accelerated tFold (3.97× faster)"
echo "  - Checkpoint every 20K steps"
echo "  - Same config as test51c (for comparison)"
