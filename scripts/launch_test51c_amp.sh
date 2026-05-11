#!/bin/bash
# test51c_amp: Resume from test51c 100K checkpoint with AMP-accelerated tFold
# Hypothesis: AMP provides 4× faster training without quality regression
# Key config: affinity_model=tfold_amp, resume_from=milestone_100000.pt

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Wait for checkpoint to exist
CHECKPOINT="output/test51c_no_decoy_long_ep/checkpoints/milestone_100000.pt"
while [ ! -f "$CHECKPOINT" ]; do
    echo "Waiting for checkpoint: $CHECKPOINT"
    sleep 300  # Check every 5 minutes
done

echo "Checkpoint found! Launching test51c_amp..."

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/test51c_amp.yaml \
    --run_name test51c_amp \
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
    --resume_from "$CHECKPOINT" \
    > logs/test51c_amp_train.log 2>&1 &

echo "test51c_amp launched on GPU 2"
echo "Monitor: tail -f logs/test51c_amp_train.log"
echo "Hypothesis: 4× faster training (AMP) with same quality as test51c"
