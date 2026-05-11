#!/bin/bash
# test51c_decoy: Resume from test51c 100K checkpoint with decoy penalty
# Hypothesis: Adding decoy penalty after warm-start improves specificity
# Key config: w_decoy=0.4 (conservative), resume_from=milestone_100000.pt

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Wait for checkpoint to exist
CHECKPOINT="output/test51c_no_decoy_long_ep/checkpoints/milestone_100000.pt"
while [ ! -f "$CHECKPOINT" ]; do
    echo "Waiting for checkpoint: $CHECKPOINT"
    sleep 300  # Check every 5 minutes
done

echo "Checkpoint found! Launching test51c_decoy..."

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/test51c_decoy.yaml \
    --run_name test51c_decoy \
    --seed 42 \
    --reward_mode v2_full \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 4 \
    --w_affinity 1.0 \
    --w_decoy 0.4 \
    --w_naturalness 0.5 \
    --w_diversity 0.2 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db \
    --resume_from "$CHECKPOINT" \
    > logs/test51c_decoy_train.log 2>&1 &

echo "test51c_decoy launched on GPU 3"
echo "Monitor: tail -f logs/test51c_decoy_train.log"
echo "Hypothesis: Decoy penalty (w=0.4) improves specificity without destabilizing training"
