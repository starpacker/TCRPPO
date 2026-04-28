#!/bin/bash
# test31: Two-phase training — warm-start contrastive from strong ERGO binding
# Phase 1 (test22b): 1M steps pure ERGO → R≈1.8 (strong binding)
# Phase 2 (this): resume from test22b 500K ckpt, switch to contrastive_ergo
# Hypothesis: Starting from strong binding makes contrastive margin positive from day 1.
#   Cold-start contrastive (test28/29) stuck at R=-0.3 because policy can't
#   simultaneously learn binding AND specificity. Warm-start solves the cold-start problem.
# Key config: resume from test22b, contrastive_ergo, mean(8 decoys), entropy decay
# Expected: Positive reward from start, eventually best Top1_AUC

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /tmp/tcrppo_v2_code

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config /tmp/tcrppo_config.yaml \
    --run_name test31_twophase_contrastive \
    --seed 42 \
    --resume_from /tmp/tcrppo_v2_code/output/test22b_ergo_only/checkpoints/milestone_500000.pt \
    --resume_change_reward_mode contrastive_ergo \
    --resume_reset_optimizer \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 8 \
    --contrastive_agg mean \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 500000 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > /tmp/logs/test31_twophase_contrastive_train.log 2>&1 &

echo "test31_twophase_contrastive launched on GPU 3, PID: $!"
echo "Monitor: tail -f /tmp/logs/test31_twophase_contrastive_train.log"
echo "Hypothesis: warm-start binding + contrastive → positive reward from start"
