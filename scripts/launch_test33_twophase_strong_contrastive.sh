#!/bin/bash
# test33: Two-phase warm-start contrastive from test22b FINAL (2M steps)
# Phase 1 (test22b): 2M steps pure ERGO → R≈2.05 (best binding ever)
# Phase 2 (this): resume from test22b 2M ckpt, switch to contrastive_ergo
# Hypothesis: Stronger warm-start (2M vs 500K) gives even better contrastive margin.
#   test31 (500K warm-start) reached avg R≈0.40 contrastive margin.
#   test22b's 2M model has R=2.05 binding (vs 1.72 at 500K).
#   Starting from stronger binding should give larger margin from day 1.
# Key config: resume from test22b final, contrastive_ergo, mean(8 decoys), entropy decay
# Also trying lower learning rate (1e-4) for fine-tuning stability

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /tmp/tcrppo_v2_code

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config /tmp/tcrppo_config.yaml \
    --run_name test33_twophase_strong_contrastive \
    --seed 42 \
    --resume_from /tmp/tcrppo_v2_code/output/test22b_ergo_only/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --resume_reset_optimizer \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 8 \
    --contrastive_agg mean \
    --learning_rate 1e-4 \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 100000 \
    --total_timesteps 1500000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > /tmp/logs/test33_twophase_strong_contrastive_train.log 2>&1 &

echo "test33_twophase_strong_contrastive launched on GPU 0, PID: $!"
echo "Monitor: tail -f /tmp/logs/test33_twophase_strong_contrastive_train.log"
echo "Hypothesis: 2M warm-start + contrastive + low LR → best specificity"
