#!/bin/bash
# test34: Warm-start contrastive from test27's concentrated policy
# Phase 1 (test27): 2M steps convex ERGO^3 + entropy decay → R≈0.56 (concentrated binding)
# Phase 2 (this): resume from test27 final, switch to contrastive_ergo
# Hypothesis: test27's concentrated policy (Ent=3.58) already focuses on exceptional TCRs.
#   Adding contrastive reward should quickly add specificity to these high-quality sequences.
#   Unlike test31/32 (from test22b 500K, Ent≈4.5), test34 starts from a more focused policy.
# Key config: resume from test27 final, contrastive_ergo, mean(8 decoys), low LR

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /tmp/tcrppo_v2_code

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config /tmp/tcrppo_config.yaml \
    --run_name test34_from_convex_contrastive \
    --seed 42 \
    --resume_from /tmp/tcrppo_v2_code/output/test27_convex_entdecay/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --resume_reset_optimizer \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 8 \
    --contrastive_agg mean \
    --learning_rate 1e-4 \
    --entropy_coef_final 0.005 \
    --entropy_decay_start 100000 \
    --total_timesteps 1500000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > /tmp/logs/test34_from_convex_contrastive_train.log 2>&1 &

echo "test34_from_convex_contrastive launched on GPU 1, PID: $!"
echo "Monitor: tail -f /tmp/logs/test34_from_convex_contrastive_train.log"
echo "Hypothesis: concentrated policy + contrastive → best Top-1 specificity"
