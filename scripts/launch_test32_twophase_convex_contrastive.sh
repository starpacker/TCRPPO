#!/bin/bash
# test32: Two-phase warm-start CONVEX contrastive
# Phase 1 (test22b): 1M steps pure ERGO → R≈1.8 (strong binding)
# Phase 2 (this): resume from test22b 500K ckpt, switch to contrastive_ergo + convex_alpha=3
# Hypothesis: Convex transform amplifies the warm-start contrastive margin.
#   test31 proved warm-start contrastive works (R=0.18 positive from start).
#   Adding convex: reward = ERGO(target)^3 - mean(ERGO(decoys))^3
#   With target ERGO ≈ 0.8 and decoy ≈ 0.5: 0.512 - 0.125 = 0.387 (vs 0.3 raw)
#   Convex amplifies the gap at high binding scores → better Top-1/Top-5
# Key config: resume test22b, contrastive_ergo, mean(8 decoys), convex_alpha=3, entropy decay
# Expected: Higher contrastive margin than test31, best Top-1 quality

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /tmp/tcrppo_v2_code

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config /tmp/tcrppo_config.yaml \
    --run_name test32_twophase_convex_contrastive \
    --seed 42 \
    --resume_from /tmp/tcrppo_v2_code/output/test22b_ergo_only/checkpoints/milestone_500000.pt \
    --resume_change_reward_mode contrastive_ergo \
    --resume_reset_optimizer \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 8 \
    --contrastive_agg mean \
    --convex_alpha 3.0 \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 500000 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > /tmp/logs/test32_twophase_convex_contrastive_train.log 2>&1 &

echo "test32_twophase_convex_contrastive launched on GPU 4, PID: $!"
echo "Monitor: tail -f /tmp/logs/test32_twophase_convex_contrastive_train.log"
echo "Hypothesis: warm-start + convex contrastive → amplified margin for Top-1"
