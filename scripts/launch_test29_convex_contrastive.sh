#!/bin/bash
# test29: Convex contrastive — combines ERGO^3 with max-over-decoys
# Hypothesis: Best of both worlds — convex reward pushes for exceptional binding,
#             max-decoy ensures specificity. Entropy decay lets policy concentrate.
# reward = ERGO(target)^3 - max(ERGO(decoys))^3
# Key config: contrastive_ergo + convex_alpha=3 + max + entropy_decay
# Expected: Highest Composite score

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test29_convex_contrastive \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg max \
    --convex_alpha 3.0 \
    --entropy_coef_final 0.005 \
    --entropy_decay_start 800000 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > logs/test29_convex_contrastive_train.log 2>&1 &

echo "test29_convex_contrastive launched on GPU 4, PID: $!"
echo "Monitor: tail -f logs/test29_convex_contrastive_train.log"
echo "Hypothesis: ERGO^3 + max(16 decoys) + entropy decay → best Composite"
