#!/bin/bash
# test28: Contrastive + max-over-16-decoys
# Hypothesis: Using max instead of mean over 16 decoys forces the TCR to have
#             low affinity to ALL decoys (worst-case specificity), not just low average.
#             Combined with more decoys (16 vs 4), this gives a much stronger specificity signal.
# Key config: contrastive_ergo, n_contrast_decoys=16, contrastive_agg=max
# Expected: Best Top1_AUC and Composite, possibly lower binding (specificity-binding tradeoff)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test28_contrastive_max16 \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg max \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > logs/test28_contrastive_max16_train.log 2>&1 &

echo "test28_contrastive_max16 launched on GPU 3, PID: $!"
echo "Monitor: tail -f logs/test28_contrastive_max16_train.log"
echo "Hypothesis: max(16 decoys) → best specificity, worst-case optimization"
