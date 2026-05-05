#!/bin/bash
# test49: Cascade Scorer (ERGO pre-filter + tFold verify)
# Hypothesis: Adaptive scoring (fast early, accurate late) improves tFold AUROC
# Key config: Resume from test41, cascade scorer (threshold=0.3, tfold_weight=0.7)

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test49_cascade_ergo_tfold_cacheonly \
    --seed 42 \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --affinity_scorer cascade \
    --cascade_threshold 0.5 \
    --cascade_tfold_weight 0.7 \
    --cascade_ergo_weight 0.3 \
    --tfold_cache_only \
    --tfold_cache_miss_score 0.3 \
    --encoder esm2 \
    --total_timesteps 3500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt \
    > logs/test49_cascade_ergo_tfold_cacheonly_train.log 2>&1 &

echo "test49_cascade_ergo_tfold_cacheonly launched on GPU 4"
echo "Monitor: tail -f logs/test49_cascade_ergo_tfold_cacheonly_train.log"
echo "Hypothesis: Cascade scoring (ERGO → tFold cache-only if score > 0.5) improves tFold AUROC"
echo "Key change: cache_only=True, threshold=0.5, cache_miss_score=0.3"
