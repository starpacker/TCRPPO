#!/bin/bash
# test23: Contrastive ERGO reward (target vs decoys)
# Hypothesis: reward = ERGO(target) - mean(ERGO(decoys)) breaks train-eval coupling
# Key config: contrastive_ergo, n_contrast_decoys=4, ESM-2, ban_stop, seed=42

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test23_contrastive_ergo \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 4 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --esm_cache_path /tmp/esm_cache_test23.db \
    --decoy_library_path /tmp/pMHC_decoy_library \
    --ban_stop \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --device cuda \
    > logs/test23_contrastive_ergo_train.log 2>&1 &

echo "test23_contrastive_ergo launched on GPU 0"
echo "Monitor: tail -f logs/test23_contrastive_ergo_train.log"
echo "Hypothesis: Contrastive reward breaks ERGO train-eval coupling"
