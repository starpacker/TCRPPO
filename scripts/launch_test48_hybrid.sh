#!/bin/bash
# test48: Hybrid Scorer (90% ERGO + 10% tFold)
# Hypothesis: tFold's 10% signal corrects ERGO's sequence biases → better tFold AUROC
# Key config: Resume from test41, hybrid scorer (90% ERGO + 10% tFold), 16 decoys

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test48_hybrid_90ergo_10tfold \
    --seed 42 \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --affinity_scorer hybrid \
    --hybrid_tfold_ratio 0.1 \
    --encoder esm2 \
    --total_timesteps 3500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/mcpas_12_targets.txt \
    > logs/test48_hybrid_90ergo_10tfold_train.log 2>&1 &

echo "test48_hybrid_90ergo_10tfold launched on GPU 2"
echo "Monitor: tail -f logs/test48_hybrid_90ergo_10tfold_train.log"
echo "Hypothesis: 10% tFold signal → better tFold AUROC without sacrificing ERGO"
echo "Expected training time: ~30 hours"
