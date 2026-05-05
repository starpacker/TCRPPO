#!/bin/bash
# test47: 32 Decoys + 45 Filtered Peptides (from test41)
# Hypothesis: More decoys + more peptides → better specificity (target 0.64-0.67 AUROC)
# Key config: Resume from test41, 32 decoys, 45 peptides, 1M steps

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test47_32decoys_45peptides \
    --seed 42 \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --n_contrast_decoys 32 \
    --contrastive_agg mean \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/45_filtered_peptides.txt \
    > logs/test47_32decoys_45peptides_train.log 2>&1 &

echo "test47_32decoys_45peptides launched on GPU 4"
echo "Monitor: tail -f logs/test47_32decoys_45peptides_train.log"
echo "Hypothesis: 32 decoys + 45 peptides → AUROC 0.64-0.67"
