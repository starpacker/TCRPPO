#!/bin/bash
# test50: tFold Cascade on Excellent Peptides (AUC≥0.8)
# Hypothesis: Training on tFold-excellent peptides with cascade scorer produces TCRs that score high under tFold evaluation
# Key config: cascade scorer, 20 tFold-excellent peptides, contrastive_ergo reward

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test50_tfold_cascade_excellent \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer cascade \
    --cascade_threshold 0.5 \
    --cascade_tfold_weight 0.7 \
    --cascade_ergo_weight 0.3 \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/tfold_excellent_peptides.txt \
    > logs/test50_tfold_cascade_excellent_train.log 2>&1 &

echo "test50_tfold_cascade_excellent launched on GPU 3"
echo "Monitor: tail -f logs/test50_tfold_cascade_excellent_train.log"
echo "Hypothesis: Cascade scorer on tFold-excellent peptides produces tFold-validated TCRs"
