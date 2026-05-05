#!/bin/bash
# test43c: Aggressive contrastive + 32 decoys (2x test41), no naturalness — control
# Hypothesis: Is peptide filtering + more decoys sufficient without naturalness/curriculum?
# Key config: 2M steps, warm-start from test41, 32 decoys, no schedule, 45 ERGO-good peptides
# GPU: 6

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test43c_32decoys_filtered \
    --seed 42 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --reward_mode contrastive_ergo \
    --total_timesteps 4500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --ban_stop \
    --train_targets data/ergo_good_peptides.txt \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --resume_reset_optimizer \
    --n_contrast_decoys 32 --contrastive_agg mean \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 0 \
    > logs/test43c_32decoys_filtered_train.log 2>&1 &

echo "test43c_32decoys_filtered launched on GPU 1"
echo "Monitor: tail -f logs/test43c_32decoys_filtered_train.log"
echo "Hypothesis: peptide filtering + 32 decoys without naturalness/curriculum"
