#!/bin/bash
# test42 Phase 2: Contrastive fine-tuning with NetTCR (8 decoys)
# Hypothesis: NetTCR contrastive reward can improve specificity like ERGO
# Key config: resume from phase1@2M, contrastive_nettcr, 8 decoys

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test42_nettcr_phase2 \
    --seed 42 \
    --resume_from output/test42_nettcr_phase1/checkpoints/final.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --n_contrast_decoys 8 \
    --contrastive_agg mean \
    --total_timesteps 1500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 100000 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.2 \
    --curriculum_l2 0.3 \
    --device cuda \
    > logs/test42_nettcr_phase2_train.log 2>&1 &

echo "test42_nettcr_phase2 launched on GPU 2"
echo "Monitor: tail -f logs/test42_nettcr_phase2_train.log"
echo "Hypothesis: NetTCR contrastive (8 decoys) to match test33's performance"
