#!/bin/bash
# test42 Phase 3: Increase decoys to 16 (equivalent to test41)
# Hypothesis: 16 decoys with NetTCR will match or exceed test41's 0.6243 AUROC
# Key config: resume from phase2@1M, 16 decoys

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python

cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test42_nettcr_phase3 \
    --seed 42 \
    --resume_from output/test42_nettcr_phase2/checkpoints/milestone_1000000.pt \
    --resume_change_reward_mode contrastive_ergo \
    --reward_mode contrastive_ergo \
    --affinity_scorer nettcr \
    --encoder esm2 \
    --n_contrast_decoys 16 \
    --contrastive_agg mean \
    --total_timesteps 1000000 \
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
    > logs/test42_nettcr_phase3_train.log 2>&1 &

echo "test42_nettcr_phase3 launched on GPU 2"
echo "Monitor: tail -f logs/test42_nettcr_phase3_train.log"
echo "Hypothesis: NetTCR + 16 decoys to match test41's 0.6243 AUROC"
