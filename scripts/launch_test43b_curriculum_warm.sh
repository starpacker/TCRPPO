#!/bin/bash
# test43b: Warm-start from test41 + naturalness phase then contrastive
# Hypothesis: Starting from test41 (AUROC 0.6243), can naturalness+contrastive phase improve further?
# Key config: 2M steps, warm-start from test41, 2-phase schedule at 0/500K, 45 ERGO-good peptides
# GPU: 4

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test43b_curriculum_warm \
    --seed 42 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --reward_mode raw_multi_penalty \
    --total_timesteps 4500000 \
    --n_envs 8 \
    --learning_rate 1e-4 \
    --ban_stop \
    --train_targets data/ergo_good_peptides.txt \
    --resume_from output/test41_from_test33_1m_16decoys/checkpoints/final.pt \
    --resume_change_reward_mode raw_multi_penalty \
    --resume_reset_optimizer \
    --w_naturalness 0.1 --w_decoy 0.02 --w_diversity 0.0 \
    --reward_schedule '[{"step":2500000,"mode":"raw_multi_penalty","w_nat":0.1,"w_decoy":0.02,"w_diversity":0.0},{"step":3000000,"mode":"contrastive_ergo","n_decoys":16}]' \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 0 \
    > logs/test43b_curriculum_warm_train.log 2>&1 &

echo "test43b_curriculum_warm launched on GPU 3"
echo "Monitor: tail -f logs/test43b_curriculum_warm_train.log"
echo "Hypothesis: naturalness phase then contrastive from test41 warm-start"
