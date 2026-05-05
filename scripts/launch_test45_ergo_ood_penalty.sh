#!/bin/bash
# test45: ERGO with OOD Penalty
# Hypothesis: Constraining ERGO to its reliable domain prevents exploitation and improves specificity
# Key config: v1_ergo_ood_penalty mode, soft penalty, threshold=0.15, weight=1.0
# Risk mitigation: soft penalty only penalizes excess beyond threshold, starts conservative

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test45_ergo_ood_penalty \
    --seed 42 \
    --reward_mode v1_ergo_ood_penalty \
    --affinity_scorer ergo \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/ergo_positive_peptides.txt \
    --ood_threshold 0.15 \
    --ood_penalty_weight 1.0 \
    --ood_penalty_mode soft \
    > logs/test45_ergo_ood_penalty_train.log 2>&1 &

echo "test45_ergo_ood_penalty launched on GPU 4"
echo "Monitor: tail -f logs/test45_ergo_ood_penalty_train.log"
echo "Hypothesis: OOD penalty constrains agent to ERGO's reliable domain"
echo "Expected: OOD trigger rate should decrease over time as agent learns to stay in-domain"
echo "Risk mitigation: soft penalty (only excess beyond threshold), conservative weight=1.0"
echo "If OOD trigger rate > 90% at 100K steps, consider lowering threshold or weight"
