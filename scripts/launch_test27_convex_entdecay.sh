#!/bin/bash
# test27: Convex reward (ERGO^3) + entropy decay
# Hypothesis: Convex reward amplifies gradient at high scores, pushing policy
#             to concentrate on exceptional TCRs. Entropy decay after 800K lets
#             the policy exploit discovered high-quality modes.
# Key config: reward=ERGO^3, entropy_coef=0.05→0.005 (decay from 800K)
# Expected: Higher Top1_AUC and Hit@0.7 vs test23/test26

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test27_convex_entdecay \
    --seed 42 \
    --reward_mode v1_ergo_convex \
    --convex_alpha 3.0 \
    --entropy_coef_final 0.005 \
    --entropy_decay_start 800000 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --ban_stop \
    --device cuda \
    > logs/test27_convex_entdecay_train.log 2>&1 &

echo "test27_convex_entdecay launched on GPU 1, PID: $!"
echo "Monitor: tail -f logs/test27_convex_entdecay_train.log"
echo "Hypothesis: ERGO^3 + entropy decay → better Top-1/Top-5 quality"
