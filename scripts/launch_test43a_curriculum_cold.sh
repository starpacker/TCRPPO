#!/bin/bash
# test43a: Cold-start 3-phase curriculum (ERGO → naturalness+decoy → contrastive)
# Hypothesis: Can a single run with 3-phase curriculum beat test41's 0.6243 from scratch?
# Key config: 3M steps, 3-phase schedule at 0/500K/1.5M, 45 ERGO-good peptides
# GPU: 0

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=7 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test43a_curriculum_cold \
    --seed 42 \
    --affinity_scorer ergo \
    --encoder esm2 \
    --reward_mode v1_ergo_only \
    --total_timesteps 3000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --ban_stop \
    --train_targets data/ergo_good_peptides.txt \
    --reward_schedule '[{"step":0,"mode":"v1_ergo_only"},{"step":500000,"mode":"raw_multi_penalty","w_nat":0.1,"w_decoy":0.02,"w_diversity":0.0},{"step":1500000,"mode":"contrastive_ergo","n_decoys":16,"lr":1e-4}]' \
    --entropy_coef_final 0.01 \
    --entropy_decay_start 500000 \
    --curriculum_l0 0.5 --curriculum_l1 0.2 --curriculum_l2 0.3 \
    > logs/test43a_curriculum_cold_train.log 2>&1 &

echo "test43a_curriculum_cold launched on GPU 7"
echo "Monitor: tail -f logs/test43a_curriculum_cold_train.log"
echo "Hypothesis: 3-phase curriculum (ergo→multi_penalty→contrastive) from cold-start"
