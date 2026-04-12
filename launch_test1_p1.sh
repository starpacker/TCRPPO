#!/bin/bash
# Launch Test 1: Two-Phase Training (Phase 1 - v1_ergo_only for 1M steps)

CUDA_VISIBLE_DEVICES=0 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test1_two_phase_p1 \
    --reward_mode v1_ergo_only \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --seed 42 \
    > output/test1_two_phase_p1_train.log 2>&1 &

echo "Test 1 Phase 1 launched on GPU 0, PID: $!"
echo "Log: output/test1_two_phase_p1_train.log"
