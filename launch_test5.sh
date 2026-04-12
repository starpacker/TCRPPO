#!/bin/bash
# Launch Test 5: Threshold-based penalties (Option C)

CUDA_VISIBLE_DEVICES=4 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test5_threshold \
    --reward_mode threshold_penalty \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --w_decoy 0.05 \
    --w_naturalness 0.02 \
    --w_diversity 0.01 \
    --seed 42 \
    > output/test5_threshold_train.log 2>&1 &

echo "Test 5 launched on GPU 4, PID: $!"
echo "Log: output/test5_threshold_train.log"
