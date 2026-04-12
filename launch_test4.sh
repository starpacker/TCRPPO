#!/bin/bash
# Launch Test 4: Raw multi-penalty (Option A)

CUDA_VISIBLE_DEVICES=3 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test4_raw_multi \
    --reward_mode raw_multi_penalty \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --w_decoy 0.05 \
    --w_naturalness 0.02 \
    --w_diversity 0.01 \
    --seed 42 \
    > output/test4_raw_multi_train.log 2>&1 &

echo "Test 4 launched on GPU 3, PID: $!"
echo "Log: output/test4_raw_multi_train.log"
