#!/bin/bash
# Launch Test 2: min_steps=6 + raw_decoy

CUDA_VISIBLE_DEVICES=1 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test2_min6_raw \
    --reward_mode raw_decoy \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --w_decoy 0.05 \
    --min_steps 6 \
    --min_steps_penalty -3.0 \
    --seed 42 \
    > output/test2_min6_raw_train.log 2>&1 &

echo "Test 2 launched on GPU 1, PID: $!"
echo "Log: output/test2_min6_raw_train.log"
