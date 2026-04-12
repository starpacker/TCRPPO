#!/bin/bash
# Launch Test 3: Step-wise raw terminal reward

CUDA_VISIBLE_DEVICES=2 nohup /home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python -u \
    tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test3_stepwise \
    --reward_mode v1_ergo_stepwise \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --seed 42 \
    > output/test3_stepwise_train.log 2>&1 &

echo "Test 3 launched on GPU 2, PID: $!"
echo "Log: output/test3_stepwise_train.log"
