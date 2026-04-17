#!/bin/bash
# test16: ERGO on CPU + lightweight encoder — control: same ERGO scorer but lightweight state
export CUDA_VISIBLE_DEVICES=
cd /share/liuyutian/tcrppo_v2
mkdir -p output/test16_ergo_lightweight
nohup conda run -n tcrppo_v2 python -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test16_ergo_lightweight \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder lightweight \
    --encoder_dim 256 \
    --device cpu \
    --total_timesteps 2000000 \
    --seed 42 \
    > output/test16_ergo_lightweight/train.log 2>&1 &
echo "PID: $!"
