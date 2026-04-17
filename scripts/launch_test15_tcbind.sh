#!/bin/bash
# test15: TCBind scorer (GPU) + lightweight encoder — breaks ERGO train-eval coupling
export CUDA_VISIBLE_DEVICES=0
cd /share/liuyutian/tcrppo_v2
mkdir -p output/test15_tcbind_lightweight
nohup conda run -n tcrppo_v2 python -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test15_tcbind_lightweight \
    --reward_mode v1_ergo_only \
    --affinity_scorer tcbind \
    --encoder lightweight \
    --encoder_dim 256 \
    --device cuda \
    --total_timesteps 2000000 \
    --seed 42 \
    > output/test15_tcbind_lightweight/train.log 2>&1 &
echo "PID: $!"
