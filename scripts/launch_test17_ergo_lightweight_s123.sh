#!/bin/bash
# test17: ERGO lightweight seed=123 (check seed dependence with lightweight encoder)
export CUDA_VISIBLE_DEVICES=
conda run -n tcrppo_v2 python -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test17_ergo_lightweight_s123 \
    --reward_mode v1_ergo_only \
    --affinity_scorer ergo \
    --encoder lightweight \
    --encoder_dim 256 \
    --total_timesteps 2000000 \
    --seed 123 \
    2>&1 | tee output/test17_ergo_lightweight_s123/train.log
