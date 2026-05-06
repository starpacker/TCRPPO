#!/bin/bash
# test44: Pure tFold PPO Training
# Hypothesis: tFold's structure-aware scoring provides better specificity than ERGO
# Key config: tfold scorer, n_envs=4 (reduced for speed), checkpoint_freq=50K

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/default.yaml \
    --run_name test44_pure_tfold_nocache \
    --seed 42 \
    --reward_mode v1_ergo_only \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 4 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 8 \
    --ban_stop \
    --train_targets data/tfold_good_peptides.txt \
    > logs/test44_pure_tfold_train.log 2>&1 &

echo "test44_pure_tfold_nocache launched on GPU 0"
echo "Monitor: tail -f logs/test44_pure_tfold_train.log"
echo "Hypothesis: tFold structure-aware scoring achieves >0.65 AUROC (better than test41's 0.6243)"
echo "Speed: Expect 1-4 min per rollout depending on cache hit rate"
echo "Early abort: Will stop if cache hit rate < 50% at 100K steps"
