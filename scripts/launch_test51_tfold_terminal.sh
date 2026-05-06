#!/bin/bash
# test51: Pure tFold with Terminal-Only Reward
# Hypothesis: Terminal reward (3 tFold calls/episode) + naturalness bonus improves training efficiency
# Key config: tfold scorer, terminal_reward_only, 2 decoys (1A+1B), max_steps=4, w_naturalness=0.1

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/test51.yaml \
    --run_name test51_tfold_terminal \
    --seed 42 \
    --reward_mode contrastive_ergo \
    --affinity_scorer tfold \
    --encoder esm2 \
    --total_timesteps 2000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --max_steps 4 \
    --ban_stop \
    --terminal_reward_only \
    --n_contrast_decoys 2 \
    --contrastive_agg mean \
    --w_naturalness 0.1 \
    --curriculum_l0 0.5 \
    --curriculum_l1 0.0 \
    --curriculum_l2 0.5 \
    --train_targets data/tfold_excellent_peptides.txt \
    --tfold_cache_path data/tfold_feature_cache.db \
    > logs/test51_tfold_terminal_train.log 2>&1 &

echo "test51_tfold_terminal launched on GPU 4"
echo "Monitor: tail -f logs/test51_tfold_terminal_train.log"
echo ""
echo "Hypothesis: Terminal-only reward reduces tFold calls from 68/episode to 3/episode"
echo "  - Per episode: 1 initial + 1 target + 1 decoy_mean(A,B) = 3 tFold calls"
echo "  - vs step-wise: 4 steps × (1 target + 2 decoys) = 12 calls (4x slower)"
echo "  - Reward: affinity(target) - mean(affinity(decoy_A), affinity(decoy_B)) + 0.1 * naturalness"
echo ""
echo "Config:"
echo "  - Scorer: tFold (structure-aware)"
echo "  - Decoys: 2 (1A + 1B tier)"
echo "  - Max steps: 4 (reduced from 8)"
echo "  - Curriculum: L0=0.5 (known binder variants), L2=0.5 (random TCRdb)"
echo "  - Naturalness weight: 0.1"
echo "  - Checkpoints: every 200K steps"
echo "  - No training evaluation (eval_interval=999999999)"
echo ""
echo "Expected training time: ~12h (vs 100h+ for step-wise)"
