#!/bin/bash
# trace103 Stage 1: Pure naturalness pretraining (10k steps)
#
# Strategy: Train ONLY on naturalness for 10k steps to build natural foundation.
# No affinity scoring, no tFold server needed.
# After completion, manually switch to stage2 for delta reward finetune.
#
# Expected outcome after 10k steps:
#   - Mean naturalness score: 0.8+ (like trace97)
#   - Model learns to generate natural-looking TCR sequences
#   - Ready for multi-objective finetune in stage2

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

echo "==================================================================="
echo "trace103 Stage 1: Pure Naturalness Pretraining"
echo "==================================================================="
echo "Duration: 10k steps (~1250 episodes)"
echo "Reward: Pure naturalness (AE+GMM combined score 0~1)"
echo "No affinity scoring in this stage"
echo ""
echo "After 10k steps complete:"
echo "  1. Kill this training"
echo "  2. Launch stage2: ./scripts/launch_trace103_stage2_finetune.sh"
echo "==================================================================="
echo ""

# Launch training
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace103_stage1_nat_pretrain.yaml \
    --run_name trace103_stage1_nat_pretrain \
    > logs/trace103_stage1_nat_pretrain_train.log 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > logs/trace103_stage1_nat_pretrain.pid
echo "Training launched on GPU 3, PID: $TRAIN_PID"
echo ""

echo "Monitor commands:"
echo "  tail -f logs/trace103_stage1_nat_pretrain_train.log"
echo "  watch -n 10 'tail -50 logs/trace103_stage1_nat_pretrain_train.log | grep Step'"
echo ""
echo "Check progress:"
echo "  tail -100 logs/trace103_stage1_nat_pretrain_train.log | grep Episode | tail -20"
echo ""
echo "Stop command:"
echo "  kill $TRAIN_PID"
echo ""
echo "When step reaches 10,000:"
echo "  1. kill $TRAIN_PID"
echo "  2. ./scripts/launch_trace103_stage2_finetune.sh"
