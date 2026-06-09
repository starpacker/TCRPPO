#!/bin/bash
# IL Training Experiments: Early Stopping + From Scratch

set -e

DATASET="data/il/highaff03_trace11_29_61_62_63_tchard_il.jsonl"
CONFIG="configs/trace62_multi_gates.yaml"
BASE_CKPT="output/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy/checkpoints/milestone_580000.pt"
DEVICE="cuda:6"

echo "=========================================="
echo "IL Training Experiments"
echo "=========================================="

# Experiment 1: Original (3 epochs, resume from base)
echo ""
echo "[1/4] Baseline: 3 epochs with base checkpoint"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --base-checkpoint $BASE_CKPT \
  --out output/il_exp1_baseline_3epoch/checkpoints/latest.pt \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 3e-5 \
  --device $DEVICE \
  --seed 42

# Experiment 2: Early stopping (10 epochs max, patience=3, 10% val split)
echo ""
echo "[2/4] Early stopping: 10 epochs max, patience=3, 10% val"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --base-checkpoint $BASE_CKPT \
  --out output/il_exp2_early_stopping/checkpoints/latest.pt \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 3e-5 \
  --device $DEVICE \
  --seed 42 \
  --val-split 0.1 \
  --patience 3 \
  --save-every-epoch

# Experiment 3: From scratch (no base checkpoint, 8 epochs)
echo ""
echo "[3/4] From scratch: 8 epochs, no base checkpoint"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --out output/il_exp3_from_scratch_8epoch/checkpoints/latest.pt \
  --epochs 8 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --device $DEVICE \
  --seed 42 \
  --from-scratch \
  --save-every-epoch

# Experiment 4: From scratch + early stopping
echo ""
echo "[4/4] From scratch + early stopping: 15 epochs max, patience=4"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --out output/il_exp4_scratch_early_stop/checkpoints/latest.pt \
  --epochs 15 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --device $DEVICE \
  --seed 42 \
  --from-scratch \
  --val-split 0.1 \
  --patience 4 \
  --save-every-epoch

echo ""
echo "=========================================="
echo "All IL experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - output/il_exp1_baseline_3epoch/"
echo "  - output/il_exp2_early_stopping/"
echo "  - output/il_exp3_from_scratch_8epoch/"
echo "  - output/il_exp4_scratch_early_stop/"
echo ""
echo "Next steps:"
echo "  1. Compare training curves"
echo "  2. Evaluate each checkpoint with eval_checkpoint_decoy_reward_tfold.py"
echo "  3. Use best checkpoint for RL fine-tuning"
