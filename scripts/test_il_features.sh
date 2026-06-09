#!/bin/bash
# Quick test for new IL features: early stopping and from-scratch training

set -e

echo "=========================================="
echo "Testing IL New Features"
echo "=========================================="

DATASET="data/il/highaff03_trace11_29_61_62_63_tchard_il.jsonl"
CONFIG="configs/trace62_multi_gates.yaml"
BASE_CKPT="output/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy/checkpoints/milestone_580000.pt"
DEVICE="cuda:6"

# Test 1: Early stopping (small scale)
echo ""
echo "[Test 1] Early stopping with 5 epochs, patience=2"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --base-checkpoint $BASE_CKPT \
  --out output/il_test_early_stop/checkpoints/latest.pt \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 3e-5 \
  --device $DEVICE \
  --seed 42 \
  --val-split 0.1 \
  --patience 2 \
  --save-every-epoch

echo ""
echo "✓ Test 1 completed. Check output/il_test_early_stop/checkpoints/"
echo "  Expected files: latest.pt, best.pt, epoch_1.pt, epoch_2.pt, ..."

# Test 2: From scratch (small scale)
echo ""
echo "[Test 2] From scratch with 3 epochs"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --out output/il_test_from_scratch/checkpoints/latest.pt \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --device $DEVICE \
  --seed 42 \
  --from-scratch \
  --save-every-epoch

echo ""
echo "✓ Test 2 completed. Check output/il_test_from_scratch/checkpoints/"
echo "  Expected files: latest.pt, epoch_1.pt, epoch_2.pt, epoch_3.pt"

# Test 3: From scratch + early stopping
echo ""
echo "[Test 3] From scratch + early stopping"
python scripts/pretrain_il.py \
  --config $CONFIG \
  --dataset $DATASET \
  --out output/il_test_scratch_early/checkpoints/latest.pt \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --device $DEVICE \
  --seed 42 \
  --from-scratch \
  --val-split 0.1 \
  --patience 2 \
  --save-every-epoch

echo ""
echo "✓ Test 3 completed. Check output/il_test_scratch_early/checkpoints/"

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Test 1: Early stopping ✓"
echo "  Test 2: From scratch ✓"
echo "  Test 3: Combined ✓"
echo ""
echo "Next: Compare training logs to verify:"
echo "  1. Early stopping triggers correctly"
echo "  2. From-scratch shows 'Training from scratch' message"
echo "  3. best.pt is saved when val_loss improves"
