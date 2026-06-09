#!/bin/bash
# trace102: Soft naturalness gate from trace61 (clean baseline)
#
# Strategy: Resume from trace61 (balanced baseline) instead of trace99 (affinity-biased).
# Use softer gate threshold (-0.45 vs -0.3) to give model more exploration room.
#
# Expected outcome:
#   - Week 1: Nat pass rate 10% -> 40%
#   - Week 2: Nat pass rate -> 55%, Affinity>0 -> 50%
#   - Week 3: Both satisfied -> 35%+

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
cd /share/liuyutian/tcrppo_v2

# Resume from trace61 fp32 restart checkpoint
RESUME_CKPT="output/trace61_fp32_restart/checkpoints/latest.pt"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: Checkpoint not found: $RESUME_CKPT"
    echo "Available checkpoints:"
    ls -lh output/trace61_fp32_restart/checkpoints/
    exit 1
fi

echo "==================================================================="
echo "trace102: Soft Naturalness Gate from trace61 (Clean Start)"
echo "==================================================================="
echo "Resume from: $RESUME_CKPT"
echo "Key changes:"
echo "  - naturalness_gate_affinity: true (GATE ENABLED)"
echo "  - naturalness_gate_threshold: -0.45 (softer than trace100's -0.3)"
echo "  - w_naturalness: 8.0 (gentler than trace100's 10.0)"
echo "  - learning_rate: 5e-5 (prevent forgetting)"
echo "  - entropy_coef: 0.02 (more exploration)"
echo ""
echo "Advantages vs trace100:"
echo "  - trace61 baseline: less affinity-biased than trace99"
echo "  - Softer gate: easier for model to find solutions"
echo "  - More balanced starting point"
echo ""
echo "Target metrics:"
echo "  - Naturalness pass rate: 10% -> 55%"
echo "  - Affinity > 0 rate: -> 50%+"
echo "  - Both satisfied: 35%+"
echo "==================================================================="
echo ""

# Launch training
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace102_nat_gate_from_trace61.yaml \
    --run_name trace102_nat_gate_from_trace61 \
    --resume_from "$RESUME_CKPT" \
    > logs/trace102_nat_gate_from_trace61_train.log 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > logs/trace102_nat_gate_from_trace61_train.pid
echo "Training launched on GPU 2, PID: $TRAIN_PID"
echo ""

# Launch tFold server
sleep 5
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace102.sock \
    --gpu 2 \
    > logs/trace102_nat_gate_from_trace61_tfold_server.log 2>&1 &

TFOLD_PID=$!
echo $TFOLD_PID > logs/trace102_nat_gate_from_trace61_tfold.pid
echo "tFold server launched, PID: $TFOLD_PID"
echo ""

echo "Monitor commands:"
echo "  tail -f logs/trace102_nat_gate_from_trace61_train.log"
echo "  python3 scripts/monitor_trace102.py"
echo ""
echo "Stop commands:"
echo "  kill $TRAIN_PID $TFOLD_PID"
echo ""
echo "Hypothesis: Starting from trace61's balanced baseline with softer gate"
echo "will achieve better naturalness without sacrificing affinity capability."
