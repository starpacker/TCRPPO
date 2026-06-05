#!/bin/bash
# Launch trace62: Multi-gate curriculum (gates at -2, -1, 0.0, 0.5) - RELAUNCH

set -e

cd /share/liuyutian/tcrppo_v2

# Check if trace29 checkpoint exists
RESUME_CKPT="output/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy/checkpoints/milestone_580000.pt"
if [ ! -f "$RESUME_CKPT" ]; then
    echo "Error: trace29 checkpoint not found at $RESUME_CKPT"
    exit 1
fi

echo "Starting trace62: Multi-gate curriculum (RELAUNCH v4)"
echo "Resume from: $RESUME_CKPT (trace29@580K - CORRECT checkpoint!)"
echo "Gates: -3.0, -2.0, -1.0, 0.0, 0.5"
echo "Bonuses: 0.5, 1.0, 1.5, 2.0, 2.5"
echo "Cache: Using existing large cache (read-only)"

# Start tFold server
echo "Starting tFold server on GPU 2..."
nohup python scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace62_multi_gates.sock \
    --gpu 2 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace62_multi_gates_tfold_completion.log \
    > logs/trace62_multi_gates_tfold_amp_server.log 2>&1 &

TFOLD_PID=$!
echo "tFold server PID: $TFOLD_PID"
echo "Waiting for tFold server to be ready..."
sleep 5

# Wait for socket to be created
for i in {1..30}; do
    if [ -S /tmp/tfold_server_trace62_multi_gates.sock ]; then
        echo "tFold server socket ready!"
        break
    fi
    echo "Waiting for socket... ($i/30)"
    sleep 2
done

if [ ! -S /tmp/tfold_server_trace62_multi_gates.sock ]; then
    echo "ERROR: tFold server socket not found after 60 seconds!"
    echo "Check logs/trace62_multi_gates_tfold_amp_server.log"
    exit 1
fi

echo "Waiting additional 10 seconds for server to fully initialize..."
sleep 10

# Start training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=2 conda run -n tcrppo_v2 --no-capture-output \
    python -c "
import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')
from tcrppo_v2.data.tcr_pool_trace61_patch import *
import tcrppo_v2.ppo_trainer as trainer_module
trainer_module.main()
" \
    --config configs/trace62_multi_gates.yaml \
    --run_name trace62_multi_gates \
    --seed 42 \
    --resume_from "$RESUME_CKPT" \
    > logs/trace62_multi_gates_train.log 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

echo ""
echo "trace62 launched successfully!"
echo "  tFold server PID: $TFOLD_PID"
echo "  Training PID: $TRAIN_PID"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/trace62_multi_gates_train.log"
echo "  tail -f logs/trace62_multi_gates_tfold_completion.log"
echo ""
echo "Check progress:"
echo "  ls -lh output/trace62_multi_gates/checkpoints/"
echo ""
