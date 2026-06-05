#!/bin/bash
# Launch trace63: Smooth gate reward with sigmoid transition

set -e

cd /share/liuyutian/tcrppo_v2

# Check if trace61 checkpoint exists
RESUME_CKPT="output/trace61_dynamic_pool/checkpoints/latest.pt"
if [ ! -f "$RESUME_CKPT" ]; then
    echo "Error: trace61 checkpoint not found at $RESUME_CKPT"
    exit 1
fi

echo "Starting trace63: Smooth gate reward"
echo "Resume from: $RESUME_CKPT"
echo "Gate: -1.0"
echo "Bonus: 1.0"
echo "Temperature: 0.5 (sigmoid steepness)"

# Start tFold server
echo "Starting tFold server on GPU 6..."
nohup python scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace63_smooth_gate.sock \
    --gpu 6 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace63_smooth_gate_tfold_completion.log \
    > logs/trace63_smooth_gate_tfold_amp_server.log 2>&1 &

TFOLD_PID=$!
echo "tFold server PID: $TFOLD_PID"
sleep 10

# Start training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=6 conda run -n tcrppo_v2 --no-capture-output \
    python -c "
import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')
from tcrppo_v2.data.tcr_pool_trace61_patch import *
import tcrppo_v2.ppo_trainer as trainer_module
trainer_module.main()
" \
    --config configs/trace63_smooth_gate.yaml \
    --run_name trace63_smooth_gate \
    --seed 42 \
    --resume_from "$RESUME_CKPT" \
    > logs/trace63_smooth_gate_train.log 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

echo ""
echo "trace63 launched successfully!"
echo "  tFold server PID: $TFOLD_PID"
echo "  Training PID: $TRAIN_PID"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/trace63_smooth_gate_train.log"
echo "  tail -f logs/trace63_smooth_gate_tfold_completion.log"
echo ""
echo "Check progress:"
echo "  ls -lh output/trace63_smooth_gate/checkpoints/"
echo ""
