#!/bin/bash
# Launch trace73: curriculum learning + increased exploration

set -e

cd /share/liuyutian/tcrppo_v2

# GPU 3
export CUDA_VISIBLE_DEVICES=3

# Start tFold server on GPU 3
echo "Starting tFold server on GPU 3..."
tmux new-session -d -s tfold_trace73 \
  "python scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_trace73_curriculum_exploration.sock \
    --gpu 3 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/trace73_curriculum_exploration_tfold_completion.log \
    2>&1 | tee logs/trace73_curriculum_exploration_tfold_amp_server.log"

echo "Waiting for tFold server to start..."
sleep 10

# Check if server is ready
if [ ! -S /tmp/tfold_server_trace73_curriculum_exploration.sock ]; then
    echo "Error: tFold server socket not found!"
    exit 1
fi

echo "tFold server ready!"

# Start training
echo "Starting trace73 training..."
tmux new-session -d -s trace73_train \
  "conda run -n tcrppo_v2 --no-capture-output \
    python -c '
import sys
sys.path.insert(0, \"/share/liuyutian/tcrppo_v2\")
from tcrppo_v2.data.tcr_pool_trace61_patch import *
import tcrppo_v2.ppo_trainer as trainer_module
trainer_module.main()
' \
    --config configs/trace73_curriculum_exploration.yaml \
    --run_name trace73_curriculum_exploration \
    --seed 42 \
    --resume_from output/trace70_gate_m1p5_from_trace61/checkpoints/latest.pt \
    2>&1 | tee logs/trace73_curriculum_exploration_train.log"

echo ""
echo "✅ trace73 launched!"
echo ""
echo "Monitor training:"
echo "  tmux attach -t trace73_train"
echo ""
echo "Monitor tFold server:"
echo "  tmux attach -t tfold_trace73"
echo ""
echo "View logs:"
echo "  tail -f logs/trace73_curriculum_exploration_train.log"
echo ""
