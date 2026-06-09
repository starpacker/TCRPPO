#!/bin/bash
# Launch SFT + RL Fine-tuning
# Base: SFT filtered model checkpoint
# Goal: Improve affinity from -5.49 to > -2.0 using RL

set -e

cd /share/liuyutian/tcrppo_v2

# GPU 0
export CUDA_VISIBLE_DEVICES=0

# Start tFold server on GPU 0
echo "Starting tFold server on GPU 0..."
tmux new-session -d -s tfold_sft_rl \
  "python scripts/tfold_feature_server.py \
    --socket /tmp/tfold_server_sft_rl_finetune.sock \
    --gpu 0 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log logs/sft_rl_finetune_tfold_completion.log \
    2>&1 | tee logs/sft_rl_finetune_tfold_server.log"

echo "Waiting for tFold server to start..."
sleep 10

# Check if server is ready
if [ ! -S /tmp/tfold_server_sft_rl_finetune.sock ]; then
    echo "Error: tFold server socket not found!"
    exit 1
fi

echo "tFold server ready!"

# Start training
echo "Starting SFT + RL fine-tuning..."
tmux new-session -d -s sft_rl_train \
  "conda run -n tcrppo_v2 --no-capture-output \
    python -c '
import sys
sys.path.insert(0, \"/share/liuyutian/tcrppo_v2\")
from tcrppo_v2.data.tcr_pool_trace61_patch import *
import tcrppo_v2.ppo_trainer as trainer_module
trainer_module.main()
' \
    --config configs/sft_rl_finetune.yaml \
    --run_name sft_rl_finetune \
    --seed 42 \
    --resume_from output/sft_filtered_training/checkpoint_final.pt \
    --resume_reset_optimizer \
    2>&1 | tee logs/sft_rl_finetune_train.log"

echo ""
echo "✅ SFT + RL fine-tuning launched!"
echo ""
echo "Monitor training:"
echo "  tmux attach -t sft_rl_train"
echo ""
echo "Monitor tFold server:"
echo "  tmux attach -t tfold_sft_rl"
echo ""
echo "View logs:"
echo "  tail -f logs/sft_rl_finetune_train.log"
echo ""
echo "Expected timeline:"
echo "  100K steps: affinity > -3.0"
echo "  200K steps: affinity > -2.0 (minimum target)"
echo "  600K steps: affinity > -1.0 (ideal target)"
echo ""
