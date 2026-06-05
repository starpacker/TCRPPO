#!/bin/bash
# Launch trace78: Aggressive push to break mean affinity -0.5
#
# Goal: Mean affinity > -0.5 within 50K steps
# Base: trace73 latest checkpoint
# GPU: 2

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "🚀 Launching trace78: Aggressive Push"
echo "=========================================="
echo ""
echo "Goal: Break mean affinity -0.5 today!"
echo "Base: trace73 latest checkpoint"
echo "GPU: 2"
echo "Estimated time: 2-3 hours"
echo ""

# Check if trace73 checkpoint exists
CHECKPOINT="output/trace73_curriculum_exploration/checkpoints/latest.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: trace73 checkpoint not found at $CHECKPOINT"
    echo "Please ensure trace73 has run and saved a checkpoint."
    exit 1
fi

echo "✓ Found trace73 checkpoint"
echo ""

# Check GPU availability
echo "Checking GPU 2 availability..."
nvidia-smi -i 2 | grep -E "(MiB|%)" || {
    echo "❌ Error: GPU 2 not available"
    exit 1
}
echo "✓ GPU 2 available"
echo ""

# Launch tFold server
echo "=========================================="
echo "Step 1: Launching tFold server on GPU 2"
echo "=========================================="

TFOLD_SOCKET="/tmp/tfold_server_trace78_aggressive_push.sock"
TFOLD_LOG="logs/trace78_aggressive_push_tfold_server.log"
TFOLD_COMPLETION="logs/trace78_aggressive_push_tfold_completion.log"

# Kill existing server if any
if [ -S "$TFOLD_SOCKET" ]; then
    echo "Cleaning up existing socket..."
    rm -f "$TFOLD_SOCKET"
fi

# Check if tFold server is already running
TFOLD_PID=$(ps aux | grep "tfold_feature_server.py.*trace78" | grep -v grep | awk '{print $2}')
if [ -n "$TFOLD_PID" ]; then
    echo "Found existing tFold server (PID: $TFOLD_PID), killing..."
    kill $TFOLD_PID
    sleep 2
fi

echo "Starting tFold server..."
nohup python scripts/tfold_feature_server.py \
    --socket "$TFOLD_SOCKET" \
    --gpu 2 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log "$TFOLD_COMPLETION" \
    > "$TFOLD_LOG" 2>&1 &

TFOLD_PID=$!
echo "✓ tFold server started (PID: $TFOLD_PID)"
echo "  Log: $TFOLD_LOG"
echo "  Socket: $TFOLD_SOCKET"
echo ""

# Wait for socket to be ready
echo "Waiting for tFold server to be ready..."
for i in {1..30}; do
    if [ -S "$TFOLD_SOCKET" ]; then
        echo "✓ tFold server ready"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "❌ Error: tFold server failed to start"
        echo "Check log: $TFOLD_LOG"
        exit 1
    fi
done
echo ""

# Launch training
echo "=========================================="
echo "Step 2: Launching PPO training"
echo "=========================================="

TRAIN_LOG="logs/trace78_aggressive_push_train.log"

echo "Starting training..."
echo "  Config: configs/trace78_aggressive_push.yaml"
echo "  Resume from: $CHECKPOINT"
echo "  Log: $TRAIN_LOG"
echo ""

CUDA_VISIBLE_DEVICES=2 nohup python -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace78_aggressive_push.yaml \
    --run_name trace78_aggressive_push \
    --seed 42 \
    --resume_from "$CHECKPOINT" \
    --resume_reset_optimizer \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "✓ Training started (PID: $TRAIN_PID)"
echo ""

# Wait a bit and check if training started successfully
sleep 5
if ! ps -p $TRAIN_PID > /dev/null; then
    echo "❌ Error: Training process died immediately"
    echo "Check log: $TRAIN_LOG"
    tail -50 "$TRAIN_LOG"
    exit 1
fi

echo "=========================================="
echo "✅ trace78 launched successfully!"
echo "=========================================="
echo ""
echo "📊 Monitor progress:"
echo "  tail -f $TRAIN_LOG"
echo "  watch -n 10 'tail -30 $TRAIN_LOG | grep Step'"
echo ""
echo "📈 Check affinity:"
echo "  python analyze_affinity_distribution.py"
echo "  python visualize_single_trace.py 78"
echo ""
echo "🎯 Target: Mean affinity > -0.5"
echo "⏱️  Estimated time: 2-3 hours"
echo ""
echo "Process IDs:"
echo "  tFold server: $TFOLD_PID"
echo "  Training: $TRAIN_PID"
echo ""
echo "To stop:"
echo "  kill $TRAIN_PID $TFOLD_PID"
echo ""
