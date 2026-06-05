#!/bin/bash
# Launch trace61 FP32 restart: Continue from trace61 checkpoint with pure FP32 tFold

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

GPU_TRAIN=4
GPU_TFOLD=5

SOCKET_PATH="/tmp/tfold_server_trace61_fp32_restart.sock"
CACHE_PATH="data/tfold_cache_trace61_fp32_restart.db"
RUN_NAME="trace61_fp32_restart"

echo "=========================================="
echo "trace61 FP32 Restart"
echo "=========================================="
echo "Checkpoint: output/trace61_dynamic_pool/checkpoints/latest.pt"
echo "Config: configs/trace61_fp32_restart.yaml"
echo "Training GPU: $GPU_TRAIN"
echo "tFold GPU: $GPU_TFOLD (pure FP32, no AMP)"
echo "Socket: $SOCKET_PATH"
echo "Cache: $CACHE_PATH"
echo ""

# Clean up old socket
rm -f "$SOCKET_PATH"

# Create cache directory
mkdir -p data

echo "Step 1: Starting pure FP32 tFold server on GPU $GPU_TFOLD..."
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket "$SOCKET_PATH" \
    --gpu $GPU_TFOLD \
    --chunk-size 64 \
    --completion-log "logs/${RUN_NAME}_tfold_completions.log" \
    > logs/${RUN_NAME}_tfold_server.log 2>&1 &
TFOLD_PID=$!

echo "tFold server started: PID=$TFOLD_PID"
echo "Waiting 30 seconds for server initialization..."
sleep 30

# Verify server is ready
if ! ps -p $TFOLD_PID > /dev/null; then
    echo "ERROR: tFold server failed to start!"
    cat logs/${RUN_NAME}_tfold_server.log
    exit 1
fi

if ! grep -q "READY" logs/${RUN_NAME}_tfold_server.log; then
    echo "WARNING: Server may not be ready yet. Check logs/${RUN_NAME}_tfold_server.log"
fi

echo ""
echo "Step 2: Launching RL training on GPU $GPU_TRAIN..."

CUDA_VISIBLE_DEVICES=$GPU_TRAIN nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config configs/trace61_fp32_restart.yaml \
    --resume_from output/trace61_dynamic_pool/checkpoints/latest.pt \
    --run_name $RUN_NAME \
    --seed 61 \
    > logs/${RUN_NAME}_train.log 2>&1 &
TRAIN_PID=$!

echo "Training started: PID=$TRAIN_PID"
echo ""
echo "=========================================="
echo "Launch complete!"
echo "=========================================="
echo ""
echo "Monitor commands:"
echo "  Training log:  tail -f logs/${RUN_NAME}_train.log"
echo "  tFold server:  tail -f logs/${RUN_NAME}_tfold_server.log"
echo "  tFold stats:   tail -f logs/${RUN_NAME}_tfold_completions.log"
echo "  TensorBoard:   tensorboard --logdir output/${RUN_NAME}/tensorboard"
echo ""
echo "Process IDs:"
echo "  Training: $TRAIN_PID"
echo "  tFold:    $TFOLD_PID"
echo ""
echo "Checkpoints: output/${RUN_NAME}/checkpoints/"
echo ""
echo "Expected training time: ~24 hours for 2M steps"
echo "=========================================="
