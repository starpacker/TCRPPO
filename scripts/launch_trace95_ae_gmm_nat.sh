#!/bin/bash
# Launch trace95: AE-GMM naturalness scorer (instead of ESM-2)

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

GPU_TRAIN=1
GPU_TFOLD=7

SOCKET_PATH="/tmp/tfold_server_trace95.sock"
CACHE_PATH="data/tfold_cache_trace95.db"
RUN_NAME="trace95_ae_gmm_nat"

echo "=========================================="
echo "trace95: AE-GMM Naturalness Scorer"
echo "=========================================="
echo "Config: configs/trace95_ae_gmm_nat.yaml"
echo "Training GPU: $GPU_TRAIN"
echo "tFold GPU: $GPU_TFOLD (pure FP32, no AMP)"
echo "Socket: $SOCKET_PATH"
echo "Cache: $CACHE_PATH"
echo ""
echo "Key Changes vs trace94:"
echo "  - Naturalness scorer: AE-GMM (was ESM-2 perplexity)"
echo "  - AE-GMM threshold: 0.8"
echo "  - All other settings identical to trace94"
echo ""
echo "Expected: AE-GMM should provide stronger/more sensitive"
echo "          naturalness signal than ESM-2 pseudo-perplexity"
echo ""

# Clean up old socket
rm -f "$SOCKET_PATH"

# Create cache directory and logs
mkdir -p data logs

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
    --config configs/trace95_ae_gmm_nat.yaml \
    --run_name $RUN_NAME \
    --seed 95 \
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
