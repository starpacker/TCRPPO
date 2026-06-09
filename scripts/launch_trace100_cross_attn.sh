#!/bin/bash
# Launch trace100: trace98 + TCR-peptide cross attention
#
# Adds explicit TCR-peptide interaction modeling via cross-attention.
# TCR features attend to peptide features before fusion into policy backbone.

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

GPU_TRAIN=${GPU_TRAIN:-0}
GPU_TFOLD=${GPU_TFOLD:-3}

SOCKET_PATH="/tmp/tfold_server_trace100.sock"
CACHE_PATH="data/tfold_cache_trace100.db"
RUN_NAME="trace100_cross_attn"
CONFIG="configs/trace100_cross_attn.yaml"

echo "=========================================="
echo "trace100: trace98 + TCR-peptide cross attention"
echo "=========================================="
echo "Config:       $CONFIG"
echo "Training GPU: $GPU_TRAIN"
echo "tFold GPU:    $GPU_TFOLD  (FP32)"
echo "Socket:       $SOCKET_PATH"
echo "Cache:        $CACHE_PATH"
echo ""

# Sanity checks
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG"; exit 1
fi

# Clean up old socket
rm -f "$SOCKET_PATH"
mkdir -p data logs

# ---- Step 1: tFold server ----
echo "Step 1: Starting FP32 tFold server on GPU $GPU_TFOLD..."
nohup $TFOLD_PYTHON scripts/tfold_feature_server.py \
    --socket "$SOCKET_PATH" \
    --gpu $GPU_TFOLD \
    --chunk-size 64 \
    --completion-log "logs/${RUN_NAME}_tfold_completions.log" \
    > logs/${RUN_NAME}_tfold_server.log 2>&1 &
TFOLD_PID=$!
echo "tFold server started: PID=$TFOLD_PID"
echo $TFOLD_PID > logs/${RUN_NAME}_tfold.pid

echo "Waiting up to 90s for tFold server READY..."
for i in $(seq 1 90); do
    if grep -q "READY" logs/${RUN_NAME}_tfold_server.log 2>/dev/null; then
        echo "  tFold ready after ${i}s"; break
    fi
    if ! ps -p $TFOLD_PID > /dev/null; then
        echo "ERROR: tFold server died before READY!"
        tail -50 logs/${RUN_NAME}_tfold_server.log; exit 1
    fi
    sleep 1
done
if ! grep -q "READY" logs/${RUN_NAME}_tfold_server.log 2>/dev/null; then
    echo "WARN: 'READY' not detected in 90s, but server still alive — continuing."
fi

# ---- Step 2: PPO training ----
echo ""
echo "Step 2: Launching PPO training on GPU $GPU_TRAIN..."

CUDA_VISIBLE_DEVICES=$GPU_TRAIN nohup $PYTHON -u tcrppo_v2/ppo_trainer.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    > logs/${RUN_NAME}_train.log 2>&1 &
TRAIN_PID=$!
echo "PPO training started: PID=$TRAIN_PID"
echo $TRAIN_PID > logs/${RUN_NAME}_train.pid

echo ""
echo "=========================================="
echo "trace100 launched successfully!"
echo "=========================================="
echo "tFold PID:    $TFOLD_PID  (logs/${RUN_NAME}_tfold_server.log)"
echo "Training PID: $TRAIN_PID  (logs/${RUN_NAME}_train.log)"
echo ""
echo "Monitor training:"
echo "  tail -f logs/${RUN_NAME}_train.log"
echo ""
echo "Monitor tFold:"
echo "  tail -f logs/${RUN_NAME}_tfold_server.log"
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "Stop tFold:"
echo "  kill $TFOLD_PID"
echo ""
