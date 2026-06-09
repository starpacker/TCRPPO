#!/bin/bash
# Launch trace99: resume from trace61 fp32_restart latest.pt and apply
# trace98's high-naturalness recipe (w_naturalness=5.0, AE+GMM).
#
# Goal: keep trace61's A>=0 capability while fixing its naturalness.

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

GPU_TRAIN=${GPU_TRAIN:-0}
GPU_TFOLD=${GPU_TFOLD:-3}

SOCKET_PATH="/tmp/tfold_server_trace99.sock"
CACHE_PATH="data/tfold_cache_trace99.db"
RUN_NAME="trace99_finetune_nat5_from_trace61"
RESUME_CKPT="output/trace61_fp32_restart/checkpoints/latest.pt"
CONFIG="configs/trace99_finetune_nat5_from_trace61.yaml"

echo "=========================================="
echo "trace99: trace61 resume + w_naturalness=5.0 + AE+GMM"
echo "=========================================="
echo "Resume ckpt:  $RESUME_CKPT"
echo "Config:       $CONFIG"
echo "Training GPU: $GPU_TRAIN"
echo "tFold GPU:    $GPU_TFOLD  (FP32)"
echo "Socket:       $SOCKET_PATH"
echo "Cache:        $CACHE_PATH"
echo ""

# Sanity checks
if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "ERROR: resume checkpoint not found: $RESUME_CKPT"; exit 1
fi
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
    --resume_from "$RESUME_CKPT" \
    --resume_reset_optimizer \
    --run_name "$RUN_NAME" \
    --seed 99 \
    > logs/${RUN_NAME}_train.log 2>&1 &
TRAIN_PID=$!

echo "Training started: PID=$TRAIN_PID"
echo "$TRAIN_PID" > logs/${RUN_NAME}_train.pid
echo "$TFOLD_PID" > logs/${RUN_NAME}_tfold.pid

echo ""
echo "=========================================="
echo "Launched."
echo "  Train log:  tail -f logs/${RUN_NAME}_train.log"
echo "  tFold srv:  tail -f logs/${RUN_NAME}_tfold_server.log"
echo "  PIDs:       train=$TRAIN_PID  tfold=$TFOLD_PID"
echo "=========================================="
