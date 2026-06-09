#!/bin/bash
# Launch trace104: Resume trace61 + triple constraint (nat>0.7, target>0.6, decoy<0.6)
#
# Goal: Design TCRs satisfying all 3 constraints simultaneously:
#   1. Naturalness > 0.7 (AE+GMM)
#   2. Target affinity > 0.6
#   3. Max decoy affinity < 0.6

set -e

PYTHON=/home/liuyutian/server/miniconda3/envs/tcrppo_v2/bin/python
TFOLD_PYTHON=/home/liuyutian/server/miniconda3/envs/tfold/bin/python
PROJECT_ROOT=/share/liuyutian/tcrppo_v2
cd $PROJECT_ROOT

GPU_TRAIN=${GPU_TRAIN:-1}
GPU_TFOLD=${GPU_TFOLD:-4}

SOCKET_PATH="/tmp/tfold_server_trace104.sock"
CACHE_PATH="data/tfold_cache_trace104.db"
RUN_NAME="trace104_triple_constraint"
RESUME_CKPT="output/trace61_fp32_restart/checkpoints/latest.pt"
CONFIG="configs/trace104_triple_constraint.yaml"

echo "=========================================="
echo "trace104: Triple Constraint Experiment"
echo "=========================================="
echo "Resume ckpt:  $RESUME_CKPT"
echo "Config:       $CONFIG"
echo "Training GPU: $GPU_TRAIN"
echo "tFold GPU:    $GPU_TFOLD"
echo "Socket:       $SOCKET_PATH"
echo "Cache:        $CACHE_PATH"
echo ""
echo "Constraints:"
echo "  1. Naturalness > 0.7 (AE+GMM)"
echo "  2. Target affinity > 0.6"
echo "  3. Max decoy affinity < 0.6"
echo ""
echo "Reward: Large bonus (+3.0) only when ALL 3 satisfied"
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
    --seed 104 \
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
echo ""
echo "Monitor first episodes:"
echo "  grep '^Episode' logs/${RUN_NAME}_train.log | head -20"
echo "=========================================="
