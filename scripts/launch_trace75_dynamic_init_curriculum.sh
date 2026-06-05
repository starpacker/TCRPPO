#!/bin/bash
# Launch trace75: dynamic initial-affinity curriculum.

set -e

cd /share/liuyutian/tcrppo_v2

RUN_NAME="trace75_dynamic_init_curriculum"
CONFIG="configs/trace75_dynamic_init_curriculum.yaml"
RESUME_CKPT="output/trace73_curriculum_exploration/checkpoints/latest.pt"
TFOLD_SOCKET="/tmp/tfold_server_${RUN_NAME}.sock"
GPU_ID="${GPU_ID:-4}"
SEED="${SEED:-42}"

echo "=========================================="
echo "trace75: dynamic initial-affinity curriculum"
echo "=========================================="
echo "Run name: $RUN_NAME"
echo "Config: $CONFIG"
echo "Resume from: $RESUME_CKPT"
echo "GPU: $GPU_ID"
echo "Seed: $SEED"
echo "=========================================="

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: checkpoint not found: $RESUME_CKPT"
    exit 1
fi

if [ ! -f "output/trace73_curriculum_exploration/online_tcr_pool_snapshot.json" ]; then
    echo "ERROR: trace73 online pool snapshot not found"
    exit 1
fi

echo ""
echo "Preparing independent trace75 tFold cache..."
for suffix in "" "-wal" "-shm"; do
    src="data/tfold_feature_cache_trace73_curriculum_exploration.db${suffix}"
    dst="data/tfold_feature_cache_trace75_dynamic_init_curriculum.db${suffix}"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp -f "$src" "$dst"
    fi
done

echo ""
echo "Starting tFold feature server..."
rm -f "$TFOLD_SOCKET"
setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n tfold --no-capture-output \
    python scripts/tfold_feature_server.py \
    --socket "$TFOLD_SOCKET" \
    --gpu 0 \
    --use-amp-wrapper \
    --chunk-size 64 \
    --completion-log "logs/${RUN_NAME}_tfold_completion.log" \
    > "logs/${RUN_NAME}_tfold_amp_server.log" 2>&1 < /dev/null &

TFOLD_PID=$!
echo "tFold server PID: $TFOLD_PID"

echo "Waiting for tFold socket..."
for i in {1..60}; do
    if [ -S "$TFOLD_SOCKET" ]; then
        echo "tFold socket ready"
        break
    fi
    sleep 2
done

if [ ! -S "$TFOLD_SOCKET" ]; then
    echo "ERROR: tFold socket not created"
    echo "Check logs/${RUN_NAME}_tfold_amp_server.log"
    exit 1
fi

echo ""
echo "Starting training..."
PY_CMD="
import sys
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')
from tcrppo_v2.data.tcr_pool_trace61_patch import *
import tcrppo_v2.ppo_trainer as trainer_module
trainer_module.main()
"

setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n tcrppo_v2 --no-capture-output \
    python -c "
$PY_CMD" \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --seed "$SEED" \
    --resume_from "$RESUME_CKPT" \
    --resume_reset_optimizer \
    > "logs/${RUN_NAME}_train.log" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo ""
echo "trace75 launched."
echo "Monitor with:"
echo "  tail -f logs/${RUN_NAME}_train.log"
echo "  tail -f logs/${RUN_NAME}_tfold_amp_server.log"
