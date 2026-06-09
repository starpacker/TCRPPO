#!/bin/bash
# Monitor SFT training progress

LOG_FILE="logs/sft_filtered_training.log"

echo "=== SFT Training Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

# Check if training is running
PID=$(pgrep -f "train_sft_esm.py.*filtered_sft_trajectories")
if [ -z "$PID" ]; then
    echo "❌ Training process not found!"
else
    echo "✓ Training running (PID: $PID)"
fi

echo ""
echo "=== Latest Progress ==="
grep -E "Epoch [0-9]+: 100%" $LOG_FILE | tail -5

echo ""
echo "=== Current Epoch ==="
grep -E "=== Epoch" $LOG_FILE | tail -1

echo ""
echo "=== Latest Loss ==="
grep -E "loss=" $LOG_FILE | tail -1 | sed 's/.*loss=/loss=/'

echo ""
echo "=== Checkpoints Saved ==="
ls -lh output/sft_filtered_training/checkpoint_*.pt 2>/dev/null | wc -l
ls -lh output/sft_filtered_training/checkpoint_*.pt 2>/dev/null | tail -3

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1

echo ""
echo "=== Estimated Time ==="
COMPLETED_EPOCHS=$(grep -c "Epoch [0-9]*: 100%" $LOG_FILE)
TOTAL_EPOCHS=50
if [ $COMPLETED_EPOCHS -gt 0 ]; then
    FIRST_EPOCH_TIME=$(grep "Epoch 1: 100%" $LOG_FILE | grep -oP '\d+\.\d+s/it' | grep -oP '\d+\.\d+')
    if [ ! -z "$FIRST_EPOCH_TIME" ]; then
        REMAINING=$((TOTAL_EPOCHS - COMPLETED_EPOCHS))
        ESTIMATED_SECONDS=$(echo "$FIRST_EPOCH_TIME * 5 * $REMAINING" | bc)
        ESTIMATED_HOURS=$(echo "scale=1; $ESTIMATED_SECONDS / 3600" | bc)
        echo "  Completed: $COMPLETED_EPOCHS/$TOTAL_EPOCHS epochs"
        echo "  Estimated remaining: ${ESTIMATED_HOURS}h"
    fi
fi

echo ""
echo "=== Recent Errors ==="
grep -i "error\|traceback\|exception" $LOG_FILE | tail -3
if [ $? -ne 0 ]; then
    echo "  No errors found ✓"
fi
