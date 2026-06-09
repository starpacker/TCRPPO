#!/bin/bash
# Monitor SFT training progress

LOG_FILE="/share/liuyutian/tcrppo_v2/output/sft_esm_training.log"

echo "=== SFT Training Monitor ==="
echo

# Check if process is running
PID=$(pgrep -f "train_sft_esm.py" | head -1)
if [ -n "$PID" ]; then
    echo "✓ Training process running (PID: $PID)"
    ps -p $PID -o etime,cmd | tail -1
    echo
else
    echo "✗ Training process not found"
    echo
fi

# Show latest progress
echo "=== Latest Progress ==="
grep -E "^=== Epoch|^Train losses|^Validation|checkpoint" "$LOG_FILE" | tail -20

echo
echo "=== Current Batch ==="
tail -1 "$LOG_FILE" | grep -oP "Epoch \d+:.*"

echo
echo "=== Loss Trend (last 10 batches) ==="
grep -oP "loss=\K[0-9.]+" "$LOG_FILE" | tail -10 | awk '{sum+=$1; print NR". "$1} END {if (NR>0) print "Average: "sum/NR}'

echo
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1

echo
echo "=== Estimated Time ==="
TOTAL_BATCHES=$(grep -oP "Epoch 1:.*?/\K\d+" "$LOG_FILE" | head -1)
CURRENT_BATCH=$(grep -oP "Epoch 1:\s+\d+%.*?\|\s+\K\d+" "$LOG_FILE" | tail -1)
if [ -n "$TOTAL_BATCHES" ] && [ -n "$CURRENT_BATCH" ]; then
    echo "Epoch 1: $CURRENT_BATCH / $TOTAL_BATCHES batches"
    PROGRESS=$(echo "scale=2; $CURRENT_BATCH / $TOTAL_BATCHES * 100" | bc)
    echo "Progress: ${PROGRESS}%"

    # Estimate time per epoch (assuming 4.5s/batch average)
    TIME_PER_EPOCH=$(echo "$TOTAL_BATCHES * 4.5 / 60" | bc)
    TOTAL_TIME=$(echo "$TIME_PER_EPOCH * 50 / 60" | bc)
    echo "Estimated time per epoch: ${TIME_PER_EPOCH} minutes"
    echo "Estimated total time: ${TOTAL_TIME} hours"
fi

echo
echo "To monitor in real-time: tail -f $LOG_FILE"
