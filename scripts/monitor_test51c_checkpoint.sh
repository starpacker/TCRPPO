#!/bin/bash
# Monitor test51c progress and auto-launch new experiments at 100K checkpoint

CHECKPOINT_DIR="output/test51c_no_decoy_long_ep/checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/milestone_100000.pt"
LOG_FILE="logs/test51c_no_decoy_long_ep_train.log"

echo "=========================================="
echo "test51c Checkpoint Monitor"
echo "=========================================="
echo "Waiting for: $CHECKPOINT_FILE"
echo "Monitoring: $LOG_FILE"
echo ""

# Function to get current step from log
get_current_step() {
    if [ -f "$LOG_FILE" ]; then
        grep -oP "Step \K[0-9]+" "$LOG_FILE" | tail -1
    else
        echo "0"
    fi
}

# Function to estimate time to 100K
estimate_time() {
    local current_step=$1
    local remaining=$((100000 - current_step))

    if [ $current_step -gt 0 ]; then
        # Calculate steps/min from recent progress
        local recent_steps=$(grep "Step [0-9]+" "$LOG_FILE" | tail -100 | grep -oP "Step \K[0-9]+" | awk '{sum+=$1; count++} END {if(count>1) print (sum/count)*60; else print 50}')
        local steps_per_min=${recent_steps:-50}
        local minutes_remaining=$((remaining / steps_per_min))
        local hours=$((minutes_remaining / 60))
        local mins=$((minutes_remaining % 60))

        echo "${hours}h ${mins}m (${steps_per_min} steps/min)"
    else
        echo "Unknown"
    fi
}

# Monitor loop
while [ ! -f "$CHECKPOINT_FILE" ]; do
    current_step=$(get_current_step)
    eta=$(estimate_time $current_step)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step: $current_step / 100,000 | ETA: $eta"

    # Check if test51c is still running
    if ! pgrep -f "test51c_no_decoy_long_ep" > /dev/null; then
        echo "WARNING: test51c process not found!"
    fi

    sleep 300  # Check every 5 minutes
done

echo ""
echo "=========================================="
echo "✓ Checkpoint reached: $CHECKPOINT_FILE"
echo "=========================================="
echo ""

# Verify checkpoint file
if [ -f "$CHECKPOINT_FILE" ]; then
    checkpoint_size=$(du -h "$CHECKPOINT_FILE" | cut -f1)
    echo "Checkpoint size: $checkpoint_size"
    echo ""

    # Launch new experiments
    echo "Launching new experiments..."
    echo ""

    echo "[1/2] Launching test51c_amp (GPU 2)..."
    bash scripts/launch_test51c_amp.sh
    sleep 5

    echo "[2/2] Launching test51c_decoy (GPU 3)..."
    bash scripts/launch_test51c_decoy.sh
    sleep 5

    echo ""
    echo "=========================================="
    echo "✓ All experiments launched!"
    echo "=========================================="
    echo ""
    echo "Running experiments:"
    echo "  - test51c (baseline)     : GPU 1 (continues to 2M)"
    echo "  - test51c_amp (AMP)      : GPU 2 (100K → 2M)"
    echo "  - test51c_decoy (decoy)  : GPU 3 (100K → 2M)"
    echo ""
    echo "Monitor logs:"
    echo "  tail -f logs/test51c_no_decoy_long_ep_train.log"
    echo "  tail -f logs/test51c_amp_train.log"
    echo "  tail -f logs/test51c_decoy_train.log"
    echo ""
    echo "Check GPU usage:"
    echo "  nvidia-smi"
    echo ""
else
    echo "ERROR: Checkpoint file not found after detection!"
    exit 1
fi
