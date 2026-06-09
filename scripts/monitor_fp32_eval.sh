#!/bin/bash
# Monitor FP32 evaluation progress

OUTPUT_DIR="results/fp32_eval_sequential_20260603_104721"

echo "=========================================="
echo "FP32 Evaluation Progress Monitor"
echo "=========================================="
echo ""

# Check if evaluation is still running
if ps aux | grep -q "[e]val_three_traces_fp32_sequential.sh"; then
    echo "✓ Evaluation script is running"
else
    echo "✗ Evaluation script is NOT running"
fi

if ps aux | grep -q "[t]fold_feature_server.py"; then
    echo "✓ tFold server is running"
    ps aux | grep "[t]fold_feature_server.py" | awk '{print "  PID:", $2, "GPU:", $NF}'
else
    echo "✗ tFold server is NOT running"
fi

echo ""
echo "----------------------------------------"
echo "Log Files Status:"
echo "----------------------------------------"

for trace in trace61 trace72 trace73; do
    log_file="$OUTPUT_DIR/${trace}.log"
    if [ -f "$log_file" ]; then
        lines=$(wc -l < "$log_file")
        last_update=$(stat -c %y "$log_file" | cut -d. -f1)
        echo "$trace: $lines lines, last update: $last_update"

        # Extract progress info
        if grep -q "Target:" "$log_file"; then
            current_target=$(grep "Target:" "$log_file" | tail -1 | awk '{print $2}')
            echo "  Current target: $current_target"
        fi

        # Count completed targets
        completed=$(grep -c "============================================================" "$log_file" || echo 0)
        echo "  Completed targets: $((completed / 2)) / 20"

    else
        echo "$trace: NOT STARTED"
    fi
done

echo ""
echo "----------------------------------------"
echo "Main Log Tail:"
echo "----------------------------------------"
tail -10 logs/eval_sequential.log

echo ""
echo "=========================================="
