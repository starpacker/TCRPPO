#!/bin/bash
# Monitor all new experiments progress

echo "=== New Experiments Status ($(date)) ==="
echo ""

for test in test1_two_phase_p1 test2_min6_raw test3_stepwise test4_raw_multi test5_threshold; do
    log="output/${test}_train.log"
    if [ -f "$log" ]; then
        echo "--- $test ---"
        # Get last training step line
        grep "^Step" "$log" | tail -1 || echo "Not started training yet"
        echo ""
    fi
done

echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "=== Process Status ==="
ps aux | grep "ppo_trainer.py" | grep -v grep | awk '{print "PID", $2, "GPU", $NF}'
