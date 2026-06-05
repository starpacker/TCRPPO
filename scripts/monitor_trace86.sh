#!/bin/bash
# Monitor trace86 progress

LOG_FILE="logs/trace86_per_step_reward_train.log"

echo "=== trace86 Training Monitor ==="
echo ""

# Check if training is running
PID=$(ps aux | grep "trace86_per_step_reward" | grep -v grep | grep python | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Training not running"
else
    echo "✅ Training running (PID: $PID)"
fi

echo ""
echo "=== Latest Episodes ==="
tail -1000 "$LOG_FILE" | grep "Episode" | tail -10

echo ""
echo "=== tFold Performance ==="
echo "Recent extraction times:"
tail -100 "$LOG_FILE" | grep "path_ms=" | tail -5 | grep -oP 'path_ms=\K[0-9.]+' | awk '{sum+=$1; count++} END {if(count>0) printf "  Average: %.1f ms (%.1f sec)\n", sum/count, sum/count/1000}'

echo ""
echo "Cache hit rate:"
tail -100 "$LOG_FILE" | grep "Cache" | tail -10

echo ""
echo "=== Performance Summary ==="
echo "Total episodes:"
grep -c "Episode" "$LOG_FILE"

echo ""
echo "Best affinity so far:"
grep "Episode" "$LOG_FILE" | grep -oP ' A=[+-]?\d+\.\d+' | sed 's/ A=//' | sort -n -r | head -1

echo ""
echo "Mean affinity (last 50 episodes):"
grep "Episode" "$LOG_FILE" | tail -50 | grep -oP ' A=[+-]?\d+\.\d+' | sed 's/ A=//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}'

echo ""
echo "=== Monitor Commands ==="
echo "  Watch live: tail -f $LOG_FILE"
echo "  This script: bash scripts/monitor_trace86.sh"
echo "  Kill training: kill $PID"
