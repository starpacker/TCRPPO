#!/bin/bash
# Monitor trace84 progress

LOG_FILE="logs/trace84_push_to_zero_train.log"

echo "=== trace84 Training Monitor ==="
echo ""

# Check if training is running
PID=$(ps aux | grep "trace84_push_to_zero" | grep -v grep | grep python | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Training not running"
else
    echo "✅ Training running (PID: $PID)"
    echo "   Runtime: $(ps -p $PID -o etime= | xargs)"
fi

echo ""
echo "=== Latest Episodes ==="
tail -500 "$LOG_FILE" | grep "Episode" | tail -10

echo ""
echo "=== Performance Summary ==="
echo "Current step:"
tail -100 "$LOG_FILE" | grep "Episode" | tail -1 | grep -oP 'Step \K[0-9]+'

echo ""
echo "Mean reward (last 50 episodes):"
tail -500 "$LOG_FILE" | grep "Episode" | tail -50 | grep -oP 'R=[+-]?\K[0-9.-]+' | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f\n", sum/count; else print "N/A"}'

echo ""
echo "Mean affinity (last 50 episodes):"
tail -500 "$LOG_FILE" | grep "Episode" | tail -50 | grep -oP ' A=[+-]?\K[0-9.-]+' | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f\n", sum/count; else print "N/A"}'

echo ""
echo "Best affinity so far:"
grep "Episode" "$LOG_FILE" | grep -oP ' A=[+-]?\d+\.\d+' | sed 's/ A=//' | sort -n -r | head -1

echo ""
echo "=== Monitor Commands ==="
echo "  Watch live: tail -f $LOG_FILE"
echo "  This script: bash scripts/monitor_trace84.sh"
echo "  Kill training: kill $PID"
