#!/bin/bash
# Real-time trace78 monitoring - shows latest episodes as they complete

TRAIN_LOG="logs/trace78_aggressive_push_train.log"

echo "=========================================="
echo "🔴 LIVE trace78 Monitor"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Show initial status
echo "📊 Current Status:"
EPISODES=$(grep -c "^Episode" "$TRAIN_LOG" 2>/dev/null || echo "0")
echo "  Total episodes completed: $EPISODES"
echo ""

# Tail and filter for episodes
echo "📈 Latest Episodes (live):"
echo "----------------------------------------"
tail -f "$TRAIN_LOG" | grep --line-buffered "Episode" | while read line; do
    ep=$(echo "$line" | grep -oP 'Episode \K\d+')
    step=$(echo "$line" | grep -oP 'Step \K\d+')
    aff=$(echo "$line" | grep -oP 'A=\K[-\d.]+' | head -1)

    # Color code based on affinity
    if (( $(echo "$aff > -0.5" | bc -l 2>/dev/null || echo "0") )); then
        color="\033[1;32m"  # Green - TARGET!
    elif (( $(echo "$aff > -1.0" | bc -l 2>/dev/null || echo "0") )); then
        color="\033[1;33m"  # Yellow - Good
    else
        color="\033[0m"  # Normal
    fi

    printf "${color}Ep %4d | Step %7d | A=%7.3f${color}\033[0m\n" "$ep" "$step" "$aff"
done
