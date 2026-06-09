#!/bin/bash
# Real-time trace79 monitoring - shows latest episodes as they complete

TRAIN_LOG="logs/trace79_curated_targets_train.log"

echo "=========================================="
echo "🔴 LIVE trace79 Monitor (Curated Targets)"
echo "=========================================="
echo ""
echo "Strategy: 5 learnable main + 5 easy decoy targets"
echo "Goal: Stably reach affinity > 0.0"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Show initial status
echo "📊 Current Status:"
EPISODES=$(grep -c "^Episode" "$TRAIN_LOG" 2>/dev/null || echo "0")
echo "  Total episodes completed: $EPISODES"

if [ "$EPISODES" -gt 0 ]; then
    echo ""
    echo "  Quick stats:"
    grep "^Episode" "$TRAIN_LOG" | grep -oP '\| A=\K[-\d.]+' | python3 -c "
import sys
import numpy as np
affs = [float(line.strip()) for line in sys.stdin]
if affs:
    print(f'    Mean affinity: {np.mean(affs):.3f}')
    print(f'    Best affinity: {max(affs):.3f}')
    print(f'    > 0.0: {sum(1 for a in affs if a > 0)} ({100*sum(1 for a in affs if a > 0)/len(affs):.1f}%)')
    print(f'    > -0.5: {sum(1 for a in affs if a > -0.5)} ({100*sum(1 for a in affs if a > -0.5)/len(affs):.1f}%)')
"
fi

echo ""
echo "📈 Latest Episodes (live):"
echo "----------------------------------------"
tail -f "$TRAIN_LOG" | grep --line-buffered "Episode" | while read line; do
    ep=$(echo "$line" | grep -oP 'Episode \K\d+')
    step=$(echo "$line" | grep -oP 'Step \K\d+')
    aff=$(echo "$line" | grep -oP '\| A=\K[-\d.]+' | head -1)

    # Color code based on affinity
    if (( $(echo "$aff > 0.0" | bc -l 2>/dev/null || echo "0") )); then
        color="\033[1;32m"  # Green - TARGET REACHED!
    elif (( $(echo "$aff > -0.5" | bc -l 2>/dev/null || echo "0") )); then
        color="\033[1;33m"  # Yellow - Very good
    elif (( $(echo "$aff > -1.0" | bc -l 2>/dev/null || echo "0") )); then
        color="\033[0;33m"  # Dim yellow - Good
    else
        color="\033[0m"  # Normal
    fi

    printf "${color}Ep %4d | Step %7d | A=%7.3f\033[0m\n" "$ep" "$step" "$aff"
done
