#!/bin/bash
# Monitor trace78 progress towards mean affinity -0.5

TRAIN_LOG="logs/trace78_aggressive_push_train.log"

if [ ! -f "$TRAIN_LOG" ]; then
    echo "❌ Training log not found: $TRAIN_LOG"
    echo "Has trace78 been launched?"
    exit 1
fi

echo "=========================================="
echo "📊 trace78 Progress Monitor"
echo "=========================================="
echo ""

# Check if training is running
TRAIN_PID=$(ps aux | grep "trace78_aggressive_push" | grep ppo_trainer | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PID" ]; then
    echo "✅ Training is RUNNING (PID: $TRAIN_PID)"
else
    echo "⚠️  Training is NOT running"
fi
echo ""

# Show latest training step
echo "📈 Latest Training Steps:"
echo "----------------------------------------"
tail -100 "$TRAIN_LOG" | grep "Step" | tail -5
echo ""

# Show recent episodes with affinity
echo "🎯 Recent Episodes (Affinity):"
echo "----------------------------------------"
tail -200 "$TRAIN_LOG" | grep "Episode" | tail -10 | while read line; do
    # Extract episode number, step, and affinity
    ep=$(echo "$line" | grep -oP 'Episode \K\d+')
    step=$(echo "$line" | grep -oP 'Step \K\d+')
    aff=$(echo "$line" | grep -oP 'A=\K[-\d.]+')
    init_aff=$(echo "$line" | grep -oP 'InitA=\K[-\d.]+')
    delta=$(echo "$line" | grep -oP 'DeltaA=\K[-\d.]+')

    # Color code based on affinity
    if (( $(echo "$aff > 0" | bc -l) )); then
        color="\033[1;32m"  # Green
    elif (( $(echo "$aff > -0.5" | bc -l) )); then
        color="\033[1;33m"  # Yellow
    else
        color="\033[0m"  # Normal
    fi

    printf "${color}Ep %4d | Step %7d | A=%7.3f | InitA=%7.3f | Δ=%6.3f\033[0m\n" \
        "$ep" "$step" "$aff" "$init_aff" "$delta"
done
echo ""

# Calculate quick stats from recent episodes
echo "📊 Quick Stats (last 50 episodes):"
echo "----------------------------------------"
tail -200 "$TRAIN_LOG" | grep "Episode" | tail -50 | \
    grep -oP 'A=\K[-\d.]+' | \
    python3 -c "
import sys
import numpy as np
affs = [float(line.strip()) for line in sys.stdin]
if affs:
    print(f'  Count: {len(affs)}')
    print(f'  Mean:  {np.mean(affs):.3f}')
    print(f'  Best:  {max(affs):.3f}')
    print(f'  Worst: {min(affs):.3f}')
    print(f'  >0:    {sum(1 for a in affs if a > 0)} ({100*sum(1 for a in affs if a > 0)/len(affs):.1f}%)')
    print(f'  >-0.5: {sum(1 for a in affs if a > -0.5)} ({100*sum(1 for a in affs if a > -0.5)/len(affs):.1f}%)')

    # Progress indicator
    mean = np.mean(affs)
    target = -0.5
    if mean > target:
        print(f'\n  ✅ TARGET REACHED! Mean {mean:.3f} > {target}')
    else:
        gap = target - mean
        print(f'\n  Gap to target: {gap:.3f}')
        print(f'  Progress: {max(0, 100*(1 - gap/0.7)):.1f}% (assuming start at -1.2)')
"
echo ""

# Show online pool stats
echo "🏊 Online Pool Stats:"
echo "----------------------------------------"
tail -100 "$TRAIN_LOG" | grep "OnlinePool:" | tail -1
echo ""

# Show tFold cache stats
echo "💾 tFold Cache Stats:"
echo "----------------------------------------"
tail -100 "$TRAIN_LOG" | grep "Cache:" | tail -1
echo ""

echo "=========================================="
echo "💡 Commands:"
echo "=========================================="
echo "  Full log:     tail -f $TRAIN_LOG"
echo "  Analyze:      python analyze_affinity_distribution.py"
echo "  Visualize:    python visualize_single_trace.py 78"
echo "  This monitor: watch -n 10 ./scripts/monitor_trace78.sh"
echo ""
