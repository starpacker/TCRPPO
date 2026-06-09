#!/bin/bash
# Quick statistics for trace83

cd /share/liuyutian/tcrppo_v2
TRAIN_LOG="logs/trace83_curated_from_trace73_train.log"

echo "=========================================="
echo "📊 trace83 Quick Stats (Checkpoint Resume)"
echo "=========================================="
echo ""
echo "Strategy: Resume from trace73 checkpoint + 10 curated targets"
echo "Config: terminal_reward_only=true, gate=-1.0, use_delta_reward=true"
echo ""

# Count episodes
TOTAL_EPS=$(grep "^Episode " "$TRAIN_LOG" 2>/dev/null | wc -l)
echo "Total episodes: $TOTAL_EPS"
echo ""

if [ "$TOTAL_EPS" -eq 0 ]; then
    echo "No episodes completed yet."
    exit 0
fi

# Extract affinities and compute stats
echo "Affinity Statistics:"
echo "----------------------------------------"
grep "^Episode" "$TRAIN_LOG" | grep -oP '\| A=\K[-\d.]+' | head -n "$TOTAL_EPS" | python3 -c "
import sys
import numpy as np

affs = [float(line.strip()) for line in sys.stdin]
if affs:
    print(f'  Count:     {len(affs)}')
    print(f'  Mean:      {np.mean(affs):.3f}')
    print(f'  Median:    {np.median(affs):.3f}')
    print(f'  Best:      {max(affs):.3f}')
    print(f'  Worst:     {min(affs):.3f}')
    print(f'  Std:       {np.std(affs):.3f}')
    print()
    print('Distribution:')
    print(f'  > 0.0:     {sum(1 for a in affs if a > 0)} ({100*sum(1 for a in affs if a > 0)/len(affs):.1f}%)')
    print(f'  > -0.5:    {sum(1 for a in affs if a > -0.5)} ({100*sum(1 for a in affs if a > -0.5)/len(affs):.1f}%)')
    print(f'  > -1.0:    {sum(1 for a in affs if a > -1.0)} ({100*sum(1 for a in affs if a > -1.0)/len(affs):.1f}%)')
    print(f'  > -1.5:    {sum(1 for a in affs if a > -1.5)} ({100*sum(1 for a in affs if a > -1.5)/len(affs):.1f}%)')
    print(f'  > -2.0:    {sum(1 for a in affs if a > -2.0)} ({100*sum(1 for a in affs if a > -2.0)/len(affs):.1f}%)')
    print()
"

echo ""
echo "Delta (Improvement) Statistics:"
echo "----------------------------------------"
grep "^Episode" "$TRAIN_LOG" | grep -oP 'DeltaA=\K[-\d.]+' | head -n "$TOTAL_EPS" | python3 -c "
import sys
import numpy as np

deltas = [float(line.strip()) for line in sys.stdin]
if deltas:
    print(f'  Mean DeltaA: {np.mean(deltas):.3f}')
    print(f'  Median:      {np.median(deltas):.3f}')
    print(f'  Best:        {max(deltas):.3f}')
    print(f'  Worst:       {min(deltas):.3f}')
    print()
    print(f'  Positive deltas: {sum(1 for d in deltas if d > 0)} ({100*sum(1 for d in deltas if d > 0)/len(deltas):.1f}%)')
    print(f'  Negative deltas: {sum(1 for d in deltas if d < 0)} ({100*sum(1 for d in deltas if d < 0)/len(deltas):.1f}%)')
"

echo ""
echo "PPO Training Metrics:"
echo "----------------------------------------"
grep "^Step" "$TRAIN_LOG" | tail -5 | while read line; do
    step=$(echo "$line" | grep -oP 'Step\s+\K[\d,]+' | tr -d ',')
    r=$(echo "$line" | grep -oP 'R:\s+\K[-\d.]+')
    pg=$(echo "$line" | grep -oP 'PG:\s+\K[-\d.]+')
    vf=$(echo "$line" | grep -oP 'VF:\s+\K[-\d.]+')
    ent=$(echo "$line" | grep -oP 'Ent:\s+\K[-\d.]+')
    kl=$(echo "$line" | grep -oP 'KL:\s+\K[-\d.]+')
    clip=$(echo "$line" | grep -oP 'Clip:\s+\K[-\d.]+')
    printf "  Step %6s: R=%7.3f PG=%7.4f VF=%7.4f Ent=%5.3f KL=%7.5f Clip=%4.2f\n" "$step" "$r" "$pg" "$vf" "$ent" "$kl" "$clip"
done

echo ""
echo "Latest 10 episodes:"
echo "----------------------------------------"
grep "^Episode" "$TRAIN_LOG" | tail -10 | while read line; do
    ep=$(echo "$line" | grep -oP 'Episode \K\d+')
    aff=$(echo "$line" | grep -oP '\| A=\K[-\d.]+' | head -1)
    delta=$(echo "$line" | grep -oP 'DeltaA=\K[-\d.]+' | head -1)
    printf "  Ep %4d: A=%7.3f  ΔA=%+7.3f\n" "$ep" "$aff" "$delta"
done

echo ""
echo "Comparison to trace81 (from-scratch terminal_reward_only):"
echo "----------------------------------------"
echo "  trace81 @ 160 eps: mean A=-7.763, mean ΔA=-0.179, VF=1.14, KL=0.001"
echo "  trace83 @ current: mean A=$(grep "^Episode" "$TRAIN_LOG" | grep -oP '\| A=\K[-\d.]+' | python3 -c 'import sys,numpy as np; print(f\"{np.mean([float(x) for x in sys.stdin]):.3f}\")')"
echo "                     mean ΔA=$(grep "^Episode" "$TRAIN_LOG" | grep -oP 'DeltaA=\K[-\d.]+' | python3 -c 'import sys,numpy as np; print(f\"{np.mean([float(x) for x in sys.stdin]):.3f}\")')"
