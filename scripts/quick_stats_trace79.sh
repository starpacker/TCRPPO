#!/bin/bash
# Quick statistics for trace79

TRAIN_LOG="logs/trace79_curated_targets_train.log"

echo "=========================================="
echo "📊 trace79 Quick Stats (Curated Targets)"
echo "=========================================="
echo ""
echo "Strategy: 5 learnable main + 5 easy decoy targets"
echo "Targets: YLQPRTFLL, LLLDRLNQL, GILGFVFTL, CINGVCWTV, NLVPMVATV"
echo "         LLWRGSIYKL, QLFNYVATI, HMNMMHIYV, KRGGPLPAK, KVNGVHHIKV"
echo ""

# Count episodes
TOTAL_EPS=$(grep -c "^Episode" "$TRAIN_LOG" 2>/dev/null || echo "0")
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
    print()

    # Progress toward 0.0
    mean = np.mean(affs)
    target = 0.0
    if mean > target:
        print(f'  ✅ TARGET REACHED! Mean {mean:.3f} > {target}')
    else:
        gap = target - mean
        print(f'  Gap to 0.0: {gap:.3f}')

    # Compare to trace73
    trace73_mean = -1.172
    improvement = trace73_mean - mean
    print(f'  vs trace73: {improvement:+.3f} (trace73 mean={trace73_mean:.3f})')
"

echo ""
echo "Latest 10 episodes:"
echo "----------------------------------------"
grep "^Episode" "$TRAIN_LOG" | tail -10 | while read line; do
    ep=$(echo "$line" | grep -oP 'Episode \K\d+')
    aff=$(echo "$line" | grep -oP '\| A=\K[-\d.]+' | head -1)
    printf "  Ep %4d: A=%7.3f\n" "$ep" "$aff"
done

echo ""
echo "Comparison to baselines:"
echo "----------------------------------------"
echo "  trace73 (20 targets):  mean=-1.172, >-0.5=16.2%, >0=1.4%"
echo "  trace78 (aggressive):  mean=-1.730 @ 32 eps"
echo "  trace79 (curated):     mean=$(grep "^Episode" "$TRAIN_LOG" | grep -oP '\| A=\K[-\d.]+' | python3 -c 'import sys,numpy as np; print(f\"{np.mean([float(x) for x in sys.stdin]):.3f}\")')"
