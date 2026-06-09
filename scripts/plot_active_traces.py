#!/usr/bin/env python3
"""
Extract affinity curves from active training logs and plot them.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Active traces
traces = [
    "trace53_terminal_trace29_reward_L2only",
    "trace78_aggressive_push",
    "trace83_curated_from_trace73",
    "trace84_push_to_zero"
]

log_dir = Path("/share/liuyutian/tcrppo_v2/logs")
output_dir = Path("/share/liuyutian/tcrppo_v2/figures")
output_dir.mkdir(exist_ok=True)

# Extract data
data = {}
for trace in traces:
    log_file = log_dir / f"{trace}_train.log"
    if not log_file.exists():
        print(f"⚠️  {trace}: log file not found")
        continue

    steps = []
    affinities = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'Episode' in line and 'Step' in line and ' A=' in line:
                # Extract step
                step_match = re.search(r'Step (\d+)', line)
                # Extract affinity (with space before A to avoid DeltaA)
                aff_match = re.search(r' A=([+-]?\d+\.\d+)', line)

                if step_match and aff_match:
                    step = int(step_match.group(1))
                    aff = float(aff_match.group(1))
                    steps.append(step)
                    affinities.append(aff)

    if steps:
        data[trace] = {
            'steps': np.array(steps),
            'affinities': np.array(affinities),
            'count': len(steps)
        }
        print(f"✓ {trace}: {len(steps)} episodes, steps {steps[0]}-{steps[-1]}")
    else:
        print(f"⚠️  {trace}: no data found")

# Plot
if not data:
    print("No data to plot!")
    exit(1)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (trace, color) in enumerate(zip(traces, colors)):
    if trace not in data:
        axes[idx].text(0.5, 0.5, f'{trace}\nNo data',
                      ha='center', va='center', fontsize=14)
        axes[idx].set_title(trace)
        continue

    d = data[trace]
    steps = d['steps']
    affs = d['affinities']

    # Plot raw data (semi-transparent)
    axes[idx].scatter(steps, affs, alpha=0.3, s=10, color=color, label='Episodes')

    # Plot moving average (window=50)
    window = min(50, len(affs) // 10)
    if window > 1:
        ma = np.convolve(affs, np.ones(window)/window, mode='valid')
        ma_steps = steps[window-1:]
        axes[idx].plot(ma_steps, ma, color=color, linewidth=2, label=f'MA({window})')

    # Add horizontal line at 0
    axes[idx].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (A=0)')

    # Statistics
    mean_aff = np.mean(affs)
    best_aff = np.max(affs)
    latest_mean = np.mean(affs[-50:]) if len(affs) >= 50 else mean_aff
    positive_rate = np.sum(affs >= 0) / len(affs) * 100

    axes[idx].set_title(f'{trace}\n'
                       f'Episodes: {len(affs)} | Steps: {steps[0]:,}-{steps[-1]:,}\n'
                       f'Mean: {mean_aff:.3f} | Best: {best_aff:.3f} | Latest(50): {latest_mean:.3f}\n'
                       f'Positive: {positive_rate:.1f}%',
                       fontsize=10)
    axes[idx].set_xlabel('Training Step')
    axes[idx].set_ylabel('Affinity (A)')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(loc='lower right', fontsize=8)

    # Set y-axis limits
    y_min = min(-10, np.percentile(affs, 1))
    y_max = max(2, np.percentile(affs, 99))
    axes[idx].set_ylim(y_min, y_max)

plt.tight_layout()
output_file = output_dir / "active_traces_affinity_curves.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Trace':<40} {'Episodes':>10} {'Steps':>15} {'Mean A':>10} {'Best A':>10} {'Latest(50)':>12} {'Pos%':>8}")
print("-"*80)

for trace in traces:
    if trace not in data:
        print(f"{trace:<40} {'N/A':>10} {'N/A':>15} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>8}")
        continue

    d = data[trace]
    steps = d['steps']
    affs = d['affinities']

    mean_aff = np.mean(affs)
    best_aff = np.max(affs)
    latest_mean = np.mean(affs[-50:]) if len(affs) >= 50 else mean_aff
    positive_rate = np.sum(affs >= 0) / len(affs) * 100

    step_range = f"{steps[0]:,}-{steps[-1]:,}"

    print(f"{trace:<40} {len(affs):>10} {step_range:>15} {mean_aff:>10.3f} {best_aff:>10.3f} {latest_mean:>12.3f} {positive_rate:>7.1f}%")

print("="*80)
