#!/usr/bin/env python3
"""
Plot affinity curves for top 5 alive traces.
"""

import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_affinity_curve(log_path, max_lines=50000):
    """Parse training log to extract affinity over time."""
    try:
        result = subprocess.run(['tail', '-n', str(max_lines), log_path],
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
    except:
        return None, None

    steps = []
    affinities = []

    # Pattern for episode lines
    episode_pattern = re.compile(
        r'Episode\s+\d+\s+\|\s+Step\s+(\d+)\s+\|\s+R=([-\d.]+)\s+\|\s+Len=[\d.]+\s+\|\s+A=([-\d.]+)'
    )

    for line in lines:
        match = episode_pattern.search(line)
        if match:
            step = int(match.group(1))
            affinity = float(match.group(3))
            steps.append(step)
            affinities.append(affinity)

    return steps, affinities

def smooth_curve(x, y, window=100):
    """Apply moving average smoothing."""
    if len(y) < window:
        return x, y

    smoothed = []
    for i in range(len(y)):
        start = max(0, i - window // 2)
        end = min(len(y), i + window // 2)
        smoothed.append(np.mean(y[start:end]))

    return x, smoothed

def main():
    # Top 5 traces
    top_traces = [
        ('trace72_delta_from_trace70', 'trace72: delta_from_trace70'),
        ('trace70_gate_m1p5_from_trace61', 'trace70: gate_m1p5'),
        ('trace61_dynamic_pool', 'trace61: dynamic_pool'),
        ('trace78_aggressive_push', 'trace78: aggressive_push'),
        ('trace53_terminal_trace29_reward_L2only', 'trace53: terminal_reward'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot individual traces
    for idx, (trace_name, label) in enumerate(top_traces):
        ax = axes[idx]

        # Find log file
        log_files = list(Path('/share/liuyutian/tcrppo_v2/logs').glob(f'{trace_name}*train.log'))
        if not log_files:
            ax.text(0.5, 0.5, f'No log file found\nfor {trace_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        log_file = max(log_files, key=lambda p: p.stat().st_mtime)

        print(f"Processing {trace_name}...")
        steps, affinities = parse_log_affinity_curve(str(log_file))

        if steps is None or len(steps) == 0:
            ax.text(0.5, 0.5, f'No data found\nfor {trace_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Plot raw data (semi-transparent)
        ax.plot(steps, affinities, alpha=0.2, color=colors[idx], linewidth=0.5)

        # Plot smoothed curve
        _, smoothed = smooth_curve(steps, affinities, window=100)
        ax.plot(steps, smoothed, color=colors[idx], linewidth=2, label='Smoothed (window=100)')

        # Add horizontal line at y=0
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Affinity = 0')

        # Statistics
        mean_aff = np.mean(affinities[-500:]) if len(affinities) >= 500 else np.mean(affinities)
        max_aff = np.max(affinities[-500:]) if len(affinities) >= 500 else np.max(affinities)

        ax.set_title(f'{label}\nMean: {mean_aff:.3f}, Max: {max_aff:.3f}', fontsize=10)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Affinity')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        print(f"  Steps: {len(steps):,}, Mean: {mean_aff:.4f}, Max: {max_aff:.4f}")

    # Combined plot in the last subplot
    ax = axes[5]
    for idx, (trace_name, label) in enumerate(top_traces):
        log_files = list(Path('/share/liuyutian/tcrppo_v2/logs').glob(f'{trace_name}*train.log'))
        if not log_files:
            continue

        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        steps, affinities = parse_log_affinity_curve(str(log_file))

        if steps is None or len(steps) == 0:
            continue

        # Plot smoothed curve only
        _, smoothed = smooth_curve(steps, affinities, window=100)
        ax.plot(steps, smoothed, color=colors[idx], linewidth=2, label=label, alpha=0.8)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('All Top 5 Traces Combined', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Affinity')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    plt.tight_layout()

    # Save figure
    output_path = '/share/liuyutian/tcrppo_v2/logs/top5_traces_affinity_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    plt.close()

    # Create a zoomed version focusing on recent steps
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for idx, (trace_name, label) in enumerate(top_traces):
        log_files = list(Path('/share/liuyutian/tcrppo_v2/logs').glob(f'{trace_name}*train.log'))
        if not log_files:
            continue

        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        steps, affinities = parse_log_affinity_curve(str(log_file))

        if steps is None or len(steps) == 0:
            continue

        # Take last 10000 episodes
        if len(steps) > 10000:
            steps = steps[-10000:]
            affinities = affinities[-10000:]

        # Plot smoothed curve
        _, smoothed = smooth_curve(steps, affinities, window=50)
        ax.plot(steps, smoothed, color=colors[idx], linewidth=2.5, label=label, alpha=0.9)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Affinity = 0')
    ax.set_title('Top 5 Traces - Recent Training (Last 10K Episodes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Affinity', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()

    output_path_zoomed = '/share/liuyutian/tcrppo_v2/logs/top5_traces_affinity_curves_recent.png'
    plt.savefig(output_path_zoomed, dpi=300, bbox_inches='tight')
    print(f"Saved zoomed version to: {output_path_zoomed}")

if __name__ == '__main__':
    main()
