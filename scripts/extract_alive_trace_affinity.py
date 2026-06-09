#!/usr/bin/env python3
"""
Extract affinity statistics from alive trace logs.
"""

import re
import subprocess
import pandas as pd
from collections import defaultdict
from pathlib import Path

def is_trace_alive(trace_name):
    """Check if a trace is currently running."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return trace_name in result.stdout
    except:
        return False

def get_trace_config(trace_name):
    """Get trace configuration from config file."""
    config_path = f'/share/liuyutian/tcrppo_v2/configs/{trace_name}.yaml'
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            # Extract train_targets
            match = re.search(r'train_targets:\s*(.+)', content)
            if match:
                targets_file = match.group(1).strip()
                targets_path = f'/share/liuyutian/tcrppo_v2/{targets_file}'
                if Path(targets_path).exists():
                    with open(targets_path, 'r') as tf:
                        peptides = [line.strip() for line in tf if line.strip() and not line.startswith('#')]
                        return peptides
    except:
        pass
    return []

def parse_log_file(log_path, n_lines=5000):
    """Parse training log to extract affinity statistics."""
    try:
        result = subprocess.run(['tail', '-n', str(n_lines), log_path],
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
    except:
        return None

    # Extract episode data
    episodes = []
    peptide_affinities = defaultdict(list)

    # Pattern for episode lines
    episode_pattern = re.compile(
        r'Episode\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+R=([-\d.]+)\s+\|\s+Len=([\d.]+)\s+\|\s+A=([-\d.]+)'
    )

    # Pattern for step summary lines (contains mean affinity)
    step_pattern = re.compile(
        r'Step\s+[\d,]+\s+\|.*?\|\s+A:\s+([-\d.]+)'
    )

    last_step = 0
    mean_affinities = []

    for line in lines:
        # Try episode pattern
        match = episode_pattern.search(line)
        if match:
            episode_num = int(match.group(1))
            step = int(match.group(2))
            reward = float(match.group(3))
            length = float(match.group(4))
            affinity = float(match.group(5))

            episodes.append({
                'episode': episode_num,
                'step': step,
                'reward': reward,
                'length': length,
                'affinity': affinity
            })

            last_step = max(last_step, step)
            continue

        # Try step summary pattern
        match = step_pattern.search(line)
        if match:
            mean_aff = float(match.group(1))
            mean_affinities.append(mean_aff)

    if not episodes:
        return None

    # Calculate statistics from recent episodes
    recent_episodes = episodes[-500:] if len(episodes) >= 500 else episodes

    affinities = [ep['affinity'] for ep in recent_episodes]
    rewards = [ep['reward'] for ep in recent_episodes]

    stats = {
        'last_step': last_step,
        'num_episodes': len(episodes),
        'mean_affinity_last500': sum(affinities) / len(affinities) if affinities else None,
        'max_affinity_last500': max(affinities) if affinities else None,
        'min_affinity_last500': min(affinities) if affinities else None,
        'mean_reward_last500': sum(rewards) / len(rewards) if rewards else None,
        'mean_affinity_from_summary': mean_affinities[-1] if mean_affinities else None,
    }

    return stats

def main():
    # List of alive traces (from ps aux output)
    alive_traces = [
        'trace53_terminal_trace29_reward_L2only',
        'trace61_dynamic_pool',
        'trace62_multi_gates',
        'trace70_gate_m1p5_from_trace61',
        'trace72_delta_from_trace70',
        'trace73_curriculum_exploration',
        'trace78_aggressive_push',
        'trace83_curated_from_trace73',
        'trace84_push_to_zero',
        'trace86_per_step_reward',
    ]

    results = []

    for trace_name in alive_traces:
        print(f"Processing {trace_name}...")

        # Check if actually alive
        is_alive = is_trace_alive(trace_name)
        if not is_alive:
            print(f"  Warning: {trace_name} not found in ps aux")
            continue

        # Find log file
        log_files = list(Path('/share/liuyutian/tcrppo_v2/logs').glob(f'{trace_name}*train.log'))
        if not log_files:
            print(f"  No log file found")
            continue

        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"  Log: {log_file}")

        # Parse log
        stats = parse_log_file(str(log_file))
        if stats is None:
            print(f"  Failed to parse log")
            continue

        # Get peptides from config
        peptides = get_trace_config(trace_name)

        results.append({
            'trace': trace_name,
            'alive': True,
            'last_step': stats['last_step'],
            'num_episodes': stats['num_episodes'],
            'mean_affinity_last500': stats['mean_affinity_last500'],
            'max_affinity_last500': stats['max_affinity_last500'],
            'min_affinity_last500': stats['min_affinity_last500'],
            'mean_reward_last500': stats['mean_reward_last500'],
            'num_peptides': len(peptides),
            'peptides': ', '.join(peptides) if peptides else 'N/A'
        })

        print(f"  Last step: {stats['last_step']:,}")
        print(f"  Episodes: {stats['num_episodes']:,}")
        print(f"  Mean affinity (last 500): {stats['mean_affinity_last500']:.4f}")
        print(f"  Max affinity (last 500): {stats['max_affinity_last500']:.4f}")
        print(f"  Peptides: {len(peptides)}")
        print()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by mean affinity
    df_sorted = df.sort_values('mean_affinity_last500', ascending=False)

    print("\n" + "="*100)
    print("ALIVE TRACES SUMMARY (sorted by mean affinity)")
    print("="*100)
    print(df_sorted.to_string(index=False))

    # Save to CSV
    output_csv = '/share/liuyutian/tcrppo_v2/logs/alive_traces_affinity_summary_v2.csv'
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n\nSaved to: {output_csv}")

    # Print per-peptide breakdown for traces with peptide info
    print("\n" + "="*100)
    print("PER-PEPTIDE BREAKDOWN")
    print("="*100)

    for idx, row in df_sorted.iterrows():
        if row['peptides'] != 'N/A':
            print(f"\n{row['trace']}:")
            print(f"  Peptides: {row['peptides']}")
            print(f"  Mean affinity: {row['mean_affinity_last500']:.4f}")
            print(f"  Max affinity: {row['max_affinity_last500']:.4f}")

if __name__ == '__main__':
    main()
