#!/usr/bin/env python3
"""
Extract per-peptide max affinity from alive trace logs.
"""

import re
import subprocess
import pandas as pd
import yaml
from collections import defaultdict
from pathlib import Path

def is_trace_alive(trace_name):
    """Check if a trace is currently running."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return trace_name in result.stdout
    except:
        return False

def get_trace_peptides(trace_name):
    """Get peptides from trace config."""
    config_path = f'/share/liuyutian/tcrppo_v2/configs/{trace_name}.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'train_targets' in config:
                targets_file = config['train_targets']
                targets_path = f'/share/liuyutian/tcrppo_v2/{targets_file}'
                if Path(targets_path).exists():
                    with open(targets_path, 'r') as tf:
                        peptides = [line.strip() for line in tf if line.strip() and not line.startswith('#')]
                        return peptides
    except Exception as e:
        print(f"  Error reading config: {e}")
    return []

def parse_log_for_peptide_affinity(log_path, peptides, n_lines=10000):
    """Parse log to extract per-peptide affinity statistics."""
    try:
        result = subprocess.run(['tail', '-n', str(n_lines), log_path],
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
    except:
        return None

    # Track affinity per peptide
    peptide_affinities = defaultdict(list)

    # Pattern to extract episode info
    # We need to track which peptide each episode is for
    # This is tricky because the log doesn't always show peptide name

    # Alternative: look for checkpoint evaluation results
    # Pattern: "Target: PEPTIDE | Mean: X.XX | Max: X.XX"
    eval_pattern = re.compile(
        r'Target:\s+(\w+)\s+\|\s+Mean:\s+([-\d.]+)\s+\|\s+Max:\s+([-\d.]+)'
    )

    # Pattern for episode lines with affinity
    episode_pattern = re.compile(
        r'Episode\s+\d+\s+\|\s+Step\s+\d+\s+\|\s+R=([-\d.]+)\s+\|\s+Len=[\d.]+\s+\|\s+A=([-\d.]+)'
    )

    # Extract from evaluation results
    for line in lines:
        match = eval_pattern.search(line)
        if match:
            peptide = match.group(1)
            mean_aff = float(match.group(2))
            max_aff = float(match.group(3))
            if peptide in peptides:
                peptide_affinities[peptide].append({
                    'mean': mean_aff,
                    'max': max_aff
                })

    # If no eval results, try to extract from episodes
    # This is less reliable without peptide labels
    if not peptide_affinities:
        all_affinities = []
        for line in lines:
            match = episode_pattern.search(line)
            if match:
                affinity = float(match.group(2))
                all_affinities.append(affinity)

        # If we have affinities but no per-peptide breakdown,
        # return aggregate stats
        if all_affinities:
            return {
                'aggregate': {
                    'mean': sum(all_affinities) / len(all_affinities),
                    'max': max(all_affinities),
                    'min': min(all_affinities),
                    'count': len(all_affinities)
                }
            }

    # Aggregate per-peptide stats
    result = {}
    for peptide, records in peptide_affinities.items():
        if records:
            result[peptide] = {
                'mean': sum(r['mean'] for r in records) / len(records),
                'max': max(r['max'] for r in records),
                'count': len(records)
            }

    return result if result else None

def extract_recent_affinity_stats(log_path, n_lines=5000):
    """Extract overall affinity stats from recent episodes."""
    try:
        result = subprocess.run(['tail', '-n', str(n_lines), log_path],
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
    except:
        return None

    episode_pattern = re.compile(
        r'Episode\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+R=([-\d.]+)\s+\|\s+Len=([\d.]+)\s+\|\s+A=([-\d.]+)'
    )

    episodes = []
    last_step = 0

    for line in lines:
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

    if not episodes:
        return None

    recent = episodes[-500:] if len(episodes) >= 500 else episodes
    affinities = [ep['affinity'] for ep in recent]
    rewards = [ep['reward'] for ep in recent]

    return {
        'last_step': last_step,
        'num_episodes': len(episodes),
        'mean_affinity': sum(affinities) / len(affinities),
        'max_affinity': max(affinities),
        'min_affinity': min(affinities),
        'mean_reward': sum(rewards) / len(rewards),
    }

def main():
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

    summary_results = []
    per_peptide_results = []

    print("="*100)
    print("EXTRACTING ALIVE TRACE STATISTICS")
    print("="*100)

    for trace_name in alive_traces:
        print(f"\n{trace_name}")
        print("-" * 80)

        # Check if alive
        if not is_trace_alive(trace_name):
            print("  WARNING: Not found in ps aux")
            continue

        # Find log file
        log_files = list(Path('/share/liuyutian/tcrppo_v2/logs').glob(f'{trace_name}*train.log'))
        if not log_files:
            print("  No log file found")
            continue

        log_file = max(log_files, key=lambda p: p.stat().st_mtime)

        # Get peptides
        peptides = get_trace_peptides(trace_name)
        print(f"  Peptides: {len(peptides)} - {', '.join(peptides[:5])}{'...' if len(peptides) > 5 else ''}")

        # Extract overall stats
        stats = extract_recent_affinity_stats(str(log_file))
        if stats:
            print(f"  Last step: {stats['last_step']:,}")
            print(f"  Mean affinity (last 500 eps): {stats['mean_affinity']:.4f}")
            print(f"  Max affinity (last 500 eps): {stats['max_affinity']:.4f}")
            print(f"  Mean reward (last 500 eps): {stats['mean_reward']:.4f}")

            summary_results.append({
                'trace': trace_name,
                'last_step': stats['last_step'],
                'num_episodes': stats['num_episodes'],
                'mean_affinity': stats['mean_affinity'],
                'max_affinity': stats['max_affinity'],
                'min_affinity': stats['min_affinity'],
                'mean_reward': stats['mean_reward'],
                'num_peptides': len(peptides),
            })

        # Extract per-peptide stats
        peptide_stats = parse_log_for_peptide_affinity(str(log_file), peptides)
        if peptide_stats and 'aggregate' not in peptide_stats:
            print(f"  Per-peptide max affinity:")
            for peptide in sorted(peptide_stats.keys(), key=lambda p: peptide_stats[p]['max'], reverse=True):
                pep_stat = peptide_stats[peptide]
                print(f"    {peptide:15s}: max={pep_stat['max']:7.4f}  mean={pep_stat['mean']:7.4f}")

                per_peptide_results.append({
                    'trace': trace_name,
                    'peptide': peptide,
                    'max_affinity': pep_stat['max'],
                    'mean_affinity': pep_stat['mean'],
                    'count': pep_stat['count']
                })

    # Save summary
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary = df_summary.sort_values('mean_affinity', ascending=False)

        print("\n" + "="*100)
        print("SUMMARY TABLE (sorted by mean affinity)")
        print("="*100)
        print(df_summary.to_string(index=False))

        output_csv = '/share/liuyutian/tcrppo_v2/logs/alive_traces_summary.csv'
        df_summary.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")

    # Save per-peptide
    if per_peptide_results:
        df_peptide = pd.DataFrame(per_peptide_results)
        df_peptide = df_peptide.sort_values(['trace', 'max_affinity'], ascending=[True, False])

        output_csv = '/share/liuyutian/tcrppo_v2/logs/alive_traces_per_peptide.csv'
        df_peptide.to_csv(output_csv, index=False)
        print(f"Per-peptide data saved to: {output_csv}")

        # Print top peptides per trace
        print("\n" + "="*100)
        print("TOP 3 PEPTIDES PER TRACE")
        print("="*100)
        for trace in df_peptide['trace'].unique():
            trace_data = df_peptide[df_peptide['trace'] == trace].head(3)
            print(f"\n{trace}:")
            for _, row in trace_data.iterrows():
                print(f"  {row['peptide']:15s}: {row['max_affinity']:7.4f}")

if __name__ == '__main__':
    main()
