#!/usr/bin/env python3
"""
Query all alive traces and extract their mean affinity and per-peptide max affinity.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

def is_trace_alive(trace_name):
    """Check if a trace is currently running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        return trace_name in result.stdout
    except:
        return False

def extract_affinity_from_tensorboard(trace_dir):
    """Extract affinity metrics from tensorboard logs."""
    from tensorboard.backend.event_processing import event_accumulator

    tb_dir = os.path.join(trace_dir, 'tensorboard')
    if not os.path.exists(tb_dir):
        return None

    event_files = glob.glob(os.path.join(tb_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None

    # Use the most recent event file
    event_file = max(event_files, key=os.path.getmtime)

    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Get available tags
        tags = ea.Tags()

        result = {
            'mean_affinity': None,
            'per_peptide_max': {},
            'last_step': 0
        }

        # Extract mean affinity
        if 'scalars' in tags and 'episode/mean_affinity' in tags['scalars']:
            affinity_events = ea.Scalars('episode/mean_affinity')
            if affinity_events:
                result['mean_affinity'] = affinity_events[-1].value
                result['last_step'] = affinity_events[-1].step

        # Extract per-peptide max affinity
        if 'scalars' in tags:
            for tag in tags['scalars']:
                if tag.startswith('peptide_max_affinity/'):
                    peptide = tag.replace('peptide_max_affinity/', '')
                    events = ea.Scalars(tag)
                    if events:
                        result['per_peptide_max'][peptide] = events[-1].value

        return result
    except Exception as e:
        print(f"Error reading tensorboard for {trace_dir}: {e}")
        return None

def extract_affinity_from_logs(trace_dir):
    """Extract affinity from training logs as fallback."""
    log_files = glob.glob(os.path.join('/share/liuyutian/tcrppo_v2/logs', f'{os.path.basename(trace_dir)}*train.log'))

    if not log_files:
        return None

    log_file = max(log_files, key=os.path.getmtime)

    try:
        # Read last 1000 lines
        result = subprocess.run(
            ['tail', '-n', '1000', log_file],
            capture_output=True,
            text=True
        )

        lines = result.stdout.split('\n')

        # Look for affinity metrics
        mean_affinities = []
        peptide_max = {}

        for line in lines:
            if 'mean_affinity' in line.lower():
                # Try to extract the value
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'mean_affinity' in part.lower() and i + 1 < len(parts):
                        try:
                            val = float(parts[i + 1].strip(','))
                            mean_affinities.append(val)
                        except:
                            pass

            # Look for per-peptide max
            if 'peptide_max_affinity' in line.lower() or 'max_aff' in line.lower():
                # Extract peptide name and value
                pass  # This would need more specific parsing

        if mean_affinities:
            return {
                'mean_affinity': mean_affinities[-1],
                'per_peptide_max': peptide_max,
                'last_step': None
            }

        return None
    except Exception as e:
        print(f"Error reading log for {trace_dir}: {e}")
        return None

def main():
    output_dir = Path('/share/liuyutian/tcrppo_v2/output')

    # Find all trace directories
    trace_dirs = sorted([d for d in output_dir.glob('trace*') if d.is_dir()])

    results = []

    for trace_dir in trace_dirs:
        trace_name = trace_dir.name

        # Skip non-trace directories (like trace61_*.png files)
        if not trace_name.startswith('trace') or not trace_name[5:].split('_')[0].isdigit():
            continue

        # Check if alive
        is_alive = is_trace_alive(trace_name)

        # Try to extract affinity from tensorboard
        affinity_data = extract_affinity_from_tensorboard(str(trace_dir))

        # Fallback to logs if tensorboard fails
        if affinity_data is None:
            affinity_data = extract_affinity_from_logs(str(trace_dir))

        if affinity_data is None:
            affinity_data = {
                'mean_affinity': None,
                'per_peptide_max': {},
                'last_step': None
            }

        results.append({
            'trace': trace_name,
            'alive': is_alive,
            'mean_affinity': affinity_data['mean_affinity'],
            'last_step': affinity_data['last_step'],
            'num_peptides': len(affinity_data['per_peptide_max']),
            'peptide_max_affinities': affinity_data['per_peptide_max']
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Filter to alive traces
    alive_df = df[df['alive'] == True].copy()

    print("\n" + "="*80)
    print("ALIVE TRACES SUMMARY")
    print("="*80)
    print(f"\nTotal traces: {len(df)}")
    print(f"Alive traces: {len(alive_df)}")
    print("\n" + "-"*80)

    if len(alive_df) > 0:
        # Sort by mean affinity (descending)
        alive_df_sorted = alive_df.sort_values('mean_affinity', ascending=False, na_position='last')

        for idx, row in alive_df_sorted.iterrows():
            print(f"\n{row['trace']}")
            print(f"  Status: ALIVE")
            print(f"  Last step: {row['last_step']}")
            print(f"  Mean affinity: {row['mean_affinity']:.4f}" if row['mean_affinity'] is not None else "  Mean affinity: N/A")
            print(f"  Num peptides tracked: {row['num_peptides']}")

            if row['peptide_max_affinities']:
                print(f"  Per-peptide max affinity:")
                for peptide, aff in sorted(row['peptide_max_affinities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"    {peptide}: {aff:.4f}")

    # Save to CSV
    output_csv = '/share/liuyutian/tcrppo_v2/logs/alive_traces_affinity_summary.csv'
    alive_df_sorted.to_csv(output_csv, index=False)
    print(f"\n\nSaved to: {output_csv}")

    # Also save detailed per-peptide data
    detailed_rows = []
    for idx, row in alive_df.iterrows():
        for peptide, aff in row['peptide_max_affinities'].items():
            detailed_rows.append({
                'trace': row['trace'],
                'peptide': peptide,
                'max_affinity': aff,
                'mean_affinity': row['mean_affinity'],
                'last_step': row['last_step']
            })

    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv = '/share/liuyutian/tcrppo_v2/logs/alive_traces_per_peptide_affinity.csv'
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"Detailed per-peptide data saved to: {detailed_csv}")

if __name__ == '__main__':
    main()
