#!/usr/bin/env python3
"""
Generate comprehensive report of alive traces with affinity statistics.
"""

import pandas as pd
import yaml
from pathlib import Path

def load_peptides(trace_name):
    """Load peptides for a trace."""
    config_path = f'/share/liuyutian/tcrppo_v2/configs/{trace_name}.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'train_targets' in config:
                targets_file = config['train_targets']
                targets_path = f'/share/liuyutian/tcrppo_v2/{targets_file}'
                if Path(targets_path).exists():
                    with open(targets_path, 'r') as tf:
                        return [line.strip() for line in tf if line.strip() and not line.startswith('#')]
    except:
        pass
    return []

def main():
    # Load summary
    summary_df = pd.read_csv('/share/liuyutian/tcrppo_v2/logs/alive_traces_summary.csv')

    print("="*100)
    print("ALIVE TRACES COMPREHENSIVE REPORT")
    print("="*100)
    print(f"\nGenerated: 2026-06-01")
    print(f"Total alive traces: {len(summary_df)}")

    print("\n" + "="*100)
    print("SUMMARY TABLE (sorted by mean affinity, last 500 episodes)")
    print("="*100)
    print()

    # Format the table nicely
    for idx, row in summary_df.iterrows():
        print(f"{row['trace']}")
        print(f"  Last step:       {row['last_step']:>10,}")
        print(f"  Episodes:        {row['num_episodes']:>10,}")
        print(f"  Mean affinity:   {row['mean_affinity']:>10.4f}")
        print(f"  Max affinity:    {row['max_affinity']:>10.4f}")
        print(f"  Min affinity:    {row['min_affinity']:>10.4f}")
        print(f"  Mean reward:     {row['mean_reward']:>10.4f}")
        print(f"  Num peptides:    {row['num_peptides']:>10}")

        # Load and display peptides
        peptides = load_peptides(row['trace'])
        if peptides:
            print(f"  Peptides: {', '.join(peptides)}")
        print()

    print("\n" + "="*100)
    print("KEY OBSERVATIONS")
    print("="*100)

    best_trace = summary_df.iloc[0]
    print(f"\n1. Best performing trace (highest mean affinity):")
    print(f"   {best_trace['trace']}")
    print(f"   Mean affinity: {best_trace['mean_affinity']:.4f}")
    print(f"   Max affinity: {best_trace['max_affinity']:.4f}")

    print(f"\n2. Traces with positive max affinity (at least one TCR with affinity > 0):")
    positive_traces = summary_df[summary_df['max_affinity'] > 0]
    for idx, row in positive_traces.iterrows():
        print(f"   {row['trace']:45s}: max={row['max_affinity']:7.4f}, mean={row['mean_affinity']:7.4f}")

    print(f"\n3. Training progress (by step count):")
    sorted_by_step = summary_df.sort_values('last_step', ascending=False)
    for idx, row in sorted_by_step.head(5).iterrows():
        print(f"   {row['trace']:45s}: {row['last_step']:>10,} steps")

    print(f"\n4. Traces with different peptide sets:")
    peptide_counts = summary_df.groupby('num_peptides').size()
    for count, num_traces in peptide_counts.items():
        print(f"   {count} peptides: {num_traces} traces")

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    print("""
1. Focus on top 3 traces with highest mean affinity:
   - trace72_delta_from_trace70 (mean: -0.97)
   - trace70_gate_m1p5_from_trace61 (mean: -1.22)
   - trace61_dynamic_pool (mean: -1.22)

2. Traces with positive max affinity show promise:
   - These have generated at least some TCRs with positive binding affinity
   - Consider evaluating their generated TCRs for specificity

3. trace86_per_step_reward is very early (8K steps):
   - Too early to evaluate, needs more training

4. trace62_multi_gates and trace83_curated_from_trace73 are struggling:
   - Mean affinity < -4.0, may need intervention or termination
""")

    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("""
1. Run decoy specificity evaluation on top 3 traces
2. Generate 50 TCRs per peptide from best checkpoints
3. Compare against baseline (trace11, trace29)
4. Analyze what makes trace72 perform best (delta reward?)
""")

if __name__ == '__main__':
    main()
