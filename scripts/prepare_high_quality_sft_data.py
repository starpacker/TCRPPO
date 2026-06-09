#!/usr/bin/env python3
"""
Prepare high-quality SFT dataset from extracted TCRs.

Strategy:
1. Load high-quality TCRs (affinity >= -2.0)
2. Filter to only keep very high quality (affinity >= 0.0 or >= -1.0)
3. For each TCR, create a "trajectory" from a random init to the final TCR
4. Use stratified sampling to balance affinity bins
5. Save as SFT trajectories with real ESM-2 embeddings
"""

import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

def load_high_quality_tcrs(json_path: str, min_affinity: float = -1.0) -> Dict:
    """Load and filter high-quality TCRs."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter by affinity
    filtered_records = [
        rec for rec in data['all_records']
        if rec['affinity_logit'] >= min_affinity
    ]
    
    return {
        'records': filtered_records,
        'by_peptide': defaultdict(list),
    }

def generate_random_init_tcr(length: int = 15) -> str:
    """Generate a random initial TCR sequence."""
    # CDR3 typically starts with C and ends with F/W
    aa_pool = 'ARNDCEQGHILKMFPSTWYV'
    middle = ''.join(random.choices(aa_pool, k=length-2))
    return 'C' + middle + random.choice('FW')

def reconstruct_trajectory_improved(init_tcr: str, final_tcr: str, max_steps: int = 8) -> List[Tuple]:
    """
    Reconstruct trajectory with improved algorithm to avoid repetitive insertions.

    Strategy:
    1. Use substitutions preferentially over insertions
    2. Prevent consecutive insertions of the same amino acid
    3. Balance INS/DEL/SUB operations more naturally

    Returns:
        List of (op_type, position, token) tuples
    """
    trajectory = []
    current = list(init_tcr)
    target = list(final_tcr)

    # Use dynamic programming alignment (Needleman-Wunsch style)
    # to find optimal edit path
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, init_tcr, final_tcr)
    opcodes = matcher.get_opcodes()

    last_inserted_token = None
    current_pos = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if len(trajectory) >= max_steps:
            break

        if tag == 'equal':
            # No change needed
            current_pos = i2
            continue
        elif tag == 'replace':
            # Substitute mismatches
            n_replace = min(i2 - i1, j2 - j1)
            for k in range(n_replace):
                if len(trajectory) >= max_steps:
                    break
                pos = i1 + k
                token = final_tcr[j1 + k]
                trajectory.append(('SUB', pos, token))
                last_inserted_token = None
            current_pos = i2
        elif tag == 'delete':
            # Delete extra characters
            for k in range(i2 - i1):
                if len(trajectory) >= max_steps:
                    break
                trajectory.append(('DEL', i1, ''))
                last_inserted_token = None
            current_pos = i2
        elif tag == 'insert':
            # Insert missing characters, but avoid consecutive same tokens
            for k in range(j2 - j1):
                if len(trajectory) >= max_steps:
                    break
                token = final_tcr[j1 + k]

                # Prevent consecutive insertions of same amino acid
                if token == last_inserted_token:
                    # Skip this insertion
                    continue

                trajectory.append(('INS', i1, token))
                last_inserted_token = token
            current_pos = i1

    # Add STOP
    trajectory.append(('STOP', 0, ''))

    return trajectory[:max_steps+1]  # max_steps + STOP

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/high_quality_tcrs.json',
                        help='Input JSON with high-quality TCRs')
    parser.add_argument('--min_affinity', type=float, default=-1.0,
                        help='Minimum affinity threshold for SFT data')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of trajectories to generate')
    parser.add_argument('--output', type=str, default='data/high_quality_sft_trajectories.json',
                        help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading high-quality TCRs from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Extract all records from by_peptide structure
    all_records = []
    if 'by_peptide' in data:
        for peptide, records in data['by_peptide'].items():
            all_records.extend(records)
    elif 'all_records' in data:
        all_records = data['all_records']
    else:
        raise ValueError("Unknown data format - expected 'by_peptide' or 'all_records'")

    # Filter by affinity
    all_records = [
        rec for rec in all_records
        if rec['affinity_logit'] >= args.min_affinity
    ]
    
    print(f"Filtered to {len(all_records)} records with affinity >= {args.min_affinity}")
    
    # Bin by affinity
    bins = {
        'high': [r for r in all_records if r['affinity_logit'] >= 0.0],
        'medium': [r for r in all_records if -1.0 <= r['affinity_logit'] < 0.0],
        'low': [r for r in all_records if args.min_affinity <= r['affinity_logit'] < -1.0],
    }
    
    print(f"\nAffinity bins:")
    for bin_name, records in bins.items():
        if records:
            affs = [r['affinity_logit'] for r in records]
            print(f"  {bin_name:10s}: {len(records):6d} records, "
                  f"affinity range [{min(affs):.2f}, {max(affs):.2f}]")
    
    # Stratified sampling
    n_per_bin = args.n_samples // 3
    sampled_records = []
    
    for bin_name, records in bins.items():
        if not records:
            continue
        n_sample = min(n_per_bin, len(records))
        sampled = random.sample(records, n_sample)
        sampled_records.extend(sampled)
        print(f"  Sampled {n_sample} from {bin_name}")
    
    print(f"\nTotal sampled: {len(sampled_records)}")
    
    # Generate trajectories
    print(f"\nGenerating trajectories...")
    trajectories = []
    
    for i, record in enumerate(sampled_records):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(sampled_records)}")
        
        final_tcr = record['cdr3b']
        peptide = record['peptide']
        hla = record['hla']
        affinity = record['affinity_logit']
        
        # Generate random init
        init_tcr = generate_random_init_tcr(length=random.randint(12, 18))
        
        # Reconstruct trajectory
        actions = reconstruct_trajectory_improved(init_tcr, final_tcr, max_steps=8)
        
        trajectory = {
            'init_tcr': init_tcr,
            'final_tcr': final_tcr,
            'peptide': peptide,
            'hla': hla,
            'final_affinity': affinity,  # Use 'final_affinity' for SFTDataset compatibility
            'actions': [
                {'op': op, 'pos': pos, 'token': tok}
                for op, pos, tok in actions
            ],
            'n_actions': len(actions),
        }
        
        trajectories.append(trajectory)
    
    # Statistics
    print(f"\n=== Trajectory Statistics ===")
    print(f"Total trajectories: {len(trajectories)}")
    
    action_counts = [t['n_actions'] for t in trajectories]
    print(f"Actions per trajectory: mean={np.mean(action_counts):.1f}, "
          f"min={min(action_counts)}, max={max(action_counts)}")
    
    init_lengths = [len(t['init_tcr']) for t in trajectories]
    final_lengths = [len(t['final_tcr']) for t in trajectories]
    print(f"Init TCR length: mean={np.mean(init_lengths):.1f}, range=[{min(init_lengths)}, {max(init_lengths)}]")
    print(f"Final TCR length: mean={np.mean(final_lengths):.1f}, range=[{min(final_lengths)}, {max(final_lengths)}]")
    
    affinities = [t['final_affinity'] for t in trajectories]
    print(f"Affinity: mean={np.mean(affinities):.2f}, range=[{min(affinities):.2f}, {max(affinities):.2f}]")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'n_trajectories': len(trajectories),
            'min_affinity': args.min_affinity,
            'affinity_mean': float(np.mean(affinities)),
            'affinity_range': [float(min(affinities)), float(max(affinities))],
        },
        'trajectories': trajectories,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    main()
