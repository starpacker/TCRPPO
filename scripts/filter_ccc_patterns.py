#!/usr/bin/env python3
"""Filter out TCRs with repetitive patterns (CCC, YYY, etc.)."""

import json
import argparse
from pathlib import Path
from collections import Counter


def has_repeat_pattern(seq, min_repeat=3):
    """Check if sequence has any amino acid repeated min_repeat times consecutively."""
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        if aa * min_repeat in seq:
            return True
    return False


def filter_tcrs(input_path, output_path, min_repeat=3, keep_ratio=1.0):
    """Filter TCRs to remove repetitive patterns.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        min_repeat: Minimum consecutive repeats to filter (default: 3)
        keep_ratio: Ratio of filtered data to keep (default: 1.0 = all)
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Collect all TCRs
    all_tcrs = []
    for peptide, records in data['by_peptide'].items():
        all_tcrs.extend(records)

    print(f"Total TCRs: {len(all_tcrs)}")

    # Filter out repetitive patterns
    filtered_tcrs = []
    removed_tcrs = []

    for record in all_tcrs:
        tcr = record['cdr3b']
        if has_repeat_pattern(tcr, min_repeat):
            removed_tcrs.append(record)
        else:
            filtered_tcrs.append(record)

    print(f"\nFiltering results:")
    print(f"  Removed: {len(removed_tcrs)} ({len(removed_tcrs)/len(all_tcrs)*100:.2f}%)")
    print(f"  Kept: {len(filtered_tcrs)} ({len(filtered_tcrs)/len(all_tcrs)*100:.2f}%)")

    # Analyze affinity distribution
    if filtered_tcrs:
        mean_aff_filtered = sum(r['affinity_logit'] for r in filtered_tcrs) / len(filtered_tcrs)
        print(f"\nAffinity statistics:")
        print(f"  Original mean: {sum(r['affinity_logit'] for r in all_tcrs) / len(all_tcrs):.4f}")
        print(f"  Filtered mean: {mean_aff_filtered:.4f}")
        print(f"  Removed mean: {sum(r['affinity_logit'] for r in removed_tcrs) / len(removed_tcrs):.4f}")

    # Apply keep_ratio if specified
    if keep_ratio < 1.0:
        import random
        random.seed(42)
        n_keep = int(len(filtered_tcrs) * keep_ratio)
        filtered_tcrs = random.sample(filtered_tcrs, n_keep)
        print(f"\nApplied keep_ratio={keep_ratio}: {len(filtered_tcrs)} TCRs")

    # Reorganize by peptide
    filtered_by_peptide = {}
    for record in filtered_tcrs:
        peptide = record['peptide']
        if peptide not in filtered_by_peptide:
            filtered_by_peptide[peptide] = []
        filtered_by_peptide[peptide].append(record)

    # Create output data structure
    output_data = {
        'metadata': {
            'total_records': len(filtered_tcrs),
            'unique_peptides': len(filtered_by_peptide),
            'min_repeat_filtered': min_repeat,
            'affinity_mean': mean_aff_filtered if filtered_tcrs else 0.0,
            'affinity_range': [
                min(r['affinity_logit'] for r in filtered_tcrs),
                max(r['affinity_logit'] for r in filtered_tcrs)
            ] if filtered_tcrs else [0, 0]
        },
        'by_peptide': filtered_by_peptide
    }

    # Save filtered data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving filtered data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Done! Filtered data saved.")

    # Print peptide distribution
    print(f"\nPeptide distribution (top 10):")
    peptide_counts = [(p, len(records)) for p, records in filtered_by_peptide.items()]
    peptide_counts.sort(key=lambda x: x[1], reverse=True)
    for peptide, count in peptide_counts[:10]:
        print(f"  {peptide}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Filter TCRs with repetitive patterns')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--min_repeat', type=int, default=3,
                        help='Minimum consecutive repeats to filter (default: 3)')
    parser.add_argument('--keep_ratio', type=float, default=1.0,
                        help='Ratio of filtered data to keep (default: 1.0)')

    args = parser.parse_args()

    filter_tcrs(args.input, args.output, args.min_repeat, args.keep_ratio)


if __name__ == '__main__':
    main()
