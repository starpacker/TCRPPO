#!/usr/bin/env python3
"""
Extract high-quality TCRs from trace logs for SFT training.

Strategy:
1. Parse all tFoldScore records from trace logs
2. Filter by affinity threshold (e.g., affinity_logit > -2.0 or > 0.0)
3. Group by peptide
4. Save as JSON for SFT training
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_tfold_score_line(line: str) -> Dict:
    """Parse a tFoldScore log line.
    
    Example:
    [tFoldScore] ts=2026-05-28 06:21:29 source=extract_ok path_ms=8749.09 
    classify_ms=1.97 end_to_end_ms=8763.95 affinity_logit=-0.6653 conf=1.00 
    cdr3b=CALSIDHGSGNAQYFCCC peptide=GLCTLVAML hla=HLA-A*02:01
    """
    pattern = r'affinity_logit=([-\d.]+).*cdr3b=(\w+).*peptide=(\w+).*hla=([\w\*:]+)'
    match = re.search(pattern, line)
    
    if not match:
        return None
    
    affinity_logit = float(match.group(1))
    cdr3b = match.group(2)
    peptide = match.group(3)
    hla = match.group(4)
    
    return {
        'cdr3b': cdr3b,
        'peptide': peptide,
        'hla': hla,
        'affinity_logit': affinity_logit,
    }

def extract_from_log(log_path: Path, min_affinity: float = -2.0) -> List[Dict]:
    """Extract high-quality TCRs from a single log file."""
    records = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if '[tFoldScore]' not in line:
                continue
            
            record = parse_tfold_score_line(line)
            if record and record['affinity_logit'] >= min_affinity:
                records.append(record)
    
    return records

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory containing trace logs')
    parser.add_argument('--min_affinity', type=float, default=-2.0,
                        help='Minimum affinity threshold')
    parser.add_argument('--output', type=str, default='data/high_quality_tcrs.json',
                        help='Output JSON file')
    parser.add_argument('--trace_pattern', type=str, default='trace*.log',
                        help='Pattern to match trace log files')
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    log_files = sorted(logs_dir.glob(args.trace_pattern))
    
    print(f"Found {len(log_files)} log files matching '{args.trace_pattern}'")
    
    all_records = []
    by_peptide = defaultdict(list)
    
    for log_file in log_files:
        print(f"Processing {log_file.name}...", end=' ')
        records = extract_from_log(log_file, args.min_affinity)
        all_records.extend(records)
        
        for rec in records:
            by_peptide[rec['peptide']].append(rec)
        
        print(f"{len(records)} records")
    
    print(f"\n=== Summary ===")
    print(f"Total records: {len(all_records)}")
    print(f"Unique peptides: {len(by_peptide)}")
    print(f"Min affinity threshold: {args.min_affinity}")
    
    # Statistics by peptide
    print(f"\nRecords per peptide:")
    for peptide in sorted(by_peptide.keys(), key=lambda p: len(by_peptide[p]), reverse=True):
        records = by_peptide[peptide]
        affinities = [r['affinity_logit'] for r in records]
        print(f"  {peptide:15s}: {len(records):5d} TCRs, "
              f"affinity range [{min(affinities):.2f}, {max(affinities):.2f}]")
    
    # Affinity distribution
    affinities = [r['affinity_logit'] for r in all_records]
    print(f"\nAffinity distribution:")
    print(f"  Min: {min(affinities):.4f}")
    print(f"  Max: {max(affinities):.4f}")
    print(f"  Mean: {sum(affinities)/len(affinities):.4f}")
    
    # Count by bins
    bins = [
        (float('inf'), 0.0, 'A >= 0.0'),
        (0.0, -1.0, '-1.0 <= A < 0.0'),
        (-1.0, -2.0, '-2.0 <= A < -1.0'),
        (-2.0, float('-inf'), 'A < -2.0'),
    ]
    
    print(f"\nAffinity bins:")
    for upper, lower, label in bins:
        count = sum(1 for a in affinities if lower <= a < upper)
        pct = count / len(affinities) * 100
        print(f"  {label:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'total_records': len(all_records),
            'unique_peptides': len(by_peptide),
            'min_affinity': args.min_affinity,
            'affinity_range': [min(affinities), max(affinities)],
            'affinity_mean': sum(affinities) / len(affinities),
        },
        'by_peptide': {
            peptide: records
            for peptide, records in by_peptide.items()
        },
        'all_records': all_records,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    main()
