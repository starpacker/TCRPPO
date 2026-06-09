#!/usr/bin/env python
"""Analyze partial FP32 evaluation results from logs."""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_log_file(log_path: str):
    """Parse tFoldScore entries from log file."""
    results = defaultdict(lambda: defaultdict(list))

    pattern = r'\[tFoldScore\].*affinity_logit=([-\d.]+).*cdr3b=(\S+) peptide=(\S+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                affinity_logit = float(match.group(1))
                cdr3b = match.group(2)
                peptide = match.group(3)
                results[peptide][cdr3b].append(affinity_logit)

    return results


def analyze_trace(trace_name: str, log_path: str):
    """Analyze results for one trace."""
    print(f"\n{'='*80}")
    print(f"Trace: {trace_name}")
    print(f"{'='*80}")

    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return None

    results = parse_log_file(log_path)

    if not results:
        print("No results found in log file")
        return None

    peptides = sorted(results.keys())
    print(f"\nPeptides evaluated: {len(peptides)}")

    peptide_stats = {}

    for peptide in peptides:
        tcrs = results[peptide]
        n_tcrs = len(tcrs)

        # Get max affinity per TCR (if multiple evaluations)
        max_affinities = [max(scores) for scores in tcrs.values()]

        mean_max = np.mean(max_affinities)
        max_overall = np.max(max_affinities)

        peptide_stats[peptide] = {
            'n_tcrs': n_tcrs,
            'mean_max_affinity': mean_max,
            'max_affinity': max_overall,
        }

    # Summary table
    print(f"\n{'Peptide':<15} {'N TCRs':<10} {'Mean Max':<12} {'Max':<12}")
    print("-" * 55)

    for peptide in peptides:
        stats = peptide_stats[peptide]
        print(f"{peptide:<15} {stats['n_tcrs']:<10} {stats['mean_max_affinity']:<12.4f} {stats['max_affinity']:<12.4f}")

    # Overall stats
    all_means = [s['mean_max_affinity'] for s in peptide_stats.values()]
    all_maxs = [s['max_affinity'] for s in peptide_stats.values()]

    print("-" * 55)
    print(f"{'OVERALL':<15} {len(peptides):<10} {np.mean(all_means):<12.4f} {np.max(all_maxs):<12.4f}")

    return {
        'trace_name': trace_name,
        'peptide_stats': peptide_stats,
        'overall_mean_of_means': np.mean(all_means),
        'overall_max': np.max(all_maxs),
    }


def main():
    output_dir = "results/fp32_eval_sequential_20260603_104721"

    traces = {
        'trace61': f"{output_dir}/trace61.log",
        'trace72': f"{output_dir}/trace72.log",
        'trace73': f"{output_dir}/trace73.log",
    }

    all_results = {}

    for trace_name, log_path in traces.items():
        result = analyze_trace(trace_name, log_path)
        if result:
            all_results[trace_name] = result

    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"\n{'Trace':<15} {'Mean of Means':<20} {'Overall Max':<20}")
        print("-" * 60)

        for trace_name in sorted(all_results.keys()):
            result = all_results[trace_name]
            print(f"{trace_name:<15} {result['overall_mean_of_means']:<20.4f} {result['overall_max']:<20.4f}")

        # Find best
        best_trace = max(all_results.items(), key=lambda x: x[1]['overall_mean_of_means'])
        print("-" * 60)
        print(f"\nBest trace (by mean of means): {best_trace[0]} ({best_trace[1]['overall_mean_of_means']:.4f})")


if __name__ == "__main__":
    main()
