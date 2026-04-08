#!/usr/bin/env python3
"""Analyze TCRPPO test results to reproduce paper metrics."""
import sys
import numpy as np
from collections import defaultdict


def analyze(result_file):
    peptide_results = defaultdict(list)

    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            peptide = parts[0]
            init_tcr = parts[1]
            final_tcr = parts[2]
            ergo_score = float(parts[3])
            seq_edit_dist = float(parts[4])
            gmm_likelihood = float(parts[5])
            peptide_results[peptide].append({
                'init': init_tcr,
                'final': final_tcr,
                'ergo': ergo_score,
                'edit_dist': seq_edit_dist,
                'gmm': gmm_likelihood
            })

    if not peptide_results:
        print("No results found in file.")
        return

    print(f"{'Peptide':<25} {'Count':>6} {'AvgERGO':>8} {'ERGO>0.9':>9} "
          f"{'AvgGMM':>8} {'AvgEdit':>8} {'Unique':>7} {'Changed':>8}")
    print("-" * 95)

    all_ergos = []
    all_gmms = []
    all_edits = []
    total_unique = 0
    total_count = 0
    total_changed = 0

    for peptide in sorted(peptide_results.keys()):
        results = peptide_results[peptide]
        n = len(results)
        ergos = [r['ergo'] for r in results]
        gmms = [r['gmm'] for r in results]
        edits = [r['edit_dist'] for r in results]
        unique_tcrs = len(set(r['final'] for r in results))
        changed = sum(1 for r in results if r['init'] != r['final'])
        high_ergo = sum(1 for e in ergos if e >= 0.9)

        all_ergos.extend(ergos)
        all_gmms.extend(gmms)
        all_edits.extend(edits)
        total_unique += unique_tcrs
        total_count += n
        total_changed += changed

        print(f"{peptide:<25} {n:>6} {np.mean(ergos):>8.4f} {high_ergo/n*100:>8.1f}% "
              f"{np.mean(gmms):>8.4f} {np.mean(edits):>8.4f} {unique_tcrs:>7} {changed:>8}")

    print("-" * 95)
    high_total = sum(1 for e in all_ergos if e >= 0.9)
    print(f"{'OVERALL':<25} {total_count:>6} {np.mean(all_ergos):>8.4f} "
          f"{high_total/len(all_ergos)*100:>8.1f}% "
          f"{np.mean(all_gmms):>8.4f} {np.mean(all_edits):>8.4f} "
          f"{total_unique:>7} {total_changed:>8}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <result_file>")
        print("  e.g. python analyze_results.py results/ae_mcpas_results.txt")
        sys.exit(1)
    analyze(sys.argv[1])
