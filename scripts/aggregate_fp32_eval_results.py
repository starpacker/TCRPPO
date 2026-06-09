#!/usr/bin/env python
"""Aggregate FP32 evaluation results from test_tcrs.py outputs."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_eval_result(result_dir: str) -> Dict:
    """Load evaluation result from test_tcrs.py output."""
    result_file = Path(result_dir) / "evaluation_results.json"
    if not result_file.exists():
        return None

    with open(result_file, "r") as f:
        return json.load(f)


def aggregate_results(results_dir: str) -> None:
    """Aggregate results from three traces."""
    results_dir = Path(results_dir)

    traces = ["trace61", "trace72", "trace73"]
    trace_results = {}

    print("\n" + "=" * 80)
    print("FP32 Evaluation Results Aggregation")
    print("=" * 80 + "\n")

    # Load all results
    for trace in traces:
        trace_dir = results_dir / trace
        result = load_eval_result(trace_dir)
        if result:
            trace_results[trace] = result
            print(f"Loaded {trace}: {len(result.get('per_peptide', {}))} peptides")
        else:
            print(f"WARNING: No results found for {trace}")

    if not trace_results:
        print("ERROR: No evaluation results found!")
        return

    print("\n" + "-" * 80)
    print("Summary Statistics (Mean Affinity)")
    print("-" * 80)
    print(f"{'Trace':<30} {'Mean Affinity':<20} {'Max Affinity':<20}")
    print("-" * 80)

    best_trace = None
    best_mean_affinity = -float('inf')

    summary_data = {}

    for trace, result in sorted(trace_results.items()):
        per_peptide = result.get("per_peptide", {})

        # Calculate mean and max affinity across all peptides
        all_means = []
        all_maxs = []

        for peptide, data in per_peptide.items():
            scores = data.get("affinity_scores", {}).get("tfold", [])
            if scores:
                all_means.append(np.mean(scores))
                all_maxs.append(np.max(scores))

        if all_means:
            mean_of_means = np.mean(all_means)
            max_of_maxs = np.max(all_maxs)

            summary_data[trace] = {
                "mean_affinity": mean_of_means,
                "max_affinity": max_of_maxs,
                "mean_of_means": mean_of_means,
                "n_peptides": len(all_means),
            }

            print(f"{trace:<30} {mean_of_means:<20.4f} {max_of_maxs:<20.4f}")

            if mean_of_means > best_mean_affinity:
                best_mean_affinity = mean_of_means
                best_trace = trace

    print("-" * 80)
    print(f"\nBest trace by mean affinity: {best_trace} ({best_mean_affinity:.4f})")
    print("-" * 80)

    # Per-peptide comparison
    print("\n\nPer-Peptide Mean Affinity Comparison:")
    print("-" * 100)

    # Get all peptides (from first trace)
    first_trace = list(trace_results.keys())[0]
    peptides = list(trace_results[first_trace].get("per_peptide", {}).keys())

    header = f"{'Peptide':<15}"
    for trace in sorted(trace_results.keys()):
        header += f" {trace:<20}"
    print(header)
    print("-" * 100)

    for peptide in peptides:
        row = f"{peptide:<15}"
        for trace in sorted(trace_results.keys()):
            per_peptide = trace_results[trace].get("per_peptide", {})
            if peptide in per_peptide:
                scores = per_peptide[peptide].get("affinity_scores", {}).get("tfold", [])
                mean_score = np.mean(scores) if scores else 0.0
                row += f" {mean_score:<20.4f}"
            else:
                row += f" {'N/A':<20}"
        print(row)

    print("-" * 100)

    # Save summary
    summary = {
        "best_trace": best_trace,
        "best_mean_affinity": best_mean_affinity,
        "traces": summary_data,
    }

    summary_file = results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {summary_file}")
    print("\n" + "=" * 80)
    print(f"RECOMMENDED: Relaunch {best_trace} with FP32 tFold scorer")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate FP32 evaluation results")
    parser.add_argument("--results-dir", required=True, help="Directory with evaluation results")

    args = parser.parse_args()

    aggregate_results(args.results_dir)


if __name__ == "__main__":
    main()
