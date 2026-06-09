#!/usr/bin/env python
"""Compare FP32 evaluation results and select best trace."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_result(result_path: str) -> Dict:
    """Load a single evaluation result."""
    with open(result_path, "r") as f:
        return json.load(f)


def compare_results(results_dir: str) -> None:
    """Compare evaluation results and generate report."""
    results_dir = Path(results_dir)

    # Load all results
    traces = {}
    for result_file in results_dir.glob("*_fp32_eval.json"):
        result = load_result(result_file)
        trace_name = result["trace_name"]
        traces[trace_name] = result

    if not traces:
        print("No evaluation results found!")
        return

    print("\n" + "=" * 80)
    print("FP32 Evaluation Results Comparison")
    print("=" * 80 + "\n")

    # Summary table
    print("Summary Statistics:")
    print("-" * 80)
    print(f"{'Trace':<35} {'Mean of Means':<15} {'Mean of Max':<15} {'Overall Max':<15}")
    print("-" * 80)

    best_trace = None
    best_mean_of_means = -float('inf')

    for trace_name in sorted(traces.keys()):
        result = traces[trace_name]
        summary = result["summary"]

        mean_of_means = summary["mean_of_mean_affinities"]
        mean_of_max = summary["mean_of_max_affinities"]
        overall_max = summary["overall_max_affinity"]

        print(f"{trace_name:<35} {mean_of_means:<15.4f} {mean_of_max:<15.4f} {overall_max:<15.4f}")

        if mean_of_means > best_mean_of_means:
            best_mean_of_means = mean_of_means
            best_trace = trace_name

    print("-" * 80)
    print(f"\nBest trace by mean affinity: {best_trace} (mean={best_mean_of_means:.4f})")
    print("-" * 80)

    # Per-peptide comparison
    print("\n\nPer-Peptide Mean Affinity Comparison:")
    print("-" * 100)

    # Get all peptides
    peptides = list(traces[list(traces.keys())[0]]["per_peptide"].keys())

    header = f"{'Peptide':<15}"
    for trace_name in sorted(traces.keys()):
        short_name = trace_name.replace("trace", "t").replace("_", "")[:12]
        header += f" {short_name:<12}"
    print(header)
    print("-" * 100)

    for peptide in peptides:
        row = f"{peptide:<15}"
        for trace_name in sorted(traces.keys()):
            mean_aff = traces[trace_name]["per_peptide"][peptide]["mean_affinity"]
            row += f" {mean_aff:<12.4f}"
        print(row)

    print("-" * 100)

    # Per-peptide max affinity comparison
    print("\n\nPer-Peptide Max Affinity Comparison:")
    print("-" * 100)

    print(header)
    print("-" * 100)

    for peptide in peptides:
        row = f"{peptide:<15}"
        for trace_name in sorted(traces.keys()):
            max_aff = traces[trace_name]["per_peptide"][peptide]["max_affinity"]
            row += f" {max_aff:<12.4f}"
        print(row)

    print("-" * 100)

    # Save comparison report
    report = {
        "best_trace": best_trace,
        "best_mean_of_means": best_mean_of_means,
        "summary_table": {
            trace_name: result["summary"]
            for trace_name, result in traces.items()
        },
        "per_peptide_comparison": {
            peptide: {
                trace_name: result["per_peptide"][peptide]
                for trace_name, result in traces.items()
            }
            for peptide in peptides
        }
    }

    report_path = results_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n\nComparison report saved to: {report_path}")
    print("\n" + "=" * 80)
    print(f"RECOMMENDED: Relaunch {best_trace} with FP32 tFold scorer")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 evaluation results")
    parser.add_argument("--results-dir", required=True, help="Directory with evaluation results")

    args = parser.parse_args()

    compare_results(args.results_dir)


if __name__ == "__main__":
    main()
