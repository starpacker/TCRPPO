#!/usr/bin/env python3
"""Compute comprehensive evaluation metrics and compare with paper results.

Reads TCRPPO test output files and produces:
  - Per-peptide statistics table (ERGO score, GMM, edit distance, diversity)
  - Overall aggregate metrics
  - Paper comparison table
  - Detailed CSV export for downstream analysis

Usage:
    python eval_metrics.py --result <result_file> [--out_dir <output_dir>]
    python eval_metrics.py --result results1.txt --result results2.txt --labels mcpas vdjdb
"""
import argparse
import csv
import os
import sys
import numpy as np
from collections import defaultdict, OrderedDict

from eval_utils import (
    parse_result_file, flatten_results, normalized_edit_distance, ensure_dir,
    AMINO_ACIDS,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------
def compute_peptide_metrics(records):
    """Compute metrics for a list of records belonging to one peptide.

    Returns dict with all scalar metrics.
    """
    n = len(records)
    if n == 0:
        return {}

    ergos = np.array([r["ergo_score"] for r in records])
    edits = np.array([r["edit_dist"] for r in records])
    gmms = np.array([r["gmm_likelihood"] for r in records])

    final_tcrs = [r["final_tcr"] for r in records]
    init_tcrs = [r["init_tcr"] for r in records]
    unique_finals = set(final_tcrs)
    changed = sum(1 for i, f in zip(init_tcrs, final_tcrs) if i != f)

    # Compute actual Levenshtein distances
    lev_dists = [normalized_edit_distance(r["init_tcr"], r["final_tcr"]) for r in records]

    # Amino acid distribution of generated TCRs
    aa_counts = defaultdict(int)
    total_aa = 0
    for tcr in final_tcrs:
        for aa in tcr:
            aa_counts[aa] += 1
            total_aa += 1

    # Mutation position distribution
    mutation_positions = []
    for r in records:
        init, final = r["init_tcr"], r["final_tcr"]
        for pos in range(min(len(init), len(final))):
            if init[pos] != final[pos]:
                mutation_positions.append(pos)

    return {
        "count": n,
        "avg_ergo": float(np.mean(ergos)),
        "std_ergo": float(np.std(ergos)),
        "median_ergo": float(np.median(ergos)),
        "max_ergo": float(np.max(ergos)),
        "min_ergo": float(np.min(ergos)),
        "ergo_gt_09": int(np.sum(ergos >= 0.9)),
        "ergo_gt_09_pct": float(np.sum(ergos >= 0.9) / n * 100),
        "ergo_gt_08": int(np.sum(ergos >= 0.8)),
        "ergo_gt_08_pct": float(np.sum(ergos >= 0.8) / n * 100),
        "ergo_gt_05": int(np.sum(ergos >= 0.5)),
        "ergo_gt_05_pct": float(np.sum(ergos >= 0.5) / n * 100),
        "avg_edit_dist": float(np.mean(edits)),
        "std_edit_dist": float(np.std(edits)),
        "avg_gmm": float(np.mean(gmms)),
        "std_gmm": float(np.std(gmms)),
        "avg_lev_dist": float(np.mean(lev_dists)),
        "num_unique": len(unique_finals),
        "unique_pct": float(len(unique_finals) / n * 100),
        "num_changed": changed,
        "changed_pct": float(changed / n * 100),
        "avg_tcr_len": float(np.mean([len(t) for t in final_tcrs])),
        "num_mutations": len(mutation_positions),
        "avg_mutations_per_seq": float(len(mutation_positions) / n) if n > 0 else 0,
    }


def compute_baseline_metrics(records):
    """Compute ERGO scores for initial (unoptimized) TCRs as baseline."""
    # edit_dist for initial TCR should be ~1.0 (no edits), gmm may vary
    # The "baseline" ERGO score is what ERGO would give to the init_tcr + peptide
    # We approximate this by noting that if init_tcr == final_tcr, ergo_score is the baseline
    unchanged = [r for r in records if r["init_tcr"] == r["final_tcr"]]
    if unchanged:
        return float(np.mean([r["ergo_score"] for r in unchanged]))
    return None


# ---------------------------------------------------------------------------
# Printing / formatting
# ---------------------------------------------------------------------------
def print_peptide_table(all_metrics, peptide_order=None):
    """Print the per-peptide summary table matching paper format."""
    if peptide_order is None:
        peptide_order = sorted(all_metrics.keys())

    header = (f"{'Peptide':<25} {'Count':>6} {'AvgERGO':>8} {'StdERGO':>8} "
              f"{'ERGO>0.9':>9} {'AvgGMM':>8} {'AvgEdit':>8} "
              f"{'Unique':>7} {'Uniq%':>6} {'Changed':>8}")
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for pep in peptide_order:
        m = all_metrics[pep]
        print(f"{pep:<25} {m['count']:>6} {m['avg_ergo']:>8.4f} {m['std_ergo']:>8.4f} "
              f"{m['ergo_gt_09_pct']:>8.1f}% {m['avg_gmm']:>8.4f} {m['avg_edit_dist']:>8.4f} "
              f"{m['num_unique']:>7} {m['unique_pct']:>5.1f}% {m['num_changed']:>8}")

    print(sep)


def print_overall_summary(overall):
    """Print aggregate summary."""
    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"Total samples:           {overall['count']}")
    print(f"Avg ERGO Score:          {overall['avg_ergo']:.4f} +/- {overall['std_ergo']:.4f}")
    print(f"Median ERGO Score:       {overall['median_ergo']:.4f}")
    print(f"ERGO > 0.9:              {overall['ergo_gt_09_pct']:.1f}% ({overall['ergo_gt_09']})")
    print(f"ERGO > 0.8:              {overall['ergo_gt_08_pct']:.1f}% ({overall['ergo_gt_08']})")
    print(f"ERGO > 0.5:              {overall['ergo_gt_05_pct']:.1f}% ({overall['ergo_gt_05']})")
    print(f"Avg GMM Likelihood:      {overall['avg_gmm']:.4f} +/- {overall['std_gmm']:.4f}")
    print(f"Avg Edit Conservation:   {overall['avg_edit_dist']:.4f} +/- {overall['std_edit_dist']:.4f}")
    print(f"Avg Levenshtein Dist:    {overall['avg_lev_dist']:.4f}")
    print(f"Unique Final TCRs:       {overall['num_unique']} ({overall['unique_pct']:.1f}%)")
    print(f"Changed TCRs:            {overall['num_changed']} ({overall['changed_pct']:.1f}%)")
    print(f"Avg Mutations/Sequence:  {overall['avg_mutations_per_seq']:.2f}")
    print(f"Avg Final TCR Length:    {overall['avg_tcr_len']:.1f}")
    print(f"{'='*60}")


def print_paper_comparison(overall):
    """Print comparison table with paper reproduction criteria."""
    print(f"\n{'='*60}")
    print("PAPER REPRODUCTION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Criterion':<35} {'Value':>10} {'Target':>12} {'Status':>8}")
    print("-" * 68)

    # Criterion 1: ERGO score improvement (paper expects avg 0.3-0.5 above baseline)
    ergo_status = "PASS" if overall["avg_ergo"] >= 0.7 else "CHECK"
    print(f"{'Avg ERGO Score':<35} {overall['avg_ergo']:>10.4f} {'>=0.70':>12} {ergo_status:>8}")

    # Criterion 2: High-affinity rate
    high_status = "PASS" if overall["ergo_gt_09_pct"] >= 50 else "CHECK"
    print(f"{'ERGO > 0.9 Rate':<35} {overall['ergo_gt_09_pct']:>9.1f}% {'>=50%':>12} {high_status:>8}")

    # Criterion 3: GMM stability
    gmm_status = "PASS" if overall["avg_gmm"] > 0 else "FAIL"
    print(f"{'Avg GMM Likelihood (positive)':<35} {overall['avg_gmm']:>10.4f} {'>0':>12} {gmm_status:>8}")

    # Criterion 4: Diversity
    div_status = "PASS" if overall["unique_pct"] >= 10 else "FAIL"
    print(f"{'Unique/Count Ratio':<35} {overall['unique_pct']:>9.1f}% {'>=10%':>12} {div_status:>8}")

    # Criterion 5: Modification rate
    mod_status = "PASS" if overall["changed_pct"] >= 50 else "CHECK"
    print(f"{'Changed TCR Rate':<35} {overall['changed_pct']:>9.1f}% {'>=50%':>12} {mod_status:>8}")

    print("-" * 68)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_per_peptide_csv(all_metrics, overall, out_path):
    """Export per-peptide metrics to CSV."""
    fieldnames = list(list(all_metrics.values())[0].keys())
    fieldnames.insert(0, "peptide")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pep in sorted(all_metrics.keys()):
            row = {"peptide": pep, **all_metrics[pep]}
            writer.writerow(row)
        row = {"peptide": "OVERALL", **overall}
        writer.writerow(row)
    print(f"Per-peptide CSV: {out_path}")


def export_detailed_csv(records, out_path):
    """Export all individual results to CSV for downstream analysis."""
    fieldnames = ["peptide", "init_tcr", "final_tcr", "ergo_score",
                  "edit_dist", "gmm_likelihood", "changed", "tcr_len",
                  "levenshtein_dist"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "peptide": r["peptide"],
                "init_tcr": r["init_tcr"],
                "final_tcr": r["final_tcr"],
                "ergo_score": f"{r['ergo_score']:.4f}",
                "edit_dist": f"{r['edit_dist']:.4f}",
                "gmm_likelihood": f"{r['gmm_likelihood']:.4f}",
                "changed": int(r["init_tcr"] != r["final_tcr"]),
                "tcr_len": len(r["final_tcr"]),
                "levenshtein_dist": f"{normalized_edit_distance(r['init_tcr'], r['final_tcr']):.4f}",
            })
    print(f"Detailed CSV:    {out_path}")


# ---------------------------------------------------------------------------
# Multi-result comparison
# ---------------------------------------------------------------------------
def compare_results(result_files, labels):
    """Compare metrics across multiple result files side by side."""
    print(f"\n{'='*80}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*80}")

    all_overalls = {}
    for rfile, label in zip(result_files, labels):
        pep_results = parse_result_file(rfile)
        all_records = flatten_results(pep_results)
        overall = compute_peptide_metrics(all_records)
        all_overalls[label] = overall

    metrics_to_compare = [
        ("Avg ERGO Score", "avg_ergo", ".4f"),
        ("Std ERGO Score", "std_ergo", ".4f"),
        ("ERGO > 0.9 %", "ergo_gt_09_pct", ".1f"),
        ("ERGO > 0.8 %", "ergo_gt_08_pct", ".1f"),
        ("Avg GMM", "avg_gmm", ".4f"),
        ("Avg Edit Dist", "avg_edit_dist", ".4f"),
        ("Unique %", "unique_pct", ".1f"),
        ("Changed %", "changed_pct", ".1f"),
        ("Avg Mutations", "avg_mutations_per_seq", ".2f"),
        ("Count", "count", "d"),
    ]

    header = f"{'Metric':<25}"
    for label in labels:
        header += f" {label:>15}"
    print(header)
    print("-" * len(header))

    for name, key, fmt in metrics_to_compare:
        row = f"{name:<25}"
        for label in labels:
            val = all_overalls[label].get(key, 0)
            row += f" {val:>15{fmt}}"
        print(row)

    # Find overlapping peptides for paired comparison
    all_pep_results = {}
    for rfile, label in zip(result_files, labels):
        all_pep_results[label] = parse_result_file(rfile)

    common_peps = set.intersection(*[set(pr.keys()) for pr in all_pep_results.values()])
    if common_peps:
        print(f"\n  Overlapping peptides ({len(common_peps)}): {', '.join(sorted(common_peps))}")
        print(f"\n  Per-peptide ERGO comparison (overlapping peptides):")
        header2 = f"  {'Peptide':<25}"
        for label in labels:
            header2 += f" {label:>12}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for pep in sorted(common_peps):
            row = f"  {pep:<25}"
            for label in labels:
                records = all_pep_results[label][pep]
                avg = np.mean([r["ergo_score"] for r in records])
                row += f" {avg:>12.4f}"
            print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze_single(result_file, out_dir=None):
    """Full analysis of a single result file."""
    pep_results = parse_result_file(result_file)
    if not pep_results:
        print(f"No results found in {result_file}")
        return

    # Per-peptide metrics
    all_metrics = OrderedDict()
    for pep in sorted(pep_results.keys()):
        all_metrics[pep] = compute_peptide_metrics(pep_results[pep])

    # Overall metrics
    all_records = flatten_results(pep_results)
    overall = compute_peptide_metrics(all_records)

    # Print tables
    print_peptide_table(all_metrics)
    print_overall_summary(overall)
    print_paper_comparison(overall)

    # Export CSVs
    if out_dir:
        ensure_dir(out_dir)
        base = os.path.splitext(os.path.basename(result_file))[0]
        export_per_peptide_csv(all_metrics, overall, os.path.join(out_dir, f"{base}_per_peptide.csv"))
        export_detailed_csv(all_records, os.path.join(out_dir, f"{base}_detailed.csv"))

    return all_metrics, overall


def main():
    parser = argparse.ArgumentParser(description="Compute TCRPPO evaluation metrics")
    parser.add_argument("--result", type=str, nargs="+", required=True,
                        help="Path(s) to result file(s)")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="Labels for each result file (for comparison)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for CSV exports (default: evaluation/results/)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if len(args.result) == 1:
        analyze_single(args.result[0], args.out_dir)
    else:
        labels = args.labels or [os.path.splitext(os.path.basename(f))[0] for f in args.result]
        if len(labels) != len(args.result):
            print("ERROR: Number of labels must match number of result files.")
            sys.exit(1)
        for rfile, label in zip(args.result, labels):
            print(f"\n{'#'*60}")
            print(f"# Analysis: {label}")
            print(f"{'#'*60}")
            analyze_single(rfile, args.out_dir)
        compare_results(args.result, labels)


if __name__ == "__main__":
    main()
