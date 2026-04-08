#!/usr/bin/env python3
"""Visualization suite for TCRPPO evaluation results.

Generates publication-quality plots:
  - ERGO score distributions (per-peptide and overall)
  - GMM likelihood distributions
  - Edit distance distributions
  - Per-peptide bar charts
  - Mutation position heatmap
  - Amino acid substitution matrix
  - TCR length distribution comparison
  - Score correlation scatter plots

Usage:
    python visualize.py --result <result_file> [--out_dir <output_dir>]
"""
import argparse
import os
import sys
import numpy as np
from collections import defaultdict, Counter

from eval_utils import (
    parse_result_file, flatten_results, normalized_edit_distance,
    ensure_dir, AMINO_ACIDS,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


def setup_style():
    """Set up a clean plot style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })
    if HAS_SNS:
        sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# 1. ERGO Score Distributions
# ---------------------------------------------------------------------------
def plot_ergo_distribution(pep_results, out_dir):
    """Plot ERGO score distribution: overall histogram + per-peptide violins."""
    all_scores = []
    for records in pep_results.values():
        all_scores.extend([r["ergo_score"] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall histogram
    ax = axes[0]
    ax.hist(all_scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.9, color="red", linestyle="--", linewidth=1.5, label="Threshold=0.9")
    ax.axvline(x=np.mean(all_scores), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean={np.mean(all_scores):.3f}")
    ax.set_xlabel("ERGO Binding Score")
    ax.set_ylabel("Count")
    ax.set_title("Overall ERGO Score Distribution")
    ax.legend()

    # Per-peptide boxplot
    ax = axes[1]
    peptides = sorted(pep_results.keys())
    data = [np.array([r["ergo_score"] for r in pep_results[p]]) for p in peptides]
    short_labels = [p[:12] + ".." if len(p) > 14 else p for p in peptides]
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True, vert=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("ERGO Score")
    ax.set_title("Per-Peptide ERGO Score Distribution")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(out_dir, "ergo_score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 2. GMM & Edit Distance Distributions
# ---------------------------------------------------------------------------
def plot_gmm_edit_distributions(pep_results, out_dir):
    """Plot GMM likelihood and edit distance distributions."""
    all_records = flatten_results(pep_results)
    gmms = [r["gmm_likelihood"] for r in all_records]
    edits = [r["edit_dist"] for r in all_records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(gmms, bins=50, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(gmms), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean={np.mean(gmms):.3f}")
    ax.set_xlabel("GMM TCR-likeness Score")
    ax.set_ylabel("Count")
    ax.set_title("GMM Likelihood Distribution")
    ax.legend()

    ax = axes[1]
    ax.hist(edits, bins=50, color="sandybrown", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(edits), color="blue", linestyle="-", linewidth=1.5,
               label=f"Mean={np.mean(edits):.3f}")
    ax.set_xlabel("Sequence Conservation (1 - edit_dist)")
    ax.set_ylabel("Count")
    ax.set_title("Edit Distance Distribution")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "gmm_edit_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Per-Peptide Bar Charts
# ---------------------------------------------------------------------------
def plot_per_peptide_bars(pep_results, out_dir):
    """Bar chart comparing key metrics across peptides."""
    peptides = sorted(pep_results.keys())
    metrics = {}
    for p in peptides:
        records = pep_results[p]
        ergos = [r["ergo_score"] for r in records]
        gmms = [r["gmm_likelihood"] for r in records]
        metrics[p] = {
            "avg_ergo": np.mean(ergos),
            "ergo_gt_09": sum(1 for e in ergos if e >= 0.9) / len(ergos) * 100,
            "avg_gmm": np.mean(gmms),
            "unique_pct": len(set(r["final_tcr"] for r in records)) / len(records) * 100,
        }

    short_labels = [p[:12] + ".." if len(p) > 14 else p for p in peptides]
    x = np.arange(len(peptides))
    width = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Avg ERGO
    ax = axes[0, 0]
    vals = [metrics[p]["avg_ergo"] for p in peptides]
    bars = ax.bar(x, vals, width, color="steelblue")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("Avg ERGO Score")
    ax.set_title("Average ERGO Binding Score per Peptide")

    # ERGO > 0.9 %
    ax = axes[0, 1]
    vals = [metrics[p]["ergo_gt_09"] for p in peptides]
    ax.bar(x, vals, width, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("% with ERGO > 0.9")
    ax.set_title("High-Affinity Rate per Peptide")

    # Avg GMM
    ax = axes[1, 0]
    vals = [metrics[p]["avg_gmm"] for p in peptides]
    ax.bar(x, vals, width, color="mediumpurple")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("Avg GMM Likelihood")
    ax.set_title("TCR-likeness (GMM) per Peptide")

    # Diversity
    ax = axes[1, 1]
    vals = [metrics[p]["unique_pct"] for p in peptides]
    ax.bar(x, vals, width, color="seagreen")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("Unique TCR %")
    ax.set_title("Diversity (Unique %) per Peptide")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "per_peptide_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 4. Mutation Analysis
# ---------------------------------------------------------------------------
def plot_mutation_heatmap(pep_results, out_dir, max_len=27):
    """Heatmap of mutation positions across TCR CDR3 positions."""
    all_records = flatten_results(pep_results)
    mutation_matrix = np.zeros((20, max_len))  # amino acid x position

    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    for r in all_records:
        init, final = r["init_tcr"], r["final_tcr"]
        for pos in range(min(len(init), len(final), max_len)):
            if init[pos] != final[pos] and final[pos] in aa_to_idx:
                mutation_matrix[aa_to_idx[final[pos]], pos] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mutation_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(20))
    ax.set_yticklabels(AMINO_ACIDS)
    ax.set_xlabel("CDR3 Position")
    ax.set_ylabel("Substituted Amino Acid")
    ax.set_title("Mutation Heatmap: Position x Amino Acid")
    plt.colorbar(im, ax=ax, label="Count")

    plt.tight_layout()
    path = os.path.join(out_dir, "mutation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_substitution_matrix(pep_results, out_dir):
    """Amino acid substitution matrix: original -> new."""
    all_records = flatten_results(pep_results)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    sub_matrix = np.zeros((20, 20))

    for r in all_records:
        init, final = r["init_tcr"], r["final_tcr"]
        for pos in range(min(len(init), len(final))):
            if init[pos] != final[pos]:
                if init[pos] in aa_to_idx and final[pos] in aa_to_idx:
                    sub_matrix[aa_to_idx[init[pos]], aa_to_idx[final[pos]]] += 1

    if sub_matrix.sum() == 0:
        return

    # Normalize rows
    row_sums = sub_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sub_matrix_norm = sub_matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SNS:
        sns.heatmap(sub_matrix_norm, xticklabels=AMINO_ACIDS, yticklabels=AMINO_ACIDS,
                    cmap="Blues", ax=ax, square=True, linewidths=0.5,
                    cbar_kws={"label": "Normalized Frequency"})
    else:
        im = ax.imshow(sub_matrix_norm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(20))
        ax.set_xticklabels(AMINO_ACIDS)
        ax.set_yticks(range(20))
        ax.set_yticklabels(AMINO_ACIDS)
        plt.colorbar(im, ax=ax, label="Normalized Frequency")

    ax.set_xlabel("New Amino Acid")
    ax.set_ylabel("Original Amino Acid")
    ax.set_title("Amino Acid Substitution Matrix")

    plt.tight_layout()
    path = os.path.join(out_dir, "substitution_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Score Correlations
# ---------------------------------------------------------------------------
def plot_score_correlations(pep_results, out_dir):
    """Scatter plots showing correlations between metrics."""
    all_records = flatten_results(pep_results)
    ergos = np.array([r["ergo_score"] for r in all_records])
    gmms = np.array([r["gmm_likelihood"] for r in all_records])
    edits = np.array([r["edit_dist"] for r in all_records])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ERGO vs GMM
    ax = axes[0]
    ax.scatter(ergos, gmms, alpha=0.1, s=5, c="steelblue")
    ax.set_xlabel("ERGO Score")
    ax.set_ylabel("GMM Likelihood")
    ax.set_title("ERGO vs GMM")

    # ERGO vs Edit Distance
    ax = axes[1]
    ax.scatter(ergos, edits, alpha=0.1, s=5, c="coral")
    ax.set_xlabel("ERGO Score")
    ax.set_ylabel("Edit Conservation")
    ax.set_title("ERGO vs Edit Distance")

    # GMM vs Edit Distance
    ax = axes[2]
    ax.scatter(gmms, edits, alpha=0.1, s=5, c="mediumpurple")
    ax.set_xlabel("GMM Likelihood")
    ax.set_ylabel("Edit Conservation")
    ax.set_title("GMM vs Edit Distance")

    plt.tight_layout()
    path = os.path.join(out_dir, "score_correlations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 6. TCR Length Distribution
# ---------------------------------------------------------------------------
def plot_tcr_length_comparison(pep_results, out_dir):
    """Compare length distribution of initial vs final TCRs."""
    all_records = flatten_results(pep_results)
    init_lens = [len(r["init_tcr"]) for r in all_records]
    final_lens = [len(r["final_tcr"]) for r in all_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(min(min(init_lens), min(final_lens)),
                 max(max(init_lens), max(final_lens)) + 2)
    ax.hist(init_lens, bins=bins, alpha=0.5, label="Initial TCR", color="steelblue", edgecolor="white")
    ax.hist(final_lens, bins=bins, alpha=0.5, label="Final TCR", color="coral", edgecolor="white")
    ax.set_xlabel("TCR CDR3 Length")
    ax.set_ylabel("Count")
    ax.set_title("TCR Length Distribution: Initial vs Final")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "tcr_length_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 7. Summary Dashboard
# ---------------------------------------------------------------------------
def plot_summary_dashboard(pep_results, out_dir):
    """Single-page summary dashboard with key metrics."""
    all_records = flatten_results(pep_results)
    ergos = [r["ergo_score"] for r in all_records]
    gmms = [r["gmm_likelihood"] for r in all_records]
    edits = [r["edit_dist"] for r in all_records]
    n = len(all_records)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. ERGO histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(ergos, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.9, color="red", linestyle="--", label="0.9 threshold")
    ax.set_title(f"ERGO Score (mean={np.mean(ergos):.3f})")
    ax.set_xlabel("Score")
    ax.legend(fontsize=8)

    # 2. GMM histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(gmms, bins=40, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.set_title(f"GMM Likelihood (mean={np.mean(gmms):.3f})")
    ax.set_xlabel("Likelihood")

    # 3. Edit dist histogram
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(edits, bins=40, color="sandybrown", edgecolor="white", alpha=0.8)
    ax.set_title(f"Edit Conservation (mean={np.mean(edits):.3f})")
    ax.set_xlabel("Conservation")

    # 4. Per-peptide avg ERGO bar
    ax = fig.add_subplot(gs[1, 0])
    peptides = sorted(pep_results.keys())
    avg_ergos = [np.mean([r["ergo_score"] for r in pep_results[p]]) for p in peptides]
    short = [p[:10] + ".." if len(p) > 12 else p for p in peptides]
    ax.barh(range(len(peptides)), avg_ergos, color="steelblue")
    ax.set_yticks(range(len(peptides)))
    ax.set_yticklabels(short, fontsize=8)
    ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Avg ERGO Score")
    ax.set_title("Per-Peptide ERGO")

    # 5. ERGO > 0.9 rate bar
    ax = fig.add_subplot(gs[1, 1])
    rates = [sum(1 for r in pep_results[p] if r["ergo_score"] >= 0.9) / len(pep_results[p]) * 100
             for p in peptides]
    ax.barh(range(len(peptides)), rates, color="coral")
    ax.set_yticks(range(len(peptides)))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("% ERGO > 0.9")
    ax.set_title("High-Affinity Rate")

    # 6. Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    high_pct = sum(1 for e in ergos if e >= 0.9) / n * 100
    unique_pct = len(set(r["final_tcr"] for r in all_records)) / n * 100
    changed_pct = sum(1 for r in all_records if r["init_tcr"] != r["final_tcr"]) / n * 100
    summary_text = (
        f"TCRPPO Evaluation Summary\n"
        f"{'='*30}\n\n"
        f"Total Samples:    {n:,}\n"
        f"Peptides:         {len(peptides)}\n\n"
        f"Avg ERGO:         {np.mean(ergos):.4f}\n"
        f"ERGO > 0.9:       {high_pct:.1f}%\n"
        f"Avg GMM:          {np.mean(gmms):.4f}\n"
        f"Avg Edit:         {np.mean(edits):.4f}\n"
        f"Unique %:         {unique_pct:.1f}%\n"
        f"Changed %:        {changed_pct:.1f}%\n"
    )
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    path = os.path.join(out_dir, "evaluation_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all_plots(result_file, out_dir):
    """Generate all visualization plots for a result file."""
    if not HAS_MPL:
        print("ERROR: matplotlib is required for visualization.")
        return

    setup_style()
    ensure_dir(out_dir)

    print(f"Parsing results from: {result_file}")
    pep_results = parse_result_file(result_file)
    if not pep_results:
        print("No results found.")
        return

    total = sum(len(v) for v in pep_results.values())
    print(f"Loaded {total} results across {len(pep_results)} peptides\n")
    print("Generating plots...")

    plot_ergo_distribution(pep_results, out_dir)
    plot_gmm_edit_distributions(pep_results, out_dir)
    plot_per_peptide_bars(pep_results, out_dir)
    plot_mutation_heatmap(pep_results, out_dir)
    plot_substitution_matrix(pep_results, out_dir)
    plot_score_correlations(pep_results, out_dir)
    plot_tcr_length_comparison(pep_results, out_dir)
    plot_summary_dashboard(pep_results, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize TCRPPO evaluation results")
    parser.add_argument("--result", type=str, required=True,
                        help="Path to result file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots (default: evaluation/figures/)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

    generate_all_plots(args.result, args.out_dir)


if __name__ == "__main__":
    main()
