#!/usr/bin/env python3
"""Plot visualizations from the per-pair CSV produced by ``eval_decoy.py``.

Generates:
  1. ``boxplot_target_vs_decoy.png`` — per-target boxplot of target ERGO score
     vs the decoy ERGO score distribution.
  2. ``violin_by_hla.png``           — per-target violin plots colored by the
     decoy peptide's HLA allele.
  3. ``uncertainty_calibration.png`` — scatter of mean ERGO score vs MC dropout
     std, colored by target/decoy. Diagnoses whether the model is more
     uncertain on borderline predictions.
  4. ``top_off_target_hits.png``     — bar chart of the K worst decoy hits
     across all targets, annotated with HLA + tier.

Dependencies: matplotlib + numpy only (no seaborn).

Usage:
    python eval_decoy_visualize.py --csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv
"""
import argparse
import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NUMERIC_FIELDS = ("edit_dist", "gmm_likelihood", "ergo_mean", "ergo_std", "is_target_pep")


def load_csv(csv_path):
    rows = []
    with io.open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in NUMERIC_FIELDS:
                if k in r and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            r["is_target_pep"] = int(r.get("is_target_pep", 0))
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
def plot_boxplot_target_vs_decoy(rows, out_path):
    by_target = defaultdict(lambda: {"target": [], "decoy": []})
    for r in rows:
        bucket = "target" if r["is_target_pep"] == 1 else "decoy"
        by_target[r["target_pep"]][bucket].append(r["ergo_mean"])

    targets = sorted(by_target.keys())
    n = len(targets)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * n), 6))
    positions_t = np.arange(n) * 2.5
    positions_d = positions_t + 1.0

    target_data = [by_target[t]["target"] for t in targets]
    decoy_data = [by_target[t]["decoy"] for t in targets]

    bp1 = ax.boxplot(target_data, positions=positions_t, widths=0.8,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(decoy_data, positions=positions_d, widths=0.8,
                     patch_artist=True, showfliers=False)
    for box in bp1["boxes"]:
        box.set_facecolor("#2E86AB")
        box.set_alpha(0.85)
    for box in bp2["boxes"]:
        box.set_facecolor("#E63946")
        box.set_alpha(0.65)

    ax.set_xticks(positions_t + 0.5)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ERGO mean score (MC dropout)")
    ax.set_title("Target vs Decoy ERGO scores per target peptide")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(0.0, 0.91, "high-binding (0.9)", fontsize=8, color="gray")
    ax.set_ylim(-0.02, 1.05)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#2E86AB", alpha=0.85, label="Target peptide"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#E63946", alpha=0.65, label="Decoy pool"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print("[plot] {}".format(out_path))


# ----------------------------------------------------------------------------
def plot_violin_by_hla(rows, out_path, max_targets=12):
    by_target = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["is_target_pep"] == 1:
            continue
        by_target[r["target_pep"]][r["decoy_hla"]].append(r["ergo_mean"])

    targets = list(sorted(by_target.keys()))[:max_targets]
    if not targets:
        return

    n = len(targets)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n),
                              squeeze=False)

    for i, tgt in enumerate(targets):
        ax = axes[i // cols][i % cols]
        hla_groups = by_target[tgt]
        # Top 6 HLAs by sample count
        sorted_hlas = sorted(hla_groups.keys(),
                              key=lambda h: -len(hla_groups[h]))[:6]
        data = [hla_groups[h] for h in sorted_hlas]
        if not any(len(d) > 1 for d in data):
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(tgt, fontsize=10)
            continue
        parts = ax.violinplot(data, showmeans=True, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#9D4EDD")
            pc.set_alpha(0.6)
        ax.set_xticks(np.arange(1, len(sorted_hlas) + 1))
        ax.set_xticklabels(sorted_hlas, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("ERGO mean")
        ax.set_title(tgt, fontsize=10)
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylim(-0.02, 1.05)

    # Hide unused axes
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Decoy ERGO scores by HLA allele (per target)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print("[plot] {}".format(out_path))


# ----------------------------------------------------------------------------
def plot_uncertainty_calibration(rows, out_path):
    target_means = [r["ergo_mean"] for r in rows if r["is_target_pep"] == 1]
    target_stds = [r["ergo_std"] for r in rows if r["is_target_pep"] == 1]
    decoy_means = [r["ergo_mean"] for r in rows if r["is_target_pep"] == 0]
    decoy_stds = [r["ergo_std"] for r in rows if r["is_target_pep"] == 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    if decoy_means:
        ax.scatter(decoy_means, decoy_stds, s=4, alpha=0.25,
                   color="#E63946", label="Decoy ({})".format(len(decoy_means)))
    if target_means:
        ax.scatter(target_means, target_stds, s=14, alpha=0.6,
                   color="#2E86AB", label="Target ({})".format(len(target_means)))
    ax.set_xlabel("ERGO mean score (MC dropout)")
    ax.set_ylabel("MC dropout std (uncertainty)")
    ax.set_title("Prediction uncertainty vs mean score")
    ax.set_xlim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print("[plot] {}".format(out_path))


# ----------------------------------------------------------------------------
def plot_top_off_target_hits(rows, out_path, k=20):
    decoy_only = [r for r in rows if r["is_target_pep"] == 0]
    decoy_only.sort(key=lambda r: -r["ergo_mean"])
    top = decoy_only[:k]
    if not top:
        return

    labels = ["{}\n→{} ({})".format(r["target_pep"], r["decoy_pep"], r["decoy_source_tier"])
              for r in top]
    means = [r["ergo_mean"] for r in top]
    stds = [r["ergo_std"] for r in top]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(top)), 7))
    x = np.arange(len(top))
    bars = ax.bar(x, means, yerr=stds, capsize=3,
                   color="#E63946", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("ERGO mean (+/- MC std)")
    ax.set_title("Top {} off-target hits across all targets".format(k))
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print("[plot] {}".format(out_path))


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Visualize eval_decoy.py results")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: evaluation/figures/decoy/)")
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print("ERROR: CSV not found: {}".format(args.csv))
        sys.exit(1)

    if args.out_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out_dir = os.path.join(here, "figures", "decoy")
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    rows = load_csv(args.csv)
    print("[load] {} rows from {}".format(len(rows), args.csv))

    plot_boxplot_target_vs_decoy(rows, os.path.join(args.out_dir, "boxplot_target_vs_decoy.png"))
    plot_violin_by_hla(rows, os.path.join(args.out_dir, "violin_by_hla.png"))
    plot_uncertainty_calibration(rows, os.path.join(args.out_dir, "uncertainty_calibration.png"))
    plot_top_off_target_hits(rows, os.path.join(args.out_dir, "top_off_target_hits.png"),
                              k=args.top_k)
    print("\n[done] Figures saved to: {}".format(args.out_dir))


if __name__ == "__main__":
    main()
