"""Generate comparison plots and tables for evaluation results."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# V1 baseline AUROCs
V1_AUROCS = {
    "GILGFVFTL": 0.3200,
    "NLVPMVATV": 0.4022,
    "GLCTLVAML": 0.6778,
    "LLWNGPMAV": 0.3472,
    "YLQPRTFLL": 0.3028,
    "FLYALALLL": 0.4133,
    "SLYNTVATL": 0.8776,
    "KLGGALQAK": 0.5200,
    "AVFDRKSDAK": 0.4561,
    "IVTDFSVIK": 0.3022,
    "SPRWYFYYL": 0.6056,
    "RLRAEAQVK": 0.2311,
}


def plot_auroc_comparison(results_files: dict, output_dir: str):
    """Plot per-target AUROC comparison bar chart.

    Args:
        results_files: Dict mapping run_name -> eval_results.json path.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    targets = list(V1_AUROCS.keys())
    x = np.arange(len(targets))
    width = 0.8 / (len(results_files) + 1)  # +1 for v1 baseline

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # V1 baseline
    v1_aurocs = [V1_AUROCS[t] for t in targets]
    ax.bar(x - width * len(results_files) / 2, v1_aurocs, width, label="v1 baseline", color="gray", alpha=0.7)

    # Other runs
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (name, path) in enumerate(results_files.items()):
        with open(path) as f:
            data = json.load(f)
        aurocs = []
        for t in targets:
            if t in data:
                aurocs.append(data[t]["specificity"]["auroc"])
            else:
                aurocs.append(0.0)
        offset = -width * len(results_files) / 2 + width * (i + 1)
        ax.bar(x + offset, aurocs, width, label=name, color=colors[i % len(colors)], alpha=0.8)

    ax.set_ylabel("AUROC")
    ax.set_title("Decoy Specificity AUROC by Target")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="random")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auroc_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/auroc_comparison.png")


def plot_score_distribution(results_file: str, output_dir: str, run_name: str):
    """Plot target vs decoy score distributions per target."""
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file) as f:
        data = json.load(f)

    targets = [t for t in data if not t.startswith("_")]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, target in enumerate(targets):
        if i >= 12:
            break
        ax = axes[i]
        s = data[target]["specificity"]
        auroc = s["auroc"]
        target_score = s["mean_target_score"]
        decoy_score = s["mean_decoy_score"]

        ax.bar(["Target", "Decoy"], [target_score, decoy_score],
               color=["#2196F3", "#F44336"], alpha=0.7)
        ax.set_title(f"{target}\nAUROC={auroc:.3f}", fontsize=9)
        ax.set_ylim(0, 1)

    plt.suptitle(f"Target vs Decoy Scores — {run_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"score_dist_{run_name}.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/score_dist_{run_name}.png")


def print_comparison_table(results_files: dict):
    """Print markdown comparison table."""
    targets = list(V1_AUROCS.keys())

    # Header
    header = "| Target |"
    header += " v1 Baseline |"
    for name in results_files:
        header += f" {name} |"
    print(header)

    sep = "|--------|"
    sep += "-------------|"
    for _ in results_files:
        sep += "-------------|"
    print(sep)

    all_data = {}
    for name, path in results_files.items():
        with open(path) as f:
            all_data[name] = json.load(f)

    # Per-target rows
    v1_aurocs = []
    run_aurocs = {name: [] for name in results_files}
    for target in targets:
        row = f"| {target} |"
        v1_auc = V1_AUROCS[target]
        v1_aurocs.append(v1_auc)
        row += f" {v1_auc:.4f} |"
        for name in results_files:
            if target in all_data[name]:
                auc = all_data[name][target]["specificity"]["auroc"]
            else:
                auc = 0.0
            run_aurocs[name].append(auc)
            row += f" {auc:.4f} |"
        print(row)

    # Mean row
    row = "| **Mean** |"
    row += f" **{np.mean(v1_aurocs):.4f}** |"
    for name in results_files:
        mean = np.mean(run_aurocs[name])
        delta = mean - np.mean(v1_aurocs)
        row += f" **{mean:.4f}** ({delta:+.4f}) |"
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--results", nargs="+", required=True,
                        help="result_name:path pairs, e.g., 'v2_full:results/v2_full/eval_results.json'")
    parser.add_argument("--output_dir", default="figures", help="Output directory for plots")
    args = parser.parse_args()

    results_files = {}
    for r in args.results:
        name, path = r.split(":", 1)
        results_files[name] = path

    print("\n=== AUROC Comparison Table ===\n")
    print_comparison_table(results_files)

    print("\n=== Generating Plots ===\n")
    plot_auroc_comparison(results_files, args.output_dir)

    for name, path in results_files.items():
        plot_score_distribution(path, args.output_dir, name)


if __name__ == "__main__":
    main()
