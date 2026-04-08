#!/usr/bin/env python3
"""Evaluate and visualize the TCRPPO training process.

Parses training logs (stdout captured to file) and generates plots for:
  - Episode reward curves
  - ERGO binding score progression
  - PPO loss / KL / clip fraction / entropy over updates
  - Episode length distribution
  - TCR-likeness (edit distance + GMM) during training

Usage:
    python eval_training.py --log <training_log_file> [--out_dir <output_dir>]
"""
import argparse
import os
import numpy as np

from eval_utils import parse_training_log, ensure_dir

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def smooth(values, window=50):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_episode_metrics(episodes, out_dir):
    """Plot per-episode metrics from training log."""
    scores = [ep["score"] for ep in episodes if "score" in ep]
    rewards = [ep["rewards"] for ep in episodes if "rewards" in ep]
    score1 = [ep["score1"] for ep in episodes if "score1" in ep]
    score2 = [ep["score2"] for ep in episodes if "score2" in ep]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ERGO binding score
    if scores:
        ax = axes[0, 0]
        ax.plot(smooth(scores, 100), linewidth=0.8, color="tab:blue")
        ax.set_title("ERGO Binding Score (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="threshold=0.9")
        ax.legend()

    # Episode reward
    if rewards:
        ax = axes[0, 1]
        ax.plot(smooth(rewards, 100), linewidth=0.8, color="tab:green")
        ax.set_title("Episode Reward (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")

    # Edit distance (score1 = 1 - normalized_edit_dist)
    if score1:
        ax = axes[1, 0]
        ax.plot(smooth(score1, 100), linewidth=0.8, color="tab:orange")
        ax.set_title("Sequence Conservation (1-edit_dist)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Conservation")

    # GMM likelihood (score2)
    if score2:
        ax = axes[1, 1]
        ax.plot(smooth(score2, 100), linewidth=0.8, color="tab:purple")
        ax.set_title("GMM TCR-likeness")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Likelihood")

    plt.tight_layout()
    path = os.path.join(out_dir, "training_episode_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_ppo_metrics(ppo_updates, out_dir):
    """Plot PPO training metrics from SB3-style log blocks."""
    if not ppo_updates:
        print("  No PPO update metrics found in log.")
        return

    metric_keys = {
        "train_entropy_loss": ("Entropy Loss", "tab:red"),
        "train_policy_gradient_loss": ("Policy Gradient Loss", "tab:blue"),
        "train_value_loss": ("Value Loss", "tab:green"),
        "train_approx_kl": ("Approx KL Divergence", "tab:orange"),
        "train_clip_fraction": ("Clip Fraction", "tab:purple"),
        "train_explained_variance": ("Explained Variance", "tab:brown"),
        "train_loss": ("Total Loss", "tab:cyan"),
    }

    available = {}
    for key in metric_keys:
        vals = [u[key] for u in ppo_updates if key in u]
        if vals:
            available[key] = vals

    if not available:
        print("  No recognized PPO metrics in log.")
        return

    n = len(available)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (key, vals) in enumerate(available.items()):
        title, color = metric_keys[key]
        ax = axes[i]
        ax.plot(vals, linewidth=0.8, color=color)
        ax.set_title(title)
        ax.set_xlabel("Update")
        ax.set_ylabel(key.split("_")[-1])

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_ppo_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_score_distribution_over_time(episodes, out_dir, n_bins=5):
    """Show how ERGO score distribution shifts during training."""
    scores = [ep["score"] for ep in episodes if "score" in ep]
    if not scores or len(scores) < n_bins * 10:
        return

    chunk_size = len(scores) // n_bins
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n_bins):
        start = i * chunk_size
        end = start + chunk_size
        chunk = scores[start:end]
        label = f"Episodes {start}-{end}"
        ax.hist(chunk, bins=30, alpha=0.5, label=label, density=True)

    ax.set_xlabel("ERGO Binding Score")
    ax.set_ylabel("Density")
    ax.set_title("ERGO Score Distribution Shift During Training")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "training_score_distribution_shift.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_training_summary(episodes, ppo_updates):
    """Print a text summary of training statistics."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print(f"Total episodes parsed: {len(episodes)}")
    print(f"Total PPO updates parsed: {len(ppo_updates)}")

    scores = [ep["score"] for ep in episodes if "score" in ep]
    if scores:
        n = len(scores)
        q1 = n // 4
        q3 = 3 * n // 4
        print(f"\nERGO Score progression:")
        print(f"  First 25% mean:  {np.mean(scores[:q1]):.4f}")
        print(f"  Last 25% mean:   {np.mean(scores[q3:]):.4f}")
        print(f"  Overall mean:    {np.mean(scores):.4f}")
        print(f"  Max:             {max(scores):.4f}")
        print(f"  ERGO>0.9 rate (last 25%): {sum(1 for s in scores[q3:] if s >= 0.9)/len(scores[q3:])*100:.1f}%")

    rewards = [ep["rewards"] for ep in episodes if "rewards" in ep]
    if rewards:
        n = len(rewards)
        q3 = 3 * n // 4
        print(f"\nEpisode Reward:")
        print(f"  Last 25% mean: {np.mean(rewards[q3:]):.4f}")
        print(f"  Max:           {max(rewards):.4f}")

    if ppo_updates:
        kls = [u["train_approx_kl"] for u in ppo_updates if "train_approx_kl" in u]
        if kls:
            print(f"\nApprox KL (last 10 updates): {np.mean(kls[-10:]):.6f}")
        ev = [u["train_explained_variance"] for u in ppo_updates if "train_explained_variance" in u]
        if ev:
            print(f"Explained Variance (last 10): {np.mean(ev[-10:]):.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCRPPO training process")
    parser.add_argument("--log", type=str, required=True, help="Path to training log file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots (default: evaluation/figures/training)")
    args = parser.parse_args()

    if not HAS_MPL:
        print("WARNING: matplotlib not available. Only text summary will be printed.")

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "training")
    ensure_dir(args.out_dir)

    print(f"Parsing training log: {args.log}")
    data = parse_training_log(args.log)
    episodes = data["episodes"]
    ppo_updates = data["ppo_updates"]

    print_training_summary(episodes, ppo_updates)

    if HAS_MPL:
        print("\nGenerating training plots...")
        if episodes:
            plot_episode_metrics(episodes, args.out_dir)
            plot_score_distribution_over_time(episodes, args.out_dir)
        if ppo_updates:
            plot_ppo_metrics(ppo_updates, args.out_dir)
        print("Done.")


if __name__ == "__main__":
    main()
