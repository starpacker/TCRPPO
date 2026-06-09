#!/usr/bin/env python3
"""Plot reward curves for the currently live experiments."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "current_live_reward_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    (
        "trace29 baseline",
        ROOT / "logs/test62_simple_target_gated_decoy_trace29_simple_target_gated_decoy_train.log",
        "keep",
    ),
    (
        "trace48 cross-attn raw",
        ROOT / "logs/trace48_cross_attn_train.log",
        "cross-attn",
    ),
    (
        "trace51 multi-gate phase1",
        ROOT / "logs/curriculum_climbing_phase1_trace51_target0p6_phase1_train.log",
        "multi-gate",
    ),
    (
        "trace50 curriculum v1",
        ROOT / "logs/curriculum_climbing_v1_trace50_train.log",
        "multi-gate",
    ),
    (
        "trace23 delta stop-min2",
        ROOT / "logs/test55_delta_ablation_trace23_delta_stop_min2_train.log",
        "stop candidate",
    ),
    (
        "trace26 active clip",
        ROOT / "logs/test57_active_clip_trace26_active_clip_train.log",
        "stop candidate",
    ),
    (
        "trace30 maxstep ms7",
        ROOT / "logs/test56_maxstep_curriculum_trace30_maxstep_curriculum_1to8_adaptive_ms7_train.log",
        "stop candidate",
    ),
    (
        "SAC h8 livebest",
        ROOT / "logs_sac/sac_tfold_amp_h8_edit_multipep_livebest.log",
        "confirm owner",
    ),
]

EPISODE_RE = re.compile(
    r"Episode\s+(?P<episode>[0-9,]+)\s+\|\s+Step\s+(?P<step>[0-9,]+)\s+\|\s+R=(?P<reward>[-+0-9.eE]+)"
)
A_RE = re.compile(r"\bA=(?P<affinity>[-+0-9.eE]+)")


def parse_episode_rewards(path: Path) -> list[tuple[int, int, float, float]]:
    rows: list[tuple[int, int, float, float]] = []
    if not path.exists():
        return rows
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            match = EPISODE_RE.search(line)
            if not match:
                continue
            episode = int(match.group("episode").replace(",", ""))
            step = int(match.group("step").replace(",", ""))
            reward = float(match.group("reward"))
            a_match = A_RE.search(line)
            affinity = float(a_match.group("affinity")) if a_match else math.nan
            if math.isfinite(reward):
                rows.append((episode, step, reward, affinity))
    return rows


def rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    total = 0.0
    q: list[float] = []
    for value in values:
        total += value
        q.append(value)
        if len(q) > window:
            total -= q.pop(0)
        out.append(total / len(q))
    return out


def thin(xs: list[float], ys: list[float], max_points: int = 2500) -> tuple[list[float], list[float]]:
    if len(xs) <= max_points:
        return xs, ys
    stride = math.ceil(len(xs) / max_points)
    return xs[::stride], ys[::stride]


def plot_overlay(all_data: dict[str, dict[str, object]]) -> Path:
    fig, (ax_full, ax_recent) = plt.subplots(2, 1, figsize=(15, 10), sharey=False)
    fig.suptitle("Current Live Experiments: Episode Reward Curves", fontsize=16, fontweight="bold")

    for name, data in all_data.items():
        steps = data["steps"]
        rewards_roll = data["reward_roll"]
        assert isinstance(steps, list)
        assert isinstance(rewards_roll, list)
        xs, ys = thin(steps, rewards_roll)
        ax_full.plot(xs, ys, linewidth=1.8, alpha=0.85, label=name)

        recent_steps = steps[-1500:]
        recent_rewards = rewards_roll[-1500:]
        if recent_steps:
            rel_steps = [step - recent_steps[0] for step in recent_steps]
            rx, ry = thin(rel_steps, recent_rewards)
            ax_recent.plot(rx, ry, linewidth=1.8, alpha=0.85, label=name)

    ax_full.set_title("Full History, Rolling Mean Reward")
    ax_full.set_xlabel("Training step")
    ax_full.set_ylabel("Reward, rolling mean")
    ax_full.grid(alpha=0.25)
    ax_full.legend(fontsize=8, ncol=2)

    ax_recent.set_title("Recent Window, Last 1500 Episodes Per Run")
    ax_recent.set_xlabel("Relative step within recent window")
    ax_recent.set_ylabel("Reward, rolling mean")
    ax_recent.grid(alpha=0.25)
    ax_recent.legend(fontsize=8, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "current_live_reward_overlay.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_small_multiples(all_data: dict[str, dict[str, object]]) -> Path:
    n = len(all_data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.4 * rows), squeeze=False)
    fig.suptitle("Current Live Experiments: Individual Reward Curves", fontsize=16, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    for ax, (name, data) in zip(axes.flat, all_data.items()):
        ax.axis("on")
        episodes = data["episodes"]
        rewards = data["rewards"]
        rewards_roll = data["reward_roll"]
        status = data["status"]
        assert isinstance(episodes, list)
        assert isinstance(rewards, list)
        assert isinstance(rewards_roll, list)

        xs_raw, ys_raw = thin(episodes, rewards, max_points=1800)
        xs_roll, ys_roll = thin(episodes, rewards_roll, max_points=1800)
        ax.plot(xs_raw, ys_raw, color="#9aa0a6", linewidth=0.6, alpha=0.25, label="episode R")
        ax.plot(xs_roll, ys_roll, linewidth=1.8, color="#1f77b4", label="rolling mean")

        final_reward = rewards[-1]
        final_roll = rewards_roll[-1]
        best_roll = max(rewards_roll)
        last_step = data["last_step"]
        ax.scatter([episodes[-1]], [final_roll], color="#d62728", s=18, zorder=3)
        ax.set_title(f"{name} ({status})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.22)
        ax.text(
            0.02,
            0.98,
            f"step={last_step:,}\nlast R={final_reward:.3f}\nroll={final_roll:.3f}\nbest roll={best_roll:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    out = OUT_DIR / "current_live_reward_individual.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_affinity(all_data: dict[str, dict[str, object]]) -> Path:
    n = len(all_data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.4 * rows), squeeze=False)
    fig.suptitle("Current Live Experiments: Target Affinity A Curves", fontsize=16, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    for ax, (name, data) in zip(axes.flat, all_data.items()):
        affinities = data["affinities"]
        episodes = data["episodes"]
        status = data["status"]
        assert isinstance(affinities, list)
        assert isinstance(episodes, list)
        valid = [(ep, a) for ep, a in zip(episodes, affinities) if math.isfinite(a)]
        if not valid:
            continue
        valid_episodes = [row[0] for row in valid]
        valid_affinities = [row[1] for row in valid]
        window = min(100, max(10, len(valid_affinities) // 20))
        affinity_roll = rolling_mean(valid_affinities, window)

        ax.axis("on")
        xs_raw, ys_raw = thin(valid_episodes, valid_affinities, max_points=1800)
        xs_roll, ys_roll = thin(valid_episodes, affinity_roll, max_points=1800)
        ax.plot(xs_raw, ys_raw, color="#9aa0a6", linewidth=0.6, alpha=0.2, label="episode A")
        ax.plot(xs_roll, ys_roll, color="#2ca02c", linewidth=1.8, label="rolling mean")
        ax.axhline(0.6, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7, label="target 0.6")
        ax.set_title(f"{name} ({status})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Target affinity A")
        ax.grid(alpha=0.22)
        ax.text(
            0.02,
            0.98,
            f"last A={valid_affinities[-1]:.3f}\nroll{window}={affinity_roll[-1]:.3f}\nbest roll={max(affinity_roll):.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    out = OUT_DIR / "current_live_affinity_individual.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_summary(all_data: dict[str, dict[str, object]]) -> Path:
    out = OUT_DIR / "current_live_reward_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run",
                "status",
                "points",
                "first_step",
                "last_step",
                "last_episode",
                "last_reward",
                "rolling100_last",
                "rolling100_best",
                "rolling100_delta_last500",
                "affinity_last",
                "affinity_roll_last",
                "affinity_roll_best",
                "log_path",
            ]
        )
        for name, data in all_data.items():
            rewards_roll = data["reward_roll"]
            steps = data["steps"]
            episodes = data["episodes"]
            rewards = data["rewards"]
            affinities = data["affinities"]
            assert isinstance(rewards_roll, list)
            assert isinstance(steps, list)
            assert isinstance(episodes, list)
            assert isinstance(rewards, list)
            assert isinstance(affinities, list)
            if len(rewards_roll) > 500:
                delta_last500 = rewards_roll[-1] - rewards_roll[-501]
            else:
                delta_last500 = rewards_roll[-1] - rewards_roll[0]
            valid_affinities = [a for a in affinities if math.isfinite(a)]
            if valid_affinities:
                affinity_roll = rolling_mean(valid_affinities, min(100, max(10, len(valid_affinities) // 20)))
                affinity_last = f"{valid_affinities[-1]:.6f}"
                affinity_roll_last = f"{affinity_roll[-1]:.6f}"
                affinity_roll_best = f"{max(affinity_roll):.6f}"
            else:
                affinity_last = ""
                affinity_roll_last = ""
                affinity_roll_best = ""
            writer.writerow(
                [
                    name,
                    data["status"],
                    len(rewards),
                    steps[0],
                    steps[-1],
                    episodes[-1],
                    f"{rewards[-1]:.6f}",
                    f"{rewards_roll[-1]:.6f}",
                    f"{max(rewards_roll):.6f}",
                    f"{delta_last500:.6f}",
                    affinity_last,
                    affinity_roll_last,
                    affinity_roll_best,
                    data["path"],
                ]
            )
    return out


def main() -> None:
    all_data: dict[str, dict[str, object]] = {}
    for name, path, status in RUNS:
        rows = parse_episode_rewards(path)
        if not rows:
            print(f"skip: {name}: no episode rewards in {path}")
            continue
        episodes = [row[0] for row in rows]
        steps = [row[1] for row in rows]
        rewards = [row[2] for row in rows]
        affinities = [row[3] for row in rows]
        window = min(100, max(10, len(rewards) // 20))
        reward_roll = rolling_mean(rewards, window)
        all_data[name] = {
            "status": status,
            "path": str(path.relative_to(ROOT)),
            "episodes": episodes,
            "steps": steps,
            "rewards": rewards,
            "affinities": affinities,
            "reward_roll": reward_roll,
            "last_step": steps[-1],
        }
        print(
            f"{name}: n={len(rewards):,} step={steps[-1]:,} "
            f"last_R={rewards[-1]:.3f} roll{window}={reward_roll[-1]:.3f}"
        )

    if not all_data:
        raise SystemExit("No live reward data found.")

    overlay = plot_overlay(all_data)
    individual = plot_small_multiples(all_data)
    affinity = plot_affinity(all_data)
    summary = write_summary(all_data)
    print(f"saved: {overlay}")
    print(f"saved: {individual}")
    print(f"saved: {affinity}")
    print(f"saved: {summary}")


if __name__ == "__main__":
    main()
