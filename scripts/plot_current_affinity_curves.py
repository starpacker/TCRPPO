#!/usr/bin/env python3
"""Plot current TCRPPO v2 affinity curves from training logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACE_CONFIGS = {
    "trace53": {
        "log": "logs/trace53_terminal_trace29_reward_L2only_train.log",
        "label": "trace53 terminal L2",
        "color": "#4C78A8",
    },
    "trace61": {
        "log": "logs/trace61_dynamic_pool_train.log",
        "label": "trace61 dynamic pool baseline",
        "color": "#72B7B2",
    },
    "trace62": {
        "log": "logs/trace62_multi_gates_train.log",
        "label": "trace62 multi gates",
        "color": "#B279A2",
    },
    "trace70": {
        "log": "logs/trace70_gate_m1p5_from_trace61_train.log",
        "label": "trace70 gate -1.5",
        "color": "#54A24B",
    },
    "trace71": {
        "log": "logs/trace71_gate_m0p8_from_trace61_train.log",
        "label": "trace71 gate -0.8",
        "color": "#E45756",
    },
    "trace72": {
        "log": "logs/trace72_delta_from_trace70_train.log",
        "label": "trace72 adaptive/delta",
        "color": "#F58518",
    },
    "trace73": {
        "log": "logs/trace73_curriculum_exploration_train.log",
        "label": "trace73 curriculum exploration",
        "color": "#2F80ED",
    },
}


EPISODE_RE = re.compile(
    r"^Episode\s+(?P<episode>\d+)\s+\|\s+Step\s+(?P<step>[\d,]+)\s+\|\s+"
    r"R=(?P<R>-?\d+(?:\.\d+)?)\s+\|\s+Len=(?P<length>\d+)\s+\|\s+"
    r"A=(?P<A>-?\d+(?:\.\d+)?)\s+InitA=(?P<InitA>-?\d+(?:\.\d+)?)\s+"
    r"DeltaA=(?P<DeltaA>-?\d+(?:\.\d+)?)"
)

STEP_RE = re.compile(
    r"^Step\s+(?P<step>[\d,]+)\s+\|\s+Eps:\s+(?P<episode>\d+)\s+\|\s+"
    r"R:\s+(?P<R>-?\d+(?:\.\d+)?).*?\|\s+A:\s+(?P<A>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"InitA:\s+(?P<InitA>-?\d+(?:\.\d+)?)\s+\|\s+DeltaA:\s+(?P<DeltaA>-?\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/current_affinity_plots"),
    )
    parser.add_argument("--rolling", type=int, default=100)
    parser.add_argument("--zoom-episodes", type=int, default=2500)
    return parser.parse_args()


def parse_log(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    episode_rows: list[dict[str, float]] = []
    step_rows: list[dict[str, float]] = []

    if not path.exists():
        return {}, {}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = EPISODE_RE.match(line)
            if match:
                row = match.groupdict()
                episode_rows.append(
                    {
                        "episode": int(row["episode"]),
                        "step": int(row["step"].replace(",", "")),
                        "R": float(row["R"]),
                        "A": float(row["A"]),
                        "InitA": float(row["InitA"]),
                        "DeltaA": float(row["DeltaA"]),
                    }
                )
                continue

            match = STEP_RE.match(line)
            if match:
                row = match.groupdict()
                step_rows.append(
                    {
                        "episode": int(row["episode"]),
                        "step": int(row["step"].replace(",", "")),
                        "R": float(row["R"]),
                        "A": float(row["A"]),
                        "InitA": float(row["InitA"]),
                        "DeltaA": float(row["DeltaA"]),
                    }
                )

    return rows_to_arrays(episode_rows), rows_to_arrays(step_rows)


def rows_to_arrays(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: np.array([row[key] for row in rows]) for key in keys}


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return values, np.array([], dtype=int)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    smooth = np.convolve(values, kernel, mode="valid")
    idx = np.arange(window - 1, len(values))
    return smooth, idx


def trend_per_10k_steps(data: dict[str, np.ndarray], key: str, tail_n: int) -> float:
    if len(data.get(key, [])) < 3:
        return float("nan")
    x = data["step"][-tail_n:]
    y = data[key][-tail_n:]
    if len(np.unique(x)) < 2:
        return float("nan")
    slope = np.polyfit(x, y, deg=1)[0]
    return float(slope * 10_000)


def summarize_trace(name: str, data: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    if not data:
        return {
            "trace": name,
            "episodes": 0,
            "latest_step": "",
            "mean_A_last100": "",
            "mean_InitA_last100": "",
            "mean_DeltaA_last100": "",
            "mean_R_last100": "",
            "trend_A_per_10k_last500": "",
            "best_A": "",
            "frac_A_gt_m1_last100": "",
            "frac_A_gt_m1p5_last100": "",
        }

    n100 = min(100, len(data["A"]))
    n500 = min(500, len(data["A"]))
    tail_a = data["A"][-n100:]
    return {
        "trace": name,
        "episodes": int(len(data["A"])),
        "latest_step": int(data["step"][-1]),
        "mean_A_last100": float(np.mean(tail_a)),
        "mean_InitA_last100": float(np.mean(data["InitA"][-n100:])),
        "mean_DeltaA_last100": float(np.mean(data["DeltaA"][-n100:])),
        "mean_R_last100": float(np.mean(data["R"][-n100:])),
        "trend_A_per_10k_last500": trend_per_10k_steps(data, "A", n500),
        "best_A": float(np.max(data["A"])),
        "frac_A_gt_m1_last100": float(np.mean(tail_a > -1.0)),
        "frac_A_gt_m1p5_last100": float(np.mean(tail_a > -1.5)),
    }


def plot_panel(ax, all_data, key: str, title: str, ylabel: str, rolling: int, zoom: bool) -> None:
    for trace_name, payload in all_data.items():
        data = payload["episodes"]
        if not data:
            continue

        if zoom:
            start = max(0, len(data[key]) - payload["zoom_episodes"])
        else:
            start = 0

        x = data["step"][start:]
        y = data[key][start:]
        color = payload["config"]["color"]
        label = payload["config"]["label"]
        ax.scatter(x, y, s=5, alpha=0.12, color=color)

        smooth, idx = rolling_mean(y, rolling)
        if len(smooth):
            ax.plot(x[idx], smooth, color=color, lw=2.0, label=label)

        steps = payload["steps"]
        if steps:
            mask = steps["step"] >= x[0]
            ax.plot(
                steps["step"][mask],
                steps[key][mask],
                "o",
                color=color,
                ms=3,
                alpha=0.75,
            )

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def make_figure(all_data, out_path: Path, rolling: int, zoom: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=False)
    suffix = "latest segment" if zoom else "full logs"
    fig.suptitle(f"TCRPPO v2 affinity curves ({suffix}, rolling={rolling})", fontsize=15)

    plot_panel(axes[0, 0], all_data, "A", "Terminal affinity A", "logit", rolling, zoom)
    plot_panel(axes[0, 1], all_data, "DeltaA", "Delta affinity A - InitA", "logit", rolling, zoom)
    axes[0, 1].axhline(0, color="black", lw=1, alpha=0.4)
    plot_panel(axes[1, 0], all_data, "InitA", "Initial affinity", "logit", rolling, zoom)
    plot_panel(axes[1, 1], all_data, "R", "Episode reward", "reward", rolling, zoom)

    for ax in axes.ravel():
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, summaries: list[dict[str, float | str | int]]) -> None:
    fieldnames = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    summaries = []
    for trace_name, config in TRACE_CONFIGS.items():
        log_path = root / config["log"]
        episodes, steps = parse_log(log_path)
        all_data[trace_name] = {
            "config": config,
            "episodes": episodes,
            "steps": steps,
            "zoom_episodes": args.zoom_episodes,
        }
        summary = summarize_trace(trace_name, episodes)
        summaries.append(summary)
        print(
            f"{trace_name:7s} episodes={summary['episodes']} "
            f"latest_step={summary['latest_step']} "
            f"mean_A_last100={summary['mean_A_last100']}"
        )

    write_summary(out_dir / "current_affinity_summary.csv", summaries)
    make_figure(
        all_data,
        out_dir / "current_affinity_comparison_full.png",
        args.rolling,
        zoom=False,
    )
    make_figure(
        all_data,
        out_dir / "current_affinity_comparison_zoom.png",
        args.rolling,
        zoom=True,
    )
    print(f"Saved plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
