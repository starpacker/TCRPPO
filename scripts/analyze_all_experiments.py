#!/usr/bin/env python3
"""Batch-extract experiment configs and affinity data from all traces/tests."""

import json
import os
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/share/liuyutian/tcrppo_v2")
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

###############################################################################
# 1. Collect experiment.json configs
###############################################################################

def collect_configs():
    """Read all experiment.json files and return a list of config dicts."""
    results = []
    for exp_dir in sorted(OUTPUT_DIR.iterdir()):
        exp_json = exp_dir / "experiment.json"
        if not exp_json.exists():
            continue
        try:
            with open(exp_json) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            results.append({
                "name": data.get("name", exp_dir.name),
                "launched_at": data.get("launched_at", ""),
                "status": data.get("status", ""),
                "seed": cfg.get("seed"),
                "reward_mode": cfg.get("reward_mode", ""),
                "total_timesteps": cfg.get("total_timesteps"),
                "n_envs": cfg.get("n_envs"),
                "learning_rate": cfg.get("learning_rate"),
                "hidden_dim": cfg.get("hidden_dim"),
                "max_steps": cfg.get("max_steps"),
                "affinity_scorer": cfg.get("affinity_scorer", ""),
                "w_affinity": cfg.get("weights", {}).get("affinity"),
                "w_decoy": cfg.get("weights", {}).get("decoy"),
                "w_naturalness": cfg.get("weights", {}).get("naturalness"),
                "w_diversity": cfg.get("weights", {}).get("diversity"),
                "entropy_coef": cfg.get("entropy_coef"),
                "active_clipping": cfg.get("active_clipping", False),
                "terminal_reward_only": cfg.get("terminal_reward_only"),
                "ban_stop": cfg.get("ban_stop"),
                "use_znorm": cfg.get("use_znorm", False),
                "n_contrast_decoys": cfg.get("n_contrast_decoys", 0),
                "decoy_K": cfg.get("decoy_K"),
                "train_targets": cfg.get("train_targets", ""),
                "pmhc_embedding_transform": cfg.get("pmhc_embedding_transform", "none"),
                "notes": data.get("notes", ""),
            })
        except Exception as e:
            print(f"  WARN: failed to read {exp_json}: {e}", file=sys.stderr)
    return results

###############################################################################
# 2. Extract affinity data from training logs
###############################################################################

# Patterns for extracting episode metrics from logs
# Example line: [Episode 100] step=12800 R=-2.345 A=-1.234 ...
EP_PATTERN = re.compile(
    r'\[Episode\s+(\d+)\]\s+step=([0-9,]+)\s+R=([0-9eE.+-]+)\s+A=([0-9eE.+-]+)'
)

# Also look for periodic summary lines
# Example: Step 100000 | ep=125 | R_mean=-1.23 | A_mean=-0.89 ...
SUMMARY_PATTERN = re.compile(
    r'Step\s+([0-9,]+)\s*\|\s*ep=(\d+)\s*\|.*?A_mean=([0-9eE.+-]+)'
)

# Also: [Gate Schedule] Step 644,096: gate -2.0 -> -2.0
GATE_PATTERN = re.compile(
    r'\[Gate Schedule\]\s+Step\s+([0-9,]+):\s+gate\s+([0-9eE.+-]+)\s*->\s*([0-9eE.+-]+)'
)

def parse_step(s):
    """Parse step string like '644,096' to int."""
    return int(s.replace(",", ""))

def extract_affinity_from_log(log_path):
    """Extract episode-level affinity data from a training log."""
    episodes = []
    gates = []
    resumed_at = None

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Check for resume info
                if "Resumed at step" in line:
                    m = re.search(r'Resumed at step\s+([0-9,]+)', line)
                    if m:
                        resumed_at = parse_step(m.group(1))

                # Episode metrics
                m = EP_PATTERN.search(line)
                if m:
                    episodes.append({
                        "episode": int(m.group(1)),
                        "step": parse_step(m.group(2)),
                        "reward": float(m.group(3)),
                        "affinity": float(m.group(4)),
                    })
                    continue

                # Gate schedule
                m = GATE_PATTERN.search(line)
                if m:
                    gates.append({
                        "step": parse_step(m.group(1)),
                        "gate_from": float(m.group(2)),
                        "gate_to": float(m.group(3)),
                    })
    except Exception as e:
        print(f"  WARN: failed to parse {log_path}: {e}", file=sys.stderr)

    return {
        "log_path": str(log_path),
        "resumed_at": resumed_at,
        "num_episodes": len(episodes),
        "episodes": episodes,
        "gates": gates,
    }

def compute_affinity_stats(episodes, window=500):
    """Compute affinity statistics from episode data."""
    if not episodes:
        return {}

    all_aff = [e["affinity"] for e in episodes]
    last_n = all_aff[-window:] if len(all_aff) >= window else all_aff

    last_step = episodes[-1]["step"]
    first_step = episodes[0]["step"]

    # Find best sustained affinity (rolling average over 50 episodes)
    best_rolling = float("-inf")
    rolling_window = min(50, len(all_aff))
    for i in range(len(all_aff) - rolling_window + 1):
        avg = sum(all_aff[i:i+rolling_window]) / rolling_window
        best_rolling = max(best_rolling, avg)

    return {
        "first_step": first_step,
        "last_step": last_step,
        "total_episodes": len(episodes),
        "mean_affinity_all": sum(all_aff) / len(all_aff),
        "max_affinity_all": max(all_aff),
        "min_affinity_all": min(all_aff),
        "mean_affinity_last500": sum(last_n) / len(last_n),
        "max_affinity_last500": max(last_n),
        "min_affinity_last500": min(last_n),
        "best_rolling_50": best_rolling,
    }

###############################################################################
# 3. Match log files to experiment names
###############################################################################

def find_log_for_experiment(name):
    """Find the training log file for a given experiment name."""
    # Try direct match first
    candidates = [
        LOGS_DIR / f"{name}_train.log",
    ]

    # Also try with test prefix variations
    for f in LOGS_DIR.glob("*train.log"):
        if name in f.name:
            candidates.append(f)

    # Check archive too
    for f in (LOGS_DIR / "archive" / "early_small_tests").glob("*train.log"):
        if name in f.name:
            candidates.append(f)

    for c in candidates:
        if c.exists():
            return c
    return None

###############################################################################
# 4. Main
###############################################################################

def main():
    print("=" * 80)
    print("TCRPPO v2 Experiment Analysis")
    print("=" * 80)

    # Collect configs
    print("\n[1/3] Collecting experiment configs...")
    configs = collect_configs()
    print(f"  Found {len(configs)} experiments with experiment.json")

    # Extract affinity data
    print("\n[2/3] Extracting affinity data from training logs...")
    all_data = []

    for cfg in configs:
        name = cfg["name"]
        log_path = find_log_for_experiment(name)

        if log_path:
            log_data = extract_affinity_from_log(log_path)
            stats = compute_affinity_stats(log_data["episodes"])
            cfg.update(stats)
            cfg["has_log"] = True
            cfg["log_file"] = log_path.name
            cfg["num_episodes_parsed"] = log_data["num_episodes"]
            cfg["resumed_from_step"] = log_data["resumed_at"]
        else:
            cfg["has_log"] = False
            cfg["log_file"] = ""

        all_data.append(cfg)

    # Sort by mean_affinity_last500 (best first)
    all_data.sort(key=lambda x: x.get("mean_affinity_last500", float("-inf")), reverse=True)

    # Save full data as JSON
    output_json = LOGS_DIR / "experiment_analysis_full.json"
    with open(output_json, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  Saved full analysis to {output_json}")

    # Save summary CSV
    output_csv = LOGS_DIR / "experiment_analysis_summary.csv"
    fields = [
        "name", "affinity_scorer", "reward_mode", "learning_rate", "entropy_coef",
        "active_clipping", "terminal_reward_only", "max_steps", "n_envs",
        "w_affinity", "w_decoy", "w_naturalness", "w_diversity",
        "use_znorm", "n_contrast_decoys", "decoy_K", "train_targets",
        "total_episodes", "first_step", "last_step",
        "mean_affinity_last500", "max_affinity_last500", "best_rolling_50",
        "has_log", "log_file", "resumed_from_step",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for d in all_data:
            writer.writerow(d)
    print(f"  Saved summary CSV to {output_csv}")

    # Print top 20 by affinity
    print("\n[3/3] Top experiments by mean_affinity_last500:")
    print(f"{'Rank':>4} {'Name':<50} {'Scorer':<8} {'MeanAff':<10} {'MaxAff':<10} {'BestRoll50':<10} {'Steps':<10}")
    print("-" * 112)
    for i, d in enumerate(all_data[:30]):
        mean_aff = d.get("mean_affinity_last500")
        max_aff = d.get("max_affinity_last500")
        best_roll = d.get("best_rolling_50")
        last_step = d.get("last_step")
        scorer = d.get("affinity_scorer", "?")

        mean_s = f"{mean_aff:.4f}" if mean_aff is not None else "N/A"
        max_s = f"{max_aff:.4f}" if max_aff is not None else "N/A"
        roll_s = f"{best_roll:.4f}" if best_roll is not None else "N/A"
        step_s = f"{last_step:,}" if last_step is not None else "N/A"

        print(f"{i+1:>4} {d['name']:<50} {scorer:<8} {mean_s:<10} {max_s:<10} {roll_s:<10} {step_s:<10}")

if __name__ == "__main__":
    main()
