#!/usr/bin/env python3
"""Archive all completed TCRPPO v2 experiments with configs and results.

Creates output/<name>/experiment.json for each experiment with:
- Reconstructed configuration (from logs + known launch parameters)
- Evaluation results (from results/<name>/eval_results.json or eval logs)
- Training metadata (steps, speed, git commit)

Run once: python scripts/archive_experiments.py
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"


# ─── Experiment definitions (reconstructed from logs, tracker, launch docs) ───

EXPERIMENTS = [
    {
        "name": "v1_ergo_only_ablation",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v1_ergo_only",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.0,
        "w_naturalness": 0.0,
        "w_diversity": 0.0,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "git_commit": "phase5",
        "gpu": None,
        "notes": "v1 baseline reproduction with v2 architecture. BEST result but seed-dependent (seed=42 outlier).",
        "train_log": "v1_ergo_only_2M_train.log",
        "eval_log": "v1_ergo_only_eval.log",
        "results_dir": "v1_ergo_only_ablation",
    },
    {
        "name": "v2_full_run1",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v2_full",
        "total_timesteps": 2_000_000,
        "n_envs": 20,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.8,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": True,
        "use_znorm": True,
        "min_steps": 0,
        "git_commit": "phase5",
        "gpu": None,
        "notes": "Full v2 pipeline with all 4 reward components. Early stopping (avg 2 steps) due to penalty weights too strong.",
        "train_log": "v2_full_2M_train.log",
        "eval_log": "v2_full_run1_eval.log",
        "results_dir": "v2_full_run1",
    },
    {
        "name": "v2_no_decoy_ablation",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v2_no_decoy",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.0,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": True,
        "use_znorm": True,
        "min_steps": 0,
        "git_commit": "phase7",
        "gpu": 7,
        "notes": "v2 without decoy penalty (ablation). Stopped at ~1.74M/2M steps. Z-norm still hurts.",
        "train_log": "v2_no_decoy_2M_train.log",
        "eval_log": "v2_no_decoy_eval.log",
        "results_dir": None,  # no structured results dir
    },
    {
        "name": "exp1_decoy_only",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v2_decoy_only",
        "total_timesteps": 500_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.3,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": True,
        "use_znorm": True,
        "min_steps": 0,
        "git_commit": "ec1be82",
        "gpu": 0,
        "notes": "Decoy penalty only (d=0.3) with z-norm. Even light decoy + z-norm compresses affinity signal.",
        "train_log": "exp1_decoy_only_train.log",
        "eval_log": "exp1_decoy_only_eval.log",
        "results_dir": None,
    },
    {
        "name": "exp2_light",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v2_full",
        "total_timesteps": 500_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.2,
        "w_naturalness": 0.1,
        "w_diversity": 0.05,
        "use_delta_reward": True,
        "use_znorm": True,
        "min_steps": 0,
        "git_commit": "ec1be82",
        "gpu": 2,
        "notes": "Dramatically reduced penalty weights (10x from v2_full). Even 10x lighter still fails with z-norm.",
        "train_log": "exp2_light_train.log",
        "eval_log": "exp2_light_eval.log",
        "results_dir": None,
    },
    {
        "name": "exp3_ergo_delta",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v1_ergo_delta",
        "total_timesteps": 500_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.8,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": True,
        "use_znorm": False,
        "min_steps": 0,
        "git_commit": "ec1be82",
        "gpu": 1,
        "notes": "Raw delta per-step reward (no z-norm, no penalties). Long edits (7.9 steps) but poor binding.",
        "train_log": "exp3_ergo_delta_train.log",
        "eval_log": "exp3_ergo_delta_eval.log",
        "results_dir": None,
    },
    {
        "name": "exp4_min_steps",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v2_full",
        "total_timesteps": 500_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.4,
        "w_naturalness": 0.2,
        "w_diversity": 0.1,
        "use_delta_reward": True,
        "use_znorm": True,
        "min_steps": 3,
        "min_steps_penalty": -2.0,
        "git_commit": "ec1be82",
        "gpu": 3,
        "notes": "Force min 3 steps + z-norm. Min-steps doesn't fix z-norm compression.",
        "train_log": "exp4_min_steps_train.log",
        "eval_log": "exp4_min_steps_eval.log",
        "results_dir": None,
    },
    {
        "name": "test1_two_phase_p2",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v1_ergo_only -> raw_decoy",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.05,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "two_phase": True,
        "phase1_steps": 1_000_000,
        "phase1_reward_mode": "v1_ergo_only",
        "phase2_steps": 1_000_000,
        "phase2_reward_mode": "raw_decoy",
        "resume_from": "output/test1_two_phase_p1/checkpoints/milestone_1000000.pt",
        "git_commit": "68a6f75",
        "gpu": 0,
        "notes": "Two-phase: 1M pure ERGO, then 1M raw_decoy (d=0.05). Decoy penalty in P2 degraded binding.",
        "train_log": "test1_two_phase_p2_train.log",
        "eval_log": "test1_p2_eval.log",
        "results_dir": "test1_two_phase_p2",
    },
    {
        "name": "test2_min6_raw",
        "status": "completed",
        "seed": 42,
        "reward_mode": "raw_decoy",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.05,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 6,
        "min_steps_penalty": -3.0,
        "git_commit": "68a6f75",
        "gpu": 1,
        "notes": "Min 6 steps + raw decoy penalty (d=0.05, no z-norm). Raw decoy penalty still hurts.",
        "train_log": "test2_min6_raw_train.log",
        "eval_log": None,
        "results_dir": "test2_min6_raw",
    },
    {
        "name": "test3_stepwise",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v1_ergo_stepwise",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.8,
        "w_naturalness": 0.5,
        "w_diversity": 0.2,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "git_commit": "68a6f75",
        "gpu": 2,
        "notes": "Per-step absolute ERGO score (not delta, not terminal). Underperforms terminal reward.",
        "train_log": "test3_stepwise_train.log",
        "eval_log": "test3_stepwise_eval.log",
        "results_dir": "test3_stepwise",
    },
    {
        "name": "test4_raw_multi",
        "status": "completed",
        "seed": 42,
        "reward_mode": "raw_multi_penalty",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.05,
        "w_naturalness": 0.02,
        "w_diversity": 0.01,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "git_commit": "68a6f75",
        "gpu": 3,
        "notes": "Raw multi-penalty (d=0.05, n=0.02, v=0.01, no z-norm). Multi-penalty still hurts vs pure ERGO.",
        "train_log": "test4_raw_multi_train.log",
        "eval_log": "test4_raw_multi_eval.log",
        "results_dir": "test4_raw_multi",
    },
    {
        "name": "test5_threshold",
        "status": "completed",
        "seed": 42,
        "reward_mode": "threshold_penalty",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.05,
        "w_naturalness": 0.02,
        "w_diversity": 0.01,
        "affinity_threshold": 0.5,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "git_commit": "68a6f75",
        "gpu": 4,
        "notes": "Conditional penalties at affinity>0.5 threshold. Threshold gating doesn't help.",
        "train_log": "test5_threshold_train.log",
        "eval_log": "test5_threshold_eval.log",
        "results_dir": "test5_threshold",
    },
    {
        "name": "test6_pure_v2_arch",
        "status": "completed",
        "seed": 42,
        "reward_mode": "v1_ergo_only",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.0,
        "w_naturalness": 0.0,
        "w_diversity": 0.0,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "architecture_changes": "A1+A2+A10 only, NO L0 curriculum",
        "git_commit": "a3039b5",
        "gpu": 5,
        "notes": "Pure v2 architecture (A1+A2+A10) without L0 curriculum. Simpler arch performs same or better.",
        "train_log": "test6_pure_v2_arch_train.log",
        "eval_log": "test6_pure_v2_arch_eval.log",
        "results_dir": "test6_pure_v2_arch",
    },
    {
        "name": "test7_v1ergo_repro",
        "status": "completed",
        "seed": 123,
        "reward_mode": "v1_ergo_only",
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hidden_dim": 512,
        "max_steps": 8,
        "affinity_scorer": "ergo",
        "w_affinity": 1.0,
        "w_decoy": 0.0,
        "w_naturalness": 0.0,
        "w_diversity": 0.0,
        "use_delta_reward": False,
        "use_znorm": False,
        "min_steps": 0,
        "git_commit": "a3039b5",
        "gpu": 2,
        "notes": "Reproduction of v1_ergo_only with seed=123. AUROC dropped to 0.5462 (from 0.8075 with seed=42). Confirms seed dependence.",
        "train_log": "test7_v1ergo_repro_train.log",
        "eval_log": "test7_v1ergo_repro_eval.log",
        "results_dir": "test7_v1ergo_repro",
    },
]


def parse_eval_results_json(results_dir: str) -> dict:
    """Load structured eval_results.json from results/<name>/."""
    path = RESULTS_DIR / results_dir / "eval_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    per_target = {}
    total_auroc = 0
    total_target = 0
    total_decoy = 0
    total_steps = 0
    count = 0
    for target, info in data.items():
        if target.startswith("_") or "specificity" not in info:
            continue  # skip metadata keys like _summary
        spec = info.get("specificity", {})
        auroc = spec.get("auroc", 0)
        per_target[target] = {
            "auroc": round(auroc, 4),
            "mean_target_score": round(spec.get("mean_target_score", 0), 4),
            "mean_decoy_score": round(spec.get("mean_decoy_score", 0), 4),
            "n_unique": info.get("n_unique", 0),
            "mean_steps": info.get("mean_steps", 0),
        }
        total_auroc += auroc
        total_target += spec.get("mean_target_score", 0)
        total_decoy += spec.get("mean_decoy_score", 0)
        total_steps += info.get("mean_steps", 0)
        count += 1

    if count == 0:
        return {}

    return {
        "mean_auroc": round(total_auroc / count, 4),
        "mean_target_score": round(total_target / count, 4),
        "mean_decoy_score": round(total_decoy / count, 4),
        "mean_steps": round(total_steps / count, 1),
        "n_targets": count,
        "per_target": per_target,
    }


def parse_eval_log(eval_log: str) -> dict:
    """Parse AUROC from eval log file (for experiments without structured results)."""
    path = OUTPUT_DIR / eval_log
    if not path.exists():
        return {}

    text = path.read_text()

    # Extract per-target AUROC from log
    per_target = {}
    # Pattern: target line followed by AUROC line
    target_pattern = re.compile(r"Target: (\w+)")
    auroc_pattern = re.compile(r"AUROC:\s+([\d.]+)")

    lines = text.split("\n")
    current_target = None
    for line in lines:
        tm = target_pattern.search(line)
        if tm:
            current_target = tm.group(1)
        am = auroc_pattern.search(line)
        if am and current_target:
            per_target[current_target] = {"auroc": round(float(am.group(1)), 4)}
            current_target = None

    # Extract mean AUROC
    mean_match = re.search(r"Mean AUROC:\s+([\d.]+)", text)
    mean_auroc = float(mean_match.group(1)) if mean_match else 0

    return {
        "mean_auroc": round(mean_auroc, 4),
        "n_targets": len(per_target),
        "per_target": per_target,
    }


def get_output_dir_name(exp_name: str) -> str:
    """Map experiment name to its output directory."""
    # Some experiments have different output dir names
    mapping = {
        "test1_two_phase_p2": "test1_two_phase_p2",
    }
    return mapping.get(exp_name, exp_name)


def archive_experiment(exp: dict) -> dict:
    """Create experiment.json for a single experiment."""
    name = exp["name"]
    output_name = get_output_dir_name(name)
    output_path = OUTPUT_DIR / output_name

    # Build config section
    config = {
        "seed": exp["seed"],
        "reward_mode": exp["reward_mode"],
        "total_timesteps": exp["total_timesteps"],
        "n_envs": exp["n_envs"],
        "learning_rate": exp["learning_rate"],
        "hidden_dim": exp["hidden_dim"],
        "max_steps": exp["max_steps"],
        "affinity_scorer": exp["affinity_scorer"],
        "weights": {
            "affinity": exp["w_affinity"],
            "decoy": exp["w_decoy"],
            "naturalness": exp["w_naturalness"],
            "diversity": exp["w_diversity"],
        },
        "use_delta_reward": exp["use_delta_reward"],
        "use_znorm": exp["use_znorm"],
    }

    if exp.get("min_steps", 0) > 0:
        config["min_steps"] = exp["min_steps"]
        config["min_steps_penalty"] = exp.get("min_steps_penalty", -2.0)
    if exp.get("affinity_threshold"):
        config["affinity_threshold"] = exp["affinity_threshold"]
    if exp.get("two_phase"):
        config["two_phase"] = True
        config["phase1_steps"] = exp["phase1_steps"]
        config["phase1_reward_mode"] = exp["phase1_reward_mode"]
        config["phase2_steps"] = exp["phase2_steps"]
        config["phase2_reward_mode"] = exp["phase2_reward_mode"]
        config["resume_from"] = exp.get("resume_from", "")
    if exp.get("architecture_changes"):
        config["architecture_changes"] = exp["architecture_changes"]

    # Get results
    results = {}
    if exp.get("results_dir"):
        results = parse_eval_results_json(exp["results_dir"])
    if not results and exp.get("eval_log"):
        results = parse_eval_log(exp["eval_log"])

    # Build archive
    archive = {
        "name": name,
        "status": exp["status"],
        "archived_at": datetime.now().isoformat(),
        "git_commit": exp.get("git_commit", "unknown"),
        "gpu": exp.get("gpu"),
        "config": config,
        "results": results,
        "notes": exp.get("notes", ""),
        "logs": {
            "train": exp.get("train_log", ""),
            "eval": exp.get("eval_log", ""),
        },
    }

    return archive


def main():
    print(f"Archiving {len(EXPERIMENTS)} experiments...")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    archived = 0
    skipped = 0

    for exp in EXPERIMENTS:
        name = exp["name"]
        output_name = get_output_dir_name(name)
        output_path = OUTPUT_DIR / output_name

        if not output_path.exists():
            print(f"  SKIP {name}: output dir not found at {output_path}")
            skipped += 1
            continue

        archive = archive_experiment(exp)
        out_file = output_path / "experiment.json"

        with open(out_file, "w") as f:
            json.dump(archive, f, indent=2)

        auroc = archive["results"].get("mean_auroc", "N/A")
        print(f"  OK   {name}: AUROC={auroc}, saved to {out_file}")
        archived += 1

    print()
    print(f"Done: {archived} archived, {skipped} skipped")

    # Summary table
    print()
    print("=" * 80)
    print(f"{'Experiment':<30} {'AUROC':>8} {'Reward Mode':<20} {'Seed':>5}")
    print("-" * 80)
    for exp in EXPERIMENTS:
        name = exp["name"]
        output_name = get_output_dir_name(name)
        out_file = OUTPUT_DIR / output_name / "experiment.json"
        if out_file.exists():
            with open(out_file) as f:
                data = json.load(f)
            auroc = data["results"].get("mean_auroc", "N/A")
            print(f"  {name:<28} {auroc:>8} {exp['reward_mode']:<20} {exp['seed']:>5}")
    print("=" * 80)


if __name__ == "__main__":
    main()
