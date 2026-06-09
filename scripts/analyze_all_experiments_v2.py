#!/usr/bin/env python3
"""Comprehensive experiment analysis for TCRPPO v2.
Reads all experiment.json configs + parses training logs to extract affinity.
Outputs a full analysis JSON for the markdown generation step.
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/share/liuyutian/tcrppo_v2")
OUTPUT_DIR = PROJECT / "output"
LOGS_DIR = PROJECT / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive" / "early_small_tests"

# ─── Log parsing patterns ───────────────────────────────────────────────
# New format: Episode N | Step XXXXX | R=... | Len=... | A=... InitA=... DeltaA=... ...
EP_NEW = re.compile(
    r'Episode\s+(\d+)\s*\|\s*Step\s+([0-9,]+)\s*\|\s*R=([0-9eE.+-]+)\s*\|\s*Len=(\d+)\s*\|\s*A=([0-9eE.+-]+)'
)
# New format can also have InitA and DeltaA
EP_NEW_DETAIL = re.compile(
    r'A=([0-9eE.+-]+)\s+InitA=([0-9eE.+-]+)\s+DeltaA=([0-9eE.+-]+)'
)
# Old format: Step  XXXXX | Eps: NNNN | R: X.XXX | Len: X.X | ...
STEP_OLD = re.compile(
    r'Step\s+([0-9,]+)\s*\|\s*Eps:\s*(\d+)\s*\|\s*R:\s*([0-9eE.+-]+)\s*\|\s*Len:\s*([0-9.]+)'
)
# Gate schedule
GATE_RE = re.compile(
    r'\[Gate Schedule\]\s+Step\s+([0-9,]+):\s+gate\s+([0-9eE.+-]+)\s*->\s*([0-9eE.+-]+)'
)
# Resume
RESUME_RE = re.compile(r'Resumed at step\s+([0-9,]+)')
# VecEnv config line
VECENV_RE = re.compile(r'VecEnv:.*active_clipping=(True|False)')
# Online pool config
ONLINE_POOL_RE = re.compile(r'Online TCR pool enabled.*min_affinity=([0-9eE.+-]+)')
# Reward mode line
REWARD_MODE_RE = re.compile(r'reward_mode=(\S+)')
# Peptide count
PEPTIDE_RE = re.compile(r'Filtered targets:\s+(\d+)\s+peptides')
# n_envs
NENVS_RE = re.compile(r'n_envs=(\d+)')

def ps(s):
    return int(s.replace(",", ""))

def parse_log(path):
    """Parse a training log and return structured data."""
    episodes = []
    step_summaries = []
    gates = []
    resumed_at = None
    active_clip = None
    reward_mode = None
    n_pep = None
    n_envs_log = None
    online_pool_min_aff = None

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Resume
                m = RESUME_RE.search(line)
                if m:
                    resumed_at = ps(m.group(1))

                # New episode format
                m = EP_NEW.search(line)
                if m:
                    ep = {
                        "ep": int(m.group(1)),
                        "step": ps(m.group(2)),
                        "R": float(m.group(3)),
                        "len": int(m.group(4)),
                        "A": float(m.group(5)),
                    }
                    m2 = EP_NEW_DETAIL.search(line)
                    if m2:
                        ep["InitA"] = float(m2.group(2))
                        ep["DeltaA"] = float(m2.group(3))
                    # Check for decoy info
                    dm = re.search(r'DecViol=([0-9eE.+-]+)', line)
                    if dm:
                        ep["DecViol"] = float(dm.group(1))
                    dm = re.search(r'DecA=([0-9eE.+-]+)', line)
                    if dm:
                        ep["DecA"] = float(dm.group(1))
                    episodes.append(ep)
                    continue

                # Old step summary format
                m = STEP_OLD.search(line)
                if m:
                    step_summaries.append({
                        "step": ps(m.group(1)),
                        "eps": int(m.group(2)),
                        "R": float(m.group(3)),
                        "len": float(m.group(4)),
                    })
                    continue

                # Gate
                m = GATE_RE.search(line)
                if m:
                    gates.append({
                        "step": ps(m.group(1)),
                        "gate_to": float(m.group(3)),
                    })

                # Config extraction from log
                if active_clip is None:
                    m = VECENV_RE.search(line)
                    if m:
                        active_clip = m.group(1) == "True"
                if reward_mode is None and "reward_mode=" in line:
                    m = REWARD_MODE_RE.search(line)
                    if m:
                        reward_mode = m.group(1).rstrip(",")
                if n_pep is None:
                    m = PEPTIDE_RE.search(line)
                    if m:
                        n_pep = int(m.group(1))
                if n_envs_log is None and "n_envs=" in line:
                    m = NENVS_RE.search(line)
                    if m:
                        n_envs_log = int(m.group(1))
                if online_pool_min_aff is None:
                    m = ONLINE_POOL_RE.search(line)
                    if m:
                        online_pool_min_aff = float(m.group(1))
    except Exception as e:
        print(f"  WARN: {path}: {e}", file=sys.stderr)

    return {
        "episodes": episodes,
        "step_summaries": step_summaries,
        "gates": gates,
        "resumed_at": resumed_at,
        "active_clip_from_log": active_clip,
        "reward_mode_from_log": reward_mode,
        "n_peptides_from_log": n_pep,
        "n_envs_from_log": n_envs_log,
        "online_pool_min_aff": online_pool_min_aff,
    }

def affinity_stats(episodes, window=500):
    """Compute affinity statistics from new-format episodes."""
    if not episodes:
        return {}
    affs = [e["A"] for e in episodes]
    last_n = affs[-window:]
    roll_w = min(50, len(affs))
    best_roll = max(sum(affs[i:i+roll_w])/roll_w for i in range(len(affs)-roll_w+1)) if len(affs) >= roll_w else max(affs)

    # Delta A stats if available
    deltas = [e.get("DeltaA") for e in episodes if e.get("DeltaA") is not None]
    delta_stats = {}
    if deltas:
        delta_stats = {
            "mean_deltaA_last500": sum(deltas[-window:]) / len(deltas[-window:]),
        }

    return {
        "first_step": episodes[0]["step"],
        "last_step": episodes[-1]["step"],
        "total_episodes": len(episodes),
        "mean_A_all": sum(affs)/len(affs),
        "mean_A_last500": sum(last_n)/len(last_n),
        "max_A_all": max(affs),
        "max_A_last500": max(last_n),
        "min_A_last500": min(last_n),
        "best_rolling_50": best_roll,
        **delta_stats,
    }

def reward_stats_old(step_summaries, window=50):
    """Compute stats from old-format step summaries (no per-episode A)."""
    if not step_summaries:
        return {}
    rs = [s["R"] for s in step_summaries]
    last_n = rs[-window:]
    return {
        "first_step": step_summaries[0]["step"],
        "last_step": step_summaries[-1]["step"],
        "total_step_summaries": len(step_summaries),
        "mean_R_all": sum(rs)/len(rs),
        "mean_R_last50": sum(last_n)/len(last_n),
        "max_R_all": max(rs),
        "note": "old_format_no_affinity",
    }

def find_log(name):
    """Find training log for experiment name."""
    candidates = []
    for d in [LOGS_DIR, ARCHIVE_DIR]:
        if not d.exists():
            continue
        for f in d.glob("*train.log"):
            if name in f.stem:
                candidates.append(f)
    # Prefer exact match
    for c in candidates:
        if c.stem == f"{name}_train":
            return c
    # Prefer longer match (more specific)
    if candidates:
        candidates.sort(key=lambda p: len(p.stem), reverse=False)
        return candidates[0]
    return None

def main():
    # ─── 1. Read all experiment.json ───────────────────────────────────
    all_experiments = []
    for exp_dir in sorted(OUTPUT_DIR.iterdir()):
        ej = exp_dir / "experiment.json"
        if not ej.exists():
            continue
        try:
            with open(ej) as f:
                data = json.load(f)
        except:
            continue
        cfg = data.get("config", {})
        weights = cfg.get("weights", {})

        rec = {
            "name": data.get("name", exp_dir.name),
            "dir": exp_dir.name,
            "launched_at": data.get("launched_at", ""),
            "status": data.get("status", ""),
            "notes": data.get("notes", ""),
            # Config
            "seed": cfg.get("seed"),
            "reward_mode": cfg.get("reward_mode", ""),
            "total_timesteps": cfg.get("total_timesteps"),
            "n_envs": cfg.get("n_envs"),
            "lr": cfg.get("learning_rate"),
            "hidden_dim": cfg.get("hidden_dim"),
            "max_steps": cfg.get("max_steps"),
            "scorer": cfg.get("affinity_scorer", ""),
            "w_aff": weights.get("affinity"),
            "w_dec": weights.get("decoy"),
            "w_nat": weights.get("naturalness"),
            "w_div": weights.get("diversity"),
            "entropy_coef": cfg.get("entropy_coef"),
            "active_clip": cfg.get("active_clipping", False),
            "terminal_only": cfg.get("terminal_reward_only"),
            "ban_stop": cfg.get("ban_stop"),
            "use_znorm": cfg.get("use_znorm", False),
            "decoy_K": cfg.get("decoy_K"),
            "n_contrast_decoys": cfg.get("n_contrast_decoys", 0),
            "train_targets": cfg.get("train_targets", ""),
            "pmhc_transform": cfg.get("pmhc_embedding_transform", "none"),
            "sub_only": cfg.get("sub_only", False),
        }
        all_experiments.append(rec)

    print(f"Found {len(all_experiments)} experiments with experiment.json")

    # ─── 2. Parse training logs ────────────────────────────────────────
    for rec in all_experiments:
        log = find_log(rec["name"])
        if not log:
            rec["has_log"] = False
            continue
        rec["has_log"] = True
        rec["log_file"] = log.name

        parsed = parse_log(log)

        # Override config from log if not in experiment.json
        if parsed["active_clip_from_log"] is not None and not rec.get("active_clip"):
            rec["active_clip_log"] = parsed["active_clip_from_log"]
        if parsed["n_peptides_from_log"]:
            rec["n_peptides"] = parsed["n_peptides_from_log"]
        rec["resumed_from"] = parsed["resumed_at"]
        rec["online_pool_min_aff"] = parsed["online_pool_min_aff"]
        rec["has_online_pool"] = parsed["online_pool_min_aff"] is not None

        if parsed["episodes"]:
            stats = affinity_stats(parsed["episodes"])
            rec.update(stats)
            rec["log_format"] = "new"
        elif parsed["step_summaries"]:
            stats = reward_stats_old(parsed["step_summaries"])
            rec.update(stats)
            rec["log_format"] = "old"
        else:
            rec["log_format"] = "empty"

        if parsed["gates"]:
            rec["gate_schedule"] = parsed["gates"][-1]["gate_to"]
            rec["n_gate_changes"] = len(parsed["gates"])

    # ─── 3. Sort by affinity (best first) ──────────────────────────────
    def sort_key(r):
        a = r.get("mean_A_last500")
        if a is not None:
            return (2, a)  # New format with affinity - best
        a = r.get("mean_R_last50")
        if a is not None:
            return (1, a)  # Old format with reward
        return (0, 0)  # No data

    all_experiments.sort(key=sort_key, reverse=True)

    # ─── 4. Save ───────────────────────────────────────────────────────
    out_path = LOGS_DIR / "experiment_analysis_full.json"
    with open(out_path, "w") as f:
        json.dump(all_experiments, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    # ─── 5. Print summary ─────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"{'Rank':>4} {'Name':<52} {'Scorer':<10} {'MeanA500':<10} {'MaxA':<9} {'Best50':<9} {'Steps':<10} {'LR':<10} {'EntC':<7} {'AClip':<6} {'RMode':<20}")
    print(f"{'='*130}")

    for i, r in enumerate(all_experiments):
        ma = r.get("mean_A_last500")
        mx = r.get("max_A_all")
        br = r.get("best_rolling_50")
        ls = r.get("last_step")
        sc = r.get("scorer", "?")[:9]
        lr = r.get("lr")
        ec = r.get("entropy_coef")
        ac = r.get("active_clip", False)
        rm = r.get("reward_mode", "")[:19]

        if ma is not None:
            print(f"{i+1:>4} {r['name']:<52} {sc:<10} {ma:>9.4f} {mx:>8.4f} {br:>8.4f} {ls:>9,} {lr or 0:>9.6f} {ec or 0:>6.4f} {'Y' if ac else 'N':<6} {rm:<20}")
        elif r.get("mean_R_last50") is not None:
            mr = r["mean_R_last50"]
            ls2 = r.get("last_step", 0)
            print(f"{i+1:>4} {r['name']:<52} {sc:<10} {'(R only)':<10} {mr:>8.4f} {'N/A':<9} {ls2:>9,} {lr or 0:>9.6f} {ec or 0:>6.4f} {'Y' if ac else 'N':<6} {rm:<20}")

    # Count by type
    new_fmt = sum(1 for r in all_experiments if r.get("log_format") == "new")
    old_fmt = sum(1 for r in all_experiments if r.get("log_format") == "old")
    no_log = sum(1 for r in all_experiments if not r.get("has_log"))
    print(f"\nLog format: {new_fmt} new (with A=), {old_fmt} old (R only), {no_log} no log found")

if __name__ == "__main__":
    main()
