#!/usr/bin/env python3
"""Compare trained-PPO decoy eval against the random-TCR baseline.

Reads two CSVs produced by ``eval_decoy.py`` (the trained run) and
``eval_decoy_random_baseline.py`` (the null baseline), and prints a
side-by-side AUROC table per target peptide. The conclusion of the
specificity investigation hinges on the **delta** between these two columns:

* Delta ≈ 0  → ERGO is the bottleneck. PPO neither helps nor hurts; the
  scorer simply gives the same predictions to similar inputs.

* Delta < 0 (PPO worse than random) → PPO training actively destroys
  specificity. The reward function is collapsing TCRs into a region where
  ERGO can no longer discriminate.

* Delta > 0 (PPO better than random) → PPO training is doing some real work,
  even if absolute AUROC is still below the user's threshold.

Usage
-----
    python evaluation/compare_trained_vs_random.py \
        --trained_csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv \
        --random_csv  evaluation/results/decoy/eval_decoy_random_pool_ae_mcpas.csv

Note: this script intentionally re-implements only the AUROC comparison; for
full per-tier and uncertainty analysis run ``eval_decoy_metrics.py`` on each
CSV separately.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eval_decoy_metrics import load_csv, auroc, safe_mean  # noqa: E402


def per_target_auroc(rows):
    """Return {target_pep: auroc_target_vs_decoys} for the given CSV rows."""
    by_target = defaultdict(lambda: {"t": [], "d": []})
    for r in rows:
        if r["is_target_pep"] == 1:
            by_target[r["target_pep"]]["t"].append(r["ergo_mean"])
        else:
            by_target[r["target_pep"]]["d"].append(r["ergo_mean"])
    out = {}
    for tgt, b in by_target.items():
        out[tgt] = {
            "auroc": auroc(b["t"], b["d"]),
            "target_mean": safe_mean(b["t"]),
            "decoy_mean": safe_mean(b["d"]),
            "n_target": len(b["t"]),
            "n_decoy": len(b["d"]),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trained_csv", required=True,
                   help="CSV from eval_decoy.py (trained PPO run)")
    p.add_argument("--random_csv", required=True,
                   help="CSV from eval_decoy_random_baseline.py (null baseline)")
    args = p.parse_args()

    for path in (args.trained_csv, args.random_csv):
        if not os.path.isfile(path):
            print("ERROR: not found: {}".format(path))
            sys.exit(1)

    print("[load] trained: {}".format(args.trained_csv))
    trained_rows = load_csv(args.trained_csv)
    print("[load] random:  {}".format(args.random_csv))
    random_rows = load_csv(args.random_csv)

    trained = per_target_auroc(trained_rows)
    random_ = per_target_auroc(random_rows)

    targets = sorted(set(trained.keys()) | set(random_.keys()))

    print("\n" + "=" * 90)
    print("TRAINED vs RANDOM-TCR BASELINE — per-target AUROC comparison")
    print("=" * 90)
    print("{:<14} {:>10} {:>10} {:>10} {:>12} {:>14}".format(
        "Target", "AUROC_RL", "AUROC_RND", "Delta", "TgtMean_RL", "TgtMean_RND"))
    print("-" * 90)

    delta_sum = 0.0
    delta_n = 0
    rl_aurocs = []
    rnd_aurocs = []
    for tgt in targets:
        rl = trained.get(tgt, {})
        rnd = random_.get(tgt, {})
        a_rl = rl.get("auroc", float("nan"))
        a_rnd = rnd.get("auroc", float("nan"))
        delta = a_rl - a_rnd if not (np.isnan(a_rl) or np.isnan(a_rnd)) else float("nan")
        if not np.isnan(delta):
            delta_sum += delta
            delta_n += 1
        rl_aurocs.append(a_rl)
        rnd_aurocs.append(a_rnd)

        print("{:<14} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f} {:>14.4f}".format(
            tgt[:14],
            a_rl,
            a_rnd,
            delta,
            rl.get("target_mean", float("nan")),
            rnd.get("target_mean", float("nan")),
        ))

    print("-" * 90)
    mean_rl = np.nanmean(rl_aurocs)
    mean_rnd = np.nanmean(rnd_aurocs)
    mean_delta = (delta_sum / delta_n) if delta_n else float("nan")
    print("{:<14} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "MEAN", mean_rl, mean_rnd, mean_delta))
    print("=" * 90)

    # Verdict
    print("\nVerdict:")
    if abs(mean_delta) < 0.02:
        print("  Δ AUROC ≈ 0  →  ERGO is the bottleneck.")
        print("  The scorer gives near-identical predictions to similar inputs")
        print("  regardless of which TCRs you feed it. Switching to a different")
        print("  binding predictor (pMTnet, NetTCR-2.x, TITAN, …) is the only")
        print("  thing that can move this number.")
    elif mean_delta < -0.02:
        print("  Δ AUROC < 0  →  PPO training is ACTIVELY DESTROYING specificity.")
        print("  Random TCRs do better than the trained agent. The reward function")
        print("  is collapsing TCRs into a region where ERGO can no longer")
        print("  discriminate. Reward redesign (contrastive / multi-objective)")
        print("  is the right intervention.")
    else:
        print("  Δ AUROC > 0  →  PPO is doing real work but absolute AUROC is low.")
        print("  Both the scorer and the reward function contribute. The first")
        print("  cheap improvement is probably reward shaping.")
    print()


if __name__ == "__main__":
    main()
