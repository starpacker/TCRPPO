#!/usr/bin/env python3
"""Scorer Discrimination Benchmark: Test ERGO/TCBind/NetTCR ability to distinguish target vs decoy peptides.

Uses tc-hard known TCR-peptide pairs + decoy library to evaluate whether affinity scorers
can rank true target peptide above decoy peptides.

Metrics:
- Per-scorer, per-target, per-tier AUROC
- Mean AUROC across targets
- Discrimination ability ranking
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.data.pmhc_loader import EVAL_TARGETS


def load_tc_hard_pairs(tc_hard_path: str, targets: List[str], max_tcrs_per_target: int = 50) -> Dict[str, List[str]]:
    """Load known TCR-peptide pairs from tc-hard."""
    with open(tc_hard_path) as f:
        tc_hard = json.load(f)

    pairs = {}
    for target in targets:
        if target in tc_hard:
            target_data = tc_hard[target]
            # tc-hard format: {peptide: {hla, n_positive, n_unique_cdr3b, cdr3b_sequences}}
            if isinstance(target_data, dict) and "cdr3b_sequences" in target_data:
                tcrs = target_data["cdr3b_sequences"]
            elif isinstance(target_data, list):
                tcrs = target_data
            else:
                continue

            # Sample up to max_tcrs_per_target
            if len(tcrs) > max_tcrs_per_target:
                rng = np.random.default_rng(42)
                tcrs = rng.choice(tcrs, size=max_tcrs_per_target, replace=False).tolist()
            pairs[target] = tcrs

    return pairs


def load_decoys_tier_a(decoy_base: str, target: str, max_decoys: int = 100) -> List[str]:
    """Load tier A decoys (point mutants)."""
    json_path = os.path.join(decoy_base, "decoy_a", target, "decoy_a_results.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path) as f:
        data = json.load(f)

    decoys = [entry["sequence"] for entry in data if "sequence" in entry]
    if len(decoys) > max_decoys:
        rng = np.random.default_rng(42)
        decoys = rng.choice(decoys, size=max_decoys, replace=False).tolist()

    return decoys


def load_decoys_tier_b(decoy_base: str, target: str, max_decoys: int = 50) -> List[str]:
    """Load tier B decoys (structural variants)."""
    json_path = os.path.join(decoy_base, "decoy_b", target, "decoy_b_results.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path) as f:
        data = json.load(f)

    decoys = [entry["sequence"] for entry in data if "sequence" in entry]
    if len(decoys) > max_decoys:
        rng = np.random.default_rng(42)
        decoys = rng.choice(decoys, size=max_decoys, replace=False).tolist()

    return decoys


def load_decoys_tier_d(decoy_base: str, target: str, max_decoys: int = 100) -> List[str]:
    """Load tier D decoys (VDJdb/IEDB known binders)."""
    csv_path = os.path.join(decoy_base, "decoy_d", target, "decoy_d_results.csv")
    if not os.path.exists(csv_path):
        return []

    try:
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns:
            return []

        decoys = df["sequence"].dropna().tolist()
        # Filter to valid peptide lengths (8-11 AA)
        decoys = [d for d in decoys if isinstance(d, str) and 8 <= len(d) <= 11]

        if len(decoys) > max_decoys:
            rng = np.random.default_rng(42)
            decoys = rng.choice(decoys, size=max_decoys, replace=False).tolist()

        return decoys
    except Exception as e:
        print(f"Warning: Failed to load tier D for {target}: {e}")
        return []


def compute_auroc_for_target(
    scorer,
    tcrs: List[str],
    target_peptide: str,
    decoy_peptides: List[str],
) -> float:
    """Compute AUROC for one target: can scorer rank target above decoys?

    For each TCR:
    - Score (TCR, target_peptide) -> label=1
    - Score (TCR, decoy_peptide) for each decoy -> label=0

    Returns AUROC across all (TCR, peptide) pairs.
    """
    if not decoy_peptides:
        return np.nan

    scores = []
    labels = []

    for tcr in tcrs:
        # Target score
        if hasattr(scorer, 'score_batch_fast'):
            target_score = scorer.score_batch_fast([tcr], [target_peptide])[0]
        else:
            target_score, _ = scorer.score(tcr, target_peptide)
        scores.append(target_score)
        labels.append(1)

        # Decoy scores
        if hasattr(scorer, 'score_batch_fast'):
            decoy_scores = scorer.score_batch_fast([tcr] * len(decoy_peptides), decoy_peptides)
        else:
            decoy_scores = [scorer.score(tcr, d)[0] for d in decoy_peptides]

        scores.extend(decoy_scores)
        labels.extend([0] * len(decoy_peptides))

    # Compute AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        # All same label or other issue
        auroc = np.nan

    return auroc


def main():
    parser = argparse.ArgumentParser(description="Benchmark scorer discrimination ability")
    parser.add_argument("--tc_hard_path", default="data/tc_hard_targets.json")
    parser.add_argument("--decoy_base", default="/share/liuyutian/pMHC_decoy_library/data")
    parser.add_argument("--max_tcrs_per_target", type=int, default=50)
    parser.add_argument("--max_decoys_per_tier", type=int, default=100)
    parser.add_argument("--scorers", nargs="+", default=["ergo"], choices=["ergo", "tcbind", "nettcr"])
    parser.add_argument("--output_dir", default="results/scorer_discrimination/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tc-hard pairs
    print(f"Loading tc-hard pairs from {args.tc_hard_path}...")
    tc_hard_pairs = load_tc_hard_pairs(args.tc_hard_path, EVAL_TARGETS, args.max_tcrs_per_target)
    print(f"  Loaded {len(tc_hard_pairs)} targets")
    for target, tcrs in tc_hard_pairs.items():
        print(f"    {target}: {len(tcrs)} TCRs")

    # Initialize scorers
    scorers = {}
    if "ergo" in args.scorers:
        print("\nInitializing ERGO scorer...")
        ergo_model = os.path.join(PROJECT_ROOT, "tcrppo_v2", "ERGO", "models", "ae_mcpas1.pt")
        scorers["ergo"] = AffinityERGOScorer(model_file=ergo_model, device=args.device)

    # TODO: Add TCBind and NetTCR when available
    if "tcbind" in args.scorers:
        print("Warning: TCBind scorer not yet implemented, skipping")
    if "nettcr" in args.scorers:
        print("Warning: NetTCR scorer not yet implemented, skipping")

    # Run benchmark
    results = defaultdict(lambda: defaultdict(dict))

    for scorer_name, scorer in scorers.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking {scorer_name.upper()}")
        print(f"{'='*60}")

        for target in tqdm(tc_hard_pairs.keys(), desc=f"{scorer_name} targets"):
            tcrs = tc_hard_pairs[target]

            # Load decoys for each tier
            decoys_a = load_decoys_tier_a(args.decoy_base, target, args.max_decoys_per_tier)
            decoys_b = load_decoys_tier_b(args.decoy_base, target, args.max_decoys_per_tier)
            decoys_d = load_decoys_tier_d(args.decoy_base, target, args.max_decoys_per_tier)

            # Compute AUROC for each tier
            if decoys_a:
                auroc_a = compute_auroc_for_target(scorer, tcrs, target, decoys_a)
                results[scorer_name][target]["tier_a"] = {
                    "auroc": auroc_a,
                    "n_tcrs": len(tcrs),
                    "n_decoys": len(decoys_a),
                }

            if decoys_b:
                auroc_b = compute_auroc_for_target(scorer, tcrs, target, decoys_b)
                results[scorer_name][target]["tier_b"] = {
                    "auroc": auroc_b,
                    "n_tcrs": len(tcrs),
                    "n_decoys": len(decoys_b),
                }

            if decoys_d:
                auroc_d = compute_auroc_for_target(scorer, tcrs, target, decoys_d)
                results[scorer_name][target]["tier_d"] = {
                    "auroc": auroc_d,
                    "n_tcrs": len(tcrs),
                    "n_decoys": len(decoys_d),
                }

    # Compute summary statistics
    summary = {}
    for scorer_name in scorers.keys():
        summary[scorer_name] = {}
        for tier in ["tier_a", "tier_b", "tier_d"]:
            aurocs = [
                results[scorer_name][target][tier]["auroc"]
                for target in tc_hard_pairs.keys()
                if tier in results[scorer_name][target]
                and not np.isnan(results[scorer_name][target][tier]["auroc"])
            ]
            if aurocs:
                summary[scorer_name][tier] = {
                    "mean_auroc": float(np.mean(aurocs)),
                    "std_auroc": float(np.std(aurocs)),
                    "n_targets": len(aurocs),
                }
            else:
                summary[scorer_name][tier] = {
                    "mean_auroc": np.nan,
                    "std_auroc": np.nan,
                    "n_targets": 0,
                }

        # Overall mean across all tiers
        all_aurocs = []
        for target in tc_hard_pairs.keys():
            for tier in ["tier_a", "tier_b", "tier_d"]:
                if tier in results[scorer_name][target]:
                    auroc = results[scorer_name][target][tier]["auroc"]
                    if not np.isnan(auroc):
                        all_aurocs.append(auroc)

        if all_aurocs:
            summary[scorer_name]["overall"] = {
                "mean_auroc": float(np.mean(all_aurocs)),
                "std_auroc": float(np.std(all_aurocs)),
                "n_samples": len(all_aurocs),
            }
        else:
            summary[scorer_name]["overall"] = {
                "mean_auroc": np.nan,
                "std_auroc": np.nan,
                "n_samples": 0,
            }

    # Save results
    output_file = os.path.join(args.output_dir, "discrimination_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "results": dict(results),
            "summary": summary,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("DISCRIMINATION BENCHMARK SUMMARY")
    print(f"{'='*60}\n")

    # Print summary table
    print(f"{'Scorer':<10} {'Tier':<8} {'Mean AUROC':<12} {'Std':<8} {'N Targets':<10}")
    print("-" * 60)
    for scorer_name in sorted(scorers.keys()):
        for tier in ["tier_a", "tier_b", "tier_d", "overall"]:
            if tier in summary[scorer_name]:
                stats = summary[scorer_name][tier]
                mean_auroc = stats["mean_auroc"]
                std_auroc = stats["std_auroc"]
                n = stats.get("n_targets", stats.get("n_samples", 0))

                if not np.isnan(mean_auroc):
                    print(f"{scorer_name:<10} {tier:<8} {mean_auroc:<12.4f} {std_auroc:<8.4f} {n:<10}")
                else:
                    print(f"{scorer_name:<10} {tier:<8} {'N/A':<12} {'N/A':<8} {n:<10}")

    print(f"\nResults saved to {output_file}")

    # Interpretation guide
    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print("AUROC > 0.7: Good discrimination (scorer can distinguish target from decoys)")
    print("AUROC 0.5-0.7: Moderate discrimination")
    print("AUROC < 0.5: Poor discrimination (worse than random)")
    print("\nTier difficulty:")
    print("  tier_a (point mutants): Easiest — should have highest AUROC")
    print("  tier_b (structural variants): Moderate")
    print("  tier_d (VDJdb known binders): Hardest — cross-reactive peptides")


if __name__ == "__main__":
    main()
