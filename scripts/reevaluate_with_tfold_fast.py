#!/usr/bin/env python
"""Re-evaluate top models with tFold scorer - fast sampling version.

CRITICAL BUG FIX (2026-04-30):
- Previous version used cache_miss_score=-0.5, which produced raw scores of -1.5
  (outside valid range [-1, 0]). This caused cache misses to be undetected and
  all cache-miss TCRs to be scored as -1.5, resulting in AUROC=0.5 (no discrimination).
- Fixed: cache_miss_score=0.5 (default), which produces raw = -0.5 for cache misses.
- Cache miss detection checks for s == -0.5 (line 113).

This script:
1. Evaluates only top N TCRs per target (default: 10)
2. Uses only M decoys per TCR (default: 5, sampled from tiers A/B/D)
3. Merges tFold results into eval_results_with_tfold.json

Usage:
    python scripts/reevaluate_with_tfold_fast.py --models test41 --n_tcrs 10 --n_decoys 5
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer


def load_decoys_for_peptide(peptide: str, n_decoys: int = 5) -> Dict[str, List[str]]:
    """Load a sample of decoys from decoy library for a given peptide."""
    decoy_library_path = "/share/liuyutian/pMHC_decoy_library"
    data_dir = os.path.join(decoy_library_path, "data")

    all_decoys = []
    decoy_tiers = []

    # Load tier A (point mutants)
    tier_a_dir = os.path.join(data_dir, "decoy_a", peptide)
    if os.path.exists(tier_a_dir):
        tier_a_file = os.path.join(tier_a_dir, "decoy_a_results.json")
        if os.path.exists(tier_a_file):
            with open(tier_a_file) as f:
                data = json.load(f)
                tier_a_seqs = [entry["sequence"] for entry in data][:5]
                all_decoys.extend(tier_a_seqs)
                decoy_tiers.extend(["A"] * len(tier_a_seqs))

    # Load tier B (2-3 AA mutants)
    tier_b_dir = os.path.join(data_dir, "decoy_b", peptide)
    if os.path.exists(tier_b_dir):
        tier_b_file = os.path.join(tier_b_dir, "decoy_b_results.json")
        if os.path.exists(tier_b_file):
            with open(tier_b_file) as f:
                data = json.load(f)
                tier_b_seqs = [entry["sequence"] for entry in data][:5]
                all_decoys.extend(tier_b_seqs)
                decoy_tiers.extend(["B"] * len(tier_b_seqs))

    # Load tier D (known binders)
    tier_d_dir = os.path.join(data_dir, "decoy_d", peptide)
    if os.path.exists(tier_d_dir):
        tier_d_file = os.path.join(tier_d_dir, "decoy_d_results.json")
        if os.path.exists(tier_d_file):
            with open(tier_d_file) as f:
                data = json.load(f)
                tier_d_seqs = [entry["sequence"] for entry in data][:5]
                all_decoys.extend(tier_d_seqs)
                decoy_tiers.extend(["D"] * len(tier_d_seqs))

    # Sample n_decoys from all available
    if len(all_decoys) > n_decoys:
        indices = np.random.choice(len(all_decoys), n_decoys, replace=False)
        all_decoys = [all_decoys[i] for i in indices]
        decoy_tiers = [decoy_tiers[i] for i in indices]

    return {"decoys": all_decoys, "tiers": decoy_tiers}


def evaluate_with_tfold(
    tcrs: List[str],
    peptide: str,
    tfold_scorer: AffinityTFoldScorer,
    tfold_scorer_cache_only: AffinityTFoldScorer,
    n_tcrs: int = 10,
    n_decoys: int = 5,
) -> Dict:
    """Evaluate top N TCRs with tFold scorer using M decoys."""

    # Take only top N TCRs
    tcrs = tcrs[:n_tcrs]
    n_tcrs_actual = len(tcrs)

    print(f"  Evaluating top {n_tcrs_actual} TCRs for {peptide} with tFold (fast sampling)...")

    # Get decoys
    decoy_data = load_decoys_for_peptide(peptide, n_decoys)
    all_decoys = decoy_data["decoys"]
    decoy_tier_labels = decoy_data["tiers"]

    if not all_decoys:
        print(f"    WARNING: No decoys found for {peptide}")
        return None

    print(f"    Using {len(all_decoys)} decoys (A:{decoy_tier_labels.count('A')}, B:{decoy_tier_labels.count('B')}, D:{decoy_tier_labels.count('D')})")

    # Phase 1: Try cache-only scoring for all targets
    print(f"    Phase 1: Cache-only scoring for {n_tcrs_actual} targets...")
    target_scores_cached, _ = tfold_scorer_cache_only.score_batch(
        tcrs, [peptide] * n_tcrs_actual
    )

    # Identify cache misses (score == -0.5 means cache miss in cache_only mode)
    # CRITICAL: With cache_miss_score=0.5, raw score = 0.5 - 1.0 = -0.5
    # DO NOT check for -1.5 (that was the bug when cache_miss_score=-0.5)
    cache_miss_indices = [i for i, s in enumerate(target_scores_cached) if s == -0.5]
    cache_hits = n_tcrs_actual - len(cache_miss_indices)
    print(f"      Cache hits: {cache_hits}/{n_tcrs_actual}, misses: {len(cache_miss_indices)}")

    # Phase 2: Score cache misses one-by-one with retries
    target_scores = list(target_scores_cached)
    if cache_miss_indices:
        print(f"    Phase 2: Scoring {len(cache_miss_indices)} cache misses...")
        for idx_num, i in enumerate(cache_miss_indices):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    score, _ = tfold_scorer.score(tcrs[i], peptide)
                    target_scores[i] = score
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"        Retry {attempt+1} for TCR {i}: {str(e)[:50]}")
                        time.sleep(5)
                    else:
                        print(f"        ERROR: Failed TCR {i} after {max_retries} attempts")
                        target_scores[i] = 0.0

    # Phase 3: Score decoys for each TCR (cache-first)
    print(f"    Phase 3: Scoring decoys ({len(all_decoys)} per TCR)...")
    per_tcr_decoy_scores = []
    per_tcr_aurocs = []

    for i, tcr in enumerate(tcrs):
        # Try cache-only first
        decoy_scores_cached, _ = tfold_scorer_cache_only.score_batch(
            [tcr] * len(all_decoys), all_decoys
        )

        # Score cache misses (cache miss = -0.5 in cache_only mode with cache_miss_score=0.5)
        # CRITICAL: Check for -0.5, not -1.5 (that was the bug)
        decoy_scores = list(decoy_scores_cached)
        decoy_miss_indices = [j for j, s in enumerate(decoy_scores) if s == -0.5]

        if decoy_miss_indices:
            for j in decoy_miss_indices:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        score, _ = tfold_scorer.score(tcr, all_decoys[j])
                        decoy_scores[j] = score
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            decoy_scores[j] = 0.0

        per_tcr_decoy_scores.append(decoy_scores)

        # Per-TCR AUROC
        labels = np.array([1] + [0] * len(decoy_scores))
        scores_arr = np.concatenate([[target_scores[i]], decoy_scores])
        try:
            auroc = roc_auc_score(labels, scores_arr)
        except ValueError:
            auroc = 0.5
        per_tcr_aurocs.append(auroc)

    # Aggregate metrics
    mean_auroc = float(np.mean(per_tcr_aurocs))
    mean_target = float(np.mean(target_scores))
    mean_decoy = float(np.mean([s for ds in per_tcr_decoy_scores for s in ds]))
    std_target = float(np.std(target_scores))
    std_decoy = float(np.std([s for ds in per_tcr_decoy_scores for s in ds]))

    # Per-tier AUROC
    tier_aurocs = {}
    unique_tiers = sorted(set(decoy_tier_labels))
    for tier in unique_tiers:
        tier_indices = [j for j, t in enumerate(decoy_tier_labels) if t == tier]
        if not tier_indices:
            continue
        tier_per_tcr = []
        for i in range(n_tcrs_actual):
            pos = np.array([target_scores[i]])
            neg = np.array([per_tcr_decoy_scores[i][j] for j in tier_indices])
            if len(neg) == 0:
                continue
            labels = np.array([1] + [0] * len(neg))
            scores_arr = np.concatenate([pos, neg])
            try:
                a = roc_auc_score(labels, scores_arr)
            except ValueError:
                a = 0.5
            tier_per_tcr.append(a)
        if tier_per_tcr:
            tier_aurocs[tier] = float(np.mean(tier_per_tcr))

    # Ranking metrics
    target_arr = np.array(target_scores)
    auroc_arr = np.array(per_tcr_aurocs)
    ranked_idx = np.argsort(target_arr)[::-1]
    ranked_target = target_arr[ranked_idx]
    ranked_auroc = auroc_arr[ranked_idx]

    n = len(ranked_target)
    top1_target = float(ranked_target[0]) if n >= 1 else 0.0
    top3_target = float(np.mean(ranked_target[:min(3, n)])) if n >= 1 else 0.0
    top5_target = float(np.mean(ranked_target[:min(5, n)])) if n >= 1 else 0.0
    top1_auroc = float(ranked_auroc[0]) if n >= 1 else 0.5
    top3_auroc = float(np.mean(ranked_auroc[:min(3, n)])) if n >= 1 else 0.5
    top5_auroc = float(np.mean(ranked_auroc[:min(5, n)])) if n >= 1 else 0.5
    hit_rate_07 = float(np.mean(target_arr > 0.7)) if n >= 1 else 0.0
    top1_composite = float(ranked_target[0] * ranked_auroc[0]) if n >= 1 else 0.0
    k5 = min(5, n)
    top5_composite = float(np.mean(ranked_target[:k5] * ranked_auroc[:k5])) if n >= 1 else 0.0

    # Per-TCR ranked details
    per_tcr_ranked = []
    for i in range(n):
        idx = ranked_idx[i]
        per_tcr_ranked.append({
            "tcr": tcrs[idx],
            "target_score": float(target_arr[idx]),
            "auroc": float(auroc_arr[idx]),
            "composite": float(target_arr[idx] * auroc_arr[idx]),
            "rank": i + 1
        })

    return {
        "auroc": mean_auroc,
        "mean_target_score": mean_target,
        "mean_decoy_score": mean_decoy,
        "std_target_score": std_target,
        "std_decoy_score": std_decoy,
        "n_tcrs": n_tcrs_actual,
        "n_decoys_per_tcr": len(all_decoys),
        "per_tier_auroc": tier_aurocs,
        "top1_target": top1_target,
        "top3_target": top3_target,
        "top5_target": top5_target,
        "top1_auroc": top1_auroc,
        "top3_auroc": top3_auroc,
        "top5_auroc": top5_auroc,
        "hit_rate_07": hit_rate_07,
        "top1_composite": top1_composite,
        "top5_composite": top5_composite,
        "per_tcr_ranked": per_tcr_ranked
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for tFold")
    parser.add_argument("--n_tcrs", type=int, default=10, help="Number of top TCRs to evaluate per target")
    parser.add_argument("--n_decoys", type=int, default=5, help="Number of decoys to use per TCR")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    # Initialize scorers
    print("Loading tFold scorers...")
    tfold_scorer = AffinityTFoldScorer(device=args.device, cache_only=False)
    # CRITICAL: cache_miss_score=0.5 (default) produces raw score = 0.5 - 1.0 = -0.5
    # This is what we check for in cache miss detection (line 113: s == -0.5)
    # DO NOT change this to -0.5, which would produce raw = -1.5 (out of range)
    tfold_scorer_cache_only = AffinityTFoldScorer(device=args.device, cache_only=True, cache_miss_score=0.5)
    print("tFold scorers loaded successfully")

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")

        eval_dir = f"results/{model_name}_eval"
        input_file = f"{eval_dir}/eval_results.json"
        output_file = f"{eval_dir}/eval_results_with_tfold.json"

        if not os.path.exists(input_file):
            print(f"  ERROR: {input_file} not found, skipping")
            continue

        # Load existing ERGO results
        with open(input_file, "r") as f:
            data = json.load(f)

        # Process each target
        for peptide, target_data in data.items():
            if peptide == "_summary":
                continue

            print(f"\nTarget: {peptide}")

            # Extract TCR sequences from ERGO evaluation
            if "specificity" not in target_data or "ergo" not in target_data["specificity"]:
                print(f"  WARNING: No ERGO results found, skipping")
                continue

            ergo_data = target_data["specificity"]["ergo"]
            if "per_tcr_ranked" not in ergo_data:
                print(f"  WARNING: No per_tcr_ranked data, skipping")
                continue

            # Get TCRs (already ranked by ERGO)
            tcrs = [item["tcr"] for item in ergo_data["per_tcr_ranked"]]

            # Evaluate with tFold
            tfold_results = evaluate_with_tfold(
                tcrs, peptide, tfold_scorer, tfold_scorer_cache_only,
                n_tcrs=args.n_tcrs, n_decoys=args.n_decoys
            )

            if tfold_results is None:
                print(f"  WARNING: tFold evaluation failed, skipping")
                continue

            # Merge into data structure
            if "specificity" not in target_data:
                target_data["specificity"] = {}
            target_data["specificity"]["tfold"] = tfold_results

            print(f"  tFold AUROC: {tfold_results['auroc']:.4f} (n_tcrs={tfold_results['n_tcrs']}, n_decoys={tfold_results['n_decoys_per_tcr']})")
            print(f"  ERGO AUROC:  {ergo_data['auroc']:.4f}")
            print(f"  Difference:  {tfold_results['auroc'] - ergo_data['auroc']:+.4f}")

        # Save updated results
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Saved to: {output_file}")

        # Print summary comparison
        print(f"\n{'='*60}")
        print(f"Summary for {model_name}:")
        print(f"{'='*60}")
        print(f"{'Target':<20} {'ERGO AUROC':>12} {'tFold AUROC':>12} {'Diff':>8}")
        print("-" * 60)

        ergo_aurocs = []
        tfold_aurocs = []
        for peptide, target_data in data.items():
            if peptide == "_summary":
                continue
            if "specificity" in target_data:
                ergo_auc = target_data["specificity"].get("ergo", {}).get("auroc", 0.0)
                tfold_auc = target_data["specificity"].get("tfold", {}).get("auroc", 0.0)
                ergo_aurocs.append(ergo_auc)
                tfold_aurocs.append(tfold_auc)
                diff = tfold_auc - ergo_auc
                print(f"{peptide:<20} {ergo_auc:>12.4f} {tfold_auc:>12.4f} {diff:>+8.4f}")

        if ergo_aurocs and tfold_aurocs:
            mean_ergo = np.mean(ergo_aurocs)
            mean_tfold = np.mean(tfold_aurocs)
            print("-" * 60)
            print(f"{'MEAN':<20} {mean_ergo:>12.4f} {mean_tfold:>12.4f} {mean_tfold - mean_ergo:>+8.4f}")


if __name__ == "__main__":
    main()
