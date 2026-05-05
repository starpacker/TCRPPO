#!/usr/bin/env python
"""Re-evaluate top models with tFold scorer - parallel batch version.

This script:
1. Loads existing ERGO evaluation results from results/<model>_eval/eval_results.json
2. Re-evaluates the same generated TCRs using tFold scorer (parallel batches)
3. Merges tFold results into the JSON under specificity.tfold
4. Saves updated results to results/<model>_eval/eval_results_with_tfold.json

Usage:
    python scripts/reevaluate_with_tfold_parallel.py --models test41 --device cuda --batch_size 50
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer


def load_decoys_for_peptide(peptide: str) -> Dict[str, List[str]]:
    """Load decoys from decoy library for a given peptide."""
    decoy_library_path = "/share/liuyutian/pMHC_decoy_library"
    data_dir = os.path.join(decoy_library_path, "data")

    decoys_dict = {}

    # Load tier A (point mutants)
    tier_a_dir = os.path.join(data_dir, "decoy_a", peptide)
    if os.path.exists(tier_a_dir):
        tier_a_file = os.path.join(tier_a_dir, "decoy_a_results.json")
        if os.path.exists(tier_a_file):
            with open(tier_a_file) as f:
                data = json.load(f)
                decoys_dict["A"] = [entry["sequence"] for entry in data][:15]

    # Load tier B (2-3 AA mutants)
    tier_b_dir = os.path.join(data_dir, "decoy_b", peptide)
    if os.path.exists(tier_b_dir):
        tier_b_file = os.path.join(tier_b_dir, "decoy_b_results.json")
        if os.path.exists(tier_b_file):
            with open(tier_b_file) as f:
                data = json.load(f)
                decoys_dict["B"] = [entry["sequence"] for entry in data][:15]

    # Load tier D (known binders)
    tier_d_dir = os.path.join(data_dir, "decoy_d", peptide)
    if os.path.exists(tier_d_dir):
        tier_d_file = os.path.join(tier_d_dir, "decoy_d_results.json")
        if os.path.exists(tier_d_file):
            with open(tier_d_file) as f:
                data = json.load(f)
                decoys_dict["D"] = [entry["sequence"] for entry in data][:20]

    return decoys_dict


def evaluate_with_tfold(
    tcrs: List[str],
    peptide: str,
    tfold_scorer: AffinityTFoldScorer,
    batch_size: int = 50,
) -> Dict:
    """Evaluate TCRs with tFold scorer using parallel batching."""

    n_tcrs = len(tcrs)
    print(f"  Evaluating {n_tcrs} TCRs for {peptide} with tFold (batch_size={batch_size})...")

    # Get decoys
    decoys_dict = load_decoys_for_peptide(peptide)
    all_decoys = []
    decoy_tier_labels = []
    for tier in ["A", "B", "D"]:
        if tier in decoys_dict:
            all_decoys.extend(decoys_dict[tier])
            decoy_tier_labels.extend([tier] * len(decoys_dict[tier]))

    if not all_decoys:
        print(f"    WARNING: No decoys found for {peptide}")
        return None

    print(f"    Found {len(all_decoys)} decoys (A:{len(decoys_dict.get('A',[]))}, B:{len(decoys_dict.get('B',[]))}, D:{len(decoys_dict.get('D',[]))})")

    # Score all target pairs in parallel batches
    print(f"    Scoring {n_tcrs} targets in batches of {batch_size}...")
    target_scores = []
    for i in range(0, n_tcrs, batch_size):
        batch_tcrs = tcrs[i:i+batch_size]
        batch_peptides = [peptide] * len(batch_tcrs)

        batch_scores, _ = tfold_scorer.score_batch(batch_tcrs, batch_peptides)
        target_scores.extend(batch_scores)

        if (i + batch_size) % 100 == 0 or i + batch_size >= n_tcrs:
            print(f"      Progress: {min(i+batch_size, n_tcrs)}/{n_tcrs} targets")

    # Score all decoy pairs for each TCR in parallel batches
    print(f"    Scoring decoys ({len(all_decoys)} per TCR) in batches of {batch_size}...")
    per_tcr_decoy_scores = []
    per_tcr_aurocs = []

    for i, tcr in enumerate(tcrs):
        if (i + 1) % 20 == 0:
            print(f"      Progress: {i+1}/{n_tcrs} TCRs")

        # Score all decoys for this TCR in batches
        decoy_scores = []
        for j in range(0, len(all_decoys), batch_size):
            batch_decoys = all_decoys[j:j+batch_size]
            batch_tcrs = [tcr] * len(batch_decoys)

            batch_scores, _ = tfold_scorer.score_batch(batch_tcrs, batch_decoys)
            decoy_scores.extend(batch_scores)

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
        for i in range(n_tcrs):
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
    for i in range(min(n, 50)):  # Top 50 TCRs
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
        "n_tcrs": n_tcrs,
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
    parser.add_argument("--device", type=str, default="cuda", help="Device for tFold")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for parallel scoring")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    # Initialize scorer
    print("Loading tFold scorer...")
    tfold_scorer = AffinityTFoldScorer(device=args.device)
    print("tFold scorer loaded successfully")

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

            # Get all TCRs
            tcrs = [item["tcr"] for item in ergo_data["per_tcr_ranked"]]

            # Evaluate with tFold
            tfold_results = evaluate_with_tfold(tcrs, peptide, tfold_scorer, args.batch_size)

            if tfold_results is None:
                print(f"  WARNING: tFold evaluation failed, skipping")
                continue

            # Merge into data structure
            if "specificity" not in target_data:
                target_data["specificity"] = {}
            target_data["specificity"]["tfold"] = tfold_results

            print(f"  tFold AUROC: {tfold_results['auroc']:.4f}")
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
