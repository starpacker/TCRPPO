#!/usr/bin/env python3
"""Re-evaluate TCRPPO v1 results with tFold scorer.

This script:
1. Reads v1's eval_decoy_ae_mcpas_v2.csv (contains generated TCRs + ERGO scores)
2. Extracts unique TCRs per target peptide
3. Re-scores with tFold against decoy library
4. Compares tFold AUROC vs v1's ERGO AUROC (0.4538)

Usage:
    python scripts/eval_v1_csv_with_tfold.py --n_decoys 50
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer


# 12 McPAS target peptides
TARGET_PEPTIDES = [
    "GILGFVFTL",
    "NLVPMVATV",
    "GLCTLVAML",
    "LLWNGPMAV",
    "YLQPRTFLL",
    "FLYALALLL",
    "SLYNTVATL",
    "KLGGALQAK",
    "AVFDRKSDAK",
    "IVTDFSVIK",
    "SPRWYFYYL",
    "RLRAEAQVK",
]


def load_v1_tcrs_from_csv(csv_path: str) -> Dict[str, List[str]]:
    """Extract unique TCRs per target from v1 eval CSV."""
    print(f"Loading v1 TCRs from {csv_path}...")

    tcrs_per_target = defaultdict(list)
    seen_per_target = defaultdict(set)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_pep = row['target_pep']
            final_tcr = row['final_tcr']
            is_target = int(row['is_target_pep'])

            # Only collect from target peptide rows (not decoy rows)
            if is_target == 1 and target_pep in TARGET_PEPTIDES:
                if final_tcr not in seen_per_target[target_pep]:
                    tcrs_per_target[target_pep].append(final_tcr)
                    seen_per_target[target_pep].add(final_tcr)

    # Print summary
    for pep in TARGET_PEPTIDES:
        print(f"  {pep}: {len(tcrs_per_target[pep])} unique TCRs")

    return dict(tcrs_per_target)


def load_decoys_for_peptide(peptide: str, n_decoys: int = 50) -> Dict[str, List]:
    """Load decoys from decoy library for a given peptide."""
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
                tier_a_seqs = [entry["sequence"] for entry in data]
                all_decoys.extend(tier_a_seqs)
                decoy_tiers.extend(["A"] * len(tier_a_seqs))

    # Load tier B (2-3 AA mutants)
    tier_b_dir = os.path.join(data_dir, "decoy_b", peptide)
    if os.path.exists(tier_b_dir):
        tier_b_file = os.path.join(tier_b_dir, "decoy_b_results.json")
        if os.path.exists(tier_b_file):
            with open(tier_b_file) as f:
                data = json.load(f)
                tier_b_seqs = [entry["sequence"] for entry in data]
                all_decoys.extend(tier_b_seqs)
                decoy_tiers.extend(["B"] * len(tier_b_seqs))

    # Load tier D (known binders)
    tier_d_dir = os.path.join(data_dir, "decoy_d", peptide)
    if os.path.exists(tier_d_dir):
        tier_d_file = os.path.join(tier_d_dir, "decoy_d_results.json")
        if os.path.exists(tier_d_file):
            with open(tier_d_file) as f:
                data = json.load(f)
                tier_d_seqs = [entry["sequence"] for entry in data]
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
    n_decoys: int = 50,
) -> Dict:
    """Evaluate TCRs with tFold scorer."""

    print(f"  Evaluating {len(tcrs)} TCRs for {peptide} with tFold...")

    # Get decoys
    decoy_data = load_decoys_for_peptide(peptide, n_decoys)
    all_decoys = decoy_data["decoys"]
    decoy_tier_labels = decoy_data["tiers"]

    if not all_decoys:
        print(f"    WARNING: No decoys found for {peptide}")
        return None

    print(f"    Using {len(all_decoys)} decoys (A:{decoy_tier_labels.count('A')}, "
          f"B:{decoy_tier_labels.count('B')}, D:{decoy_tier_labels.count('D')})")

    # Score targets
    print(f"    Scoring {len(tcrs)} targets...")
    target_scores, _ = tfold_scorer.score_batch(tcrs, [peptide] * len(tcrs))

    # Score decoys for each TCR
    print(f"    Scoring decoys...")
    per_tcr_aurocs = []
    per_tcr_decoy_scores = []

    for i, tcr in enumerate(tcrs):
        if (i + 1) % 10 == 0:
            print(f"      TCR {i+1}/{len(tcrs)}")

        # Score all decoys for this TCR
        decoy_scores, _ = tfold_scorer.score_batch(
            [tcr] * len(all_decoys), all_decoys
        )
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

    print(f"    AUROC: {mean_auroc:.4f}, Target: {mean_target:.4f}, Decoy: {mean_decoy:.4f}")

    return {
        "auroc": mean_auroc,
        "mean_target_score": mean_target,
        "mean_decoy_score": mean_decoy,
        "std_target_score": std_target,
        "std_decoy_score": std_decoy,
        "n_tcrs": len(tcrs),
        "n_decoys": len(all_decoys),
        "per_tcr_aurocs": per_tcr_aurocs,
        "target_scores": target_scores,
        "decoy_tiers": decoy_tier_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1_csv",
        default="/share/liuyutian/TCRPPO/evaluation/results/decoy/eval_decoy_ae_mcpas_v2.csv",
        help="Path to v1 eval CSV",
    )
    parser.add_argument("--n_decoys", type=int, default=50, help="Decoys per TCR")
    parser.add_argument(
        "--output",
        default="results/v1_tfold_eval/eval_results.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load v1 TCRs from CSV
    tcrs_per_target = load_v1_tcrs_from_csv(args.v1_csv)

    # Initialize tFold scorer
    print("\nInitializing tFold scorer...")
    tfold_scorer = AffinityTFoldScorer()

    # Evaluate each target
    results = {}
    all_aurocs = []

    for peptide in TARGET_PEPTIDES:
        print(f"\n{'='*60}")
        print(f"[{peptide}]")
        print(f"{'='*60}")

        tcrs = tcrs_per_target.get(peptide, [])
        if not tcrs:
            print(f"  WARNING: No TCRs found for {peptide}")
            continue

        # Evaluate with tFold
        eval_result = evaluate_with_tfold(tcrs, peptide, tfold_scorer, args.n_decoys)

        if eval_result:
            results[peptide] = {
                "specificity": {
                    "auroc": eval_result["auroc"],
                    "mean_target_score": eval_result["mean_target_score"],
                    "mean_decoy_score": eval_result["mean_decoy_score"],
                    "std_target_score": eval_result["std_target_score"],
                    "std_decoy_score": eval_result["std_decoy_score"],
                    "n_tcrs": eval_result["n_tcrs"],
                    "n_decoys": eval_result["n_decoys"],
                },
                "generated_tcrs": tcrs,
            }
            all_aurocs.append(eval_result["auroc"])

    # Summary
    mean_auroc = float(np.mean(all_aurocs))
    results["_summary"] = {
        "mean_auroc_tfold": mean_auroc,
        "v1_ergo_baseline": 0.4538,
        "delta_vs_ergo": mean_auroc - 0.4538,
        "scorer": "tFold",
        "n_targets": len(all_aurocs),
        "n_decoys_per_tcr": args.n_decoys,
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Mean AUROC (tFold):  {mean_auroc:.4f}")
    print(f"v1 Baseline (ERGO):  0.4538")
    print(f"Delta:               {mean_auroc - 0.4538:+.4f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
