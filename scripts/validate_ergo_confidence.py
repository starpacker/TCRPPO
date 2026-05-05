#!/usr/bin/env python3
"""
Validate ERGO MC Dropout confidence as a signal for reward weighting.

This script checks:
1. Distribution of MC Dropout std across random TCRs
2. Correlation between MC Dropout std and per-TCR AUROC
3. Whether confidence = 1 - std is a useful signal

If validation passes, we proceed with test46 (confidence-weighted reward).
If validation fails, we document why and abandon the approach.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.data.pmhc_loader import PMHCLoader
from tcrppo_v2.data.tcr_pool import TCRPool


def sample_random_tcrs(tcr_pool, n=1000):
    """Sample random TCRs from TCRdb."""
    tcrs = []
    for _ in range(n):
        tcr = tcr_pool.get_random_tcr()
        tcrs.append(tcr)
    return tcrs


def compute_mc_dropout_stats(ergo_scorer, tcrs, peptide, n_forward=10):
    """Compute MC Dropout mean and std for each TCR."""
    results = []

    for i, tcr in enumerate(tcrs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(tcrs)} TCRs...")

        # MC Dropout: use the scorer's built-in mc_dropout_score
        means, stds = ergo_scorer.mc_dropout_score([tcr], [peptide])
        mean_score = float(means[0])
        std_score = float(stds[0])
        confidence = 1.0 - std_score

        results.append({
            'tcr': tcr,
            'mean_score': mean_score,
            'std_score': std_score,
            'confidence': confidence
        })

    return results


def compute_per_tcr_auroc(ergo_scorer, tcrs, peptide, decoy_peptides, n_decoys=20):
    """
    Compute per-TCR AUROC by scoring TCR against target and decoy peptides.

    AUROC measures: does ERGO score this TCR higher on target than on decoys?
    High AUROC = good specificity, low AUROC = poor specificity.
    """
    from sklearn.metrics import roc_auc_score

    results = []

    for i, tcr in enumerate(tcrs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(tcrs)} TCRs...")

        # Score on target (use mean from MC Dropout)
        means, _ = ergo_scorer.mc_dropout_score([tcr], [peptide])
        target_score = float(means[0])

        # Score on decoys
        decoy_scores = []
        selected_decoys = np.random.choice(decoy_peptides, size=min(n_decoys, len(decoy_peptides)), replace=False)
        for decoy in selected_decoys:
            means, _ = ergo_scorer.mc_dropout_score([tcr], [decoy])
            decoy_score = float(means[0])
            decoy_scores.append(decoy_score)

        # Compute AUROC
        if len(decoy_scores) > 0:
            y_true = [1] + [0] * len(decoy_scores)
            y_score = [target_score] + decoy_scores

            try:
                auroc = roc_auc_score(y_true, y_score)
            except:
                auroc = 0.5  # If all scores are identical
        else:
            auroc = 0.5

        results.append({
            'tcr': tcr,
            'target_score': target_score,
            'mean_decoy_score': np.mean(decoy_scores) if decoy_scores else 0.0,
            'auroc': auroc
        })

    return results


def load_decoy_peptides():
    """Load decoy peptides from decoy library."""
    decoy_file = '/share/liuyutian/pMHC_decoy_library/data/decoy_c/decoy_library.json'

    if not os.path.exists(decoy_file):
        print(f"Warning: Decoy library not found at {decoy_file}")
        print("Using hardcoded decoy peptides instead.")
        # Use some common decoys
        return [
            'AAAAAAAAA', 'KKKKKKKKK', 'EEEEEEEEE', 'RRRRRRRRR',
            'LLLLLLLL', 'VVVVVVVV', 'IIIIIIIII', 'FFFFFFFFF',
            'YYYYYYYYYY', 'WWWWWWWWW', 'PPPPPPPP', 'GGGGGGGG'
        ]

    with open(decoy_file) as f:
        data = json.load(f)

    # Extract peptide sequences from entries
    decoys = []
    if 'entries' in data:
        for entry in data['entries']:
            if 'peptide_info' in entry and 'decoy_sequence' in entry['peptide_info']:
                decoys.append(entry['peptide_info']['decoy_sequence'])

    if not decoys:
        # Fallback to hardcoded decoys
        print(f"Warning: Could not parse decoy library, using hardcoded decoys")
        return [
            'AAAAAAAAA', 'KKKKKKKKK', 'EEEEEEEEE', 'RRRRRRRRR',
            'LLLLLLLL', 'VVVVVVVV', 'IIIIIIIII', 'FFFFFFFFF',
            'YYYYYYYYYY', 'WWWWWWWWW', 'PPPPPPPP', 'GGGGGGGG'
        ]

    print(f"Loaded {len(decoys)} decoy peptides from library")
    return decoys


def main():
    print("=" * 70)
    print("ERGO MC Dropout Confidence Validation")
    print("=" * 70)

    # Setup
    print("\n[1/6] Loading ERGO scorer...")
    model_file = '/share/liuyutian/tcrppo_v2/tcrppo_v2/ERGO/models/ae_mcpas1.pt'
    ergo_scorer = AffinityERGOScorer(model_file=model_file)

    print("\n[2/6] Loading TCR pool...")
    tcr_pool = TCRPool()

    print("\n[3/6] Loading pMHC data...")
    pmhc_loader = PMHCLoader()

    # Use a representative peptide
    test_peptide = "GILGFVFTL"
    print(f"\nTest peptide: {test_peptide}")

    # Sample random TCRs
    n_tcrs = 200  # Reduced from 1000 for speed
    print(f"\n[4/6] Sampling {n_tcrs} random TCRs...")
    tcrs = sample_random_tcrs(tcr_pool, n=n_tcrs)
    print(f"Sampled {len(tcrs)} TCRs")

    # Compute MC Dropout stats
    print(f"\n[5/6] Computing MC Dropout stats (10 forward passes per TCR)...")
    mc_results = compute_mc_dropout_stats(ergo_scorer, tcrs, test_peptide, n_forward=10)

    # Extract stats
    mean_scores = [r['mean_score'] for r in mc_results]
    std_scores = [r['std_score'] for r in mc_results]
    confidences = [r['confidence'] for r in mc_results]

    print("\n" + "=" * 70)
    print("MC Dropout std Distribution:")
    print("=" * 70)
    print(f"Mean std:   {np.mean(std_scores):.4f}")
    print(f"Median std: {np.median(std_scores):.4f}")
    print(f"Std of std: {np.std(std_scores):.4f}")
    print(f"Min std:    {np.min(std_scores):.4f}")
    print(f"Max std:    {np.max(std_scores):.4f}")
    print(f"5th percentile:  {np.percentile(std_scores, 5):.4f}")
    print(f"95th percentile: {np.percentile(std_scores, 95):.4f}")

    # Check if std has reasonable variance
    std_range = np.max(std_scores) - np.min(std_scores)
    print(f"\nStd range: {std_range:.4f}")

    if std_range < 0.05:
        print("⚠️  WARNING: MC Dropout std has very low variance!")
        print("   Confidence signal may not be informative.")

    # Plot std distribution
    plt.figure(figsize=(10, 6))
    plt.hist(std_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('MC Dropout std', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'ERGO MC Dropout std Distribution (n={n_tcrs}, peptide={test_peptide})', fontsize=14)
    plt.axvline(np.mean(std_scores), color='red', linestyle='--', label=f'Mean: {np.mean(std_scores):.4f}')
    plt.axvline(np.median(std_scores), color='green', linestyle='--', label=f'Median: {np.median(std_scores):.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/ergo_mc_dropout_std_distribution.png', dpi=150)
    print("\nSaved plot: figures/ergo_mc_dropout_std_distribution.png")

    # Compute per-TCR AUROC
    print(f"\n[6/6] Computing per-TCR AUROC (specificity)...")
    decoy_peptides = load_decoy_peptides()
    auroc_results = compute_per_tcr_auroc(ergo_scorer, tcrs, test_peptide, decoy_peptides, n_decoys=20)

    aurocs = [r['auroc'] for r in auroc_results]

    print("\n" + "=" * 70)
    print("Per-TCR AUROC Distribution:")
    print("=" * 70)
    print(f"Mean AUROC:   {np.mean(aurocs):.4f}")
    print(f"Median AUROC: {np.median(aurocs):.4f}")
    print(f"Std AUROC:    {np.std(aurocs):.4f}")
    print(f"Min AUROC:    {np.min(aurocs):.4f}")
    print(f"Max AUROC:    {np.max(aurocs):.4f}")

    # Merge results
    for i in range(len(tcrs)):
        mc_results[i]['auroc'] = auroc_results[i]['auroc']

    # Compute correlations
    print("\n" + "=" * 70)
    print("Correlation Analysis:")
    print("=" * 70)

    corr_mean_auroc = np.corrcoef(mean_scores, aurocs)[0, 1]
    corr_std_auroc = np.corrcoef(std_scores, aurocs)[0, 1]
    corr_conf_auroc = np.corrcoef(confidences, aurocs)[0, 1]

    print(f"corr(mean_score, AUROC):  {corr_mean_auroc:+.4f}")
    print(f"corr(std_score, AUROC):   {corr_std_auroc:+.4f}")
    print(f"corr(confidence, AUROC):  {corr_conf_auroc:+.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)

    if corr_conf_auroc > 0.3:
        print("✅ POSITIVE correlation: High confidence → High specificity")
        print("   This is GOOD. Confidence-weighted reward should work.")
    elif corr_conf_auroc < -0.3:
        print("❌ NEGATIVE correlation: High confidence → Low specificity")
        print("   This is BAD. Confidence-weighted reward will harm performance.")
    else:
        print("⚠️  WEAK correlation: Confidence does not predict specificity")
        print("   Confidence signal is not informative. Approach may not work.")

    # Plot correlations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Mean score vs AUROC
    axes[0].scatter(mean_scores, aurocs, alpha=0.5, s=20)
    axes[0].set_xlabel('ERGO Mean Score', fontsize=12)
    axes[0].set_ylabel('Per-TCR AUROC', fontsize=12)
    axes[0].set_title(f'Mean Score vs AUROC\ncorr = {corr_mean_auroc:+.3f}', fontsize=14)
    axes[0].grid(alpha=0.3)

    # Std vs AUROC
    axes[1].scatter(std_scores, aurocs, alpha=0.5, s=20, color='orange')
    axes[1].set_xlabel('MC Dropout std', fontsize=12)
    axes[1].set_ylabel('Per-TCR AUROC', fontsize=12)
    axes[1].set_title(f'Std vs AUROC\ncorr = {corr_std_auroc:+.3f}', fontsize=14)
    axes[1].grid(alpha=0.3)

    # Confidence vs AUROC
    axes[2].scatter(confidences, aurocs, alpha=0.5, s=20, color='green')
    axes[2].set_xlabel('Confidence (1 - std)', fontsize=12)
    axes[2].set_ylabel('Per-TCR AUROC', fontsize=12)
    axes[2].set_title(f'Confidence vs AUROC\ncorr = {corr_conf_auroc:+.3f}', fontsize=14)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/ergo_confidence_auroc_correlation.png', dpi=150)
    print("\nSaved plot: figures/ergo_confidence_auroc_correlation.png")

    # Save results
    output = {
        'peptide': test_peptide,
        'n_tcrs': n_tcrs,
        'n_forward_passes': 10,
        'n_decoys': 20,
        'std_distribution': {
            'mean': float(np.mean(std_scores)),
            'median': float(np.median(std_scores)),
            'std': float(np.std(std_scores)),
            'min': float(np.min(std_scores)),
            'max': float(np.max(std_scores)),
            'p5': float(np.percentile(std_scores, 5)),
            'p95': float(np.percentile(std_scores, 95)),
            'range': float(std_range)
        },
        'auroc_distribution': {
            'mean': float(np.mean(aurocs)),
            'median': float(np.median(aurocs)),
            'std': float(np.std(aurocs)),
            'min': float(np.min(aurocs)),
            'max': float(np.max(aurocs))
        },
        'correlations': {
            'mean_score_auroc': float(corr_mean_auroc),
            'std_auroc': float(corr_std_auroc),
            'confidence_auroc': float(corr_conf_auroc)
        },
        'per_tcr_results': mc_results
    }

    os.makedirs('results/validation', exist_ok=True)
    with open('results/validation/ergo_confidence_validation.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nSaved results: results/validation/ergo_confidence_validation.json")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT:")
    print("=" * 70)

    if corr_conf_auroc > 0.3:
        print("✅ PASS: Confidence signal is reliable.")
        print("   Recommendation: Proceed with test46 (confidence-weighted reward)")
        verdict = "PASS"
    elif abs(corr_conf_auroc) < 0.1 and std_range < 0.05:
        print("❌ FAIL: Confidence signal is uninformative (low variance + no correlation)")
        print("   Recommendation: Do NOT proceed with test46")
        verdict = "FAIL"
    elif corr_conf_auroc < -0.1:
        print("❌ FAIL: Confidence signal is negatively correlated with specificity")
        print("   Recommendation: Do NOT proceed with test46")
        verdict = "FAIL"
    else:
        print("⚠️  MARGINAL: Confidence signal is weakly correlated")
        print("   Recommendation: Proceed with caution, or try alternative OOD signals")
        verdict = "MARGINAL"

    output['verdict'] = verdict

    with open('results/validation/ergo_confidence_validation.json', 'w') as f:
        json.dump(output, f, indent=2)

    return verdict


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    verdict = main()
    sys.exit(0 if verdict == "PASS" else 1)
