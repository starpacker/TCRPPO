"""Test scorer ability to discriminate target peptides from decoy peptides.

This is a critical test for RL training. If scorers cannot distinguish TCRs
binding to target vs decoy peptides, the contrastive reward signal will be
meaningless and RL training should not proceed.

Test design:
1. Select target peptides from decoy library
2. For each target, load decoys from tiers A, B, D
3. For each TCR-target pair with high affinity:
   - Score TCR vs target peptide
   - Score TCR vs all decoy peptides
   - Measure score separation
4. Compute metrics:
   - Mean score difference (target - decoy)
   - AUC for target vs decoy discrimination
   - Percentage of cases where target scores higher than all decoys
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.utils.constants import ERGO_MODEL_DIR


def load_decoy_library(target_peptide, decoy_library_root='/share/liuyutian/pMHC_decoy_library/data', max_decoys_per_tier=30):
    """Load decoys for a target peptide from tiers A, B, D.

    Args:
        target_peptide: Target peptide sequence
        decoy_library_root: Root directory of decoy library
        max_decoys_per_tier: Maximum number of decoys to sample per tier (for speed)

    Returns:
        dict: {tier: [decoy_sequences]}
    """
    decoys = {}
    np.random.seed(42)

    # Tier A: Hamming distance variants
    tier_a_file = Path(decoy_library_root) / 'decoy_a' / target_peptide / 'decoy_a_results.json'
    if tier_a_file.exists():
        with open(tier_a_file) as f:
            data = json.load(f)
            all_decoys = [d['sequence'] for d in data if d['sequence'] != target_peptide]
            if len(all_decoys) > max_decoys_per_tier:
                all_decoys = list(np.random.choice(all_decoys, max_decoys_per_tier, replace=False))
            decoys['A'] = all_decoys

    # Tier B: Structure-based similar peptides
    tier_b_file = Path(decoy_library_root) / 'decoy_b' / target_peptide / 'decoy_b_results.json'
    if tier_b_file.exists():
        with open(tier_b_file) as f:
            data = json.load(f)
            all_decoys = [d['sequence'] for d in data if d['sequence'] != target_peptide]
            if len(all_decoys) > max_decoys_per_tier:
                all_decoys = list(np.random.choice(all_decoys, max_decoys_per_tier, replace=False))
            decoys['B'] = all_decoys

    # Tier D: Known binders from VDJdb/IEDB
    tier_d_file = Path(decoy_library_root) / 'decoy_d' / target_peptide / 'decoy_d_results.json'
    if tier_d_file.exists():
        with open(tier_d_file) as f:
            data = json.load(f)
            all_decoys = [d['sequence'] for d in data if d['sequence'] != target_peptide]
            if len(all_decoys) > max_decoys_per_tier:
                all_decoys = list(np.random.choice(all_decoys, max_decoys_per_tier, replace=False))
            decoys['D'] = all_decoys

    return decoys


def get_high_affinity_tcrs(target_peptide, scorer, n_tcrs=50, tcr_pool_file='/share/liuyutian/TCRPPO/data/tcrdb/train_uniq_tcr_seqs.txt'):
    """Sample TCRs and select those with high affinity to target.

    Args:
        target_peptide: Target peptide sequence
        scorer: Scorer instance
        n_tcrs: Number of high-affinity TCRs to return
        tcr_pool_file: Path to TCR pool

    Returns:
        list: High-affinity TCR sequences
    """
    print(f"  Sampling TCRs for {target_peptide}...")

    # Load TCR pool
    with open(tcr_pool_file) as f:
        all_tcrs = [line.strip() for line in f if line.strip()]

    # Sample 1000 TCRs
    np.random.seed(42)
    sampled_tcrs = np.random.choice(all_tcrs, size=min(1000, len(all_tcrs)), replace=False).tolist()

    # Score all TCRs
    peptides = [target_peptide] * len(sampled_tcrs)
    scores, _ = scorer.score_batch(sampled_tcrs, peptides)

    # Select top N
    top_indices = np.argsort(scores)[-n_tcrs:]
    high_affinity_tcrs = [sampled_tcrs[i] for i in top_indices]

    print(f"    Selected {len(high_affinity_tcrs)} TCRs with scores {scores[top_indices[0]]:.3f} to {scores[top_indices[-1]]:.3f}")

    return high_affinity_tcrs


def evaluate_discrimination(scorer, scorer_name, target_peptide, tcrs, decoys_by_tier, output_dir):
    """Evaluate scorer's ability to discriminate target from decoys.

    Args:
        scorer: Scorer instance
        scorer_name: Name for logging
        target_peptide: Target peptide sequence
        tcrs: List of TCR sequences
        decoys_by_tier: Dict of {tier: [decoy_sequences]}
        output_dir: Output directory

    Returns:
        dict: Discrimination metrics
    """
    print(f"\n  Evaluating {scorer_name} on {target_peptide}...")

    results = {
        'scorer': scorer_name,
        'target': target_peptide,
        'n_tcrs': len(tcrs),
        'tiers': {}
    }

    # Score TCRs vs target
    target_scores, _ = scorer.score_batch(tcrs, [target_peptide] * len(tcrs))

    # For each tier, score TCRs vs decoys
    for tier, decoy_seqs in decoys_by_tier.items():
        if not decoy_seqs:
            continue

        print(f"    Tier {tier}: {len(decoy_seqs)} decoys")

        # Score all TCR-decoy pairs
        decoy_scores_matrix = []
        for decoy in tqdm(decoy_seqs, desc=f"      Scoring decoys", leave=False):
            scores, _ = scorer.score_batch(tcrs, [decoy] * len(tcrs))
            decoy_scores_matrix.append(scores)

        decoy_scores_matrix = np.array(decoy_scores_matrix)  # Shape: (n_decoys, n_tcrs)

        # Compute metrics
        # For each TCR, compare target score vs all decoy scores
        target_wins = 0
        score_diffs = []

        for i, tcr in enumerate(tcrs):
            target_score = target_scores[i]
            decoy_scores_for_tcr = decoy_scores_matrix[:, i]

            # Does target score higher than all decoys?
            if target_score > decoy_scores_for_tcr.max():
                target_wins += 1

            # Score difference
            score_diffs.append(target_score - decoy_scores_for_tcr.mean())

        # AUC: treat target as positive class, decoys as negative
        # For each TCR, we have 1 target score and N decoy scores
        y_true = []
        y_score = []
        for i in range(len(tcrs)):
            y_true.append(1)  # Target
            y_score.append(target_scores[i])
            for j in range(len(decoy_seqs)):
                y_true.append(0)  # Decoy
                y_score.append(decoy_scores_matrix[j, i])

        auc = roc_auc_score(y_true, y_score)

        results['tiers'][tier] = {
            'n_decoys': len(decoy_seqs),
            'mean_score_diff': float(np.mean(score_diffs)),
            'std_score_diff': float(np.std(score_diffs)),
            'target_wins_pct': float(target_wins / len(tcrs) * 100),
            'auc': float(auc),
            'target_scores_mean': float(np.mean(target_scores)),
            'target_scores_std': float(np.std(target_scores)),
            'decoy_scores_mean': float(np.mean(decoy_scores_matrix)),
            'decoy_scores_std': float(np.std(decoy_scores_matrix))
        }

        print(f"      AUC: {auc:.3f}")
        print(f"      Mean score diff: {np.mean(score_diffs):.4f} ± {np.std(score_diffs):.4f}")
        print(f"      Target wins: {target_wins}/{len(tcrs)} ({target_wins/len(tcrs)*100:.1f}%)")

    return results


def plot_discrimination_results(all_results, output_dir):
    """Generate plots for discrimination results."""

    # Prepare data for plotting
    plot_data = []
    for result in all_results:
        scorer = result['scorer']
        target = result['target']
        for tier, metrics in result['tiers'].items():
            plot_data.append({
                'Scorer': scorer,
                'Target': target,
                'Tier': tier,
                'AUC': metrics['auc'],
                'Score Diff': metrics['mean_score_diff'],
                'Target Wins %': metrics['target_wins_pct']
            })

    df = pd.DataFrame(plot_data)

    # Plot 1: AUC by scorer and tier
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # AUC
    ax = axes[0]
    for scorer in df['Scorer'].unique():
        scorer_df = df[df['Scorer'] == scorer]
        for tier in ['A', 'B', 'D']:
            tier_df = scorer_df[scorer_df['Tier'] == tier]
            if not tier_df.empty:
                ax.scatter([tier] * len(tier_df), tier_df['AUC'],
                          label=f'{scorer} (mean={tier_df["AUC"].mean():.3f})',
                          alpha=0.6, s=100)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Decoy Tier')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Target vs Decoy Discrimination (AUC)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Score difference
    ax = axes[1]
    for scorer in df['Scorer'].unique():
        scorer_df = df[df['Scorer'] == scorer]
        for tier in ['A', 'B', 'D']:
            tier_df = scorer_df[scorer_df['Tier'] == tier]
            if not tier_df.empty:
                ax.scatter([tier] * len(tier_df), tier_df['Score Diff'],
                          label=f'{scorer}', alpha=0.6, s=100)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No separation')
    ax.set_xlabel('Decoy Tier')
    ax.set_ylabel('Mean Score(target) - Score(decoy)')
    ax.set_title('Score Separation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Target wins percentage
    ax = axes[2]
    for scorer in df['Scorer'].unique():
        scorer_df = df[df['Scorer'] == scorer]
        for tier in ['A', 'B', 'D']:
            tier_df = scorer_df[scorer_df['Tier'] == tier]
            if not tier_df.empty:
                ax.scatter([tier] * len(tier_df), tier_df['Target Wins %'],
                          label=f'{scorer}', alpha=0.6, s=100)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Decoy Tier')
    ax.set_ylabel('% TCRs where target > all decoys')
    ax.set_title('Target Wins Percentage')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discrimination_summary.png'), dpi=150)
    print(f"\nSaved discrimination summary to {output_dir}/discrimination_summary.png")


def main():
    print("=" * 80)
    print("TCR-Peptide Scorer Decoy Discrimination Test")
    print("=" * 80)

    # Configuration
    output_dir = '/share/liuyutian/tcrppo_v2/results/scorer_decoy_discrimination'
    os.makedirs(output_dir, exist_ok=True)

    # Select test targets (from candidate_targets.json)
    test_targets = [
        'GILGFVFTL',  # Influenza M1
        'GLCTLVAML',  # EBV BMLF1
        'NLVPMVATV',  # CMV pp65
    ]

    # Initialize scorers
    print("\nInitializing scorers...")
    print("  Loading NetTCR...")
    nettcr_scorer = AffinityNetTCRScorer(device='cpu')

    print("  Loading ERGO...")
    ergo_model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
    ergo_scorer = AffinityERGOScorer(model_file=ergo_model_file, device='cuda', mc_samples=1)

    # Run discrimination tests
    all_results = []

    for target in test_targets:
        print("\n" + "=" * 80)
        print(f"Testing target: {target}")
        print("=" * 80)

        # Load decoys (sample 30 per tier for speed)
        print("\nLoading decoys...")
        decoys = load_decoy_library(target, max_decoys_per_tier=30)
        for tier, seqs in decoys.items():
            print(f"  Tier {tier}: {len(seqs)} decoys")

        if not decoys:
            print(f"  No decoys found for {target}, skipping")
            continue

        # Get high-affinity TCRs using NetTCR (use 30 for speed)
        tcrs = get_high_affinity_tcrs(target, nettcr_scorer, n_tcrs=30)

        # Evaluate NetTCR
        nettcr_results = evaluate_discrimination(
            nettcr_scorer, 'NetTCR', target, tcrs, decoys, output_dir
        )
        all_results.append(nettcr_results)

        # Evaluate ERGO
        ergo_results = evaluate_discrimination(
            ergo_scorer, 'ERGO', target, tcrs, decoys, output_dir
        )
        all_results.append(ergo_results)

    # Save results
    results_file = os.path.join(output_dir, 'discrimination_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for scorer_name in ['NetTCR', 'ERGO']:
        print(f"\n{scorer_name}:")
        scorer_results = [r for r in all_results if r['scorer'] == scorer_name]

        for tier in ['A', 'B', 'D']:
            tier_metrics = []
            for result in scorer_results:
                if tier in result['tiers']:
                    tier_metrics.append(result['tiers'][tier])

            if tier_metrics:
                mean_auc = np.mean([m['auc'] for m in tier_metrics])
                mean_score_diff = np.mean([m['mean_score_diff'] for m in tier_metrics])
                mean_target_wins = np.mean([m['target_wins_pct'] for m in tier_metrics])

                print(f"  Tier {tier}:")
                print(f"    Mean AUC: {mean_auc:.3f}")
                print(f"    Mean score diff: {mean_score_diff:.4f}")
                print(f"    Mean target wins: {mean_target_wins:.1f}%")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80)
    plot_discrimination_results(all_results, output_dir)

    # Go/no-go decision
    print("\n" + "=" * 80)
    print("GO/NO-GO DECISION")
    print("=" * 80)

    # Compute overall metrics
    nettcr_aucs = []
    ergo_aucs = []
    nettcr_diffs = []
    ergo_diffs = []

    for result in all_results:
        for tier_metrics in result['tiers'].values():
            if result['scorer'] == 'NetTCR':
                nettcr_aucs.append(tier_metrics['auc'])
                nettcr_diffs.append(tier_metrics['mean_score_diff'])
            else:
                ergo_aucs.append(tier_metrics['auc'])
                ergo_diffs.append(tier_metrics['mean_score_diff'])

    print(f"\nNetTCR:")
    print(f"  Mean AUC: {np.mean(nettcr_aucs):.3f} (threshold: > 0.55)")
    print(f"  Mean score diff: {np.mean(nettcr_diffs):.4f} (threshold: > 0.05)")

    print(f"\nERGO:")
    print(f"  Mean AUC: {np.mean(ergo_aucs):.3f} (threshold: > 0.55)")
    print(f"  Mean score diff: {np.mean(ergo_diffs):.4f} (threshold: > 0.05)")

    # Decision logic
    nettcr_pass = np.mean(nettcr_aucs) > 0.55 and np.mean(nettcr_diffs) > 0.05
    ergo_pass = np.mean(ergo_aucs) > 0.55 and np.mean(ergo_diffs) > 0.05

    print("\n" + "-" * 80)
    if nettcr_pass or ergo_pass:
        print("DECISION: GO")
        print("\nAt least one scorer can discriminate target from decoys.")
        if nettcr_pass and ergo_pass:
            print("Recommendation: Use ensemble (NetTCR + ERGO average)")
        elif nettcr_pass:
            print("Recommendation: Use NetTCR as primary scorer")
        else:
            print("Recommendation: Use ERGO as primary scorer")
        print("\nProceed to pilot RL training with multi-component reward.")
    else:
        print("DECISION: NO-GO")
        print("\nNeither scorer can reliably discriminate target from decoys.")
        print("Recommendation: Obtain TITAN or better scorer before RL training.")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("Discrimination Test Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
