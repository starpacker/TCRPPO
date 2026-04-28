"""Test script to evaluate scorer consistency on tc-hard data.

This script tests NetTCR-2.0, ERGO, and DeepAIR scorers on tc-hard dataset
and analyzes their score consistency and correlation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.scorers.affinity_deepair import AffinityDeepAIRScorer
from tcrppo_v2.utils.constants import ERGO_MODEL_DIR


def load_tc_hard_data(data_dir: str, max_peptides: int = 10) -> Dict[str, List[Tuple[str, str]]]:
    """Load tc-hard test data.

    Returns:
        Dict mapping peptide -> list of (tcr, peptide) pairs
    """
    tc_hard_data = {}

    # List all peptide files
    peptide_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    # Limit number of peptides for faster testing
    peptide_files = peptide_files[:max_peptides]

    for pep_file in peptide_files:
        peptide = pep_file.replace('.txt', '')
        file_path = os.path.join(data_dir, pep_file)

        with open(file_path, 'r') as f:
            tcrs = [line.strip() for line in f if line.strip()]

        # Create TCR-peptide pairs
        pairs = [(tcr, peptide) for tcr in tcrs]
        tc_hard_data[peptide] = pairs

    return tc_hard_data


def evaluate_scorer(scorer, data: Dict[str, List[Tuple[str, str]]], scorer_name: str) -> pd.DataFrame:
    """Evaluate a scorer on tc-hard data.

    Returns:
        DataFrame with columns: peptide, tcr, score
    """
    print(f"\nEvaluating {scorer_name}...")

    results = []

    for peptide, pairs in data.items():
        tcrs = [tcr for tcr, _ in pairs]
        peptides = [pep for _, pep in pairs]

        # Score batch
        scores, _ = scorer.score_batch(tcrs, peptides)

        for tcr, score in zip(tcrs, scores):
            results.append({
                'peptide': peptide,
                'tcr': tcr,
                'score': score,
                'scorer': scorer_name
            })

        print(f"  {peptide}: {len(tcrs)} TCRs, mean score: {np.mean(scores):.4f}")

    return pd.DataFrame(results)


def compute_correlations(df1: pd.DataFrame, df2: pd.DataFrame,
                         name1: str, name2: str) -> Tuple[float, float]:
    """Compute Pearson and Spearman correlations between two scorers."""
    # Merge on peptide and tcr
    merged = df1.merge(df2, on=['peptide', 'tcr'], suffixes=('_1', '_2'))

    scores1 = merged['score_1'].values
    scores2 = merged['score_2'].values

    pearson_r, pearson_p = pearsonr(scores1, scores2)
    spearman_r, spearman_p = spearmanr(scores1, scores2)

    print(f"\n{name1} vs {name2}:")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")

    return pearson_r, spearman_r


def plot_score_distributions(results_df: pd.DataFrame, output_dir: str):
    """Plot score distributions for each scorer."""
    plt.figure(figsize=(12, 4))

    scorers = results_df['scorer'].unique()

    for i, scorer in enumerate(scorers, 1):
        plt.subplot(1, len(scorers), i)
        scores = results_df[results_df['scorer'] == scorer]['score']
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(f'{scorer}\n(mean={scores.mean():.3f}, std={scores.std():.3f})')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
    print(f"\nSaved score distributions to {output_dir}/score_distributions.png")


def plot_correlation_matrix(nettcr_df: pd.DataFrame, ergo_df: pd.DataFrame,
                            deepair_df: pd.DataFrame, output_dir: str):
    """Plot correlation heatmap between scorers."""
    # Merge all dataframes
    merged = nettcr_df.merge(ergo_df, on=['peptide', 'tcr'], suffixes=('_nettcr', '_ergo'))
    merged = merged.merge(deepair_df, on=['peptide', 'tcr'])

    # Create correlation matrix
    score_matrix = merged[['score_nettcr', 'score_ergo', 'score']].values
    corr_matrix = np.corrcoef(score_matrix.T)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=['NetTCR', 'ERGO', 'DeepAIR'],
                yticklabels=['NetTCR', 'ERGO', 'DeepAIR'],
                vmin=-1, vmax=1, center=0)
    plt.title('Scorer Correlation Matrix (Pearson)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
    print(f"Saved correlation matrix to {output_dir}/correlation_matrix.png")


def plot_pairwise_scatter(nettcr_df: pd.DataFrame, ergo_df: pd.DataFrame,
                          deepair_df: pd.DataFrame, output_dir: str):
    """Plot pairwise scatter plots between scorers."""
    # Merge all dataframes
    merged = nettcr_df.merge(ergo_df, on=['peptide', 'tcr'], suffixes=('_nettcr', '_ergo'))
    merged = merged.merge(deepair_df, on=['peptide', 'tcr'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # NetTCR vs ERGO
    axes[0].scatter(merged['score_nettcr'], merged['score_ergo'], alpha=0.3, s=10)
    axes[0].set_xlabel('NetTCR Score')
    axes[0].set_ylabel('ERGO Score')
    axes[0].set_title('NetTCR vs ERGO')
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0].grid(alpha=0.3)

    # NetTCR vs DeepAIR
    axes[1].scatter(merged['score_nettcr'], merged['score'], alpha=0.3, s=10)
    axes[1].set_xlabel('NetTCR Score')
    axes[1].set_ylabel('DeepAIR Score')
    axes[1].set_title('NetTCR vs DeepAIR')
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1].grid(alpha=0.3)

    # ERGO vs DeepAIR
    axes[2].scatter(merged['score_ergo'], merged['score'], alpha=0.3, s=10)
    axes[2].set_xlabel('ERGO Score')
    axes[2].set_ylabel('DeepAIR Score')
    axes[2].set_title('ERGO vs DeepAIR')
    axes[2].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_scatter.png'), dpi=150)
    print(f"Saved pairwise scatter plots to {output_dir}/pairwise_scatter.png")


def main():
    print("=" * 80)
    print("TC-Hard Scorer Consistency Evaluation")
    print("=" * 80)

    # Configuration
    data_dir = '/share/liuyutian/tcrppo_v2/data/l0_seeds_tchard'
    output_dir = '/share/liuyutian/tcrppo_v2/results/scorer_consistency'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading tc-hard data...")
    tc_hard_data = load_tc_hard_data(data_dir, max_peptides=10)
    total_pairs = sum(len(pairs) for pairs in tc_hard_data.values())
    print(f"Loaded {len(tc_hard_data)} peptides, {total_pairs} TCR-peptide pairs")

    # Initialize scorers
    print("\nInitializing scorers...")

    print("  Loading NetTCR-2.0...")
    nettcr_scorer = AffinityNetTCRScorer(device='cpu')

    print("  Loading ERGO...")
    ergo_model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
    ergo_scorer = AffinityERGOScorer(model_file=ergo_model_file, device='cuda', mc_samples=1)

    print("  Loading DeepAIR...")
    deepair_scorer = AffinityDeepAIRScorer(device='cuda')

    # Evaluate each scorer
    nettcr_results = evaluate_scorer(nettcr_scorer, tc_hard_data, 'NetTCR')
    ergo_results = evaluate_scorer(ergo_scorer, tc_hard_data, 'ERGO')
    deepair_results = evaluate_scorer(deepair_scorer, tc_hard_data, 'DeepAIR')

    # Combine results
    all_results = pd.concat([nettcr_results, ergo_results, deepair_results], ignore_index=True)

    # Save raw results
    all_results.to_csv(os.path.join(output_dir, 'scorer_results.csv'), index=False)
    print(f"\nSaved raw results to {output_dir}/scorer_results.csv")

    # Compute correlations
    print("\n" + "=" * 80)
    print("Correlation Analysis")
    print("=" * 80)

    pearson_ne, spearman_ne = compute_correlations(nettcr_results, ergo_results, 'NetTCR', 'ERGO')
    pearson_nd, spearman_nd = compute_correlations(nettcr_results, deepair_results, 'NetTCR', 'DeepAIR')
    pearson_ed, spearman_ed = compute_correlations(ergo_results, deepair_results, 'ERGO', 'DeepAIR')

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for scorer in ['NetTCR', 'ERGO', 'DeepAIR']:
        scores = all_results[all_results['scorer'] == scorer]['score']
        print(f"\n{scorer}:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
        print(f"  Median: {scores.median():.4f}")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80)

    plot_score_distributions(all_results, output_dir)
    plot_correlation_matrix(nettcr_results, ergo_results, deepair_results, output_dir)
    plot_pairwise_scatter(nettcr_results, ergo_results, deepair_results, output_dir)

    # Save summary report
    summary = {
        'n_peptides': len(tc_hard_data),
        'n_pairs': total_pairs,
        'correlations': {
            'NetTCR_vs_ERGO': {'pearson': float(pearson_ne), 'spearman': float(spearman_ne)},
            'NetTCR_vs_DeepAIR': {'pearson': float(pearson_nd), 'spearman': float(spearman_nd)},
            'ERGO_vs_DeepAIR': {'pearson': float(pearson_ed), 'spearman': float(spearman_ed)},
        },
        'statistics': {
            scorer: {
                'mean': float(all_results[all_results['scorer'] == scorer]['score'].mean()),
                'std': float(all_results[all_results['scorer'] == scorer]['score'].std()),
                'min': float(all_results[all_results['scorer'] == scorer]['score'].min()),
                'max': float(all_results[all_results['scorer'] == scorer]['score'].max()),
            }
            for scorer in ['NetTCR', 'ERGO', 'DeepAIR']
        }
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {output_dir}/summary.json")
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
