"""Evaluate TCR-peptide scorers on labeled ground-truth data.

This script tests NetTCR and ERGO on the NetTCR test dataset which has
experimental binder/non-binder labels. Computes AUC, accuracy, and other
classification metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.utils.constants import ERGO_MODEL_DIR


def evaluate_scorer(scorer, data_df, scorer_name, batch_size=128):
    """Evaluate a scorer on labeled data.

    Args:
        scorer: Scorer instance
        data_df: DataFrame with columns [CDR3b, peptide, binder]
        scorer_name: Name for logging
        batch_size: Batch size for scoring

    Returns:
        DataFrame with predictions
    """
    print(f"\nEvaluating {scorer_name}...")
    print(f"  Total samples: {len(data_df)}")

    scores = []

    # Score in batches
    for i in tqdm(range(0, len(data_df), batch_size), desc=f"  Scoring"):
        batch = data_df.iloc[i:i+batch_size]
        tcrs = batch['CDR3b'].tolist()
        peptides = batch['peptide'].tolist()

        batch_scores, _ = scorer.score_batch(tcrs, peptides)
        scores.extend(batch_scores)

    # Add scores to dataframe
    result_df = data_df.copy()
    result_df['score'] = scores
    result_df['scorer'] = scorer_name

    return result_df


def compute_metrics(df, scorer_name):
    """Compute classification metrics."""
    y_true = df['binder'].values
    y_score = df['score'].values

    # AUC metrics
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Predictions at optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'scorer': scorer_name,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return metrics


def plot_roc_curves(results_dict, output_dir):
    """Plot ROC curves for all scorers."""
    plt.figure(figsize=(8, 6))

    for scorer_name, df in results_dict.items():
        y_true = df['binder'].values
        y_score = df['score'].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f'{scorer_name} (AUC={auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Binder Classification')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    print(f"Saved ROC curves to {output_dir}/roc_curves.png")


def plot_pr_curves(results_dict, output_dir):
    """Plot Precision-Recall curves."""
    plt.figure(figsize=(8, 6))

    for scorer_name, df in results_dict.items():
        y_true = df['binder'].values
        y_score = df['score'].values
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)
        plt.plot(recall, precision, label=f'{scorer_name} (AP={auc_pr:.3f})', linewidth=2)

    baseline = (df['binder'] == 1).mean()
    plt.axhline(baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150)
    print(f"Saved PR curves to {output_dir}/pr_curves.png")


def plot_score_distributions(results_dict, output_dir):
    """Plot score distributions for binders vs non-binders."""
    n_scorers = len(results_dict)
    fig, axes = plt.subplots(1, n_scorers, figsize=(6*n_scorers, 4))
    if n_scorers == 1:
        axes = [axes]

    for idx, (scorer_name, df) in enumerate(results_dict.items()):
        ax = axes[idx]
        binder_scores = df[df['binder'] == 1]['score']
        nonbinder_scores = df[df['binder'] == 0]['score']

        ax.hist(nonbinder_scores, bins=50, alpha=0.5, label='Non-binders', color='red', density=True)
        ax.hist(binder_scores, bins=50, alpha=0.5, label='Binders', color='green', density=True)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{scorer_name}\nBinder mean={binder_scores.mean():.3f}, Non-binder mean={nonbinder_scores.mean():.3f}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
    print(f"Saved score distributions to {output_dir}/score_distributions.png")


def main():
    print("=" * 80)
    print("TCR-Peptide Scorer Evaluation on Labeled Data")
    print("=" * 80)

    # Configuration
    data_file = '/share/liuyutian/tcrppo_v2/data/nettcr_test.csv'
    output_dir = '/share/liuyutian/tcrppo_v2/results/scorer_labeled_eval'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data_df = pd.read_csv(data_file)
    print(f"Loaded {len(data_df)} samples")
    print(f"  Binders: {(data_df['binder']==1).sum()}")
    print(f"  Non-binders: {(data_df['binder']==0).sum()}")
    print(f"  Unique peptides: {data_df['peptide'].nunique()}")

    # Sample for faster testing - stratified by peptide and binder
    print("\nSampling 5000 samples (stratified by peptide and binder)...")
    sampled_dfs = []
    for peptide in data_df['peptide'].unique():
        pep_df = data_df[data_df['peptide'] == peptide]
        n_samples = min(len(pep_df), max(50, int(5000 * len(pep_df) / len(data_df))))
        sampled_dfs.append(pep_df.sample(n=n_samples, random_state=42))
    data_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Sampled {len(data_df)} samples")
    print(f"  Binders: {(data_df['binder']==1).sum()}")
    print(f"  Non-binders: {(data_df['binder']==0).sum()}")

    # Initialize scorers
    print("\nInitializing scorers...")

    print("  Loading NetTCR...")
    nettcr_scorer = AffinityNetTCRScorer(device='cpu')

    print("  Loading ERGO...")
    ergo_model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
    ergo_scorer = AffinityERGOScorer(model_file=ergo_model_file, device='cuda', mc_samples=1)

    # Evaluate scorers
    nettcr_results = evaluate_scorer(nettcr_scorer, data_df, 'NetTCR', batch_size=256)
    ergo_results = evaluate_scorer(ergo_scorer, data_df, 'ERGO', batch_size=256)

    # Compute metrics
    print("\n" + "=" * 80)
    print("Classification Metrics")
    print("=" * 80)

    nettcr_metrics = compute_metrics(nettcr_results, 'NetTCR')
    ergo_metrics = compute_metrics(ergo_results, 'ERGO')

    metrics_df = pd.DataFrame([nettcr_metrics, ergo_metrics])

    print("\n" + metrics_df.to_string(index=False))

    # Save results
    all_results = pd.concat([nettcr_results, ergo_results], ignore_index=True)
    all_results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\nSaved predictions to {output_dir}/predictions.csv")

    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"Saved metrics to {output_dir}/metrics.csv")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80)

    results_dict = {'NetTCR': nettcr_results, 'ERGO': ergo_results}
    plot_roc_curves(results_dict, output_dir)
    plot_pr_curves(results_dict, output_dir)
    plot_score_distributions(results_dict, output_dir)

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

