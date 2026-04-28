"""Evaluate NetTCR and ERGO per-peptide AUC on tc-hard dataset.

For each peptide with sufficient labeled data, compute AUC-ROC to understand
per-peptide prediction accuracy.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_nettcr import AffinityNetTCRScorer
from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer
from tcrppo_v2.utils.constants import ERGO_MODEL_DIR


def main():
    print("=" * 80)
    print("Per-Peptide AUC Evaluation on tc-hard")
    print("=" * 80)

    output_dir = '/share/liuyutian/tcrppo_v2/results/scorer_per_peptide_tchard'
    os.makedirs(output_dir, exist_ok=True)

    # Load tc-hard
    print("\nLoading tc-hard dataset...")
    df = pd.read_csv('/share/liuyutian/TCRdata/tc-hard/ds.csv', low_memory=False)
    df = df.dropna(subset=['cdr3.beta', 'antigen.epitope', 'label'])
    df['label'] = df['label'].astype(int)
    print(f"Total samples: {len(df):,}")

    # Filter peptides with >= 20 pos and >= 20 neg
    print("\nFiltering peptides with sufficient samples...")
    pep_stats = df.groupby('antigen.epitope').agg(
        total=('label', 'count'),
        pos=('label', 'sum')
    )
    pep_stats['neg'] = pep_stats['total'] - pep_stats['pos']
    valid_peptides = pep_stats[(pep_stats['pos'] >= 20) & (pep_stats['neg'] >= 20)].index.tolist()
    print(f"Valid peptides: {len(valid_peptides)}")

    # Sample up to 500 per peptide for speed (stratified)
    print("\nSampling data...")
    sampled_dfs = []
    for pep in tqdm(valid_peptides, desc="Sampling"):
        pep_df = df[df['antigen.epitope'] == pep]
        if len(pep_df) > 500:
            pos = pep_df[pep_df['label'] == 1]
            neg = pep_df[pep_df['label'] == 0]
            n_pos = min(len(pos), 250)
            n_neg = min(len(neg), 250)
            pep_df = pd.concat([
                pos.sample(n=n_pos, random_state=42),
                neg.sample(n=n_neg, random_state=42)
            ])
        sampled_dfs.append(pep_df)

    data = pd.concat(sampled_dfs, ignore_index=True)

    # Filter out non-standard amino acids
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    def is_valid(seq):
        return all(c in standard_aa for c in str(seq))
    mask = data['cdr3.beta'].apply(is_valid) & data['antigen.epitope'].apply(is_valid)
    n_before = len(data)
    data = data[mask].reset_index(drop=True)
    print(f"Filtered non-standard AAs: {n_before} -> {len(data)} ({n_before - len(data)} removed)")

    print(f"Sampled: {len(data):,} samples, {data['antigen.epitope'].nunique()} peptides")
    print(f"  Positive: {(data['label']==1).sum():,}")
    print(f"  Negative: {(data['label']==0).sum():,}")

    # Initialize scorers
    print("\nInitializing scorers...")
    nettcr = AffinityNetTCRScorer(device='cpu')
    ergo = AffinityERGOScorer(
        model_file=os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt"),
        device='cuda', mc_samples=1
    )

    # Score all pairs
    print("\nScoring with NetTCR...")
    cdr3b = data['cdr3.beta'].tolist()
    peptides = data['antigen.epitope'].tolist()

    nettcr_scores = []
    batch_size = 512
    for i in tqdm(range(0, len(cdr3b), batch_size)):
        batch_tcrs = cdr3b[i:i+batch_size]
        batch_peps = peptides[i:i+batch_size]
        scores, _ = nettcr.score_batch(batch_tcrs, batch_peps)
        nettcr_scores.extend(scores)

    print("Scoring with ERGO...")
    ergo_scores = []
    for i in tqdm(range(0, len(cdr3b), batch_size)):
        batch_tcrs = cdr3b[i:i+batch_size]
        batch_peps = peptides[i:i+batch_size]
        scores, _ = ergo.score_batch(batch_tcrs, batch_peps)
        ergo_scores.extend(scores)

    data['nettcr_score'] = nettcr_scores
    data['ergo_score'] = ergo_scores

    # Per-peptide metrics
    print("\n" + "=" * 80)
    print("Computing per-peptide metrics...")
    print("=" * 80)

    results = []
    for pep in tqdm(valid_peptides):
        pep_data = data[data['antigen.epitope'] == pep]
        if len(pep_data) < 20:
            continue

        y = pep_data['label'].values
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        row = {
            'peptide': pep,
            'n_total': len(pep_data),
            'n_pos': n_pos,
            'n_neg': n_neg
        }

        for scorer_name, col in [('nettcr', 'nettcr_score'), ('ergo', 'ergo_score')]:
            scores = pep_data[col].values

            # Skip if all same label
            if n_pos == 0 or n_neg == 0:
                row[f'{scorer_name}_auc'] = np.nan
                row[f'{scorer_name}_ap'] = np.nan
                row[f'{scorer_name}_pos_mean'] = np.nan
                row[f'{scorer_name}_neg_mean'] = np.nan
                row[f'{scorer_name}_separation'] = np.nan
                continue

            try:
                auc = roc_auc_score(y, scores)
            except:
                auc = np.nan
            try:
                ap = average_precision_score(y, scores)
            except:
                ap = np.nan

            pos_mean = scores[y == 1].mean()
            neg_mean = scores[y == 0].mean()

            row[f'{scorer_name}_auc'] = auc
            row[f'{scorer_name}_ap'] = ap
            row[f'{scorer_name}_pos_mean'] = pos_mean
            row[f'{scorer_name}_neg_mean'] = neg_mean
            row[f'{scorer_name}_separation'] = pos_mean - neg_mean

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('nettcr_auc', ascending=False)

    # Save full results
    results_df.to_csv(os.path.join(output_dir, 'per_peptide_metrics.csv'), index=False)
    print(f"\nSaved full results to {output_dir}/per_peptide_metrics.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"\nTotal peptides evaluated: {len(results_df)}")
    print(f"\nNetTCR:")
    print(f"  Mean AUC: {results_df['nettcr_auc'].mean():.3f}")
    print(f"  Median AUC: {results_df['nettcr_auc'].median():.3f}")
    print(f"  Mean separation: {results_df['nettcr_separation'].mean():.4f}")
    print(f"\nERGO:")
    print(f"  Mean AUC: {results_df['ergo_auc'].mean():.3f}")
    print(f"  Median AUC: {results_df['ergo_auc'].median():.3f}")
    print(f"  Mean separation: {results_df['ergo_separation'].mean():.4f}")

    # Count by AUC threshold
    print(f"\nPeptides by AUC threshold:")
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        n_nettcr = (results_df['nettcr_auc'] > threshold).sum()
        n_ergo = (results_df['ergo_auc'] > threshold).sum()
        pct_nettcr = n_nettcr / len(results_df) * 100
        pct_ergo = n_ergo / len(results_df) * 100
        print(f"  AUC > {threshold}: NetTCR={n_nettcr} ({pct_nettcr:.1f}%), ERGO={n_ergo} ({pct_ergo:.1f}%)")

    # Print top/bottom peptides
    print(f"\n{'='*80}")
    print("Top 20 peptides by NetTCR AUC:")
    print(f"{'='*80}")
    print(f"{'Peptide':<20} {'N':>6} {'Pos':>5} {'Neg':>5} {'NetTCR AUC':>11} {'ERGO AUC':>10} {'NetTCR Sep':>11}")
    print("-" * 80)
    for _, r in results_df.head(20).iterrows():
        print(f"{r['peptide']:<20} {r['n_total']:>6} {r['n_pos']:>5} {r['n_neg']:>5} "
              f"{r['nettcr_auc']:>11.3f} {r['ergo_auc']:>10.3f} {r['nettcr_separation']:>11.4f}")

    print(f"\n{'='*80}")
    print("Bottom 20 peptides by NetTCR AUC:")
    print(f"{'='*80}")
    print(f"{'Peptide':<20} {'N':>6} {'Pos':>5} {'Neg':>5} {'NetTCR AUC':>11} {'ERGO AUC':>10} {'NetTCR Sep':>11}")
    print("-" * 80)
    for _, r in results_df.tail(20).iterrows():
        print(f"{r['peptide']:<20} {r['n_total']:>6} {r['n_pos']:>5} {r['n_neg']:>5} "
              f"{r['nettcr_auc']:>11.3f} {r['ergo_auc']:>10.3f} {r['nettcr_separation']:>11.4f}")

    # Generate plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: AUC histogram
    ax = axes[0, 0]
    ax.hist(results_df['nettcr_auc'].dropna(), bins=30, alpha=0.6, label='NetTCR', color='blue', density=True)
    ax.hist(results_df['ergo_auc'].dropna(), bins=30, alpha=0.6, label='ERGO', color='orange', density=True)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('AUC-ROC')
    ax.set_ylabel('Density')
    ax.set_title(f'Per-Peptide AUC Distribution (n={len(results_df)} peptides)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: NetTCR vs ERGO scatter
    ax = axes[0, 1]
    ax.scatter(results_df['nettcr_auc'], results_df['ergo_auc'], alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('NetTCR AUC')
    ax.set_ylabel('ERGO AUC')
    ax.set_title('NetTCR vs ERGO per-peptide AUC')
    ax.grid(alpha=0.3)

    # Add correlation
    corr = results_df[['nettcr_auc', 'ergo_auc']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Score separation
    ax = axes[1, 0]
    ax.hist(results_df['nettcr_separation'].dropna(), bins=30, alpha=0.6, label='NetTCR', color='blue', density=True)
    ax.hist(results_df['ergo_separation'].dropna(), bins=30, alpha=0.6, label='ERGO', color='orange', density=True)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Score Separation (Pos mean - Neg mean)')
    ax.set_ylabel('Density')
    ax.set_title('Per-Peptide Score Separation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: AUC vs sample size
    ax = axes[1, 1]
    ax.scatter(results_df['n_total'], results_df['nettcr_auc'], alpha=0.5, s=20, label='NetTCR', color='blue')
    ax.scatter(results_df['n_total'], results_df['ergo_auc'], alpha=0.5, s=20, label='ERGO', color='orange')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Sample size')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC vs Sample Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_peptide_summary.png'), dpi=150)
    print(f"Saved plot to {output_dir}/per_peptide_summary.png")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
