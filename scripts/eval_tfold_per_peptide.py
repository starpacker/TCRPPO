"""Evaluate tFold V3.4 per-peptide AUC on tc-hard dataset.

Loads pre-extracted tFold features from HDF5 and runs the V3.4 classifier
to compute per-peptide AUC. Merges results with existing NetTCR/ERGO metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')
sys.path.insert(0, '/share/liuyutian/tfold')

import importlib
_mod = importlib.import_module("TCR_PMHC_pred.4_16.model")
ClassifierV34 = _mod.ClassifierV34


def load_model(checkpoint_path, device='cuda'):
    """Load V3.4 classifier from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('model_config', {})

    model = ClassifierV34(
        d_sfea=config.get('d_sfea', 192),
        n_heads=config.get('n_heads', 4),
        n_rbf=config.get('n_rbf', 16),
        pfea_dim=config.get('pfea_dim', 128),
        n_attn_layers=config.get('n_attn_layers', 2),
        mlp_hidden=config.get('mlp_hidden', 256),
        dropout=config.get('dropout', 0.1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded: {model._n_params:,} params")
    print(f"  Trained epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val PerEpiAUC: {ckpt.get('val_per_epitope_mean_auc', '?')}")

    return model


def predict_batch(model, batch, device='cuda'):
    """Run model inference on a batch of features."""
    # Move batch to device
    batch_gpu = {}
    for key in ['sfea_cdr3b', 'sfea_cdr3a', 'sfea_pep',
                'ca_cdr3b', 'ca_cdr3a', 'ca_pep',
                'pfea_cdr3b_pep', 'pfea_cdr3a_pep', 'v33_feat']:
        batch_gpu[key] = batch[key].to(device)
    for key in ['len_cdr3b', 'len_cdr3a', 'len_pep']:
        batch_gpu[key] = batch[key].to(device)

    with torch.no_grad():
        gate_logits = model(batch_gpu).cpu().numpy()

    # Gate logit: higher = more likely non-binding
    # Binding score: negate sigmoid so higher = more likely binding
    binding_scores = -1.0 / (1.0 + np.exp(-gate_logits))

    return binding_scores


def main():
    print("=" * 80)
    print("tFold V3.4 Per-Peptide AUC Evaluation")
    print("=" * 80)

    h5_path = '/share/liuyutian/tfold/outputs/sfea_pfea_cord_vregion/tchard_v34_features.h5'
    model_path = '/share/liuyutian/tfold/TCR_PMHC_pred/4_16/weights/best_v34.pth'
    output_dir = '/share/liuyutian/tcrppo_v2/results/scorer_per_peptide_tchard'
    device = 'cuda'

    # Load existing NetTCR/ERGO results
    print("\nLoading existing NetTCR/ERGO results...")
    existing_df = pd.read_csv(os.path.join(output_dir, 'per_peptide_metrics.csv'))
    nettcr_peptides = set(existing_df['peptide'].tolist())
    print(f"  Found {len(existing_df)} peptides with NetTCR/ERGO scores")

    # Load tFold features
    print("\nLoading tFold features from HDF5...")
    with h5py.File(h5_path, 'r') as f:
        epitopes = [x.decode() for x in f['epitope'][:]]
        labels = f['label'][:]
        print(f"  Total samples: {len(epitopes):,}")
        print(f"  Unique epitopes: {len(set(epitopes))}")

    # Group by epitope and filter for sufficient samples
    print("\nGrouping by epitope...")
    epi_indices = {}
    for i, (epi, lab) in enumerate(zip(epitopes, labels)):
        if epi not in epi_indices:
            epi_indices[epi] = {'pos': [], 'neg': []}
        if lab == 1:
            epi_indices[epi]['pos'].append(i)
        else:
            epi_indices[epi]['neg'].append(i)

    # Filter peptides with >=20 pos and >=20 neg that are also in NetTCR results
    valid_peptides = []
    for epi, idx in epi_indices.items():
        if len(idx['pos']) >= 20 and len(idx['neg']) >= 20 and epi in nettcr_peptides:
            valid_peptides.append(epi)

    print(f"  Valid peptides (>=20 pos, >=20 neg, in NetTCR): {len(valid_peptides)}")

    # Load model
    print("\nLoading tFold V3.4 model...")
    model = load_model(model_path, device=device)

    # Score each peptide
    print("\nScoring peptides...")
    results = []
    batch_size = 256

    for pep in tqdm(valid_peptides, desc="Peptides"):
        pos_idx = epi_indices[pep]['pos']
        neg_idx = epi_indices[pep]['neg']
        all_idx = pos_idx + neg_idx
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        # Load features for this peptide
        with h5py.File(h5_path, 'r') as f:
            batch = {
                'sfea_cdr3b': torch.from_numpy(f['sfea_cdr3b'][all_idx]).float(),
                'sfea_cdr3a': torch.from_numpy(f['sfea_cdr3a'][all_idx]).float(),
                'sfea_pep': torch.from_numpy(f['sfea_pep'][all_idx]).float(),
                'ca_cdr3b': torch.from_numpy(f['ca_cdr3b'][all_idx]).float(),
                'ca_cdr3a': torch.from_numpy(f['ca_cdr3a'][all_idx]).float(),
                'ca_pep': torch.from_numpy(f['ca_pep'][all_idx]).float(),
                'pfea_cdr3b_pep': torch.from_numpy(f['pfea_cdr3b_pep'][all_idx]).float(),
                'pfea_cdr3a_pep': torch.from_numpy(f['pfea_cdr3a_pep'][all_idx]).float(),
                'v33_feat': torch.from_numpy(f['v33_feat'][all_idx]).float(),
                'len_cdr3b': torch.from_numpy(f['len_cdr3b'][all_idx]).long(),
                'len_cdr3a': torch.from_numpy(f['len_cdr3a'][all_idx]).long(),
                'len_pep': torch.from_numpy(f['len_pep'][all_idx]).long(),
            }
            y = torch.from_numpy(labels[all_idx]).numpy()

        # Predict in batches
        scores = []
        for i in range(0, len(all_idx), batch_size):
            batch_slice = {k: v[i:i+batch_size] for k, v in batch.items()}
            batch_scores = predict_batch(model, batch_slice, device=device)
            scores.extend(batch_scores)

        scores = np.array(scores)

        # Compute metrics
        try:
            auc = roc_auc_score(y, scores)
        except:
            auc = np.nan
        try:
            ap = average_precision_score(y, scores)
        except:
            ap = np.nan

        pos_mean = scores[:n_pos].mean()
        neg_mean = scores[n_pos:].mean()
        separation = pos_mean - neg_mean

        results.append({
            'peptide': pep,
            'tfold_auc': auc,
            'tfold_ap': ap,
            'tfold_pos_mean': pos_mean,
            'tfold_neg_mean': neg_mean,
            'tfold_separation': separation,
            'tfold_n_samples': len(all_idx),
        })

    # Merge with existing results
    print("\nMerging with existing results...")
    tfold_df = pd.DataFrame(results)
    merged_df = existing_df.merge(tfold_df, on='peptide', how='left')

    # Save updated results
    output_path = os.path.join(output_dir, 'per_peptide_metrics.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"Saved updated results to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("tFold Summary Statistics")
    print("=" * 80)
    print(f"\nPeptides evaluated: {len(tfold_df)}")
    print(f"\ntFold V3.4:")
    print(f"  Mean AUC: {tfold_df['tfold_auc'].mean():.3f}")
    print(f"  Median AUC: {tfold_df['tfold_auc'].median():.3f}")
    print(f"  Std AUC: {tfold_df['tfold_auc'].std():.3f}")
    print(f"  Mean separation: {tfold_df['tfold_separation'].mean():.4f}")

    # Count by AUC threshold
    print(f"\nPeptides by AUC threshold:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        n = (tfold_df['tfold_auc'] > threshold).sum()
        pct = n / len(tfold_df) * 100
        print(f"  AUC > {threshold}: {n} ({pct:.1f}%)")

    # Compare with NetTCR/ERGO on overlapping peptides
    print("\n" + "=" * 80)
    print("Comparison on 37 overlapping peptides")
    print("=" * 80)
    overlap_df = merged_df[merged_df['tfold_auc'].notna()].copy()

    print(f"\nMean AUC:")
    print(f"  NetTCR: {overlap_df['nettcr_auc'].mean():.3f}")
    print(f"  ERGO:   {overlap_df['ergo_auc'].mean():.3f}")
    print(f"  tFold:  {overlap_df['tfold_auc'].mean():.3f}")

    print(f"\nMedian AUC:")
    print(f"  NetTCR: {overlap_df['nettcr_auc'].median():.3f}")
    print(f"  ERGO:   {overlap_df['ergo_auc'].median():.3f}")
    print(f"  tFold:  {overlap_df['tfold_auc'].median():.3f}")

    # Head-to-head comparison
    tfold_vs_nettcr = (overlap_df['tfold_auc'] > overlap_df['nettcr_auc']).sum()
    tfold_vs_ergo = (overlap_df['tfold_auc'] > overlap_df['ergo_auc']).sum()
    print(f"\nHead-to-head wins:")
    print(f"  tFold > NetTCR: {tfold_vs_nettcr}/{len(overlap_df)} ({tfold_vs_nettcr/len(overlap_df)*100:.1f}%)")
    print(f"  tFold > ERGO:   {tfold_vs_ergo}/{len(overlap_df)} ({tfold_vs_ergo/len(overlap_df)*100:.1f}%)")

    # Top peptides by tFold
    print("\n" + "=" * 80)
    print("Top 10 peptides by tFold AUC")
    print("=" * 80)
    top10 = overlap_df.nlargest(10, 'tfold_auc')
    print(f"{'Peptide':<20} {'tFold AUC':>10} {'NetTCR AUC':>11} {'ERGO AUC':>10}")
    print("-" * 80)
    for _, r in top10.iterrows():
        print(f"{r['peptide']:<20} {r['tfold_auc']:>10.3f} {r['nettcr_auc']:>11.3f} {r['ergo_auc']:>10.3f}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
