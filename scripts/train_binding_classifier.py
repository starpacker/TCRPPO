#!/usr/bin/env python
"""Train the fast CDR3β×Peptide binding classifier on tc-hard data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_binding_classifier.py \
        --data /share/liuyutian/TCRdata/tc-hard/reconstructed/ds_with_full_seq_v2.csv \
        --out-dir runs/binding_classifier \
        --epochs 20 \
        --batch-size 512 \
        --lr 3e-4

Trains on 566K (CDR3β, peptide, label) samples with hard negatives.
Evaluates per-epitope AUC on held-out peptides (no epitope leakage).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tcrppo_v2.scorers.tcr_binding_model import (
    AA_VOCAB, build_model, encode_sequence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_binding")


# ============================================================================
# Dataset
# ============================================================================

class TCRBindingDataset(Dataset):
    """CDR3β × Peptide binding dataset."""

    def __init__(self, cdr3b_list, pep_list, labels, max_cdr3=30, max_pep=25):
        self.cdr3b = cdr3b_list
        self.pep = pep_list
        self.labels = labels
        self.max_cdr3 = max_cdr3
        self.max_pep = max_pep

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cdr3b_ids = encode_sequence(self.cdr3b[idx], self.max_cdr3)
        pep_ids = encode_sequence(self.pep[idx], self.max_pep)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return cdr3b_ids, pep_ids, label


# ============================================================================
# Data Loading
# ============================================================================

def load_data(csv_path: str, min_epitope_samples: int = 10):
    """Load and filter the tc-hard dataset."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Raw rows: {len(df)}")

    # Filter: need CDR3β, peptide, valid label
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    mask = (
        df["cdr3.beta"].notna()
        & df["antigen.epitope"].notna()
        & df["label"].isin([0.0, 1.0])
    )
    df = df[mask].copy()
    logger.info(f"After basic filter: {len(df)}")

    # Filter out sequences with non-standard AAs
    df = df[df["cdr3.beta"].apply(lambda s: all(c in valid_aa for c in str(s)))]
    df = df[df["antigen.epitope"].apply(lambda s: all(c in valid_aa for c in str(s)))]
    logger.info(f"After AA filter: {len(df)}")

    # Filter CDR3β length: [8, 25]
    df = df[df["cdr3.beta"].str.len().between(8, 25)]
    # Filter peptide length: [8, 20]
    df = df[df["antigen.epitope"].str.len().between(8, 20)]
    logger.info(f"After length filter: {len(df)}")

    # Remove epitopes with too few samples
    epi_counts = df["antigen.epitope"].value_counts()
    valid_epis = epi_counts[epi_counts >= min_epitope_samples].index
    df = df[df["antigen.epitope"].isin(valid_epis)]
    logger.info(f"After min-sample filter ({min_epitope_samples}): {len(df)}")
    logger.info(f"Unique epitopes: {df['antigen.epitope'].nunique()}")
    logger.info(f"Label distribution: pos={sum(df['label']==1.0)}, neg={sum(df['label']==0.0)}")

    return df


def split_by_epitope(df, test_size=0.15, seed=42):
    """Split data by epitope groups — no epitope leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["antigen.epitope"]
    train_idx, val_idx = next(gss.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_epis = set(df_train["antigen.epitope"].unique())
    val_epis = set(df_val["antigen.epitope"].unique())
    overlap = train_epis & val_epis
    logger.info(f"Train: {len(df_train)} samples, {len(train_epis)} epitopes")
    logger.info(f"Val: {len(df_val)} samples, {len(val_epis)} epitopes")
    logger.info(f"Epitope overlap: {len(overlap)} (should be 0)")

    return df_train, df_val


# ============================================================================
# Training
# ============================================================================

def compute_per_epitope_auc(labels, probs, epitopes):
    """Compute per-epitope AUC and mean."""
    results = {}
    unique_epis = sorted(set(epitopes))
    for epi in unique_epis:
        mask = [e == epi for e in epitopes]
        y = [labels[i] for i, m in enumerate(mask) if m]
        p = [probs[i] for i, m in enumerate(mask) if m]
        if len(set(y)) < 2:
            continue
        try:
            auc = roc_auc_score(y, p)
            results[epi] = {"auc": auc, "n": len(y)}
        except ValueError:
            pass
    aucs = [v["auc"] for v in results.values()]
    mean_auc = np.mean(aucs) if aucs else 0.0
    return mean_auc, results


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch. Returns avg loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for cdr3b_ids, pep_ids, labels in loader:
        cdr3b_ids = cdr3b_ids.to(device)
        pep_ids = pep_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(cdr3b_ids, pep_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epitopes_list):
    """Evaluate: loss, global AUC, per-epitope AUC."""
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0
    n_batches = 0
    idx = 0

    for cdr3b_ids, pep_ids, labels in loader:
        cdr3b_ids = cdr3b_ids.to(device)
        pep_ids = pep_ids.to(device)
        labels_d = labels.to(device)

        logits = model(cdr3b_ids, pep_ids)
        loss = criterion(logits, labels_d)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
        idx += len(labels)

    avg_loss = total_loss / max(n_batches, 1)

    # Global AUC
    try:
        global_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        global_auc = 0.5

    # Per-epitope AUC
    epis = epitopes_list[:len(all_labels)]
    mean_epi_auc, per_epi = compute_per_epitope_auc(all_labels, all_probs, epis)

    return avg_loss, global_auc, mean_epi_auc, per_epi


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CDR3β×Peptide binding classifier")
    parser.add_argument("--data", required=True, help="Path to tc-hard CSV")
    parser.add_argument("--out-dir", default="runs/binding_classifier", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-attn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="Positive class weight for BCE. 0=auto-compute from data.")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    df = load_data(args.data)
    df_train, df_val = split_by_epitope(df, test_size=0.15, seed=args.seed)

    # Datasets
    train_ds = TCRBindingDataset(
        df_train["cdr3.beta"].tolist(),
        df_train["antigen.epitope"].tolist(),
        df_train["label"].tolist(),
    )
    val_ds = TCRBindingDataset(
        df_val["cdr3.beta"].tolist(),
        df_val["antigen.epitope"].tolist(),
        df_val["label"].tolist(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = build_model({
        "hidden_dim": args.hidden_dim,
        "n_attn_layers": args.n_attn_layers,
        "dropout": args.dropout,
    })
    model = model.to(device)
    logger.info(f"Model params: {model.n_params:,}")

    # Loss with class weighting
    if args.pos_weight <= 0:
        n_pos = sum(df_train["label"] == 1.0)
        n_neg = sum(df_train["label"] == 0.0)
        pw = n_neg / max(n_pos, 1)
        logger.info(f"Auto pos_weight: {pw:.2f} (neg/pos = {n_neg}/{n_pos})")
    else:
        pw = args.pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_val_auc = 0.0
    val_epitopes = df_val["antigen.epitope"].tolist()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_global_auc, val_mean_epi_auc, val_per_epi = evaluate(
            model, val_loader, criterion, device, val_epitopes,
        )
        scheduler.step()
        dt = time.time() - t0

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_global_auc={val_global_auc:.4f}  val_mean_epi_auc={val_mean_epi_auc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  time={dt:.1f}s"
        )

        # Log per-epitope AUC
        if epoch == 1 or epoch == args.epochs or val_mean_epi_auc > best_val_auc:
            logger.info(f"  Per-epitope AUC ({len(val_per_epi)} epitopes):")
            for epi in sorted(val_per_epi.keys()):
                info = val_per_epi[epi]
                logger.info(f"    {epi}: AUC={info['auc']:.4f} (n={info['n']})")

        # Save best
        if val_mean_epi_auc > best_val_auc:
            best_val_auc = val_mean_epi_auc
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "n_attn_layers": args.n_attn_layers,
                    "dropout": args.dropout,
                },
                "val_global_auc": val_global_auc,
                "val_mean_epi_auc": val_mean_epi_auc,
                "val_per_epi": val_per_epi,
            }, save_path)
            logger.info(f"  ★ New best! val_mean_epi_auc={val_mean_epi_auc:.4f} saved to {save_path}")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "n_attn_layers": args.n_attn_layers,
            "dropout": args.dropout,
        },
        "val_global_auc": val_global_auc,
        "val_mean_epi_auc": val_mean_epi_auc,
    }, os.path.join(args.out_dir, "final_model.pt"))

    logger.info(f"\nDone! Best val_mean_epi_auc: {best_val_auc:.4f}")
    logger.info(f"Models saved to {args.out_dir}")


if __name__ == "__main__":
    main()
