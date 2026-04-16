#!/usr/bin/env python
"""Train an ESM-2-based binding classifier on tc-hard data.

Uses precomputed ESM-2 mean embeddings (1280-dim) as input features,
with a projection + cross-attention + MLP classifier on top.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/train_esm_classifier.py \
        --data /share/liuyutian/TCRdata/tc-hard/reconstructed/ds_with_full_seq_v2.csv \
        --embeddings data/esm2_embeddings.pt \
        --out-dir runs/esm_classifier_v1 \
        --epochs 30 \
        --batch-size 2048
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_esm_clf")


# ============================================================================
# Model: ESM embedding → project → cross-attend → classify
# ============================================================================

class ESMBindingClassifier(nn.Module):
    """CDR3β × Peptide classifier using precomputed ESM-2 embeddings."""

    def __init__(self, esm_dim=1280, proj_dim=128, n_heads=4, mlp_hidden=256, dropout=0.15):
        super().__init__()
        self.proj_cdr3b = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_pep = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bilinear interaction
        self.bilinear = nn.Bilinear(proj_dim, proj_dim, proj_dim)
        self.bilinear_norm = nn.LayerNorm(proj_dim)

        # Classifier on concatenated features
        cat_dim = proj_dim * 3  # cdr3b + pep + interaction
        self.classifier = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, cdr3b_emb, pep_emb):
        """Forward pass on ESM embeddings. Returns logits [B]."""
        h_c = self.proj_cdr3b(cdr3b_emb)   # [B, proj_dim]
        h_p = self.proj_pep(pep_emb)        # [B, proj_dim]

        # Bilinear interaction
        h_int = self.bilinear(h_c, h_p)
        h_int = self.bilinear_norm(h_int)

        # Classify
        h = torch.cat([h_c, h_p, h_int], dim=-1)
        return self.classifier(h).squeeze(-1)


# ============================================================================
# Dataset
# ============================================================================

class ESMBindingDataset(Dataset):
    """Dataset using precomputed ESM embeddings."""

    def __init__(self, cdr3b_list, pep_list, labels, embeddings_dict):
        self.cdr3b = cdr3b_list
        self.pep = pep_list
        self.labels = labels
        self.emb = embeddings_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cdr3b_emb = self.emb.get(self.cdr3b[idx])
        pep_emb = self.emb.get(self.pep[idx])

        if cdr3b_emb is None:
            cdr3b_emb = torch.zeros(1280, dtype=torch.float16)
        if pep_emb is None:
            pep_emb = torch.zeros(1280, dtype=torch.float16)

        return cdr3b_emb.float(), pep_emb.float(), torch.tensor(self.labels[idx], dtype=torch.float32)


# ============================================================================
# Data Loading
# ============================================================================

def load_data(csv_path: str, min_epitope_samples: int = 10):
    """Load and filter dataset."""
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    mask = df["cdr3.beta"].notna() & df["antigen.epitope"].notna() & df["label"].isin([0.0, 1.0])
    df = df[mask]
    df = df[df["cdr3.beta"].apply(lambda s: all(c in valid_aa for c in str(s)))]
    df = df[df["antigen.epitope"].apply(lambda s: all(c in valid_aa for c in str(s)))]
    df = df[df["cdr3.beta"].str.len().between(8, 25)]
    df = df[df["antigen.epitope"].str.len().between(8, 20)]

    epi_counts = df["antigen.epitope"].value_counts()
    valid_epis = epi_counts[epi_counts >= min_epitope_samples].index
    df = df[df["antigen.epitope"].isin(valid_epis)]

    logger.info(f"Filtered: {len(df)} samples, {df['antigen.epitope'].nunique()} epitopes")
    return df


def compute_per_epitope_auc(labels, probs, epitopes):
    """Compute per-epitope AUC."""
    results = {}
    for epi in sorted(set(epitopes)):
        idxs = [i for i, e in enumerate(epitopes) if e == epi]
        y = [labels[i] for i in idxs]
        p = [probs[i] for i in idxs]
        if len(set(y)) < 2:
            continue
        try:
            results[epi] = {"auc": roc_auc_score(y, p), "n": len(y)}
        except ValueError:
            pass
    aucs = [v["auc"] for v in results.values()]
    return (np.mean(aucs) if aucs else 0.0), results


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for cdr3b_emb, pep_emb, labels in loader:
        cdr3b_emb = cdr3b_emb.to(device)
        pep_emb = pep_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(cdr3b_emb, pep_emb)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epitopes_list):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0
    n = 0

    for cdr3b_emb, pep_emb, labels in loader:
        cdr3b_emb = cdr3b_emb.to(device)
        pep_emb = pep_emb.to(device)
        labels_d = labels.to(device)

        logits = model(cdr3b_emb, pep_emb)
        loss = criterion(logits, labels_d)
        total_loss += loss.item()
        n += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(n, 1)
    try:
        global_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        global_auc = 0.5

    epis = epitopes_list[:len(all_labels)]
    mean_epi_auc, per_epi = compute_per_epitope_auc(all_labels, all_probs, epis)
    return avg_loss, global_auc, mean_epi_auc, per_epi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--embeddings", default="data/esm2_embeddings.pt")
    parser.add_argument("--out-dir", default="runs/esm_classifier_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load embeddings
    logger.info("Loading ESM-2 embeddings...")
    emb_dict = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    logger.info(f"Loaded {len(emb_dict)} embeddings")

    # Load data
    df = load_data(args.data)

    # Filter to sequences that have embeddings
    has_emb = df["cdr3.beta"].isin(emb_dict) & df["antigen.epitope"].isin(emb_dict)
    df = df[has_emb]
    logger.info(f"After embedding filter: {len(df)} samples")

    # Split by epitope (zero-shot)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    groups = df["antigen.epitope"]
    train_idx, val_idx = next(gss.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_epis = set(df_train["antigen.epitope"].unique())
    val_epis = set(df_val["antigen.epitope"].unique())
    logger.info(f"Train: {len(df_train)} ({len(train_epis)} epi), Val: {len(df_val)} ({len(val_epis)} epi)")
    logger.info(f"Epitope overlap: {len(train_epis & val_epis)}")

    # Datasets
    train_ds = ESMBindingDataset(
        df_train["cdr3.beta"].tolist(), df_train["antigen.epitope"].tolist(),
        df_train["label"].tolist(), emb_dict,
    )
    val_ds = ESMBindingDataset(
        df_val["cdr3.beta"].tolist(), df_val["antigen.epitope"].tolist(),
        df_val["label"].tolist(), emb_dict,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = ESMBindingClassifier(proj_dim=args.proj_dim, mlp_hidden=args.mlp_hidden, dropout=args.dropout).to(device)
    logger.info(f"Model params: {model.n_params:,}")

    # Class weighting
    n_pos = sum(df_train["label"] == 1.0)
    n_neg = sum(df_train["label"] == 0.0)
    pw = n_neg / max(n_pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))
    logger.info(f"pos_weight: {pw:.2f}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    val_epitopes = df_val["antigen.epitope"].tolist()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
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

        # Save best
        if val_mean_epi_auc > best_val_auc:
            best_val_auc = val_mean_epi_auc
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "esm_dim": 1280,
                    "proj_dim": args.proj_dim,
                    "mlp_hidden": args.mlp_hidden,
                    "dropout": args.dropout,
                },
                "val_global_auc": val_global_auc,
                "val_mean_epi_auc": val_mean_epi_auc,
                "val_per_epi": val_per_epi,
            }, save_path)
            logger.info(f"  ★ New best! val_mean_epi_auc={val_mean_epi_auc:.4f}")

            # Print per-epitope
            for epi in sorted(val_per_epi.keys()):
                info = val_per_epi[epi]
                logger.info(f"    {epi}: AUC={info['auc']:.4f} (n={info['n']})")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "model_config": {"esm_dim": 1280, "proj_dim": args.proj_dim, "mlp_hidden": args.mlp_hidden, "dropout": args.dropout},
        "val_global_auc": val_global_auc,
        "val_mean_epi_auc": val_mean_epi_auc,
    }, os.path.join(args.out_dir, "final_model.pt"))

    logger.info(f"\nDone! Best val_mean_epi_auc: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
