#!/usr/bin/env python
"""Precompute ESM-2 embeddings for all unique CDR3β and peptide sequences.

Saves a dictionary of {sequence: embedding_tensor} to a .pt file.
These embeddings are then used by the ESM-based binding classifier.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/precompute_esm_embeddings.py \
        --data /share/liuyutian/TCRdata/tc-hard/reconstructed/ds_with_full_seq_v2.csv \
        --output data/esm2_embeddings.pt \
        --batch-size 64
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("esm_embed")


def load_unique_sequences(csv_path: str):
    """Extract unique CDR3β and peptide sequences from the dataset."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    mask = (
        df["cdr3.beta"].notna()
        & df["antigen.epitope"].notna()
        & df["label"].isin([0.0, 1.0])
    )
    df = df[mask]
    df = df[df["cdr3.beta"].apply(lambda s: all(c in valid_aa for c in str(s)))]
    df = df[df["antigen.epitope"].apply(lambda s: all(c in valid_aa for c in str(s)))]
    df = df[df["cdr3.beta"].str.len().between(8, 25)]
    df = df[df["antigen.epitope"].str.len().between(8, 20)]

    unique_cdr3b = sorted(df["cdr3.beta"].unique())
    unique_pep = sorted(df["antigen.epitope"].unique())

    logger.info(f"Unique CDR3β: {len(unique_cdr3b)}")
    logger.info(f"Unique peptides: {len(unique_pep)}")

    return unique_cdr3b, unique_pep


def batch_encode_esm2(
    model, alphabet, sequences: List[str], device: str, batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Encode sequences with ESM-2 and return per-sequence mean embeddings."""
    batch_converter = alphabet.get_batch_converter()
    embeddings = {}
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33]  # [B, L+2, 1280]

        # Extract per-residue embeddings (skip BOS and EOS)
        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # token_repr[j, 1:seq_len+1] = actual residue embeddings
            emb = token_repr[j, 1 : seq_len + 1, :].cpu()  # [seq_len, 1280]
            # Store mean embedding for fast lookup
            embeddings[seq] = emb.mean(dim=0).half()  # [1280], float16 to save space

        batch_idx = i // batch_size + 1
        if batch_idx % 100 == 0 or batch_idx == n_batches:
            logger.info(f"  Batch {batch_idx}/{n_batches} ({len(embeddings)} sequences done)")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Precompute ESM-2 embeddings")
    parser.add_argument("--data", required=True, help="Path to tc-hard CSV")
    parser.add_argument("--output", default="data/esm2_embeddings.pt", help="Output .pt file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load sequences
    unique_cdr3b, unique_pep = load_unique_sequences(args.data)
    all_seqs = list(unique_cdr3b) + list(unique_pep)
    logger.info(f"Total sequences to embed: {len(all_seqs)}")

    # Load ESM-2
    logger.info("Loading ESM-2 model...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    logger.info("ESM-2 loaded")

    # Encode
    t0 = time.time()
    logger.info("Encoding sequences...")
    embeddings = batch_encode_esm2(model, alphabet, all_seqs, device, args.batch_size)
    dt = time.time() - t0
    logger.info(f"Encoding done: {len(embeddings)} sequences in {dt:.1f}s ({dt/len(embeddings)*1000:.1f}ms/seq)")

    # Save
    torch.save(embeddings, args.output)
    file_size = os.path.getsize(args.output) / 1e6
    logger.info(f"Saved to {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
