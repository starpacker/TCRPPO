#!/usr/bin/env python3
"""
Precompute ESM-2 embeddings for all sequences in high-quality SFT trajectories.

This script extends the existing ESM-2 cache with embeddings for any new
CDR3β and peptide sequences found in the SFT training data.
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to SFT trajectories JSON')
    parser.add_argument('--esm_cache', type=str, default='data/esm2_embeddings.pt',
                        help='Existing ESM-2 cache to extend')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: same as esm_cache)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.esm_cache

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Load trajectories
    print(f"Loading trajectories from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    trajectories = data['trajectories']

    # Collect all unique sequences needed
    all_seqs = set()
    for traj in trajectories:
        all_seqs.add(traj['init_tcr'])
        all_seqs.add(traj['final_tcr'])
        all_seqs.add(traj['peptide'])

        # Also add intermediate TCR states (after each action)
        current_tcr = list(traj['init_tcr'])
        for action in traj['actions']:
            op = action.get('op', action.get('op_type', 'STOP'))
            pos = action.get('pos', action.get('position', 0))
            tok = action.get('token', 'A')

            if op == 'SUB' or op == 0:
                if 0 <= pos < len(current_tcr):
                    current_tcr[pos] = tok
            elif op == 'INS' or op == 1:
                if 0 <= pos <= len(current_tcr):
                    current_tcr.insert(pos, tok)
            elif op == 'DEL' or op == 2:
                if 0 <= pos < len(current_tcr):
                    current_tcr.pop(pos)

            all_seqs.add(''.join(current_tcr))

    print(f"Total unique sequences needed: {len(all_seqs)}")

    # Load existing cache
    if os.path.exists(args.esm_cache):
        print(f"Loading existing cache from {args.esm_cache}...")
        cache = torch.load(args.esm_cache, map_location='cpu')
        print(f"  Existing cache size: {len(cache)}")
    else:
        cache = {}
        print("  No existing cache found, starting fresh")

    # Find missing sequences
    missing = [s for s in all_seqs if s not in cache]
    print(f"Sequences to encode: {len(missing)} (already cached: {len(all_seqs) - len(missing)})")

    if not missing:
        print("All sequences already cached!")
        return

    # Load ESM-2
    print("Loading ESM-2 model...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Encode missing sequences
    print(f"Encoding {len(missing)} sequences...")
    t0 = time.time()
    n_batches = (len(missing) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(missing), args.batch_size):
        batch_seqs = missing[i:i + args.batch_size]
        data_batch = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        _, _, batch_tokens = batch_converter(data_batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33]

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = token_repr[j, 1:seq_len+1, :].cpu()
            cache[seq] = emb.mean(dim=0).half()

        batch_idx = i // args.batch_size + 1
        if batch_idx % 50 == 0 or batch_idx == n_batches:
            print(f"  Batch {batch_idx}/{n_batches} ({len(missing) - (i + len(batch_seqs))} remaining)")

    dt = time.time() - t0
    print(f"Encoding done in {dt:.1f}s ({dt/len(missing)*1000:.1f}ms/seq)")

    # Save extended cache
    print(f"Saving extended cache to {args.output}...")
    torch.save(cache, args.output)
    file_size = os.path.getsize(args.output) / 1e6
    print(f"  Cache size: {len(cache)} sequences, {file_size:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
