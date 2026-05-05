#!/usr/bin/env python
"""Parallel tFold cache warmup with multiple GPUs.

Strategy:
- Generate diverse TCRs for each peptide (mutations, insertions, deletions)
- Score with tFold and cache features
- Use multiple GPUs to parallelize
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, '/share/liuyutian/tcrppo_v2')

from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer

# Test3's 4 empirically validated peptides
PEPTIDES = ["KLWASPLHV", "FPRPWLHGL", "KAFSPEVIPMF", "HSKKKCDEL"]

# Amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"


def generate_diverse_tcrs(seed_tcrs: List[str], n_target: int, seed: int = 42) -> List[str]:
    """Generate diverse TCRs via mutations, insertions, deletions."""
    rng = np.random.default_rng(seed)
    tcrs = set(seed_tcrs)
    tcr_list = list(seed_tcrs)  # Keep a list for fast sampling

    batch_size = 1000
    while len(tcrs) < n_target:
        # Generate a batch of candidates
        for _ in range(batch_size):
            # Pick a random seed from the list (O(1) instead of O(n))
            base = tcr_list[rng.integers(0, len(tcr_list))]

            # Random operation
            op = rng.choice(['mutate', 'insert', 'delete', 'random'])

            if op == 'mutate' and len(base) > 0:
                # Point mutation
                pos = rng.integers(0, len(base))
                new_aa = rng.choice(list(AA))
                new_tcr = base[:pos] + new_aa + base[pos+1:]
            elif op == 'insert' and len(base) < 27:
                # Insert amino acid
                pos = rng.integers(0, len(base)+1)
                new_aa = rng.choice(list(AA))
                new_tcr = base[:pos] + new_aa + base[pos:]
            elif op == 'delete' and len(base) > 8:
                # Delete amino acid
                pos = rng.integers(0, len(base))
                new_tcr = base[:pos] + base[pos+1:]
            else:
                # Random TCR
                length = rng.integers(10, 20)
                new_tcr = ''.join(rng.choice(list(AA)) for _ in range(length))

            if 8 <= len(new_tcr) <= 27 and new_tcr not in tcrs:
                tcrs.add(new_tcr)
                tcr_list.append(new_tcr)

            if len(tcrs) >= n_target:
                break

    return list(tcrs)


def warmup_worker(gpu_id: int, peptide: str, tcrs: List[str], cache_path: str):
    """Worker process: score TCRs on a specific GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Load tFold scorer (cache_only=False to enable feature extraction)
    scorer = AffinityTFoldScorer(
        cache_path=cache_path,
        device='cuda',
        cache_only=False,  # CRITICAL: Allow cache misses to call tFold
        cache_miss_score=None,  # Not used when cache_only=False
    )

    print(f"[GPU {gpu_id}] Starting warmup for {peptide}: {len(tcrs)} TCRs")

    t0 = time.time()
    for i, tcr in enumerate(tcrs):
        try:
            score, std = scorer.score(tcr, peptide)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tcrs) - i - 1) / rate / 3600
                print(f"[GPU {gpu_id}] {peptide}: {i+1}/{len(tcrs)} ({rate:.2f} TCR/s, ETA {eta:.1f}h)")
        except Exception as e:
            print(f"[GPU {gpu_id}] Error scoring {tcr} on {peptide}: {e}")

    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] Completed {peptide}: {len(tcrs)} TCRs in {elapsed/3600:.1f}h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tcrs_per_peptide', type=int, default=50000,
                        help='Number of TCRs to generate per peptide')
    parser.add_argument('--gpus', type=str, default='0,1,3,4',
                        help='Comma-separated GPU IDs to use')
    parser.add_argument('--cache_path', type=str,
                        default='/share/liuyutian/tcrppo_v2/data/tfold_feature_cache.db')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(',')]

    print("="*60)
    print("tFold Cache Warmup (Parallel)")
    print("="*60)
    print(f"Peptides: {PEPTIDES}")
    print(f"TCRs per peptide: {args.n_tcrs_per_peptide}")
    print(f"Total TCRs: {len(PEPTIDES) * args.n_tcrs_per_peptide}")
    print(f"GPUs: {gpu_ids}")
    print(f"Cache: {args.cache_path}")
    print()

    # Load seed TCRs from TCRdb
    print("Loading seed TCRs from TCRdb...")
    tcrdb_path = "/share/liuyutian/TCRPPO/data/tcrdb/train_uniq_tcr_seqs.txt"
    with open(tcrdb_path) as f:
        seed_tcrs = [line.strip() for line in f if line.strip()][:1000]  # 1000 seed TCRs
    print(f"Loaded {len(seed_tcrs)} seed TCRs from TCRdb")

    # Generate diverse TCRs for each peptide
    print(f"\nGenerating {args.n_tcrs_per_peptide} diverse TCRs per peptide...")
    peptide_tcrs = {}
    for pep in PEPTIDES:
        tcrs = generate_diverse_tcrs(seed_tcrs, args.n_tcrs_per_peptide, seed=args.seed)
        peptide_tcrs[pep] = tcrs
        print(f"  {pep}: {len(tcrs)} TCRs")

    # Distribute work across GPUs
    print(f"\nStarting parallel warmup on {len(gpu_ids)} GPUs...")
    print(f"Estimated time: {args.n_tcrs_per_peptide * len(PEPTIDES) / len(gpu_ids) / 3600:.1f}h per GPU")
    print()

    # Create worker processes
    processes = []
    for i, pep in enumerate(PEPTIDES):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        p = mp.Process(target=warmup_worker, args=(gpu_id, pep, peptide_tcrs[pep], args.cache_path))
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts

    # Wait for all workers
    for p in processes:
        p.join()

    print("\n" + "="*60)
    print("Warmup complete!")
    print("="*60)


if __name__ == '__main__':
    main()
