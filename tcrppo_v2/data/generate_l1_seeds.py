"""Generate L1 curriculum seeds: top-K TCRdb sequences per target by ERGO score.

Usage:
    python -m tcrppo_v2.data.generate_l1_seeds --targets GILGFVFTL NLVPMVATV ...
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.utils.constants import (
    ERGO_MODEL_DIR, ERGO_AE_FILE, TCRDB_PATH,
    MIN_TCR_LEN, MAX_TCR_LEN,
)
from tcrppo_v2.utils.encoding import is_valid_tcr
from tcrppo_v2.data.pmhc_loader import EVAL_TARGETS


def load_tcrdb_sample(tcrdb_path: str, n_sample: int = 50000, seed: int = 42) -> List[str]:
    """Load a random subsample of TCRdb for L1 scoring."""
    train_file = os.path.join(tcrdb_path, "train_uniq_tcr_seqs.txt")
    all_seqs = []
    with open(train_file) as f:
        for line in f:
            seq = line.strip()
            if seq and is_valid_tcr(seq) and MIN_TCR_LEN <= len(seq) <= MAX_TCR_LEN:
                all_seqs.append(seq)

    rng = np.random.default_rng(seed)
    if len(all_seqs) > n_sample:
        indices = rng.choice(len(all_seqs), size=n_sample, replace=False)
        return [all_seqs[i] for i in indices]
    return all_seqs


def generate_l1_seeds(
    targets: List[str],
    output_dir: str,
    n_sample: int = 50000,
    top_k: int = 500,
    batch_size: int = 512,
    device: str = "cuda",
) -> None:
    """Generate L1 seeds for given targets."""
    from tcrppo_v2.scorers.affinity_ergo import AffinityERGOScorer

    model_file = os.path.join(ERGO_MODEL_DIR, "ae_mcpas1.pt")
    scorer = AffinityERGOScorer(
        model_file=model_file,
        ae_file=ERGO_AE_FILE,
        device=device,
        mc_samples=1,  # Fast mode, no MC dropout needed for ranking
    )

    # Load TCRdb subsample
    print(f"Loading TCRdb sample (n={n_sample})...")
    tcr_seqs = load_tcrdb_sample(TCRDB_PATH, n_sample=n_sample)
    print(f"Loaded {len(tcr_seqs)} valid TCRdb sequences")

    os.makedirs(output_dir, exist_ok=True)

    for target in targets:
        t0 = time.time()
        print(f"\nScoring {len(tcr_seqs)} TCRs against {target}...")

        # Score in batches
        all_scores = []
        for i in range(0, len(tcr_seqs), batch_size):
            batch_tcrs = tcr_seqs[i : i + batch_size]
            batch_peps = [target] * len(batch_tcrs)
            scores = scorer.score_batch_fast(batch_tcrs, batch_peps)
            all_scores.extend(scores)

            if (i // batch_size) % 20 == 0:
                print(f"  Scored {min(i + batch_size, len(tcr_seqs))}/{len(tcr_seqs)}")

        # Get top-K
        scores_arr = np.array(all_scores)
        top_indices = np.argsort(scores_arr)[-top_k:][::-1]

        top_seqs = [tcr_seqs[i] for i in top_indices]
        top_scores_vals = [scores_arr[i] for i in top_indices]

        # Save
        out_file = os.path.join(output_dir, f"{target}.txt")
        with open(out_file, "w") as f:
            for seq in top_seqs:
                f.write(seq + "\n")

        elapsed = time.time() - t0
        print(
            f"  {target}: top-{top_k} saved to {out_file} "
            f"(best={top_scores_vals[0]:.4f}, worst={top_scores_vals[-1]:.4f}, "
            f"time={elapsed:.1f}s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate L1 curriculum seeds")
    parser.add_argument(
        "--targets", nargs="+", default=list(EVAL_TARGETS.keys()),
        help="Target peptides"
    )
    parser.add_argument("--output_dir", default="data/l1_seeds", help="Output directory")
    parser.add_argument("--n_sample", type=int, default=50000, help="TCRdb subsample size")
    parser.add_argument("--top_k", type=int, default=500, help="Top-K per target")
    parser.add_argument("--batch_size", type=int, default=512, help="ERGO batch size")
    parser.add_argument("--device", default="cuda", help="Torch device")

    args = parser.parse_args()
    generate_l1_seeds(
        targets=args.targets,
        output_dir=args.output_dir,
        n_sample=args.n_sample,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
    )
