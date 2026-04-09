#!/usr/bin/env python3
"""Random-TCR null-hypothesis baseline for the decoy specificity evaluation.

Why this exists
---------------
``eval_decoy.py`` measures whether a *trained* PPO agent can produce TCRs that
are more specific to a target peptide than to ~3000 decoys. The headline number
in ``progress_3.md`` is ``mean AUROC = 0.4538`` — below 0.5. That number is
only meaningful if we know what AUROC we'd get from completely random TCRs.

Two possible worlds:

  (a) Random TCRs also give AUROC ≈ 0.45.
      → Then the problem is in **ERGO**: it gives similar scores to similar
      peptides regardless of which TCR you feed it. PPO training is innocent;
      reward design is irrelevant; the underlying scorer is the bottleneck.

  (b) Random TCRs give AUROC ≈ 0.50 and PPO TCRs give 0.45.
      → Then the problem is in **PPO training**: optimising the ERGO score
      collapses TCRs into a region where ERGO can no longer discriminate. The
      reward function (binding-only, no specificity term) is the bottleneck.

This script answers that question with the same metric pipeline as the trained
evaluation, so the two AUROCs are directly comparable.

What the script does
--------------------
For each target peptide, it generates ``--num_tcrs_per_target`` random CDR3β
sequences and then calls the SAME ``score_against_decoys`` /
``mc_dropout_predict_chunked`` codepath used by ``eval_decoy.py``. The output
is a CSV with the SAME schema as ``eval_decoy.py``'s output, so you can run
``eval_decoy_metrics.py`` on it without modification.

Two random-TCR generation modes:

* ``--mode pool`` (default): sample CDR3βs uniformly from
  ``data/tcrdb/test_uniq_tcr_seqs.txt``. These are real human TCRs that the
  trained agent has never seen — the cleanest "untrained but biologically
  plausible" baseline.

* ``--mode synthetic``: generate fully synthetic sequences by sampling
  amino acids uniformly. These are NOT biologically valid but provide a
  completely model-free baseline.

Usage
-----
    # Pool baseline (recommended)
    python evaluation/eval_decoy_random_baseline.py \
        --ergo_model ae_mcpas \
        --num_tcrs_per_target 50 \
        --n_mc_samples 20 \
        --out_csv evaluation/results/decoy/eval_decoy_random_pool.csv

    # Then run the same metrics pipeline used for the trained eval:
    python evaluation/eval_decoy_metrics.py \
        --csv evaluation/results/decoy/eval_decoy_random_pool.csv

Compare the resulting per-target AUROCs against the trained-model run to draw
the (a) vs (b) conclusion above.
"""
import argparse
import csv
import io
import os
import random
import sys

import numpy as np

# --- Path setup -----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CODE_DIR = os.path.join(_REPO, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# Re-use the SAME helpers as eval_decoy.py so the output is byte-for-byte
# format-compatible with eval_decoy_metrics.py.
from eval_utils import ERGO_MODELS, TEST_TCR_FILE, ensure_dir, load_tcrs  # noqa: E402
from eval_decoy import (  # noqa: E402
    CSV_FIELDS,
    load_decoys,
    load_target_peptides,
    score_against_decoys,
    write_results_row,
    load_reward_model,
)


# ============================================================================
#  Random TCR generators
# ============================================================================
AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"


def random_tcrs_from_pool(pool, n, rng):
    """Sample ``n`` TCRs uniformly from ``pool`` (with replacement if needed)."""
    if not pool:
        raise ValueError("TCR pool is empty")
    if n <= len(pool):
        return rng.sample(pool, n)
    # If we ask for more than the pool size, sample with replacement.
    return [rng.choice(pool) for _ in range(n)]


def random_tcrs_synthetic(n, rng, min_len=10, max_len=18):
    """Generate ``n`` purely random CDR3-like sequences.

    These are NOT biologically realistic — they ignore CDR3 conserved residues
    (C..F/W) and the V/J gene constraints — but they form a completely
    model-free null distribution.
    """
    out = []
    for _ in range(n):
        L = rng.randint(min_len, max_len)
        seq = "".join(rng.choice(AMINO_ACIDS) for _ in range(L))
        out.append(seq)
    return out


# ============================================================================
#  Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Random-TCR null baseline for decoy specificity evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ERGO model
    p.add_argument("--ergo_model", type=str, default="ae_mcpas",
                   help="ERGO model: 'ae_mcpas', 'ae_vdjdb', or path to .pt file")

    # Random TCR config
    p.add_argument("--mode", type=str, default="pool", choices=["pool", "synthetic"],
                   help="pool: sample from data/tcrdb/test_uniq_tcr_seqs.txt; "
                        "synthetic: generate fully random AA sequences")
    p.add_argument("--tcr_pool_file", type=str, default=TEST_TCR_FILE,
                   help="TCR pool to sample from when --mode=pool")
    p.add_argument("--num_tcrs_per_target", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # Decoy library (same as eval_decoy.py)
    p.add_argument("--decoy_library_root", type=str,
                   default=os.path.normpath(os.path.join(_REPO, "..", "decoy_library")))
    p.add_argument("--target_peptides_file", type=str, default=None)
    p.add_argument("--max_targets", type=int, default=0)
    p.add_argument("--no_decoy_a", action="store_true")
    p.add_argument("--no_decoy_b", action="store_true")
    p.add_argument("--no_decoy_c", action="store_true")
    p.add_argument("--no_decoy_d", action="store_true")

    # MC Dropout
    p.add_argument("--n_mc_samples", type=int, default=20)
    p.add_argument("--ergo_chunk_size", type=int, default=4096)

    # The Reward constructor needs these — values don't affect the baseline
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gmm_stop_criteria", type=float, default=1.2577)

    # Output
    p.add_argument("--out_csv", type=str, default=None)

    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # --- Load TCR pool ---------------------------------------------------
    if args.mode == "pool":
        if not os.path.isfile(args.tcr_pool_file):
            print("ERROR: TCR pool file not found: {}".format(args.tcr_pool_file))
            sys.exit(1)
        pool = load_tcrs(args.tcr_pool_file)
        print("[load] TCR pool: {} unique sequences from {}".format(
            len(pool), args.tcr_pool_file))
    else:
        pool = None
        print("[mode] synthetic — generating purely random AA sequences")

    # --- Load targets and decoys (same as eval_decoy.py) -----------------
    if not os.path.isdir(args.decoy_library_root):
        print("ERROR: decoy_library_root not found: {}".format(args.decoy_library_root))
        sys.exit(1)

    targets = load_target_peptides(args.decoy_library_root, args.target_peptides_file)
    if args.max_targets and args.max_targets > 0:
        targets = targets[:args.max_targets]
    target_seqs = [t["sequence"] for t in targets if t["sequence"]]
    print("[load] Targets: {}".format(len(target_seqs)))
    for t in targets:
        print("  - {} ({}, {})".format(t["sequence"], t["hla_allele"], t["source_protein"]))

    decoys = load_decoys(
        args.decoy_library_root,
        target_sequences=target_seqs,
        include_a=not args.no_decoy_a,
        include_b=not args.no_decoy_b,
        include_c=not args.no_decoy_c,
        include_d=not args.no_decoy_d,
    )
    print("[load] Total unique decoys: {}".format(len(decoys)))
    tier_counts = {}
    for d in decoys:
        tier_counts[d["tier"]] = tier_counts.get(d["tier"], 0) + 1
    print("       by tier: {}".format(tier_counts))

    if args.dry_run:
        print("[dry_run] Skipping reward-model load and scoring.")
        return

    # --- Output path -----------------------------------------------------
    if args.out_csv is None:
        ergo_tag = args.ergo_model if args.ergo_model in ERGO_MODELS else "custom"
        out_dir = os.path.join(_HERE, "results", "decoy")
        ensure_dir(out_dir)
        args.out_csv = os.path.join(
            out_dir, "eval_decoy_random_{}_{}.csv".format(args.mode, ergo_tag))
    else:
        ensure_dir(os.path.dirname(os.path.abspath(args.out_csv)))

    # --- Reward model for MC scoring -------------------------------------
    print("\n[run] Loading Reward + ERGO for MC dropout scoring...")
    # load_reward_model in eval_decoy.py only reads .ergo_model / .beta /
    # .gmm_stop_criteria, all of which we set above.
    reward_model = load_reward_model(args)

    # --- Score and write CSV ---------------------------------------------
    print("\n[run] Writing CSV: {}".format(args.out_csv))
    with io.open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for ti, target_meta in enumerate(targets):
            target_seq = target_meta["sequence"]

            # Generate random TCRs for this target.
            if args.mode == "pool":
                tcrs = random_tcrs_from_pool(pool, args.num_tcrs_per_target, rng)
            else:
                tcrs = random_tcrs_synthetic(args.num_tcrs_per_target, rng)

            print("\n[target {}/{}] {} ({}) — {} random TCRs".format(
                ti + 1, len(targets), target_seq, target_meta["hla_allele"], len(tcrs)))

            # Filter out X-containing TCRs (ERGO can't handle them)
            tcrs = [t for t in tcrs if t and "X" not in t]
            if not tcrs:
                print("  [skip] no usable TCRs")
                continue

            # Build fake "records" so we can re-use write_results_row
            fake_records = [{
                "init_tcr": t,
                "final_tcr": t,
                "edit_dist": 0.0,
                "gmm_likelihood": 0.0,
            } for t in tcrs]

            mean_grid, std_grid, all_meta = score_against_decoys(
                reward_model, tcrs, target_seq, decoys,
                n_mc_samples=args.n_mc_samples,
                ergo_chunk_size=args.ergo_chunk_size,
            )

            for i, rec in enumerate(fake_records):
                for j, dmeta in enumerate(all_meta):
                    is_target = (j == 0)
                    write_results_row(writer, target_meta, dmeta, rec,
                                      mean_grid[i, j], std_grid[i, j], is_target)

    print("\n[done] Wrote {}".format(args.out_csv))


if __name__ == "__main__":
    main()
