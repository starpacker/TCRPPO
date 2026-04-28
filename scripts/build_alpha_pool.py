#!/usr/bin/env python3
"""Build TCRα pairing pool from tc-hard dataset for tFold oracle.

For each of the 12 eval targets, extract paired TCRα V-regions from
tc-hard ds_with_full_seq_v2.csv and save to data/alpha_pool/{target}.json.

This enables tFold scoring during training by providing paired TCRα sequences
for agent-generated CDR3β sequences.
"""

import os
import sys
import json
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.data.pmhc_loader import EVAL_TARGETS

# V-region extraction (copied from tFold data_pipeline.py)
_CONST_PATTERNS_BETA = ["EDLNKVFPP", "EDLKNVFPP", "DLNKVFPP", "DLKNVFPP"]
_CONST_PATTERNS_ALPHA = ["NIQNPDPAVYQ", "IQNPDPAVYQ"]
_CDR3_POS_IN_VREGION_BETA = 92
_CDR3_POS_IN_VREGION_ALPHA = 90


def _find_constant_start(full_seq: str, chain_type: str) -> int:
    """Find where the constant region starts."""
    patterns = _CONST_PATTERNS_BETA if chain_type == "beta" else _CONST_PATTERNS_ALPHA
    for pat in patterns:
        pos = full_seq.find(pat)
        if pos > 0:
            return pos
    return -1


def _find_leader_end(full_seq: str, cdr3: str, chain_type: str) -> int:
    """Estimate where the leader peptide ends."""
    cdr3_pos = full_seq.find(cdr3)
    if cdr3_pos < 0:
        return -1
    expected_cdr3_pos = (
        _CDR3_POS_IN_VREGION_BETA if chain_type == "beta"
        else _CDR3_POS_IN_VREGION_ALPHA
    )
    leader_end = cdr3_pos - expected_cdr3_pos
    return max(leader_end, 0)


def extract_vregion(full_seq: str, cdr3: str, chain_type: str) -> Optional[str]:
    """Extract V(D)J variable domain from a full-length TCR sequence."""
    if not full_seq or not cdr3 or len(full_seq) < 50:
        return None

    const_start = _find_constant_start(full_seq, chain_type)
    if const_start < 0:
        return None

    leader_end = _find_leader_end(full_seq, cdr3, chain_type)
    if leader_end < 0:
        return None

    if leader_end >= const_start:
        return None

    vregion = full_seq[leader_end:const_start]

    # Sanity check: V-region should be 90-140 aa
    if not (90 <= len(vregion) <= 140):
        return None

    # Sanity check: CDR3 should be present in the V-region
    if cdr3 not in vregion:
        return None

    return vregion


def build_alpha_pool(
    tc_hard_csv: str,
    targets: List[str],
    output_dir: str,
    max_pairs_per_target: int = 50,
):
    """Build alpha pool for each target from tc-hard paired data.

    For each target:
    1. Filter positive pairs with valid full_alpha_seq and full_beta_seq
    2. Extract TCRα V-region
    3. Extract TCRβ V-region (for pairing verification)
    4. Save paired data: {cdr3b, alpha_vregion, beta_vregion, cdr3a}

    Args:
        tc_hard_csv: Path to ds_with_full_seq_v2.csv
        targets: List of target peptides
        output_dir: Output directory for alpha pool JSON files
        max_pairs_per_target: Max pairs to save per target
    """
    print(f"Loading tc-hard data from {tc_hard_csv}...")
    df = pd.read_csv(tc_hard_csv)
    print(f"  Total rows: {len(df)}")

    # Filter for positive pairs with full sequences
    df_pos = df[
        (df["label"] == 1.0)
        & (df["full_alpha_seq"].notna())
        & (df["full_beta_seq"].notna())
        & (df["full_alpha_seq"].str.len() > 50)
        & (df["full_beta_seq"].str.len() > 50)
        & (df["cdr3.alpha"].notna())
        & (df["cdr3.beta"].notna())
    ].copy()
    print(f"  Positive pairs with full sequences: {len(df_pos)}")

    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    for target in tqdm(targets, desc="Building alpha pool"):
        target_df = df_pos[df_pos["antigen.epitope"] == target]

        if len(target_df) == 0:
            print(f"  {target}: No paired data found")
            summary[target] = {"n_pairs": 0, "reason": "no_data"}
            continue

        pairs = []
        n_alpha_fail = 0
        n_beta_fail = 0

        for _, row in target_df.iterrows():
            cdr3a = row["cdr3.alpha"]
            cdr3b = row["cdr3.beta"]
            full_alpha = row["full_alpha_seq"]
            full_beta = row["full_beta_seq"]

            # Extract V-regions
            alpha_vregion = extract_vregion(full_alpha, cdr3a, "alpha")
            if alpha_vregion is None:
                n_alpha_fail += 1
                continue

            beta_vregion = extract_vregion(full_beta, cdr3b, "beta")
            if beta_vregion is None:
                n_beta_fail += 1
                continue

            pairs.append({
                "cdr3b": cdr3b,
                "cdr3a": cdr3a,
                "alpha_vregion": alpha_vregion,
                "beta_vregion": beta_vregion,
                "alpha_vregion_len": len(alpha_vregion),
                "beta_vregion_len": len(beta_vregion),
            })

        # Deduplicate by (cdr3b, cdr3a) pair
        seen = set()
        unique_pairs = []
        for p in pairs:
            key = (p["cdr3b"], p["cdr3a"])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)

        # Sample if too many
        if len(unique_pairs) > max_pairs_per_target:
            import numpy as np
            rng = np.random.default_rng(42)
            indices = rng.choice(len(unique_pairs), size=max_pairs_per_target, replace=False)
            unique_pairs = [unique_pairs[i] for i in sorted(indices)]

        # Save
        output_file = os.path.join(output_dir, f"{target}.json")
        with open(output_file, "w") as f:
            json.dump({
                "target": target,
                "n_pairs": len(unique_pairs),
                "n_raw_positive": len(target_df),
                "n_alpha_fail": n_alpha_fail,
                "n_beta_fail": n_beta_fail,
                "pairs": unique_pairs,
            }, f, indent=2)

        summary[target] = {
            "n_pairs": len(unique_pairs),
            "n_raw_positive": len(target_df),
            "n_alpha_fail": n_alpha_fail,
            "n_beta_fail": n_beta_fail,
        }
        print(f"  {target}: {len(unique_pairs)} pairs "
              f"(raw={len(target_df)}, alpha_fail={n_alpha_fail}, beta_fail={n_beta_fail})")

    # Save summary
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAlpha pool built: {output_dir}")
    print(f"Summary saved: {summary_file}")

    # Print coverage table
    print(f"\n{'Target':<15} {'Pairs':<8} {'Raw+':<8} {'α fail':<8} {'β fail':<8}")
    print("-" * 50)
    total_pairs = 0
    for target in targets:
        if target in summary:
            s = summary[target]
            n = s["n_pairs"]
            total_pairs += n
            print(f"{target:<15} {n:<8} {s.get('n_raw_positive', 0):<8} "
                  f"{s.get('n_alpha_fail', 0):<8} {s.get('n_beta_fail', 0):<8}")
        else:
            print(f"{target:<15} {'N/A':<8}")

    print(f"\nTotal pairs: {total_pairs}")
    targets_with_data = sum(1 for t in targets if summary.get(t, {}).get("n_pairs", 0) > 0)
    print(f"Targets with data: {targets_with_data}/{len(targets)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TCRα pairing pool from tc-hard")
    parser.add_argument(
        "--tc_hard_csv",
        default="/share/liuyutian/TCRdata/tc-hard/reconstructed/ds_with_full_seq_v2.csv",
    )
    parser.add_argument("--output_dir", default="data/alpha_pool")
    parser.add_argument("--max_pairs_per_target", type=int, default=50)
    args = parser.parse_args()

    build_alpha_pool(
        tc_hard_csv=args.tc_hard_csv,
        targets=EVAL_TARGETS,
        output_dir=args.output_dir,
        max_pairs_per_target=args.max_pairs_per_target,
    )
