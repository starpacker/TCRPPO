#!/usr/bin/env python3
"""Compute summary metrics from the per-pair CSV produced by ``eval_decoy.py``.

The input CSV has one row per (target_pep, generated_tcr, scored_peptide)
triple, with columns including ``ergo_mean``, ``ergo_std``, ``is_target_pep``,
``decoy_source_tier``, ``decoy_evidence_level``, ``decoy_hla``, etc.

Outputs (printed to stdout, plus optional CSV exports):

  1. **Per-target summary**: For each target peptide, mean+/-std ERGO on the
     target itself vs the decoy pool, specificity ratio, AUROC, and the
     fraction of decoys that score >= a configurable high-binding threshold.

  2. **Per-peptide table**: One row per (target, scored_peptide) — averaged
     across all generated TCRs. This is the "for each peptide, report HLA"
     view requested by the user.

  3. **Evidence-level bucketing**: Average decoy ERGO score grouped by the
     literature evidence level (Level 1 clinical fatal vs Level 3 in silico).

  4. **Uncertainty calibration**: Mean MC-dropout std for target peptide hits
     vs decoy hits, and within high-confidence vs low-confidence bands.

  5. **Top-K worst decoys**: The K highest-scoring decoys per target — these
     are the most concerning off-target hits and warrant closer inspection.

Usage:
    python eval_decoy_metrics.py --csv evaluation/results/decoy/eval_decoy_ae_mcpas.csv
    python eval_decoy_metrics.py --csv <file> --out_dir analysis/ --high_threshold 0.9
"""
import argparse
import csv
import io
import os
import sys
from collections import defaultdict, OrderedDict

import numpy as np


# ----------------------------------------------------------------------------
#  Loader
# ----------------------------------------------------------------------------
NUMERIC_FIELDS = ("edit_dist", "gmm_likelihood", "ergo_mean", "ergo_std", "is_target_pep")


def load_csv(csv_path):
    """Load the eval_decoy.py output CSV into a list of dicts (numeric fields cast)."""
    rows = []
    with io.open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in NUMERIC_FIELDS:
                if k in r and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            r["is_target_pep"] = int(r.get("is_target_pep", 0))
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
#  Metric helpers
# ----------------------------------------------------------------------------
def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def safe_std(xs):
    return float(np.std(xs)) if len(xs) else float("nan")


def auroc(positive_scores, negative_scores):
    """Compute ROC-AUC: positives = target ERGO scores, negatives = decoy scores.

    Implemented manually with rank statistics to avoid an sklearn dependency
    when this script is run on a minimal env.
    """
    if not len(positive_scores) or not len(negative_scores):
        return float("nan")
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    n_pos = len(pos)
    n_neg = len(neg)
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average rank for ties
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = ranks[:n_pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------------
#  Per-target summary
# ----------------------------------------------------------------------------
def per_target_summary(rows, high_threshold=0.9, top_k=10):
    """Group rows by target_pep, compute target vs decoy stats and AUROC."""
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target_pep"]].append(r)

    out = []
    for target_pep in sorted(by_target.keys()):
        recs = by_target[target_pep]
        target_rows = [r for r in recs if r["is_target_pep"] == 1]
        decoy_rows = [r for r in recs if r["is_target_pep"] == 0]
        target_scores = [r["ergo_mean"] for r in target_rows]
        decoy_scores = [r["ergo_mean"] for r in decoy_rows]
        target_stds = [r["ergo_std"] for r in target_rows]
        decoy_stds = [r["ergo_std"] for r in decoy_rows]

        n_decoy_high = sum(1 for s in decoy_scores if s >= high_threshold)
        n_decoy = max(1, len(decoy_scores))

        # Specificity: how much higher is the target score vs the decoy mean?
        target_mean = safe_mean(target_scores)
        decoy_mean = safe_mean(decoy_scores)
        spec_ratio = (target_mean / decoy_mean) if decoy_mean > 0 else float("inf")
        spec_diff = target_mean - decoy_mean

        out.append({
            "target_pep": target_pep,
            "target_hla": target_rows[0]["target_hla"] if target_rows else "?",
            "target_source_protein": target_rows[0]["target_source_protein"] if target_rows else "",
            "n_tcrs": len(target_rows),
            "n_decoys": len(decoy_rows),
            "target_ergo_mean": target_mean,
            "target_ergo_std_across_tcrs": safe_std(target_scores),
            "target_mc_uncertainty_mean": safe_mean(target_stds),
            "decoy_ergo_mean": decoy_mean,
            "decoy_ergo_std_across_pairs": safe_std(decoy_scores),
            "decoy_mc_uncertainty_mean": safe_mean(decoy_stds),
            "specificity_diff": spec_diff,
            "specificity_ratio": spec_ratio,
            "auroc_target_vs_decoys": auroc(target_scores, decoy_scores),
            "decoys_above_threshold": n_decoy_high,
            "decoys_above_threshold_pct": 100.0 * n_decoy_high / n_decoy,
        })
    return out


def print_per_target_table(summary, high_threshold):
    print("\n" + "=" * 110)
    print("PER-TARGET SUMMARY (high-binding threshold = {})".format(high_threshold))
    print("=" * 110)
    header = ("{:<12} {:<14} {:>6} {:>7} {:>11} {:>11} {:>10} {:>8} {:>10} {:>11}").format(
        "Target", "HLA", "#TCRs", "#Decoys", "Tgt(mean)", "Dcy(mean)",
        "Diff", "AUROC", "MCstd_T", ">=hi(decoy)")
    print(header)
    print("-" * 110)
    for s in summary:
        print(("{:<12} {:<14} {:>6d} {:>7d} {:>11.4f} {:>11.4f} {:>10.4f} "
               "{:>8.4f} {:>10.4f} {:>9d}({:>4.1f}%)").format(
            s["target_pep"][:12],
            s["target_hla"][:14],
            s["n_tcrs"],
            s["n_decoys"],
            s["target_ergo_mean"],
            s["decoy_ergo_mean"],
            s["specificity_diff"],
            s["auroc_target_vs_decoys"],
            s["target_mc_uncertainty_mean"],
            s["decoys_above_threshold"],
            s["decoys_above_threshold_pct"],
        ))
    print("=" * 110)


# ----------------------------------------------------------------------------
#  Per-peptide table (one row per (target, scored peptide))
# ----------------------------------------------------------------------------
def per_peptide_table(rows):
    """One row per (target_pep, decoy_pep) — averaged across generated TCRs.

    This is the per-peptide HLA-aware view requested by the user.
    """
    grouped = defaultdict(list)
    for r in rows:
        key = (r["target_pep"], r["decoy_pep"])
        grouped[key].append(r)

    table = []
    for (tgt, pep), recs in grouped.items():
        scores = [r["ergo_mean"] for r in recs]
        stds = [r["ergo_std"] for r in recs]
        rec0 = recs[0]
        table.append({
            "target_pep": tgt,
            "target_hla": rec0["target_hla"],
            "scored_pep": pep,
            "scored_pep_hla": rec0["decoy_hla"],
            "is_target_pep": rec0["is_target_pep"],
            "tier": rec0["decoy_source_tier"],
            "evidence_level": rec0["decoy_evidence_level"],
            "source_protein": rec0["decoy_source_protein"],
            "n_tcrs": len(recs),
            "ergo_mean": safe_mean(scores),
            "ergo_std_across_tcrs": safe_std(scores),
            "mc_uncertainty_mean": safe_mean(stds),
        })
    table.sort(key=lambda r: (r["target_pep"], -r["is_target_pep"], -r["ergo_mean"]))
    return table


def print_per_peptide_top(table, top_per_target=10):
    print("\n" + "=" * 110)
    print("PER-PEPTIDE TABLE (top {} highest-scoring scored peptides per target)".format(top_per_target))
    print("=" * 110)
    print("{:<12} {:<14} {:<14} {:<14} {:>4} {:>4} {:>10} {:>8}".format(
        "Target", "TgtHLA", "ScoredPep", "ScoredHLA", "Tier", "Tgt?", "Mean", "MCstd"))
    print("-" * 110)
    by_target = defaultdict(list)
    for r in table:
        by_target[r["target_pep"]].append(r)
    for tgt in sorted(by_target.keys()):
        rows = by_target[tgt][:top_per_target]
        for r in rows:
            tag = "Y" if r["is_target_pep"] else " "
            print("{:<12} {:<14} {:<14} {:<14} {:>4} {:>4} {:>10.4f} {:>8.4f}".format(
                r["target_pep"][:12],
                r["target_hla"][:14],
                r["scored_pep"][:14],
                r["scored_pep_hla"][:14],
                r["tier"][:4],
                tag,
                r["ergo_mean"],
                r["mc_uncertainty_mean"],
            ))
        print("-" * 110)


# ----------------------------------------------------------------------------
#  Per-tier AUROC breakdown
# ----------------------------------------------------------------------------
def per_tier_auroc(rows):
    """Compute AUROC of (target peptide score) vs (decoy score) **separately**
    for each decoy tier (A/B/C/D), per target peptide AND globally.

    Tier A = 1-2 aa mutants of the target — the hardest test.
    Tier B = 2-3 aa mutants — slightly easier.
    Tier C = global random/literature pool — sequence-unrelated decoys.
    Tier D = other known binders — biologically plausible cross-reactivity.

    A model that simply rank-orders peptides by sequence similarity will get
    AUROC ~0.5 on Tier A but high AUROC on Tier C. A model that has learnt
    real binding specificity should achieve >0.5 on Tier A as well.

    Returns:
        list of dicts, sorted by (target, tier).
    """
    by_target_tier = defaultdict(lambda: {"target": [], "decoy": []})
    target_scores_by_target = defaultdict(list)
    decoy_scores_by_tier = defaultdict(list)
    target_scores_global = []

    for r in rows:
        tgt = r["target_pep"]
        if r["is_target_pep"] == 1:
            target_scores_by_target[tgt].append(r["ergo_mean"])
            target_scores_global.append(r["ergo_mean"])
        else:
            tier = r["decoy_source_tier"] or "?"
            by_target_tier[(tgt, tier)]["decoy"].append(r["ergo_mean"])
            decoy_scores_by_tier[tier].append(r["ergo_mean"])

    # Per-target × per-tier
    out = []
    for (tgt, tier), bucket in sorted(by_target_tier.items()):
        tgt_scores = target_scores_by_target.get(tgt, [])
        out.append({
            "target_pep": tgt,
            "tier": tier,
            "n_target": len(tgt_scores),
            "n_decoy": len(bucket["decoy"]),
            "target_mean": safe_mean(tgt_scores),
            "decoy_mean": safe_mean(bucket["decoy"]),
            "auroc": auroc(tgt_scores, bucket["decoy"]),
        })

    # Global per-tier (across ALL targets)
    global_rows = []
    for tier in sorted(decoy_scores_by_tier.keys()):
        global_rows.append({
            "target_pep": "<ALL>",
            "tier": tier,
            "n_target": len(target_scores_global),
            "n_decoy": len(decoy_scores_by_tier[tier]),
            "target_mean": safe_mean(target_scores_global),
            "decoy_mean": safe_mean(decoy_scores_by_tier[tier]),
            "auroc": auroc(target_scores_global, decoy_scores_by_tier[tier]),
        })
    return out, global_rows


def print_per_tier_table(per_tier_rows, global_rows):
    print("\n" + "=" * 90)
    print("PER-TIER AUROC BREAKDOWN")
    print("=" * 90)
    print("Tier-level breakdown reveals WHERE the model fails:")
    print("  Tier A = 1-2 aa mutants    (hardest — sequence near target)")
    print("  Tier B = 2-3 aa mutants    (slightly easier)")
    print("  Tier C = unrelated random  (easiest — should be near 1.0)")
    print("  Tier D = other binders     (biologically plausible cross-reactivity)")
    print("-" * 90)

    print("{:<14} {:<6} {:>8} {:>8} {:>11} {:>11} {:>10}".format(
        "Target", "Tier", "#Tgt", "#Dcy", "Tgt(mean)", "Dcy(mean)", "AUROC"))
    print("-" * 90)
    for r in per_tier_rows:
        print("{:<14} {:<6} {:>8d} {:>8d} {:>11.4f} {:>11.4f} {:>10.4f}".format(
            r["target_pep"][:14],
            r["tier"][:6],
            r["n_target"],
            r["n_decoy"],
            r["target_mean"],
            r["decoy_mean"],
            r["auroc"],
        ))

    print("-" * 90)
    print("GLOBAL (across all targets):")
    for r in global_rows:
        print("{:<14} {:<6} {:>8d} {:>8d} {:>11.4f} {:>11.4f} {:>10.4f}".format(
            r["target_pep"][:14],
            r["tier"][:6],
            r["n_target"],
            r["n_decoy"],
            r["target_mean"],
            r["decoy_mean"],
            r["auroc"],
        ))
    print("=" * 90)


# ----------------------------------------------------------------------------
#  Evidence-level bucketing
# ----------------------------------------------------------------------------
def evidence_level_breakdown(rows):
    bucket = defaultdict(list)
    for r in rows:
        if r["is_target_pep"] == 1:
            continue
        ev = r["decoy_evidence_level"] or "(no_label)"
        bucket[ev].append(r["ergo_mean"])
    print("\n" + "=" * 70)
    print("DECOY ERGO SCORES BY EVIDENCE LEVEL (decoy rows only)")
    print("=" * 70)
    print("{:<40} {:>8} {:>10} {:>10}".format("Evidence level", "n", "mean", "std"))
    print("-" * 70)
    for ev in sorted(bucket.keys(), key=lambda k: -len(bucket[k])):
        s = bucket[ev]
        print("{:<40} {:>8d} {:>10.4f} {:>10.4f}".format(
            ev[:40], len(s), safe_mean(s), safe_std(s)))


# ----------------------------------------------------------------------------
#  Uncertainty diagnostics
# ----------------------------------------------------------------------------
def uncertainty_diagnostics(rows):
    target_stds = [r["ergo_std"] for r in rows if r["is_target_pep"] == 1]
    decoy_stds = [r["ergo_std"] for r in rows if r["is_target_pep"] == 0]

    target_means = [r["ergo_mean"] for r in rows if r["is_target_pep"] == 1]
    decoy_means = [r["ergo_mean"] for r in rows if r["is_target_pep"] == 0]

    print("\n" + "=" * 70)
    print("MC DROPOUT UNCERTAINTY DIAGNOSTICS")
    print("=" * 70)
    print("{:<40} {:>10} {:>10}".format("", "mean(std)", "max(std)"))
    print("-" * 70)
    print("{:<40} {:>10.4f} {:>10.4f}".format(
        "Target peptide pairs", safe_mean(target_stds), float(np.max(target_stds)) if target_stds else float("nan")))
    print("{:<40} {:>10.4f} {:>10.4f}".format(
        "Decoy peptide pairs", safe_mean(decoy_stds), float(np.max(decoy_stds)) if decoy_stds else float("nan")))

    # Correlation between mean and std (often: extremes have lower uncertainty)
    if len(target_means) >= 2 and len(decoy_means) >= 2:
        all_means = np.array(target_means + decoy_means, dtype=np.float64)
        all_stds = np.array(target_stds + decoy_stds, dtype=np.float64)
        # Pearson r
        if all_means.std() > 0 and all_stds.std() > 0:
            r = float(np.corrcoef(all_means, all_stds)[0, 1])
            print("\nPearson r(ergo_mean, ergo_std) over all rows: {:.4f}".format(r))


# ----------------------------------------------------------------------------
#  Top-K worst decoys
# ----------------------------------------------------------------------------
def top_worst_decoys(rows, k=10):
    """Top-K decoy peptides by ERGO score, across all targets — flag potential off-target risks."""
    decoy_only = [r for r in rows if r["is_target_pep"] == 0]
    decoy_only.sort(key=lambda r: -r["ergo_mean"])
    print("\n" + "=" * 110)
    print("TOP {} WORST OFF-TARGET HITS (decoys with highest ERGO score, all targets)".format(k))
    print("=" * 110)
    print("{:<12} {:<14} {:<14} {:<14} {:<5} {:<28} {:>8} {:>8}".format(
        "Target", "TgtHLA", "DecoyPep", "DecoyHLA", "Tier", "EvidenceLevel", "Mean", "MCstd"))
    print("-" * 110)
    for r in decoy_only[:k]:
        print("{:<12} {:<14} {:<14} {:<14} {:<5} {:<28} {:>8.4f} {:>8.4f}".format(
            r["target_pep"][:12],
            r["target_hla"][:14],
            r["decoy_pep"][:14],
            r["decoy_hla"][:14],
            r["decoy_source_tier"][:5],
            (r["decoy_evidence_level"] or "")[:28],
            r["ergo_mean"],
            r["ergo_std"],
        ))


# ----------------------------------------------------------------------------
#  CSV exports
# ----------------------------------------------------------------------------
def export_per_target_csv(summary, out_path):
    fieldnames = list(summary[0].keys()) if summary else []
    with io.open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summary:
            row = {k: ("{:.6f}".format(v) if isinstance(v, float) else v)
                   for k, v in s.items()}
            w.writerow(row)
    print("[export] {}".format(out_path))


def export_per_peptide_csv(table, out_path):
    fieldnames = list(table[0].keys()) if table else []
    with io.open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in table:
            row = {k: ("{:.6f}".format(v) if isinstance(v, float) else v)
                   for k, v in t.items()}
            w.writerow(row)
    print("[export] {}".format(out_path))


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Compute summary metrics from eval_decoy.py CSV")
    p.add_argument("--csv", required=True, help="Per-pair CSV produced by eval_decoy.py")
    p.add_argument("--out_dir", default=None,
                   help="Optional output directory for per-target and per-peptide CSV exports")
    p.add_argument("--high_threshold", type=float, default=0.9,
                   help="Decoys scoring >= this are flagged as off-target hits")
    p.add_argument("--top_k", type=int, default=15,
                   help="Top-K worst decoys to print across all targets")
    p.add_argument("--top_per_target", type=int, default=8,
                   help="Top scored peptides to print per target in the per-peptide table")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print("ERROR: CSV not found: {}".format(args.csv))
        sys.exit(1)

    print("[load] {}".format(args.csv))
    rows = load_csv(args.csv)
    print("[load] {} rows".format(len(rows)))
    if not rows:
        print("ERROR: empty CSV")
        sys.exit(1)

    summary = per_target_summary(rows, high_threshold=args.high_threshold)
    print_per_target_table(summary, args.high_threshold)

    per_tier_rows, per_tier_global = per_tier_auroc(rows)
    print_per_tier_table(per_tier_rows, per_tier_global)

    table = per_peptide_table(rows)
    print_per_peptide_top(table, top_per_target=args.top_per_target)

    evidence_level_breakdown(rows)
    uncertainty_diagnostics(rows)
    top_worst_decoys(rows, k=args.top_k)

    if args.out_dir:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        export_per_target_csv(summary, os.path.join(args.out_dir, "per_target_summary.csv"))
        export_per_peptide_csv(table, os.path.join(args.out_dir, "per_peptide_table.csv"))
        # Per-tier AUROC export
        per_tier_path = os.path.join(args.out_dir, "per_tier_auroc.csv")
        with io.open(per_tier_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["target_pep", "tier", "n_target", "n_decoy",
                          "target_mean", "decoy_mean", "auroc"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_tier_rows + per_tier_global:
                w.writerow({k: ("{:.6f}".format(v) if isinstance(v, float) else v)
                            for k, v in r.items()})
        print("[export] {}".format(per_tier_path))


if __name__ == "__main__":
    main()
