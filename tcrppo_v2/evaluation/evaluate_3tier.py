"""4-Tier Evaluation System for TCRPPO v2.

Tier 0: tFold V3.4 AUROC — structure-aware binding prediction (mean AUC=0.800)
Tier 1: ERGO AUROC — fast LSTM baseline (mean AUC=0.541)
Tier 2: Cross-model validation (NetTCR-2.0) — CNN independent verification (mean AUC=0.601)
Tier 3: Sequence analysis — motif enrichment, diversity, binder distance

Rationale for tFold as primary scorer:
  - 33% higher mean AUC than NetTCR (0.800 vs 0.601) on 37 overlapping peptides
  - 48% higher mean AUC than ERGO (0.800 vs 0.541)
  - Structure-aware: uses 735M tFold model features capturing 3D binding geometry
  - Superior generalization: maintains high AUC even on peptides where NetTCR fails
    (e.g., RAKFKQLL: tFold 0.933 vs NetTCR 0.428)
  - 59.5% of peptides achieve AUC>0.8 (vs NetTCR 3.7%, ERGO 0.9%)

See evaluation/EVALUATION_METHODOLOGY.md for detailed justification.

Usage:
    python -m tcrppo_v2.evaluation.evaluate_3tier \
        --checkpoint output/v2_full_run1/checkpoints/latest.pt \
        --output results/v2_full_4tier.json \
        --tiers 0 1 2 3
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tcrppo_v2.data.pmhc_loader import PMHCLoader, EVAL_TARGETS


def load_generated_tcrs(results_dir: str) -> Dict[str, List[str]]:
    """Load generated TCRs from test_tcrs.py output.

    Expects JSON files: results/<run_name>/<target>.json
    Each with key "generated_tcrs" -> list of CDR3b strings.
    """
    tcrs = {}
    if not os.path.isdir(results_dir):
        return tcrs
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            target = fname[:-5]
            filepath = os.path.join(results_dir, fname)
            with open(filepath) as f:
                data = json.load(f)
            if "generated_tcrs" in data:
                tcrs[target] = data["generated_tcrs"]
            elif "tcrs" in data:
                tcrs[target] = data["tcrs"]
    return tcrs


def run_tier0_tfold(
    generated_tcrs: Dict[str, List[str]],
    n_decoys: int = 50,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Tier 0: tFold V3.4 structure-aware evaluation.

    Uses pre-trained tFold classifier (1.57M params) operating on features
    from 735M tFold structure prediction model. Achieves mean AUC=0.800
    on tc-hard dataset, significantly outperforming ERGO and NetTCR.

    Falls back to empty results if tFold scorer unavailable.
    """
    try:
        from tcrppo_v2.scorers.affinity_tfold import AffinityTFoldScorer
    except ImportError:
        print("[warn] tFold scorer not available, skipping Tier 0")
        return {}

    from tcrppo_v2.data.decoy_sampler import DecoySampler
    from sklearn.metrics import roc_auc_score

    try:
        scorer = AffinityTFoldScorer(device=device)
    except Exception as e:
        print(f"[warn] Failed to initialize tFold scorer: {e}")
        return {}

    targets = list(generated_tcrs.keys())
    sampler = DecoySampler(targets=targets)
    sampler.update_unlocked_tiers(100_000_000)

    results = {}
    for target, tcrs in generated_tcrs.items():
        if not tcrs:
            continue
        decoys = sampler.sample_decoys(target, k=n_decoys)

        # Score each TCR against target and decoys
        target_scores = []
        decoy_scores = []

        pmhc_loader = PMHCLoader()
        pmhc_str = pmhc_loader.get_pmhc_string(target)

        for tcr in tcrs:
            # Target affinity
            try:
                ts, _ = scorer.score(tcr, pmhc_str)
                target_scores.append(ts)
            except Exception as e:
                print(f"[warn] tFold scoring failed for {tcr[:10]}...: {e}")
                continue

            # Decoy scores
            for decoy in decoys:
                try:
                    ds, _ = scorer.score(tcr, decoy)
                    decoy_scores.append(ds)
                except Exception:
                    continue

        if not target_scores or not decoy_scores:
            print(f"[warn] Insufficient tFold scores for {target}, skipping")
            continue

        # Compute AUROC
        labels = [1] * len(target_scores) + [0] * len(decoy_scores)
        scores = target_scores + decoy_scores
        try:
            auroc = float(roc_auc_score(labels, scores))
        except ValueError:
            auroc = 0.5

        results[target] = {
            "auroc": auroc,
            "mean_target_score": float(np.mean(target_scores)),
            "mean_decoy_score": float(np.mean(decoy_scores)),
            "n_tcrs": len(tcrs),
            "n_decoys": n_decoys,
        }

    return results


def run_tier1_ergo(
    generated_tcrs: Dict[str, List[str]],
    n_decoys: int = 50,
    mc_samples: int = 10,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Tier 1: ERGO AUROC evaluation.

    Uses the existing test_tcrs.py evaluate_specificity logic.
    """
    from tcrppo_v2.scorers.affinity_ergo import ERGOScorer
    from tcrppo_v2.data.decoy_sampler import DecoySampler
    from sklearn.metrics import roc_auc_score

    scorer = ERGOScorer(device=device, mc_samples=mc_samples)
    targets = list(generated_tcrs.keys())
    sampler = DecoySampler(targets=targets)
    # Unlock all tiers
    sampler.update_unlocked_tiers(100_000_000)

    results = {}
    for target, tcrs in generated_tcrs.items():
        if not tcrs:
            continue
        decoys = sampler.sample_decoys(target, k=n_decoys)

        # Score each TCR against target and decoys
        target_scores = []
        decoy_scores = []

        pmhc_loader = PMHCLoader()
        pmhc_str = pmhc_loader.get_pmhc_string(target)

        for tcr in tcrs:
            # Target affinity
            ts, _ = scorer.score(tcr, pmhc_str)
            target_scores.append(ts)

            # Decoy scores
            for decoy in decoys:
                decoy_pmhc = decoy  # Simplified — use peptide directly
                ds, _ = scorer.score(tcr, decoy_pmhc)
                decoy_scores.append(ds)

        # Compute AUROC: can we distinguish target from decoy scores?
        labels = [1] * len(target_scores) + [0] * len(decoy_scores)
        scores = target_scores + decoy_scores
        try:
            auroc = float(roc_auc_score(labels, scores))
        except ValueError:
            auroc = 0.5

        results[target] = {
            "auroc": auroc,
            "mean_target_score": float(np.mean(target_scores)),
            "mean_decoy_score": float(np.mean(decoy_scores)),
            "n_tcrs": len(tcrs),
            "n_decoys": n_decoys,
        }

    return results


def run_tier2_nettcr(
    generated_tcrs: Dict[str, List[str]],
    n_decoys: int = 50,
) -> Dict[str, Dict]:
    """Tier 2: NetTCR cross-model validation.

    Independent CNN model scores TCR-peptide binding.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tcrppo_v2.evaluation.nettcr_scorer import NetTCRScorer
    from tcrppo_v2.data.decoy_sampler import DecoySampler

    scorer = NetTCRScorer()
    targets = list(generated_tcrs.keys())
    sampler = DecoySampler(targets=targets)
    sampler.update_unlocked_tiers(100_000_000)

    results = {}
    for target, tcrs in generated_tcrs.items():
        if not tcrs:
            continue
        decoys = sampler.sample_decoys(target, k=n_decoys)

        auroc = scorer.compute_auroc(tcrs, target, decoys)

        # Also get mean binding scores
        target_scores = scorer.score_batch(tcrs, [target] * len(tcrs))
        mean_target = float(np.mean(target_scores))

        results[target] = {
            "auroc": auroc,
            "mean_target_binding_prob": mean_target,
            "n_tcrs": len(tcrs),
            "n_decoys": n_decoys,
        }

    return results


def run_tier3_sequence(
    generated_tcrs: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """Tier 3: Sequence analysis (no external model needed)."""
    from tcrppo_v2.evaluation.sequence_analysis import analyze_generated_tcrs
    return analyze_generated_tcrs(generated_tcrs)


def run_3tier_evaluation(
    generated_tcrs: Dict[str, List[str]],
    tiers: List[int] = None,
    n_decoys: int = 50,
    device: str = "cuda",
) -> Dict:
    """Run full 4-tier evaluation (Tier 0-3).

    Args:
        generated_tcrs: Dict of target -> list of CDR3b sequences.
        tiers: Which tiers to run (default: all [0, 1, 2, 3]).
        n_decoys: Number of decoy peptides per target.
        device: CUDA device for Tier 0 and Tier 1.

    Returns:
        Dict with results for each tier.
    """
    if tiers is None:
        tiers = [0, 1, 2, 3]

    results = {"metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_targets": len(generated_tcrs),
        "targets": list(generated_tcrs.keys()),
        "tiers_run": tiers,
    }}

    if 0 in tiers:
        print("\n=== Tier 0: tFold V3.4 (Structure-Aware) ===")
        t0 = time.time()
        results["tier0_tfold"] = run_tier0_tfold(generated_tcrs, n_decoys=n_decoys, device=device)
        if results["tier0_tfold"]:
            aurocs = [r["auroc"] for r in results["tier0_tfold"].values()]
            print(f"  Mean AUROC: {np.mean(aurocs):.4f} (took {time.time()-t0:.1f}s)")
            for t, r in results["tier0_tfold"].items():
                print(f"  {t}: AUROC={r['auroc']:.4f}")
        else:
            print("  [warn] tFold evaluation failed or unavailable")

    if 1 in tiers:
        print("\n=== Tier 1: ERGO AUROC ===")
        t0 = time.time()
        results["tier1_ergo"] = run_tier1_ergo(generated_tcrs, n_decoys=n_decoys, device=device)
        aurocs = [r["auroc"] for r in results["tier1_ergo"].values()]
        print(f"  Mean AUROC: {np.mean(aurocs):.4f} (took {time.time()-t0:.1f}s)")
        for t, r in results["tier1_ergo"].items():
            print(f"  {t}: AUROC={r['auroc']:.4f}")

    if 2 in tiers:
        print("\n=== Tier 2: NetTCR Cross-Validation ===")
        t0 = time.time()
        results["tier2_nettcr"] = run_tier2_nettcr(generated_tcrs, n_decoys=n_decoys)
        aurocs = [r["auroc"] for r in results["tier2_nettcr"].values()]
        print(f"  Mean AUROC: {np.mean(aurocs):.4f} (took {time.time()-t0:.1f}s)")
        for t, r in results["tier2_nettcr"].items():
            print(f"  {t}: AUROC={r['auroc']:.4f}, mean_binding={r['mean_target_binding_prob']:.4f}")

    if 3 in tiers:
        print("\n=== Tier 3: Sequence Analysis ===")
        t0 = time.time()
        results["tier3_sequence"] = run_tier3_sequence(generated_tcrs)
        print(f"  Completed ({time.time()-t0:.1f}s)")
        for t, r in results["tier3_sequence"].items():
            div = r.get("diversity", {})
            dist = r.get("distance_to_binders", {})
            print(f"  {t}: unique={div.get('n_unique', '?')}/{div.get('n_total', '?')}, "
                  f"mean_pairwise_lev={div.get('mean_pairwise_levenshtein', '?'):.2f}"
                  + (f", mean_min_dist_to_binders={dist.get('mean_min_distance', '?'):.2f}"
                     if dist.get("available", False) else ""))

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Target':<15} {'T0 tFold':>10} {'T1 ERGO':>10} {'T2 NetTCR':>10} {'Unique':>8} {'Binder Dist':>12}")
    print("-" * 75)
    for target in sorted(generated_tcrs.keys()):
        t0 = results.get("tier0_tfold", {}).get(target, {}).get("auroc", "N/A")
        t1 = results.get("tier1_ergo", {}).get(target, {}).get("auroc", "N/A")
        t2 = results.get("tier2_nettcr", {}).get(target, {}).get("auroc", "N/A")
        t3 = results.get("tier3_sequence", {}).get(target, {})
        uniq = t3.get("diversity", {}).get("n_unique", "N/A")
        bdist = t3.get("distance_to_binders", {}).get("mean_min_distance", "N/A")

        t0_str = f"{t0:.4f}" if isinstance(t0, float) else t0
        t1_str = f"{t1:.4f}" if isinstance(t1, float) else t1
        t2_str = f"{t2:.4f}" if isinstance(t2, float) else t2
        bd_str = f"{bdist:.2f}" if isinstance(bdist, float) else str(bdist)

        print(f"{target:<15} {t0_str:>10} {t1_str:>10} {t2_str:>10} {str(uniq):>8} {bd_str:>12}")

    return results


def main():
    parser = argparse.ArgumentParser(description="4-Tier TCR Evaluation")
    parser.add_argument("--results-dir", required=True, help="Dir with generated TCR JSON files")
    parser.add_argument("--output", default="results/4tier_eval.json", help="Output JSON file")
    parser.add_argument("--tiers", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--n-decoys", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generated_tcrs = load_generated_tcrs(args.results_dir)
    if not generated_tcrs:
        print(f"No generated TCRs found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded TCRs for {len(generated_tcrs)} targets")
    print(f"Running tiers: {args.tiers}")
    results = run_3tier_evaluation(
        generated_tcrs, tiers=args.tiers, n_decoys=args.n_decoys, device=args.device
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
